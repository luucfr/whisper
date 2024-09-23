from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import os
import uuid
import threading
import queue

# Charger le modèle Whisper (large)
model = whisper.load_model("large")

app = FastAPI()

# File d'attente pour les requêtes
request_queue = queue.Queue()
results = {}

# Fonction pour traiter le fichier WAV sans conversion
def process_audio(file: UploadFile):
    extension = file.filename.split('.')[-1].lower()
    wav_filename = f"temp_audio_{uuid.uuid4()}.wav"  # Fichier temporaire unique

    if extension == 'wav' or file.content_type == 'audio/wav':
        with open(wav_filename, "wb") as f:
            f.write(file.file.read())
        return wav_filename
    elif extension == 'mp3' or file.content_type == 'audio/mpeg':
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file.file, format="mp3")
            audio.export(wav_filename, format="wav")
            return wav_filename
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors de la conversion MP3: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Format non supporté. Veuillez fournir un fichier MP3 ou WAV.")

# Fonction worker pour traiter les requêtes
def worker():
    while True:
        file_id, wav_file = request_queue.get()
        if wav_file is None:
            break  # Quitter la boucle si None est reçu

        try:
            result = model.transcribe(wav_file)
            transcription = result["text"]
            results[file_id] = transcription
        except Exception as e:
            results[file_id] = f"Erreur lors de la transcription: {str(e)}"
        finally:
            if os.path.exists(wav_file):
                os.remove(wav_file)
            request_queue.task_done()

# Démarrer le thread worker
threading.Thread(target=worker, daemon=True).start()

# Route pour transcrire un fichier audio
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    wav_file = process_audio(file)
    file_id = str(uuid.uuid4())  # Identifiant unique pour cette requête

    # Mettre la tâche dans la file d'attente
    request_queue.put((file_id, wav_file))

    # Attendre la fin du traitement
    request_queue.join()

    # Récupérer le résultat
    transcription = results.pop(file_id, None)

    if transcription is None:
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de la transcription.")

    return {"transcription": transcription}
