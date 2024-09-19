from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import os

# Charger le modèle Whisper (large)
model = whisper.load_model("large")

app = FastAPI()

# Fonction pour traiter le fichier WAV sans conversion
def process_audio(file: UploadFile):
    extension = file.filename.split('.')[-1].lower()

    if extension == 'wav' or file.content_type == 'audio/wav':
        # Traiter comme un fichier WAV
        wav_filename = "temp_audio.wav"
        with open(wav_filename, "wb") as f:
            f.write(file.file.read())
        return wav_filename
    elif extension == 'mp3' or file.content_type == 'audio/mpeg':
        # Traiter comme un fichier MP3 et convertir en WAV
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file.file, format="mp3")
            wav_filename = "temp_audio.wav"
            audio.export(wav_filename, format="wav")
            return wav_filename
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors de la conversion MP3: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Format non supporté. Veuillez fournir un fichier MP3 ou WAV.")

# Route pour transcrire un fichier audio
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Traiter le fichier audio
    wav_file = process_audio(file)

    try:
        # Utiliser Whisper pour transcrire l'audio
        result = model.transcribe(wav_file)
        transcription = result["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription de l'audio: {str(e)}")
    finally:
        # Supprimer le fichier temporaire
        if os.path.exists(wav_file):
            os.remove(wav_file)

    # Retourner la transcription
    return {"transcription": transcription}
