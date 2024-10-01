from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import os
import uuid
import threading
import queue
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment


class AudioProcessor:
    def __init__(self):
        self.model = whisper.load_model("large")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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"
        )
        self.pipeline.to(torch.device("cuda"))
        self.request_queue = queue.Queue()
        self.results = {}

    def process_audio(self, file: UploadFile):
        """Process the uploaded audio file."""
        extension = file.filename.split('.')[-1].lower()
        wav_filename = f"temp_audio_{uuid.uuid4()}.wav"

        if extension in ['wav', 'mp3'] or file.content_type in ['audio/wav', 'audio/mpeg']:
            with open(wav_filename, "wb") as f:
                f.write(file.file.read())
            return wav_filename
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Please provide a MP3 or WAV file.")

    def diarize_audio(self, wav_file):
        """Perform speaker diarization on the audio file."""
        diarization = self.pipeline({"uri": wav_file, "audio": wav_file})
        return [
            {"speaker": speaker, "start": turn.start, "end": turn.end}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

    def transcribe_audio(self, wav_file):
        """Transcribe the audio file using Whisper."""
        result = self.model.transcribe(wav_file, word_timestamps=True)
        return result["segments"]

    def merge_transcriptions(self, speaker_segments, transcription_segments):
        """Merge diarization and transcription, avoiding duplicates."""
        speaker_mapping = {}
        speaker_count = 1
        merged_transcription = []

        for segment in speaker_segments:
            start, end, speaker = segment["start"], segment["end"], segment["speaker"]

            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"SPEAKER {speaker_count}"
                speaker_count += 1

            speaker_label = speaker_mapping[speaker]
            segment_text = self.extract_segment_text(start, end, transcription_segments)

            if segment_text:
                final_segment = f"{speaker_label}: {' '.join(segment_text)}"
                merged_transcription.append(final_segment)

        return "\n".join(merged_transcription)

    def extract_segment_text(self, start, end, transcription_segments):
        """Extract unique text segments within the given time range."""
        segment_text = set()
        previous_end = None

        for t in transcription_segments:
            if t['start'] < end and t['end'] > start:
                # Ignore overlapping segments
                if previous_end is not None and t['start'] < previous_end + 0.5:
                    continue
                previous_end = t['end']
                segment_text.add(t['text'])

        return list(segment_text)

    def process_file(self, wav_file):
        """Process the audio file and return the transcription."""
        speaker_segments = self.diarize_audio(wav_file)
        transcription_segments = self.transcribe_audio(wav_file)
        return self.merge_transcriptions(speaker_segments, transcription_segments)


# Initialize the FastAPI app
app = FastAPI()
audio_processor = AudioProcessor()

# Worker function to handle requests
def worker():
    while True:
        file_id, wav_file = audio_processor.request_queue.get()
        if wav_file is None:
            break

        try:
            transcription = audio_processor.process_file(wav_file)
            audio_processor.results[file_id] = transcription
        except Exception as e:
            audio_processor.results[file_id] = f"Error during transcription: {str(e)}"
        finally:
            if os.path.exists(wav_file):
                os.remove(wav_file)
            audio_processor.request_queue.task_done()

# Start the worker thread
threading.Thread(target=worker, daemon=True).start()

# Route for audio transcription
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    wav_file = audio_processor.process_audio(file)
    file_id = str(uuid.uuid4())

    # Add task to the queue
    audio_processor.request_queue.put((file_id, wav_file))
    audio_processor.request_queue.join()

    # Retrieve the result
    transcription = audio_processor.results.pop(file_id, None)
    if transcription is None:
        raise HTTPException(status_code=500, detail="Error retrieving transcription.")

    return {"transcription": transcription}