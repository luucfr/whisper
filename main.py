from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import uuid
import threading
import queue
from typing import Dict, List, Optional
import torch
from pyannote.audio import Pipeline
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CUDA settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class AudioProcessor:
    def __init__(self):
        self.model: Optional[whisper.Whisper] = None
        self.pipeline: Optional[Pipeline] = None
        self.request_queue: queue.Queue = queue.Queue()
        self.results: Dict[str, str] = {}
        self.lock = threading.Lock()
        self.initialize_models()

    def initialize_models(self) -> None:
        """Initialize Whisper and Pyannote models."""
        try:
            self.model = whisper.load_model("large")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token="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"
            )
            self.pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise RuntimeError("Failed to initialize audio processing models")

    @contextmanager
    def handle_audio_file(self, file: UploadFile) -> str:
        """Context manager for handling audio file processing and cleanup."""
        temp_filename = f"temp_audio_{uuid.uuid4()}.wav"
        try:
            yield temp_filename
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    logger.error(f"Failed to remove temporary file {temp_filename}: {e}")

    async def process_audio(self, file: UploadFile) -> str:
        """Process the uploaded audio file."""
        extension = file.filename.split('.')[-1].lower()
        
        if extension not in ['wav', 'mp3'] and file.content_type not in ['audio/wav', 'audio/mpeg']:
            raise HTTPException(
                status_code=400,
                detail="Unsupported format. Please provide a MP3 or WAV file."
            )

        with self.handle_audio_file(file) as wav_filename:
            content = await file.read()
            with open(wav_filename, "wb") as f:
                f.write(content)
            return wav_filename

    def diarize_audio(self, wav_file: str) -> List[Dict]:
        """Perform speaker diarization on the audio file."""
        try:
            diarization = self.pipeline({"uri": wav_file, "audio": wav_file})
            return [
                {
                    "speaker": speaker,
                    "start": float(turn.start),
                    "end": float(turn.end)
                }
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise RuntimeError(f"Speaker diarization failed: {str(e)}")

    def transcribe_audio(self, wav_file: str) -> List[Dict]:
        """Transcribe the audio file using Whisper."""
        try:
            result = self.model.transcribe(wav_file, word_timestamps=True)
            return result["segments"]
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Audio transcription failed: {str(e)}")

    def merge_transcriptions(self, speaker_segments: List[Dict], transcription_segments: List[Dict]) -> str:
        """Merge diarization and transcription results."""
        speaker_mapping = {}
        speaker_count = 1
        merged_transcription = []
        last_segment = {"text": None, "speaker": None}

        for segment in speaker_segments:
            start, end, speaker = segment["start"], segment["end"], segment["speaker"]

            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"SPEAKER {speaker_count}"
                speaker_count += 1

            speaker_label = speaker_mapping[speaker]
            segment_text = self.extract_segment_text(start, end, transcription_segments)

            if segment_text:
                current_text = " ".join(segment_text)
                # Avoid consecutive duplicate segments from the same speaker
                if (current_text != last_segment["text"] or 
                    speaker_label != last_segment["speaker"]):
                    final_segment = f"{speaker_label}: {current_text}"
                    merged_transcription.append(final_segment)
                    last_segment = {"text": current_text, "speaker": speaker_label}

        return "\n".join(merged_transcription)

    def extract_segment_text(self, start: float, end: float, 
                           transcription_segments: List[Dict]) -> List[str]:
        """Extract text segments within the given time range."""
        return [
            t['text']
            for t in transcription_segments
            if t['start'] < end and t['end'] > start
        ]

    async def process_file(self, wav_file: str) -> str:
        """Process the audio file and return the transcription."""
        try:
            speaker_segments = self.diarize_audio(wav_file)
            transcription_segments = self.transcribe_audio(wav_file)
            return self.merge_transcriptions(speaker_segments, transcription_segments)
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise RuntimeError(f"Audio processing failed: {str(e)}")

# Initialize the FastAPI app
app = FastAPI(title="Audio Transcription Service")
audio_processor = AudioProcessor()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_audio_task(file_id: str, wav_file: str) -> None:
    """Background task for processing audio files."""
    try:
        transcription = await audio_processor.process_file(wav_file)
        with audio_processor.lock:
            audio_processor.results[file_id] = transcription
    except Exception as e:
        logger.error(f"Error processing file {file_id}: {e}")
        with audio_processor.lock:
            audio_processor.results[file_id] = f"Error during transcription: {str(e)}"

@app.post("/transcribe/")
async def transcribe(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict[str, str]:
    """
    Endpoint for audio transcription.
    Returns a file ID immediately and processes the audio in the background.
    """
    try:
        wav_file = await audio_processor.process_audio(file)
        file_id = str(uuid.uuid4())
        
        background_tasks.add_task(process_audio_task, file_id, wav_file)
        
        return {
            "file_id": file_id,
            "status": "processing",
            "message": "Audio file is being processed"
        }
    except Exception as e:
        logger.error(f"Transcription request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{file_id}")
async def get_transcription_status(file_id: str) -> Dict[str, str]:
    """Get the status or result of a transcription job."""
    with audio_processor.lock:
        if file_id not in audio_processor.results:
            return {"status": "processing"}
        
        transcription = audio_processor.results.pop(file_id)
        return {
            "status": "completed",
            "transcription": transcription
        }