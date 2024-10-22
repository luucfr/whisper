from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import uuid
import threading
import queue
from typing import Dict, List, Optional, Tuple
import torch
from pyannote.audio import Pipeline
import logging
from contextlib import contextmanager
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.model: Optional[whisper.Whisper] = None
        self.pipeline: Optional[Pipeline] = None
        self.results: Dict[str, str] = {}
        self.lock = threading.Lock()
        self.temp_dir = "temp_audio_files"
        self.initialize_models()
        self.ensure_temp_directory()

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

    def ensure_temp_directory(self) -> None:
        """Ensure temporary directory exists."""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            logger.info(f"Created temporary directory: {self.temp_dir}")

    def get_temp_filepath(self, file_id: str) -> str:
        """Generate temporary file path."""
        return os.path.join(self.temp_dir, f"{file_id}.wav")

    async def save_upload_file(self, upload_file: UploadFile, file_path: str) -> None:
        """Save uploaded file to temporary location."""
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(upload_file.file, f)
        except Exception as e:
            logger.error(f"Failed to save upload file: {e}")
            raise RuntimeError(f"Failed to save upload file: {str(e)}")
        finally:
            upload_file.file.close()

    def cleanup_file(self, file_path: str) -> None:
        """Clean up temporary file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")

    async def process_audio(self, file: UploadFile) -> Tuple[str, str]:
        """Process the uploaded audio file and return file_id and path."""
        extension = file.filename.split('.')[-1].lower()
        
        if extension not in ['wav', 'mp3'] and file.content_type not in ['audio/wav', 'audio/mpeg']:
            raise HTTPException(
                status_code=400,
                detail="Unsupported format. Please provide a MP3 or WAV file."
            )

        file_id = str(uuid.uuid4())
        file_path = self.get_temp_filepath(file_id)
        await self.save_upload_file(file, file_path)
        
        return file_id, file_path

    def diarize_audio(self, wav_file: str) -> List[Dict]:
        """Perform speaker diarization on the audio file."""
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"Audio file not found: {wav_file}")
            
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
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"Audio file not found: {wav_file}")
            
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
    finally:
        audio_processor.cleanup_file(wav_file)

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
        file_id, wav_file = await audio_processor.process_audio(file)
        
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

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up temporary files on shutdown."""
    try:
        shutil.rmtree(audio_processor.temp_dir)
        logger.info("Cleaned up temporary directory on shutdown")
    except Exception as e:
        logger.error(f"Failed to cleanup temporary directory: {e}")