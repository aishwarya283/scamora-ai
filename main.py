from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, UploadFile, HTTPException

# Allow backend to import files from ML folder
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ML"

sys.path.insert(0, str(ML_DIR))

from ml_engine import analyze_call


app = FastAPI(
    title="Scamora AI API",
    description="Real-time scam and deepfake voice detection API",
    version="1.0.0",
)


import os
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


def transcribe_with_groq(audio_path: str) -> str:
    """Transcribes audio using Groq Whisper model."""
    if not groq_client:
        return ""
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="json",
            )
            return transcription.text.strip()
    except Exception as e:
        print(f"Groq Whisper transcription error: {e}")
        return ""


@app.get("/")
def root():
    return {
        "message": "Scamora AI API is running",
        "status": "success",
    }


@app.post("/score-audio")
async def score_audio(
    audio: UploadFile = File(...),
    transcript: Optional[str] = Form(None),
):
    try:
        # Temporary location for uploaded audio
        temp_dir = BASE_DIR / "backend" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_path = temp_dir / audio.filename
        contents = await audio.read()

        with open(audio_path, "wb") as file:
            file.write(contents)

        # Determine transcript: if absent, empty, or default live-call label, transcribe via Groq
        final_transcript = (transcript or "").strip()
        if (
            not final_transcript
            or final_transcript.startswith("Live call incoming audio screening")
            or final_transcript == "[AUTO_TRANSCRIBE]"
        ):
            transcribed = transcribe_with_groq(str(audio_path))
            if transcribed:
                final_transcript = transcribed
            elif not final_transcript:
                final_transcript = "Incoming caller voice audio"

        # Run ML analysis
        result = analyze_call(
            audio_path=str(audio_path),
            text=final_transcript,
        )

        # Delete temporary audio
        audio_path.unlink(missing_ok=True)

        # Attach transcript to the result
        result["transcript"] = final_transcript

        return {
            "success": True,
            "result": result,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )