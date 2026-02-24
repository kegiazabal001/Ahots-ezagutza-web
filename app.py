import asyncio
import json
import os
import subprocess
import tempfile
import wave

import numpy as np
import noisereduce as nr
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from starlette.responses import StreamingResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_NAME = "xezpeleta/whisper-medium-eu-ct2"
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")

VAD_OPTS = VadOptions(
    threshold=0.15,
    min_silence_duration_ms=200,
    speech_pad_ms=300,
    min_speech_duration_ms=100,
)



def enhance_audio(audio: np.ndarray, sample_rate: int = 16000, denoise: bool = True) -> np.ndarray:
    """Reduce background noise and normalize audio level.

    Uses spectral subtraction (noisereduce) to clean the signal, then normalizes
    to a target RMS so quiet/whispered speech is amplified consistently.
    """
    if len(audio) == 0:
        return audio

    if denoise and len(audio) >= sample_rate // 4:  # at least 0.25s for reliable noise profile
        audio = nr.reduce_noise(y=audio, sr=sample_rate, stationary=False, prop_decrease=0.75)

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio
    gain = 0.15 / rms
    gain = min(gain, 40.0)  # allow more headroom for whispered speech
    return np.clip(audio * gain, -1.0, 1.0)


def numpy_to_wav_bytes(audio: np.ndarray) -> str:
    """Write float32 audio array to a temporary WAV file. Returns the temp file path."""
    samples = (audio * 32768).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    with wave.open(tmp_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    return tmp_path


def audio_bytes_to_numpy(audio_bytes: bytes) -> np.ndarray | None:
    """Convert raw audio bytes (any format) to 16kHz mono float32 numpy array via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
        input=audio_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    wav_bytes = result.stdout
    # Skip WAV header (44 bytes) and convert
    if len(wav_bytes) <= 44:
        return None
    return np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_audio(audio: np.ndarray) -> list[dict]:
    """Transcribe audio array using VAD + whisper. Returns list of {start, end, text}."""
    speech_chunks = get_speech_timestamps(audio, VAD_OPTS)
    results = []
    for chunk in speech_chunks:
        start_sample = chunk["start"]
        end_sample = chunk["end"]
        chunk_audio = enhance_audio(audio[start_sample:end_sample])

        tmp_chunk = numpy_to_wav_bytes(chunk_audio)
        segs, _ = model.transcribe(tmp_chunk, language="eu", no_speech_threshold=0.99)
        text = " ".join(s.text.strip() for s in segs).strip()
        os.unlink(tmp_chunk)

        if text:
            results.append({
                "start": round(start_sample / 16000, 2),
                "end": round(end_sample / 16000, 2),
                "text": text,
            })
    return results


@app.get("/")
async def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.post("/api/transcribe")
async def transcribe_file(file: UploadFile):
    audio_bytes = await file.read()
    audio = audio_bytes_to_numpy(audio_bytes)
    if audio is None:
        return {"error": "Could not process audio file"}
    audio = enhance_audio(audio)
    results = await asyncio.to_thread(transcribe_audio, audio)
    return {"segments": results}


def transcribe_chunk(chunk_audio: np.ndarray, beam_size: int = 1) -> str:
    """Transcribe a single VAD chunk. Runs in a thread."""
    chunk_audio = enhance_audio(chunk_audio)
    tmp = numpy_to_wav_bytes(chunk_audio)
    segs, _ = model.transcribe(tmp, language="eu", beam_size=beam_size, no_speech_threshold=0.99)
    text = " ".join(s.text.strip() for s in segs).strip()
    os.unlink(tmp)
    return text


@app.post("/api/transcribe-stream")
async def transcribe_file_stream(file: UploadFile):
    audio_bytes = await file.read()
    audio = audio_bytes_to_numpy(audio_bytes)
    if audio is None:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Could not process audio file'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    audio = enhance_audio(audio)
    speech_chunks = await asyncio.to_thread(get_speech_timestamps, audio, VAD_OPTS)

    async def event_stream():
        total = len(speech_chunks)
        yield f"data: {json.dumps({'type': 'progress', 'current': 0, 'total': total})}\n\n"
        for i, chunk in enumerate(speech_chunks):
            start_sample = chunk["start"]
            end_sample = chunk["end"]
            chunk_audio = audio[start_sample:end_sample]
            text = await asyncio.to_thread(transcribe_chunk, chunk_audio)
            if text:
                segment = {
                    "start": round(start_sample / 16000, 2),
                    "end": round(end_sample / 16000, 2),
                    "text": text,
                }
                yield f"data: {json.dumps({'type': 'segment', 'data': segment})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': total})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
    )
