import asyncio
import os
import subprocess
import tempfile
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_NAME = "xezpeleta/whisper-medium-eu-ct2"
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")

VAD_OPTS = VadOptions(
    threshold=0.3,
    min_silence_duration_ms=200,
    speech_pad_ms=300,
    min_speech_duration_ms=100,
)


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
        chunk_audio = audio[start_sample:end_sample]

        tmp_chunk = tempfile.mktemp(suffix=".wav")
        with wave.open(tmp_chunk, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            samples = (chunk_audio * 32768).astype(np.int16)
            wf.writeframes(samples.tobytes())

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
    results = await asyncio.to_thread(transcribe_audio, audio)
    return {"segments": results}


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    audio_buffer = np.array([], dtype=np.float32)
    raw_bytes = bytearray()
    prev_decoded_len = 0
    last_transcribed_end = 0  # samples

    try:
        while True:
            data = await ws.receive_bytes()
            raw_bytes.extend(data)

            # Decode the full accumulated WebM stream (header is only in first chunk)
            full_audio = await asyncio.to_thread(audio_bytes_to_numpy, bytes(raw_bytes))
            if full_audio is None or len(full_audio) == 0:
                continue

            # Only take newly decoded samples to avoid reprocessing
            if len(full_audio) > prev_decoded_len:
                new_samples = full_audio[prev_decoded_len:]
                audio_buffer = np.concatenate([audio_buffer, new_samples])
                prev_decoded_len = len(full_audio)
            else:
                continue

            buffer_duration = len(audio_buffer) / 16000

            # Only run VAD+transcription if we have at least 2s of audio
            if buffer_duration < 2.0:
                continue

            speech_chunks = await asyncio.to_thread(get_speech_timestamps, audio_buffer, VAD_OPTS)

            if not speech_chunks:
                continue

            # Only consider chunks we haven't transcribed yet
            new_chunks = [c for c in speech_chunks if c["end"] > last_transcribed_end]
            if not new_chunks:
                continue

            last_chunk = new_chunks[-1]
            silence_after_last = len(audio_buffer) - last_chunk["end"]

            # Wait for enough silence after last speech chunk (0.8s)
            if silence_after_last < int(0.8 * 16000):
                continue

            # Transcribe new speech chunks
            for chunk in new_chunks:
                chunk_audio_data = audio_buffer[chunk["start"]:chunk["end"]]

                tmp = tempfile.mktemp(suffix=".wav")
                with wave.open(tmp, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    samples = (chunk_audio_data * 32768).astype(np.int16)
                    wf.writeframes(samples.tobytes())

                segs, _ = await asyncio.to_thread(
                    model.transcribe, tmp, language="eu", no_speech_threshold=0.99
                )
                text = " ".join(s.text.strip() for s in segs).strip()
                os.unlink(tmp)

                if text:
                    await ws.send_json({"text": text})

            last_transcribed_end = last_chunk["end"]

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
    )
