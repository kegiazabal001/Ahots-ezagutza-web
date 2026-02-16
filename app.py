import asyncio
import json
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
from starlette.responses import StreamingResponse

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

VAD_OPTS_STREAM = VadOptions(
    threshold=0.3,
    min_silence_duration_ms=500,
    speech_pad_ms=400,
    min_speech_duration_ms=250,
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


def transcribe_chunk(chunk_audio: np.ndarray) -> str:
    """Transcribe a single VAD chunk. Runs in a thread."""
    tmp = tempfile.mktemp(suffix=".wav")
    with wave.open(tmp, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = (chunk_audio * 32768).astype(np.int16)
        wf.writeframes(samples.tobytes())
    segs, _ = model.transcribe(tmp, language="eu", no_speech_threshold=0.99)
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


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    audio_buffer = np.array([], dtype=np.float32)
    last_transcribed_end = 0  # samples

    try:
        while True:
            data = await ws.receive_bytes()

            # Frontend sends raw PCM float32 at 16kHz
            new_samples = np.frombuffer(data, dtype=np.float32)
            if len(new_samples) == 0:
                continue
            audio_buffer = np.concatenate([audio_buffer, new_samples])

            buffer_duration = len(audio_buffer) / 16000

            # Only run VAD+transcription if we have at least 3s of audio
            if buffer_duration < 3.0:
                continue

            # Run VAD on a sliding window: last 30s or full buffer if shorter
            vad_window = min(len(audio_buffer), 30 * 16000)
            vad_offset = len(audio_buffer) - vad_window
            vad_audio = audio_buffer[vad_offset:]

            speech_chunks = await asyncio.to_thread(get_speech_timestamps, vad_audio, VAD_OPTS_STREAM)

            if not speech_chunks:
                continue

            # Adjust chunk positions to absolute buffer coordinates
            for c in speech_chunks:
                c["start"] += vad_offset
                c["end"] += vad_offset

            # Only consider chunks we haven't transcribed yet
            new_chunks = [c for c in speech_chunks if c["end"] > last_transcribed_end]
            if not new_chunks:
                continue

            last_chunk = new_chunks[-1]
            silence_after_last = len(audio_buffer) - last_chunk["end"]

            # Wait for enough silence after last speech chunk (1.5s)
            if silence_after_last < int(1.5 * 16000):
                continue

            # Group chunks separated by less than 1s into single segments
            grouped = []
            for chunk in new_chunks:
                if grouped and chunk["start"] - grouped[-1]["end"] < 16000:
                    grouped[-1]["end"] = chunk["end"]
                else:
                    grouped.append({"start": chunk["start"], "end": chunk["end"]})

            # Transcribe grouped speech segments
            for chunk in grouped:
                chunk_audio_data = audio_buffer[chunk["start"]:chunk["end"]]

                tmp = tempfile.mktemp(suffix=".wav")
                with wave.open(tmp, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    samples = (chunk_audio_data * 32768).astype(np.int16)
                    wf.writeframes(samples.tobytes())

                segs, _ = await asyncio.to_thread(
                    model.transcribe, tmp, language="eu", beam_size=5, no_speech_threshold=0.99
                )
                text = " ".join(s.text.strip() for s in segs).strip()
                os.unlink(tmp)

                if text:
                    await ws.send_json({"text": text})

            last_transcribed_end = grouped[-1]["end"]

            # Trim buffer: keep from last transcribed position minus 5s padding
            trim_point = max(0, last_transcribed_end - 5 * 16000)
            if trim_point > 0:
                audio_buffer = audio_buffer[trim_point:]
                last_transcribed_end -= trim_point

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
