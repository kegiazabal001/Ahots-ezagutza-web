import os
import subprocess
import tempfile
import wave

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

AUDIO_FILE = "2-06 Lesson 8 Additional Vocabulary, 144.mp3"

model = WhisperModel("xezpeleta/whisper-medium-eu-ct2", device="cpu", compute_type="int8")

# Convert to 16kHz mono WAV for VAD
tmp_wav = tempfile.mktemp(suffix=".wav")
subprocess.run(
    ["ffmpeg", "-y", "-i", AUDIO_FILE, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav],
    capture_output=True,
)

with wave.open(tmp_wav) as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
os.unlink(tmp_wav)

# Detect speech segments with Silero VAD
vad_opts = VadOptions(
    threshold=0.3,
    min_silence_duration_ms=200,
    speech_pad_ms=300,
    min_speech_duration_ms=100,
)
speech_chunks = get_speech_timestamps(audio, vad_opts)

print(f"Duration: {len(audio)/16000:.1f}s | Speech segments: {len(speech_chunks)}")

# Transcribe each speech chunk individually
for chunk in speech_chunks:
    start_sample = chunk["start"]
    end_sample = chunk["end"]
    start = start_sample / 16000
    end = end_sample / 16000
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
        print(f"[{start:.2f}s -> {end:.2f}s] {text}")
