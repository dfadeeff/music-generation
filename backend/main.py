"""FastAPI backend for MusicGen music generation."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torch
import scipy
import os
import time

app = FastAPI(title="Music to My Ears")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Load model once at startup
print(f"Loading MusicGen on {DEVICE}...")
MODEL_NAME = "facebook/musicgen-small"
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
SAMPLE_RATE = model.config.audio_encoder.sampling_rate
print(f"Model loaded. Sample rate: {SAMPLE_RATE}Hz")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str
    duration: int = 256  # tokens: 256=5s, 512=10s, 1024=20s


@app.post("/generate")
def generate(req: GenerateRequest):
    inputs = processor(text=[req.prompt], padding=True, return_tensors="pt").to(DEVICE)
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=req.duration)

    filename = f"gen_{int(time.time())}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=audio_values[0, 0].cpu().numpy())

    return FileResponse(filepath, media_type="audio/wav", filename=filename)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME}