"""FastAPI backend for MusicGen music generation."""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from openai import OpenAI
from dotenv import load_dotenv
import torch
import scipy
import json
import os
import time
import base64

load_dotenv()

app = FastAPI(title="Music to My Ears")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client for prompt enhancement + image description
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Load model once at startup
print(f"Loading MusicGen on {DEVICE}...")
MODEL_NAME = "facebook/musicgen-medium"
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
SAMPLE_RATE = model.config.audio_encoder.sampling_rate
print(f"Model loaded. Sample rate: {SAMPLE_RATE}Hz")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def enhance_prompt(raw_prompt: str) -> str:
    """Use GPT-4o to turn a raw user prompt into a rich MusicGen music direction."""
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a music director. Given a user's description, output a concise MusicGen prompt "
                "(1-2 sentences) specifying: genre, tempo (BPM), key, instruments, mood, and energy level. "
                "Be specific and musical. Output ONLY the prompt, nothing else."
            )},
            {"role": "user", "content": raw_prompt},
        ],
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def describe_image(image_bytes: bytes) -> str:
    """Use GPT-4o vision to describe an image for music generation."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Describe this image in terms of mood, emotion, colors, scene, and atmosphere. "
                "Then suggest a musical interpretation: genre, tempo, instruments, key, energy. "
                "Output a single MusicGen-ready prompt (1-2 sentences). Output ONLY the prompt."
            )},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def generate_audio(prompt: str, duration_tokens: int) -> str:
    """Run MusicGen and return filepath to .wav."""
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(DEVICE)
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=duration_tokens)
    filename = f"gen_{int(time.time())}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=audio_values[0, 0].cpu().numpy())
    return filepath, filename


# --- API Endpoints ---

class GenerateRequest(BaseModel):
    prompt: str
    duration: int = 256


@app.post("/generate")
def generate(req: GenerateRequest):
    enhanced = enhance_prompt(req.prompt)
    filepath, filename = generate_audio(enhanced, req.duration)

    # Save to history
    entry = {
        "id": filename.replace(".wav", ""),
        "original_prompt": req.prompt,
        "enhanced_prompt": enhanced,
        "filename": filename,
        "duration_tokens": req.duration,
        "timestamp": time.time(),
        "rating": 0,
        "source": "text",
    }
    history = load_history()
    history.insert(0, entry)
    save_history(history)

    return {
        "audio_url": f"/audio/{filename}",
        "enhanced_prompt": enhanced,
        "id": entry["id"],
    }


@app.post("/generate-from-image")
async def generate_from_image(
    image: UploadFile = File(...),
    duration: int = Form(256),
):
    image_bytes = await image.read()
    music_prompt = describe_image(image_bytes)
    filepath, filename = generate_audio(music_prompt, duration)

    entry = {
        "id": filename.replace(".wav", ""),
        "original_prompt": f"[Image: {image.filename}]",
        "enhanced_prompt": music_prompt,
        "filename": filename,
        "duration_tokens": duration,
        "timestamp": time.time(),
        "rating": 0,
        "source": "image",
    }
    history = load_history()
    history.insert(0, entry)
    save_history(history)

    return {
        "audio_url": f"/audio/{filename}",
        "enhanced_prompt": music_prompt,
        "id": entry["id"],
    }


@app.get("/audio/{filename}")
def get_audio(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(filepath, media_type="audio/wav", filename=filename)


@app.post("/rate/{gen_id}")
def rate(gen_id: str, rating: int):
    """Rate a generation: 1 = thumbs up, -1 = thumbs down."""
    history = load_history()
    for entry in history:
        if entry["id"] == gen_id:
            entry["rating"] = rating
            break
    save_history(history)
    return {"ok": True}


@app.get("/history")
def get_history():
    return load_history()


@app.get("/top")
def top_generations():
    """Return generations sorted by rating (best first)."""
    history = load_history()
    return sorted(history, key=lambda x: x["rating"], reverse=True)[:20]


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME}