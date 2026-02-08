"""FastAPI backend for MusicGen — multimodal music generation with emotion fusion."""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import scipy
import json
import os
import time
import base64
import tempfile

load_dotenv()

app = FastAPI(title="Music to My Ears")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Load MusicGen
print(f"Loading MusicGen on {DEVICE}...")
MODEL_NAME = "facebook/musicgen-medium"
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
SAMPLE_RATE = model.config.audio_encoder.sampling_rate
print(f"Model loaded. Sample rate: {SAMPLE_RATE}Hz")

# Load Whisper
print("Loading Whisper...")
import whisper
whisper_model = whisper.load_model("base")
print("Whisper loaded.")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- History ---

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def get_learned_defaults(emotion: str) -> dict:
    """Get averaged slider values from high-rated past generations with similar emotion."""
    history = load_history()
    matching = [h for h in history if h.get("rating", 0) > 0 and h.get("emotion_profile", {}).get("emotion") == emotion]
    if len(matching) < 2:
        return {}
    keys = ["energy", "style", "warmth", "arc"]
    result = {}
    for k in keys:
        vals = [h["emotion_profile"][k] for h in matching if k in h.get("emotion_profile", {})]
        if vals:
            result[k] = round(sum(vals) / len(vals))
    return result


def get_top_prompts() -> str:
    history = load_history()
    top = [h["enhanced_prompt"] for h in history if h.get("rating", 0) > 0][:3]
    if not top:
        return ""
    return "Previously well-rated prompts for style reference: " + " | ".join(top)


# --- Analysis functions (run concurrently) ---

def analyze_text(text: str) -> dict:
    """Analyze text for emotional content."""
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Analyze the emotional content of this text. Return JSON with: "
                '"summary" (1 sentence), "emotion" (single word), "energy" (0-100), '
                '"mood" (brief description). Output ONLY valid JSON.'
            )},
            {"role": "user", "content": text},
        ],
        max_tokens=150,
    )
    return {"source": "text", **json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))}


def analyze_images(images_b64: list[str]) -> dict:
    """Analyze images for emotional content."""
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in images_b64]
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Analyze the mood and emotion of these images together. Return JSON with: "
                '"caption" (1 sentence scene description), "emotion" (single word), "energy" (0-100), '
                '"mood" (brief description). Output ONLY valid JSON.'
            )},
            {"role": "user", "content": content},
        ],
        max_tokens=150,
    )
    return {"source": "image", **json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))}


def analyze_voice(audio_bytes: bytes) -> dict:
    """Transcribe voice with Whisper, then analyze emotion."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"].strip()
    finally:
        os.unlink(tmp_path)

    if not transcript:
        return {"source": "voice", "transcript": "", "emotion": "neutral", "energy": 50, "mood": "neutral tone"}

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "This is a voice transcription. Analyze the emotional tone and intent. Return JSON with: "
                '"transcript" (the text), "emotion" (single word), "energy" (0-100), '
                '"mood" (brief description). Output ONLY valid JSON.'
            )},
            {"role": "user", "content": transcript},
        ],
        max_tokens=150,
    )
    return {"source": "voice", **json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))}


# --- Emotion fusion ---

def fuse_emotions(analyses: list[dict]) -> dict:
    """Fuse multiple emotion signals into a 4-dimensional profile."""
    signals_str = json.dumps(analyses, indent=2)
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an emotion-to-music translator. Given multiple emotional signals from different sources "
                "(text, image, voice), synthesize ONE unified emotional profile. Return JSON with:\n"
                '- "emotion": dominant emotion word (e.g. "melancholy", "euphoria", "tension")\n'
                '- "energy": 0-100 (0=whisper quiet, 100=explosive)\n'
                '- "style": 0-100 (0=lo-fi intimate, 100=cinematic epic)\n'
                '- "warmth": 0-100 (0=warm analog, 100=bright digital)\n'
                '- "arc": 0-100 (0=steady constant, 100=dramatic build)\n'
                '- "description": 1 sentence describing the emotional landscape\n'
                "Output ONLY valid JSON."
            )},
            {"role": "user", "content": signals_str},
        ],
        max_tokens=200,
    )
    profile = json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))

    # Apply learned defaults from feedback loop
    learned = get_learned_defaults(profile.get("emotion", ""))
    if learned:
        for key in ["energy", "style", "warmth", "arc"]:
            if key in learned and key in profile:
                # Blend: 70% AI-detected, 30% learned from feedback
                profile[key] = round(profile[key] * 0.7 + learned[key] * 0.3)

    return profile


# --- Slider-to-prompt mapping ---

ENERGY_MAP = {
    (0, 20): "quiet, minimal, ambient, whisper-soft, ~60 BPM",
    (21, 40): "gentle, relaxed, easy-going, mellow, ~80 BPM",
    (41, 60): "moderate, steady, grooving, balanced, ~100 BPM",
    (61, 80): "energetic, driving, powerful, uplifting, ~120 BPM",
    (81, 100): "intense, explosive, soaring, maximum energy, ~140 BPM",
}
STYLE_MAP = {
    (0, 20): "lo-fi, intimate, bedroom production, raw and personal",
    (21, 40): "indie, understated, warm production, subtle details",
    (41, 60): "polished pop, clean production, radio-ready",
    (61, 80): "anthemic, big production, layered arrangement",
    (81, 100): "cinematic, orchestral, epic, massive soundscape",
}
WARMTH_MAP = {
    (0, 20): "warm analog, vinyl texture, tape saturation, vintage",
    (21, 40): "organic, acoustic instruments, natural reverb",
    (41, 60): "balanced mix of electronic and acoustic elements",
    (61, 80): "modern electronic, clean synths, digital precision",
    (81, 100): "bright digital, crystalline, futuristic, hi-fi",
}
ARC_MAP = {
    (0, 20): "steady, constant, meditative, unchanging",
    (21, 40): "gentle evolution, subtle shifts, slow progression",
    (41, 60): "clear sections, moderate dynamics, natural flow",
    (61, 80): "rising intensity, building momentum, crescendo",
    (81, 100): "dramatic build, explosive climax, cinematic arc",
}


def map_slider(val: int, mapping: dict) -> str:
    for (lo, hi), desc in mapping.items():
        if lo <= val <= hi:
            return desc
    return ""


def profile_to_prompt(profile: dict) -> str:
    """Convert emotion profile + sliders into a rich MusicGen prompt."""
    energy_desc = map_slider(profile.get("energy", 50), ENERGY_MAP)
    style_desc = map_slider(profile.get("style", 50), STYLE_MAP)
    warmth_desc = map_slider(profile.get("warmth", 50), WARMTH_MAP)
    arc_desc = map_slider(profile.get("arc", 50), ARC_MAP)

    style_context = get_top_prompts()
    system_parts = [
        "You are a music director. Create a vivid, specific MusicGen prompt (2-3 sentences) that combines these musical qualities:",
        f"Energy: {energy_desc}",
        f"Style: {style_desc}",
        f"Warmth: {warmth_desc}",
        f"Arc: {arc_desc}",
        f"Core emotion: {profile.get('emotion', 'neutral')} — {profile.get('description', '')}",
        "Include specific instruments, tempo, key, and production style. Output ONLY the prompt.",
    ]
    if style_context:
        system_parts.append(style_context)

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": "Generate the MusicGen prompt."},
        ],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


# --- Explainability ---

def explain_generation(original_inputs: str, profile: dict, music_prompt: str) -> dict:
    """Generate an explanation of how inputs became music."""
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are explaining how AI transformed a user's input into music. "
                "Write 2-3 sentences connecting the input, the detected emotion, and the musical output. "
                "Be poetic but concise. Also return a timeline of 5 steps. Return JSON with:\n"
                '"narrative": "2-3 sentence explanation",\n'
                '"timeline": [{"step": "name", "detail": "what happened"}] (5 steps)\n'
                "Output ONLY valid JSON."
            )},
            {"role": "user", "content": json.dumps({
                "inputs": original_inputs,
                "profile": profile,
                "music_prompt": music_prompt,
            })},
        ],
        max_tokens=300,
    )
    return json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))


# --- Audio generation ---

def generate_audio(prompt: str, duration_tokens: int) -> tuple[str, str]:
    """Run MusicGen and return (filepath, filename)."""
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
    energy: int | None = None
    style: int | None = None
    warmth: int | None = None
    arc: int | None = None


@app.post("/generate")
def generate(req: GenerateRequest):
    """Text-to-music with emotion analysis, sliders, and explainability."""
    # Step 1: Analyze text
    analysis = analyze_text(req.prompt)

    # Step 2: Fuse into profile
    profile = fuse_emotions([analysis])

    # Step 3: Apply user slider overrides
    overrides = []
    for key in ["energy", "style", "warmth", "arc"]:
        val = getattr(req, key)
        if val is not None:
            profile[key] = val
            overrides.append(key)
    profile["overrides"] = overrides

    # Step 4: Generate music prompt from profile
    music_prompt = profile_to_prompt(profile)

    # Step 5: Generate audio
    filepath, filename = generate_audio(music_prompt, req.duration)

    # Step 6: Explain
    explanation = explain_generation(req.prompt, profile, music_prompt)

    # Save to history
    entry = {
        "id": filename.replace(".wav", ""),
        "original_prompt": req.prompt,
        "enhanced_prompt": music_prompt,
        "emotion_profile": profile,
        "explanation": explanation,
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
        "enhanced_prompt": music_prompt,
        "emotion_profile": profile,
        "explanation": explanation,
        "id": entry["id"],
    }


@app.post("/generate-multimodal")
async def generate_multimodal(
    images: list[UploadFile] | None = File(None),
    voice: UploadFile | None = File(None),
    text: str = Form(""),
    duration: int = Form(256),
    energy: int | None = Form(None),
    style: int | None = Form(None),
    warmth: int | None = Form(None),
    arc: int | None = Form(None),
):
    """Multimodal generation: text + images + voice → emotion fusion → music."""
    analyses = []
    input_desc_parts = []

    # Concurrent analysis of all inputs
    with ThreadPoolExecutor() as executor:
        futures = {}

        if text.strip():
            futures[executor.submit(analyze_text, text.strip())] = "text"
            input_desc_parts.append(f"Text: {text.strip()}")

        if images:
            images_bytes = [await img.read() for img in images]
            images_b64 = [base64.b64encode(b).decode("utf-8") for b in images_bytes]
            image_names = [img.filename for img in images]
            futures[executor.submit(analyze_images, images_b64)] = "image"
            input_desc_parts.append(f"Images: {', '.join(image_names)}")

        if voice:
            voice_bytes = await voice.read()
            futures[executor.submit(analyze_voice, voice_bytes)] = "voice"
            input_desc_parts.append("Voice recording")

        for future in as_completed(futures):
            try:
                analyses.append(future.result())
            except Exception as e:
                print(f"Analysis error ({futures[future]}): {e}")

    if not analyses:
        return {"error": "No valid inputs provided"}

    # Fuse emotions
    profile = fuse_emotions(analyses)

    # Apply slider overrides
    overrides = []
    for key, val in [("energy", energy), ("style", style), ("warmth", warmth), ("arc", arc)]:
        if val is not None:
            profile[key] = val
            overrides.append(key)
    profile["overrides"] = overrides

    # Generate prompt + audio
    music_prompt = profile_to_prompt(profile)
    filepath, filename = generate_audio(music_prompt, duration)

    # Explain
    input_desc = " | ".join(input_desc_parts)
    explanation = explain_generation(input_desc, profile, music_prompt)

    # Save
    entry = {
        "id": filename.replace(".wav", ""),
        "original_prompt": input_desc,
        "enhanced_prompt": music_prompt,
        "emotion_profile": profile,
        "explanation": explanation,
        "analyses": analyses,
        "filename": filename,
        "duration_tokens": duration,
        "timestamp": time.time(),
        "rating": 0,
        "source": "multimodal",
    }
    history = load_history()
    history.insert(0, entry)
    save_history(history)

    return {
        "audio_url": f"/audio/{filename}",
        "enhanced_prompt": music_prompt,
        "emotion_profile": profile,
        "explanation": explanation,
        "analyses": analyses,
        "id": entry["id"],
    }


@app.get("/audio/{filename}")
def get_audio(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(filepath, media_type="audio/wav", filename=filename)


@app.post("/rate/{gen_id}")
def rate(gen_id: str, rating: int):
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
    history = load_history()
    return sorted(history, key=lambda x: x["rating"], reverse=True)[:20]


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME}