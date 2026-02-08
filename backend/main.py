"""FastAPI backend — multimodal music generation with emotion fusion, A/B comparison, and reflection learning."""

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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")
LEARNED_RULES_FILE = os.path.join(DATA_DIR, "learned_rules.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

REFLECTION_THRESHOLD = 5


# ============================================================
# JSON helpers
# ============================================================

def _load_json(path, default=None):
    if default is None:
        default = []
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_history():
    return _load_json(HISTORY_FILE, [])


def save_history(history):
    _save_json(HISTORY_FILE, history)


# ============================================================
# Feedback & Learned Rules storage
# ============================================================

def load_feedback():
    return _load_json(FEEDBACK_FILE, [])


def save_feedback_entry(entry):
    fb = load_feedback()
    fb.append(entry)
    _save_json(FEEDBACK_FILE, fb)
    _maybe_trigger_reflection(fb)


def load_learned_rules():
    return _load_json(LEARNED_RULES_FILE, {
        "version": 0,
        "reflection_count": 0,
        "last_reflection": 0,
        "global_rules": {"positive": [], "negative": []},
        "emotion_profiles": {},
        "param_insights": {},
    })


def save_learned_rules(rules):
    _save_json(LEARNED_RULES_FILE, rules)


# ============================================================
# Reflection Engine (3-phase)
# ============================================================

def _maybe_trigger_reflection(feedback_list):
    rules = load_learned_rules()
    entries_since = len(feedback_list) - (rules.get("reflection_count", 0) * REFLECTION_THRESHOLD)
    if entries_since >= REFLECTION_THRESHOLD:
        run_reflection(feedback_list)


def run_reflection(feedback_list=None):
    if feedback_list is None:
        feedback_list = load_feedback()
    if len(feedback_list) < REFLECTION_THRESHOLD:
        return

    rules = load_learned_rules()

    # Phase A: Global rules
    global_rules = _reflect_global_rules(feedback_list)

    # Phase B: Per-emotion profiles
    emotion_profiles = _reflect_emotion_profiles(feedback_list)

    # Phase C: Parameter insights
    param_insights = _compute_param_insights(feedback_list)

    rules["version"] = rules.get("version", 0) + 1
    rules["reflection_count"] = rules.get("reflection_count", 0) + 1
    rules["last_reflection"] = time.time()
    rules["global_rules"] = global_rules
    rules["emotion_profiles"] = emotion_profiles
    rules["param_insights"] = param_insights

    save_learned_rules(rules)
    print(f"Reflection #{rules['reflection_count']} complete. "
          f"Rules: {len(global_rules.get('positive', []))}+ / {len(global_rules.get('negative', []))}-, "
          f"Emotion profiles: {len(emotion_profiles)}")


def _reflect_global_rules(feedback_list):
    high = [f for f in feedback_list if f.get("rating", 0) >= 4]
    low = [f for f in feedback_list if f.get("rating", 0) <= 2]

    text = "HIGH-RATED sessions:\n"
    for f in high[-10:]:
        text += f"- Rating {f['rating']}, Prompt: {f.get('enhanced_prompt', '')}, Notes: {f.get('notes', '')}\n"
    text += "\nLOW-RATED sessions:\n"
    for f in low[-10:]:
        text += f"- Rating {f['rating']}, Prompt: {f.get('enhanced_prompt', '')}, Notes: {f.get('notes', '')}\n"

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Analyze music generation feedback. Compare high-rated vs low-rated prompts. "
                "Extract actionable rules. Return JSON:\n"
                '{"positive": ["rule1", "rule2", ...], "negative": ["anti-pattern1", ...]}\n'
                "Max 4 rules each. Be specific and actionable. Output ONLY valid JSON."
            )},
            {"role": "user", "content": text},
        ],
        max_tokens=300,
    )
    return json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))


def _reflect_emotion_profiles(feedback_list):
    by_emotion = {}
    for f in feedback_list:
        emotion = f.get("emotion_profile", {}).get("emotion", "")
        if emotion:
            by_emotion.setdefault(emotion, []).append(f)

    profiles = {}
    for emotion, entries in by_emotion.items():
        if len(entries) < 2:
            continue

        text = f"All sessions for emotion '{emotion}':\n"
        for f in entries[-10:]:
            p = f.get("emotion_profile", {})
            text += (
                f"- Rating {f.get('rating', '?')}, "
                f"Energy={p.get('energy', '?')}, Style={p.get('style', '?')}, "
                f"Warmth={p.get('warmth', '?')}, Arc={p.get('arc', '?')}, "
                f"Prompt: {f.get('enhanced_prompt', '')}, Notes: {f.get('notes', '')}\n"
            )

        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    f"Analyze feedback for emotion '{emotion}' in music generation. Return JSON:\n"
                    '{"preferred_params": {"energy": [low, high], "style": [low, high], '
                    '"warmth": [low, high], "arc": [low, high]}, '
                    '"prompt_principles": ["principle1", ...], '
                    '"anti_patterns": ["pattern1", ...], '
                    '"best_prompt_template": "a template prompt"}\n'
                    "Derive ranges from high-rated sessions. Max 3 principles/anti_patterns. "
                    "Output ONLY valid JSON."
                )},
                {"role": "user", "content": text},
            ],
            max_tokens=300,
        )
        profiles[emotion] = json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))

    return profiles


def _compute_param_insights(feedback_list):
    entries = [f for f in feedback_list if f.get("gen_params")]
    if len(entries) < 3:
        return {}

    text = "Generation parameters and ratings:\n"
    for f in entries[-15:]:
        p = f["gen_params"]
        text += f"- Rating {f.get('rating', '?')}, temp={p.get('temperature', '?')}, guidance={p.get('guidance_scale', '?')}, tokens={p.get('max_new_tokens', '?')}\n"

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Analyze correlation between MusicGen parameters and user ratings. Return JSON:\n"
                '{"temperature": "insight", "guidance_scale": "insight", "max_new_tokens": "insight"}\n'
                "Be concise. Output ONLY valid JSON."
            )},
            {"role": "user", "content": text},
        ],
        max_tokens=200,
    )
    return json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))


# ============================================================
# Learned knowledge accessors
# ============================================================

def get_learned_defaults(emotion: str) -> dict:
    feedback = load_feedback()
    matching = [f for f in feedback if f.get("rating", 0) >= 4 and f.get("emotion_profile", {}).get("emotion") == emotion]
    if len(matching) < 2:
        return {}
    result = {}
    for k in ["energy", "style", "warmth", "arc"]:
        vals = [f["emotion_profile"][k] for f in matching if k in f.get("emotion_profile", {})]
        if vals:
            result[k] = round(sum(vals) / len(vals))
    return result


def get_emotion_profile(emotion: str) -> dict:
    return load_learned_rules().get("emotion_profiles", {}).get(emotion, {})


def get_top_prompts(emotion: str) -> list:
    feedback = load_feedback()
    matching = sorted(
        [f for f in feedback if f.get("rating", 0) >= 4 and f.get("emotion_profile", {}).get("emotion") == emotion],
        key=lambda x: x.get("rating", 0), reverse=True,
    )
    return [f["enhanced_prompt"] for f in matching[:3] if f.get("enhanced_prompt")]


def get_negative_examples(emotion: str) -> list:
    feedback = load_feedback()
    matching = sorted(
        [f for f in feedback if f.get("rating", 0) <= 2 and f.get("emotion_profile", {}).get("emotion") == emotion],
        key=lambda x: x.get("rating", 0),
    )
    return [f["enhanced_prompt"] for f in matching[:3] if f.get("enhanced_prompt")]


def get_learned_rules() -> dict:
    return load_learned_rules().get("global_rules", {"positive": [], "negative": []})


# ============================================================
# Analysis functions (run concurrently)
# ============================================================

def analyze_text(text: str) -> dict:
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Analyze the emotional content of this text. Return JSON with: "
                '"summary" (1 sentence), "moods" (list of 1-3 detected emotions), '
                '"mood" (dominant emotion, single word), "energy" (0.0-1.0 float). '
                "Output ONLY valid JSON."
            )},
            {"role": "user", "content": text},
        ],
        max_tokens=150,
    )
    result = json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))
    result["source"] = "text"
    return result


def analyze_images(images_b64: list[str]) -> dict:
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in images_b64]
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Analyze the mood and emotion of these images together. Return JSON with: "
                '"caption" (1 sentence scene description), "moods" (list of 1-3 detected emotions), '
                '"mood" (dominant emotion, single word), "energy" (0.0-1.0 float). '
                "Output ONLY valid JSON."
            )},
            {"role": "user", "content": content},
        ],
        max_tokens=150,
    )
    result = json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))
    result["source"] = "image"
    return result


def analyze_voice(audio_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"].strip()
    finally:
        os.unlink(tmp_path)

    if not transcript:
        return {"source": "voice", "transcript": "", "moods": ["neutral"], "mood": "neutral", "energy": 0.5}

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "This is a voice transcription. Analyze the emotional tone. Return JSON with: "
                '"transcript" (the text), "moods" (list of 1-3 detected emotions), '
                '"mood" (dominant emotion, single word), "energy" (0.0-1.0 float). '
                "Output ONLY valid JSON."
            )},
            {"role": "user", "content": transcript},
        ],
        max_tokens=150,
    )
    result = json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))
    result["source"] = "voice"
    return result


# ============================================================
# Emotion fusion with range-clamping
# ============================================================

def _range_clamp(ai_value, learned_range):
    """Nudge AI value toward learned range if outside it.
    Within range: keep AI value. Outside: blend 70% toward nearest boundary."""
    lo, hi = learned_range
    if lo <= ai_value <= hi:
        return ai_value
    nearest = lo if ai_value < lo else hi
    return round(ai_value * 0.3 + nearest * 0.7)


def fuse_emotions(analyses: list[dict]) -> dict:
    signals_str = json.dumps(analyses, indent=2)
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an emotion-to-music translator. Given emotional signals from different sources "
                "(text, image, voice), each with 1-3 detected moods, synthesize ONE unified profile. "
                "Blend ALL detected emotions into slider values, not just the dominant one. Return JSON:\n"
                '- "emotion": dominant emotion word\n'
                '- "emotions": list of all detected emotions (deduplicated)\n'
                '- "energy": 0-100\n'
                '- "style": 0-100 (0=lo-fi intimate, 100=cinematic epic)\n'
                '- "warmth": 0-100 (0=warm analog, 100=bright digital)\n'
                '- "arc": 0-100 (0=steady, 100=dramatic build)\n'
                '- "description": 1 sentence emotional landscape\n'
                "Output ONLY valid JSON."
            )},
            {"role": "user", "content": signals_str},
        ],
        max_tokens=200,
    )
    profile = json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))
    profile["sources"] = [a["source"] for a in analyses]

    # Apply range-clamping from learned emotion profile
    emotion = profile.get("emotion", "")
    ep = get_emotion_profile(emotion)
    if ep and "preferred_params" in ep:
        for key in ["energy", "style", "warmth", "arc"]:
            if key in ep["preferred_params"] and key in profile:
                profile[key] = _range_clamp(profile[key], ep["preferred_params"][key])
    else:
        # Fallback: blend with learned defaults
        learned = get_learned_defaults(emotion)
        if learned:
            for key in ["energy", "style", "warmth", "arc"]:
                if key in learned and key in profile:
                    profile[key] = round(profile[key] * 0.7 + learned[key] * 0.3)

    return profile


# ============================================================
# Prompt construction with knowledge injection
# ============================================================

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


def _build_knowledge_context(emotion: str) -> str:
    parts = []

    top = get_top_prompts(emotion)
    if top:
        parts.append("POSITIVE EXAMPLES (emulate these styles):\n" + "\n".join(f"- {p}" for p in top))

    neg = get_negative_examples(emotion)
    if neg:
        parts.append("ANTI-PATTERNS (avoid these):\n" + "\n".join(f"- {p}" for p in neg))

    ep = get_emotion_profile(emotion)
    if ep:
        if ep.get("prompt_principles"):
            parts.append("LEARNED PRINCIPLES:\n" + "\n".join(f"- {p}" for p in ep["prompt_principles"]))
        if ep.get("anti_patterns"):
            parts.append("LEARNED ANTI-PATTERNS:\n" + "\n".join(f"- {p}" for p in ep["anti_patterns"]))

    rules = get_learned_rules()
    if rules.get("positive"):
        parts.append("GLOBAL RULES (always follow):\n" + "\n".join(f"- {r}" for r in rules["positive"]))
    if rules.get("negative"):
        parts.append("GLOBAL ANTI-RULES (always avoid):\n" + "\n".join(f"- {r}" for r in rules["negative"]))

    return "\n\n".join(parts)


def profile_to_prompt(profile: dict) -> str:
    energy_desc = map_slider(profile.get("energy", 50), ENERGY_MAP)
    style_desc = map_slider(profile.get("style", 50), STYLE_MAP)
    warmth_desc = map_slider(profile.get("warmth", 50), WARMTH_MAP)
    arc_desc = map_slider(profile.get("arc", 50), ARC_MAP)

    knowledge = _build_knowledge_context(profile.get("emotion", ""))

    system_parts = [
        "You are a music director. Create a vivid, specific MusicGen prompt (2-3 sentences) that combines these musical qualities:",
        f"Energy: {energy_desc}",
        f"Style: {style_desc}",
        f"Warmth: {warmth_desc}",
        f"Arc: {arc_desc}",
        f"Core emotion: {profile.get('emotion', 'neutral')} — {profile.get('description', '')}",
        f"All emotions to blend: {', '.join(profile.get('emotions', [profile.get('emotion', 'neutral')]))}",
        "Include specific instruments, tempo, key, and production style. Output ONLY the prompt text.",
    ]
    if knowledge:
        system_parts.append("\n--- LEARNED KNOWLEDGE ---\n" + knowledge)

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": "Generate the MusicGen prompt."},
        ],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# Explainability
# ============================================================

def explain_generation(original_inputs: str, profile: dict, music_prompt: str) -> dict:
    rules = load_learned_rules()
    learning_note = ""
    if rules.get("reflection_count", 0) > 0:
        emotions_learned = list(rules.get("emotion_profiles", {}).keys())
        learning_note = (
            f"\nThe system has completed {rules['reflection_count']} reflection cycles "
            f"and learned about emotions: {', '.join(emotions_learned) if emotions_learned else 'none yet'}. "
            "Mention how learning influenced this generation if relevant."
        )

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are explaining how AI transformed a user's input into music. "
                "Write 2-3 sentences connecting the input, the detected emotions, and the musical output. "
                "Be poetic but concise. Also return a timeline of 5 steps and key descriptors. Return JSON:\n"
                '"narrative": "2-3 sentence explanation",\n'
                '"timeline": [{"step": "name", "detail": "what happened"}] (5 steps),\n'
                '"key_descriptors": ["descriptor1", ...] (4-6 musical descriptors)\n'
                "Output ONLY valid JSON." + learning_note
            )},
            {"role": "user", "content": json.dumps({
                "inputs": original_inputs,
                "profile": profile,
                "music_prompt": music_prompt,
            })},
        ],
        max_tokens=400,
    )
    return json.loads(resp.choices[0].message.content.strip().strip("`").removeprefix("json"))


# ============================================================
# Audio generation — batched A/B
# ============================================================

def generate_audio(prompt: str, duration_tokens: int, num_variations: int = 2) -> list[dict]:
    prompts = [prompt]
    if num_variations >= 2:
        prompts.append(prompt + " with subtle variation in rhythm and texture")

    inputs = processor(text=prompts, padding=True, return_tensors="pt").to(DEVICE)
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=duration_tokens)

    ts = int(time.time())
    results = []
    for i in range(len(prompts)):
        version = chr(65 + i)  # A, B
        filename = f"gen_{ts}_{version}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=audio_values[i, 0].cpu().numpy())
        results.append({
            "filename": filename,
            "version": version,
            "audio_url": f"/audio/{filename}",
        })

    return results


# ============================================================
# API Endpoints
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str
    duration: int = 256
    energy: int | None = None
    style: int | None = None
    warmth: int | None = None
    arc: int | None = None


@app.post("/generate")
def generate(req: GenerateRequest):
    analysis = analyze_text(req.prompt)
    profile = fuse_emotions([analysis])

    overrides = []
    for key in ["energy", "style", "warmth", "arc"]:
        val = getattr(req, key)
        if val is not None:
            profile[key] = val
            overrides.append(key)
    profile["overrides"] = overrides

    music_prompt = profile_to_prompt(profile)
    variations = generate_audio(music_prompt, req.duration)
    explanation = explain_generation(req.prompt, profile, music_prompt)

    gen_id = f"gen_{int(time.time())}"
    entry = {
        "id": gen_id,
        "original_prompt": req.prompt,
        "enhanced_prompt": music_prompt,
        "emotion_profile": profile,
        "explanation": explanation,
        "variations": variations,
        "duration_tokens": req.duration,
        "timestamp": time.time(),
        "source": "text",
        "gen_params": {"temperature": 1.0, "guidance_scale": 3, "max_new_tokens": req.duration},
    }
    history = load_history()
    history.insert(0, entry)
    save_history(history)

    return {
        "id": gen_id,
        "enhanced_prompt": music_prompt,
        "emotion_profile": profile,
        "explanation": explanation,
        "variations": variations,
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
    analyses = []
    input_desc_parts = []

    with ThreadPoolExecutor() as executor:
        futures = {}

        if text.strip():
            futures[executor.submit(analyze_text, text.strip())] = "text"
            input_desc_parts.append(f"Text: {text.strip()}")

        if images:
            images_bytes = [await img.read() for img in images]
            images_b64 = [base64.b64encode(b).decode("utf-8") for b in images_bytes]
            futures[executor.submit(analyze_images, images_b64)] = "image"
            input_desc_parts.append(f"Images: {len(images)} uploaded")

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

    profile = fuse_emotions(analyses)

    overrides = []
    for key, val in [("energy", energy), ("style", style), ("warmth", warmth), ("arc", arc)]:
        if val is not None:
            profile[key] = val
            overrides.append(key)
    profile["overrides"] = overrides

    music_prompt = profile_to_prompt(profile)
    variations = generate_audio(music_prompt, duration)
    input_desc = " | ".join(input_desc_parts)
    explanation = explain_generation(input_desc, profile, music_prompt)

    gen_id = f"gen_{int(time.time())}"
    entry = {
        "id": gen_id,
        "original_prompt": input_desc,
        "enhanced_prompt": music_prompt,
        "emotion_profile": profile,
        "explanation": explanation,
        "analyses": analyses,
        "variations": variations,
        "duration_tokens": duration,
        "timestamp": time.time(),
        "source": "multimodal",
        "gen_params": {"temperature": 1.0, "guidance_scale": 3, "max_new_tokens": duration},
    }
    history = load_history()
    history.insert(0, entry)
    save_history(history)

    return {
        "id": gen_id,
        "enhanced_prompt": music_prompt,
        "emotion_profile": profile,
        "explanation": explanation,
        "analyses": analyses,
        "variations": variations,
    }


@app.get("/audio/{filename}")
def get_audio(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(filepath, media_type="audio/wav", filename=filename)


class FeedbackRequest(BaseModel):
    rating: int  # 1-5
    replay: bool = False
    preferred_version: str = ""  # "A", "B", or ""
    notes: str = ""


@app.post("/feedback/{gen_id}")
def submit_feedback(gen_id: str, req: FeedbackRequest):
    history = load_history()
    gen_entry = None
    for entry in history:
        if entry["id"] == gen_id:
            gen_entry = entry
            entry["rating"] = req.rating
            entry["replay"] = req.replay
            entry["preferred_version"] = req.preferred_version
            break
    save_history(history)

    if not gen_entry:
        return {"error": "Generation not found"}

    fb_entry = {
        "gen_id": gen_id,
        "rating": req.rating,
        "replay": req.replay,
        "preferred_version": req.preferred_version,
        "notes": req.notes,
        "original_prompt": gen_entry.get("original_prompt", ""),
        "enhanced_prompt": gen_entry.get("enhanced_prompt", ""),
        "emotion_profile": gen_entry.get("emotion_profile", {}),
        "gen_params": gen_entry.get("gen_params", {}),
        "timestamp": time.time(),
    }
    save_feedback_entry(fb_entry)

    # Return learning stats
    rules = load_learned_rules()
    feedback_list = load_feedback()
    total = len(feedback_list)
    avg_rating = round(sum(f.get("rating", 0) for f in feedback_list) / total, 1) if total else 0
    replay_rate = round(sum(1 for f in feedback_list if f.get("replay")) / total * 100) if total else 0
    entries_since = total - (rules.get("reflection_count", 0) * REFLECTION_THRESHOLD)

    return {
        "ok": True,
        "learning_stats": {
            "total_sessions": total,
            "avg_rating": avg_rating,
            "replay_rate": replay_rate,
            "reflections_completed": rules.get("reflection_count", 0),
            "rules_active": len(rules.get("global_rules", {}).get("positive", [])) + len(rules.get("global_rules", {}).get("negative", [])),
            "emotions_learned": list(rules.get("emotion_profiles", {}).keys()),
            "next_reflection_in": max(0, REFLECTION_THRESHOLD - entries_since),
        },
    }


@app.get("/history")
def get_history():
    return load_history()


@app.get("/top")
def top_generations():
    history = load_history()
    return sorted(history, key=lambda x: x.get("rating", 0), reverse=True)[:20]


@app.get("/learned")
def get_learned():
    """What the system has learned — for the Insights panel."""
    rules = load_learned_rules()
    feedback = load_feedback()
    total = len(feedback)
    return {
        "reflection_count": rules.get("reflection_count", 0),
        "global_rules": rules.get("global_rules", {"positive": [], "negative": []}),
        "emotion_profiles": {
            emotion: {
                "prompt_principles": p.get("prompt_principles", []),
                "anti_patterns": p.get("anti_patterns", []),
                "preferred_params": p.get("preferred_params", {}),
            }
            for emotion, p in rules.get("emotion_profiles", {}).items()
        },
        "param_insights": rules.get("param_insights", {}),
        "total_sessions": total,
        "emotions_learned": list(rules.get("emotion_profiles", {}).keys()),
        "next_reflection_in": max(0, REFLECTION_THRESHOLD - (total - rules.get("reflection_count", 0) * REFLECTION_THRESHOLD)),
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME}