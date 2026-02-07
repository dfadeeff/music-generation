"""Core music generation using MusicGen via HuggingFace Transformers."""

from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torch
import scipy
import os
import time

# Pick the best available device
if torch.backends.mps.is_available():
    DEVICE = "mps"       # Apple Silicon GPU (your MacBook Pro)
elif torch.cuda.is_available():
    DEVICE = "cuda"      # NVIDIA GPU
else:
    DEVICE = "cpu"

# Load model once, move to GPU
print(f"Loading MusicGen model on {DEVICE}...")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
print("Model loaded.")

SAMPLE_RATE = model.config.audio_encoder.sampling_rate
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_music(prompt: str, duration_tokens: int = 256) -> str:
    """Generate music from a text prompt. Returns path to .wav file."""
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(DEVICE)
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=duration_tokens)

    filename = f"gen_{int(time.time())}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=audio_values[0, 0].cpu().numpy())
    return filepath


def image_to_music_prompt(image_description: str) -> str:
    """Convert an image description into a MusicGen-friendly prompt.

    In a full pipeline, this calls an LLM (Claude/GPT) to translate
    visual/emotional context into musical direction. For now, a template.
    """
    return (
        f"A musical piece inspired by: {image_description}. "
        "Cinematic, emotionally rich, with dynamic instrumentation."
    )


if __name__ == "__main__":
    prompt = "epic Super Bowl halftime show, energetic drums, brass section, crowd cheering"
    print(f"Generating music for: {prompt}")
    path = generate_music(prompt, duration_tokens=256)
    print(f"Saved to: {path}")