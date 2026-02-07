# Music to My Ears

AI music generation using Meta's MusicGen via HuggingFace Transformers.

Built for the Global Hackathon — VC Track: Building the AI Backbone of the Super Bowl Halftime Music Industry.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### CLI
```bash
python generate.py
```

### Web UI (Gradio)
```bash
python app.py
```

Opens a browser with two tabs:
- **Text → Music** — describe what you want, get a .wav
- **Scene → Music** — describe a scene, AI converts it to a music prompt and generates

## How it works

1. Text prompt → MusicGen processor (T5 tokenizer)
2. Frozen T5 text encoder → hidden states
3. MusicGen decoder → discrete audio tokens
4. EnCodec decoder → 32kHz audio waveform

Runs on Apple Silicon GPU (MPS) automatically. Falls back to CPU if needed.

## Models

| Model | Size | Quality |
|-------|------|---------|
| `facebook/musicgen-small` | 300MB | Good (default) |
| `facebook/musicgen-medium` | 1.5GB | Better |
| `facebook/musicgen-large` | 3.3GB | Best |

## Token-to-duration

- 256 tokens ≈ 5 seconds
- 512 tokens ≈ 10 seconds
- 1024 tokens ≈ 20 seconds