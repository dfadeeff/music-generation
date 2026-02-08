# Music to My Ears

AI music generation platform that transforms text, images, and voice into emotionally-targeted music using Meta's MusicGen.

Built for the Global Hackathon — VC Track: Building the AI Backbone of the Super Bowl Halftime Music Industry.

## Architecture

```
User Input (text / images / voice)
        ↓
  Concurrent Emotion Analysis (GPT-4o-mini + Whisper)
        ↓
  6D Emotion Fusion (intensity, mood, complexity, tempo, texture, narrative)
        ↓
  Knowledge-Injected Prompt Generation
        ↓
  MusicGen Batched Generation → 3 Variations (A / B / C)
        ↓
  Feedback → Reflection Engine → Learned Rules → Better Generations
```

- **Backend** — FastAPI server with MusicGen, Whisper, multimodal emotion analysis, reflection learning
- **Frontend** — Next.js 14 app with multimodal input, 6 emotion sliders, A/B/C comparison, explainability, feedback loop

## Run Locally

You need two terminal windows — one for the backend, one for the frontend.

### 1. Clone and install

```bash
git clone https://github.com/dfadeeff/music-generation.git
cd music-generation
```

### 2. Set up environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Start the backend (Terminal 1)

```bash
uv sync                                                  # installs Python deps into .venv
uv run uvicorn backend.main:app --reload --port 8000     # starts API on http://localhost:8000
```

First run downloads the MusicGen model and Whisper model. After that they load from cache.
You should see:

```
Loading MusicGen on mps...
Model loaded. Sample rate: 32000Hz
Loading Whisper...
Whisper loaded.
Uvicorn running on http://127.0.0.1:8000
```

Test it works:

```bash
curl http://localhost:8000/health
# → {"status":"ok","device":"mps","model":"facebook/musicgen-medium"}
```

### 4. Start the frontend (Terminal 2)

```bash
cd frontend
npm install       # first time only
npm run dev       # starts Next.js on http://localhost:3000
```

### 5. Use it

Open http://localhost:3000 in your browser:

1. **Create** — enter text, upload images, or record voice (any combination)
2. Adjust 6 emotion sliders (Intensity, Mood, Complexity, Tempo, Texture, Narrative) or leave on Auto
3. Pick duration: 5s / 10s / 20s
4. Hit **Generate** — wait ~30-90 seconds for 3 variations
5. Listen to A / B / C, pick your favorite, rate 1-5, submit feedback
6. After 5 ratings the system runs a **reflection cycle** and learns your taste
7. Check **Insights** tab to see what the system has learned

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Text-only generation with optional slider overrides |
| `POST` | `/generate-multimodal` | Multimodal generation (text + images + voice) |
| `GET` | `/audio/{filename}` | Stream generated audio file |
| `POST` | `/feedback/{gen_id}` | Submit rating, replay, preference, notes |
| `GET` | `/history` | All past generations |
| `GET` | `/top` | Top-rated generations |
| `GET` | `/learned` | What the system has learned (rules, profiles, insights) |
| `GET` | `/health` | Server status and device info |

## Key Features

- **Multimodal Input** — text, images, and voice analyzed concurrently
- **6 Musical Dimensions** — Intensity, Mood, Complexity, Tempo, Texture, Narrative
- **3-Way A/B/C Comparison** — three distinct variations per generation
- **3-Phase Reflection Engine** — learns global rules, per-emotion profiles, and parameter insights
- **Range-Clamping** — applies learned preferences without overriding AI judgment
- **Knowledge Injection** — top prompts, anti-patterns, and principles fed into future generations
- **Explainability** — narrative + timeline showing how inputs became music

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Music Generation | Meta MusicGen-medium (HuggingFace Transformers) |
| Emotion Analysis | OpenAI GPT-4o-mini |
| Voice Transcription | OpenAI Whisper (base) |
| Backend | FastAPI (Python) |
| Frontend | Next.js 14 (React) |
| Package Manager | uv |
| GPU Support | Apple MPS / NVIDIA CUDA / CPU |

## Models

| Model | Download Size | Quality |
|-------|--------------|---------|
| `facebook/musicgen-small` | ~1.3 GB | Good |
| `facebook/musicgen-medium` | ~8 GB | Better (default) |
| `facebook/musicgen-large` | ~13 GB | Best |

Change the model in `backend/main.py` by updating `MODEL_NAME`.

## How It Works

1. **Input Analysis** — Text/images analyzed by GPT-4o-mini, voice transcribed by Whisper, all run concurrently
2. **Emotion Fusion** — Multiple emotion signals blended into a 6D profile with range-clamping from learned preferences
3. **Prompt Generation** — 6D profile + knowledge injection → rich MusicGen prompt via GPT-4o-mini
4. **Audio Generation** — MusicGen processes 3 prompt variations in a single batched forward pass
5. **Feedback Loop** — User ratings trigger reflection every 5 sessions, extracting rules and preferences

Auto-detects device: Apple Silicon GPU (MPS), NVIDIA GPU (CUDA), or CPU.

## Dependencies

- Python 3.11+
- Node.js 18+
- OpenAI API key (for GPT-4o-mini and emotion analysis)

## Deployment

### Frontend → Vercel

1. Push repo to GitHub
2. Import `frontend/` directory in Vercel
3. Set environment variable: `NEXT_PUBLIC_API_URL` = your backend URL
4. Deploy

### Backend → any server with Python + GPU

- **HuggingFace Spaces** (free T4 GPU)
- **Railway** / **Fly.io** — container deploy
- **Local** + ngrok — quickest for demo

```bash
ngrok http 8000
# then set NEXT_PUBLIC_API_URL to the ngrok URL
```