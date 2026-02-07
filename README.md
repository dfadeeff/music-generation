# Music to My Ears

AI music generation using Meta's MusicGen via HuggingFace Transformers.

Built for the Global Hackathon — VC Track: Building the AI Backbone of the Super Bowl Halftime Music Industry.

## Architecture

```
Frontend (Next.js on Vercel)  →  POST /generate  →  Backend (FastAPI + MusicGen)  →  .wav audio
```

- **Backend** — FastAPI server that loads MusicGen and generates audio from text prompts
- **Frontend** — Next.js app with prompt input, duration selector, preset prompts, and audio player

## Run Locally (before deploying)

You need two terminal windows — one for the backend, one for the frontend.

### 1. Clone and install

```bash
git clone https://github.com/dfadeeff/music-generation.git
cd music-generation
```

### 2. Start the backend (Terminal 1)

```bash
uv sync                                                  # installs Python deps into .venv
uv run uvicorn backend.main:app --reload --port 8000     # starts API on http://localhost:8000
```

First run downloads the MusicGen model (~2.4GB). After that it loads from cache in seconds.
You should see:

```
Loading MusicGen on mps...
Model loaded. Sample rate: 32000Hz
Uvicorn running on http://127.0.0.1:8000
```

Test it works:

```bash
curl -X POST http://localhost:8000/health
# → {"status":"ok","device":"mps","model":"facebook/musicgen-small"}
```

### 3. Start the frontend (Terminal 2)

```bash
cd frontend
npm install       # first time only
npm run dev       # starts Next.js on http://localhost:3000
```

### 4. Use it

Open http://localhost:3000 in your browser:

1. Type a prompt (e.g. "epic Super Bowl halftime show, energetic drums, brass section")
2. Pick duration: 5s / 10s / 20s
3. Hit **Generate** — wait 15-60 seconds depending on duration and device
4. Audio plays automatically in the browser

## API

### POST /generate

```json
{
  "prompt": "epic Super Bowl halftime show, energetic drums, brass section",
  "duration": 256
}
```

Returns a `.wav` audio file. Duration is in tokens: 256 = ~5s, 512 = ~10s, 1024 = ~20s.

### GET /health

Returns server status and device info.

## How it works

1. Text prompt → T5 tokenizer
2. Frozen T5 text encoder → hidden states
3. MusicGen decoder → discrete audio tokens
4. EnCodec decoder → 32kHz audio waveform

Auto-detects device: Apple Silicon GPU (MPS), NVIDIA GPU (CUDA), or CPU.

## Models

| Model                       | Size  | Quality          |
|-----------------------------|-------|------------------|
| `facebook/musicgen-small`   | 300MB | Good (default)   |
| `facebook/musicgen-medium`  | 1.5GB | Better           |
| `facebook/musicgen-large`   | 3.3GB | Best             |

## Deployment

### Frontend → Vercel

1. Push repo to GitHub
2. Import `frontend/` directory in Vercel
3. Set environment variable: `NEXT_PUBLIC_API_URL` = your backend URL (e.g. `https://your-backend.fly.dev`)
4. Deploy

### Backend → any server with Python

Vercel cannot run PyTorch. Deploy the backend separately:

- **HuggingFace Spaces** (free T4 GPU) — easiest for hackathon
- **Railway** / **Fly.io** — simple container deploy
- **Your MacBook** + ngrok — quickest for demo

```bash
# expose local backend publicly with ngrok
ngrok http 8000
# then set NEXT_PUBLIC_API_URL to the ngrok URL in Vercel
```
