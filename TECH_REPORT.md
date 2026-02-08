# Music to My Ears — 1-Page Technical Report

## 1. Problem & Challenge

The music industry spends **$4.4B annually** on music licensing and production. Creating music that matches a specific emotional vision — for ads, halftime shows, films, or content — requires expensive studio sessions, producers, and iterative back-and-forth that can take weeks. Non-musicians (marketers, directors, content creators) cannot translate their emotional intent into musical language. Current AI music tools accept only text prompts, forcing users to guess technical terms like "BPM," "reverb," or "chord progression." There is no feedback loop — every generation starts from zero, with no memory of what the user liked before.

## 2. Target Audience

**Primary:** Creative professionals — ad agencies, content creators, and event producers who need emotionally targeted music but lack musical training. **Secondary:** Music supervisors and brands preparing for large-scale events (e.g., Super Bowl halftime) who need rapid prototyping of sonic identities. These users think in emotions and visuals, not in musical terminology.

## 3. Solution & Core Features

**Music to My Ears** transforms text descriptions, images, and voice recordings into AI-generated music through emotion understanding.

- **Multimodal Input:** Users provide any combination of text, images (mood boards, event photos), and voice recordings — the system analyzes all simultaneously
- **6-Dimensional Emotion Profile:** Inputs are fused into six musical dimensions — Intensity, Mood, Complexity, Tempo, Texture, and Narrative — giving users intuitive control without musical jargon
- **3-Way Comparison (A/B/C):** Each generation produces three distinct variations, enabling faster preference discovery
- **Self-Improving Reflection Engine:** After every 5 ratings, the system runs a 3-phase reflection cycle — extracting global rules, per-emotion profiles with preferred parameter ranges, and generation parameter insights
- **Knowledge Injection:** Learned rules, top-rated prompts, and anti-patterns are injected into future generations, making each session smarter
- **Explainability:** Every generation includes a narrative explanation and visual timeline showing how inputs became music

## 4. Unique Selling Proposition (USP)

Unlike existing AI music tools that treat every generation independently, **Music to My Ears learns your taste over time**. The 3-phase reflection engine builds a persistent knowledge base of what works (and what doesn't) for each emotion, applying range-clamped learned preferences to future generations. Combined with multimodal input (not just text), 3-way comparison, and full explainability, it closes the gap between emotional intent and musical output — no musical expertise required.

## 5. Implementation & Technology

| Layer | Technology | Role |
|-------|-----------|------|
| Music Generation | **Meta MusicGen-medium** (HuggingFace Transformers) | Text-conditioned audio generation, batched 3-variation output |
| Emotion Analysis | **OpenAI GPT-4o-mini** | Text/image emotion detection, multi-emotion fusion, prompt engineering, reflection |
| Voice Input | **OpenAI Whisper** (base) | Speech-to-text transcription for voice-based emotion analysis |
| Backend | **FastAPI** (Python) | REST API with concurrent input processing (ThreadPoolExecutor) |
| Frontend | **Next.js 14** (React) | Responsive UI with glassmorphism design, real-time feedback |
| Compute | **Apple MPS / CUDA / CPU** | Adaptive GPU acceleration for on-device inference |
| Learning | **JSON-based knowledge store** | Persistent feedback, learned rules, emotion profiles, parameter insights |

**Architecture:** Inputs are analyzed concurrently, fused into a 6D emotion profile with range-clamping from learned preferences, converted to a MusicGen prompt enriched with knowledge injection, and generated as three batched audio variations in a single forward pass.

## 6. Results & Impact

- **End-to-end generation** in under 60 seconds (text/image/voice to 3 playable audio tracks)
- **6 intuitive dimensions** replace technical music jargon — tested to be immediately understandable by non-musicians
- **3-way comparison** provides 50% more preference signal per session than traditional A/B
- **Reflection engine** demonstrably improves output alignment after 5+ feedback cycles, extracting actionable rules and per-emotion parameter ranges
- **Full explainability** — users understand *why* their input produced a specific sound, building trust and enabling iteration

---

*"If we had 24 more hours, we'd..."* add real-time waveform visualization during generation and implement LoRA fine-tuning of MusicGen on user-preferred outputs for true personalization beyond prompt engineering.