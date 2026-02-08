"use client";

import { useState, useRef, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const DURATIONS = [
  { label: "5s", tokens: 256 },
  { label: "10s", tokens: 512 },
  { label: "20s", tokens: 1024 },
];

const PRESETS = [
  "Epic Super Bowl halftime, energetic drums, brass section, crowd energy",
  "Chill lo-fi beat, rainy mood, soft piano, vinyl crackle",
  "90s hip-hop, heavy bass, scratching, boom bap drums",
  "Orchestral cinematic, rising strings, dramatic tension",
  "Tropical house, steel drums, upbeat synth, summer vibes",
  "Dark ambient, deep drone, eerie atmosphere, suspenseful",
];

export default function Home() {
  const [tab, setTab] = useState("text");
  const [prompt, setPrompt] = useState("");
  const [duration, setDuration] = useState(256);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [images, setImages] = useState([]);
  const [imagePreviews, setImagePreviews] = useState([]);
  const [imageText, setImageText] = useState("");
  const fileRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/history`).then(r => r.json()).then(setHistory).catch(() => {});
  }, []);

  const refreshHistory = async () => {
    const r = await fetch(`${API}/history`);
    setHistory(await r.json());
  };

  const generate = async () => {
    if (tab === "text" && !prompt.trim()) return;
    if (tab === "image" && images.length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let res;
      if (tab === "image") {
        const fd = new FormData();
        images.forEach(f => fd.append("images", f));
        fd.append("text", imageText);
        fd.append("duration", duration);
        res = await fetch(`${API}/generate-from-image`, { method: "POST", body: fd });
      } else {
        res = await fetch(`${API}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: prompt.trim(), duration }),
        });
      }
      if (!res.ok) throw new Error("Generation failed");
      const data = await res.json();
      setResult({ ...data, audioUrl: `${API}${data.audio_url}` });
      await refreshHistory();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const rate = async (id, rating) => {
    await fetch(`${API}/rate/${id}?rating=${rating}`, { method: "POST" });
    await refreshHistory();
  };

  const addImages = (e) => {
    const files = Array.from(e.target.files);
    setImages(prev => [...prev, ...files]);
    setImagePreviews(prev => [...prev, ...files.map(f => URL.createObjectURL(f))]);
  };

  const removeImage = (i) => {
    setImages(prev => prev.filter((_, idx) => idx !== i));
    setImagePreviews(prev => prev.filter((_, idx) => idx !== i));
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !loading) { e.preventDefault(); generate(); }
  };

  const tabs = [
    { id: "text", label: "Text to Music" },
    { id: "image", label: "Image to Music" },
    { id: "discover", label: "Discover" },
  ];

  return (
    <div style={s.page}>
      {/* Glow effect */}
      <div style={s.glow} />

      <div style={s.container}>
        {/* Header */}
        <header style={s.header}>
          <div style={s.logoRow}>
            <div style={s.logoDot} />
            <span style={s.logoText}>Music to My Ears</span>
          </div>
          <p style={s.tagline}>Transform text and images into music with AI</p>
        </header>

        {/* Tabs */}
        <nav style={s.nav}>
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={tab === t.id ? { ...s.navBtn, ...s.navActive } : s.navBtn}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {/* === TEXT TAB === */}
        {tab === "text" && (
          <div style={s.card}>
            <textarea
              style={s.input}
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              onKeyDown={onKey}
              placeholder="Describe the music you want to create..."
              rows={4}
              disabled={loading}
            />
            <div style={s.toolbar}>
              <div style={s.durRow}>
                {DURATIONS.map(d => (
                  <button
                    key={d.tokens}
                    onClick={() => setDuration(d.tokens)}
                    style={duration === d.tokens ? { ...s.durBtn, ...s.durActive } : s.durBtn}
                  >
                    {d.label}
                  </button>
                ))}
              </div>
              <button
                onClick={generate}
                disabled={loading || !prompt.trim()}
                style={{ ...s.primaryBtn, opacity: loading || !prompt.trim() ? 0.4 : 1 }}
              >
                {loading ? "Generating..." : "Generate"}
              </button>
            </div>

            {/* Presets */}
            <div style={s.presetRow}>
              {PRESETS.map(p => (
                <button key={p} onClick={() => setPrompt(p)} style={s.preset}>
                  {p.length > 45 ? p.slice(0, 45) + "..." : p}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* === IMAGE TAB === */}
        {tab === "image" && (
          <div style={s.card}>
            {/* Image grid */}
            <div style={s.imageGrid}>
              {imagePreviews.map((src, i) => (
                <div key={i} style={s.imageThumb}>
                  <img src={src} alt="" style={s.thumbImg} />
                  <button onClick={() => removeImage(i)} style={s.removeBtn}>x</button>
                </div>
              ))}
              <div onClick={() => fileRef.current?.click()} style={s.addImageBtn}>
                <span style={{ fontSize: 28, color: "#444" }}>+</span>
                <span style={{ fontSize: 11, color: "#555" }}>Add images</span>
              </div>
            </div>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              multiple
              onChange={addImages}
              style={{ display: "none" }}
            />

            {/* Optional text context */}
            <textarea
              style={{ ...s.input, marginTop: 12 }}
              value={imageText}
              onChange={e => setImageText(e.target.value)}
              placeholder="Add text context (optional) — e.g. 'make it feel like a summer festival'"
              rows={2}
              disabled={loading}
            />

            <div style={s.toolbar}>
              <div style={s.durRow}>
                {DURATIONS.map(d => (
                  <button
                    key={d.tokens}
                    onClick={() => setDuration(d.tokens)}
                    style={duration === d.tokens ? { ...s.durBtn, ...s.durActive } : s.durBtn}
                  >
                    {d.label}
                  </button>
                ))}
              </div>
              <button
                onClick={generate}
                disabled={loading || images.length === 0}
                style={{ ...s.primaryBtn, opacity: loading || images.length === 0 ? 0.4 : 1 }}
              >
                {loading ? "Generating..." : `Generate from ${images.length} image${images.length !== 1 ? "s" : ""}`}
              </button>
            </div>
          </div>
        )}

        {/* === DISCOVER TAB === */}
        {tab === "discover" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {history.length === 0 && (
              <div style={s.emptyState}>No generations yet. Create some music first.</div>
            )}
            {history.map(item => (
              <div key={item.id} style={s.histCard}>
                <div style={s.histTop}>
                  <span style={item.source === "image" ? s.badgeImage : s.badgeText}>
                    {item.source === "image" ? "Image" : "Text"}
                  </span>
                  <span style={s.histTime}>
                    {new Date(item.timestamp * 1000).toLocaleString()}
                  </span>
                </div>
                <p style={s.histPrompt}>{item.original_prompt}</p>
                <p style={s.histEnhanced}>{item.enhanced_prompt}</p>
                <audio controls src={`${API}/audio/${item.filename}`} />
                <div style={s.rateRow}>
                  <button
                    onClick={() => rate(item.id, 1)}
                    style={item.rating === 1 ? { ...s.rateBtn, ...s.rateUp } : s.rateBtn}
                  >
                    &#9650;
                  </button>
                  <span style={s.rateScore}>{item.rating}</span>
                  <button
                    onClick={() => rate(item.id, -1)}
                    style={item.rating === -1 ? { ...s.rateBtn, ...s.rateDown } : s.rateBtn}
                  >
                    &#9660;
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* === LOADING === */}
        {loading && (
          <div style={s.loadingCard}>
            <div style={s.waveRow}>
              {[0, 1, 2, 3, 4].map(i => (
                <div key={i} style={{ ...s.waveBar, animationDelay: `${i * 0.15}s` }} />
              ))}
            </div>
            <span style={s.loadingText}>Generating your track...</span>
          </div>
        )}

        {/* === ERROR === */}
        {error && <div style={s.errorCard}>Error: {error}</div>}

        {/* === RESULT === */}
        {result && tab !== "discover" && (
          <div style={s.resultCard}>
            <p style={s.resultLabel}>Enhanced prompt</p>
            <p style={s.resultEnhanced}>{result.enhanced_prompt}</p>
            <audio controls autoPlay src={result.audioUrl} />
            <div style={s.rateRow}>
              <span style={{ fontSize: 13, color: "#555" }}>Rate:</span>
              <button onClick={() => rate(result.id, 1)} style={s.rateBtn}>&#9650; Good</button>
              <button onClick={() => rate(result.id, -1)} style={s.rateBtn}>&#9660; Bad</button>
            </div>
          </div>
        )}

        <footer style={s.footer}>
          Powered by Meta MusicGen &middot; OpenAI &middot; Built for Global Hackathon
        </footer>
      </div>
    </div>
  );
}

/* ——— Styles ——— */
const s = {
  page: {
    minHeight: "100vh",
    position: "relative",
    overflow: "hidden",
  },
  glow: {
    position: "fixed",
    top: "-200px",
    left: "50%",
    transform: "translateX(-50%)",
    width: 600,
    height: 600,
    borderRadius: "50%",
    background: "radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%)",
    pointerEvents: "none",
    zIndex: 0,
  },
  container: {
    position: "relative",
    zIndex: 1,
    maxWidth: 720,
    margin: "0 auto",
    padding: "48px 20px 40px",
    display: "flex",
    flexDirection: "column",
    gap: 20,
  },
  header: { textAlign: "center", marginBottom: 8 },
  logoRow: { display: "flex", alignItems: "center", justifyContent: "center", gap: 10 },
  logoDot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    background: "linear-gradient(135deg, #6366f1, #a855f7)",
  },
  logoText: { fontSize: 26, fontWeight: 700, letterSpacing: "-0.5px", color: "#fff" },
  tagline: { fontSize: 14, color: "#555", marginTop: 6 },

  nav: {
    display: "flex",
    gap: 2,
    background: "#0d0d0d",
    borderRadius: 12,
    padding: 3,
    border: "1px solid #1a1a1a",
  },
  navBtn: {
    flex: 1,
    padding: "10px 0",
    background: "transparent",
    border: "none",
    color: "#666",
    borderRadius: 10,
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 500,
    transition: "all 0.2s",
  },
  navActive: {
    background: "#1a1a1a",
    color: "#fff",
    boxShadow: "0 1px 4px rgba(0,0,0,0.3)",
  },

  card: {
    background: "#0d0d0d",
    borderRadius: 16,
    padding: 20,
    border: "1px solid #1a1a1a",
    animation: "slideUp 0.3s ease",
  },
  input: {
    width: "100%",
    background: "transparent",
    border: "none",
    color: "#e0e0e0",
    fontSize: 15,
    resize: "none",
    outline: "none",
    fontFamily: "inherit",
    lineHeight: 1.6,
  },
  toolbar: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 16,
    paddingTop: 16,
    borderTop: "1px solid #1a1a1a",
  },
  durRow: { display: "flex", gap: 6 },
  durBtn: {
    background: "#111",
    border: "1px solid #222",
    color: "#666",
    padding: "6px 16px",
    borderRadius: 8,
    cursor: "pointer",
    fontSize: 12,
    fontWeight: 500,
    transition: "all 0.15s",
  },
  durActive: {
    background: "#fff",
    color: "#000",
    borderColor: "#fff",
  },
  primaryBtn: {
    background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
    color: "#fff",
    border: "none",
    padding: "10px 28px",
    borderRadius: 10,
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    transition: "all 0.2s",
  },
  presetRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: 6,
    marginTop: 16,
    paddingTop: 16,
    borderTop: "1px solid #1a1a1a",
  },
  preset: {
    background: "#111",
    border: "1px solid #1a1a1a",
    color: "#555",
    padding: "5px 12px",
    borderRadius: 20,
    cursor: "pointer",
    fontSize: 11,
    transition: "all 0.15s",
  },

  /* Image grid */
  imageGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(100px, 1fr))",
    gap: 10,
    marginBottom: 4,
  },
  imageThumb: {
    position: "relative",
    borderRadius: 10,
    overflow: "hidden",
    aspectRatio: "1",
    border: "1px solid #222",
  },
  thumbImg: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  removeBtn: {
    position: "absolute",
    top: 4,
    right: 4,
    width: 22,
    height: 22,
    borderRadius: "50%",
    background: "rgba(0,0,0,0.7)",
    border: "1px solid #333",
    color: "#999",
    fontSize: 12,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  addImageBtn: {
    borderRadius: 10,
    border: "2px dashed #222",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    aspectRatio: "1",
    cursor: "pointer",
    transition: "all 0.15s",
    minHeight: 100,
  },

  /* Loading */
  loadingCard: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    padding: 20,
    background: "#0d0d0d",
    borderRadius: 16,
    border: "1px solid #1a1a1a",
  },
  waveRow: { display: "flex", alignItems: "center", gap: 3 },
  waveBar: {
    width: 3,
    height: 8,
    background: "#6366f1",
    borderRadius: 2,
    animation: "waveform 0.8s ease-in-out infinite",
  },
  loadingText: { fontSize: 13, color: "#555" },

  errorCard: {
    padding: 16,
    background: "#1a0a0a",
    borderRadius: 12,
    border: "1px solid #331111",
    color: "#ef4444",
    fontSize: 13,
  },

  /* Result */
  resultCard: {
    background: "#0d0d0d",
    borderRadius: 16,
    padding: 20,
    border: "1px solid #1a1a1a",
    animation: "slideUp 0.3s ease",
  },
  resultLabel: { fontSize: 11, color: "#444", textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 },
  resultEnhanced: { fontSize: 13, color: "#888", fontStyle: "italic", marginBottom: 16, lineHeight: 1.5 },

  /* Rating */
  rateRow: { display: "flex", alignItems: "center", gap: 8, marginTop: 12 },
  rateBtn: {
    background: "#111",
    border: "1px solid #222",
    color: "#666",
    padding: "4px 14px",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 13,
    transition: "all 0.15s",
  },
  rateUp: { background: "#0a1a0a", borderColor: "#1a3a1a", color: "#4ade80" },
  rateDown: { background: "#1a0a0a", borderColor: "#3a1a1a", color: "#f87171" },
  rateScore: { fontSize: 13, color: "#555", minWidth: 20, textAlign: "center" },

  /* Discover */
  histCard: {
    background: "#0d0d0d",
    borderRadius: 14,
    padding: 18,
    border: "1px solid #1a1a1a",
    animation: "slideUp 0.3s ease",
  },
  histTop: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  badgeText: {
    fontSize: 10,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    background: "#1a1a2a",
    color: "#6366f1",
    padding: "3px 8px",
    borderRadius: 4,
  },
  badgeImage: {
    fontSize: 10,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    background: "#1a2a1a",
    color: "#4ade80",
    padding: "3px 8px",
    borderRadius: 4,
  },
  histTime: { fontSize: 11, color: "#333" },
  histPrompt: { fontSize: 14, color: "#ccc", marginBottom: 4, lineHeight: 1.4 },
  histEnhanced: { fontSize: 12, color: "#555", fontStyle: "italic", marginBottom: 14, lineHeight: 1.4 },
  emptyState: {
    textAlign: "center",
    padding: 40,
    color: "#333",
    fontSize: 14,
  },

  footer: {
    textAlign: "center",
    fontSize: 11,
    color: "#333",
    marginTop: 20,
    paddingTop: 20,
    borderTop: "1px solid #111",
  },
};