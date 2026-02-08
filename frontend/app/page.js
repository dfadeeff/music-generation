"use client";

import { useState, useRef, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const DURATIONS = [
  { label: "5s", tokens: 256 },
  { label: "10s", tokens: 512 },
  { label: "20s", tokens: 1024 },
];

const PRESETS = [
  "Epic Super Bowl halftime, energetic drums, brass section, crowd energy",
  "Chill lo-fi beat, rainy mood, soft piano, vinyl crackle",
  "90s hip-hop, heavy bass, scratching, boom bap drums",
  "Orchestral cinematic, rising strings, dramatic tension, heroic brass",
  "Tropical house, steel drums, upbeat synth, summer vibes",
  "Dark ambient, deep drone, eerie atmosphere, suspenseful",
];

export default function Home() {
  const [tab, setTab] = useState("text");
  const [prompt, setPrompt] = useState("");
  const [duration, setDuration] = useState(256);
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [enhancedPrompt, setEnhancedPrompt] = useState(null);
  const [currentId, setCurrentId] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);

  // Load history on mount
  useEffect(() => {
    fetch(`${API_URL}/history`)
      .then((r) => r.json())
      .then(setHistory)
      .catch(() => {});
  }, []);

  async function handleGenerate() {
    if (tab === "text" && !prompt.trim()) return;
    if (tab === "image" && !imageFile) return;
    setLoading(true);
    setError(null);
    setAudioUrl(null);
    setEnhancedPrompt(null);
    setCurrentId(null);

    try {
      let res;
      if (tab === "image") {
        const formData = new FormData();
        formData.append("image", imageFile);
        formData.append("duration", duration);
        res = await fetch(`${API_URL}/generate-from-image`, {
          method: "POST",
          body: formData,
        });
      } else {
        res = await fetch(`${API_URL}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: prompt.trim(), duration }),
        });
      }

      if (!res.ok) throw new Error("Generation failed");

      const data = await res.json();
      const audioFullUrl = `${API_URL}${data.audio_url}`;
      setAudioUrl(audioFullUrl);
      setEnhancedPrompt(data.enhanced_prompt);
      setCurrentId(data.id);

      // Refresh history
      const histRes = await fetch(`${API_URL}/history`);
      setHistory(await histRes.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleRate(id, rating) {
    await fetch(`${API_URL}/rate/${id}?rating=${rating}`, { method: "POST" });
    const histRes = await fetch(`${API_URL}/history`);
    setHistory(await histRes.json());
  }

  function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && !loading) {
      e.preventDefault();
      handleGenerate();
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.main}>
        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.title}>Music to My Ears</h1>
          <p style={styles.subtitle}>
            AI-powered music generation — text or image to music
          </p>
        </div>

        {/* Tabs */}
        <div style={styles.tabs}>
          <button
            onClick={() => setTab("text")}
            style={{ ...styles.tab, ...(tab === "text" ? styles.tabActive : {}) }}
          >
            Text to Music
          </button>
          <button
            onClick={() => setTab("image")}
            style={{ ...styles.tab, ...(tab === "image" ? styles.tabActive : {}) }}
          >
            Image to Music
          </button>
          <button
            onClick={() => setTab("discover")}
            style={{ ...styles.tab, ...(tab === "discover" ? styles.tabActive : {}) }}
          >
            Discover
          </button>
        </div>

        {/* Text Input */}
        {tab === "text" && (
          <>
            <div style={styles.inputArea}>
              <textarea
                style={styles.textarea}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe the music you want to create..."
                rows={3}
                disabled={loading}
              />
              <div style={styles.controls}>
                <div style={styles.durationGroup}>
                  {DURATIONS.map((d) => (
                    <button
                      key={d.tokens}
                      onClick={() => setDuration(d.tokens)}
                      style={{
                        ...styles.durationBtn,
                        ...(duration === d.tokens ? styles.durationActive : {}),
                      }}
                    >
                      {d.label}
                    </button>
                  ))}
                </div>
                <button
                  onClick={handleGenerate}
                  disabled={loading || !prompt.trim()}
                  style={{
                    ...styles.generateBtn,
                    opacity: loading || !prompt.trim() ? 0.5 : 1,
                  }}
                >
                  {loading ? "Generating..." : "Generate"}
                </button>
              </div>
            </div>
            <div style={styles.presets}>
              {PRESETS.map((p) => (
                <button key={p} onClick={() => setPrompt(p)} style={styles.presetBtn}>
                  {p.length > 50 ? p.slice(0, 50) + "..." : p}
                </button>
              ))}
            </div>
          </>
        )}

        {/* Image Input */}
        {tab === "image" && (
          <div style={styles.inputArea}>
            <div
              onClick={() => fileInputRef.current?.click()}
              style={styles.dropZone}
            >
              {imagePreview ? (
                <img
                  src={imagePreview}
                  alt="Preview"
                  style={{ maxHeight: 200, borderRadius: 8 }}
                />
              ) : (
                <p style={{ color: "#666", margin: 0 }}>
                  Click to upload an image — AI will compose music from it
                </p>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              style={{ display: "none" }}
            />
            <div style={styles.controls}>
              <div style={styles.durationGroup}>
                {DURATIONS.map((d) => (
                  <button
                    key={d.tokens}
                    onClick={() => setDuration(d.tokens)}
                    style={{
                      ...styles.durationBtn,
                      ...(duration === d.tokens ? styles.durationActive : {}),
                    }}
                  >
                    {d.label}
                  </button>
                ))}
              </div>
              <button
                onClick={handleGenerate}
                disabled={loading || !imageFile}
                style={{
                  ...styles.generateBtn,
                  opacity: loading || !imageFile ? 0.5 : 1,
                }}
              >
                {loading ? "Generating..." : "Generate from Image"}
              </button>
            </div>
          </div>
        )}

        {/* Discover Tab */}
        {tab === "discover" && (
          <div style={styles.discoverSection}>
            <h3 style={{ margin: "0 0 16px" }}>All Generations</h3>
            {history.length === 0 && (
              <p style={{ color: "#666" }}>No generations yet. Create some music first!</p>
            )}
            {history.map((item) => (
              <div key={item.id} style={styles.discoverCard}>
                <div style={styles.discoverHeader}>
                  <span style={styles.discoverSource}>
                    {item.source === "image" ? "Image" : "Text"}
                  </span>
                  <span style={styles.discoverTime}>
                    {new Date(item.timestamp * 1000).toLocaleString()}
                  </span>
                </div>
                <p style={styles.discoverPrompt}>{item.original_prompt}</p>
                <p style={styles.discoverEnhanced}>{item.enhanced_prompt}</p>
                <audio
                  controls
                  src={`${API_URL}/audio/${item.filename}`}
                  style={{ width: "100%" }}
                />
                <div style={styles.ratingRow}>
                  <button
                    onClick={() => handleRate(item.id, 1)}
                    style={{
                      ...styles.rateBtn,
                      ...(item.rating === 1 ? styles.rateActive : {}),
                    }}
                  >
                    &#9650; {item.rating > 0 ? item.rating : ""}
                  </button>
                  <button
                    onClick={() => handleRate(item.id, -1)}
                    style={{
                      ...styles.rateBtn,
                      ...(item.rating === -1 ? styles.rateDown : {}),
                    }}
                  >
                    &#9660;
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={styles.loadingBox}>
            <div style={styles.spinner} />
            <p style={{ margin: 0, color: "#888" }}>
              Generating music... this takes 15-60 seconds
            </p>
          </div>
        )}

        {/* Error */}
        {error && <div style={styles.errorBox}>Error: {error}</div>}

        {/* Result */}
        {audioUrl && tab !== "discover" && (
          <div style={styles.playerBox}>
            {enhancedPrompt && (
              <p style={styles.enhancedLabel}>
                Enhanced prompt: <span style={styles.enhancedText}>{enhancedPrompt}</span>
              </p>
            )}
            <audio
              ref={audioRef}
              controls
              autoPlay
              src={audioUrl}
              style={{ width: "100%" }}
            />
            {currentId && (
              <div style={styles.ratingRow}>
                <span style={{ fontSize: 13, color: "#666" }}>Rate this:</span>
                <button
                  onClick={() => handleRate(currentId, 1)}
                  style={styles.rateBtn}
                >
                  &#9650; Good
                </button>
                <button
                  onClick={() => handleRate(currentId, -1)}
                  style={styles.rateBtn}
                >
                  &#9660; Bad
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    background: "#0a0a0a",
    color: "#fff",
    display: "flex",
    justifyContent: "center",
    padding: "40px 20px",
  },
  main: {
    width: "100%",
    maxWidth: 700,
    display: "flex",
    flexDirection: "column",
    gap: 24,
  },
  header: { textAlign: "center" },
  title: { fontSize: 32, margin: 0, fontWeight: 700 },
  subtitle: { fontSize: 14, color: "#888", margin: "8px 0 0" },
  tabs: { display: "flex", gap: 4, background: "#111", borderRadius: 10, padding: 4 },
  tab: {
    flex: 1,
    padding: "10px 0",
    background: "transparent",
    border: "none",
    color: "#888",
    borderRadius: 8,
    cursor: "pointer",
    fontSize: 14,
  },
  tabActive: { background: "#2a2a2a", color: "#fff" },
  inputArea: {
    background: "#1a1a1a",
    borderRadius: 12,
    padding: 16,
    border: "1px solid #333",
  },
  textarea: {
    width: "100%",
    background: "transparent",
    border: "none",
    color: "#fff",
    fontSize: 16,
    resize: "none",
    outline: "none",
    fontFamily: "inherit",
    boxSizing: "border-box",
  },
  controls: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 12,
  },
  durationGroup: { display: "flex", gap: 8 },
  durationBtn: {
    background: "#2a2a2a",
    border: "1px solid #444",
    color: "#aaa",
    padding: "6px 14px",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 13,
  },
  durationActive: { background: "#fff", color: "#000", borderColor: "#fff" },
  generateBtn: {
    background: "#fff",
    color: "#000",
    border: "none",
    padding: "10px 28px",
    borderRadius: 8,
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
  },
  presets: { display: "flex", flexWrap: "wrap", gap: 8 },
  presetBtn: {
    background: "#1a1a1a",
    border: "1px solid #333",
    color: "#aaa",
    padding: "6px 12px",
    borderRadius: 20,
    cursor: "pointer",
    fontSize: 12,
  },
  dropZone: {
    border: "2px dashed #333",
    borderRadius: 12,
    padding: 40,
    textAlign: "center",
    cursor: "pointer",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    minHeight: 120,
  },
  loadingBox: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: 20,
    background: "#1a1a1a",
    borderRadius: 12,
    border: "1px solid #333",
  },
  spinner: {
    width: 20,
    height: 20,
    border: "2px solid #333",
    borderTop: "2px solid #fff",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
  },
  errorBox: {
    padding: 16,
    background: "#2a1010",
    borderRadius: 12,
    border: "1px solid #500",
    color: "#f88",
  },
  playerBox: {
    padding: 20,
    background: "#1a1a1a",
    borderRadius: 12,
    border: "1px solid #333",
  },
  enhancedLabel: { fontSize: 12, color: "#666", margin: "0 0 12px" },
  enhancedText: { color: "#aaa" },
  ratingRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginTop: 12,
  },
  rateBtn: {
    background: "#2a2a2a",
    border: "1px solid #444",
    color: "#aaa",
    padding: "4px 12px",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 13,
  },
  rateActive: { background: "#1a3a1a", borderColor: "#4a4", color: "#4a4" },
  rateDown: { background: "#3a1a1a", borderColor: "#a44", color: "#a44" },
  discoverSection: {
    background: "#111",
    borderRadius: 12,
    padding: 20,
    border: "1px solid #222",
  },
  discoverCard: {
    background: "#1a1a1a",
    borderRadius: 10,
    padding: 16,
    marginBottom: 12,
    border: "1px solid #333",
  },
  discoverHeader: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  discoverSource: {
    fontSize: 11,
    background: "#2a2a2a",
    padding: "2px 8px",
    borderRadius: 4,
    color: "#aaa",
  },
  discoverTime: { fontSize: 11, color: "#555" },
  discoverPrompt: { fontSize: 14, margin: "8px 0 4px", color: "#ddd" },
  discoverEnhanced: { fontSize: 12, margin: "0 0 12px", color: "#777", fontStyle: "italic" },
};