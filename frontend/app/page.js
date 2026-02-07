"use client";

import { useState, useRef } from "react";

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
  const [prompt, setPrompt] = useState("");
  const [duration, setDuration] = useState(256);
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const audioRef = useRef(null);

  async function handleGenerate() {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setAudioUrl(null);

    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.trim(), duration }),
      });

      if (!res.ok) throw new Error("Generation failed");

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      setHistory((prev) => [{ prompt: prompt.trim(), url, time: new Date().toLocaleTimeString() }, ...prev]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
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
          <p style={styles.subtitle}>Type a prompt. Get music. Powered by MusicGen.</p>
        </div>

        {/* Prompt Input */}
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

        {/* Presets */}
        <div style={styles.presets}>
          {PRESETS.map((p) => (
            <button key={p} onClick={() => setPrompt(p)} style={styles.presetBtn}>
              {p.length > 50 ? p.slice(0, 50) + "..." : p}
            </button>
          ))}
        </div>

        {/* Loading */}
        {loading && (
          <div style={styles.loadingBox}>
            <div style={styles.spinner} />
            <p style={{ margin: 0, color: "#888" }}>Generating music... this takes 15-60 seconds</p>
          </div>
        )}

        {/* Error */}
        {error && <div style={styles.errorBox}>Error: {error}</div>}

        {/* Audio Player */}
        {audioUrl && (
          <div style={styles.playerBox}>
            <p style={{ margin: "0 0 12px", fontWeight: 600 }}>Generated Audio</p>
            <audio ref={audioRef} controls autoPlay src={audioUrl} style={{ width: "100%" }} />
          </div>
        )}

        {/* History */}
        {history.length > 1 && (
          <div style={styles.history}>
            <h3 style={{ margin: "0 0 12px" }}>History</h3>
            {history.slice(1).map((item, i) => (
              <div key={i} style={styles.historyItem}>
                <span style={styles.historyTime}>{item.time}</span>
                <span style={styles.historyPrompt}>{item.prompt}</span>
                <audio controls src={item.url} style={{ height: 32 }} />
              </div>
            ))}
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
  durationActive: {
    background: "#fff",
    color: "#000",
    borderColor: "#fff",
  },
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
  presets: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
  },
  presetBtn: {
    background: "#1a1a1a",
    border: "1px solid #333",
    color: "#aaa",
    padding: "6px 12px",
    borderRadius: 20,
    cursor: "pointer",
    fontSize: 12,
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
  history: {
    padding: 20,
    background: "#111",
    borderRadius: 12,
    border: "1px solid #222",
  },
  historyItem: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "8px 0",
    borderBottom: "1px solid #222",
  },
  historyTime: { fontSize: 12, color: "#555", minWidth: 70 },
  historyPrompt: { fontSize: 13, color: "#aaa", flex: 1 },
};