"use client";

import { useState, useRef, useEffect, useCallback } from "react";

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

const SLIDER_LABELS = {
  energy: { name: "Energy", left: "Whisper", right: "Explosive" },
  style: { name: "Style", left: "Lo-fi", right: "Cinematic" },
  warmth: { name: "Warmth", left: "Warm Analog", right: "Bright Digital" },
  arc: { name: "Arc", left: "Steady", right: "Dramatic Build" },
};

export default function Home() {
  const [tab, setTab] = useState("create");
  const [prompt, setPrompt] = useState("");
  const [duration, setDuration] = useState(256);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  // Images
  const [images, setImages] = useState([]);
  const [imagePreviews, setImagePreviews] = useState([]);
  const fileRef = useRef(null);

  // Voice
  const [isRecording, setIsRecording] = useState(false);
  const [voiceBlob, setVoiceBlob] = useState(null);
  const [voiceName, setVoiceName] = useState("");
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const voiceFileRef = useRef(null);

  // Sliders — null means "auto" (AI-detected)
  const [sliders, setSliders] = useState({ energy: null, style: null, warmth: null, arc: null });
  const [slidersEnabled, setSlidersEnabled] = useState({ energy: false, style: false, warmth: false, arc: false });

  useEffect(() => {
    fetch(`${API}/history`).then(r => r.json()).then(setHistory).catch(() => {});
  }, []);

  const refreshHistory = async () => {
    const r = await fetch(`${API}/history`);
    setHistory(await r.json());
  };

  // --- Voice Recording ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        setVoiceBlob(blob);
        setVoiceName("Recording");
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorderRef.current = mr;
      mr.start();
      setIsRecording(true);
    } catch (e) {
      setError("Microphone access denied");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const uploadVoice = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVoiceBlob(file);
      setVoiceName(file.name);
    }
  };

  const removeVoice = () => {
    setVoiceBlob(null);
    setVoiceName("");
  };

  // --- Image handling ---
  const addImages = (e) => {
    const files = Array.from(e.target.files);
    setImages(prev => [...prev, ...files]);
    setImagePreviews(prev => [...prev, ...files.map(f => URL.createObjectURL(f))]);
  };

  const removeImage = (i) => {
    setImages(prev => prev.filter((_, idx) => idx !== i));
    setImagePreviews(prev => prev.filter((_, idx) => idx !== i));
  };

  // --- Slider handling ---
  const toggleSlider = (key) => {
    setSlidersEnabled(prev => {
      const next = { ...prev, [key]: !prev[key] };
      if (!next[key]) setSliders(s => ({ ...s, [key]: null }));
      else setSliders(s => ({ ...s, [key]: s[key] ?? 50 }));
      return next;
    });
  };

  const updateSlider = (key, val) => {
    setSliders(prev => ({ ...prev, [key]: parseInt(val) }));
  };

  // --- Generate ---
  const generate = async () => {
    const hasText = prompt.trim().length > 0;
    const hasImages = images.length > 0;
    const hasVoice = voiceBlob !== null;
    if (!hasText && !hasImages && !hasVoice) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let res;
      const isMultimodal = hasImages || hasVoice;

      if (isMultimodal) {
        const fd = new FormData();
        if (hasText) fd.append("text", prompt.trim());
        images.forEach(f => fd.append("images", f));
        if (hasVoice) fd.append("voice", voiceBlob, "voice.wav");
        fd.append("duration", duration);
        Object.entries(sliders).forEach(([k, v]) => {
          if (v !== null) fd.append(k, v);
        });
        res = await fetch(`${API}/generate-multimodal`, { method: "POST", body: fd });
      } else {
        res = await fetch(`${API}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: prompt.trim(),
            duration,
            ...Object.fromEntries(Object.entries(sliders).filter(([, v]) => v !== null)),
          }),
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
    if (result && result.id === id) {
      setResult(prev => ({ ...prev, rated: rating }));
    }
    await refreshHistory();
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !loading) { e.preventDefault(); generate(); }
  };

  const canGenerate = prompt.trim() || images.length > 0 || voiceBlob;

  const tabs = [
    { id: "create", label: "Create" },
    { id: "discover", label: "Discover" },
  ];

  return (
    <div style={s.page}>
      <div style={s.glow} />

      <div style={s.container}>
        {/* Header */}
        <header style={s.header}>
          <div style={s.logoRow}>
            <div style={s.logoDot} />
            <span style={s.logoText}>Music to My Ears</span>
          </div>
          <p style={s.tagline}>Transform text, images & voice into music with AI</p>
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

        {/* === CREATE TAB === */}
        {tab === "create" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Text Input */}
            <div style={s.card}>
              <div style={s.sectionLabel}>Text</div>
              <textarea
                style={s.input}
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={onKey}
                placeholder="Describe the music you want to create..."
                rows={3}
                disabled={loading}
              />
              <div style={s.presetRow}>
                {PRESETS.map(p => (
                  <button key={p} onClick={() => setPrompt(p)} style={s.preset}>
                    {p.length > 40 ? p.slice(0, 40) + "..." : p}
                  </button>
                ))}
              </div>
            </div>

            {/* Image Input */}
            <div style={s.card}>
              <div style={s.sectionLabel}>Images <span style={s.optional}>(optional)</span></div>
              <div style={s.imageGrid}>
                {imagePreviews.map((src, i) => (
                  <div key={i} style={s.imageThumb}>
                    <img src={src} alt="" style={s.thumbImg} />
                    <button onClick={() => removeImage(i)} style={s.removeBtn}>x</button>
                  </div>
                ))}
                <div onClick={() => fileRef.current?.click()} style={s.addImageBtn}>
                  <span style={{ fontSize: 24, color: "#444" }}>+</span>
                  <span style={{ fontSize: 10, color: "#555" }}>Add</span>
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
            </div>

            {/* Voice Input */}
            <div style={s.card}>
              <div style={s.sectionLabel}>Voice <span style={s.optional}>(optional)</span></div>
              {voiceBlob ? (
                <div style={s.voiceReady}>
                  <div style={s.voiceIcon}>&#127908;</div>
                  <span style={s.voiceText}>{voiceName}</span>
                  <button onClick={removeVoice} style={s.voiceRemove}>Remove</button>
                </div>
              ) : (
                <div style={s.voiceRow}>
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    style={isRecording ? { ...s.voiceBtn, ...s.voiceBtnRecording } : s.voiceBtn}
                  >
                    {isRecording ? "Stop Recording" : "Record"}
                  </button>
                  <span style={{ color: "#333", fontSize: 12 }}>or</span>
                  <button onClick={() => voiceFileRef.current?.click()} style={s.voiceBtn}>
                    Upload Audio
                  </button>
                  <input
                    ref={voiceFileRef}
                    type="file"
                    accept="audio/*"
                    onChange={uploadVoice}
                    style={{ display: "none" }}
                  />
                </div>
              )}
            </div>

            {/* Emotion Sliders */}
            <div style={s.card}>
              <div style={s.sectionLabel}>Emotion Sliders <span style={s.optional}>(override AI-detected)</span></div>
              <div style={s.sliderGrid}>
                {Object.entries(SLIDER_LABELS).map(([key, label]) => (
                  <div key={key} style={s.sliderItem}>
                    <div style={s.sliderHeader}>
                      <label style={s.sliderLabel}>{label.name}</label>
                      <button
                        onClick={() => toggleSlider(key)}
                        style={slidersEnabled[key] ? { ...s.toggleBtn, ...s.toggleActive } : s.toggleBtn}
                      >
                        {slidersEnabled[key] ? "Manual" : "Auto"}
                      </button>
                    </div>
                    <div style={{ opacity: slidersEnabled[key] ? 1 : 0.3, transition: "opacity 0.2s" }}>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={sliders[key] ?? 50}
                        onChange={e => updateSlider(key, e.target.value)}
                        disabled={!slidersEnabled[key]}
                        style={s.slider}
                      />
                      <div style={s.sliderRange}>
                        <span>{label.left}</span>
                        <span style={s.sliderVal}>{sliders[key] ?? "—"}</span>
                        <span>{label.right}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Duration & Generate */}
            <div style={s.card}>
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
                  disabled={loading || !canGenerate}
                  style={{ ...s.primaryBtn, opacity: loading || !canGenerate ? 0.4 : 1 }}
                >
                  {loading ? "Generating..." : "Generate Music"}
                </button>
              </div>
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
                  <span style={
                    item.source === "multimodal" ? s.badgeMulti :
                    item.source === "image" ? s.badgeImage : s.badgeText
                  }>
                    {item.source === "multimodal" ? "Multi" : item.source === "image" ? "Image" : "Text"}
                  </span>
                  <span style={s.histTime}>
                    {new Date(item.timestamp * 1000).toLocaleString()}
                  </span>
                </div>
                <p style={s.histPrompt}>{item.original_prompt}</p>
                <p style={s.histEnhanced}>{item.enhanced_prompt}</p>

                {/* Emotion profile mini bars */}
                {item.emotion_profile && (
                  <div style={s.miniProfileRow}>
                    {["energy", "style", "warmth", "arc"].map(k => (
                      <div key={k} style={s.miniBar}>
                        <div style={s.miniBarLabel}>{k}</div>
                        <div style={s.miniBarTrack}>
                          <div style={{ ...s.miniBarFill, width: `${item.emotion_profile[k] || 0}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}

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
            <span style={s.loadingText}>Analyzing emotions & generating your track...</span>
          </div>
        )}

        {/* === ERROR === */}
        {error && <div style={s.errorCard}>Error: {error}</div>}

        {/* === RESULT === */}
        {result && tab === "create" && (
          <div style={s.resultCard}>
            {/* Emotion Profile */}
            {result.emotion_profile && (
              <div style={s.profileSection}>
                <div style={s.profileHeader}>
                  <span style={s.profileEmotion}>{result.emotion_profile.emotion}</span>
                  {result.emotion_profile.description && (
                    <span style={s.profileDesc}>{result.emotion_profile.description}</span>
                  )}
                </div>
                <div style={s.profileBars}>
                  {["energy", "style", "warmth", "arc"].map(k => (
                    <div key={k} style={s.profileBar}>
                      <div style={s.profileBarHeader}>
                        <span style={s.profileBarLabel}>{SLIDER_LABELS[k].name}</span>
                        <span style={s.profileBarVal}>{result.emotion_profile[k]}</span>
                      </div>
                      <div style={s.profileBarTrack}>
                        <div style={{
                          ...s.profileBarFill,
                          width: `${result.emotion_profile[k]}%`,
                        }} />
                      </div>
                      <div style={s.profileBarRange}>
                        <span>{SLIDER_LABELS[k].left}</span>
                        <span>{SLIDER_LABELS[k].right}</span>
                      </div>
                    </div>
                  ))}
                </div>
                {result.emotion_profile.overrides?.length > 0 && (
                  <div style={s.overrideNote}>
                    You overrode: {result.emotion_profile.overrides.join(", ")}
                  </div>
                )}
              </div>
            )}

            {/* Enhanced Prompt */}
            <div style={s.resultPromptSection}>
              <p style={s.resultLabel}>Enhanced prompt</p>
              <p style={s.resultEnhanced}>{result.enhanced_prompt}</p>
            </div>

            {/* Audio */}
            <audio controls autoPlay src={result.audioUrl} />

            {/* Explainability */}
            {result.explanation && (
              <div style={s.explainSection}>
                <p style={s.explainLabel}>How your inputs became music</p>
                <p style={s.explainNarrative}>{result.explanation.narrative}</p>
                {result.explanation.timeline && (
                  <div style={s.timeline}>
                    {result.explanation.timeline.map((step, i) => (
                      <div key={i} style={s.timelineStep}>
                        <div style={s.timelineDot}>
                          <div style={s.timelineDotInner} />
                          {i < result.explanation.timeline.length - 1 && <div style={s.timelineLine} />}
                        </div>
                        <div style={s.timelineContent}>
                          <span style={s.timelineStepName}>{step.step}</span>
                          <span style={s.timelineStepDetail}>{step.detail}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Rating */}
            <div style={s.rateRow}>
              <span style={{ fontSize: 13, color: "#555" }}>Rate this generation:</span>
              <button
                onClick={() => rate(result.id, 1)}
                style={result.rated === 1 ? { ...s.rateBtn, ...s.rateUp } : s.rateBtn}
              >
                &#9650; Good
              </button>
              <button
                onClick={() => rate(result.id, -1)}
                style={result.rated === -1 ? { ...s.rateBtn, ...s.rateDown } : s.rateBtn}
              >
                &#9660; Bad
              </button>
            </div>
          </div>
        )}

        <footer style={s.footer}>
          Powered by Meta MusicGen &middot; OpenAI &middot; Whisper &middot; Built for Global Hackathon
        </footer>
      </div>
    </div>
  );
}

/* ——— Styles ——— */
const s = {
  page: { minHeight: "100vh", position: "relative", overflow: "hidden" },
  glow: {
    position: "fixed", top: "-200px", left: "50%", transform: "translateX(-50%)",
    width: 600, height: 600, borderRadius: "50%",
    background: "radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%)",
    pointerEvents: "none", zIndex: 0,
  },
  container: {
    position: "relative", zIndex: 1, maxWidth: 720, margin: "0 auto",
    padding: "48px 20px 40px", display: "flex", flexDirection: "column", gap: 16,
  },
  header: { textAlign: "center", marginBottom: 8 },
  logoRow: { display: "flex", alignItems: "center", justifyContent: "center", gap: 10 },
  logoDot: {
    width: 10, height: 10, borderRadius: "50%",
    background: "linear-gradient(135deg, #6366f1, #a855f7)",
  },
  logoText: { fontSize: 26, fontWeight: 700, letterSpacing: "-0.5px", color: "#fff" },
  tagline: { fontSize: 14, color: "#555", marginTop: 6 },

  nav: {
    display: "flex", gap: 2, background: "#0d0d0d", borderRadius: 12,
    padding: 3, border: "1px solid #1a1a1a",
  },
  navBtn: {
    flex: 1, padding: "10px 0", background: "transparent", border: "none",
    color: "#666", borderRadius: 10, cursor: "pointer", fontSize: 13, fontWeight: 500,
    transition: "all 0.2s",
  },
  navActive: { background: "#1a1a1a", color: "#fff", boxShadow: "0 1px 4px rgba(0,0,0,0.3)" },

  card: {
    background: "#0d0d0d", borderRadius: 16, padding: 20,
    border: "1px solid #1a1a1a", animation: "slideUp 0.3s ease",
  },
  sectionLabel: {
    fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1,
    color: "#444", marginBottom: 12,
  },
  optional: { fontWeight: 400, color: "#333", textTransform: "none", letterSpacing: 0 },

  input: {
    width: "100%", background: "transparent", border: "none", color: "#e0e0e0",
    fontSize: 15, resize: "none", outline: "none", fontFamily: "inherit", lineHeight: 1.6,
  },
  toolbar: { display: "flex", justifyContent: "space-between", alignItems: "center" },
  durRow: { display: "flex", gap: 6 },
  durBtn: {
    background: "#111", border: "1px solid #222", color: "#666",
    padding: "6px 16px", borderRadius: 8, cursor: "pointer", fontSize: 12,
    fontWeight: 500, transition: "all 0.15s",
  },
  durActive: { background: "#fff", color: "#000", borderColor: "#fff" },
  primaryBtn: {
    background: "linear-gradient(135deg, #6366f1, #8b5cf6)", color: "#fff",
    border: "none", padding: "10px 28px", borderRadius: 10, fontSize: 13,
    fontWeight: 600, cursor: "pointer", transition: "all 0.2s",
  },
  presetRow: {
    display: "flex", flexWrap: "wrap", gap: 6, marginTop: 14, paddingTop: 14,
    borderTop: "1px solid #1a1a1a",
  },
  preset: {
    background: "#111", border: "1px solid #1a1a1a", color: "#555",
    padding: "5px 12px", borderRadius: 20, cursor: "pointer", fontSize: 11,
    transition: "all 0.15s",
  },

  /* Image grid */
  imageGrid: {
    display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))",
    gap: 8,
  },
  imageThumb: {
    position: "relative", borderRadius: 10, overflow: "hidden",
    aspectRatio: "1", border: "1px solid #222",
  },
  thumbImg: { width: "100%", height: "100%", objectFit: "cover" },
  removeBtn: {
    position: "absolute", top: 4, right: 4, width: 20, height: 20,
    borderRadius: "50%", background: "rgba(0,0,0,0.7)", border: "1px solid #333",
    color: "#999", fontSize: 11, cursor: "pointer", display: "flex",
    alignItems: "center", justifyContent: "center",
  },
  addImageBtn: {
    borderRadius: 10, border: "2px dashed #222", display: "flex",
    flexDirection: "column", alignItems: "center", justifyContent: "center",
    aspectRatio: "1", cursor: "pointer", transition: "all 0.15s", minHeight: 80,
  },

  /* Voice */
  voiceRow: { display: "flex", alignItems: "center", gap: 12 },
  voiceBtn: {
    background: "#111", border: "1px solid #222", color: "#888",
    padding: "8px 20px", borderRadius: 8, cursor: "pointer", fontSize: 12,
    fontWeight: 500, transition: "all 0.15s",
  },
  voiceBtnRecording: {
    background: "#2a0a0a", borderColor: "#f87171", color: "#f87171",
    animation: "pulse 1s infinite",
  },
  voiceReady: {
    display: "flex", alignItems: "center", gap: 10, padding: "8px 14px",
    background: "#111", borderRadius: 8, border: "1px solid #222",
  },
  voiceIcon: { fontSize: 18 },
  voiceText: { flex: 1, fontSize: 13, color: "#888" },
  voiceRemove: {
    background: "transparent", border: "1px solid #333", color: "#666",
    padding: "3px 10px", borderRadius: 6, cursor: "pointer", fontSize: 11,
  },

  /* Sliders */
  sliderGrid: { display: "flex", flexDirection: "column", gap: 16 },
  sliderItem: {},
  sliderHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 },
  sliderLabel: { fontSize: 13, fontWeight: 500, color: "#aaa" },
  toggleBtn: {
    background: "#111", border: "1px solid #222", color: "#555",
    padding: "3px 10px", borderRadius: 6, cursor: "pointer", fontSize: 10,
    fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5,
  },
  toggleActive: { background: "#1a1a2a", borderColor: "#6366f1", color: "#6366f1" },
  slider: {
    width: "100%", height: 4, appearance: "none", WebkitAppearance: "none",
    background: "#222", borderRadius: 2, outline: "none", cursor: "pointer",
    accentColor: "#6366f1",
  },
  sliderRange: {
    display: "flex", justifyContent: "space-between", fontSize: 10, color: "#444", marginTop: 4,
  },
  sliderVal: { fontWeight: 600, color: "#6366f1" },

  /* Loading */
  loadingCard: {
    display: "flex", alignItems: "center", gap: 16, padding: 20,
    background: "#0d0d0d", borderRadius: 16, border: "1px solid #1a1a1a",
  },
  waveRow: { display: "flex", alignItems: "center", gap: 3 },
  waveBar: {
    width: 3, height: 8, background: "#6366f1", borderRadius: 2,
    animation: "waveform 0.8s ease-in-out infinite",
  },
  loadingText: { fontSize: 13, color: "#555" },

  errorCard: {
    padding: 16, background: "#1a0a0a", borderRadius: 12,
    border: "1px solid #331111", color: "#ef4444", fontSize: 13,
  },

  /* Result */
  resultCard: {
    background: "#0d0d0d", borderRadius: 16, padding: 24,
    border: "1px solid #1a1a1a", animation: "slideUp 0.3s ease",
    display: "flex", flexDirection: "column", gap: 20,
  },

  /* Emotion Profile in Result */
  profileSection: {},
  profileHeader: { display: "flex", alignItems: "baseline", gap: 10, marginBottom: 14 },
  profileEmotion: {
    fontSize: 18, fontWeight: 700, color: "#fff", textTransform: "capitalize",
  },
  profileDesc: { fontSize: 13, color: "#666", fontStyle: "italic" },
  profileBars: { display: "flex", flexDirection: "column", gap: 10 },
  profileBar: {},
  profileBarHeader: { display: "flex", justifyContent: "space-between", marginBottom: 4 },
  profileBarLabel: { fontSize: 11, fontWeight: 500, color: "#666", textTransform: "uppercase", letterSpacing: 0.5 },
  profileBarVal: { fontSize: 11, fontWeight: 600, color: "#6366f1" },
  profileBarTrack: {
    width: "100%", height: 4, background: "#1a1a1a", borderRadius: 2, overflow: "hidden",
  },
  profileBarFill: {
    height: "100%", borderRadius: 2,
    background: "linear-gradient(90deg, #6366f1, #a855f7)",
    transition: "width 0.6s ease",
  },
  profileBarRange: {
    display: "flex", justifyContent: "space-between", fontSize: 9, color: "#333", marginTop: 2,
  },
  overrideNote: {
    marginTop: 8, fontSize: 11, color: "#6366f1", fontStyle: "italic",
  },

  /* Enhanced prompt */
  resultPromptSection: { borderTop: "1px solid #1a1a1a", paddingTop: 16 },
  resultLabel: { fontSize: 11, color: "#444", textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 },
  resultEnhanced: { fontSize: 13, color: "#888", fontStyle: "italic", lineHeight: 1.5 },

  /* Explainability */
  explainSection: { borderTop: "1px solid #1a1a1a", paddingTop: 16 },
  explainLabel: {
    fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1,
    color: "#444", marginBottom: 8,
  },
  explainNarrative: { fontSize: 13, color: "#999", lineHeight: 1.6, marginBottom: 16 },
  timeline: { display: "flex", flexDirection: "column" },
  timelineStep: { display: "flex", gap: 12, minHeight: 48 },
  timelineDot: {
    display: "flex", flexDirection: "column", alignItems: "center", width: 16, flexShrink: 0,
  },
  timelineDotInner: {
    width: 8, height: 8, borderRadius: "50%",
    background: "linear-gradient(135deg, #6366f1, #a855f7)", flexShrink: 0,
  },
  timelineLine: {
    width: 1, flex: 1, background: "#222", marginTop: 4,
  },
  timelineContent: {
    display: "flex", flexDirection: "column", gap: 2, paddingBottom: 12,
  },
  timelineStepName: { fontSize: 12, fontWeight: 600, color: "#ccc" },
  timelineStepDetail: { fontSize: 11, color: "#555", lineHeight: 1.4 },

  /* Rating */
  rateRow: { display: "flex", alignItems: "center", gap: 8, marginTop: 4 },
  rateBtn: {
    background: "#111", border: "1px solid #222", color: "#666",
    padding: "4px 14px", borderRadius: 6, cursor: "pointer", fontSize: 13,
    transition: "all 0.15s",
  },
  rateUp: { background: "#0a1a0a", borderColor: "#1a3a1a", color: "#4ade80" },
  rateDown: { background: "#1a0a0a", borderColor: "#3a1a1a", color: "#f87171" },
  rateScore: { fontSize: 13, color: "#555", minWidth: 20, textAlign: "center" },

  /* Discover */
  histCard: {
    background: "#0d0d0d", borderRadius: 14, padding: 18,
    border: "1px solid #1a1a1a", animation: "slideUp 0.3s ease",
  },
  histTop: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  badgeText: {
    fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5,
    background: "#1a1a2a", color: "#6366f1", padding: "3px 8px", borderRadius: 4,
  },
  badgeImage: {
    fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5,
    background: "#1a2a1a", color: "#4ade80", padding: "3px 8px", borderRadius: 4,
  },
  badgeMulti: {
    fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5,
    background: "#2a1a2a", color: "#a855f7", padding: "3px 8px", borderRadius: 4,
  },
  histTime: { fontSize: 11, color: "#333" },
  histPrompt: { fontSize: 14, color: "#ccc", marginBottom: 4, lineHeight: 1.4 },
  histEnhanced: { fontSize: 12, color: "#555", fontStyle: "italic", marginBottom: 14, lineHeight: 1.4 },

  /* Mini profile bars in discover */
  miniProfileRow: { display: "flex", gap: 8, marginBottom: 12 },
  miniBar: { flex: 1 },
  miniBarLabel: { fontSize: 9, color: "#444", textTransform: "uppercase", marginBottom: 3 },
  miniBarTrack: { height: 3, background: "#1a1a1a", borderRadius: 2, overflow: "hidden" },
  miniBarFill: {
    height: "100%", borderRadius: 2,
    background: "linear-gradient(90deg, #6366f1, #a855f7)",
  },

  emptyState: { textAlign: "center", padding: 40, color: "#333", fontSize: 14 },
  footer: {
    textAlign: "center", fontSize: 11, color: "#333", marginTop: 20,
    paddingTop: 20, borderTop: "1px solid #111",
  },
};