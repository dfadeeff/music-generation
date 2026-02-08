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

  // Sliders
  const [sliders, setSliders] = useState({ energy: null, style: null, warmth: null, arc: null });
  const [slidersEnabled, setSlidersEnabled] = useState({ energy: false, style: false, warmth: false, arc: false });

  // Feedback
  const [fbRating, setFbRating] = useState(0);
  const [fbReplay, setFbReplay] = useState(false);
  const [fbPreferred, setFbPreferred] = useState("");
  const [fbNotes, setFbNotes] = useState("");
  const [fbSubmitted, setFbSubmitted] = useState(false);
  const [learningStats, setLearningStats] = useState(null);

  // Insights
  const [insights, setInsights] = useState(null);

  useEffect(() => {
    fetch(`${API}/history`).then(r => r.json()).then(setHistory).catch(() => {});
    fetch(`${API}/learned`).then(r => r.json()).then(setInsights).catch(() => {});
  }, []);

  const refreshHistory = async () => {
    const r = await fetch(`${API}/history`);
    setHistory(await r.json());
  };

  const refreshInsights = async () => {
    const r = await fetch(`${API}/learned`);
    setInsights(await r.json());
  };

  // --- Voice ---
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
    } catch { setError("Microphone access denied"); }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const uploadVoice = (e) => {
    const file = e.target.files[0];
    if (file) { setVoiceBlob(file); setVoiceName(file.name); }
  };

  // --- Images ---
  const addImages = (e) => {
    const files = Array.from(e.target.files);
    setImages(prev => [...prev, ...files]);
    setImagePreviews(prev => [...prev, ...files.map(f => URL.createObjectURL(f))]);
  };

  const removeImage = (i) => {
    setImages(prev => prev.filter((_, idx) => idx !== i));
    setImagePreviews(prev => prev.filter((_, idx) => idx !== i));
  };

  // --- Sliders ---
  const toggleSlider = (key) => {
    setSlidersEnabled(prev => {
      const next = { ...prev, [key]: !prev[key] };
      if (!next[key]) setSliders(s => ({ ...s, [key]: null }));
      else setSliders(s => ({ ...s, [key]: s[key] ?? 50 }));
      return next;
    });
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
    setFbRating(0);
    setFbReplay(false);
    setFbPreferred("");
    setFbNotes("");
    setFbSubmitted(false);
    setLearningStats(null);

    try {
      let res;
      if (hasImages || hasVoice) {
        const fd = new FormData();
        if (hasText) fd.append("text", prompt.trim());
        images.forEach(f => fd.append("images", f));
        if (hasVoice) fd.append("voice", voiceBlob, "voice.wav");
        fd.append("duration", duration);
        Object.entries(sliders).forEach(([k, v]) => { if (v !== null) fd.append(k, v); });
        res = await fetch(`${API}/generate-multimodal`, { method: "POST", body: fd });
      } else {
        res = await fetch(`${API}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: prompt.trim(), duration,
            ...Object.fromEntries(Object.entries(sliders).filter(([, v]) => v !== null)),
          }),
        });
      }
      if (!res.ok) throw new Error("Generation failed");
      const data = await res.json();
      setResult(data);
      await refreshHistory();
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const submitFeedback = async () => {
    if (!result || fbRating === 0) return;
    try {
      const res = await fetch(`${API}/feedback/${result.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rating: fbRating,
          replay: fbReplay,
          preferred_version: fbPreferred,
          notes: fbNotes,
        }),
      });
      const data = await res.json();
      setFbSubmitted(true);
      setLearningStats(data.learning_stats);
      await refreshHistory();
      await refreshInsights();
    } catch { setError("Failed to submit feedback"); }
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !loading) { e.preventDefault(); generate(); }
  };

  const canGenerate = prompt.trim() || images.length > 0 || voiceBlob;

  const tabs = [
    { id: "create", label: "Create" },
    { id: "discover", label: "Discover" },
    { id: "insights", label: "Insights" },
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
            <button key={t.id} onClick={() => setTab(t.id)}
              style={tab === t.id ? { ...s.navBtn, ...s.navActive } : s.navBtn}>
              {t.label}
              {t.id === "insights" && insights?.reflection_count > 0 && (
                <span style={s.insightsDot} />
              )}
            </button>
          ))}
        </nav>

        {/* ===================== CREATE TAB ===================== */}
        {tab === "create" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Text */}
            <div style={s.card}>
              <div style={s.sectionLabel}>Text</div>
              <textarea style={s.input} value={prompt} onChange={e => setPrompt(e.target.value)}
                onKeyDown={onKey} placeholder="Describe the music you want to create..." rows={3} disabled={loading} />
              <div style={s.presetRow}>
                {PRESETS.map(p => (
                  <button key={p} onClick={() => setPrompt(p)} style={s.preset}>
                    {p.length > 40 ? p.slice(0, 40) + "..." : p}
                  </button>
                ))}
              </div>
            </div>

            {/* Images */}
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
              <input ref={fileRef} type="file" accept="image/*" multiple onChange={addImages} style={{ display: "none" }} />
            </div>

            {/* Voice */}
            <div style={s.card}>
              <div style={s.sectionLabel}>Voice <span style={s.optional}>(optional)</span></div>
              {voiceBlob ? (
                <div style={s.voiceReady}>
                  <span style={{ fontSize: 18 }}>&#127908;</span>
                  <span style={s.voiceText}>{voiceName}</span>
                  <button onClick={() => { setVoiceBlob(null); setVoiceName(""); }} style={s.voiceRemove}>Remove</button>
                </div>
              ) : (
                <div style={s.voiceRow}>
                  <button onClick={isRecording ? stopRecording : startRecording}
                    style={isRecording ? { ...s.voiceBtn, ...s.voiceBtnRec } : s.voiceBtn}>
                    {isRecording ? "Stop Recording" : "Record"}
                  </button>
                  <span style={{ color: "#333", fontSize: 12 }}>or</span>
                  <button onClick={() => voiceFileRef.current?.click()} style={s.voiceBtn}>Upload Audio</button>
                  <input ref={voiceFileRef} type="file" accept="audio/*" onChange={uploadVoice} style={{ display: "none" }} />
                </div>
              )}
            </div>

            {/* Emotion Sliders */}
            <div style={s.card}>
              <div style={s.sectionLabel}>Emotion Sliders <span style={s.optional}>(override AI-detected)</span></div>
              <div style={s.sliderGrid}>
                {Object.entries(SLIDER_LABELS).map(([key, label]) => (
                  <div key={key}>
                    <div style={s.sliderHeader}>
                      <label style={s.sliderLabel}>{label.name}</label>
                      <button onClick={() => toggleSlider(key)}
                        style={slidersEnabled[key] ? { ...s.toggleBtn, ...s.toggleActive } : s.toggleBtn}>
                        {slidersEnabled[key] ? "Manual" : "Auto"}
                      </button>
                    </div>
                    <div style={{ opacity: slidersEnabled[key] ? 1 : 0.3, transition: "opacity 0.2s" }}>
                      <input type="range" min={0} max={100} value={sliders[key] ?? 50}
                        onChange={e => setSliders(p => ({ ...p, [key]: parseInt(e.target.value) }))}
                        disabled={!slidersEnabled[key]} style={s.slider} />
                      <div style={s.sliderRange}>
                        <span>{label.left}</span>
                        <span style={s.sliderVal}>{sliders[key] ?? "\u2014"}</span>
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
                    <button key={d.tokens} onClick={() => setDuration(d.tokens)}
                      style={duration === d.tokens ? { ...s.durBtn, ...s.durActive } : s.durBtn}>{d.label}</button>
                  ))}
                </div>
                <button onClick={generate} disabled={loading || !canGenerate}
                  style={{ ...s.primaryBtn, opacity: loading || !canGenerate ? 0.4 : 1 }}>
                  {loading ? "Generating..." : "Generate Music"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ===================== DISCOVER TAB ===================== */}
        {tab === "discover" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {history.length === 0 && <div style={s.emptyState}>No generations yet. Create some music first.</div>}
            {history.map(item => (
              <div key={item.id} style={s.histCard}>
                <div style={s.histTop}>
                  <span style={item.source === "multimodal" ? s.badgeMulti : item.source === "image" ? s.badgeImage : s.badgeText}>
                    {item.source === "multimodal" ? "Multi" : item.source === "image" ? "Image" : "Text"}
                  </span>
                  {item.rating > 0 && <span style={s.histRating}>{"*".repeat(item.rating)}</span>}
                  <span style={s.histTime}>{new Date(item.timestamp * 1000).toLocaleString()}</span>
                </div>
                <p style={s.histPrompt}>{item.original_prompt}</p>
                <p style={s.histEnhanced}>{item.enhanced_prompt}</p>
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
                {item.variations ? (
                  <div style={s.histAudioRow}>
                    {item.variations.map(v => (
                      <div key={v.version} style={s.histAudioItem}>
                        <span style={s.histVersionLabel}>Version {v.version}</span>
                        <audio controls src={`${API}${v.audio_url}`} style={{ width: "100%" }} />
                      </div>
                    ))}
                  </div>
                ) : item.filename && (
                  <audio controls src={`${API}/audio/${item.filename}`} />
                )}
              </div>
            ))}
          </div>
        )}

        {/* ===================== INSIGHTS TAB ===================== */}
        {tab === "insights" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {(!insights || insights.reflection_count === 0) ? (
              <div style={s.card}>
                <div style={s.emptyState}>
                  <p>No reflections yet.</p>
                  <p style={{ marginTop: 8, fontSize: 12, color: "#444" }}>
                    The system learns after every {REFLECTION_THRESHOLD || 5} ratings.
                    {insights && insights.next_reflection_in > 0 && (
                      <> Next reflection in <strong style={{ color: "#6366f1" }}>{insights.next_reflection_in}</strong> more ratings.</>
                    )}
                  </p>
                </div>
              </div>
            ) : (
              <>
                {/* Learning Progress */}
                <div style={s.card}>
                  <div style={s.sectionLabel}>Learning Progress</div>
                  <div style={s.statsGrid}>
                    <div style={s.statBox}>
                      <div style={s.statVal}>{insights.reflection_count}</div>
                      <div style={s.statLabel}>Reflections</div>
                    </div>
                    <div style={s.statBox}>
                      <div style={s.statVal}>{insights.total_sessions}</div>
                      <div style={s.statLabel}>Sessions</div>
                    </div>
                    <div style={s.statBox}>
                      <div style={s.statVal}>{insights.emotions_learned?.length || 0}</div>
                      <div style={s.statLabel}>Emotions Learned</div>
                    </div>
                    <div style={s.statBox}>
                      <div style={s.statVal}>{insights.next_reflection_in}</div>
                      <div style={s.statLabel}>Until Next Cycle</div>
                    </div>
                  </div>
                </div>

                {/* Global Rules */}
                {(insights.global_rules?.positive?.length > 0 || insights.global_rules?.negative?.length > 0) && (
                  <div style={s.card}>
                    <div style={s.sectionLabel}>Global Rules</div>
                    {insights.global_rules.positive?.length > 0 && (
                      <div style={{ marginBottom: 12 }}>
                        <div style={s.ruleHeader}>What works well</div>
                        {insights.global_rules.positive.map((r, i) => (
                          <div key={i} style={s.rulePositive}>{r}</div>
                        ))}
                      </div>
                    )}
                    {insights.global_rules.negative?.length > 0 && (
                      <div>
                        <div style={s.ruleHeader}>What to avoid</div>
                        {insights.global_rules.negative.map((r, i) => (
                          <div key={i} style={s.ruleNegative}>{r}</div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Per-Emotion Knowledge */}
                {Object.keys(insights.emotion_profiles || {}).length > 0 && (
                  <div style={s.card}>
                    <div style={s.sectionLabel}>Emotion-Specific Knowledge</div>
                    {Object.entries(insights.emotion_profiles).map(([emotion, profile]) => (
                      <div key={emotion} style={s.emotionBlock}>
                        <div style={s.emotionName}>{emotion}</div>
                        {profile.prompt_principles?.length > 0 && (
                          <div style={{ marginBottom: 6 }}>
                            <span style={s.emotionSubLabel}>Principles: </span>
                            {profile.prompt_principles.map((p, i) => (
                              <span key={i} style={s.emotionPill}>{p}</span>
                            ))}
                          </div>
                        )}
                        {profile.preferred_params && Object.keys(profile.preferred_params).length > 0 && (
                          <div style={s.paramRanges}>
                            {Object.entries(profile.preferred_params).map(([k, range]) => (
                              <span key={k} style={s.paramRange}>{k}: {Array.isArray(range) ? `${range[0]}-${range[1]}` : range}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Param Insights */}
                {insights.param_insights && Object.keys(insights.param_insights).length > 0 && (
                  <div style={s.card}>
                    <div style={s.sectionLabel}>Parameter Insights</div>
                    {Object.entries(insights.param_insights).map(([param, insight]) => (
                      <div key={param} style={s.paramInsight}>
                        <span style={s.paramInsightName}>{param}</span>
                        <span style={s.paramInsightText}>{insight}</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ===================== LOADING ===================== */}
        {loading && (
          <div style={s.loadingCard}>
            <div style={s.waveRow}>
              {[0, 1, 2, 3, 4].map(i => (
                <div key={i} style={{ ...s.waveBar, animationDelay: `${i * 0.15}s` }} />
              ))}
            </div>
            <span style={s.loadingText}>Analyzing emotions & generating A/B variations...</span>
          </div>
        )}

        {/* ===================== ERROR ===================== */}
        {error && <div style={s.errorCard}>Error: {error}</div>}

        {/* ===================== RESULT ===================== */}
        {result && tab === "create" && (
          <div style={s.resultCard}>
            {/* Emotion Profile */}
            {result.emotion_profile && (
              <div>
                <div style={s.profileHeader}>
                  <span style={s.profileEmotion}>{result.emotion_profile.emotion}</span>
                  {result.emotion_profile.emotions?.length > 1 && (
                    <div style={s.emotionPills}>
                      {result.emotion_profile.emotions.map(e => (
                        <span key={e} style={s.emotionPillSmall}>{e}</span>
                      ))}
                    </div>
                  )}
                </div>
                {result.emotion_profile.description && (
                  <p style={s.profileDesc}>{result.emotion_profile.description}</p>
                )}
                <div style={s.profileBars}>
                  {["energy", "style", "warmth", "arc"].map(k => (
                    <div key={k}>
                      <div style={s.profileBarHeader}>
                        <span style={s.profileBarLabel}>{SLIDER_LABELS[k].name}</span>
                        <span style={s.profileBarVal}>{result.emotion_profile[k]}</span>
                      </div>
                      <div style={s.profileBarTrack}>
                        <div style={{ ...s.profileBarFill, width: `${result.emotion_profile[k]}%` }} />
                      </div>
                      <div style={s.profileBarRange}>
                        <span>{SLIDER_LABELS[k].left}</span>
                        <span>{SLIDER_LABELS[k].right}</span>
                      </div>
                    </div>
                  ))}
                </div>
                {result.emotion_profile.overrides?.length > 0 && (
                  <div style={s.overrideNote}>You overrode: {result.emotion_profile.overrides.join(", ")}</div>
                )}
              </div>
            )}

            {/* Enhanced Prompt */}
            <div style={s.section}>
              <p style={s.resultLabel}>Enhanced prompt</p>
              <p style={s.resultEnhanced}>{result.enhanced_prompt}</p>
            </div>

            {/* A/B Comparison */}
            {result.variations && (
              <div style={s.section}>
                <p style={s.resultLabel}>A/B Comparison</p>
                <div style={s.abRow}>
                  {result.variations.map(v => (
                    <div key={v.version} style={s.abCol}>
                      <div style={s.abVersionLabel}>Version {v.version}</div>
                      <audio controls autoPlay={v.version === "A"} src={`${API}${v.audio_url}`} style={{ width: "100%" }} />
                    </div>
                  ))}
                </div>
                {result.variations.length >= 2 && (
                  <div style={s.prefRow}>
                    <span style={s.prefLabel}>Which version do you prefer?</span>
                    <div style={s.prefBtns}>
                      {["A", "B", ""].map(v => (
                        <button key={v} onClick={() => setFbPreferred(v)}
                          style={fbPreferred === v ? { ...s.prefBtn, ...s.prefBtnActive } : s.prefBtn}>
                          {v || "No preference"}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Explainability */}
            {result.explanation && (
              <div style={s.section}>
                <p style={s.resultLabel}>How your inputs became music</p>
                <p style={s.explainNarrative}>{result.explanation.narrative}</p>
                {result.explanation.key_descriptors && (
                  <div style={s.descriptorRow}>
                    {result.explanation.key_descriptors.map((d, i) => (
                      <span key={i} style={s.descriptor}>{d}</span>
                    ))}
                  </div>
                )}
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

            {/* Feedback Form */}
            {!fbSubmitted ? (
              <div style={s.section}>
                <p style={s.resultLabel}>Rate this generation</p>

                {/* 1-5 Rating */}
                <div style={s.ratingRow}>
                  {[1, 2, 3, 4, 5].map(n => (
                    <button key={n} onClick={() => setFbRating(n)}
                      style={fbRating >= n ? { ...s.starBtn, ...s.starActive } : s.starBtn}>
                      {fbRating >= n ? "\u2605" : "\u2606"}
                    </button>
                  ))}
                  <span style={s.ratingLabel}>
                    {fbRating === 0 ? "" : fbRating <= 2 ? "Needs work" : fbRating <= 3 ? "Okay" : fbRating <= 4 ? "Good" : "Amazing"}
                  </span>
                </div>

                {/* Replay toggle */}
                <div style={s.replayRow}>
                  <span style={{ fontSize: 13, color: "#888" }}>Would you listen again?</span>
                  <button onClick={() => setFbReplay(!fbReplay)}
                    style={fbReplay ? { ...s.toggleChip, ...s.toggleChipActive } : s.toggleChip}>
                    {fbReplay ? "Yes" : "No"}
                  </button>
                </div>

                {/* Notes */}
                <textarea style={{ ...s.input, marginTop: 10, fontSize: 13 }} value={fbNotes}
                  onChange={e => setFbNotes(e.target.value)} placeholder="Any specific feedback? (optional)" rows={2} />

                <button onClick={submitFeedback} disabled={fbRating === 0}
                  style={{ ...s.primaryBtn, marginTop: 12, opacity: fbRating === 0 ? 0.4 : 1, width: "100%" }}>
                  Submit Feedback
                </button>
              </div>
            ) : (
              <div style={s.section}>
                <p style={s.resultLabel}>Feedback submitted</p>
                {learningStats && (
                  <div style={s.statsGrid}>
                    <div style={s.statBox}>
                      <div style={s.statVal}>{learningStats.total_sessions}</div>
                      <div style={s.statLabel}>Total Sessions</div>
                    </div>
                    <div style={s.statBox}>
                      <div style={s.statVal}>{learningStats.avg_rating}</div>
                      <div style={s.statLabel}>Avg Rating</div>
                    </div>
                    <div style={s.statBox}>
                      <div style={s.statVal}>{learningStats.replay_rate}%</div>
                      <div style={s.statLabel}>Replay Rate</div>
                    </div>
                    <div style={s.statBox}>
                      <div style={s.statVal}>{learningStats.next_reflection_in}</div>
                      <div style={s.statLabel}>Until Next Learning</div>
                    </div>
                  </div>
                )}
                {learningStats?.reflections_completed > 0 && (
                  <p style={{ fontSize: 12, color: "#6366f1", marginTop: 8 }}>
                    {learningStats.reflections_completed} reflection{learningStats.reflections_completed > 1 ? "s" : ""} completed
                    {learningStats.emotions_learned?.length > 0 && (
                      <> &middot; Learned: {learningStats.emotions_learned.join(", ")}</>
                    )}
                  </p>
                )}
                {learningStats?.next_reflection_in === 0 && (
                  <p style={{ fontSize: 12, color: "#4ade80", marginTop: 4 }}>
                    A new reflection cycle just ran! Check the Insights tab.
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        <footer style={s.footer}>
          Powered by Meta MusicGen &middot; OpenAI &middot; Whisper &middot; Built for Global Hackathon
        </footer>
      </div>
    </div>
  );
}

const REFLECTION_THRESHOLD = 5;

/* ——————————————— Styles ——————————————— */
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
  logoDot: { width: 10, height: 10, borderRadius: "50%", background: "linear-gradient(135deg, #6366f1, #a855f7)" },
  logoText: { fontSize: 26, fontWeight: 700, letterSpacing: "-0.5px", color: "#fff" },
  tagline: { fontSize: 14, color: "#555", marginTop: 6 },

  nav: { display: "flex", gap: 2, background: "#0d0d0d", borderRadius: 12, padding: 3, border: "1px solid #1a1a1a" },
  navBtn: {
    flex: 1, padding: "10px 0", background: "transparent", border: "none", color: "#666",
    borderRadius: 10, cursor: "pointer", fontSize: 13, fontWeight: 500, transition: "all 0.2s", position: "relative",
  },
  navActive: { background: "#1a1a1a", color: "#fff", boxShadow: "0 1px 4px rgba(0,0,0,0.3)" },
  insightsDot: {
    position: "absolute", top: 6, right: "30%", width: 6, height: 6, borderRadius: "50%",
    background: "#6366f1",
  },

  card: { background: "#0d0d0d", borderRadius: 16, padding: 20, border: "1px solid #1a1a1a", animation: "slideUp 0.3s ease" },
  sectionLabel: { fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1, color: "#444", marginBottom: 12 },
  optional: { fontWeight: 400, color: "#333", textTransform: "none", letterSpacing: 0 },

  input: { width: "100%", background: "transparent", border: "none", color: "#e0e0e0", fontSize: 15, resize: "none", outline: "none", fontFamily: "inherit", lineHeight: 1.6 },
  toolbar: { display: "flex", justifyContent: "space-between", alignItems: "center" },
  durRow: { display: "flex", gap: 6 },
  durBtn: { background: "#111", border: "1px solid #222", color: "#666", padding: "6px 16px", borderRadius: 8, cursor: "pointer", fontSize: 12, fontWeight: 500, transition: "all 0.15s" },
  durActive: { background: "#fff", color: "#000", borderColor: "#fff" },
  primaryBtn: { background: "linear-gradient(135deg, #6366f1, #8b5cf6)", color: "#fff", border: "none", padding: "10px 28px", borderRadius: 10, fontSize: 13, fontWeight: 600, cursor: "pointer", transition: "all 0.2s" },
  presetRow: { display: "flex", flexWrap: "wrap", gap: 6, marginTop: 14, paddingTop: 14, borderTop: "1px solid #1a1a1a" },
  preset: { background: "#111", border: "1px solid #1a1a1a", color: "#555", padding: "5px 12px", borderRadius: 20, cursor: "pointer", fontSize: 11, transition: "all 0.15s" },

  imageGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))", gap: 8 },
  imageThumb: { position: "relative", borderRadius: 10, overflow: "hidden", aspectRatio: "1", border: "1px solid #222" },
  thumbImg: { width: "100%", height: "100%", objectFit: "cover" },
  removeBtn: { position: "absolute", top: 4, right: 4, width: 20, height: 20, borderRadius: "50%", background: "rgba(0,0,0,0.7)", border: "1px solid #333", color: "#999", fontSize: 11, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" },
  addImageBtn: { borderRadius: 10, border: "2px dashed #222", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", aspectRatio: "1", cursor: "pointer", transition: "all 0.15s", minHeight: 80 },

  voiceRow: { display: "flex", alignItems: "center", gap: 12 },
  voiceBtn: { background: "#111", border: "1px solid #222", color: "#888", padding: "8px 20px", borderRadius: 8, cursor: "pointer", fontSize: 12, fontWeight: 500, transition: "all 0.15s" },
  voiceBtnRec: { background: "#2a0a0a", borderColor: "#f87171", color: "#f87171", animation: "pulse 1s infinite" },
  voiceReady: { display: "flex", alignItems: "center", gap: 10, padding: "8px 14px", background: "#111", borderRadius: 8, border: "1px solid #222" },
  voiceText: { flex: 1, fontSize: 13, color: "#888" },
  voiceRemove: { background: "transparent", border: "1px solid #333", color: "#666", padding: "3px 10px", borderRadius: 6, cursor: "pointer", fontSize: 11 },

  sliderGrid: { display: "flex", flexDirection: "column", gap: 16 },
  sliderHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 },
  sliderLabel: { fontSize: 13, fontWeight: 500, color: "#aaa" },
  toggleBtn: { background: "#111", border: "1px solid #222", color: "#555", padding: "3px 10px", borderRadius: 6, cursor: "pointer", fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 },
  toggleActive: { background: "#1a1a2a", borderColor: "#6366f1", color: "#6366f1" },
  slider: { width: "100%", height: 4, appearance: "none", WebkitAppearance: "none", background: "#222", borderRadius: 2, outline: "none", cursor: "pointer", accentColor: "#6366f1" },
  sliderRange: { display: "flex", justifyContent: "space-between", fontSize: 10, color: "#444", marginTop: 4 },
  sliderVal: { fontWeight: 600, color: "#6366f1" },

  loadingCard: { display: "flex", alignItems: "center", gap: 16, padding: 20, background: "#0d0d0d", borderRadius: 16, border: "1px solid #1a1a1a" },
  waveRow: { display: "flex", alignItems: "center", gap: 3 },
  waveBar: { width: 3, height: 8, background: "#6366f1", borderRadius: 2, animation: "waveform 0.8s ease-in-out infinite" },
  loadingText: { fontSize: 13, color: "#555" },
  errorCard: { padding: 16, background: "#1a0a0a", borderRadius: 12, border: "1px solid #331111", color: "#ef4444", fontSize: 13 },

  resultCard: { background: "#0d0d0d", borderRadius: 16, padding: 24, border: "1px solid #1a1a1a", animation: "slideUp 0.3s ease", display: "flex", flexDirection: "column", gap: 20 },
  section: { borderTop: "1px solid #1a1a1a", paddingTop: 16 },
  resultLabel: { fontSize: 11, color: "#444", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 },
  resultEnhanced: { fontSize: 13, color: "#888", fontStyle: "italic", lineHeight: 1.5 },

  profileHeader: { display: "flex", alignItems: "center", gap: 10, marginBottom: 6 },
  profileEmotion: { fontSize: 18, fontWeight: 700, color: "#fff", textTransform: "capitalize" },
  profileDesc: { fontSize: 13, color: "#666", fontStyle: "italic", marginBottom: 12 },
  emotionPills: { display: "flex", gap: 4, flexWrap: "wrap" },
  emotionPillSmall: { fontSize: 10, background: "#1a1a2a", color: "#8b8bf5", padding: "2px 8px", borderRadius: 10, textTransform: "capitalize" },
  profileBars: { display: "flex", flexDirection: "column", gap: 10 },
  profileBarHeader: { display: "flex", justifyContent: "space-between", marginBottom: 4 },
  profileBarLabel: { fontSize: 11, fontWeight: 500, color: "#666", textTransform: "uppercase", letterSpacing: 0.5 },
  profileBarVal: { fontSize: 11, fontWeight: 600, color: "#6366f1" },
  profileBarTrack: { width: "100%", height: 4, background: "#1a1a1a", borderRadius: 2, overflow: "hidden" },
  profileBarFill: { height: "100%", borderRadius: 2, background: "linear-gradient(90deg, #6366f1, #a855f7)", transition: "width 0.6s ease" },
  profileBarRange: { display: "flex", justifyContent: "space-between", fontSize: 9, color: "#333", marginTop: 2 },
  overrideNote: { marginTop: 8, fontSize: 11, color: "#6366f1", fontStyle: "italic" },

  /* A/B */
  abRow: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 8 },
  abCol: { display: "flex", flexDirection: "column", gap: 6 },
  abVersionLabel: { fontSize: 12, fontWeight: 600, color: "#888", textTransform: "uppercase", letterSpacing: 0.5 },
  prefRow: { marginTop: 12, display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" },
  prefLabel: { fontSize: 12, color: "#666" },
  prefBtns: { display: "flex", gap: 6 },
  prefBtn: { background: "#111", border: "1px solid #222", color: "#666", padding: "5px 14px", borderRadius: 6, cursor: "pointer", fontSize: 12, transition: "all 0.15s" },
  prefBtnActive: { background: "#1a1a2a", borderColor: "#6366f1", color: "#6366f1" },

  /* Explainability */
  explainNarrative: { fontSize: 13, color: "#999", lineHeight: 1.6, marginBottom: 10 },
  descriptorRow: { display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12 },
  descriptor: { fontSize: 10, background: "#111", border: "1px solid #1a1a1a", color: "#888", padding: "3px 10px", borderRadius: 12 },
  timeline: { display: "flex", flexDirection: "column" },
  timelineStep: { display: "flex", gap: 12, minHeight: 48 },
  timelineDot: { display: "flex", flexDirection: "column", alignItems: "center", width: 16, flexShrink: 0 },
  timelineDotInner: { width: 8, height: 8, borderRadius: "50%", background: "linear-gradient(135deg, #6366f1, #a855f7)", flexShrink: 0 },
  timelineLine: { width: 1, flex: 1, background: "#222", marginTop: 4 },
  timelineContent: { display: "flex", flexDirection: "column", gap: 2, paddingBottom: 12 },
  timelineStepName: { fontSize: 12, fontWeight: 600, color: "#ccc" },
  timelineStepDetail: { fontSize: 11, color: "#555", lineHeight: 1.4 },

  /* Feedback */
  ratingRow: { display: "flex", alignItems: "center", gap: 4 },
  starBtn: { background: "transparent", border: "none", fontSize: 24, cursor: "pointer", color: "#333", transition: "color 0.15s", padding: "2px" },
  starActive: { color: "#f59e0b" },
  ratingLabel: { fontSize: 12, color: "#666", marginLeft: 8 },
  replayRow: { display: "flex", alignItems: "center", gap: 10, marginTop: 10 },
  toggleChip: { background: "#111", border: "1px solid #222", color: "#666", padding: "4px 14px", borderRadius: 6, cursor: "pointer", fontSize: 12, transition: "all 0.15s" },
  toggleChipActive: { background: "#0a1a0a", borderColor: "#1a3a1a", color: "#4ade80" },

  /* Stats Grid */
  statsGrid: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 },
  statBox: { background: "#111", borderRadius: 10, padding: "12px 8px", textAlign: "center", border: "1px solid #1a1a1a" },
  statVal: { fontSize: 20, fontWeight: 700, color: "#fff" },
  statLabel: { fontSize: 9, color: "#555", textTransform: "uppercase", letterSpacing: 0.5, marginTop: 4 },

  /* Insights */
  ruleHeader: { fontSize: 11, fontWeight: 600, color: "#666", marginBottom: 6 },
  rulePositive: { fontSize: 12, color: "#4ade80", padding: "4px 0", paddingLeft: 12, borderLeft: "2px solid #1a3a1a", marginBottom: 4 },
  ruleNegative: { fontSize: 12, color: "#f87171", padding: "4px 0", paddingLeft: 12, borderLeft: "2px solid #3a1a1a", marginBottom: 4 },
  emotionBlock: { padding: "12px 0", borderBottom: "1px solid #1a1a1a" },
  emotionName: { fontSize: 14, fontWeight: 600, color: "#fff", textTransform: "capitalize", marginBottom: 6 },
  emotionSubLabel: { fontSize: 10, color: "#555", textTransform: "uppercase" },
  emotionPill: { fontSize: 11, background: "#111", color: "#888", padding: "2px 8px", borderRadius: 8, marginLeft: 4, display: "inline-block", marginBottom: 2 },
  paramRanges: { display: "flex", gap: 8, flexWrap: "wrap", marginTop: 4 },
  paramRange: { fontSize: 10, background: "#111", color: "#6366f1", padding: "2px 8px", borderRadius: 6, border: "1px solid #1a1a2a" },
  paramInsight: { display: "flex", gap: 8, alignItems: "baseline", padding: "6px 0", borderBottom: "1px solid #111" },
  paramInsightName: { fontSize: 11, fontWeight: 600, color: "#888", textTransform: "capitalize", minWidth: 100 },
  paramInsightText: { fontSize: 12, color: "#666" },

  /* Discover */
  histCard: { background: "#0d0d0d", borderRadius: 14, padding: 18, border: "1px solid #1a1a1a", animation: "slideUp 0.3s ease" },
  histTop: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, gap: 8 },
  badgeText: { fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5, background: "#1a1a2a", color: "#6366f1", padding: "3px 8px", borderRadius: 4 },
  badgeImage: { fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5, background: "#1a2a1a", color: "#4ade80", padding: "3px 8px", borderRadius: 4 },
  badgeMulti: { fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5, background: "#2a1a2a", color: "#a855f7", padding: "3px 8px", borderRadius: 4 },
  histRating: { fontSize: 12, color: "#f59e0b", letterSpacing: 2 },
  histTime: { fontSize: 11, color: "#333", marginLeft: "auto" },
  histPrompt: { fontSize: 14, color: "#ccc", marginBottom: 4, lineHeight: 1.4 },
  histEnhanced: { fontSize: 12, color: "#555", fontStyle: "italic", marginBottom: 14, lineHeight: 1.4 },
  miniProfileRow: { display: "flex", gap: 8, marginBottom: 12 },
  miniBar: { flex: 1 },
  miniBarLabel: { fontSize: 9, color: "#444", textTransform: "uppercase", marginBottom: 3 },
  miniBarTrack: { height: 3, background: "#1a1a1a", borderRadius: 2, overflow: "hidden" },
  miniBarFill: { height: "100%", borderRadius: 2, background: "linear-gradient(90deg, #6366f1, #a855f7)" },
  histAudioRow: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 8 },
  histAudioItem: { display: "flex", flexDirection: "column", gap: 4 },
  histVersionLabel: { fontSize: 10, color: "#555", fontWeight: 600, textTransform: "uppercase" },

  emptyState: { textAlign: "center", padding: 40, color: "#333", fontSize: 14 },
  footer: { textAlign: "center", fontSize: 11, color: "#333", marginTop: 20, paddingTop: 20, borderTop: "1px solid #111" },
};