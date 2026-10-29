import { useState, useCallback } from "react";
import { moderateComment } from "./api";

const TOXICITY_COLOR = {
  Toxic: { bg: "#FF3B30", dim: "#3D1210", text: "#FF6B63" },
  "Non-Toxic": { bg: "#30D158", dim: "#0D2E19", text: "#5DE084" },
};
const SPAM_COLOR = {
  Spam: { bg: "#FF9F0A", dim: "#332200", text: "#FFBF5C" },
  "Not Spam": { bg: "#30D158", dim: "#0D2E19", text: "#5DE084" },
};
const SENTIMENT_COLOR = {
  Positive: { bg: "#30D158", dim: "#0D2E19", text: "#5DE084" },
  Negative: { bg: "#FF3B30", dim: "#3D1210", text: "#FF6B63" },
  Neutral: { bg: "#636366", dim: "#1C1C1E", text: "#AEAEB2" },
};

const TASKS = [["toxicity", "Toxicity"], ["spam", "Spam"], ["sentiment", "Sentiment"]];

function Badge({ label, colorMap }) {
  const c = colorMap[label] || { bg: "#636366", dim: "#1C1C1E", text: "#AEAEB2" };
  return (
    <span style={{ display:"inline-flex", alignItems:"center", gap:6, padding:"4px 12px", borderRadius:999, background:c.dim, border:`1px solid ${c.bg}33`, color:c.text, fontSize:13, fontWeight:600 }}>
      <span style={{ width:7, height:7, borderRadius:"50%", background:c.bg, flexShrink:0 }} />
      {label}
    </span>
  );
}

function Spinner() {
  return <div style={{ width:18, height:18, border:"2px solid #ffffff30", borderTopColor:"#fff", borderRadius:"50%", animation:"spin 0.7s linear infinite" }} />;
}

export default function Dashboard() {
  const [comment, setComment] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyze = useCallback(async () => {
    if (!comment.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const data = await moderateComment(comment);
      setResult(data);
    } catch (e) {
      setError(e.message || "Error — is the backend running on port 8000?");
    } finally {
      setLoading(false);
    }
  }, [comment]);

  const EXAMPLES = [
    "This tutorial was incredibly helpful, thank you!",
    "You are an absolute idiot and should be banned",
    "Subscribe to my channel for free giveaways linktr.ee",
    "The video was uploaded last Thursday.",
  ];

  return (
    <div style={{ minHeight:"100vh", background:"#0A0A12", color:"#E8E6F0", fontFamily:"sans-serif", padding:"0 0 60px" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <div style={{ borderBottom:"1px solid #1A1A2E", padding:"16px 28px", display:"flex", alignItems:"center", gap:12, background:"#0D0D18" }}>
        <div style={{ width:34, height:34, background:"linear-gradient(135deg,#534AB7,#8B5CF6)", borderRadius:9, display:"flex", alignItems:"center", justifyContent:"center", fontSize:16 }}>🛡️</div>
        <div>
          <div style={{ fontWeight:700, fontSize:15 }}>CommentGuard AI</div>
          <div style={{ fontSize:11, color:"#636380" }}>MODERATION DASHBOARD</div>
        </div>
      </div>

      <div style={{ maxWidth:860, margin:"0 auto", padding:"36px 24px 0" }}>
        <h1 style={{ fontSize:36, fontWeight:700, letterSpacing:"-0.03em", marginBottom:8, color:"#E8E6F0" }}>AI Comment Moderation</h1>
        <p style={{ color:"#636380", fontSize:14, marginBottom:28 }}>Detect toxicity, spam and sentiment — TF-IDF + scikit-learn classifiers trained on public datasets (tweets and SMS). Predictions can be wrong.</p>

        <div style={{ background:"#111118", border:"1px solid #2A2A3D", borderRadius:16, overflow:"hidden", marginBottom:20 }}>
          <textarea value={comment} onChange={e => setComment(e.target.value)} placeholder="Paste or type a social media comment here…" rows={4}
            style={{ width:"100%", background:"transparent", border:"none", color:"#E8E6F0", fontSize:14, lineHeight:1.7, padding:"18px 20px", resize:"none", boxSizing:"border-box", fontFamily:"inherit" }} />
          <div style={{ borderTop:"1px solid #1A1A2E", padding:"10px 14px", display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:8 }}>
            <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
              <span style={{ fontSize:11, color:"#636380", alignSelf:"center" }}>Try:</span>
              {EXAMPLES.map((ex, i) => (
                <button key={i} onClick={() => setComment(ex)}
                  style={{ background:"#1C1C2E", border:"1px solid #2A2A3D", borderRadius:7, color:"#AEAEB2", fontSize:11, padding:"4px 10px", cursor:"pointer", fontFamily:"inherit" }}>
                  {["✅ Nice","🚫 Toxic","📣 Spam","😐 Neutral"][i]}
                </button>
              ))}
            </div>
            <button onClick={analyze} disabled={loading || !comment.trim()}
              style={{ background: loading || !comment.trim() ? "#2A2A3D" : "linear-gradient(135deg,#534AB7,#8B5CF6)", color: loading || !comment.trim() ? "#636380" : "#fff", border:"none", borderRadius:10, padding:"10px 20px", fontSize:13, fontWeight:600, cursor: loading || !comment.trim() ? "not-allowed" : "pointer", display:"flex", alignItems:"center", gap:8, fontFamily:"inherit" }}>
              {loading ? <Spinner /> : "⚡"}{loading ? "Analyzing…" : "Analyze Comment"}
            </button>
          </div>
        </div>

        {error && <div style={{ background:"#3D1210", border:"1px solid #FF3B3033", borderRadius:10, padding:"12px 16px", color:"#FF6B63", fontSize:13, marginBottom:20 }}>⚠️ {error}</div>}

        {result && (
          <div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:14, marginBottom:14 }}>
              {[["⚠️","Toxicity",result.toxicity,TOXICITY_COLOR],["📣","Spam",result.spam,SPAM_COLOR],["💬","Sentiment",result.sentiment,SENTIMENT_COLOR]].map(([icon,title,val,map]) => (
                <div key={title} style={{ background:"#111118", border:"1px solid #2A2A3D", borderRadius:14, padding:"16px 18px" }}>
                  <div style={{ fontSize:11, fontWeight:700, letterSpacing:"0.1em", color:"#636380", textTransform:"uppercase", marginBottom:10 }}>{icon} {title}</div>
                  <Badge label={val} colorMap={map} />
                </div>
              ))}
            </div>
            <div style={{ background:"#111118", border:"1px solid #2A2A3D", borderRadius:14, padding:"16px 18px", marginBottom:14 }}>
              <div style={{ fontSize:11, fontWeight:700, letterSpacing:"0.08em", color:"#636380", textTransform:"uppercase", marginBottom:4 }}>Predicted-class probability (per task)</div>
              <div style={{ fontSize:11, color:"#636380", marginBottom:12 }}>Model-estimated probability of each predicted label. Not a guarantee of correctness; the three tasks are independent.</div>
              {TASKS.map(([key, title]) => {
                const s = result.scores?.[key];
                if (!s) return null;
                const pct = Math.round(s.probability * 100);
                const color = s.uncertain ? "#FF9F0A" : "#30D158";
                return (
                  <div key={key} style={{ marginBottom:12 }}>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                      <span style={{ fontSize:12, color:"#AEAEB2" }}>{title}: {s.label}{s.uncertain ? "  ⚠️ low confidence — human review advised" : ""}</span>
                      <span style={{ fontSize:14, fontWeight:700, color, fontFamily:"monospace" }}>{pct}%</span>
                    </div>
                    <div style={{ height:5, borderRadius:99, background:"#2A2A3D" }}>
                      <div style={{ height:"100%", width:`${pct}%`, borderRadius:99, background:color }} />
                    </div>
                  </div>
                );
              })}
              <div style={{ fontSize:11, color:"#636380", marginTop:6 }}>Model version {result.model_version}. Automated moderation makes mistakes — keep human review for consequential decisions.</div>
            </div>
            <div style={{ background:"#111118", border:"1px solid #2A2A3D", borderRadius:14, padding:"16px 18px" }}>
              <div style={{ fontSize:11, fontWeight:700, letterSpacing:"0.08em", color:"#636380", textTransform:"uppercase", marginBottom:8 }}>Analyzed Comment</div>
              <div style={{ fontSize:13, color:"#AEAEB2", fontStyle:"italic" }}>"{result.comment}"</div>
            </div>
          </div>
        )}

        {!result && !loading && !error && (
          <div style={{ textAlign:"center", padding:"50px 20px", color:"#636380" }}>
            <div style={{ fontSize:44, marginBottom:12 }}>🛡️</div>
            <div style={{ fontSize:15, fontWeight:600, color:"#AEAEB2", marginBottom:6 }}>No comment analyzed yet</div>
            <div style={{ fontSize:13 }}>Type a comment or click an example above.</div>
          </div>
        )}
      </div>
    </div>
  );
}