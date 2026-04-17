"""
Mood Classifier — Streamlit API
================================
Deploy to Streamlit Cloud (free tier).

The app exposes one endpoint consumed by the Next.js portfolio:
  GET /?text=<url-encoded-text>  → returns JSON with mood + scores
"""

import json
import re
from pathlib import Path

import streamlit as st
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mood API", page_icon="🎵", layout="centered")

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "mood_model_onnx"

@st.cache_resource(show_spinner="Loading mood model…")
def load_model():
    if not MODEL_DIR.exists():
        st.warning("⚠️ mood_model_onnx/ not found. Using rule-based fallback.")
        return None
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    return pipeline("text-classification", model=ort_model, tokenizer=tokenizer, top_k=None)

pipe = load_model()

# ── Rule-based fallback ───────────────────────────────────────────────────────
KEYWORD_RULES = [
    ("happy",       r"happy|joy|great|amazing|excited|wonderful|fantastic|thrilled|elated|glad|cheerful"),
    ("sad",         r"sad|depressed|blue|down|lonely|miss|cry|grief|hopeless|broken|tears|hurt"),
    ("angry",       r"angry|mad|frustrated|hate|furious|rage|annoyed|irritated|livid|infuriated"),
    ("calm",        r"calm|peaceful|relaxed|chill|zen|serene|tranquil|quiet|still|gentle"),
    ("anxious",     r"anxious|stressed|worried|nervous|panic|overwhelm|tense|uneasy|scared|fear"),
    ("focused",     r"focus|work|study|concentrate|grind|productive|deep|task|code|build"),
    ("loved",       r"loved|love|cherish|adore|affection|warm|cared|appreciated|tender"),
    ("melancholic", r"nostalgic|wistful|bittersweet|longing|ache|reflective|heavy|tender|melanchol"),
    ("confident",   r"confident|strong|capable|unstoppable|bold|decisive|assertive|powerful|own it"),
]

def rule_based_predict(text: str) -> dict:
    lower = text.lower()
    scores = {}
    for mood, pattern in KEYWORD_RULES:
        matches = re.findall(pattern, lower)
        if matches:
            scores[mood] = len(matches)
    if not scores:
        scores["calm"] = 1
    top_mood = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = round((scores[top_mood] / total) * 100, 1)
    return {"mood": top_mood, "confidence": min(confidence, 82.0), "source": "rule-based"}

def model_predict(text: str) -> dict:
    if pipe is None:
        return rule_based_predict(text)
    results = pipe(text, truncation=True, max_length=128)[0]
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results_sorted[0]
    return {
        "mood":       top["label"],
        "confidence": round(top["score"] * 100, 1),
        "source":     "distilbert-onnx",
        "all_scores": {r["label"]: round(r["score"] * 100, 1) for r in results_sorted},
    }

# ── Main handler ──────────────────────────────────────────────────────────────
params = st.query_params

if "text" in params:
    # API mode — called by Next.js /api/mood
    text = params["text"]
    if not text or not text.strip():
        result = {"error": "empty text"}
    else:
        prediction = model_predict(text.strip())
        result = {
            "mood":       prediction["mood"],
            "confidence": prediction["confidence"],
            "source":     prediction.get("source", "model"),
            "all_scores": prediction.get("all_scores", {}),
        }
    st.markdown(f'<div id="api-result">{json.dumps(result)}</div>', unsafe_allow_html=True)
    st.stop()

else:
    # Browser UI mode
    st.title("🎵 Mood API")
    st.caption("Powers the Music Mood tool on altheae.dev. Spotify recommendations are handled server-side.")

    text_input = st.text_input("Describe how you're feeling", placeholder="I'm feeling really calm and peaceful today")
    if st.button("Analyze") and text_input.strip():
        with st.spinner("Running inference…"):
            prediction = model_predict(text_input.strip())
        st.success(f"**Mood:** {prediction['mood']}  |  **Confidence:** {prediction['confidence']}%  |  **Source:** {prediction.get('source')}")
        if prediction.get("all_scores"):
            st.markdown("### Score breakdown")
            st.bar_chart(prediction["all_scores"])

    st.divider()
    st.markdown("**API usage:** `GET /?text=<url-encoded-text>`")
    st.caption("✅ ONNX model loaded" if pipe is not None else "⚠️ Rule-based fallback active")
