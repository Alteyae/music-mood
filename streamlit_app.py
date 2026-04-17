"""
Mood Classifier + Spotify Recommender — Streamlit API
======================================================
Deploy this to Streamlit Cloud (free tier).

Environment variables to set in Streamlit Cloud secrets:
  SPOTIFY_CLIENT_ID     = "your_client_id"
  SPOTIFY_CLIENT_SECRET = "your_client_secret"

The app exposes one endpoint consumed by the Next.js portfolio:
  POST /  (via st.experimental_get_query_params hack — see notes below)

Because Streamlit isn't a REST framework, we use a clean pattern:
  - Query param  ?text=<url-encoded-text>  triggers inference
  - Response rendered as JSON via st.json() — scraped by the API route
  - A lightweight UI is shown for direct browser visits

For production, consider replacing with a FastAPI app on Render/Railway.
But Streamlit Cloud is free and works fine for portfolio traffic.
"""

import json
import os
import time
import base64
from pathlib import Path

import streamlit as st
import requests
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mood API",
    page_icon="🎵",
    layout="centered",
)

# ── Spotify audio feature targets per mood ────────────────────────────────────
MOOD_AUDIO_FEATURES = {
    "happy":       {"valence": 0.85, "energy": 0.80, "target_tempo": 128},
    "sad":         {"valence": 0.15, "energy": 0.25, "target_tempo": 70},
    "angry":       {"valence": 0.20, "energy": 0.90, "target_tempo": 155},
    "calm":        {"valence": 0.65, "energy": 0.20, "target_tempo": 75},
    "anxious":     {"valence": 0.25, "energy": 0.75, "target_tempo": 145},
    "focused":     {"valence": 0.50, "energy": 0.50, "target_tempo": 105},
    "loved":       {"valence": 0.80, "energy": 0.45, "target_tempo": 95},
    "melancholic": {"valence": 0.20, "energy": 0.25, "target_tempo": 72},
    "confident":   {"valence": 0.75, "energy": 0.82, "target_tempo": 122},
}

# Seed artists/genres per mood for better Spotify recommendations
MOOD_SEEDS = {
    "happy":       {"seed_genres": "pop,happy,dance"},
    "sad":         {"seed_genres": "sad,acoustic,singer-songwriter"},
    "angry":       {"seed_genres": "metal,rock,punk"},
    "calm":        {"seed_genres": "ambient,classical,sleep"},
    "anxious":     {"seed_genres": "alternative,indie,post-rock"},
    "focused":     {"seed_genres": "study,classical,instrumental"},
    "loved":       {"seed_genres": "romance,soul,r-n-b"},
    "melancholic": {"seed_genres": "indie,folk,sad"},
    "confident":   {"seed_genres": "hip-hop,power-pop,work-out"},
}

# ── Load model (cached — loads once per Streamlit session) ────────────────────
MODEL_DIR = Path(__file__).parent / "mood_model_onnx"

@st.cache_resource(show_spinner="Loading mood model…")
def load_model():
    """Load ONNX model. Falls back to rule-based if model not found."""
    if not MODEL_DIR.exists():
        st.warning("⚠️ mood_model_onnx/ not found. Using rule-based fallback.")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    pipe = pipeline("text-classification", model=ort_model, tokenizer=tokenizer, top_k=None)
    return pipe, tokenizer

pipe, tokenizer = load_model()

# ── Rule-based fallback (used when model not yet uploaded) ────────────────────
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

import re

def rule_based_predict(text: str) -> dict:
    import random
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
    # results is list of {label, score}
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results_sorted[0]
    return {
        "mood":       top["label"],
        "confidence": round(top["score"] * 100, 1),
        "source":     "distilbert-onnx",
        "all_scores": {r["label"]: round(r["score"] * 100, 1) for r in results_sorted},
    }


# ── Spotify helpers ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3500)  # token expires in 3600s
def get_spotify_token() -> str | None:
    client_id     = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {creds}"},
        data={"grant_type": "client_credentials"},
        timeout=10,
    )
    if resp.status_code == 200:
        return resp.json()["access_token"]
    return None


def get_spotify_tracks(mood: str, limit: int = 5) -> list[dict]:
    token = get_spotify_token()
    if not token:
        return []

    features = MOOD_AUDIO_FEATURES.get(mood, MOOD_AUDIO_FEATURES["calm"])
    seeds    = MOOD_SEEDS.get(mood, {"seed_genres": "pop"})

    params = {
        "limit":            limit,
        "seed_genres":      seeds["seed_genres"],
        "target_valence":   features["valence"],
        "target_energy":    features["energy"],
        "target_tempo":     features["target_tempo"],
        "min_popularity":   40,
    }

    resp = requests.get(
        "https://api.spotify.com/v1/recommendations",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=10,
    )

    if resp.status_code != 200:
        return []

    tracks = []
    for t in resp.json().get("tracks", []):
        tracks.append({
            "title":       t["name"],
            "artist":      t["artists"][0]["name"],
            "album":       t["album"]["name"],
            "preview_url": t.get("preview_url"),
            "spotify_url": t["external_urls"]["spotify"],
            "image_url":   t["album"]["images"][-1]["url"] if t["album"]["images"] else None,
        })
    return tracks


# ── Main API handler ──────────────────────────────────────────────────────────
params = st.query_params

if "text" in params:
    # ── API mode: called by Next.js /api/mood ──────────────────────────────
    text = params["text"]

    if not text or not text.strip():
        result = {"error": "empty text"}
    else:
        # Run inference
        prediction = model_predict(text.strip())
        mood = prediction["mood"]

        # Fetch Spotify tracks
        tracks = get_spotify_tracks(mood, limit=5)

        result = {
            "mood":       mood,
            "confidence": prediction["confidence"],
            "source":     prediction.get("source", "model"),
            "all_scores": prediction.get("all_scores", {}),
            "tracks":     tracks,
        }

    # Render as JSON — Next.js will scrape this via html parsing
    # We wrap in a known div so it's easy to parse
    st.markdown(
        f'<div id="api-result">{json.dumps(result)}</div>',
        unsafe_allow_html=True
    )
    st.stop()

else:
    # ── Browser UI mode ────────────────────────────────────────────────────
    st.title("🎵 Mood → Music API")
    st.caption("This app powers the Music Mood tool on altheae.dev")

    st.markdown("### Try it")
    text_input = st.text_input("Describe how you're feeling", placeholder="I'm feeling really calm and peaceful today")

    if st.button("Analyze") and text_input.strip():
        with st.spinner("Running inference…"):
            prediction = model_predict(text_input.strip())
            mood = prediction["mood"]
            tracks = get_spotify_tracks(mood)

        st.success(f"**Mood:** {mood}  |  **Confidence:** {prediction['confidence']}%  |  **Source:** {prediction.get('source')}")

        if tracks:
            st.markdown("### 🎶 Recommended tracks")
            for t in tracks:
                col1, col2 = st.columns([1, 6])
                if t.get("image_url"):
                    col1.image(t["image_url"], width=48)
                col2.markdown(f"**{t['title']}** — {t['artist']}  \n_{t['album']}_  \n[Open in Spotify]({t['spotify_url']})")
        else:
            st.info("Spotify credentials not configured — add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to Streamlit secrets.")

        if prediction.get("all_scores"):
            st.markdown("### Score breakdown")
            st.bar_chart(prediction["all_scores"])

    st.divider()
    st.markdown("**API usage** (for Next.js):  `GET /?text=<url-encoded-text>`")
    st.code('fetch("https://your-app.streamlit.app/?text=" + encodeURIComponent(text))', language="js")

    model_status = "✅ ONNX model loaded" if pipe is not None else "⚠️ Rule-based fallback (upload mood_model_onnx/ to enable full model)"
    st.caption(model_status)
