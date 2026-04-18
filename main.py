"""
Mood Classifier + Spotify Recommender — FastAPI
================================================
Deploy to Render (free tier, no Docker needed).

Environment variables to set in Render dashboard:
  SPOTIFY_CLIENT_ID     = "..."
  SPOTIFY_CLIENT_SECRET = "..."

Endpoints:
  POST /mood   { "text": "I feel calm" }
  GET  /health
"""

import base64
import os
import re
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Mood API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "mood_model_onnx"

def load_model():
    if not MODEL_DIR.exists():
        print("⚠️  mood_model_onnx/ not found — using rule-based fallback")
        return None
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    print("✅ ONNX model loaded")
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
    return {"mood": top_mood, "confidence": min(confidence, 82.0), "source": "rule-based", "all_scores": {}}

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

# ── Spotify ───────────────────────────────────────────────────────────────────
MOOD_AUDIO: dict[str, dict] = {
    "happy":       {"valence": 0.85, "energy": 0.80, "tempo": 128, "genres": "pop,happy,dance"},
    "sad":         {"valence": 0.15, "energy": 0.25, "tempo": 70,  "genres": "sad,acoustic,singer-songwriter"},
    "angry":       {"valence": 0.20, "energy": 0.90, "tempo": 155, "genres": "metal,rock,punk"},
    "calm":        {"valence": 0.65, "energy": 0.20, "tempo": 75,  "genres": "ambient,classical,sleep"},
    "anxious":     {"valence": 0.25, "energy": 0.75, "tempo": 145, "genres": "alternative,indie,post-rock"},
    "focused":     {"valence": 0.50, "energy": 0.50, "tempo": 105, "genres": "study,classical,instrumental"},
    "loved":       {"valence": 0.80, "energy": 0.45, "tempo": 95,  "genres": "romance,soul,r-n-b"},
    "melancholic": {"valence": 0.20, "energy": 0.25, "tempo": 72,  "genres": "indie,folk,sad"},
    "confident":   {"valence": 0.75, "energy": 0.82, "tempo": 122, "genres": "hip-hop,power-pop,work-out"},
}

_spotify_token: str | None = None
_spotify_token_expiry: float = 0

def get_spotify_token() -> str | None:
    global _spotify_token, _spotify_token_expiry
    client_id     = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None
    if _spotify_token and time.time() < _spotify_token_expiry:
        return _spotify_token
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {creds}"},
        data={"grant_type": "client_credentials"},
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    _spotify_token = data["access_token"]
    _spotify_token_expiry = time.time() + data["expires_in"] - 60
    return _spotify_token

def get_spotify_tracks(mood: str, limit: int = 5) -> list[dict]:
    token = get_spotify_token()
    if not token:
        return []
    audio = MOOD_AUDIO.get(mood, MOOD_AUDIO["calm"])
    params = {
        "limit":          limit,
        "seed_genres":    audio["genres"],
        "target_valence": audio["valence"],
        "target_energy":  audio["energy"],
        "target_tempo":   audio["tempo"],
        "min_popularity": 40,
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

# ── Routes ────────────────────────────────────────────────────────────────────
class MoodRequest(BaseModel):
    text: str

@app.post("/mood")
def predict_mood(req: MoodRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    prediction = model_predict(text)
    tracks = get_spotify_tracks(prediction["mood"])
    return {**prediction, "tracks": tracks}

@app.get("/health")
def health():
    return {"status": "ok", "model": "onnx" if pipe is not None else "rule-based"}
