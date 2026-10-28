"""Local CPU inference API for the React dashboard.

This module is intentionally separate from ``hf_space/app.py``. The latter is
the GPU-backed Gradio application deployed to Hugging Face.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("HF_SENTIMENT_MODEL", str(ROOT / "models" / "modernbert-sentiment"))
MAX_TEXT_LENGTH = 2_000

app = FastAPI(title="Sentiment Analyzer Local CPU API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    texts: list[str]


def _sentiment_label(raw_label: str) -> str:
    label = raw_label.lower()
    if "positive" in label:
        return "Positive"
    if "negative" in label:
        return "Negative"
    return "Neutral"


@lru_cache(maxsize=1)
def get_classifier() -> Any:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(
            f"Local model was not found at {model_path}. "
            "Set HF_SENTIMENT_MODEL to the checkpoint directory."
        )
    return pipeline(
        "sentiment-analysis",
        model=str(model_path),
        tokenizer=str(model_path),
        device=-1,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "local-cpu", "model": MODEL_PATH}


@app.post("/classify")
def classify(request: ClassifyRequest) -> dict[str, list[dict[str, Any]]]:
    texts = [text.strip() for text in request.texts]
    if not texts or any(not text for text in texts):
        raise HTTPException(status_code=400, detail="texts must contain non-empty strings")
    if any(len(text) > MAX_TEXT_LENGTH for text in texts):
        raise HTTPException(status_code=400, detail=f"Each text must be at most {MAX_TEXT_LENGTH} characters")
    try:
        predictions = get_classifier()(texts, truncation=True)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Local model could not be loaded: {exc}") from exc
    return {
        "predictions": [
            {"sentiment": _sentiment_label(item["label"]), "score": float(item["score"])}
            for item in predictions
        ]
    }
