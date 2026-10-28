"""Shared pytest fixtures.

Tests are fully deterministic and offline: the Hugging Face engine is swapped
for a lightweight stub and an isolated SQLite database is used.
"""

from __future__ import annotations

import pytest

from sentiment_analyzer.engine import SentimentEngine, SentimentResult
from sentiment_analyzer.schemas import Sentiment


class StubSentimentEngine(SentimentEngine):
    """Offline test double for the Hugging Face engine.

    Mirrors the real engine's contract (label + confidence) so the suite runs
    fast and deterministically without downloading any model weights.
    """

    name = "hf"

    POSITIVE = {
        "mantap",
        "mantul",
        "bagus",
        "elok",
        "baik",
        "puas",
        "suka",
        "best",
        "good",
        "great",
        "nice",
        "cepat",
        "murah",
        "sangat",
        "recommend",
    }
    NEGATIVE = {
        "teruk",
        "buruk",
        "rosak",
        "lambat",
        "benci",
        "bad",
        "worst",
        "slow",
        "kemik",
        "mahal",
        "tak recommend",
        "tidak disyorkan",
    }
    NEUTRAL = {"biasa", "boleh", "okay", "average", "so so"}

    def classify(self, text: str) -> Sentiment:
        if not text or not text.strip():
            return Sentiment.NEUTRAL
        lower = text.lower()
        for words, sentiment in (
            (self.NEGATIVE, Sentiment.NEGATIVE),
            (self.POSITIVE, Sentiment.POSITIVE),
            (self.NEUTRAL, Sentiment.NEUTRAL),
        ):
            if any(word in lower for word in words):
                return sentiment
        return Sentiment.NEUTRAL

    def classify_detailed(self, text: str) -> SentimentResult:
        sentiment = self.classify(text)
        score = 0.9 if sentiment != Sentiment.NEUTRAL else 0.5
        return SentimentResult(sentiment=sentiment, score=score)


@pytest.fixture
def stub_engine() -> StubSentimentEngine:
    return StubSentimentEngine()


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    db_path = tmp_path / "test_sessions.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("MAX_BATCH_ROWS", "500")

    from sentiment_analyzer import db, service
    from sentiment_analyzer.config import get_settings

    get_settings.cache_clear()
    db.reset_engine()

    # Never load real model weights in the test suite.
    monkeypatch.setattr(service, "get_engine", lambda *a, **k: StubSentimentEngine())

    db.init_db()

    yield

    get_settings.cache_clear()
    db.reset_engine()


SAMPLE_ROWS = [
    {"review_id": "1", "date": "2026-07-01", "platform": "Shopee", "review_text": "Barang sampai cepat giler. Seller mantap!"},
    {"review_id": "2", "date": "2026-07-03", "platform": "TikTok", "review_text": "Lambat gila delivery kotak pun kemik. Tak recommend."},
    {"review_id": "3", "date": "2026-07-05", "platform": "Lazada", "review_text": "Boleh la kualiti padan dengan harga. Biasa biasa je."},
    {"review_id": "4", "date": "2026-07-08", "platform": "Shopee", "review_text": "Tq seller packaging kemas. Nanti repeat order lagi."},
    {"review_id": "5", "date": "2026-07-12", "platform": "TikTok", "review_text": "x sedap langsung membazir duit je beli ni"},
]


@pytest.fixture
def sample_rows():
    return [dict(r) for r in SAMPLE_ROWS]
