"""Smoke-test the configured Hugging Face checkpoint on the sample reviews.

Usage::

    python scripts/verify_model.py                 # uses HF_SENTIMENT_MODEL
    python scripts/verify_model.py <model-id>      # override the checkpoint

Prints the label, confidence score and detected language for every row in
``sample_reviews.csv`` so you can eyeball model quality on Malay/English data.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentiment_analyzer.config import get_settings  # noqa: E402
from sentiment_analyzer.ingest import load_reviews  # noqa: E402
from sentiment_analyzer.service import AnalysisService  # noqa: E402


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else get_settings().hf_model
    print(f"Model: {model_id}")

    service = AnalysisService()
    print(f"Active engine: {service.engine_name}\n")

    rows = load_reviews((ROOT / "sample_reviews.csv").read_bytes())
    for row in rows:
        pred = service.classify_single(
            str(row.get("review_text", "")),
            review_id=str(row.get("review_id", "")),
            platform=str(row.get("platform", "Unknown")),
            date_str=str(row.get("date", "")),
        )
        print(
            f"{pred.review_id:<6} {pred.sentiment.value:<9} "
            f"{pred.score:<7.3f} {pred.language:<10} "
            f"{str(row.get('review_text', ''))[:60]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
