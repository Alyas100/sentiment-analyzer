import pytest

from sentiment_analyzer.ingest import IngestionError, load_reviews


def test_load_reviews_ok():
    csv = b"review_id,date,platform,review_text\n1,2026-07-01,Shopee,bagus\n"
    rows = load_reviews(csv)
    assert len(rows) == 1
    assert rows[0]["platform"] == "Shopee"


def test_load_reviews_missing_column():
    try:
        load_reviews(b"id,text\n1,hello")
        raise AssertionError("expected IngestionError")
    except IngestionError as exc:
        assert "review_text" in str(exc)


def test_load_reviews_row_limit():
    header = "review_id,date,platform,review_text\n"
    body = "".join(f"{i},2026-07-01,Shopee,good\n" for i in range(5))
    with pytest.raises(IngestionError):
        load_reviews((header + body).encode(), max_rows=2)


def test_load_reviews_adds_default_platform():
    rows = load_reviews(b"review_text\nhello world")
    assert rows[0]["platform"] == "Unknown"
