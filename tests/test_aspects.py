from sentiment_analyzer.aspects import absa, extract_aspects
from sentiment_analyzer.schemas import Sentiment


def test_extract_aspects_detects_delivery_and_packaging():
    aspects = extract_aspects("Delivery lambat dan kotak kemik")
    assert "delivery" in aspects
    assert "packaging" in aspects


def test_absa_returns_aspect_sentiments(stub_engine):
    results = absa("harga murah tapi delivery lambat", lambda t: stub_engine.classify(t))
    names = {r.aspect for r in results}
    assert "price" in names
    assert "delivery" in names
    assert all(isinstance(r.sentiment, Sentiment) for r in results)


def test_extract_aspects_empty():
    assert extract_aspects("") == []
