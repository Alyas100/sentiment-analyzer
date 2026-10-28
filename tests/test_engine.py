import pytest

from sentiment_analyzer.engine import (
    EngineUnavailableError,
    HuggingFaceSentimentEngine,
    SpaceSentimentEngine,
    label_to_sentiment,
)
from sentiment_analyzer.schemas import Sentiment


def test_label_to_sentiment_named_labels():
    assert label_to_sentiment("positive") == Sentiment.POSITIVE
    assert label_to_sentiment("Negative") == Sentiment.NEGATIVE
    assert label_to_sentiment("neutral") == Sentiment.NEUTRAL
    assert label_to_sentiment("Very Positive") == Sentiment.POSITIVE
    assert label_to_sentiment("Very Negative") == Sentiment.NEGATIVE
    assert label_to_sentiment("mixed") is None


def test_label_to_sentiment_star_ratings():
    assert label_to_sentiment("1 star") == Sentiment.NEGATIVE
    assert label_to_sentiment("2 stars") == Sentiment.NEGATIVE
    assert label_to_sentiment("3 stars") == Sentiment.NEUTRAL
    assert label_to_sentiment("4 stars") == Sentiment.POSITIVE
    assert label_to_sentiment("5 stars") == Sentiment.POSITIVE


def test_hf_engine_uses_pipeline_label_and_score():
    eng = HuggingFaceSentimentEngine(model_name="fake/model")
    eng._pipeline = lambda text: [{"label": "positive", "score": 0.93}]
    res = eng.classify_detailed("barang bagus giler")
    assert res.sentiment == Sentiment.POSITIVE
    assert res.score == 0.93


def test_hf_engine_empty_text_is_neutral():
    eng = HuggingFaceSentimentEngine(model_name="fake/model")
    res = eng.classify_detailed("   ")
    assert res.sentiment == Sentiment.NEUTRAL
    assert res.score == 0.0


def test_hf_engine_resolves_labels_from_checkpoint_config():
    eng = HuggingFaceSentimentEngine(model_name="fake/model")
    eng._label_map = {
        "positive": Sentiment.POSITIVE,
        "negative": Sentiment.NEGATIVE,
        "neutral": Sentiment.NEUTRAL,
    }
    assert eng._resolve("positive") == Sentiment.POSITIVE
    assert eng._resolve("negative") == Sentiment.NEGATIVE
    assert eng._resolve("neutral") == Sentiment.NEUTRAL


def test_space_engine_parses_space_labels():
    assert SpaceSentimentEngine._parse("POSITIVE") == (Sentiment.POSITIVE, 0.95)
    assert SpaceSentimentEngine._parse("NEGATIVE (confidence unavailable)") == (
        Sentiment.NEGATIVE,
        0.95,
    )
    assert SpaceSentimentEngine._parse("not a known label") == (Sentiment.NEUTRAL, 0.0)


def test_factory_returns_the_space_engine_by_default():
    import sentiment_analyzer.engine as engine_module

    engine_module.get_engine.cache_clear()
    engine = engine_module.get_engine()
    assert isinstance(engine, SpaceSentimentEngine)
    assert engine.name == "space"


def test_factory_can_select_local_hugging_face_engine(monkeypatch):
    import sentiment_analyzer.engine as engine_module
    from sentiment_analyzer.config import get_settings

    monkeypatch.setenv("SENTIMENT_ENGINE", "hf")
    get_settings.cache_clear()
    engine_module.get_engine.cache_clear()
    engine = engine_module.get_engine()
    assert isinstance(engine, HuggingFaceSentimentEngine)
    assert engine.name == "hf"


def test_hf_engine_raises_when_model_cannot_load(monkeypatch):
    transformers = pytest.importorskip("transformers")

    def boom(*args, **kwargs):
        raise OSError("no network / weights unavailable")

    monkeypatch.setattr(transformers, "pipeline", boom)
    eng = HuggingFaceSentimentEngine(model_name="fake/model")
    with pytest.raises(EngineUnavailableError):
        eng.classify_detailed("seller mantap barang bagus")
    assert eng.load_error


def test_hf_engine_honours_explicit_device(monkeypatch):
    import sentiment_analyzer.engine as engine_module

    class _Settings:
        hf_device = "cpu"
        hf_model = "fake/model"

    monkeypatch.setattr(engine_module, "get_settings", lambda: _Settings())
    eng = HuggingFaceSentimentEngine(model_name="fake/model")
    assert eng._resolve_device() == "cpu"


def test_hf_engine_auto_device_is_supported_backend(monkeypatch):
    import sentiment_analyzer.engine as engine_module

    class _Settings:
        hf_device = "auto"
        hf_model = "fake/model"

    monkeypatch.setattr(engine_module, "get_settings", lambda: _Settings())
    eng = HuggingFaceSentimentEngine(model_name="fake/model")
    assert eng._resolve_device() in {"cpu", "cuda", "mps"}


def test_service_reports_active_engine_name(stub_engine):
    from sentiment_analyzer.service import AnalysisService

    svc = AnalysisService(engine=stub_engine)
    assert svc.engine_name == "hf"
