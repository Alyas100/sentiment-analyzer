from sentiment_analyzer.schemas import Sentiment
from sentiment_analyzer.service import AnalysisService


def test_analyze_batch_counts(sample_rows, stub_engine):
    svc = AnalysisService(engine=stub_engine)
    result = svc.analyze_batch(sample_rows, platform_filter="All Shops", with_aspects=True)
    assert result.metrics.total == len(sample_rows)
    assert result.metrics.counts.positive + result.metrics.counts.negative + result.metrics.counts.neutral == len(sample_rows)
    assert 0.0 <= result.metrics.negative_ratio <= 1.0
    assert result.engine == "hf"


def test_analyze_batch_platform_filter(sample_rows, stub_engine):
    svc = AnalysisService(engine=stub_engine)
    result = svc.analyze_batch(sample_rows, platform_filter="Shopee")
    assert all(p.platform == "Shopee" for p in result.predictions)
    assert result.metrics.total == 2


def test_classify_single_sets_language_and_date(stub_engine):
    svc = AnalysisService(engine=stub_engine)
    pred = svc.classify_single("barang bagus, sampai cepat", date_str="2026-07-01")
    assert pred.sentiment == Sentiment.POSITIVE
    assert pred.date is not None
    assert pred.language in {"en", "ms", "ms-en", "unknown"}
