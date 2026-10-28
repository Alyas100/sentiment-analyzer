from sentiment_analyzer import charts
from sentiment_analyzer.service import AnalysisService


def _result(sample_rows, stub_engine):
    return AnalysisService(engine=stub_engine).analyze_batch(
        sample_rows, with_aspects=True
    )


def test_pie_chart(sample_rows, stub_engine):
    fig = charts.pie_chart(_result(sample_rows, stub_engine))
    assert fig is not None
    assert len(fig.data) == 1


def test_trend_chart_has_all_series(sample_rows, stub_engine):
    fig = charts.trend_chart(_result(sample_rows, stub_engine))
    labels = {trace.name for trace in fig.data}
    assert labels == {"Positive", "Negative", "Neutral"}


def test_trend_chart_empty(sample_rows, stub_engine):
    result = AnalysisService(engine=stub_engine).analyze_batch(
        [{"platform": "Shopee", "review_text": "bagus"}], with_aspects=False
    )
    # no dates -> should still return a figure without crashing
    fig = charts.trend_chart(result)
    assert fig is not None


def test_aspect_chart(sample_rows, stub_engine):
    fig = charts.aspect_chart(_result(sample_rows, stub_engine))
    assert fig is not None
    assert len(fig.data) == 3
