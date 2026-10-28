from sentiment_analyzer import export
from sentiment_analyzer.service import AnalysisService


def _result(sample_rows, stub_engine):
    return AnalysisService(engine=stub_engine).analyze_batch(sample_rows, with_aspects=True)


def test_to_csv(sample_rows, stub_engine):
    data = export.to_csv(_result(sample_rows, stub_engine))
    assert isinstance(data, bytes)
    text = data.decode("utf-8")
    assert "Review" in text
    assert "Sentiment" in text


def test_to_json(sample_rows, stub_engine):
    data = export.to_json(_result(sample_rows, stub_engine))
    assert b"predictions" in data
    assert b"metrics" in data


def test_to_pdf(sample_rows, stub_engine):
    data = export.to_pdf(_result(sample_rows, stub_engine))
    assert data[:4] == b"%PDF"


def test_dataframe_columns(sample_rows, stub_engine):
    df = export.result_to_dataframe(_result(sample_rows, stub_engine))
    assert list(df.columns) == [
        "ID",
        "Date",
        "Platform",
        "Review",
        "Sentiment",
        "Score",
        "Language",
        "Aspects",
    ]
