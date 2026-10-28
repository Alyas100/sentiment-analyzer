from sentiment_analyzer.repository import Repository
from sentiment_analyzer.schemas import Sentiment
from sentiment_analyzer.service import AnalysisService


def _result(sample_rows, stub_engine):
    return AnalysisService(engine=stub_engine).analyze_batch(
        sample_rows, with_aspects=True
    )


def test_create_and_fetch_session(sample_rows, stub_engine):
    repo = Repository()
    result = _result(sample_rows, stub_engine)
    session_id = repo.create_session_from_result(result, source="test.csv")

    assert session_id > 0
    detail = repo.get_session(session_id)
    assert detail is not None
    assert detail.source == "test.csv"
    assert detail.total_reviews == len(sample_rows)
    assert len(detail.predictions) == len(sample_rows)
    assert detail.predictions[0].sentiment in Sentiment


def test_list_sessions(sample_rows, stub_engine):
    repo = Repository()
    repo.create_session_from_result(_result(sample_rows, stub_engine), source="a.csv")
    repo.create_session_from_result(_result(sample_rows, stub_engine), source="b.csv")
    sessions = repo.list_sessions()
    assert len(sessions) >= 2


def test_get_missing_session_returns_none():
    assert Repository().get_session(999999) is None
