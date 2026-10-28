from sentiment_analyzer.schemas import Prediction, Sentiment
from sentiment_analyzer.topics import extract_keywords_freq, extract_topics_negative


def test_extract_keywords_freq_filters_stopwords():
    kws = extract_keywords_freq(["delivery lambat delivery", "packaging kemik"])
    assert "delivery" in kws
    assert "yang" not in kws


def test_extract_topics_negative_only_negative():
    preds = [
        Prediction(sentiment=Sentiment.NEGATIVE, review_text="delivery lambat gila"),
        Prediction(sentiment=Sentiment.POSITIVE, review_text="packaging kemas mantap"),
    ]
    topics = extract_topics_negative(preds)
    assert any("lambat" in t or "delivery" in t for t in topics)
    assert all("mantap" not in t for t in topics)


def test_extract_topics_multiple_negatives_hits_tfidf():
    preds = [
        Prediction(sentiment=Sentiment.NEGATIVE, review_text="delivery lambat kotak kemik"),
        Prediction(sentiment=Sentiment.NEGATIVE, review_text="customer service teruk chat tak reply"),
        Prediction(sentiment=Sentiment.NEGATIVE, review_text="delivery lambat lagi sekali"),
    ]
    topics = extract_topics_negative(preds)
    assert len(topics) > 0


def test_extract_keywords_tfidf_direct():
    from sentiment_analyzer.topics import extract_keywords_tfidf

    kws = extract_keywords_tfidf(
        ["harga mahal tapi bagus", "delivery lambat gila", "packaging kemik rosak"], top_n=5
    )
    assert isinstance(kws, list)
    assert len(kws) <= 5
