
from fastapi.testclient import TestClient

from sentiment_analyzer.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_analyze_single():
    resp = client.post("/analyze", json={"text": "barang mantap, seller bagus", "with_aspects": True})
    assert resp.status_code == 200
    pred = resp.json()["prediction"]
    assert pred["sentiment"] == "Positive"


def test_classify_batch():
    resp = client.post("/classify", json={"texts": ["bagus", "teruk"]})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_analyze_batch_upload_and_export(sample_rows):

    import pandas as pd

    csv_bytes = pd.DataFrame(sample_rows).to_csv(index=False).encode("utf-8")
    files = {"file": ("reviews.csv", csv_bytes, "text/csv")}
    data = {"platform": "All Shops", "with_aspects": "true", "persist": "true"}
    resp = client.post("/analyze/batch", files=files, data=data)
    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"]["total"] == len(sample_rows)
    session_id = body["session_id"]
    assert session_id

    sessions = client.get("/sessions").json()
    assert any(s["id"] == session_id for s in sessions)

    detail = client.get(f"/sessions/{session_id}").json()
    assert len(detail["predictions"]) == len(sample_rows)

    csv_export = client.get(f"/sessions/{session_id}/export?format=csv")
    assert csv_export.status_code == 200
    assert "text/csv" in csv_export.headers["content-type"]

    json_export = client.get(f"/sessions/{session_id}/export?format=json")
    assert json_export.status_code == 200

    pdf_export = client.get(f"/sessions/{session_id}/export?format=pdf")
    assert pdf_export.status_code == 200
    assert pdf_export.content[:4] == b"%PDF"


def test_analyze_batch_bad_csv():
    files = {"file": ("bad.csv", b"foo,bar\n1,2", "text/csv")}
    resp = client.post("/analyze/batch", files=files, data={"platform": "All Shops"})
    assert resp.status_code == 400


def test_get_missing_session_404():
    assert client.get("/sessions/999999").status_code == 404
