# Product Requirements Document (PRD)
## E-Commerce Customer Sentiment Dashboard

| Field | Value |
| --- | --- |
| Status | Approved / In build |
| Owner | _(candidate)_ |
| Version | 1.0 |
| Last updated | 2026-10-18 |

---

## 1. Background & Problem

Social-commerce sellers (Shopee, TikTok Shop, Lazada) receive hundreds of reviews per
week written in **code-mixed Malay/English slang** ("giler", "mantap", "aduhai",
"tak recommend"). Sellers cannot read every review, and simple star averages hide
*why* customers are unhappy.

The previous version of this project was a thin Gradio wrapper around a single LLM
call. It had no persistence, no tests, no API, no export, and no handling of
code-mixed text. It demonstrated "how to call a model", not "how to build a
product".

This PRD defines an upgraded **end-to-end analytics platform** that is production
shaped and demonstrates software engineering, data-science, and DevOps capability.

---

## 2. Goals & Non-Goals

### Goals
1. Ship an **end-to-end** system: ingest → preprocess → analyze → visualize → persist → export → serve.
2. Handle **code-mixed Malay/English social-commerce slang** correctly.
3. Provide **Aspect-Based Sentiment Analysis (ABSA)** so users learn *why* sentiment drops.
4. Provide **historical sessions** persisted in a database.
5. Provide a **documented REST API** alongside the UI.
6. Be **containerized** and covered by **CI + tests**.

### Non-Goals
- User accounts / multi-tenancy / auth.
- Real-time streaming ingestion from marketplaces.
- Fine-tuning a model from scratch (we use an off-the-shelf multilingual model).

---

## 3. Target Users

| Persona | Need |
| --- | --- |
| Marketplace seller | Understand sentiment trends and top complaints quickly |
| Ops / CX analyst | Drill into a bad day, export a report |
| Recruiter / hiring manager | Evaluate engineering quality of the repo |
| Developer / integrator | Call the REST API from another system |

---

## 4. Functional Requirements

### FR-1 Data ingestion
- Upload a CSV with columns: `review_id`, `date`, `platform`, `review_text` (case/space tolerant).
- Filter by platform (`All Shops`, `Shopee`, `TikTok`, `Lazada`).
- Validate schema and row limits; return actionable errors.

### FR-2 Preprocessing (code-mixed / noisy text)
- Normalize whitespace, casing, URLs, emojis.
- Expand a curated **Malay/Manglish slang lexicon** (e.g. `giler → sangat`, `mantap → bagus`).
- Detect language (English / Malay / code-mixed) as metadata.

### FR-3 Sentiment classification
- A single **Hugging Face engine** classifies every review (`sentiment_analyzer/engine`).
- Output a label (`Positive` / `Negative` / `Neutral`) and a confidence score.
- Default checkpoint: `clapAI/modernBERT-base-multilingual-sentiment` (150M, Apache-2.0,
  16 languages incl. Malay + Indonesian). Label semantics are derived from the
  checkpoint's `id2label` config, so the model is swappable via `HF_SENTIMENT_MODEL`.
- Weights are vendored under `models/` (git-ignored) so the app is self-contained and
  runs offline on CPU-only machines; a Hub id also works via the shared HF cache.

### FR-4 Aspect-Based Sentiment Analysis
- Extract aspects (`delivery`, `packaging`, `price`, `product_quality`, `customer_service`, `size_fit`) from each review.
- Classify sentiment **per aspect**.
- Aggregate aspect sentiment for the whole batch.

### FR-5 Topic / keyword extraction
- Extract top keywords / complaint phrases from **negative** reviews (TF-IDF, frequency fallback).
- Surface them in the UI so users see *why* sentiment dropped.

### FR-6 Visualization
- Interactive sentiment distribution (pie) and trend-over-time (line).
- **Drill-down**: selecting a point on the trend filters the results table to that date.

### FR-7 Persistence
- Persist every analysis run as a **session** with its per-review predictions.
- List and reopen historical sessions.

### FR-8 Export
- Export results and metrics as **CSV**, **JSON**, and **PDF**.

### FR-9 REST API
- `POST /analyze` — single text.
- `POST /analyze/batch` — CSV upload.
- `GET /sessions`, `GET /sessions/{id}`.
- `GET /sessions/{id}/export?format=csv|json|pdf`.
- `GET /health`.
- Auto-generated OpenAPI/Swagger at `/docs`.

### FR-10 UI
- React dashboard hosted as a Static Space and calling the hosted model Space.

---

## 5. Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | Lazy model loading; app imports without downloading weights. |
| NFR-2 | Explicit failure: if the transformer cannot be loaded, the API returns 503 with an actionable error instead of guessing a sentiment. |
| NFR-3 | Deterministic tests that do **not** require network or model weights. |
| NFR-4 | Single-command startup via `docker compose up`. |
| NFR-5 | CI runs tests + lint on every push/PR. |
| NFR-6 | Config via environment variables only; no secrets in the repo. |

---

## 6. Architecture

```
                ┌─────────────────┐        ┌──────────────────┐
  browser ─────▶│  React UI       │        │  FastAPI REST API │◀──── clients
                │ Static Space    │        │  (api.py, /docs)  │
                └────────┬────────┘        └─────────┬────────┘
                         │                            │
                         ▼                            ▼
                ┌──────────────────┐        ┌──────────────────────┐
                │ Hosted model     │        │ Analysis service     │
                │ Space /predict   │        │ local API workflows  │
                └──────────────────┘        └──────────────────────┘
```

### Module map
| Path | Responsibility |
| --- | --- |
| `sentiment_analyzer/config.py` | Env-driven settings |
| `sentiment_analyzer/preprocessing.py` | Cleaning, slang, language detection |
| `sentiment_analyzer/engine/` | Hugging Face sentiment engine |
| `sentiment_analyzer/aspects.py` | ABSA extraction + aggregation |
| `sentiment_analyzer/topics.py` | Keyword/topic extraction |
| `sentiment_analyzer/charts.py` | Plotly figures |
| `sentiment_analyzer/export.py` | CSV / JSON / PDF writers |
| `sentiment_analyzer/db.py`, `repository.py` | Persistence |
| `sentiment_analyzer/service.py` | Orchestration (use-cases) |
| `sentiment_analyzer/api.py` | FastAPI app |
| `frontend/` | React Static Space dashboard |

---

## 7. Data Model

**sessions**
| column | type | notes |
| --- | --- | --- |
| id | int PK | |
| created_at | datetime | |
| source | text | filename / "api" |
| platform | text | filter used |
| engine | text | engine name |
| total_reviews | int | |
| positive / negative / neutral | int | counts |
| top_keywords | text (JSON) | |

**predictions**
| column | type | notes |
| --- | --- | --- |
| id | int PK | |
| session_id | FK → sessions.id | |
| review_id | text | |
| date | date | |
| platform | text | |
| review_text | text | |
| sentiment | text | |
| score | float | confidence |
| language | text | |
| aspects | text (JSON) | |

---

## 8. Success Metrics
- `pytest` green, coverage of preprocessing / engine / ABSA / repository / API.
- Cold start to dashboard < 10s without model download.
- A reviewer can go from `git clone` → running dashboard with **one command**
  (`make install && make install-ml && make model`).
- API documented and explorable at `/docs`.

---

## 9. Milestones
1. **M1** Package scaffolding, config, DB, repository.
2. **M2** NLP layer: preprocessing, HF engine, ABSA, topics.
3. **M3** Service, charts, export.
4. **M4** FastAPI + Gradio UI with drill-down.
5. **M5** Docker, compose, CI, tests, docs.

---

## 10. Risks & Mitigations
| Risk | Mitigation |
| --- | --- |
| Large model download slows first run | Lazy load + vendored weights + `make model` helper |
| Slang lexicon incomplete | Extensible dict; model handles unseen slang |
| API key leakage | `.env` git-ignored; `.env.example` only — no cloud API keys needed |
| Non-deterministic tests | Stub engine in tests + in-memory SQLite |
```
