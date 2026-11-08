# 🛒 E-Commerce Customer Sentiment Dashboard

An end-to-end NLP analytics platform that turns noisy, **code-mixed Malay/English**
social-commerce reviews (Shopee, TikTok Shop, Lazada) into actionable insight:
sentiment trends, aspect-level breakdowns, complaint keywords, persisted sessions
and exportable reports — served through a React dashboard and a documented
FastAPI REST API.

[![CI](https://github.com/your-username/sentiment-analyzer/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

| Area | What it does |
| --- | --- |
| 🧠 **Hugging Face sentiment model** | Multilingual ModernBERT transformer (150M, Apache-2.0) — label + confidence straight from the checkpoint, no cloud API |
| 🌏 **Code-mixed handling** | Preprocessing + curated Malay/Manglish slang lexicon (`giler → sangat`, `mantap → bagus`, …) and language detection |
| 🔬 **Aspect-Based Sentiment (ABSA)** | Per-aspect sentiment for `delivery`, `packaging`, `price`, `product_quality`, `customer_service`, `size_fit` |
| 🔑 **Complaint mining** | TF-IDF keyword extraction over negative reviews to explain *why* sentiment drops |
| 📊 **Interactive dashboard** | Plotly pie/trend/aspect charts with date drill-down and session history
| 📤 **Exports** | Download results as **CSV**, **JSON** or a **PDF** report |
| 🗄️ **Persistence** | Every run is stored as a session (SQLAlchemy + SQLite) and can be reopened |
| 🔌 **REST API** | Documented with OpenAPI/Swagger at `/docs` |
| 🐳 **Containerized** | One command to run API + UI with `docker compose up` |
| ✅ **Tested & linted** | `pytest` suite (offline, deterministic) and `ruff` enforced in CI |

---

## 🏗️ Architecture

```
                    ┌─────────────────┐        ┌────────────────────┐
   CSV upload ─────▶│  React UI       │        │  FastAPI REST API  │◀── integrators
                    │  (:7860)        │        │  (:8000/docs)      │
                    └────────┬────────┘        └─────────┬──────────┘
                             │        shared service layer │
                    ┌────────▼───────────────────────────▼──────────┐
                    │            AnalysisService                     │
                    │  ingest → preprocess → classify → ABSA →       │
                    │  topics → metrics → persist → visualise        │
                    └───┬───────────┬───────────┬───────────┬────────┘
                        │           │           │           │
                 ┌──────▼─────┐ ┌───▼────┐ ┌────▼────┐ ┌────▼─────┐
                 │ HF model   │ │ SQLite │ │ Plotly  │ │ Export   │
                 │ ModernBERT │ │ models │ │ charts  │ │ CSV/JSON │
                 │            │ │        │ │         │ │ /PDF     │
                 └────────────┘ └────────┘ └─────────┘ └──────────┘
```

**Tech stack:** Python 3.10 · FastAPI · React · Hugging Face Transformers
(ModernBERT) · scikit-learn · SQLAlchemy · Plotly · ReportLab · Docker · GitHub Actions.

---

## 🤖 Model selection and app separation

The repository contains two intentionally separate applications:

1. **Local dashboard** — `frontend/` + `local_api.py`. This is the version
   recruiters and contributors can run locally. It loads ModernBERT from
   `models/modernbert-sentiment` and performs CPU-only inference without
   contacting Hugging Face.
2. **Hosted Space** — `hf_space/app.py`. This is the GPU-backed Gradio app
   deployed to Hugging Face and loads the Gemma + LoRA model.

The hosted Space is
**`alyaspronoob/malay-sentiment-demo`**. It calls the Space's `/predict` Gradio
endpoint, which serves the Gemma-SEA-LION + LoRA model fine-tuned for
Malay/Manglish sentiment. The main app does not download or load those large
weights locally.

The original local checkpoint remains available as an alternative engine:
**`clapAI/modernBERT-base-multilingual-sentiment`** (150M, Apache-2.0).
Label semantics are read from the checkpoint's `id2label` config, so any 3-class
sentiment model can be swapped in via `HF_SENTIMENT_MODEL` without code changes.

Macro-F1 on public multilingual benchmarks (product reviews / tweets / MTEB 29-language):

| Model | Licence | Product reviews | Tweets | MTEB | Malay (`ms`) |
| --- | --- | --- | --- | --- | --- |
| **clapAI/modernBERT-base-multilingual-sentiment** (150M) | Apache-2.0 | **0.775** | **0.784** | 0.653 | ✅ |
| Horizon-Labs/multilingual-sentiment-small (141M) | Apache-2.0 | 0.695 | 0.635 | **0.822** | ✅ |
| Horizon-Labs/multilingual-sentiment-base (308M) | Apache-2.0 | 0.703 | 0.660 | 0.857 | ✅ |
| cardiffnlp/twitter-xlm-roberta-base-sentiment (278M) | _not listed_ | 0.533 | 0.700 | 0.772 | ✅ |
| nlptown/bert-base-multilingual-uncased-sentiment (167M) | MIT | 0.645 | 0.390 | 0.683 | ❌ |

Why the default: highest product-review accuracy (the closest match to the
Shopee/TikTok review domain), explicit Apache-2.0 licence, 16 training languages
including Malay and Indonesian, and ModernBERT's 2–8× faster inference over
BERT/XLM-R, which matters when classifying a full CSV batch.
`Horizon-Labs/multilingual-sentiment-small` is the best fallback — same licence,
40 languages, and it never saw the evaluation train splits.

### Hardware

The default `space` engine needs no local ML model or GPU. Each review is sent
to the hosted Space for inference. An `HF_TOKEN` read token can provide a higher
ZeroGPU quota.

The optional `hf` engine runs on a **CPU-only laptop** — the local checkpoint is 150M params (290 MB on
disk, ~1 GB RAM) and ModernBERT is 2–8× faster than BERT/XLM-R on CPU. No GPU
required.

The weights live **inside the project** (git-ignored) so the app is
self-contained and works offline:

```bash
make model     # hf download ... --local-dir models/modernbert-sentiment  (~290 MB, once)
```

Set `SENTIMENT_ENGINE=hf` to use the local engine. `HF_SENTIMENT_MODEL=models/modernbert-sentiment` points at that folder; a Hub
id (`clapAI/modernBERT-base-multilingual-sentiment`) also works and is resolved
through the shared HF cache. Relative paths are resolved against the project
root, so the app loads the model from any working directory.

Device selection is automatic (`cuda` → `mps` on Apple Silicon → `cpu`); force it
with `HF_DEVICE=cpu|mps|cuda`. If the checkpoint cannot be loaded (offline, no
weights, OOM) the API answers **503** with the underlying error instead of
guessing a sentiment — `GET /engines` reports the active model and any error.

Sanity-check the checkpoint against the sample reviews:

```bash
python scripts/verify_model.py
# 101  Positive  0.997  ms     Barang sampai cepat giler. Seller mantap!
# 110  Negative  0.997  ms-en  Customer service teruk chat tak reply langsung! Nak refund.
```

---

## 🚀 Quickstart

### Local CPU dashboard

```bash
python -m venv .venv && source .venv/bin/activate
make install-ml       # local FastAPI + CPU PyTorch + Transformers
make model            # only if models/modernbert-sentiment is missing
make local-api        # local CPU API → http://localhost:8000/docs
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev           # local dashboard → http://localhost:5173
```

The local dashboard calls `http://localhost:8000/classify`. It does not use
Gradio, `@gradio/client`, or the Hugging Face Space.
To test it, upload
[`samples-to-test/sample_reviews.csv`](samples-to-test/sample_reviews.csv)
from the `samples-to-test` folder. This file contains 10 sample reviews.

### Option 2: Hosted Gemma + LoRA application

Open the hosted Hugging Face Space:

<https://huggingface.co/spaces/alyaspronoob/malay-sentiment-demo>

No local installation or GPU is required. Enter a Malay/Manglish review or
upload a CSV file in the Space, then run the analysis there. To test the
application, first upload
[`samples-to-test/sample_reviews-reduced-to-one.csv`](samples-to-test/sample_reviews-reduced-to-one.csv),
which contains one review. If that works, try
[`samples-to-test/sample_reviews.csv`](samples-to-test/sample_reviews.csv),
which contains 10 reviews.
Starting with one review is recommended because the hosted Hugging Face GPU
runtime has a limited execution time.

### Docker

```bash
docker compose up --build
# API  → http://localhost:8000/docs
# UI   → http://localhost:7860
```

---

## 🔌 REST API

| Method | Path | Description |
| --- | --- | --- |
| `GET`  | `/health` | Service + active engine health |
| `GET`  | `/engines` | Configured vs active engine |
| `POST` | `/analyze` | Classify a single review (optional ABSA) |
| `POST` | `/classify` | Classify a list of raw texts |
| `POST` | `/analyze/batch` | Upload a CSV and analyze a whole batch |
| `GET`  | `/sessions` | List persisted analysis sessions |
| `GET`  | `/sessions/{id}` | Fetch a session with all predictions |
| `GET`  | `/sessions/{id}/export?format=csv\|json\|pdf` | Export a session |

Example:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Barang sampai cepat giler, seller mantap!", "with_aspects": true}'
```

---

## ⚙️ Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `SENTIMENT_ENGINE` | `space` | `space` (hosted `HF_SPACE_ID`) or `hf` (local checkpoint) |
| `HF_SPACE_ID` | `alyaspronoob/malay-sentiment-demo` | Hugging Face Space used by the default engine |
| `HF_TOKEN` | — | Optional Hugging Face read token for Space access/quota |
| `HF_SENTIMENT_MODEL` | `clapAI/modernBERT-base-multilingual-sentiment` | HF model id or local path (any 3-class sentiment checkpoint works) |
| `HF_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` — `auto` picks cuda → mps (Apple Silicon) → cpu |
| `LOG_LEVEL` | `INFO` | `DEBUG` logs every per-review model prediction |
| `DATABASE_URL` | `sqlite:///data/sessions.db` | SQLAlchemy database URL |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | API bind address |
| `MAX_BATCH_ROWS` | `500` | Max rows accepted per CSV upload |

---

## 📁 Project structure

```
sentiment-analyzer/
├── frontend/                    # React local CPU dashboard
├── local_api.py                 # Local ModernBERT CPU inference API
├── sentiment_analyzer/
│   ├── api.py                   # FastAPI app + OpenAPI docs
│   ├── service.py               # Analysis orchestration (use-cases)
│   ├── engine/                  # Hugging Face sentiment engine (ModernBERT)
│   ├── preprocessing.py         # Cleaning + Malay/English slang normalization
│   ├── aspects.py               # Aspect-Based Sentiment Analysis
│   ├── topics.py                # Negative-review keyword extraction
│   ├── charts.py                # Interactive Plotly figures
│   ├── export.py                # CSV / JSON / PDF writers
│   ├── ingest.py                # CSV parsing & validation
│   ├── db.py / repository.py    # Persistence (SQLAlchemy + SQLite)
│   ├── schemas.py               # Pydantic models
│   └── config.py                # Environment-driven settings
├── tests/                       # pytest suite (offline, deterministic)
├── docs/PRD.md                  # Product Requirements Document
├── hf_space/app.py              # Hosted-only Gemma + LoRA Gradio app
├── Dockerfile / docker-compose.yml
├── .github/workflows/ci.yml     # Lint + test + docker build
├── requirements*.txt / pyproject.toml
└── sample_reviews.csv
```

---

## 🧪 Testing & quality

```bash
make lint   # ruff
make test   # pytest + coverage
```

The suite replaces the transformer with a deterministic stub engine and uses an
isolated SQLite database, so it runs fast, offline and deterministically in CI.

---

## 📝 Resume highlights

> **E-Commerce Customer Sentiment Dashboard** — *Python, FastAPI, Hugging Face
> Transformers, Gradio, scikit-learn, SQLAlchemy, Docker, GitHub Actions*
>
> - Built an end-to-end NLP analytics platform that processes code-mixed
>   Malay/English e-commerce reviews and visualizes sentiment trends over time.
> - Engineered a multilingual ModernBERT sentiment pipeline (CPU-friendly
>   inference, checkpoint-driven labels, actionable 503s when weights are missing).
> - Implemented aspect-based sentiment analysis, TF-IDF complaint extraction and
>   interactive Plotly visualisations with drill-down filtering.
> - Architected a tested (`pytest`, 88% coverage), linted (`ruff`) codebase with a
>   documented FastAPI backend, SQLite persistence, Docker Compose deployment and
>   a GitHub Actions CI pipeline.

---

## 📄 License

MIT.
