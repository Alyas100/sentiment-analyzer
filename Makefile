.PHONY: help install install-ml model api local-api test lint docker-up docker-down clean

help:
	@echo "Targets:"
	@echo "  make install      Install core + dev dependencies"
	@echo "  make install-ml   Install Hugging Face transformer dependencies (torch/transformers)"
	@echo "  make model        Download the sentiment model into ./models (once, ~290 MB)"
	@echo "  make api          Launch the FastAPI server (http://localhost:8000/docs)"
	@echo "  make local-api    Launch the local CPU inference API (http://localhost:8000/docs)"
	@echo "  make test         Run the test suite with coverage"
	@echo "  make lint         Run ruff"
	@echo "  make docker-up    Build and start API + UI via docker compose"
	@echo "  make docker-down  Stop docker compose services"

install:
	pip install -r requirements-local.txt

install-ml:
	pip install -r requirements-local.txt

model:
	hf download clapAI/modernBERT-base-multilingual-sentiment --local-dir models/modernbert-sentiment

api:
	uvicorn sentiment_analyzer.api:app --reload --host 0.0.0.0 --port 8000

local-api:
	HF_SENTIMENT_MODEL=$${HF_SENTIMENT_MODEL:-models/modernbert-sentiment} uvicorn local_api:app --reload --host 127.0.0.1 --port 8000

test:
	pytest --cov=sentiment_analyzer --cov-report=term-missing

lint:
	ruff check .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
