FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# No build toolchain: every dependency ships prebuilt cp310 manylinux wheels,
# so build-essential would only add ~230 MB to the final image.
COPY requirements.txt requirements-ml.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install transformers sentencepiece

COPY . .

# The sentiment checkpoint lives in ./models (git-ignored, ~290 MB). It is
# copied into the image by the COPY above when present locally. When it is
# absent (e.g. a fresh CI checkout) the API starts and answers /health, while
# classification requests return 503 with an actionable error.
RUN mkdir -p /app/data /app/.cache/huggingface

EXPOSE 8000 7860

CMD ["uvicorn", "sentiment_analyzer.api:app", "--host", "0.0.0.0", "--port", "8000"]
