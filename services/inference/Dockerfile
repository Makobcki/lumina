# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ARG LUMINA_EMBED_MODEL=BAAI/bge-m3
ARG LUMINA_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

WORKDIR /app

ENV HF_HOME=/models/huggingface \
    LUMINA_EMBED_MODEL=${LUMINA_EMBED_MODEL} \
    LUMINA_RERANKER_MODEL=${LUMINA_RERANKER_MODEL}

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir uv==0.8.13 \
    && uv pip install --system -r requirements.txt

RUN python -c "import os; from sentence_transformers import CrossEncoder, SentenceTransformer; SentenceTransformer(os.environ['LUMINA_EMBED_MODEL'], device='cpu'); CrossEncoder(os.environ['LUMINA_RERANKER_MODEL'], device='cpu')"

COPY app.py ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
