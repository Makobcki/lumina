# Lumina

Distributed search engine baseline for Wikipedia and IT documentation.

## Components
- `docs/architecture.md` — architecture baseline, latency considerations, schema, and rollout notes.
- `infra/qdrant/config.yaml` — Qdrant memory-oriented baseline with scalar quantization.
- `services/inference` — FastAPI inference service with SentenceTransformer-based embeddings and reranker scaffold.
- `services/gateway` — FastAPI gateway scaffold.
- `docker-compose.yml` — local development topology for Qdrant + inference + gateway.

## Quick start
```bash
docker compose up --build
```

Then check:
- Gateway: `http://localhost:8000/health`
- Inference: `http://localhost:8001/health`
- Qdrant: `http://localhost:6333/dashboard`

## Current status
This repository now contains the **distributed architecture scaffold**, but not yet:
- production crawler implementation,
- live Qdrant collection provisioning,
- React frontend.
