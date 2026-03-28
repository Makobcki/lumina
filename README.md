# Lumina

Distributed search engine baseline for Wikipedia and IT documentation.

## Components
- `docs/architecture.md` — architecture baseline, latency considerations, schema, and rollout notes.
- `infra/qdrant/config.yaml` — Qdrant memory-oriented baseline with scalar quantization.
- `services/inference` — FastAPI inference service with SentenceTransformer-based embeddings and reranker scaffold.
- `services/gateway` — FastAPI gateway scaffold.
- `docker-compose.yml` — local development topology for Qdrant + inference + gateway.
- `services/crawler` — CLI crawler for fetching, cleaning, chunking, and indexing web pages through the gateway.
- `frontend` — React + Vite + Tailwind SPA for search queries and result rendering.

## Hybrid retrieval (Dense + Sparse)
- Gateway now indexes **two vector spaces** in Qdrant collection `documents`:
  - `dense` (semantic embedding from `/embed`)
  - `sparse` (BM25-hash sparse embedding from `/embed/sparse`)
- `/search` executes Qdrant `query_points` with `prefetch` for both spaces and `Fusion.RRF`.
- This improves ranking for exact technical terms (e.g., `FastAPI CORSMiddleware`) while preserving semantic recall.

## Quick start
```bash
docker compose up --build
```

Then check:
- Gateway: `http://localhost:8000/health`
- Inference: `http://localhost:8001/health`
- Qdrant: `http://localhost:6333/dashboard`
- Indexing: `POST http://localhost:8000/index`
- Search: `POST http://localhost:8000/search`

## Crawler
Install crawler dependencies and run locally:
```bash
cd services/crawler
pip install -r requirements.txt
python main.py
```

The crawler fetches a small set of seed URLs, removes noisy HTML, chunks text, and pushes each chunk to `POST /index` on the gateway.

## Frontend
Run the SPA locally after installing dependencies:
```bash
cd frontend
npm install
npm run dev
```

The frontend talks to the gateway search API via `VITE_API_URL` (for example, an external IP in production).

Examples:
```bash
# local default
npm run dev

# custom gateway URL
VITE_API_URL=http://203.0.113.10:8000 npm run build
```

If `VITE_API_URL` is not set, the app falls back to `http://localhost:8000`.
Search results support incremental loading via a **Load more** button (10 results per page).

## Current status
This repository now contains the **distributed architecture scaffold**, but not yet:
- production-grade frontier management / politeness controls for crawling,
- dump-based Wikipedia ingestion,
- production deployment hardening for the frontend.
