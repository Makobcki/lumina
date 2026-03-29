# Lumina

Distributed search engine baseline for Wikipedia and IT documentation.

## Components
- `docs/architecture.md` — architecture baseline, latency considerations, schema, and rollout notes.
- `infra/qdrant/config.yaml` — Qdrant memory-oriented baseline with scalar quantization.
- `services/inference` — FastAPI inference service with SentenceTransformer-based embeddings and reranker scaffold.
- `services/gateway` — FastAPI gateway scaffold.
- `docker-compose.yml` — local development topology for Qdrant + PostgreSQL (raw documents) + inference + gateway.
- `services/crawler` — CLI crawler for fetching, cleaning, chunking, and indexing web pages through the gateway.
- `frontend` — React + Vite + Tailwind SPA for search queries and result rendering.

## Hybrid retrieval (Dense + Sparse)
- Gateway now indexes **two vector spaces** in Qdrant collection `documents`:
  - `dense` (semantic embedding from `/embed`)
  - `sparse` (BM25-hash sparse embedding from `/embed/sparse`)
- Raw document `content` is stored in PostgreSQL (`raw_documents`); Qdrant payload keeps only hot metadata (`title`, `url`, `document_id`, `source`).
- `/search` executes Qdrant `query_points` with `prefetch` for both spaces and `Fusion.RRF`.
- After vector retrieval, gateway performs a bulk lookup by `document_id` in PostgreSQL and sends only retrieved candidate texts to `/rerank`.
- This improves ranking for exact technical terms (e.g., `FastAPI CORSMiddleware`) while preserving semantic recall.

## Quick start
```bash
cp .env .env.local  # optional backup for local overrides
docker compose up --build
```

> Compose now uses service-specific `Dockerfile` builds and reads runtime URLs from `.env`.

Then check:
- Gateway: `http://localhost:8000/health`
- Inference: `http://localhost:8001/health`
- Qdrant: `http://localhost:6333/dashboard`
- Indexing: `POST http://localhost:8000/index`
- Search: `POST http://localhost:8000/search`
- Ask (RAG): `POST http://localhost:8000/ask`

## Crawler
Install crawler dependencies and run locally:
```bash
cd services/crawler
pip install -r requirements.txt
python main.py
```

The crawler fetches a small set of seed URLs, removes noisy HTML, chunks text, and sends chunk batches to `POST /index/bulk` on the gateway (`LUMINA_GATEWAY_BULK_INDEX_URL`). Requests are retried with exponential backoff via `tenacity`.

Queue-based ingestion is available via Redis Streams:
```bash
# producer mode (crawler -> redis stream)
LUMINA_INGESTION_MODE=redis_stream \
LUMINA_REDIS_URL=redis://localhost:6379/0 \
python main.py

# consumer mode (redis stream -> gateway /index/bulk)
LUMINA_REDIS_URL=redis://localhost:6379/0 \
python redis_stream_worker.py
```

## Wikipedia dump parser

For large Wikipedia dumps, use the streaming parser without loading the full file into memory:

```bash
cd services/crawler
pip install -r requirements.txt

# stream parse XML dump and directly bulk-index to gateway
python wikipedia_dump_parser.py \
  --input /path/to/ruwiki-latest-pages-articles.xml.bz2 \
  --format xml \
  --language ru \
  --post-to-gateway \
  --gateway-bulk-url http://localhost:8000/index/bulk \
  --bulk-size 64

# stream parse Parquet dump and write JSONL
python wikipedia_dump_parser.py \
  --input /path/to/wiki.parquet \
  --format parquet \
  --language en \
  --output-jsonl ./wikipedia.jsonl
```

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
Frontend now supports **Search / Ask** mode switch. In **Ask**, the gateway streams an LLM answer and shows source snippets used for context.

## LLM settings for `/ask`
Configure gateway RAG generation in `.env`:

```bash
LUMINA_LLM_MODEL=gpt-4o-mini
LUMINA_LLM_API_KEY=...
LUMINA_LLM_BASE_URL=... # optional (for Ollama/OpenAI-compatible endpoints)
LUMINA_LLM_MAX_TOKENS=512
LUMINA_ASK_TOP_K=5
```

Examples:

```bash
# OpenAI-compatible
LUMINA_LLM_MODEL=gpt-4o-mini
LUMINA_LLM_API_KEY=sk-...

# Ollama (OpenAI-compatible endpoint)
LUMINA_LLM_MODEL=ollama/llama3.1
LUMINA_LLM_BASE_URL=http://localhost:11434
LUMINA_LLM_API_KEY=dummy
```

## Current status
This repository now contains the **distributed architecture scaffold**, but not yet:
- production-grade frontier management / politeness controls for crawling,
- production deployment hardening for the frontend.


## Healthchecks and startup ordering
- `postgres`, `qdrant`, `inference`, and `gateway` now expose Docker healthchecks.
- `gateway` starts only after `postgres`, `qdrant`, and `inference` report healthy status.
- Services emit structured JSON logs (via `structlog`) for easier ingestion into centralized observability stacks.
