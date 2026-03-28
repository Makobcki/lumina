from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import uuid4

import asyncpg
import httpx
import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

INFERENCE_URL = os.getenv("LUMINA_INFERENCE_URL", "http://inference:8001")
QDRANT_URL = os.getenv("LUMINA_QDRANT_URL", "http://qdrant:6333")
POSTGRES_DSN = os.getenv(
    "LUMINA_POSTGRES_DSN",
    "postgresql://lumina:lumina@postgres:5432/lumina",
)
COLLECTION_NAME = os.getenv("LUMINA_QDRANT_COLLECTION", "documents")
VECTOR_SIZE = int(os.getenv("LUMINA_VECTOR_SIZE", "1024"))
RERANK_CANDIDATE_MULTIPLIER = int(os.getenv("LUMINA_RERANK_CANDIDATE_MULTIPLIER", "3"))
MAX_RERANK_CANDIDATES = int(os.getenv("LUMINA_MAX_RERANK_CANDIDATES", "30"))
DENSE_VECTOR_NAME = os.getenv("LUMINA_QDRANT_DENSE_VECTOR_NAME", "dense")
SPARSE_VECTOR_NAME = os.getenv("LUMINA_QDRANT_SPARSE_VECTOR_NAME", "sparse")

def _configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=os.getenv("LUMINA_LOG_LEVEL", "INFO"), force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


_configure_logging()
logger = structlog.get_logger("lumina.gateway")


class IndexRequest(BaseModel):
    id: str | None = None
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    content: str = Field(min_length=1)


class IndexResponse(BaseModel):
    id: str
    status: Literal["indexed"]
    collection: str
    embedding_model: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    id: str
    title: str
    score: float
    source: str
    url: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    embedding_model: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    inference_url: str
    qdrant_url: str
    collection: str
    raw_storage: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = httpx.AsyncClient(timeout=30.0)
    qdrant = AsyncQdrantClient(url=QDRANT_URL)
    pg_pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=1, max_size=10)

    app.state.http_client = http_client
    app.state.qdrant = qdrant
    app.state.pg_pool = pg_pool

    async with pg_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_documents (
                document_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

    collection_exists = await qdrant.collection_exists(collection_name=COLLECTION_NAME)
    if not collection_exists:
        await qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                )
            },
        )
        logger.info(
            "qdrant_collection_created",
            collection=COLLECTION_NAME,
            dense_vector=DENSE_VECTOR_NAME,
            vector_size=VECTOR_SIZE,
            sparse_vector=SPARSE_VECTOR_NAME,
        )
    else:
        logger.info("qdrant_collection_exists", collection=COLLECTION_NAME)

    yield

    await http_client.aclose()
    await qdrant.close()
    await pg_pool.close()


app = FastAPI(title="lumina-gateway", version="0.4.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_http_client(request: Request) -> httpx.AsyncClient:
    client = getattr(request.app.state, "http_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client is not initialized")
    return client


def _get_qdrant(request: Request) -> AsyncQdrantClient:
    client = getattr(request.app.state, "qdrant", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Qdrant client is not initialized")
    return client


def _get_pg_pool(request: Request) -> asyncpg.Pool:
    pool = getattr(request.app.state, "pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="PostgreSQL pool is not initialized")
    return pool


async def _store_raw_document(request: Request, document_id: str, content: str) -> None:
    pool = _get_pg_pool(request)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO raw_documents (document_id, content)
            VALUES ($1, $2)
            ON CONFLICT (document_id)
            DO UPDATE SET content = EXCLUDED.content
            """,
            document_id,
            content,
        )


async def _fetch_raw_documents(request: Request, document_ids: list[str]) -> dict[str, str]:
    if not document_ids:
        return {}

    pool = _get_pg_pool(request)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT document_id, content
            FROM raw_documents
            WHERE document_id = ANY($1::text[])
            """,
            document_ids,
        )

    return {str(row["document_id"]): str(row["content"]) for row in rows}


async def _post_inference(request: Request, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    client = _get_http_client(request)
    try:
        response = await client.post(f"{INFERENCE_URL}{endpoint}", json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Inference service unavailable: {exc}") from exc
    return response.json()


async def _embed_texts(request: Request, texts: list[str]) -> tuple[str, list[list[float]]]:
    payload = await _post_inference(request, "/embed", {"texts": texts})
    return payload["model"], payload["embeddings"]


async def _sparse_embed_texts(request: Request, texts: list[str]) -> tuple[str, list[models.SparseVector]]:
    payload = await _post_inference(request, "/embed/sparse", {"texts": texts})
    sparse_vectors = [
        models.SparseVector(indices=embedding["indices"], values=embedding["values"])
        for embedding in payload["embeddings"]
    ]
    return payload["model"], sparse_vectors


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        inference_url=INFERENCE_URL,
        qdrant_url=QDRANT_URL,
        collection=COLLECTION_NAME,
        raw_storage="postgresql",
    )


@app.post("/index", response_model=IndexResponse)
async def index_document(request: Request, payload: IndexRequest) -> IndexResponse:
    embedding_model, embeddings = await _embed_texts(request, [payload.content])
    _, sparse_embeddings = await _sparse_embed_texts(request, [payload.content])
    document_id = payload.id or str(uuid4())

    await _store_raw_document(request, document_id=document_id, content=payload.content)

    point = PointStruct(
        id=document_id,
        vector={
            DENSE_VECTOR_NAME: embeddings[0],
            SPARSE_VECTOR_NAME: sparse_embeddings[0],
        },
        payload={
            "title": payload.title,
            "url": payload.url,
            "document_id": document_id,
            "source": "indexed",
        },
    )

    qdrant = _get_qdrant(request)
    await qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    return IndexResponse(
        id=document_id,
        status="indexed",
        collection=COLLECTION_NAME,
        embedding_model=embedding_model,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: Request, payload: SearchRequest) -> SearchResponse:
    embedding_model, embeddings = await _embed_texts(request, [payload.query])
    _, sparse_embeddings = await _sparse_embed_texts(request, [payload.query])
    qdrant = _get_qdrant(request)
    candidate_limit = min(max(payload.top_k * RERANK_CANDIDATE_MULTIPLIER, payload.top_k), MAX_RERANK_CANDIDATES)
    query_response = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=embeddings[0],
                using=DENSE_VECTOR_NAME,
                limit=candidate_limit,
            ),
            models.Prefetch(
                query=sparse_embeddings[0],
                using=SPARSE_VECTOR_NAME,
                limit=candidate_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidate_limit,
        with_payload=True,
    )

    candidates_by_id: dict[str, dict[str, Any]] = {}
    candidate_ids: list[str] = []
    for hit in query_response.points:
        payload_data = dict(hit.payload or {})
        candidate_id = str(payload_data.get("document_id") or hit.id)
        candidate_ids.append(candidate_id)
        candidates_by_id[candidate_id] = {
            "title": str(payload_data.get("title", "")),
            "url": str(payload_data.get("url", "")),
            "source": str(payload_data.get("source", "indexed")),
            "initial_score": float(hit.score),
        }

    if not candidate_ids:
        return SearchResponse(query=payload.query, embedding_model=embedding_model, results=[])

    raw_documents = await _fetch_raw_documents(request, candidate_ids)

    rerank_documents: list[dict[str, str | None]] = []
    for candidate_id in candidate_ids:
        text = raw_documents.get(candidate_id)
        if not text:
            continue
        rerank_documents.append(
            {
                "id": candidate_id,
                "title": str(candidates_by_id[candidate_id]["title"]),
                "text": text,
            }
        )

    if not rerank_documents:
        return SearchResponse(query=payload.query, embedding_model=embedding_model, results=[])

    rerank_payload = await _post_inference(
        request,
        "/rerank",
        {"query": payload.query, "documents": rerank_documents, "top_k": payload.top_k},
    )
    reranked_results = rerank_payload["results"]

    results = [
        SearchResult(
            id=result["id"],
            title=str(candidates_by_id[result["id"]]["title"]),
            score=float(result["score"]),
            source=str(candidates_by_id[result["id"]]["source"]),
            url=str(candidates_by_id[result["id"]]["url"]),
            snippet=raw_documents.get(result["id"], "")[:240],
        )
        for result in reranked_results
        if result["id"] in candidates_by_id
    ]
    return SearchResponse(query=payload.query, embedding_model=embedding_model, results=results)
