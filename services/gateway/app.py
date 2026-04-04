from __future__ import annotations

import logging
import os
import json
import hashlib
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

import asyncpg
import httpx
import structlog
import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from litellm import acompletion
from prometheus_fastapi_instrumentator import Instrumentator
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
LLM_API_KEY = os.getenv("LUMINA_LLM_API_KEY")
LLM_BASE_URL = os.getenv("LUMINA_LLM_BASE_URL")
LLM_MODEL = os.getenv("LUMINA_LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = int(os.getenv("LUMINA_LLM_MAX_TOKENS", "512"))
ASK_TOP_K = int(os.getenv("LUMINA_ASK_TOP_K", "5"))
REDIS_URL = os.getenv("LUMINA_REDIS_URL", "redis://redis:6379/0")
SEARCH_CACHE_TTL_SECONDS = int(os.getenv("LUMINA_SEARCH_CACHE_TTL_SECONDS", "600"))

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


class BulkIndexRequest(BaseModel):
    documents: list[IndexRequest] = Field(min_length=1, max_length=1000)


class BulkIndexResponse(BaseModel):
    ids: list[str]
    indexed_count: int
    status: Literal["indexed"]
    collection: str
    embedding_model: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: dict[str, Any] | None = None


class SearchResult(BaseModel):
    id: str
    title: str
    score: float
    source: str
    url: str
    snippet: str
    indexed_at: str | None = None


class SearchResponse(BaseModel):
    query: str
    embedding_model: str
    results: list[SearchResult]


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=ASK_TOP_K, ge=1, le=10)


class AskSource(BaseModel):
    id: str
    title: str
    url: str
    source: str
    score: float
    indexed_at: str | None = None
    snippet: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    inference_url: str
    qdrant_url: str
    collection: str
    raw_storage: str


class DeleteDocumentsResponse(BaseModel):
    status: Literal["deleted"]
    deleted_count: int
    qdrant_deleted_count: int
    postgres_deleted_count: int
    filter_field: Literal["url", "source"]
    filter_value: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = httpx.AsyncClient(timeout=30.0)
    qdrant = AsyncQdrantClient(url=QDRANT_URL)
    pg_pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=1, max_size=10)
    redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

    app.state.http_client = http_client
    app.state.qdrant = qdrant
    app.state.pg_pool = pg_pool
    app.state.redis_client = redis_client

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
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_query_history (
                id BIGSERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                result_count INTEGER NOT NULL,
                searched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
    await redis_client.aclose()


app = FastAPI(title="lumina-gateway", version="0.4.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, include_in_schema=False, route_name="/metrics")
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


def _get_redis_client(request: Request) -> redis.Redis:
    client = getattr(request.app.state, "redis_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Redis client is not initialized")
    return client


def _build_search_cache_key(payload: SearchRequest) -> str:
    cache_payload = {
        "query": payload.query,
        "top_k": payload.top_k,
        "filters": payload.filters or {},
    }
    serialized_payload = json.dumps(cache_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    key_hash = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()
    return f"lumina:cache:search:{key_hash}"


async def _cache_search_response(
    redis_client: redis.Redis,
    *,
    cache_key: str,
    response: SearchResponse,
) -> None:
    try:
        await redis_client.set(
            cache_key,
            json.dumps(response.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")),
            ex=SEARCH_CACHE_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("search_cache_store_failed", cache_key=cache_key, error=str(exc))



async def _store_raw_document(request: Request, document_id: str, content: str) -> str:
    pool = _get_pg_pool(request)
    async with pool.acquire() as conn:
        stored_document_id = await conn.fetchval(
            """
            INSERT INTO raw_documents (document_id, content)
            VALUES ($1, $2)
            ON CONFLICT (document_id)
            DO UPDATE SET content = EXCLUDED.content
            RETURNING document_id
            """,
            document_id,
            content,
        )
    return str(stored_document_id)


async def _store_raw_documents_bulk(request: Request, documents: list[tuple[str, str]]) -> None:
    if not documents:
        return

    pool = _get_pg_pool(request)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO raw_documents (document_id, content)
                VALUES ($1, $2)
                ON CONFLICT (document_id)
                DO UPDATE SET content = EXCLUDED.content
                """,
                documents,
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


async def _delete_raw_documents(request: Request, document_ids: list[str]) -> int:
    if not document_ids:
        return 0

    pool = _get_pg_pool(request)
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM raw_documents
            WHERE document_id = ANY($1::text[])
            """,
            document_ids,
        )
    return int(result.split()[-1])


async def _store_search_query(
    request: Request,
    *,
    query: str,
    top_k: int,
    result_count: int,
) -> None:
    pool = _get_pg_pool(request)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO search_query_history (query, top_k, result_count)
            VALUES ($1, $2, $3)
            """,
            query,
            top_k,
            result_count,
        )


def _coerce_indexed_at(payload_data: dict[str, Any]) -> str | None:
    raw_value = payload_data.get("indexed_at")
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    deduplicated_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated_values.append(value)
    return deduplicated_values


async def _post_inference(request: Request, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    client = _get_http_client(request)
    try:
        response = await client.post(f"{INFERENCE_URL}{endpoint}", json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Пробрасываем реальную ошибку (например 422 или 500) от микросервиса
        error_body = exc.response.text
        raise HTTPException(status_code=502, detail=f"Inference error {exc.response.status_code}: {error_body}") from exc
    except httpx.HTTPError as exc:
        # Сработает, если контейнер inference упал (Connection refused)
        raise HTTPException(status_code=502, detail=f"Inference unreachable: {exc}") from exc
    return response.json()


async def _semantic_snippets(
    request: Request,
    *,
    query: str,
    documents: list[dict[str, str | None]],
) -> dict[str, str]:
    if not documents:
        return {}

    payload = await _post_inference(
        request,
        "/snippet",
        {
            "query": query,
            "documents": documents,
            "max_sentences_per_document": 12,
            "max_snippet_length": 320,
        },
    )
    snippet_map: dict[str, str] = {}
    for item in payload.get("results", []):
        document_id = str(item.get("id", ""))
        if not document_id:
            continue
        snippet_map[document_id] = str(item.get("snippet", ""))
    return snippet_map


def _build_ask_system_prompt(results: list[SearchResult]) -> str:
    snippets = "\n\n".join(
        [
            (
                f"[{index}] Заголовок: {result.title}\n"
                f"URL: {result.url}\n"
                f"Источник: {result.source}\n"
                f"Сниппет: {result.snippet}"
            )
            for index, result in enumerate(results, start=1)
        ]
    )
    return (
        "Ты полезный ассистент. Ответь на вопрос пользователя, используя ТОЛЬКО предоставленный контекст. "
        "Если контекста недостаточно, честно скажи, что в источниках нет точного ответа.\n\n"
        f"Контекст:\n{snippets}"
    )


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


async def _find_document_ids_by_payload_field(
    request: Request,
    *,
    field_name: Literal["url", "source"],
    field_value: str,
) -> list[str]:
    qdrant = _get_qdrant(request)
    next_offset: models.PointId | None = None
    document_ids: list[str] = []

    while True:
        points, next_offset = await qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=field_name,
                        match=models.MatchValue(value=field_value),
                    )
                ]
            ),
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=next_offset,
        )

        if not points:
            break

        for point in points:
            payload_data = dict(point.payload or {})
            document_id = payload_data.get("document_id") or point.id
            if document_id is None:
                continue
            document_ids.append(str(document_id))

        if next_offset is None:
            break

    return _deduplicate_preserve_order(document_ids)


async def _delete_documents_by_field(
    request: Request,
    *,
    field_name: Literal["url", "source"],
    field_value: str,
) -> tuple[int, int, int]:
    document_ids = await _find_document_ids_by_payload_field(
        request,
        field_name=field_name,
        field_value=field_value,
    )

    if not document_ids:
        return 0, 0, 0

    qdrant = _get_qdrant(request)
    await qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=document_ids),
    )

    postgres_deleted_count = await _delete_raw_documents(request, document_ids)
    deleted_count = len(document_ids)
    return deleted_count, deleted_count, postgres_deleted_count


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

    stored_document_id = await _store_raw_document(request, document_id=document_id, content=payload.content)

    point = PointStruct(
        id=stored_document_id,
        vector={
            DENSE_VECTOR_NAME: embeddings[0],
            SPARSE_VECTOR_NAME: sparse_embeddings[0],
        },
        payload={
            "title": payload.title,
            "url": payload.url,
            "document_id": stored_document_id,
            "source": "indexed",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    qdrant = _get_qdrant(request)
    await qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    return IndexResponse(
        id=stored_document_id,
        status="indexed",
        collection=COLLECTION_NAME,
        embedding_model=embedding_model,
    )


@app.post("/index/bulk", response_model=BulkIndexResponse)
async def index_documents_bulk(request: Request, payload: BulkIndexRequest) -> BulkIndexResponse:
    documents = payload.documents
    contents = [document.content for document in documents]
    embedding_model, embeddings = await _embed_texts(request, contents)
    _, sparse_embeddings = await _sparse_embed_texts(request, contents)

    document_ids = [document.id or str(uuid4()) for document in documents]
    await _store_raw_documents_bulk(
        request,
        documents=[(document_id, document.content) for document_id, document in zip(document_ids, documents, strict=True)],
    )

    points = [
        PointStruct(
            id=document_id,
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: sparse_vector,
            },
            payload={
                "title": document.title,
                "url": document.url,
                "document_id": document_id,
                "source": "indexed",
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        for document_id, document, dense_vector, sparse_vector in zip(
            document_ids,
            documents,
            embeddings,
            sparse_embeddings,
            strict=True,
        )
    ]

    qdrant = _get_qdrant(request)
    await qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    return BulkIndexResponse(
        ids=document_ids,
        indexed_count=len(document_ids),
        status="indexed",
        collection=COLLECTION_NAME,
        embedding_model=embedding_model,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: Request, payload: SearchRequest) -> SearchResponse:
    redis_client = _get_redis_client(request)
    cache_key = _build_search_cache_key(payload)
    cached_response = await redis_client.get(cache_key)
    if cached_response is not None:
        cached_data = json.loads(cached_response)
        return SearchResponse.model_validate(cached_data)

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
            "indexed_at": _coerce_indexed_at(payload_data),
            "initial_score": float(hit.score),
        }

    if not candidate_ids:
        return SearchResponse(query=payload.query, embedding_model=embedding_model, results=[])

    unique_candidate_ids = _deduplicate_preserve_order(candidate_ids)
    raw_documents = await _fetch_raw_documents(request, unique_candidate_ids)

    rerank_documents: list[dict[str, str | None]] = []
    for candidate_id in unique_candidate_ids:
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
    snippet_map = await _semantic_snippets(request, query=payload.query, documents=rerank_documents)

    results = [
        SearchResult(
            id=result["id"],
            title=str(candidates_by_id[result["id"]]["title"]),
            score=float(result["score"]),
            source=str(candidates_by_id[result["id"]]["source"]),
            url=str(candidates_by_id[result["id"]]["url"]),
            snippet=snippet_map.get(result["id"], raw_documents.get(result["id"], "")[:240]),
            indexed_at=candidates_by_id[result["id"]]["indexed_at"],
        )
        for result in reranked_results
        if result["id"] in candidates_by_id
    ]
    await _store_search_query(
        request,
        query=payload.query,
        top_k=payload.top_k,
        result_count=len(results),
    )
    response = SearchResponse(query=payload.query, embedding_model=embedding_model, results=results)
    asyncio.create_task(_cache_search_response(redis_client, cache_key=cache_key, response=response))
    return response


@app.post("/ask")
async def ask(request: Request, payload: AskRequest) -> StreamingResponse:
    search_payload = SearchRequest(query=payload.query, top_k=payload.top_k)
    search_response = await search(request, search_payload)
    context_results = search_response.results
    sources = [
        AskSource(
            id=result.id,
            title=result.title,
            url=result.url,
            source=result.source,
            score=result.score,
            indexed_at=result.indexed_at,
            snippet=result.snippet,
        )
        for result in context_results
    ]

    async def event_stream() -> AsyncIterator[str]:
        source_json = [source.model_dump() for source in sources]
        yield f"event: sources\ndata: {json.dumps(source_json, ensure_ascii=False)}\n\n"

        if not context_results:
            yield "event: done\ndata: Контекст не найден. Попробуйте переформулировать запрос.\n\n"
            return

        messages = [
            {"role": "system", "content": _build_ask_system_prompt(context_results)},
            {"role": "user", "content": payload.query},
        ]
        try:
            completion = await acompletion(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                api_key=LLM_API_KEY,
                api_base=LLM_BASE_URL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.0,
            )
        except Exception as exc:
            error_message = str(exc).replace("\n", " ")
            yield f"event: error\ndata: {error_message}\n\n"
            return

        async for chunk in completion:
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if not content:
                continue
            escaped_content = str(content).replace("\n", "\\n")
            yield f"event: token\ndata: {escaped_content}\n\n"

        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/documents", response_model=DeleteDocumentsResponse)
async def delete_documents(
    request: Request,
    url: str | None = None,
    source: str | None = None,
) -> DeleteDocumentsResponse:
    if bool(url) == bool(source):
        raise HTTPException(status_code=422, detail="Provide exactly one of: url or source")

    if url is not None:
        filter_field: Literal["url", "source"] = "url"
        filter_value = url
    else:
        filter_field = "source"
        filter_value = str(source)

    deleted_count, qdrant_deleted_count, postgres_deleted_count = await _delete_documents_by_field(
        request,
        field_name=filter_field,
        field_value=filter_value,
    )

    logger.info(
        "documents_deleted",
        filter_field=filter_field,
        filter_value=filter_value,
        deleted_count=deleted_count,
        qdrant_deleted_count=qdrant_deleted_count,
        postgres_deleted_count=postgres_deleted_count,
    )
    return DeleteDocumentsResponse(
        status="deleted",
        deleted_count=deleted_count,
        qdrant_deleted_count=qdrant_deleted_count,
        postgres_deleted_count=postgres_deleted_count,
        filter_field=filter_field,
        filter_value=filter_value,
    )
