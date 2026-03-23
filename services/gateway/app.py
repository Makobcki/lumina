from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

INFERENCE_URL = os.getenv("LUMINA_INFERENCE_URL", "http://inference:8001")
QDRANT_URL = os.getenv("LUMINA_QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("LUMINA_QDRANT_COLLECTION", "documents")
VECTOR_SIZE = int(os.getenv("LUMINA_VECTOR_SIZE", "1024"))
RERANK_CANDIDATE_MULTIPLIER = int(os.getenv("LUMINA_RERANK_CANDIDATE_MULTIPLIER", "3"))
MAX_RERANK_CANDIDATES = int(os.getenv("LUMINA_MAX_RERANK_CANDIDATES", "30"))

logger = logging.getLogger("lumina.gateway")
logging.basicConfig(level=os.getenv("LUMINA_LOG_LEVEL", "INFO"))


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = httpx.AsyncClient(timeout=30.0)
    qdrant = AsyncQdrantClient(url=QDRANT_URL)

    app.state.http_client = http_client
    app.state.qdrant = qdrant

    collection_exists = await qdrant.collection_exists(collection_name=COLLECTION_NAME)
    if not collection_exists:
        await qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' with vector size %s", COLLECTION_NAME, VECTOR_SIZE)
    else:
        logger.info("Qdrant collection '%s' already exists", COLLECTION_NAME)

    yield

    await http_client.aclose()
    await qdrant.close()


app = FastAPI(title="lumina-gateway", version="0.3.0", lifespan=lifespan)
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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        inference_url=INFERENCE_URL,
        qdrant_url=QDRANT_URL,
        collection=COLLECTION_NAME,
    )


@app.post("/index", response_model=IndexResponse)
async def index_document(request: Request, payload: IndexRequest) -> IndexResponse:
    embedding_model, embeddings = await _embed_texts(request, [payload.content])
    document_id = payload.id or str(uuid4())
    point = PointStruct(
        id=document_id,
        vector=embeddings[0],
        payload={
            "title": payload.title,
            "url": payload.url,
            "content": payload.content,
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
    qdrant = _get_qdrant(request)
    candidate_limit = min(max(payload.top_k * RERANK_CANDIDATE_MULTIPLIER, payload.top_k), MAX_RERANK_CANDIDATES)
    query_response = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embeddings[0],
        limit=candidate_limit,
        with_payload=True,
    )

    candidates_by_id: dict[str, dict[str, Any]] = {}
    rerank_documents: list[dict[str, str | None]] = []
    for hit in query_response.points:
        payload_data = dict(hit.payload or {})
        candidate_id = str(hit.id)
        candidates_by_id[candidate_id] = {
            "title": str(payload_data.get("title", "")),
            "url": str(payload_data.get("url", "")),
            "content": str(payload_data.get("content", "")),
            "source": str(payload_data.get("source", "indexed")),
            "initial_score": float(hit.score),
        }
        rerank_documents.append(
            {
                "id": candidate_id,
                "title": str(payload_data.get("title", "")),
                "text": str(payload_data.get("content", "")),
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
            snippet=str(candidates_by_id[result["id"]]["content"])[:240],
        )
        for result in reranked_results
        if result["id"] in candidates_by_id
    ]
    return SearchResponse(query=payload.query, embedding_model=embedding_model, results=results)
