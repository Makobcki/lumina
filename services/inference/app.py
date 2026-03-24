from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

VECTOR_SIZE = int(os.getenv("LUMINA_VECTOR_SIZE", "1024"))
MODEL_NAME = os.getenv("LUMINA_EMBED_MODEL", "BAAI/bge-m3")
RERANKER_NAME = os.getenv("LUMINA_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MODEL_DEVICE = os.getenv("LUMINA_EMBED_DEVICE", "cpu")

logger = logging.getLogger("lumina.inference")
logging.basicConfig(level=os.getenv("LUMINA_LOG_LEVEL", "INFO"))


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=128)
    normalize: bool = True


class EmbedResponse(BaseModel):
    model: str
    vector_size: int
    embeddings: list[list[float]]


class RerankDocument(BaseModel):
    id: str
    text: str
    title: str | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[RerankDocument] = Field(min_length=1, max_length=64)
    top_k: int = Field(default=10, ge=1, le=64)


class RerankResult(BaseModel):
    id: str
    score: float
    title: str | None = None


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    reranker: str
    vector_size: int
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model '%s' on device '%s'", MODEL_NAME, MODEL_DEVICE)
    model = SentenceTransformer(MODEL_NAME, device=MODEL_DEVICE)
    embedding_dimension = model.get_sentence_embedding_dimension()
    if embedding_dimension is None:
        raise RuntimeError(f"Unable to determine embedding dimension for model '{MODEL_NAME}'")

    logger.info("Loading reranker model '%s' on device '%s'", RERANKER_NAME, MODEL_DEVICE)
    reranker = CrossEncoder(RERANKER_NAME, device=MODEL_DEVICE)

    app.state.model = model
    app.state.reranker = reranker
    app.state.vector_size = int(embedding_dimension)
    logger.info(
        "Inference models loaded: embed='%s' (vector size %s), reranker='%s'",
        MODEL_NAME,
        embedding_dimension,
        RERANKER_NAME,
    )
    yield


app = FastAPI(title="lumina-inference", version="0.3.0", lifespan=lifespan)


def _get_model(request: Request) -> SentenceTransformer:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded")
    return model


def _get_reranker(request: Request) -> CrossEncoder:
    reranker = getattr(request.app.state, "reranker", None)
    if reranker is None:
        raise HTTPException(status_code=503, detail="Reranker model is not loaded")
    return reranker


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    vector_size = int(getattr(request.app.state, "vector_size", VECTOR_SIZE))
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        reranker=RERANKER_NAME,
        vector_size=vector_size,
        device=MODEL_DEVICE,
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(request: Request, payload: EmbedRequest) -> EmbedResponse:
    model = _get_model(request)
    embeddings = model.encode(
        payload.texts,
        normalize_embeddings=payload.normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vector_size = int(getattr(request.app.state, "vector_size", VECTOR_SIZE))
    return EmbedResponse(
        model=MODEL_NAME,
        vector_size=vector_size,
        embeddings=embeddings.tolist(),
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: Request, payload: RerankRequest) -> RerankResponse:
    reranker = _get_reranker(request)
    pairs = [[payload.query, document.text] for document in payload.documents]
    scores = reranker.predict(pairs)
    scored = [
        RerankResult(id=document.id, title=document.title, score=float(score))
        for document, score in zip(payload.documents, scores)
    ]
    results = sorted(scored, key=lambda item: item.score, reverse=True)[: payload.top_k]
    return RerankResponse(model=RERANKER_NAME, results=results)
