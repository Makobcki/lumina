from __future__ import annotations

import hashlib
import math
import os
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

VECTOR_SIZE = int(os.getenv("LUMINA_VECTOR_SIZE", "1024"))
MODEL_NAME = os.getenv("LUMINA_EMBED_MODEL", "BAAI/bge-m3")
RERANKER_NAME = os.getenv("LUMINA_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

app = FastAPI(title="lumina-inference", version="0.1.0")


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
    documents: list[RerankDocument] = Field(min_length=1, max_length=32)
    top_k: int = Field(default=10, ge=1, le=32)


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


def _hash_to_unit_vector(text: str, size: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < size:
        digest = hashlib.blake2b(f"{text}:{counter}".encode("utf-8"), digest_size=32).digest()
        for idx in range(0, len(digest), 4):
            chunk = digest[idx : idx + 4]
            integer = int.from_bytes(chunk, byteorder="little", signed=False)
            values.append((integer / 2**32) * 2.0 - 1.0)
            if len(values) == size:
                break
        counter += 1

    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=MODEL_NAME, reranker=RERANKER_NAME, vector_size=VECTOR_SIZE)


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    embeddings = [_hash_to_unit_vector(text, VECTOR_SIZE) for text in request.texts]
    return EmbedResponse(model=MODEL_NAME, vector_size=VECTOR_SIZE, embeddings=embeddings)


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest) -> RerankResponse:
    query_terms = set(request.query.lower().split())
    scored: list[RerankResult] = []
    for document in request.documents:
        document_terms = set(document.text.lower().split())
        overlap = len(query_terms & document_terms)
        length_penalty = max(len(document.text.split()), 1)
        score = overlap / length_penalty
        scored.append(RerankResult(id=document.id, title=document.title, score=score))

    results = sorted(scored, key=lambda item: item.score, reverse=True)[: request.top_k]
    return RerankResponse(model=RERANKER_NAME, results=results)
