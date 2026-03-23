from __future__ import annotations

import os
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

INFERENCE_URL = os.getenv("LUMINA_INFERENCE_URL", "http://inference:8001")

app = FastAPI(title="lumina-gateway", version="0.1.0")


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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", inference_url=INFERENCE_URL)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            embed_response = await client.post(f"{INFERENCE_URL}/embed", json={"texts": [request.query]})
            embed_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Inference service unavailable: {exc}") from exc

    embed_payload: dict[str, Any] = embed_response.json()
    mock_results = [
        SearchResult(
            id=f"demo-{index}",
            title=f"Demo result {index}",
            score=1.0 / index,
            source="demo",
            url=f"https://example.local/docs/{index}",
            snippet=f"Placeholder result for query '{request.query}' until Qdrant integration is enabled.",
        )
        for index in range(1, min(request.top_k, 5) + 1)
    ]
    return SearchResponse(query=request.query, embedding_model=embed_payload["model"], results=mock_results)
