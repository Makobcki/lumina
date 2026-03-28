from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

VECTOR_SIZE = int(os.getenv("LUMINA_VECTOR_SIZE", "1024"))
MODEL_NAME = os.getenv("LUMINA_EMBED_MODEL", "BAAI/bge-m3")
RERANKER_NAME = os.getenv("LUMINA_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MODEL_DEVICE = os.getenv("LUMINA_EMBED_DEVICE", "cpu")
EMBED_MAX_BATCH_SIZE = int(os.getenv("LUMINA_EMBED_MAX_BATCH_SIZE", "32"))
EMBED_BATCH_TIMEOUT_MS = int(os.getenv("LUMINA_EMBED_BATCH_TIMEOUT_MS", "50"))
EMBED_CONCURRENCY_LIMIT = int(os.getenv("LUMINA_EMBED_CONCURRENCY", "1"))
RERANK_CONCURRENCY_LIMIT = int(os.getenv("LUMINA_RERANK_CONCURRENCY", "1"))

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
    embed_queue_size: int
    embed_batch_size: int
    embed_batch_timeout_ms: int
    embed_concurrency_limit: int
    rerank_concurrency_limit: int


@dataclass(slots=True)
class EmbedJob:
    texts: list[str]
    normalize: bool
    future: asyncio.Future[list[list[float]]]


def _encode_texts(model: SentenceTransformer, texts: list[str], normalize: bool) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


async def _process_embed_batch(app: FastAPI, jobs: list[EmbedJob]) -> None:
    model = app.state.model
    semaphore: asyncio.Semaphore = app.state.embed_semaphore

    grouped: dict[bool, list[EmbedJob]] = {True: [], False: []}
    for job in jobs:
        grouped[job.normalize].append(job)

    async with semaphore:
        for normalize, normalized_jobs in grouped.items():
            if not normalized_jobs:
                continue
            all_texts = [text for job in normalized_jobs for text in job.texts]
            try:
                vectors = await asyncio.to_thread(_encode_texts, model, all_texts, normalize)
            except Exception as exc:  # defensive path for model/runtime failures
                for job in normalized_jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)
                continue

            cursor = 0
            for job in normalized_jobs:
                size = len(job.texts)
                chunk = vectors[cursor : cursor + size]
                cursor += size
                if not job.future.done():
                    job.future.set_result(chunk)


async def _embed_worker(app: FastAPI) -> None:
    queue: asyncio.Queue[EmbedJob] = app.state.embed_queue
    batch_timeout = EMBED_BATCH_TIMEOUT_MS / 1000
    logger.info(
        "Embed worker started (batch_size=%s, timeout=%sms, concurrency=%s)",
        EMBED_MAX_BATCH_SIZE,
        EMBED_BATCH_TIMEOUT_MS,
        EMBED_CONCURRENCY_LIMIT,
    )

    while True:
        jobs: list[EmbedJob] = []
        first_job = await queue.get()
        jobs.append(first_job)

        deadline = asyncio.get_running_loop().time() + batch_timeout
        while len(jobs) < EMBED_MAX_BATCH_SIZE:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                next_job = await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            jobs.append(next_job)

        try:
            await _process_embed_batch(app, jobs)
        finally:
            for _ in jobs:
                queue.task_done()


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
    app.state.embed_queue = asyncio.Queue()
    app.state.embed_semaphore = asyncio.Semaphore(EMBED_CONCURRENCY_LIMIT)
    app.state.rerank_semaphore = asyncio.Semaphore(RERANK_CONCURRENCY_LIMIT)
    app.state.embed_worker = asyncio.create_task(_embed_worker(app))

    logger.info(
        "Inference models loaded: embed='%s' (vector size %s), reranker='%s'",
        MODEL_NAME,
        embedding_dimension,
        RERANKER_NAME,
    )
    try:
        yield
    finally:
        worker: asyncio.Task = app.state.embed_worker
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


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
    embed_queue: asyncio.Queue[EmbedJob] = getattr(request.app.state, "embed_queue", asyncio.Queue())
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        reranker=RERANKER_NAME,
        vector_size=vector_size,
        device=MODEL_DEVICE,
        embed_queue_size=embed_queue.qsize(),
        embed_batch_size=EMBED_MAX_BATCH_SIZE,
        embed_batch_timeout_ms=EMBED_BATCH_TIMEOUT_MS,
        embed_concurrency_limit=EMBED_CONCURRENCY_LIMIT,
        rerank_concurrency_limit=RERANK_CONCURRENCY_LIMIT,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: Request, payload: EmbedRequest) -> EmbedResponse:
    _get_model(request)
    queue: asyncio.Queue[EmbedJob] = request.app.state.embed_queue
    future: asyncio.Future[list[list[float]]] = asyncio.get_running_loop().create_future()
    await queue.put(EmbedJob(texts=payload.texts, normalize=payload.normalize, future=future))
    embeddings = await future

    vector_size = int(getattr(request.app.state, "vector_size", VECTOR_SIZE))
    return EmbedResponse(
        model=MODEL_NAME,
        vector_size=vector_size,
        embeddings=embeddings,
    )


def _predict_rerank_scores(reranker: CrossEncoder, pairs: list[list[str]]) -> list[float]:
    return [float(score) for score in reranker.predict(pairs)]


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: Request, payload: RerankRequest) -> RerankResponse:
    reranker = _get_reranker(request)
    pairs = [[payload.query, document.text] for document in payload.documents]

    rerank_semaphore: asyncio.Semaphore = request.app.state.rerank_semaphore
    async with rerank_semaphore:
        scores = await asyncio.to_thread(_predict_rerank_scores, reranker, pairs)

    scored = [
        RerankResult(id=document.id, title=document.title, score=score)
        for document, score in zip(payload.documents, scores)
    ]
    results = sorted(scored, key=lambda item: item.score, reverse=True)[: payload.top_k]
    return RerankResponse(model=RERANKER_NAME, results=results)
