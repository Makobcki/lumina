from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Literal

import structlog
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

VECTOR_SIZE = int(os.getenv("LUMINA_VECTOR_SIZE", "1024"))
MODEL_NAME = os.getenv("LUMINA_EMBED_MODEL", "BAAI/bge-m3")
RERANKER_NAME = os.getenv("LUMINA_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MODEL_DEVICE = os.getenv("LUMINA_EMBED_DEVICE", "cpu")
EMBED_MAX_BATCH_SIZE = int(os.getenv("LUMINA_EMBED_MAX_BATCH_SIZE", "32"))
EMBED_BATCH_TIMEOUT_MS = int(os.getenv("LUMINA_EMBED_BATCH_TIMEOUT_MS", "50"))
EMBED_CONCURRENCY_LIMIT = int(os.getenv("LUMINA_EMBED_CONCURRENCY", "1"))
RERANK_CONCURRENCY_LIMIT = int(os.getenv("LUMINA_RERANK_CONCURRENCY", "1"))
SPARSE_VOCAB_SIZE = int(os.getenv("LUMINA_SPARSE_VOCAB_SIZE", "200000"))
SPARSE_MIN_TOKEN_LEN = int(os.getenv("LUMINA_SPARSE_MIN_TOKEN_LEN", "2"))


def _resolve_model_device(configured_device: str) -> str:
    normalized_device = configured_device.strip().lower()
    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "cuda_requested_but_unavailable_fallback_to_cpu",
            configured_device=configured_device,
        )
        return "cpu"
    return configured_device

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
logger = structlog.get_logger("lumina.inference")


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


class SnippetDocument(BaseModel):
    id: str
    text: str


class SnippetRequest(BaseModel):
    query: str
    documents: list[SnippetDocument] = Field(min_length=1, max_length=32)
    max_sentences_per_document: int = Field(default=12, ge=1, le=64)
    max_snippet_length: int = Field(default=320, ge=80, le=1024)


class SnippetResult(BaseModel):
    id: str
    snippet: str
    score: float


class SnippetResponse(BaseModel):
    model: str
    results: list[SnippetResult]


class SparseEmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=128)


class SparseVectorPayload(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbedResponse(BaseModel):
    model: str
    vocab_size: int
    embeddings: list[SparseVectorPayload]


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
    sparse_vocab_size: int


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
        "embed_worker_started",
        batch_size=EMBED_MAX_BATCH_SIZE,
        timeout_ms=EMBED_BATCH_TIMEOUT_MS,
        concurrency=EMBED_CONCURRENCY_LIMIT,
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
    runtime_device = _resolve_model_device(MODEL_DEVICE)
    logger.info("loading_embedding_model", model=MODEL_NAME, device=runtime_device)
    model = SentenceTransformer(MODEL_NAME, device=runtime_device)
    embedding_dimension = model.get_sentence_embedding_dimension()
    if embedding_dimension is None:
        raise RuntimeError(f"Unable to determine embedding dimension for model '{MODEL_NAME}'")

    logger.info("loading_reranker_model", model=RERANKER_NAME, device=runtime_device)
    reranker = CrossEncoder(RERANKER_NAME, device=runtime_device)

    app.state.model = model
    app.state.reranker = reranker
    app.state.device = runtime_device
    app.state.vector_size = int(embedding_dimension)
    app.state.embed_queue = asyncio.Queue()
    app.state.embed_semaphore = asyncio.Semaphore(EMBED_CONCURRENCY_LIMIT)
    app.state.rerank_semaphore = asyncio.Semaphore(RERANK_CONCURRENCY_LIMIT)
    app.state.embed_worker = asyncio.create_task(_embed_worker(app))

    logger.info(
        "inference_models_loaded",
        embed_model=MODEL_NAME,
        vector_size=embedding_dimension,
        reranker_model=RERANKER_NAME,
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
        device=str(getattr(request.app.state, "device", MODEL_DEVICE)),
        embed_queue_size=embed_queue.qsize(),
        embed_batch_size=EMBED_MAX_BATCH_SIZE,
        embed_batch_timeout_ms=EMBED_BATCH_TIMEOUT_MS,
        embed_concurrency_limit=EMBED_CONCURRENCY_LIMIT,
        rerank_concurrency_limit=RERANK_CONCURRENCY_LIMIT,
        sparse_vocab_size=SPARSE_VOCAB_SIZE,
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


@app.post("/embed/sparse", response_model=SparseEmbedResponse)
async def sparse_embed(payload: SparseEmbedRequest) -> SparseEmbedResponse:
    embeddings = await asyncio.to_thread(lambda: [_sparse_encode(text) for text in payload.texts])
    return SparseEmbedResponse(
        model=f"bm25-hash-{SPARSE_VOCAB_SIZE}",
        vocab_size=SPARSE_VOCAB_SIZE,
        embeddings=embeddings,
    )


def _predict_rerank_scores(reranker: CrossEncoder, pairs: list[list[str]]) -> list[float]:
    return [float(score) for score in reranker.predict(pairs)]


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_to_sentences(text: str, max_sentences: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    raw_segments = SENTENCE_SPLIT_RE.split(stripped)
    sentences: list[str] = []
    for segment in raw_segments:
        normalized = " ".join(segment.split())
        if not normalized:
            continue
        sentences.append(normalized)
        if len(sentences) >= max_sentences:
            break
    return sentences


def _truncate_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    truncated = value[:max_length].rsplit(" ", 1)[0].strip()
    if not truncated:
        return value[:max_length].strip()
    return truncated


def _build_best_snippets(
    reranker: CrossEncoder,
    query: str,
    documents: list[SnippetDocument],
    max_sentences_per_document: int,
    max_snippet_length: int,
) -> list[SnippetResult]:
    sentence_candidates: list[tuple[str, str]] = []

    for document in documents:
        sentences = _split_to_sentences(document.text, max_sentences=max_sentences_per_document)
        if not sentences:
            fallback = _truncate_text(" ".join(document.text.split()), max_length=max_snippet_length)
            if fallback:
                sentence_candidates.append((document.id, fallback))
            continue
        for sentence in sentences:
            sentence_candidates.append((document.id, sentence))

    if not sentence_candidates:
        return [SnippetResult(id=document.id, snippet="", score=0.0) for document in documents]

    pairs = [[query, sentence] for _, sentence in sentence_candidates]
    sentence_scores = [float(score) for score in reranker.predict(pairs)]

    best_by_document: dict[str, SnippetResult] = {}
    for (document_id, sentence), score in zip(sentence_candidates, sentence_scores, strict=True):
        current_best = best_by_document.get(document_id)
        if current_best is not None and score <= current_best.score:
            continue
        best_by_document[document_id] = SnippetResult(
            id=document_id,
            snippet=_truncate_text(sentence, max_length=max_snippet_length),
            score=score,
        )

    return [best_by_document.get(document.id, SnippetResult(id=document.id, snippet="", score=0.0)) for document in documents]


def _stable_sparse_index(token: str) -> int:
    value = 2166136261
    for byte in token.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value % SPARSE_VOCAB_SIZE


def _sparse_encode(text: str) -> SparseVectorPayload:
    term_weights: dict[int, float] = {}
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        if len(token) < SPARSE_MIN_TOKEN_LEN:
            continue
        sparse_index = _stable_sparse_index(token)
        term_weights[sparse_index] = term_weights.get(sparse_index, 0.0) + 1.0

    if not term_weights:
        return SparseVectorPayload(indices=[], values=[])

    sorted_weights = sorted(term_weights.items(), key=lambda item: item[0])
    indices = [index for index, _ in sorted_weights]
    values = [float(1.0 + (count / (count + 1.0))) for _, count in sorted_weights]
    return SparseVectorPayload(indices=indices, values=values)


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


@app.post("/snippet", response_model=SnippetResponse)
async def snippet(request: Request, payload: SnippetRequest) -> SnippetResponse:
    reranker = _get_reranker(request)
    rerank_semaphore: asyncio.Semaphore = request.app.state.rerank_semaphore

    async with rerank_semaphore:
        results = await asyncio.to_thread(
            _build_best_snippets,
            reranker,
            payload.query,
            payload.documents,
            payload.max_sentences_per_document,
            payload.max_snippet_length,
        )

    return SnippetResponse(model=RERANKER_NAME, results=results)
