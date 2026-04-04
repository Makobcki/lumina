from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import redis.asyncio as redis
import structlog
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

GATEWAY_BULK_INDEX_URL = os.getenv("LUMINA_GATEWAY_BULK_INDEX_URL", "http://localhost:8000/index/bulk")
GATEWAY_DELETE_URL = os.getenv("LUMINA_GATEWAY_DELETE_URL", "http://localhost:8000/documents")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LUMINA_REQUEST_TIMEOUT_SECONDS", "30.0"))
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
INDEX_BULK_SIZE = int(os.getenv("LUMINA_INDEX_BULK_SIZE", "64"))
MAX_CONCURRENT_FETCHES = 20
MAX_CONCURRENT_INDEX_REQUESTS = 20
USER_AGENT = "lumina-crawler/1.0"
DEFAULT_CRAWL_DELAY_SECONDS = 0.5
INGESTION_MODE = os.getenv("LUMINA_INGESTION_MODE", "direct")
REDIS_URL = os.getenv("LUMINA_REDIS_URL", "redis://localhost:6379/0")
REDIS_STREAM_NAME = os.getenv("LUMINA_REDIS_STREAM_NAME", "lumina:index:bulk")
REDIS_FRONTIER_QUEUE_KEY = os.getenv("LUMINA_REDIS_FRONTIER_QUEUE_KEY", "lumina:crawler:frontier:queue")
REDIS_FRONTIER_SEEN_KEY = os.getenv("LUMINA_REDIS_FRONTIER_SEEN_KEY", "lumina:crawler:frontier:seen")
MAX_CRAWL_PAGES = int(os.getenv("LUMINA_MAX_CRAWL_PAGES", "500"))
START_URLS = [
    "https://fastapi.tiangolo.com/",
    "https://ru.wikipedia.org/wiki/Очередь_(программирование)",
]
REMOVABLE_TAGS = {
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "noscript",
    "svg",
    "img",
    "form",
    "aside",
}


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
logger = structlog.get_logger("lumina.crawler")


@dataclass(slots=True)
class DomainState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_request_monotonic: float = 0.0


class PolitenessPolicy:
    def __init__(self, delay_seconds: float = DEFAULT_CRAWL_DELAY_SECONDS, user_agent: str = USER_AGENT) -> None:
        self.delay_seconds = delay_seconds
        self.user_agent = user_agent
        self._domain_states: dict[str, DomainState] = {}
        self._robots_parsers: dict[str, RobotFileParser | None] = {}
        self._robots_lock = asyncio.Lock()

    async def wait_turn(self, url: str) -> None:
        domain = self._domain(url)
        state = self._domain_states.setdefault(domain, DomainState())
        async with state.lock:
            elapsed = time.monotonic() - state.last_request_monotonic
            sleep_seconds = max(0.0, self.delay_seconds - elapsed)
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)
            state.last_request_monotonic = time.monotonic()

    async def can_fetch(self, url: str, client: httpx.AsyncClient) -> bool:
        parser = await self._get_robots_parser(url=url, client=client)
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    async def _get_robots_parser(self, url: str, client: httpx.AsyncClient) -> RobotFileParser | None:
        domain = self._domain(url)
        async with self._robots_lock:
            if domain in self._robots_parsers:
                return self._robots_parsers[domain]

        robots_url = f"{domain}/robots.txt"
        parser: RobotFileParser | None = None
        try:
            response = await client.get(robots_url, follow_redirects=True)
            if response.status_code < 400:
                parser = RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(response.text.splitlines())
        except httpx.HTTPError:
            logger.warning("could_not_fetch_robots", domain=domain)

        async with self._robots_lock:
            self._robots_parsers[domain] = parser

        return parser

    @staticmethod
    def _domain(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"


def _extract_title_and_text(html: str, fallback_title: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in REMOVABLE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else fallback_title
    raw_text = soup.get_text(separator=" ", strip=True)
    cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
    return title, cleaned_text


def _normalize_url(raw_url: str, base_url: str) -> str | None:
    absolute_url = urljoin(base_url, raw_url)
    normalized_url, _ = urldefrag(absolute_url)
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"}:
        return None
    return normalized_url


def _extract_links(html: str, page_url: str, allowed_domains: set[str]) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        normalized_url = _normalize_url(anchor["href"], base_url=page_url)
        if normalized_url is None:
            continue
        if urlparse(normalized_url).netloc not in allowed_domains:
            continue
        links.append(normalized_url)
    return links


async def fetch_and_clean(
    url: str,
    client: httpx.AsyncClient,
    politeness_policy: PolitenessPolicy,
    allowed_domains: set[str],
) -> tuple[str, str, list[str]] | None:
    if not await politeness_policy.can_fetch(url=url, client=client):
        logger.warning("robots_txt_disallows_crawling", url=url)
        return None

    await politeness_policy.wait_turn(url)
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=6),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    ):
        with attempt:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

    title, text = await asyncio.to_thread(_extract_title_and_text, response.text, url)
    links = await asyncio.to_thread(_extract_links, response.text, url, allowed_domains)
    return title, text, links


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )
    chunks: list[str] = []
    for chunk in splitter.split_text(text):
        normalized_chunk = re.sub(r"\s+", " ", chunk).strip()
        if not normalized_chunk or len(normalized_chunk) <= 1:
            continue
        chunks.append(normalized_chunk)
    return chunks


def _build_documents(url: str, title: str, batch_start_index: int, batch_chunks: list[str]) -> list[dict[str, str]]:
    return [
        {
            "title": f"{title} [chunk {chunk_index}]",
            "url": url,
            "content": chunk,
        }
        for chunk_index, chunk in enumerate(batch_chunks, start=batch_start_index)
    ]


async def _send_to_gateway(
    url: str,
    batch_start_index: int,
    batch_chunks: list[str],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    title: str,
) -> None:
    payload = {"documents": _build_documents(url=url, title=title, batch_start_index=batch_start_index, batch_chunks=batch_chunks)}
    async with semaphore:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        ):
            with attempt:
                response = await client.post(GATEWAY_BULK_INDEX_URL, json=payload)
                response.raise_for_status()

    logger.info(
        "chunks_indexed_bulk",
        url=url,
        chunk_start=batch_start_index,
        chunk_end=batch_start_index + len(batch_chunks) - 1,
        chunk_count=len(batch_chunks),
    )


async def _publish_to_redis_stream(
    url: str,
    batch_start_index: int,
    batch_chunks: list[str],
    stream_client: redis.Redis,
    title: str,
) -> None:
    payload = {"documents": _build_documents(url=url, title=title, batch_start_index=batch_start_index, batch_chunks=batch_chunks)}
    await stream_client.xadd(
        name=REDIS_STREAM_NAME,
        fields={"payload": json.dumps(payload, ensure_ascii=False)},
    )
    logger.info(
        "chunks_queued_bulk",
        url=url,
        stream=REDIS_STREAM_NAME,
        chunk_start=batch_start_index,
        chunk_end=batch_start_index + len(batch_chunks) - 1,
        chunk_count=len(batch_chunks),
    )


async def _delete_existing_document(url: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        ):
            with attempt:
                response = await client.delete(GATEWAY_DELETE_URL, params={"url": url})
                response.raise_for_status()

    logger.info("document_cleanup_completed", url=url)


async def index_chunks(
    url: str,
    title: str,
    chunks: Iterable[str],
    gateway_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    stream_client: redis.Redis | None,
) -> None:
    chunk_list = list(chunks)
    if not chunk_list:
        return

    if INDEX_BULK_SIZE < 1:
        raise ValueError("LUMINA_INDEX_BULK_SIZE must be >= 1")

    if INGESTION_MODE not in {"direct", "redis_stream"}:
        raise ValueError("LUMINA_INGESTION_MODE must be one of: direct, redis_stream")

    tasks: list[asyncio.Task[None]] = []
    for start_index in range(0, len(chunk_list), INDEX_BULK_SIZE):
        batch_start_index = start_index + 1
        batch_chunks = chunk_list[start_index : start_index + INDEX_BULK_SIZE]
        if INGESTION_MODE == "direct":
            tasks.append(
                asyncio.create_task(
                    _send_to_gateway(
                        url=url,
                        batch_start_index=batch_start_index,
                        batch_chunks=batch_chunks,
                        client=gateway_client,
                        semaphore=semaphore,
                        title=title,
                    )
                )
            )
        else:
            if stream_client is None:
                raise ValueError("Redis stream mode requires an initialized Redis client")
            tasks.append(
                asyncio.create_task(
                    _publish_to_redis_stream(
                        url=url,
                        batch_start_index=batch_start_index,
                        batch_chunks=batch_chunks,
                        stream_client=stream_client,
                        title=title,
                    )
                )
            )

    if tasks:
        await asyncio.gather(*tasks)


async def crawl_and_index(
    url: str,
    fetch_client: httpx.AsyncClient,
    gateway_client: httpx.AsyncClient,
    politeness_policy: PolitenessPolicy,
    indexing_semaphore: asyncio.Semaphore,
    stream_client: redis.Redis | None,
    allowed_domains: set[str],
) -> list[str]:
    logger.info("fetching_url", url=url)
    result = await fetch_and_clean(
        url=url,
        client=fetch_client,
        politeness_policy=politeness_policy,
        allowed_domains=allowed_domains,
    )
    if result is None:
        return []

    title, text, links = result
    chunks = chunk_text(text)
    logger.info("chunks_prepared", chunk_count=len(chunks), url=url)
    await _delete_existing_document(url=url, client=gateway_client, semaphore=indexing_semaphore)
    await index_chunks(
        url=url,
        title=title,
        chunks=chunks,
        gateway_client=gateway_client,
        semaphore=indexing_semaphore,
        stream_client=stream_client,
    )
    return links


class RedisFrontier:
    def __init__(self, redis_client: redis.Redis, queue_key: str, seen_key: str) -> None:
        self.redis_client = redis_client
        self.queue_key = queue_key
        self.seen_key = seen_key

    async def seed(self, urls: Iterable[str]) -> None:
        for url in urls:
            await self.push(url)

    async def push(self, url: str) -> bool:
        added = await self.redis_client.sadd(self.seen_key, url)
        if added:
            await self.redis_client.rpush(self.queue_key, url)
            return True
        return False

    async def pop(self) -> str | None:
        return await self.redis_client.lpop(self.queue_key)


async def run_crawler(urls: list[str]) -> None:
    if INGESTION_MODE not in {"direct", "redis_stream"}:
        raise ValueError("LUMINA_INGESTION_MODE must be one of: direct, redis_stream")

    limits = httpx.Limits(max_connections=MAX_CONCURRENT_FETCHES, max_keepalive_connections=MAX_CONCURRENT_FETCHES)
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)

    politeness_policy = PolitenessPolicy(delay_seconds=DEFAULT_CRAWL_DELAY_SECONDS)
    indexing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INDEX_REQUESTS)

    redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    await redis_client.ping()
    frontier = RedisFrontier(
        redis_client=redis_client,
        queue_key=REDIS_FRONTIER_QUEUE_KEY,
        seen_key=REDIS_FRONTIER_SEEN_KEY,
    )
    await frontier.seed(urls)

    if INGESTION_MODE == "redis_stream":
        logger.info("redis_stream_mode_enabled", redis_url=REDIS_URL, stream=REDIS_STREAM_NAME)

    allowed_domains = {urlparse(url).netloc for url in urls}

    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers={"User-Agent": USER_AGENT}) as fetch_client, httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        headers={"User-Agent": USER_AGENT},
    ) as gateway_client:
        crawled_pages = 0
        while crawled_pages < MAX_CRAWL_PAGES:
            url = await frontier.pop()
            if url is None:
                logger.info("frontier_exhausted", crawled_pages=crawled_pages)
                break

            try:
                discovered_links = await crawl_and_index(
                    url=url,
                    fetch_client=fetch_client,
                    gateway_client=gateway_client,
                    politeness_policy=politeness_policy,
                    indexing_semaphore=indexing_semaphore,
                    stream_client=redis_client if INGESTION_MODE == "redis_stream" else None,
                    allowed_domains=allowed_domains,
                )
                for discovered_url in discovered_links:
                    await frontier.push(discovered_url)
                crawled_pages += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("crawl_failed", url=url, error=str(exc))

    await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(run_crawler(START_URLS))
