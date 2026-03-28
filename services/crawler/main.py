from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import structlog
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

GATEWAY_BULK_INDEX_URL = os.getenv("LUMINA_GATEWAY_BULK_INDEX_URL", "http://localhost:8000/index/bulk")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LUMINA_REQUEST_TIMEOUT_SECONDS", "30.0"))
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
INDEX_BULK_SIZE = int(os.getenv("LUMINA_INDEX_BULK_SIZE", "64"))
MAX_CONCURRENT_FETCHES = 20
MAX_CONCURRENT_INDEX_REQUESTS = 20
USER_AGENT = "lumina-crawler/1.0"
DEFAULT_CRAWL_DELAY_SECONDS = 0.5
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
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
    cleaned_lines = [line for line in lines if line]
    cleaned_text = "\n".join(cleaned_lines)
    return title, cleaned_text


async def fetch_and_clean(
    url: str,
    client: httpx.AsyncClient,
    politeness_policy: PolitenessPolicy,
) -> tuple[str, str] | None:
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

    return await asyncio.to_thread(_extract_title_and_text, response.text, url)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )
    return [chunk for chunk in splitter.split_text(text) if chunk]


async def index_chunks(
    url: str,
    title: str,
    chunks: Iterable[str],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> None:
    chunk_list = list(chunks)
    if not chunk_list:
        return

    if INDEX_BULK_SIZE < 1:
        raise ValueError("LUMINA_INDEX_BULK_SIZE must be >= 1")

    async def index_bulk(batch_start_index: int, batch_chunks: list[str]) -> None:
        documents = [
            {
                "title": f"{title} [chunk {chunk_index}]",
                "url": url,
                "content": chunk,
            }
            for chunk_index, chunk in enumerate(batch_chunks, start=batch_start_index)
        ]
        payload = {"documents": documents}
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

    tasks = [
        index_bulk(
            batch_start_index=start_index + 1,
            batch_chunks=chunk_list[start_index : start_index + INDEX_BULK_SIZE],
        )
        for start_index in range(0, len(chunk_list), INDEX_BULK_SIZE)
    ]
    if tasks:
        await asyncio.gather(*tasks)


async def crawl_and_index(
    url: str,
    fetch_client: httpx.AsyncClient,
    gateway_client: httpx.AsyncClient,
    politeness_policy: PolitenessPolicy,
    indexing_semaphore: asyncio.Semaphore,
) -> None:
    logger.info("fetching_url", url=url)
    result = await fetch_and_clean(url=url, client=fetch_client, politeness_policy=politeness_policy)
    if result is None:
        return

    title, text = result
    chunks = chunk_text(text)
    logger.info("chunks_prepared", chunk_count=len(chunks), url=url)
    await index_chunks(url=url, title=title, chunks=chunks, client=gateway_client, semaphore=indexing_semaphore)


async def run_crawler(urls: list[str]) -> None:
    limits = httpx.Limits(max_connections=MAX_CONCURRENT_FETCHES, max_keepalive_connections=MAX_CONCURRENT_FETCHES)
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)

    politeness_policy = PolitenessPolicy(delay_seconds=DEFAULT_CRAWL_DELAY_SECONDS)
    indexing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INDEX_REQUESTS)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers={"User-Agent": USER_AGENT}) as fetch_client, httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        headers={"User-Agent": USER_AGENT},
    ) as gateway_client:
        tasks = [
            crawl_and_index(
                url=url,
                fetch_client=fetch_client,
                gateway_client=gateway_client,
                politeness_policy=politeness_policy,
                indexing_semaphore=indexing_semaphore,
            )
            for url in urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for url, result in zip(urls, results, strict=True):
        if isinstance(result, Exception):
            logger.error("crawl_failed", url=url, error=str(result))


if __name__ == "__main__":
    asyncio.run(run_crawler(START_URLS))
