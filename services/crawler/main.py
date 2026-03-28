from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

GATEWAY_INDEX_URL = "http://localhost:8000/index"
REQUEST_TIMEOUT_SECONDS = 30.0
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("lumina.crawler")


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
            logger.warning("Could not fetch robots.txt for %s", domain)

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
        logger.warning("robots.txt disallows crawling %s", url)
        return None

    await politeness_policy.wait_turn(url)
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
    async def index_one(chunk_index: int, chunk: str) -> None:
        payload = {
            "title": f"{title} [chunk {chunk_index}]",
            "url": url,
            "content": chunk,
        }
        async with semaphore:
            response = await client.post(GATEWAY_INDEX_URL, json=payload)
            response.raise_for_status()
            logger.info("Indexed %s chunk %s", url, chunk_index)

    tasks = [index_one(chunk_index=index, chunk=chunk) for index, chunk in enumerate(chunks, start=1)]
    if tasks:
        await asyncio.gather(*tasks)


async def crawl_and_index(
    url: str,
    fetch_client: httpx.AsyncClient,
    gateway_client: httpx.AsyncClient,
    politeness_policy: PolitenessPolicy,
    indexing_semaphore: asyncio.Semaphore,
) -> None:
    logger.info("Fetching %s", url)
    result = await fetch_and_clean(url=url, client=fetch_client, politeness_policy=politeness_policy)
    if result is None:
        return

    title, text = result
    chunks = chunk_text(text)
    logger.info("Prepared %s chunks for %s", len(chunks), url)
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
            logger.error("Failed to process %s: %s", url, result)


if __name__ == "__main__":
    asyncio.run(run_crawler(START_URLS))
