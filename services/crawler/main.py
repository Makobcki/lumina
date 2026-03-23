from __future__ import annotations

import logging
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

GATEWAY_INDEX_URL = "http://localhost:8000/index"
REQUEST_TIMEOUT_SECONDS = 30.0
CHUNK_SIZE = 1000
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


def fetch_and_clean(url: str, client: httpx.Client) -> tuple[str, str]:
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag_name in REMOVABLE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
    cleaned_lines = [line for line in lines if line]
    cleaned_text = "\n".join(cleaned_lines)
    return title, cleaned_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [normalized.strip()] if normalized.strip() else []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_length = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_length
        if buffer:
            chunks.append("\n\n".join(buffer).strip())
            buffer = []
            buffer_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush_buffer()
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end].strip())
                start = end
            continue

        projected = buffer_length + len(paragraph) + (2 if buffer else 0)
        if projected > chunk_size:
            flush_buffer()

        buffer.append(paragraph)
        buffer_length += len(paragraph) + (2 if len(buffer) > 1 else 0)

    flush_buffer()
    return [chunk for chunk in chunks if chunk]


def index_chunks(url: str, title: str, chunks: Iterable[str], client: httpx.Client) -> None:
    for index, chunk in enumerate(chunks, start=1):
        payload = {
            "title": f"{title} [chunk {index}]",
            "url": url,
            "content": chunk,
        }
        response = client.post(GATEWAY_INDEX_URL, json=payload)
        response.raise_for_status()
        logger.info("Indexed %s chunk %s", url, index)


def crawl_and_index(url: str, fetch_client: httpx.Client, gateway_client: httpx.Client) -> None:
    logger.info("Fetching %s", url)
    title, text = fetch_and_clean(url, fetch_client)
    chunks = chunk_text(text)
    logger.info("Prepared %s chunks for %s", len(chunks), url)
    index_chunks(url=url, title=title, chunks=chunks, client=gateway_client)


if __name__ == "__main__":
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as fetch_client, httpx.Client(
        timeout=REQUEST_TIMEOUT_SECONDS
    ) as gateway_client:
        for url in START_URLS:
            try:
                crawl_and_index(url=url, fetch_client=fetch_client, gateway_client=gateway_client)
            except httpx.HTTPError as exc:
                logger.error("Failed to process %s: %s", url, exc)
