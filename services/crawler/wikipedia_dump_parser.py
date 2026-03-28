from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import httpx

WIKI_XML_NAMESPACE = "http://www.mediawiki.org/xml/export-0.11/"
DEFAULT_BULK_SIZE = 64
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class WikipediaDocument:
    title: str
    url: str
    content: str


@dataclass(frozen=True, slots=True)
class ParserConfig:
    input_path: Path
    format: str
    language: str
    output_jsonl: Path | None
    post_to_gateway: bool
    gateway_bulk_url: str
    bulk_size: int
    timeout_seconds: float
    max_documents: int | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream parser for Wikipedia XML/Parquet dumps with optional bulk ingestion to gateway.",
    )
    parser.add_argument("--input", required=True, help="Path to dump file (.xml/.bz2/.gz/.xz/.parquet).")
    parser.add_argument(
        "--format",
        choices=("xml", "parquet"),
        required=True,
        help="Input dump format.",
    )
    parser.add_argument("--language", default="en", help="Wikipedia language subdomain, e.g. en, ru.")
    parser.add_argument("--output-jsonl", help="Optional output JSONL file path.")
    parser.add_argument(
        "--post-to-gateway",
        action="store_true",
        help="Send parsed records to gateway /index/bulk endpoint.",
    )
    parser.add_argument(
        "--gateway-bulk-url",
        default=os.getenv("LUMINA_GATEWAY_BULK_INDEX_URL", "http://localhost:8000/index/bulk"),
        help="Gateway bulk indexing URL.",
    )
    parser.add_argument("--bulk-size", type=int, default=DEFAULT_BULK_SIZE, help="Batch size for output and bulk indexing.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds for gateway requests.",
    )
    parser.add_argument("--max-documents", type=int, help="Optional cap on number of documents to process.")
    return parser


def _validate_args(args: argparse.Namespace) -> ParserConfig:
    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")

    if args.bulk_size < 1:
        raise ValueError("--bulk-size must be >= 1")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if args.max_documents is not None and args.max_documents < 1:
        raise ValueError("--max-documents must be >= 1")

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else None
    return ParserConfig(
        input_path=input_path,
        format=args.format,
        language=args.language,
        output_jsonl=output_jsonl,
        post_to_gateway=bool(args.post_to_gateway),
        gateway_bulk_url=str(args.gateway_bulk_url),
        bulk_size=int(args.bulk_size),
        timeout_seconds=float(args.timeout_seconds),
        max_documents=args.max_documents,
    )


def _open_text_stream(path: Path) -> TextIO:
    suffix = path.suffix.lower()
    if suffix == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    if suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    if suffix == ".xz":
        return lzma.open(path, mode="rt", encoding="utf-8", errors="replace")
    return path.open(mode="rt", encoding="utf-8", errors="replace")


def _normalize_wiki_title(title: str) -> str:
    return title.replace(" ", "_")


def _wiki_url(language: str, title: str) -> str:
    normalized = _normalize_wiki_title(title)
    return f"https://{language}.wikipedia.org/wiki/{normalized}"


def _extract_revision_text(page_element: ET.Element) -> str:
    revision = page_element.find(f"{{{WIKI_XML_NAMESPACE}}}revision")
    if revision is None:
        return ""
    text_element = revision.find(f"{{{WIKI_XML_NAMESPACE}}}text")
    if text_element is None or text_element.text is None:
        return ""
    return text_element.text.strip()


def _clean_wikitext(text: str) -> str:
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"==+\s*(.*?)\s*==+", r"\n\1\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_wikipedia_xml_documents(path: Path, language: str) -> Iterator[WikipediaDocument]:
    with _open_text_stream(path) as stream:
        context = ET.iterparse(stream, events=("end",))
        for _, element in context:
            if element.tag != f"{{{WIKI_XML_NAMESPACE}}}page":
                continue

            title_element = element.find(f"{{{WIKI_XML_NAMESPACE}}}title")
            namespace_element = element.find(f"{{{WIKI_XML_NAMESPACE}}}ns")

            title = title_element.text.strip() if title_element is not None and title_element.text else ""
            namespace = namespace_element.text.strip() if namespace_element is not None and namespace_element.text else ""

            if not title or namespace != "0":
                element.clear()
                continue

            raw_text = _extract_revision_text(element)
            cleaned_text = _clean_wikitext(raw_text)
            if cleaned_text:
                yield WikipediaDocument(
                    title=title,
                    url=_wiki_url(language=language, title=title),
                    content=cleaned_text,
                )

            element.clear()


def iter_wikipedia_parquet_documents(path: Path, language: str) -> Iterator[WikipediaDocument]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("Parquet support requires pyarrow. Install dependency: pyarrow==18.1.0") from exc

    dataset = ds.dataset(str(path), format="parquet")
    scanner = dataset.scanner(columns=["title", "text"])
    for record_batch in scanner.to_batches():
        titles = record_batch.column("title").to_pylist()
        texts = record_batch.column("text").to_pylist()
        for title_value, text_value in zip(titles, texts, strict=True):
            if title_value is None or text_value is None:
                continue
            title = str(title_value).strip()
            if not title:
                continue
            content = _clean_wikitext(str(text_value))
            if not content:
                continue
            yield WikipediaDocument(
                title=title,
                url=_wiki_url(language=language, title=title),
                content=content,
            )


def _iter_documents(config: ParserConfig) -> Iterator[WikipediaDocument]:
    if config.format == "xml":
        return iter_wikipedia_xml_documents(path=config.input_path, language=config.language)
    if config.format == "parquet":
        return iter_wikipedia_parquet_documents(path=config.input_path, language=config.language)
    raise ValueError(f"Unsupported format: {config.format}")


def _flush_jsonl(buffer: list[WikipediaDocument], stream: TextIO) -> None:
    for document in buffer:
        stream.write(
            json.dumps(
                {
                    "title": document.title,
                    "url": document.url,
                    "content": document.content,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    stream.flush()


def _flush_gateway(buffer: list[WikipediaDocument], client: httpx.Client, url: str) -> None:
    payload = {
        "documents": [
            {
                "title": document.title,
                "url": document.url,
                "content": document.content,
            }
            for document in buffer
        ]
    }
    response = client.post(url, json=payload)
    response.raise_for_status()


def run(config: ParserConfig) -> int:
    document_stream = _iter_documents(config)
    processed_documents = 0
    batch: list[WikipediaDocument] = []

    output_stream: TextIO | None = None
    gateway_client: httpx.Client | None = None

    try:
        if config.output_jsonl is not None:
            output_stream = config.output_jsonl.open(mode="w", encoding="utf-8")
        if config.post_to_gateway:
            gateway_client = httpx.Client(timeout=config.timeout_seconds)

        for document in document_stream:
            if config.max_documents is not None and processed_documents >= config.max_documents:
                break
            batch.append(document)
            if len(batch) >= config.bulk_size:
                if config.max_documents is not None:
                    remaining = config.max_documents - processed_documents
                    if remaining < len(batch):
                        batch = batch[:remaining]
                if output_stream is not None:
                    _flush_jsonl(batch, output_stream)
                if gateway_client is not None:
                    _flush_gateway(batch, gateway_client, config.gateway_bulk_url)
                processed_documents += len(batch)
                batch.clear()

        if batch:
            if config.max_documents is not None:
                remaining = config.max_documents - processed_documents
                if remaining <= 0:
                    batch = []
                elif remaining < len(batch):
                    batch = batch[:remaining]
        if batch:
            if output_stream is not None:
                _flush_jsonl(batch, output_stream)
            if gateway_client is not None:
                _flush_gateway(batch, gateway_client, config.gateway_bulk_url)
            processed_documents += len(batch)

        print(f"Processed documents: {processed_documents}", file=sys.stderr)
        return 0
    finally:
        if output_stream is not None:
            output_stream.close()
        if gateway_client is not None:
            gateway_client.close()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = _validate_args(args)
        return run(config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
