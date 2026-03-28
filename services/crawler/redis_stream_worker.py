from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from typing import Any

import httpx
import redis.asyncio as redis
import structlog
from dotenv import load_dotenv
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

REDIS_URL = os.getenv("LUMINA_REDIS_URL", "redis://localhost:6379/0")
REDIS_STREAM_NAME = os.getenv("LUMINA_REDIS_STREAM_NAME", "lumina:index:bulk")
REDIS_CONSUMER_GROUP = os.getenv("LUMINA_REDIS_CONSUMER_GROUP", "lumina-indexers")
REDIS_CONSUMER_NAME = os.getenv("LUMINA_REDIS_CONSUMER_NAME", socket.gethostname())
REDIS_READ_COUNT = int(os.getenv("LUMINA_REDIS_READ_COUNT", "50"))
REDIS_BLOCK_MS = int(os.getenv("LUMINA_REDIS_BLOCK_MS", "5000"))
GATEWAY_BULK_INDEX_URL = os.getenv("LUMINA_GATEWAY_BULK_INDEX_URL", "http://localhost:8000/index/bulk")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LUMINA_REQUEST_TIMEOUT_SECONDS", "30.0"))


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
logger = structlog.get_logger("lumina.redis_worker")


def _validate_config() -> None:
    if REDIS_READ_COUNT < 1:
        raise ValueError("LUMINA_REDIS_READ_COUNT must be >= 1")
    if REDIS_BLOCK_MS < 1:
        raise ValueError("LUMINA_REDIS_BLOCK_MS must be >= 1")
    if REQUEST_TIMEOUT_SECONDS <= 0:
        raise ValueError("LUMINA_REQUEST_TIMEOUT_SECONDS must be > 0")


async def _ensure_consumer_group(redis_client: redis.Redis) -> None:
    try:
        await redis_client.xgroup_create(name=REDIS_STREAM_NAME, groupname=REDIS_CONSUMER_GROUP, id="0", mkstream=True)
        logger.info(
            "redis_consumer_group_created",
            stream=REDIS_STREAM_NAME,
            group=REDIS_CONSUMER_GROUP,
        )
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def _post_bulk_documents(http_client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    ):
        with attempt:
            response = await http_client.post(GATEWAY_BULK_INDEX_URL, json=payload)
            response.raise_for_status()


async def run_worker() -> None:
    _validate_config()

    redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)

    async with httpx.AsyncClient(timeout=timeout) as http_client:
        await redis_client.ping()
        await _ensure_consumer_group(redis_client)

        logger.info(
            "redis_stream_worker_started",
            redis_url=REDIS_URL,
            stream=REDIS_STREAM_NAME,
            group=REDIS_CONSUMER_GROUP,
            consumer=REDIS_CONSUMER_NAME,
            gateway_bulk_url=GATEWAY_BULK_INDEX_URL,
        )

        try:
            while True:
                messages = await redis_client.xreadgroup(
                    groupname=REDIS_CONSUMER_GROUP,
                    consumername=REDIS_CONSUMER_NAME,
                    streams={REDIS_STREAM_NAME: ">"},
                    count=REDIS_READ_COUNT,
                    block=REDIS_BLOCK_MS,
                )
                if not messages:
                    continue

                for _, entries in messages:
                    for message_id, fields in entries:
                        payload_raw = fields.get("payload")
                        if payload_raw is None:
                            logger.error("redis_message_missing_payload", message_id=message_id)
                            await redis_client.xack(REDIS_STREAM_NAME, REDIS_CONSUMER_GROUP, message_id)
                            continue

                        try:
                            payload = json.loads(payload_raw)
                            documents = payload.get("documents")
                            if not isinstance(documents, list) or not documents:
                                raise ValueError("Payload must contain non-empty 'documents' list")
                            await _post_bulk_documents(http_client=http_client, payload=payload)
                            await redis_client.xack(REDIS_STREAM_NAME, REDIS_CONSUMER_GROUP, message_id)
                            logger.info(
                                "redis_message_processed",
                                message_id=message_id,
                                document_count=len(documents),
                            )
                        except Exception as exc:
                            logger.error("redis_message_failed", message_id=message_id, error=str(exc))
        finally:
            await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(run_worker())
