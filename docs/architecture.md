# Lumina Distributed Edition — Architecture Baseline

## Design analysis

> Note: вместо скрытого chain-of-thought здесь приведён краткий инженерный разбор, достаточный для ревью и реализации.

### 1. Hardware topology
- **Node A (GPU + storage):** x86_64 preferred, CUDA-capable GPU with >= 8 GiB VRAM, 64-byte cache-line assumption for host CPU data structures.
- **Node B (CPU crawler + gateway):** x86_64 preferred, NIC on 1 GbE LAN, NUMA kept single-socket where possible to reduce remote memory traffic.
- **ISA targets:** baseline x86_64-v3; optional AVX2/AVX-512 acceleration for tokenizer/re-ranking preprocessing. GPU embedding path is dominant, so CPU SIMD remains a secondary optimization.
- **False sharing mitigation:** per-worker queues and metrics counters should be sharded, padded to cache-line boundaries, and never share hot producer/consumer counters on the same line.

### 2. Memory model and ordering
- Python/FastAPI/Qdrant stack is process-oriented, so lock-free correctness is mostly delegated to runtime/library internals.
- For any future native hot-path queues: use SPSC ring buffers with `release` on producer index publication and `acquire` on consumer index observation.
- Informal proof sketch: payload write completes-before producer publishes tail; consumer observes tail with acquire semantics before reading payload, preventing stale reads under weak ordering.
- Cross-process boundaries use HTTP/gRPC and kernel socket buffers, which form explicit synchronization points.

### 3. Deterministic latency / jitter
Primary jitter sources:
- GPU warm-up and lazy kernel initialization.
- Python GC pauses and allocator churn.
- Major page faults from cold Qdrant segments or model weights.
- Linux scheduler context switches and shared IRQ handling.
- Network jitter on 1 GbE under buffer pressure.

Mitigations:
- Preload model weights and perform warm-up inference during startup.
- Enable hugepages where practical and keep quantized vectors resident in RAM.
- Pin latency-sensitive processes/containers to CPU sets.
- Prefer bounded async queues and backpressure over unbounded task spawning.
- Separate crawler workloads from gateway cores on Node B.

This baseline is **not interrupt-safe** in the kernel-space sense; it is user-space service software with best-effort latency control.

### 4. Branch analysis
- Hot search path should be mostly straight-line: normalize request -> embed -> hybrid retrieve -> rerank -> respond.
- Error branches are isolated at service boundaries.
- For future native parsers/token pipelines, favor table-driven token classification to reduce branch misprediction on heterogeneous documentation corpora.

### 5. Data-oriented design and layout
- Qdrant stores vectors in compact quantized segments; payload should keep only retrieval-critical metadata hot (`url`, `source`, timestamps, tags).
- Full document bodies remain off the hot vector path and can be stored compressed on disk/object storage.
- For future native extensions, require `alignas(64)` for queue headers and padded counters to avoid cache-line splits.

## Functional architecture

### Node A — “Brain and Storage”
- **Qdrant** for dense + sparse hybrid retrieval.
- **Inference Service** for embedding generation and reranker execution.
- Optional local object/blob storage for raw chunk bodies.

### Node B — “Collector and Gateway”
- **Crawler/ingestion workers** for Wikipedia dumps and IT documentation.
- **API Gateway** as the public entrypoint.
- **Frontend** served by Nginx/React.

### Wikipedia dump ingestion path
- `services/crawler/wikipedia_dump_parser.py` supports streaming XML (`iterparse`) and Parquet (`pyarrow.dataset` batch scanner) without loading full dumps into RAM.
- Gateway exposes `/index/bulk` for batch document ingestion (`documents: IndexRequest[]`) and writes both raw content (PostgreSQL) and vectors (Qdrant) in one request cycle.
- Crawler uses `/index/bulk` with configurable `LUMINA_INDEX_BULK_SIZE` to reduce per-chunk request overhead during high-volume indexing.
- Queue-based buffering (RabbitMQ / Redis Streams) remains an optional extension point between parser/crawler and gateway.

### Container build/startup optimization
- Service Dockerfiles use `uv` for dependency installation. This works both with legacy builder and BuildKit.
- Inference image preloads embedding and reranker models at build time to avoid repeated cold-start downloads on container startup.
- `docker-compose.yml` configures local BuildKit cache import/export for `inference` and `gateway` images (effective when BuildKit is enabled) and mounts a shared Hugging Face cache volume (`hf_cache`) for runtime reuse.

## Search contract
1. Client calls gateway `/search` with text query.
2. Gateway calls inference `/embed` and `/embed/sparse` on Node A.
3. Gateway calls Qdrant hybrid search using `prefetch` (dense + sparse) and `Fusion.RRF`; payload in Qdrant contains only hot metadata (`title`, `url`, `document_id`, `source`) without raw `content`.
4. Gateway performs bulk point lookup in raw storage (PostgreSQL) by `document_id` for retrieved candidates.
5. Retrieved raw texts are passed to reranker model.
6. Gateway returns ranked results plus metadata and snippets.

### Complexity and latency targets
- Embedding: `O(tokens)` per query.
- ANN retrieval in Qdrant: sublinear approximate search, target p95 < 50 ms on warm cache for 1M vectors.
- Reranking: `O(K * tokens)` with `K <= 10`, target p95 < 40 ms.
- End-to-end warm query target over LAN: **p95 < 150 ms**, **p99 < 250 ms**.

## Recommended Python notes
- Python 3.14 features such as subinterpreters are promising, but the ecosystem support is still uneven. Treat them as an optimization track, not a day-one dependency for crawler correctness.
- Zero-copy transfer should rely on memoryview/shared memory boundaries where possible, but HTML/Markdown parsing still incurs decode/tokenization costs.

## Schema baseline
| Field | Type | Index | Description |
| --- | --- | --- | --- |
| `id` | UUID/Text | Primary | Stable point ID (matches `document_id`) |
| `vector` | Vector(1024) | HNSW + int8 | Dense semantic embedding |
| `sparse_vector` | SparseVector | Inverted | Sparse lexical embedding |
| `metadata` | JSON | Payload | `title`, `url`, `document_id`, `source` (without `content`) |

Raw storage (PostgreSQL `raw_documents`) keeps:

| Field | Type | Index | Description |
| --- | --- | --- | --- |
| `document_id` | Text | Primary | Stable document/chunk ID |
| `content` | Text | No | Full chunk body used only for reranking/snippets |
| `created_at` | TIMESTAMPTZ | Optional | Ingestion timestamp |

## Qdrant configuration baseline
```yaml
indexing_threshold: 20000
optimizers:
  default_segment_number: 2
quantization_config:
  scalar:
    type: int8
    quantile: 0.99
    always_ram: true
```

## Risks and open points
- Real support maturity for Python 3.14 subinterpreters in Scrapy/parsing dependencies must be validated before committing the crawler design.
- Hybrid sparse retrieval support in Qdrant must be pinned to a concrete release and tested against IT-symbol-heavy corpora.
- BGE-M3 vector dimensionality and operational footprint should be fixed in config and validated empirically before schema freeze.
