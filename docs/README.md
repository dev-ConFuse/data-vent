# data-vent

Hybrid retrieval & fusion engine for the ConFuse platform.

> This is the canonical README for the service; **all documentation lives in
> [`docs/`](./)** (this folder). See [index.md](index.md) for the docs index.

**Service**: `data-vent/` · **Port**: `3002` (`DATA_VENT_PORT`)
**Stack**: Rust (axum), FalkorDB, HNSW acceleration, multi-level caches

## Overview

`data-vent` answers retrieval queries by fusing vector similarity and graph
traversal:

- **Hybrid fusion** — WRRF fusion of vector + graph ranked lists, PageRank boost
- **Query decomposition** — multi-chunk query planning
- **Multi-level caches** — embedding / vector / algorithm / result caches
- **HNSW acceleration** — configurable index (m, ef) with cosine similarity
- **Graph algorithms** — BFS, PageRank, betweenness, WCC, s-paths

## Key API surface

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root status |
| `/health` | GET | Health + performance overview |
| `/metrics` | GET | Real-time metrics & cache hit rates |
| `/api/v1/retrieve` | POST | Hybrid retrieval |
| `/api/v1/retrieve/stream` | POST | Streaming retrieval |
| `/api/v1/retrieve/context` | POST | Context-aware retrieval |
| `/api/v1/retrieve/multi-hop` | POST | Multi-hop graph retrieval |
| `/api/v1/cache/invalidate` | POST | Cache invalidation |
| `/api/v1/hnsw/update` | POST | HNSW index update |

Full schemas: [api-reference.md](api-reference.md).

## How to run the microservice

```bash
cargo build
cargo run
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_VENT_PORT` | HTTP port | `3002` |
| `FALKORDB_HOST` / `FALKORDB_PORT` | FalkorDB (Redis protocol) | `localhost:6379` |
| `FALKORDB_VECTOR_DIMENSION` | Vector dimension | `768` |
| `FUSION_ALGORITHM` | Fusion strategy | `wrrf` |

## Testing

Tests for this service live in the central test module (this service contains
no tests):

```bash
cd ../ConFuse-test-module
pip install -r requirements.txt
pytest data_vent perf -v
```

## Documentation

| Document | Contents |
|----------|----------|
| [index.md](index.md) | Overview & docs index |
| [api-reference.md](api-reference.md) | Full endpoint reference |
| [architecture.md](architecture.md) | Internal architecture |
| [setup.md](setup.md) | Local development setup |
| [deployment.md](deployment.md) | Deployment guide |
| [performance.md](performance.md) | Performance characteristics |
