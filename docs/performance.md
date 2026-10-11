# Performance Guide - ConFuse Retrieval Engine

This guide details the performance architecture, baseline benchmarks, configuration tuning, and observability features for `data-vent`.

## Baseline Performance Targets

| Metric | Target | Verified Status |
|---|---|---|
| **Single-call latency** | <100ms @ 1000 RPS | Passed (`load_test.rs`) |
| **Multi-call latency** | <200ms @ 500 RPS | Passed (`load_test.rs`) |
| **Cache hit rate** | >60% on repeated queries | Passed (`cache_performance.rs`) |
| **Fast query decomposition** | <1ms for simple queries | Passed (`performance_regression.rs`) |
| **Algorithm execution** | <50ms per algorithm call | Passed (`algorithm_performance.rs`) |
| **Memory usage** | Bounded with zero memory leaks | Passed (`memory_leak.rs`) |
| **Concurrency** | 100+ concurrent requests without locks/deadlocks | Passed (`concurrent_access.rs`) |

---

## HNSW Tuning Recommendations

FalkorDB vector search uses HNSW indexes configured via `CALL db.idx.vector.configureNodeIndex`. Three pre-calibrated modes are supported:

1. **`low_latency`** (High Throughput / Real-time Agents):
   - `M = 16`, `efConstruction = 200`, `efRuntime = 10`, `similarityFunction = COSINE`
   - Optimal for interactive conversational turns and single-token search queries.
2. **`balanced`** (Default Production Profile):
   - `M = 24`, `efConstruction = 250`, `efRuntime = 25`, `similarityFunction = COSINE`
   - Balanced recall (>95%) with sub-10ms raw vector lookup time.
3. **`high_recall`** (Deep Research / Complex Multi-hop Reasoning):
   - `M = 32`, `efConstruction = 300`, `efRuntime = 50`, `similarityFunction = COSINE`
   - Maximizes precision for dense architectural concepts and multi-token queries.

Runtime adjustment endpoint:
```http
POST /api/v1/hnsw/update
Content-Type: application/json

{
  "mode": "high_recall",
  "falkordb_graph_name": "confuse_graph"
}
```

---

## Multi-Level Caching (4 Tiers)

`data-vent` implements a thread-safe 4-tier LRU cache:
- **L1 Query Embeddings**: Maps raw text -> 2048d embedding vector (TTL: 3600s). Avoids redundant NVIDIA NIM inference calls.
- **L2 Vector Search Results**: Maps `(graph_name, vector_hash, limit)` -> top-K chunk candidates (TTL: 300s).
- **L3 Graph Algorithm Computations**: Caches PageRank, Betweenness Centrality, and WCC component assignments (TTL: 1800s).
- **L4 Aggregated Responses**: Stores finalized WRRF candidate ranking for identical query contexts (TTL: 120s).

### Cache Warming & Auto-Tuning
- `warm_frequent_queries`: Ingests top frequent queries and pre-populates L1 embeddings and L2 vector results.
- `auto_tune_cache_sizes`: Automatically expands vector and embedding cache capacities when hit rates fall below the target threshold (e.g. 60%).

---

## Observability & Metrics

1. **Metrics Endpoint** (`GET /metrics`):
   Returns real-time hit rates across all 4 cache tiers along with algorithm enable flags:
   ```json
   {
     "cache_performance": {
       "embedding_hit_rate": 0.85,
       "vector_hit_rate": 0.72,
       "algorithm_hit_rate": 0.60,
       "result_hit_rate": 0.65
     },
     "service_info": {
       "version": "0.3.0",
       "algorithms_enabled": {
         "bfs": true,
         "pagerank": true,
         "betweenness": false,
         "wcc": false,
         "sppaths": false
       }
     }
   }
   ```

2. **Health Check** (`GET /health`):
   Provides liveness status and performance overview for load balancers.

3. **Performance Logging & Alerting**:
   Structured logs (`PERFORMANCE_LOG`) emit operation latency and cache hit status. `PerformanceAlerts` automatically emits warnings whenever latency exceeds 100ms or hit rates fall below 60%.
