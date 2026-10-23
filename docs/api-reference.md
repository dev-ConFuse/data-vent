# data-vent API Reference

## Overview
This document outlines the HTTP API endpoints exposed by `data-vent`.

## HTTP Endpoints

### `GET /health`
Returns the health status of the service.
- **Response**: `200 OK` with service status

### `POST /api/v1/retrieve`
Standard retrieval endpoint for intelligent search.
- **Request Body**:
  ```json
  {
    "intent": "string",
    "keywords": ["string"],
    "limit": 10,
    "falkordb_graph_name": "optional string"
  }
  ```
- **Response**: JSON with search results and metadata, plus a `guidance` block
  (see [Research guidance](#research-guidance)).

### `POST /api/v1/retrieve/stream`
Streaming retrieval endpoint for real-time results via Server-Sent Events.
- **Request Body**: Same as `/api/v1/retrieve`
- **Response**: SSE stream with progressive results. The terminal `done` event
  carries the same `guidance` block as the batch response.

### `GET /`
Root endpoint for basic health monitoring.
- **Response**: `{"status": "ok"}`

## Research guidance

Every retrieval response (batch, SSE `done` event, and `/retrieve/context`)
includes a `guidance` object. Retrieval is deliberately not a one-shot answer
service: this block tells the caller how much of the question is backed by
evidence and what to do next, so agents keep reasoning instead of answering
from a single pass. `client-connector` renders it as the `[GUIDANCE]` block on
every MCP tool result.

```json
"guidance": {
  "completeness": 0.42,
  "status": "partial",
  "covered_sub_queries": 2,
  "total_sub_queries": 5,
  "uncovered_sub_queries": ["how X persists state"],
  "sub_query_coverage": [
    { "sub_query": "how X persists state", "intent": "relationship", "hits": 0, "errored": false }
  ],
  "next_actions": [
    {
      "tool": "search_knowledge",
      "intent": "how X persists state",
      "keywords": ["X", "persist"],
      "why": "sub-query produced no evidence — retry with tighter terms before answering"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `completeness` | `0.0`–`1.0`. Sub-query coverage blended with source corroboration. |
| `status` | `ready_to_answer` \| `partial` \| `needs_more_evidence` \| `not_covered`. |
| `covered_sub_queries` / `total_sub_queries` | How many decomposed sub-queries returned evidence. |
| `uncovered_sub_queries` | Sub-queries with zero hits — the concrete evidence gaps. |
| `sub_query_coverage` | Per-sub-query hit counts (`hits`), including errored searches. |
| `next_actions` | Ready-to-execute follow-ups. `intent` + `keywords` map 1:1 onto the `client-connector` `search_knowledge` arguments. |

**Status semantics.** `ready_to_answer` is returned only when nothing is
missing: every sub-query matched *and* at least two independent sources
corroborate the evidence. A single source yields `partial` at best, which is the
uncorroborated case being flagged rather than asserted.

**Note** that `completion_reached` and `guidance.completeness` measure different
things: the former is the fusion engine's saturation flag (enough high-scoring
chunks fused, or the result budget filled), the latter is how much of the
*question* is covered by evidence. Prefer `guidance` when deciding whether to
retrieve more.
