mod infra;
mod services;

use axum::{
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use envconfig::Envconfig;
use serde::Deserialize;
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{error, info};

use infra::Config;
use services::cache::MultiLevelCache;
use services::context_aware_retrieval::{ContextAwareRetriever, ContextOptions};
use services::graph_algorithms::PathResult;
use services::intelligent_retriever::IntelligentRetriever;
use services::parallel_search::ParallelSearchDispatcher;
use services::query_decomposer::{QueryChunk, QueryDecomposer};
use services::result_aggregator::{ResearchGuidance, ResultAggregator};
use services::vector_search::FalkorDbClient;

#[derive(Clone)]
struct AppState {
    retriever: Arc<IntelligentRetriever>,
    decomposer: Arc<QueryDecomposer>,
    dispatcher: Arc<ParallelSearchDispatcher>,
    aggregator: Arc<ResultAggregator>,
    context_retriever: Arc<ContextAwareRetriever>,
    cache: Arc<MultiLevelCache>,
    default_graph_name: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Setup panic handler to catch and log panics
    std::panic::set_hook(Box::new(|panic_info| {
        tracing::error!("Panic occurred: {}", panic_info);
    }));

    // Load config
    tracing::info!("Loading environment variables...");
    dotenvy::from_filename_override(".map.env").ok();
    dotenvy::from_filename_override(".secret.env").ok();
    dotenvy::from_filename_override(".env.local").ok();
    tracing::info!("Environment variables loaded");

    infra::logging::init_tracing();

    info!(service = "data-vent", "Starting service");
    // Fail fast when required env config is missing — values live in
    // .map.env / .secret.env (no in-code defaults; see .cline/rules/env-policy.md)
    let config = Config::init_from_env().map_err(|e| {
        anyhow::anyhow!(
            "Missing or invalid required environment configuration (see .map.env / .secret.env): {}",
            e
        )
    })?;

    // Initialize FalkorDB (optional - allow service to start without it)
    let falkordb_client = match FalkorDbClient::new(&config).await {
        Ok(client) => {
            info!("FalkorDB connected successfully");
            // NOTE: we deliberately do NOT pre-create an ambient default graph
            // here. Historically this called initialize_indexes() with
            // FALKORDB_GRAPH_NAME ("knowledge-layer"), which created a shared
            // graph at startup that silently accumulated every tenant's data.
            // Graphs are per-tenant (graph-<x-user-id>) and are created on
            // first write; there is no shared default any more.
            client
        }
        Err(e) => {
            error!(
                "Failed to connect to FalkorDB: {}. Service will start in degraded mode.",
                e
            );
            FalkorDbClient::new_dummy()
        }
    };

    // Initialize Multi-Level Cache
    let cache = Arc::new(MultiLevelCache::new(&config));

    // Initialize Services
    let retriever = Arc::new(IntelligentRetriever::new(falkordb_client, cache.clone(), &config));
    let decomposer = Arc::new(QueryDecomposer::new(config.pipeline_max_query_chunks));
    let dispatcher = Arc::new(ParallelSearchDispatcher::new(
        config.pipeline_per_chunk_timeout,
        config.pipeline_vector_top_k,
        config.pipeline_dfs_depth,
        config.pipeline_dfs_min_relevance,
        config.pipeline_dfs_max_results,
        config.enable_bfs,
    ));
    let aggregator = Arc::new(ResultAggregator::new(
        config.pipeline_max_total_results,
        config.pipeline_dfs_min_relevance,
        3,
        config.pipeline_vector_weight,
        config.pipeline_graph_weight,
        config.pipeline_cross_chunk_weight,
        config.fusion_algorithm.clone(),
        config.wrrf_k,
        config.pagerank_boost_weight,
    ));
    let context_retriever = Arc::new(ContextAwareRetriever::new(
        retriever.clone(),
        decomposer.clone(),
        dispatcher.clone(),
        aggregator.clone(),
        retriever.graph_algo.clone(),
    ));

    let state = AppState {
        retriever: retriever.clone(),
        decomposer: decomposer.clone(),
        dispatcher: dispatcher.clone(),
        aggregator: aggregator.clone(),
        context_retriever: context_retriever.clone(),
        cache: cache.clone(),
        default_graph_name: config.falkordb_graph_name.clone(),
    };

    // Define REST routes
    let app = Router::new()
        .route("/", get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }))
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_handler))
        .route("/api/v1/retrieve", post(retrieve_handler))
        .route("/api/v1/retrieve/stream", post(retrieve_stream_handler))
        .route("/api/v1/retrieve/context", post(context_retrieve_handler))
        .route("/api/v1/retrieve/multi-hop", post(multi_hop_handler))
        .route("/api/v1/cache/invalidate", post(cache_invalidate_handler))
        .route("/api/v1/hnsw/update", post(update_hnsw_handler))
        .layer(axum::middleware::from_fn(
            infra::logging::access_log_middleware,
        ))
        .with_state(state);

    // Use PORT from environment (Render) or fall back to config
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(config.app_port);

    let addr: SocketAddr = format!("{}:{}", config.host, port).parse()?;

    info!("HTTP server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Json<serde_json::Value> {
    let cache_stats = state.cache.get_cache_stats().await;
    Json(serde_json::json!({
        "status": "healthy",
        "service": "data-vent",
        "version": "0.3.0",
        "performance": {
            "cache_hit_rate": cache_stats.vector_hit_rate,
            "embedding_hit_rate": cache_stats.embedding_hit_rate,
            "algorithm_hit_rate": cache_stats.algorithm_hit_rate,
            "result_hit_rate": cache_stats.result_hit_rate,
        },
        "algorithms": ["bfs", "sppaths", "pagerank", "betweenness", "wcc"],
        "fusion": "wrrf",
        "cache": "4-tier-lru"
    }))
}

async fn metrics_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Json<serde_json::Value> {
    let cache_stats = state.cache.get_cache_stats().await;
    Json(serde_json::json!({
        "cache_performance": {
            "embedding_hit_rate": cache_stats.embedding_hit_rate,
            "vector_hit_rate": cache_stats.vector_hit_rate,
            "algorithm_hit_rate": cache_stats.algorithm_hit_rate,
            "result_hit_rate": cache_stats.result_hit_rate,
        },
        "service_info": {
            "version": "0.3.0",
            "algorithms_enabled": {
                "bfs": true,
                "pagerank": true,
                "betweenness": false,
                "wcc": false,
                "sppaths": false,
            }
        }
    }))
}

/// Serialize retrieval-side research guidance for the calling agent.
///
/// Every retrieve response carries this block so agents can see how much of the
/// question is covered by evidence and what to do next instead of treating a
/// single retrieval pass as a finished answer.
fn guidance_json(g: &ResearchGuidance) -> serde_json::Value {
    serde_json::json!({
        "completeness": g.completeness,
        "status": g.status,
        "covered_sub_queries": g.covered_sub_queries,
        "total_sub_queries": g.total_sub_queries,
        "uncovered_sub_queries": g.uncovered_sub_queries,
        "sub_query_coverage": g
            .sub_query_coverage
            .iter()
            .map(|c| serde_json::json!({
                "sub_query": c.sub_query,
                "intent": c.intent,
                "hits": c.hits,
                "errored": c.errored,
            }))
            .collect::<Vec<_>>(),
        "next_actions": g
            .next_actions
            .iter()
            .map(|a| serde_json::json!({
                "tool": a.tool,
                "intent": a.intent,
                "keywords": a.keywords,
                "why": a.why,
            }))
            .collect::<Vec<_>>(),
    })
}

#[derive(Deserialize, Debug, Clone)]
pub struct AlgorithmOptions {
    #[serde(default = "default_true")]
    pub enable_bfs: bool,
    #[serde(default = "default_true")]
    pub enable_pagerank: bool,
    #[serde(default)]
    pub enable_betweenness: bool,
    #[serde(default)]
    pub enable_wcc: bool,
    #[serde(default)]
    pub enable_sppaths: bool,
}
fn default_true() -> bool {
    true
}

#[derive(Deserialize, Debug, Clone)]
pub struct HnswRequestConfig {
    #[serde(default = "default_hnsw_mode")]
    pub mode: String,
}
fn default_hnsw_mode() -> String {
    "balanced".to_string()
}

#[derive(Deserialize)]
struct RetrieveRequest {
    intent: String,
    keywords: Vec<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    _source_ids: Option<Vec<String>>,
    pub falkordb_graph_name: Option<String>,
    pub algorithms: Option<AlgorithmOptions>,
    pub hnsw_config: Option<HnswRequestConfig>,
}
fn default_limit() -> usize {
    20
}

#[derive(Deserialize)]
struct ContextRetrieveRequest {
    intent: String,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    pub falkordb_graph_name: Option<String>,
    pub context: ContextOptions,
}

#[derive(Deserialize)]
struct MultiHopRequest {
    source_chunk_id: String,
    target_chunk_id: String,
    #[serde(default = "default_max_hops")]
    max_hops: usize,
    #[serde(default)]
    relationship_types: Vec<String>,
    pub falkordb_graph_name: Option<String>,
}
fn default_max_hops() -> usize {
    3
}

#[derive(Deserialize)]
struct CacheInvalidateRequest {
    pub graph_name: Option<String>,
}

async fn retrieve_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<RetrieveRequest>,
) -> Json<serde_json::Value> {
    let start = std::time::Instant::now();
    let decomp_res = state.decomposer.decompose(&req.intent).await;

    let mut all_chunks = decomp_res.chunks;
    for kw in req.keywords {
        all_chunks.push(QueryChunk {
            text: kw.clone(),
            intent: "entity_lookup".to_string(),
            weight: 1.0,
            original_span: (0, 0),
            tokens: kw.split_whitespace().map(|s| s.to_string()).collect(),
        });
    }

    // Graph scope is ALWAYS explicit. There is deliberately no ambient default:
    // falling back to a shared graph (historically "knowledge-layer") silently
    // mixed every tenant's data into one graph. The caller scopes via
    // x-user-id, or names the graph explicitly in the request.
    let graph_name = match headers
        .get("x-user-id")
        .and_then(|h| h.to_str().ok())
        .map(str::trim)
        .filter(|u| !u.is_empty())
    {
        Some(user_id) => format!("graph-{}", user_id),
        None => match req
            .falkordb_graph_name
            .as_deref()
            .map(str::trim)
            .filter(|g| !g.is_empty())
        {
            Some(graph) => graph.to_string(),
            None => {
                tracing::warn!("retrieve rejected: no x-user-id header and no explicit graph name");
                return Json(serde_json::json!({
                    "error": "graph scope required",
                    "detail": "provide an x-user-id header (preferred) or an explicit falkordb_graph_name",
                }));
            }
        },
    };

    if let Some(ref hnsw) = req.hnsw_config {
        let hnsw_cfg = services::vector_search::HNSWConfig::from_mode(&hnsw.mode);
        let _ = state.retriever.update_hnsw_parameters(&graph_name, &hnsw_cfg).await;
    }

    let search_res = state.dispatcher.dispatch(&graph_name, all_chunks, &state.retriever).await;

    // Optional PageRank & Betweenness Centrality boosting
    let candidate_ids: Vec<String> = search_res
        .chunk_results
        .iter()
        .flat_map(|cr| {
            cr.vector_results
                .iter()
                .chain(cr.graph_results.iter())
                .map(|r| r.chunk_id.clone())
        })
        .collect();

    let pagerank_scores = if req.algorithms.as_ref().map_or(true, |a| a.enable_pagerank) {
        state.retriever.get_pagerank_scores(&graph_name, &candidate_ids).await
    } else {
        HashMap::new()
    };

    let betweenness_scores = if req.algorithms.as_ref().map_or(false, |a| a.enable_betweenness) {
        state.retriever.get_betweenness_scores(&graph_name, &candidate_ids, 64).await
    } else {
        HashMap::new()
    };

    let agg_res = state.aggregator.aggregate_with_centrality(
        search_res,
        &req.intent,
        req.limit,
        &pagerank_scores,
        &betweenness_scores,
    );

    let mut results = vec![];
    for c in agg_res.chunks {
        results.push(serde_json::json!({
            "chunk_id": c.chunk_id,
            "content": c.content,
            "final_score": c.final_score,
            "vector_score": c.vector_score,
            "graph_score": c.graph_score,
            "cross_chunk_boost": c.cross_chunk_boost,
            "pagerank_score": c.pagerank_score,
            "chunk_type": c.chunk_type,
            "source_id": c.source_id,
            "document_id": c.document_id,
            "metadata": c.metadata,
            "matched_by_chunks": c.matched_by_chunks,
        }));
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Json(serde_json::json!({
        "results": results,
        "total_results": agg_res.total_results,
        "unique_sources": agg_res.unique_sources,
        "vector_matches": agg_res.vector_matches,
        "graph_matches": agg_res.graph_matches,
        "completion_reached": agg_res.completion_reached,
        "total_time_ms": elapsed,
        "guidance": guidance_json(&agg_res.guidance),
    }))
}

async fn retrieve_stream_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<RetrieveRequest>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let (mut tx, rx) = futures::channel::mpsc::channel::<Result<Event, Infallible>>(100);

    tokio::spawn(async move {
        let start = std::time::Instant::now();
        let decomp_res = state.decomposer.decompose(&req.intent).await;

        let mut all_chunks = decomp_res.chunks;
        for kw in &req.keywords {
            all_chunks.push(QueryChunk {
                text: kw.clone(),
                intent: "entity_lookup".to_string(),
                weight: 1.0,
                original_span: (0, 0),
                tokens: kw.split_whitespace().map(|s| s.to_string()).collect(),
            });
        }

        let chunk_infos: Vec<serde_json::Value> = all_chunks
            .iter()
            .map(|c| {
                serde_json::json!({
                    "text": c.text,
                    "intent": c.intent,
                    "weight": c.weight,
                })
            })
            .collect();

        let _ = tx.try_send(Ok(Event::default().event("query_decomposition").data(
            serde_json::json!({
                "query_chunks": chunk_infos,
                "decomposition_time_ms": decomp_res.decomposition_time_ms,
            })
            .to_string(),
        )));

        // No ambient default graph: scope must come from x-user-id or the request.
        let graph_name = match headers
            .get("x-user-id")
            .and_then(|h| h.to_str().ok())
            .map(str::trim)
            .filter(|u| !u.is_empty())
        {
            Some(user_id) => format!("graph-{}", user_id),
            None => match req
                .falkordb_graph_name
                .as_deref()
                .map(str::trim)
                .filter(|g| !g.is_empty())
            {
                Some(graph) => graph.to_string(),
                None => {
                    tracing::warn!("stream rejected: no x-user-id header and no explicit graph name");
                    let _ = tx.try_send(Ok(Event::default().event("error").data(
                        serde_json::json!({
                            "error": "graph scope required",
                            "detail": "provide an x-user-id header (preferred) or an explicit falkordb_graph_name",
                        })
                        .to_string(),
                    )));
                    return;
                }
            },
        };

        let search_res = state.dispatcher.dispatch(&graph_name, all_chunks, &state.retriever).await;

        let candidate_ids: Vec<String> = search_res
            .chunk_results
            .iter()
            .flat_map(|cr| {
                cr.vector_results
                    .iter()
                    .chain(cr.graph_results.iter())
                    .map(|r| r.chunk_id.clone())
            })
            .collect();

        let pagerank_scores = if req.algorithms.as_ref().map_or(true, |a| a.enable_pagerank) {
            state.retriever.get_pagerank_scores(&graph_name, &candidate_ids).await
        } else {
            HashMap::new()
        };

        let agg_res = state.aggregator.aggregate_with_centrality(
            search_res,
            &req.intent,
            req.limit,
            &pagerank_scores,
            &HashMap::new(),
        );

        for (idx, c) in agg_res.chunks.iter().enumerate() {
            let chunk_data = serde_json::json!({
                "index": idx,
                "chunk_id": c.chunk_id,
                "content": c.content,
                "final_score": c.final_score,
                "vector_score": c.vector_score,
                "graph_score": c.graph_score,
                "cross_chunk_boost": c.cross_chunk_boost,
                "pagerank_score": c.pagerank_score,
                "chunk_type": c.chunk_type,
                "source_id": c.source_id,
                "document_id": c.document_id,
                "metadata": c.metadata,
                "matched_by_chunks": c.matched_by_chunks,
            });

            let _ = tx.try_send(Ok(Event::default().event("chunk_result").data(chunk_data.to_string())));
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let _ = tx.try_send(Ok(Event::default().event("done").data(
            serde_json::json!({
                "total_results": agg_res.total_results,
                "unique_sources": agg_res.unique_sources,
                "vector_matches": agg_res.vector_matches,
                "graph_matches": agg_res.graph_matches,
                "completion_reached": agg_res.completion_reached,
                "total_time_ms": elapsed,
                "guidance": guidance_json(&agg_res.guidance),
            })
            .to_string(),
        )));
    });

    Sse::new(rx).keep_alive(KeepAlive::default())
}

async fn context_retrieve_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ContextRetrieveRequest>,
) -> Json<serde_json::Value> {
    // Graph scope is ALWAYS explicit. There is deliberately no ambient default:
    // falling back to a shared graph (historically "knowledge-layer") silently
    // mixed every tenant's data into one graph. The caller scopes via
    // x-user-id, or names the graph explicitly in the request.
    let graph_name = match headers
        .get("x-user-id")
        .and_then(|h| h.to_str().ok())
        .map(str::trim)
        .filter(|u| !u.is_empty())
    {
        Some(user_id) => format!("graph-{}", user_id),
        None => match req
            .falkordb_graph_name
            .as_deref()
            .map(str::trim)
            .filter(|g| !g.is_empty())
        {
            Some(graph) => graph.to_string(),
            None => {
                tracing::warn!("retrieve rejected: no x-user-id header and no explicit graph name");
                return Json(serde_json::json!({
                    "error": "graph scope required",
                    "detail": "provide an x-user-id header (preferred) or an explicit falkordb_graph_name",
                }));
            }
        },
    };

    let agg_res = state
        .context_retriever
        .retrieve_with_context(&graph_name, &req.intent, req.keywords, req.context, req.limit)
        .await;

    let mut results = vec![];
    for c in agg_res.chunks {
        results.push(serde_json::json!({
            "chunk_id": c.chunk_id,
            "content": c.content,
            "final_score": c.final_score,
            "vector_score": c.vector_score,
            "graph_score": c.graph_score,
            "cross_chunk_boost": c.cross_chunk_boost,
            "pagerank_score": c.pagerank_score,
            "chunk_type": c.chunk_type,
            "source_id": c.source_id,
            "document_id": c.document_id,
            "metadata": c.metadata,
            "matched_by_chunks": c.matched_by_chunks,
        }));
    }

    Json(serde_json::json!({
        "results": results,
        "total_results": agg_res.total_results,
        "unique_sources": agg_res.unique_sources,
        "vector_matches": agg_res.vector_matches,
        "graph_matches": agg_res.graph_matches,
        "completion_reached": agg_res.completion_reached,
        "total_time_ms": agg_res.aggregation_time_ms,
        "guidance": guidance_json(&agg_res.guidance),
    }))
}

async fn multi_hop_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<MultiHopRequest>,
) -> Json<serde_json::Value> {
    let start = std::time::Instant::now();
    // Graph scope is ALWAYS explicit. There is deliberately no ambient default:
    // falling back to a shared graph (historically "knowledge-layer") silently
    // mixed every tenant's data into one graph. The caller scopes via
    // x-user-id, or names the graph explicitly in the request.
    let graph_name = match headers
        .get("x-user-id")
        .and_then(|h| h.to_str().ok())
        .map(str::trim)
        .filter(|u| !u.is_empty())
    {
        Some(user_id) => format!("graph-{}", user_id),
        None => match req
            .falkordb_graph_name
            .as_deref()
            .map(str::trim)
            .filter(|g| !g.is_empty())
        {
            Some(graph) => graph.to_string(),
            None => {
                tracing::warn!("retrieve rejected: no x-user-id header and no explicit graph name");
                return Json(serde_json::json!({
                    "error": "graph scope required",
                    "detail": "provide an x-user-id header (preferred) or an explicit falkordb_graph_name",
                }));
            }
        },
    };

    let paths: Vec<PathResult> = state
        .context_retriever
        .multi_hop_reasoning(
            &graph_name,
            &req.source_chunk_id,
            &req.target_chunk_id,
            req.max_hops,
            req.relationship_types,
        )
        .await;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Json(serde_json::json!({
        "paths": paths,
        "total_paths": paths.len(),
        "source_chunk_id": req.source_chunk_id,
        "target_chunk_id": req.target_chunk_id,
        "total_time_ms": elapsed,
    }))
}

async fn cache_invalidate_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<CacheInvalidateRequest>,
) -> Json<serde_json::Value> {
    let Some(graph) = req
        .graph_name
        .as_deref()
        .map(str::trim)
        .filter(|g| !g.is_empty())
        .map(str::to_string)
    else {
        tracing::warn!("cache invalidate rejected: no explicit graph name");
        return Json(serde_json::json!({
            "error": "graph scope required",
            "detail": "provide an explicit graph_name; there is no shared default graph",
        }));
    };
    state.cache.invalidate_graph(&graph).await;
    Json(serde_json::json!({
        "status": "success",
        "message": format!("Cache invalidated for graph: {}", graph),
    }))
}

#[derive(Deserialize)]
struct HnswUpdateRequest {
    pub mode: String, // "high_recall", "balanced", "low_latency"
    pub falkordb_graph_name: Option<String>,
}

async fn update_hnsw_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<HnswUpdateRequest>,
) -> Json<serde_json::Value> {
    let config = services::vector_search::HNSWConfig::from_mode(&req.mode);
    let Some(graph_name) = req
        .falkordb_graph_name
        .as_deref()
        .map(str::trim)
        .filter(|g| !g.is_empty())
        .map(str::to_string)
    else {
        tracing::warn!("hnsw update rejected: no explicit graph name");
        return Json(serde_json::json!({
            "error": "graph scope required",
            "detail": "provide an explicit falkordb_graph_name; there is no shared default graph",
        }));
    };

    match state.retriever.update_hnsw_parameters(&graph_name, &config).await {
        Ok(_) => Json(serde_json::json!({
            "status": "success",
            "mode": req.mode,
            "config": config
        })),
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "error": e.to_string()
        }))
    }
}

