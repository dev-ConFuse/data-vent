use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::infra::Config;
use crate::services::cache::{MultiLevelCache, VectorCacheKey};
use crate::services::graph_algorithms::{GraphAlgorithms, PathResult};
use crate::services::vector_search::FalkorDBClient;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub content: String,
    pub score: f64,
    pub metadata: serde_json::Value,
    pub source: String,
    pub chunk_type: String,
    pub source_id: String,
    pub document_id: String,
    pub depth: i32,
    #[serde(default)]
    pub matched_by_chunks: Vec<String>,
}

pub struct IntelligentRetriever {
    pub falkordb_client: FalkorDBClient,
    pub graph_algo: GraphAlgorithms,
    pub cache: Arc<MultiLevelCache>,
    http_client: reqwest::Client,
    nim_api_key: String,
    nim_base_url: String,
    embedding_model: String,
}

impl IntelligentRetriever {
    pub fn new(falkordb_client: FalkorDBClient, cache: Arc<MultiLevelCache>, config: &Config) -> Self {
        let graph_algo = GraphAlgorithms::new(falkordb_client.clone());
        Self {
            falkordb_client,
            graph_algo,
            cache,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_default(),
            nim_api_key: config.nvidia_nim_api_key.clone().unwrap_or_default(),
            nim_base_url: config.nvidia_nim_base_url.clone(),
            embedding_model: config.default_embedding_model.clone(),
        }
    }

    pub async fn _close(&self) {
        let _ = self.falkordb_client._close().await;
    }

    pub async fn update_hnsw_parameters(&self, graph_name: &str, config: &crate::services::vector_search::HNSWConfig) -> anyhow::Result<()> {
        self.falkordb_client.update_hnsw_parameters(graph_name, config).await
    }

    pub async fn vectorize_query(&self, query: &str) -> Vec<f64> {
        if query.trim().is_empty() {
            return vec![];
        }

        // Check L1 Embedding Cache
        if let Some(cached_vec) = self.cache.get_embedding(query).await {
            return cached_vec;
        }

        if self.nim_api_key.is_empty() {
            warn!("NVIDIA_NIM_API_KEY not set");
            return vec![];
        }

        let url = format!("{}/v1/embeddings", self.nim_base_url);

        let body = serde_json::json!({
            "input": [query],
            "model": self.embedding_model,
            "input_type": "query"
        });

        // Transient network failures (connection resets on a slow link) must not
        // silently zero out the search: retry before giving up, and surface the
        // HTTP status when the API rejects us - previously a non-200 was swallowed
        // with no log at all.
        let mut last_err = String::new();
        for attempt in 1..=3 {
            match self
                .http_client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.nim_api_key))
                .json(&body)
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        let txt = resp.text().await.unwrap_or_default();
                        last_err = format!("HTTP {}: {}", status, txt.chars().take(160).collect::<String>());
                        warn!("NIM embeddings call failed (attempt {}/3): {}", attempt, last_err);
                    } else {
                        match resp.json::<serde_json::Value>().await {
                            Ok(data) => {
                                if let Some(embedding) = data
                                    .get("data")
                                    .and_then(|d| d.as_array())
                                    .and_then(|arr| arr.first())
                                    .and_then(|first| first.get("embedding"))
                                    .and_then(|e| e.as_array())
                                {
                                    let result: Vec<f64> =
                                        embedding.iter().filter_map(|v| v.as_f64()).collect();
                                    if !result.is_empty() {
                                        self.cache
                                            .put_embedding(query.to_string(), result.clone())
                                            .await;
                                        return result;
                                    }
                                    last_err = "empty embedding vector in response".to_string();
                                } else {
                                    last_err = "response missing data[0].embedding".to_string();
                                }
                                warn!(
                                    "NIM embeddings unusable (attempt {}/3): {}",
                                    attempt, last_err
                                );
                            }
                            Err(e) => {
                                last_err = format!("bad JSON body: {e}");
                                warn!(
                                    "NIM embeddings parse failed (attempt {}/3): {}",
                                    attempt, last_err
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    last_err = format!("{e}");
                    warn!("NIM embeddings request failed (attempt {}/3): {}", attempt, last_err);
                }
            }
            if attempt < 3 {
                tokio::time::sleep(std::time::Duration::from_millis(400 * attempt as u64)).await;
            }
        }
        error!("vectorize_query_failed after 3 attempts: {}", last_err);
        vec![]
    }

    pub async fn vector_search(&self, graph_name: &str, query_vectors: &[f64], limit: usize) -> Vec<SearchResult> {
        if query_vectors.is_empty() {
            return vec![];
        }

        let cache_key = VectorCacheKey {
            graph_name: graph_name.to_string(),
            vector_hash: MultiLevelCache::hash_embedding(query_vectors),
            limit,
        };

        // Check L2 Vector Search Cache
        if let Some(cached_results) = self.cache.get_vector_results(&cache_key).await {
            return cached_results;
        }

        let embedding_str = serde_json::to_string(query_vectors).unwrap_or_default();
        let cypher = format!(
            "CALL db.idx.vector.queryNodes('Vector_Chunk', 'embeddings', {}, vecf32({})) YIELD node, score \
             RETURN node.id AS chunk_id, node.content AS content, node.chunk_type AS chunk_type, \
             node.source_id AS source_id, node.metadata AS metadata, score AS score \
             ORDER BY score DESC",
            limit, embedding_str
        );

        match self.falkordb_client.query(graph_name, &cypher).await {
            Ok(val) => {
                let results = self.parse_graph_results(&val, true);
                self.cache.put_vector_results(cache_key, results.clone()).await;
                results
            }
            Err(e) => {
                error!("vector_search_failed: {}", e);
                vec![]
            }
        }
    }

    pub async fn text_search(&self, graph_name: &str, query: &str, limit: usize) -> Vec<SearchResult> {
        // Implement simple text search via regex for word boundaries and length > 2
        let words: Vec<&str> = query.split_whitespace().collect();
        let keywords: Vec<&str> = words.into_iter().filter(|w| w.len() > 2).collect();
        if keywords.is_empty() {
            return vec![];
        }
        let fts_query = keywords.join(" ");
        let cypher = format!(
            "CALL db.idx.fulltext.queryNodes('Vector_Chunk', '{}') YIELD node, score \
             RETURN node.id AS chunk_id, node.content AS content, node.chunk_type AS chunk_type, \
             node.source_id AS source_id, node.metadata AS metadata, score AS score \
             ORDER BY score DESC LIMIT {}",
            fts_query, limit
        );

        match self.falkordb_client.query(graph_name, &cypher).await {
            Ok(val) => self.parse_graph_results(&val, true),
            Err(e) => {
                error!("text_search_failed: {}", e);
                vec![]
            }
        }
    }

    pub async fn dfs_traversal(
        &self,
        graph_name: &str,
        start_chunk_ids: &[String],
        max_depth: usize,
        min_relevance: f64,
        max_results: usize,
    ) -> Vec<SearchResult> {
        if start_chunk_ids.is_empty() {
            return vec![];
        }

        let ids_str = start_chunk_ids
            .iter()
            .map(|id| format!("'{}'", id))
            .collect::<Vec<_>>()
            .join(", ");

        let cypher = format!(
            "MATCH path = (start:Vector_Chunk)-[*1..{}]-(n:Vector_Chunk) \
             WHERE start.id IN [{}] AND NOT n.id IN [{}] \
             RETURN DISTINCT n.id AS chunk_id, n.content AS content, n.chunk_type AS chunk_type, \
             n.source_id AS source_id, n.metadata AS metadata, length(path) AS depth \
             LIMIT {}",
            max_depth, ids_str, ids_str, max_results
        );

        match self.falkordb_client.query(graph_name, &cypher).await {
            Ok(val) => {
                let mut chunks = self.parse_graph_results(&val, false);
                for chunk in &mut chunks {
                    chunk.score = 0.1f64.max(1.0 - (chunk.depth as f64 * 0.2));
                }
                chunks.into_iter().filter(|c| c.score >= min_relevance).collect()
            }
            Err(e) => {
                error!("dfs_traversal_failed: {}", e);
                vec![]
            }
        }
    }

    fn parse_graph_results(&self, raw_result: &serde_json::Value, is_vector: bool) -> Vec<SearchResult> {
        let mut results = vec![];
        if let Some(arr) = raw_result.as_array() {
            if arr.len() >= 2 {
                let headers_raw = &arr[0];
                let rows_raw = &arr[1];

                let mut headers = vec![];
                if let Some(h_arr) = headers_raw.as_array() {
                    for h in h_arr {
                        // Compact replies (if ever re-enabled) encode headers as
                        // `[type_code, name]` — so prefer the last string in the
                        // pair, which is the name.
                        if let Some(pair) = h.as_array() {
                            if let Some(name) = pair.iter().rev().find_map(|v| v.as_str()) {
                                headers.push(name.to_string());
                                continue;
                            }
                        }
                        if let Some(s) = h.as_str() {
                            headers.push(s.to_string());
                        } else {
                            headers.push(h.to_string());
                        }
                    }
                }

                if let Some(r_arr) = rows_raw.as_array() {
                    for row in r_arr {
                        if let Some(cols) = row.as_array() {
                            let mut chunk_id = String::new();
                            let mut content = String::new();
                            let mut chunk_type = String::new();
                            let mut source_id = String::new();
                            let mut metadata = serde_json::Value::Null;
                            let mut score = 0.0;
                            let mut depth = 0;

                            for (i, val) in cols.iter().enumerate() {
                                if i >= headers.len() {
                                    continue;
                                }
                                let col_name = &headers[i];
                                // Strings pass through; numbers/bools are rendered so
                                // score/depth stay parseable below; nulls stay empty.
                                let v_str = match val {
                                    serde_json::Value::Null => String::new(),
                                    serde_json::Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                };

                                match col_name.as_str() {
                                    "chunk_id" => chunk_id = v_str,
                                    "content" => content = v_str,
                                    "chunk_type" => chunk_type = v_str,
                                    "source_id" => source_id = v_str,
                                    "metadata" => {
                                        if !v_str.is_empty() && v_str != "None" {
                                            if let Ok(m) = serde_json::from_str(&v_str) {
                                                metadata = m;
                                            }
                                        }
                                    }
                                    "score" if is_vector => {
                                        if let Some(n) = val.as_f64() {
                                            score = n;
                                        } else if let Ok(n) = v_str.parse::<f64>() {
                                            score = n;
                                        }
                                    }
                                    "depth" => {
                                        if let Some(n) = val.as_i64() {
                                            depth = n as i32;
                                        } else if let Ok(n) = v_str.parse::<i32>() {
                                            depth = n;
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            results.push(SearchResult {
                                chunk_id,
                                content,
                                score,
                                metadata,
                                source: "falkordb".to_string(),
                                chunk_type,
                                document_id: source_id.clone(),
                                source_id,
                                depth,
                                matched_by_chunks: vec![],
                            });
                        }
                    }
                }
            }
        }
        results
    }

    #[allow(dead_code)]
    pub async fn retrieve(
        &self,
        graph_name: &str,
        query: &str,
        _group_ids: Option<Vec<String>>,
        num_results: usize,
        _center_node_uuid: Option<String>,
    ) -> Vec<SearchResult> {
        let vector = self.vectorize_query(query).await;
        let mut results = vec![];

        if !vector.is_empty() {
            results = self.vector_search(graph_name, &vector, num_results).await;
        }

        if results.is_empty() {
            info!("Vector search returned 0 results, falling back to text search for: {}", query);
            results = self.text_search(graph_name, query, num_results).await;
        }
        results
    }

    pub async fn bfs_traversal(
        &self,
        graph_name: &str,
        start_chunk_ids: &[String],
        max_depth: usize,
        max_results: usize,
    ) -> Vec<SearchResult> {
        self.graph_algo
            .bfs_traversal(graph_name, start_chunk_ids, max_depth, max_results)
            .await
    }

    #[allow(dead_code)]
    pub async fn shortest_paths(
        &self,
        graph_name: &str,
        source_id: &str,
        target_id: &str,
        max_hops: usize,
        path_count: usize,
        rel_types: &[String],
    ) -> Vec<PathResult> {
        self.graph_algo
            .shortest_paths(graph_name, source_id, target_id, max_hops, path_count, rel_types)
            .await
    }

    pub async fn get_pagerank_scores(
        &self,
        graph_name: &str,
        candidate_ids: &[String],
    ) -> HashMap<String, f64> {
        self.graph_algo
            .get_pagerank_scores(graph_name, candidate_ids)
            .await
    }

    pub async fn get_betweenness_scores(
        &self,
        graph_name: &str,
        candidate_ids: &[String],
        sampling_size: usize,
    ) -> HashMap<String, f64> {
        self.graph_algo
            .get_betweenness_scores(graph_name, candidate_ids, sampling_size)
            .await
    }

    #[allow(dead_code)]
    pub async fn get_wcc_components(
        &self,
        graph_name: &str,
        candidate_ids: &[String],
    ) -> HashMap<String, i64> {
        self.graph_algo
            .get_wcc_components(graph_name, candidate_ids)
            .await
    }
}
