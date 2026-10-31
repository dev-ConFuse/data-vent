use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{error, info, warn};

use crate::services::intelligent_retriever::SearchResult;
use crate::services::vector_search::FalkorDbClient;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathResult {
    pub source_id: String,
    pub target_id: String,
    pub hops: usize,
    pub path_weight: f64,
    pub node_ids: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityScores {
    pub chunk_id: String,
    pub pagerank: f64,
    pub betweenness: f64,
    pub component_id: Option<i64>,
}

#[derive(Clone)]
pub struct GraphAlgorithms {
    falkordb_client: FalkorDbClient,
}

impl GraphAlgorithms {
    pub fn new(falkordb_client: FalkorDbClient) -> Self {
        Self { falkordb_client }
    }

    /// BFS layer-by-layer traversal using FalkorDB algo.bfs:
    /// CALL algo.bfs(start_node, max_depth, relationship) YIELD nodes, edges
    pub async fn bfs_traversal(
        &self,
        graph_name: &str,
        start_chunk_ids: &[String],
        max_depth: usize,
        max_results: usize,
    ) -> Vec<SearchResult> {
        if start_chunk_ids.is_empty() {
            return vec![];
        }

        let ids_str = start_chunk_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ");

        // FalkorDB official algo.bfs syntax
        let cypher_algo = format!(
            "MATCH (start:Vector_Chunk) WHERE start.id IN [{}] \
             CALL algo.bfs(start, {}, 'RELATES_TO') YIELD nodes \
             WITH nodes[size(nodes)-1] AS n, size(nodes) - 1 AS depth \
             WHERE NOT n.id IN [{}] \
             RETURN DISTINCT n.id AS chunk_id, n.content AS content, n.chunk_type AS chunk_type, \
             n.source_id AS source_id, n.metadata AS metadata, depth \
             ORDER BY depth ASC \
             LIMIT {}",
            ids_str, max_depth, ids_str, max_results
        );

        match self.falkordb_client.query(graph_name, &cypher_algo).await {
            Ok(val) => {
                let results = self.parse_nodes(&val, 1);
                if !results.is_empty() {
                    return results;
                }
            }
            Err(e) => {
                warn!("algo.bfs execution fallback to Cypher traversal: {}", e);
            }
        }

        // Fallback to Cypher variable-length path traversal
        let fallback_cypher = format!(
            "MATCH path = (start:Vector_Chunk)-[:RELATES_TO*1..{}]-(n:Vector_Chunk) \
             WHERE start.id IN [{}] AND NOT n.id IN [{}] \
             RETURN DISTINCT n.id AS chunk_id, n.content AS content, n.chunk_type AS chunk_type, \
             n.source_id AS source_id, n.metadata AS metadata, length(path) AS depth \
             ORDER BY depth ASC LIMIT {}",
            max_depth, ids_str, ids_str, max_results
        );

        match self.falkordb_client.query(graph_name, &fallback_cypher).await {
            Ok(val) => self.parse_nodes(&val, 1),
            Err(e) => {
                error!("BFS fallback traversal failed: {}", e);
                vec![]
            }
        }
    }

    /// Shortest paths using FalkorDB algo.SPpaths:
    /// CALL algo.SPpaths({sourceNode: a, targetNode: b, relTypes: [...], maxLen: 3, relDirection: 'both', pathCount: 3}) YIELD path, pathWeight, pathCost
    pub async fn shortest_paths(
        &self,
        graph_name: &str,
        source_id: &str,
        target_id: &str,
        max_hops: usize,
        path_count: usize,
        rel_types: &[String],
    ) -> Vec<PathResult> {
        let clean_source = source_id.replace('\'', "\\'");
        let clean_target = target_id.replace('\'', "\\'");

        let rel_types_str = if rel_types.is_empty() {
            "'RELATES_TO', 'REFERENCES'".to_string()
        } else {
            rel_types
                .iter()
                .map(|r| format!("'{}'", r.replace('\'', "\\'")))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let cypher_algo = format!(
            "MATCH (source:Vector_Chunk {{id: '{}'}}), (target:Vector_Chunk {{id: '{}'}}) \
             CALL algo.SPpaths({{ \
                 sourceNode: source, \
                 targetNode: target, \
                 relTypes: [{}], \
                 maxLen: {}, \
                 relDirection: 'both', \
                 pathCount: {} \
             }}) YIELD path, pathWeight \
             RETURN pathWeight, [n IN nodes(path) | n.id] AS nodeIds, length(path) AS hops",
            clean_source, clean_target, rel_types_str, max_hops, path_count
        );

        if let Ok(val) = self.falkordb_client.query(graph_name, &cypher_algo).await {
            let paths = self.parse_sppaths(&val, source_id, target_id);
            if !paths.is_empty() {
                return paths;
            }
        }

        // Fallback to Cypher shortestPath()
        let fallback_cypher = format!(
            "MATCH (source:Vector_Chunk {{id: '{}'}}), (target:Vector_Chunk {{id: '{}'}}) \
             MATCH path = shortestPath((source)-[*..{}]-(target)) \
             RETURN 1.0 AS pathWeight, [n IN nodes(path) | n.id] AS nodeIds, length(path) AS hops",
            clean_source, clean_target, max_hops
        );

        match self.falkordb_client.query(graph_name, &fallback_cypher).await {
            Ok(val) => self.parse_sppaths(&val, source_id, target_id),
            Err(e) => {
                error!("Shortest path search failed: {}", e);
                vec![]
            }
        }
    }

    /// PageRank scores using FalkorDB algo.pageRank:
    /// CALL algo.pageRank(label, relationship_type) YIELD node, score
    pub async fn get_pagerank_scores(
        &self,
        graph_name: &str,
        candidate_ids: &[String],
    ) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        if candidate_ids.is_empty() {
            return scores;
        }

        let ids_str = candidate_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ");

        // First check if pagerank is precomputed as node property
        let cypher_prop = format!(
            "MATCH (n:Vector_Chunk) WHERE n.id IN [{}] \
             RETURN n.id AS chunk_id, coalesce(n.pagerank, 0.0) AS pagerank",
            ids_str
        );

        if let Ok(val) = self.falkordb_client.query(graph_name, &cypher_prop).await {
            scores = self.parse_key_float(&val, "chunk_id", "pagerank");
            if scores.values().any(|&v| v > 0.0) {
                return scores;
            }
        }

        // Run native algo.pageRank
        let cypher_algo = format!(
            "CALL algo.pageRank('Vector_Chunk', 'RELATES_TO') YIELD node, score \
             WHERE node.id IN [{}] \
             RETURN node.id AS chunk_id, score AS pagerank",
            ids_str
        );

        if let Ok(val) = self.falkordb_client.query(graph_name, &cypher_algo).await {
            let algo_scores = self.parse_key_float(&val, "chunk_id", "pagerank");
            if !algo_scores.is_empty() {
                return algo_scores;
            }
        }

        scores
    }

    /// Betweenness centrality using FalkorDB algo.betweenness:
    /// CALL algo.betweenness({nodeLabels: [...], relationshipTypes: [...], samplingSize: 32, samplingSeed: 0}) YIELD node, score
    pub async fn get_betweenness_scores(
        &self,
        graph_name: &str,
        candidate_ids: &[String],
        sampling_size: usize,
    ) -> HashMap<String, f64> {
        if candidate_ids.is_empty() {
            return HashMap::new();
        }

        let ids_str = candidate_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ");

        let cypher = format!(
            "CALL algo.betweenness({{ \
                'nodeLabels': ['Vector_Chunk'], \
                'relationshipTypes': ['RELATES_TO'], \
                'samplingSize': {}, \
                'samplingSeed': 0 \
             }}) YIELD node, score \
             WHERE node.id IN [{}] \
             RETURN node.id AS chunk_id, score AS betweenness",
            sampling_size, ids_str
        );

        match self.falkordb_client.query(graph_name, &cypher).await {
            Ok(val) => self.parse_key_float(&val, "chunk_id", "betweenness"),
            Err(e) => {
                warn!("Betweenness centrality query failed: {}", e);
                HashMap::new()
            }
        }
    }

    /// Weakly Connected Components using FalkorDB algo.WCC:
    /// CALL algo.WCC({nodeLabels: [...], relationshipTypes: [...]}) YIELD node, componentId
    pub async fn get_wcc_components(
        &self,
        graph_name: &str,
        candidate_ids: &[String],
    ) -> HashMap<String, i64> {
        if candidate_ids.is_empty() {
            return HashMap::new();
        }

        let ids_str = candidate_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ");

        // Check if component_id property is already stored on nodes
        let prop_cypher = format!(
            "MATCH (n:Vector_Chunk) WHERE n.id IN [{}] \
             RETURN n.id AS chunk_id, coalesce(n.component_id, -1) AS component_id",
            ids_str
        );

        if let Ok(val) = self.falkordb_client.query(graph_name, &prop_cypher).await {
            let comp_map = self.parse_key_int(&val, "chunk_id", "component_id");
            if comp_map.values().any(|&v| v >= 0) {
                return comp_map;
            }
        }

        // Run native algo.WCC
        let cypher = format!(
            "CALL algo.WCC({{ \
                'nodeLabels': ['Vector_Chunk'], \
                'relationshipTypes': ['RELATES_TO'] \
             }}) YIELD node, componentId \
             WHERE node.id IN [{}] \
             RETURN node.id AS chunk_id, componentId AS component_id",
            ids_str
        );

        match self.falkordb_client.query(graph_name, &cypher).await {
            Ok(val) => self.parse_key_int(&val, "chunk_id", "component_id"),
            Err(e) => {
                warn!("WCC algorithm query failed: {}", e);
                HashMap::new()
            }
        }
    }

    /// Community Detection via Label Propagation using FalkorDB algo.labelPropagation:
    /// CALL algo.labelPropagation({nodeLabels: [...], relationshipTypes: [...], maxIterations: 10}) YIELD node, communityId
    #[allow(dead_code)]
    pub async fn get_cdlp_communities(
        &self,
        graph_name: &str,
        candidate_ids: &[String],
        max_iterations: usize,
    ) -> HashMap<String, i64> {
        if candidate_ids.is_empty() {
            return HashMap::new();
        }

        let ids_str = candidate_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ");

        let cypher = format!(
            "CALL algo.labelPropagation({{ \
                'nodeLabels': ['Vector_Chunk'], \
                'relationshipTypes': ['RELATES_TO'], \
                'maxIterations': {} \
             }}) YIELD node, communityId \
             WHERE node.id IN [{}] \
             RETURN node.id AS chunk_id, communityId AS community_id",
            max_iterations, ids_str
        );

        match self.falkordb_client.query(graph_name, &cypher).await {
            Ok(val) => self.parse_key_int(&val, "chunk_id", "community_id"),
            Err(e) => {
                warn!("CDLP label propagation query failed: {}", e);
                HashMap::new()
            }
        }
    }

    /// Precompute PageRank scores on graph updates and write back to nodes
    #[allow(dead_code)]
    pub async fn precompute_pagerank(&self, graph_name: &str) -> anyhow::Result<usize> {
        info!("Running PageRank pre-computation on graph: {}", graph_name);
        let cypher = "CALL algo.pageRank('Vector_Chunk', 'RELATES_TO') YIELD node, score \
                      SET node.pagerank = score RETURN count(node) AS updated_count";
        let res = self.falkordb_client.query(graph_name, cypher).await?;
        let count_map = self.parse_key_int(&res, "updated_count", "updated_count");
        let count = count_map.values().cloned().sum::<i64>() as usize;
        info!("PageRank pre-computation complete. {} nodes updated.", count);
        Ok(count)
    }

    /// Precompute WCC component IDs on graph updates and write back to nodes
    #[allow(dead_code)]
    pub async fn precompute_wcc(&self, graph_name: &str) -> anyhow::Result<usize> {
        info!("Running WCC pre-computation on graph: {}", graph_name);
        let cypher = "CALL algo.WCC({'nodeLabels': ['Vector_Chunk'], 'relationshipTypes': ['RELATES_TO']}) YIELD node, componentId \
                      SET node.component_id = componentId RETURN count(node) AS updated_count";
        let res = self.falkordb_client.query(graph_name, cypher).await?;
        let count_map = self.parse_key_int(&res, "updated_count", "updated_count");
        let count = count_map.values().cloned().sum::<i64>() as usize;
        info!("WCC pre-computation complete. {} nodes assigned to communities.", count);
        Ok(count)
    }

    // --- Result parsing helpers ---

    fn parse_nodes(&self, raw: &serde_json::Value, default_depth: i32) -> Vec<SearchResult> {
        let mut results = vec![];
        if let Some(arr) = raw.as_array() {
            if arr.len() >= 2 {
                let headers = match arr[0].as_array() {
                    Some(h) => h.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect::<Vec<_>>(),
                    None => return results,
                };
                if let Some(rows) = arr[1].as_array() {
                    for row in rows {
                        if let Some(cols) = row.as_array() {
                            let mut chunk_id = String::new();
                            let mut content = String::new();
                            let mut chunk_type = String::new();
                            let mut source_id = String::new();
                            let mut metadata = serde_json::Value::Null;
                            let mut depth = default_depth;

                            for (i, val) in cols.iter().enumerate() {
                                if i >= headers.len() {
                                    continue;
                                }
                                let col_name = &headers[i];
                                let v_str = val.as_str().unwrap_or("").to_string();

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

                            if !chunk_id.is_empty() {
                                results.push(SearchResult {
                                    chunk_id: chunk_id.clone(),
                                    content,
                                    score: 0.1f64.max(1.0 - (depth as f64 * 0.15)),
                                    metadata,
                                    source: "falkordb_bfs".to_string(),
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
        }
        results
    }

    fn parse_sppaths(&self, raw: &serde_json::Value, source_id: &str, target_id: &str) -> Vec<PathResult> {
        let mut paths = vec![];
        if let Some(arr) = raw.as_array() {
            if arr.len() >= 2 {
                let headers = match arr[0].as_array() {
                    Some(h) => h.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect::<Vec<_>>(),
                    None => return paths,
                };
                let weight_idx = headers.iter().position(|h| h == "pathWeight");
                let nodes_idx = headers.iter().position(|h| h == "nodeIds");
                let hops_idx = headers.iter().position(|h| h == "hops");

                if let Some(rows) = arr[1].as_array() {
                    for row in rows {
                        if let Some(cols) = row.as_array() {
                            let path_weight = weight_idx
                                .and_then(|i| cols.get(i))
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0);

                            let hops = hops_idx
                                .and_then(|i| cols.get(i))
                                .and_then(|v| v.as_i64())
                                .unwrap_or(1) as usize;

                            let mut node_ids = vec![];
                            if let Some(n_i) = nodes_idx {
                                if let Some(n_val) = cols.get(n_i) {
                                    if let Some(arr_val) = n_val.as_array() {
                                        for item in arr_val {
                                            if let Some(s) = item.as_str() {
                                                node_ids.push(s.to_string());
                                            }
                                        }
                                    }
                                }
                            }

                            paths.push(PathResult {
                                source_id: source_id.to_string(),
                                target_id: target_id.to_string(),
                                hops,
                                path_weight,
                                node_ids,
                            });
                        }
                    }
                }
            }
        }
        paths
    }

    fn parse_key_float(&self, raw: &serde_json::Value, key_col: &str, val_col: &str) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        if let Some(arr) = raw.as_array() {
            if arr.len() >= 2 {
                let headers = match arr[0].as_array() {
                    Some(h) => h.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect::<Vec<_>>(),
                    None => return map,
                };
                let key_idx = headers.iter().position(|h| h == key_col);
                let val_idx = headers.iter().position(|h| h == val_col);

                if let (Some(k_i), Some(v_i)) = (key_idx, val_idx) {
                    if let Some(rows) = arr[1].as_array() {
                        for row in rows {
                            if let Some(cols) = row.as_array() {
                                if cols.len() > k_i && cols.len() > v_i {
                                    let k = cols[k_i].as_str().unwrap_or("").to_string();
                                    let v = cols[v_i].as_f64().unwrap_or_else(|| {
                                        cols[v_i].as_str().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0)
                                    });
                                    if !k.is_empty() {
                                        map.insert(k, v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        map
    }

    fn parse_key_int(&self, raw: &serde_json::Value, key_col: &str, val_col: &str) -> HashMap<String, i64> {
        let mut map = HashMap::new();
        if let Some(arr) = raw.as_array() {
            if arr.len() >= 2 {
                let headers = match arr[0].as_array() {
                    Some(h) => h.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect::<Vec<_>>(),
                    None => return map,
                };
                let key_idx = headers.iter().position(|h| h == key_col);
                let val_idx = headers.iter().position(|h| h == val_col);

                if let (Some(k_i), Some(v_i)) = (key_idx, val_idx) {
                    if let Some(rows) = arr[1].as_array() {
                        for row in rows {
                            if let Some(cols) = row.as_array() {
                                if cols.len() > k_i && cols.len() > v_i {
                                    let k = cols[k_i].as_str().unwrap_or("").to_string();
                                    let v = cols[v_i].as_i64().unwrap_or_else(|| {
                                        cols[v_i].as_str().and_then(|s| s.parse::<i64>().ok()).unwrap_or(-1)
                                    });
                                    if !k.is_empty() {
                                        map.insert(k, v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        map
    }

    #[allow(dead_code)]
    pub async fn run_parallel_algorithms(
        &self,
        graph_name: &str,
        candidate_ids: &[String],
    ) -> ParallelAlgorithmResults {
        let start = std::time::Instant::now();
        let (pagerank, betweenness, wcc) = tokio::join!(
            self.get_pagerank_scores(graph_name, candidate_ids),
            self.get_betweenness_scores(graph_name, candidate_ids, 64),
            self.get_wcc_components(graph_name, candidate_ids),
        );

        ParallelAlgorithmResults {
            pagerank,
            betweenness,
            wcc,
            total_duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelAlgorithmResults {
    pub pagerank: HashMap<String, f64>,
    pub betweenness: HashMap<String, f64>,
    pub wcc: HashMap<String, i64>,
    pub total_duration_ms: f64,
}

