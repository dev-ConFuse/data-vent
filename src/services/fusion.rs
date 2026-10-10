use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::services::intelligent_retriever::SearchResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredChunk {
    pub chunk_id: String,
    pub content: String,
    pub final_score: f64,
    pub vector_score: f64,
    pub graph_score: f64,
    pub cross_chunk_boost: f64,
    pub pagerank_score: f64,
    pub chunk_type: String,
    pub source_id: String,
    pub document_id: String,
    pub metadata: serde_json::Value,
    pub matched_by_chunks: Vec<String>,
    pub depth: i32,
}

#[derive(Debug, Clone)]
pub struct RankedList {
    pub name: String,
    pub weight: f64,
    pub items: Vec<SearchResult>,
}

#[derive(Debug, Clone)]
pub struct FusionEngine {
    pub wrrf_k: f64,
    pub pagerank_boost_weight: f64,
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self {
            wrrf_k: 60.0,
            pagerank_boost_weight: 0.15,
        }
    }
}

impl FusionEngine {
    pub fn new(wrrf_k: f64, pagerank_boost_weight: f64) -> Self {
        Self {
            wrrf_k,
            pagerank_boost_weight,
        }
    }

    /// Weighted Reciprocal Rank Fusion (WRRF) across multiple candidate lists
    pub fn wrrf_fuse(
        &self,
        ranked_lists: &[RankedList],
        pagerank_scores: &HashMap<String, f64>,
        betweenness_scores: &HashMap<String, f64>,
        total_query_chunks: usize,
        limit: usize,
    ) -> Vec<ScoredChunk> {
        let mut score_map: HashMap<String, f64> = HashMap::new();
        let mut best_vector_scores: HashMap<String, f64> = HashMap::new();
        let mut best_graph_scores: HashMap<String, f64> = HashMap::new();
        let mut matched_by_map: HashMap<String, HashSet<String>> = HashMap::new();
        let mut node_details: HashMap<String, SearchResult> = HashMap::new();

        let k = self.wrrf_k;

        for list in ranked_lists {
            let weight = list.weight;
            let is_vector = list.name.starts_with("vector");
            let is_graph = list.name.starts_with("graph") || list.name.starts_with("bfs") || list.name.starts_with("dfs");

            for (rank, item) in list.items.iter().enumerate() {
                let chunk_id = &item.chunk_id;
                let rr_score = weight / (k + (rank as f64) + 1.0);
                *score_map.entry(chunk_id.clone()).or_insert(0.0) += rr_score;

                if is_vector {
                    let prev = best_vector_scores.get(chunk_id).cloned().unwrap_or(0.0);
                    if item.score > prev {
                        best_vector_scores.insert(chunk_id.clone(), item.score);
                    }
                } else if is_graph {
                    let prev = best_graph_scores.get(chunk_id).cloned().unwrap_or(0.0);
                    if item.score > prev {
                        best_graph_scores.insert(chunk_id.clone(), item.score);
                    }
                }

                let matched_set = matched_by_map.entry(chunk_id.clone()).or_insert_with(HashSet::new);
                for m in &item.matched_by_chunks {
                    matched_set.insert(m.clone());
                }

                node_details.entry(chunk_id.clone()).or_insert_with(|| item.clone());
            }
        }

        if score_map.is_empty() {
            return vec![];
        }

        let max_raw_score = score_map.values().cloned().fold(0.0, f64::max);
        let num_chunks = total_query_chunks.max(1) as f64;

        let mut results: Vec<ScoredChunk> = Vec::with_capacity(score_map.len());

        for (chunk_id, raw_rrf) in score_map {
            let item = match node_details.get(&chunk_id) {
                Some(n) => n,
                None => continue,
            };

            let normalized_base = if max_raw_score > 0.0 {
                raw_rrf / max_raw_score
            } else {
                raw_rrf
            };

            let matched = matched_by_map.remove(&chunk_id).unwrap_or_default();
            let cross_boost = (matched.len() as f64) / num_chunks;

            // PageRank and Betweenness Centrality boosting
            let pr = pagerank_scores.get(&chunk_id).cloned().unwrap_or(0.0);
            let bc = betweenness_scores.get(&chunk_id).cloned().unwrap_or(0.0);

            let centrality_multiplier = 1.0 + (pr * self.pagerank_boost_weight) + (bc * 0.1);
            let final_score = (normalized_base * 0.8 + cross_boost * 0.2) * centrality_multiplier;

            results.push(ScoredChunk {
                chunk_id: chunk_id.clone(),
                content: item.content.clone(),
                final_score,
                vector_score: best_vector_scores.get(&chunk_id).cloned().unwrap_or(0.0),
                graph_score: best_graph_scores.get(&chunk_id).cloned().unwrap_or(0.0),
                cross_chunk_boost: cross_boost,
                pagerank_score: pr,
                chunk_type: item.chunk_type.clone(),
                source_id: item.source_id.clone(),
                document_id: item.document_id.clone(),
                metadata: item.metadata.clone(),
                matched_by_chunks: matched.into_iter().collect(),
                depth: item.depth,
            });
        }

        results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 && results.len() > limit {
            results.truncate(limit);
        }

        results
    }

    #[allow(dead_code)]
    pub fn benchmark_fusion_methods(
        &self,
        ranked_lists: &[RankedList],
        pagerank_scores: &HashMap<String, f64>,
        betweenness_scores: &HashMap<String, f64>,
        total_query_chunks: usize,
        limit: usize,
    ) -> FusionBenchmarkResult {
        let start_wrrf = std::time::Instant::now();
        let wrrf_results = self.wrrf_fuse(
            ranked_lists,
            pagerank_scores,
            betweenness_scores,
            total_query_chunks,
            limit,
        );
        let wrrf_duration = start_wrrf.elapsed();

        let start_legacy = std::time::Instant::now();
        let legacy_results = self.legacy_fusion(
            ranked_lists,
            pagerank_scores,
            betweenness_scores,
            total_query_chunks,
            limit,
        );
        let legacy_duration = start_legacy.elapsed();

        FusionBenchmarkResult {
            wrrf_duration_ms: wrrf_duration.as_secs_f64() * 1000.0,
            legacy_duration_ms: legacy_duration.as_secs_f64() * 1000.0,
            wrrf_result_count: wrrf_results.len(),
            legacy_result_count: legacy_results.len(),
        }
    }

    #[allow(dead_code)]
    pub fn legacy_fusion(
        &self,
        ranked_lists: &[RankedList],
        pagerank_scores: &HashMap<String, f64>,
        betweenness_scores: &HashMap<String, f64>,
        total_query_chunks: usize,
        limit: usize,
    ) -> Vec<ScoredChunk> {
        let mut score_map: HashMap<String, f64> = HashMap::new();
        let mut best_vector_scores: HashMap<String, f64> = HashMap::new();
        let mut best_graph_scores: HashMap<String, f64> = HashMap::new();
        let mut matched_by_map: HashMap<String, HashSet<String>> = HashMap::new();
        let mut node_details: HashMap<String, SearchResult> = HashMap::new();

        for list in ranked_lists {
            let weight = list.weight;
            let is_vector = list.name.starts_with("vector");
            let is_graph = list.name.starts_with("graph") || list.name.starts_with("bfs") || list.name.starts_with("dfs");

            for item in &list.items {
                let chunk_id = &item.chunk_id;
                *score_map.entry(chunk_id.clone()).or_insert(0.0) += item.score * weight;

                if is_vector {
                    let prev = best_vector_scores.get(chunk_id).cloned().unwrap_or(0.0);
                    if item.score > prev {
                        best_vector_scores.insert(chunk_id.clone(), item.score);
                    }
                } else if is_graph {
                    let prev = best_graph_scores.get(chunk_id).cloned().unwrap_or(0.0);
                    if item.score > prev {
                        best_graph_scores.insert(chunk_id.clone(), item.score);
                    }
                }

                let matched_set = matched_by_map.entry(chunk_id.clone()).or_insert_with(HashSet::new);
                for m in &item.matched_by_chunks {
                    matched_set.insert(m.clone());
                }

                node_details.entry(chunk_id.clone()).or_insert_with(|| item.clone());
            }
        }

        if score_map.is_empty() {
            return vec![];
        }

        let num_chunks = total_query_chunks.max(1) as f64;
        let mut results: Vec<ScoredChunk> = Vec::with_capacity(score_map.len());

        for (chunk_id, raw_score) in score_map {
            let item = match node_details.get(&chunk_id) {
                Some(n) => n,
                None => continue,
            };

            let matched = matched_by_map.remove(&chunk_id).unwrap_or_default();
            let cross_boost = (matched.len() as f64) / num_chunks;

            let pr = pagerank_scores.get(&chunk_id).cloned().unwrap_or(0.0);
            let bc = betweenness_scores.get(&chunk_id).cloned().unwrap_or(0.0);
            let centrality_multiplier = 1.0 + (pr * self.pagerank_boost_weight) + (bc * 0.1);
            let final_score = (raw_score * 0.8 + cross_boost * 0.2) * centrality_multiplier;

            results.push(ScoredChunk {
                chunk_id: chunk_id.clone(),
                content: item.content.clone(),
                final_score,
                vector_score: best_vector_scores.get(&chunk_id).cloned().unwrap_or(0.0),
                graph_score: best_graph_scores.get(&chunk_id).cloned().unwrap_or(0.0),
                cross_chunk_boost: cross_boost,
                pagerank_score: pr,
                chunk_type: item.chunk_type.clone(),
                source_id: item.source_id.clone(),
                document_id: item.document_id.clone(),
                metadata: item.metadata.clone(),
                matched_by_chunks: matched.into_iter().collect(),
                depth: item.depth,
            });
        }

        results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 && results.len() > limit {
            results.truncate(limit);
        }

        results
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FusionBenchmarkResult {
    pub wrrf_duration_ms: f64,
    pub legacy_duration_ms: f64,
    pub wrrf_result_count: usize,
    pub legacy_result_count: usize,
}



