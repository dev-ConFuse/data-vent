use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tracing::info;

pub use crate::services::fusion::ScoredChunk;
use crate::services::fusion::{FusionEngine, RankedList};
use crate::services::intelligent_retriever::SearchResult;
use crate::services::parallel_search::ParallelSearchResult;

pub struct AggregatedResult {
    pub chunks: Vec<ScoredChunk>,
    pub total_results: usize,
    pub unique_sources: usize,
    pub vector_matches: usize,
    pub graph_matches: usize,
    /// Fusion-engine saturation flag: enough high-scoring chunks were fused, or
    /// the result budget was filled. Distinct from `guidance.completeness`, which
    /// measures how much of the *question* is backed by evidence.
    pub completion_reached: bool,
    #[allow(dead_code)]
    pub aggregation_time_ms: f64,
    /// Research guidance for the calling agent (see the Research Protocol).
    pub guidance: ResearchGuidance,
}

/// Coverage of one sub-query produced by query decomposition.
#[derive(Debug, Clone)]
pub struct SubQueryCoverage {
    pub sub_query: String,
    pub intent: String,
    pub hits: usize,
    pub errored: bool,
}

/// A concrete, ready-to-execute follow-up that closes an evidence gap.
///
/// `intent` + `keywords` map 1:1 onto the client-connector `search_knowledge`
/// tool arguments so an agent can execute the action without inventing params.
#[derive(Debug, Clone)]
pub struct NextAction {
    pub tool: String,
    pub intent: String,
    pub keywords: Vec<String>,
    pub why: String,
}

/// Retrieval-side research guidance.
///
/// Retrieval is deliberately not a one-shot answer service: every response
/// reports how much of the question is actually backed by evidence and what the
/// caller should do next, so agents keep reasoning instead of answering from a
/// single pass.
#[derive(Debug, Clone)]
pub struct ResearchGuidance {
    /// 0.0..=1.0 — sub-query coverage blended with source corroboration.
    pub completeness: f64,
    /// "ready_to_answer" | "partial" | "needs_more_evidence" | "not_covered"
    pub status: String,
    pub covered_sub_queries: usize,
    pub total_sub_queries: usize,
    pub uncovered_sub_queries: Vec<String>,
    pub sub_query_coverage: Vec<SubQueryCoverage>,
    pub next_actions: Vec<NextAction>,
}

impl ResearchGuidance {
    pub fn empty() -> Self {
        Self {
            completeness: 0.0,
            status: "not_covered".to_string(),
            covered_sub_queries: 0,
            total_sub_queries: 0,
            uncovered_sub_queries: vec![],
            sub_query_coverage: vec![],
            next_actions: vec![],
        }
    }
}

pub struct ResultAggregator {
    max_results: usize,
    min_avg_score: f64,
    min_chunks_for_completion: usize,
    vector_weight: f64,
    graph_weight: f64,
    cross_chunk_weight: f64,
    fusion_algorithm: String,
    fusion_engine: FusionEngine,
}

impl ResultAggregator {
    pub fn new(
        max_results: usize,
        min_avg_score: f64,
        min_chunks_for_completion: usize,
        vector_weight: f64,
        graph_weight: f64,
        cross_chunk_weight: f64,
        fusion_algorithm: String,
        wrrf_k: f64,
        pagerank_boost_weight: f64,
    ) -> Self {
        Self {
            max_results,
            min_avg_score,
            min_chunks_for_completion,
            vector_weight,
            graph_weight,
            cross_chunk_weight,
            fusion_algorithm,
            fusion_engine: FusionEngine::new(wrrf_k, pagerank_boost_weight),
        }
    }

    pub fn aggregate(
        &self,
        parallel_result: ParallelSearchResult,
        original_query: &str,
        limit: usize,
    ) -> AggregatedResult {
        self.aggregate_with_centrality(
            parallel_result,
            original_query,
            limit,
            &HashMap::new(),
            &HashMap::new(),
        )
    }

    pub fn aggregate_with_centrality(
        &self,
        parallel_result: ParallelSearchResult,
        _original_query: &str,
        limit: usize,
        pagerank_scores: &HashMap<String, f64>,
        betweenness_scores: &HashMap<String, f64>,
    ) -> AggregatedResult {
        let start = Instant::now();
        let limit = if limit == 0 { self.max_results } else { limit };

        if parallel_result.chunk_results.is_empty() {
            return AggregatedResult {
                chunks: vec![],
                total_results: 0,
                unique_sources: 0,
                vector_matches: 0,
                graph_matches: 0,
                completion_reached: false,
                aggregation_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                guidance: ResearchGuidance::empty(),
            };
        }

        let scored = if self.fusion_algorithm.to_lowercase() == "wrrf" {
            // Build RankedLists from chunk results
            let mut ranked_lists = vec![];
            for (i, cr) in parallel_result.chunk_results.iter().enumerate() {
                if cr.error.is_some() {
                    continue;
                }
                let weight = cr.query_chunk.weight;

                if !cr.vector_results.is_empty() {
                    let mut vec_items = cr.vector_results.clone();
                    for item in &mut vec_items {
                        item.matched_by_chunks.push(cr.query_chunk.text.clone());
                    }
                    ranked_lists.push(RankedList {
                        name: format!("vector_{}", i),
                        weight: weight * self.vector_weight,
                        items: vec_items,
                    });
                }

                if !cr.graph_results.is_empty() {
                    let mut graph_items = cr.graph_results.clone();
                    for item in &mut graph_items {
                        item.matched_by_chunks.push(cr.query_chunk.text.clone());
                    }
                    ranked_lists.push(RankedList {
                        name: format!("graph_{}", i),
                        weight: weight * self.graph_weight,
                        items: graph_items,
                    });
                }
            }

            self.fusion_engine.wrrf_fuse(
                &ranked_lists,
                pagerank_scores,
                betweenness_scores,
                parallel_result.chunks_searched,
                limit,
            )
        } else {
            // Legacy weighted linear combination
            let mut chunk_map: HashMap<String, ChunkAccumulator> = HashMap::new();
            for chunk_result in &parallel_result.chunk_results {
                if chunk_result.error.is_some() {
                    continue;
                }

                let chunk_text = &chunk_result.query_chunk.text;
                let chunk_weight = chunk_result.query_chunk.weight;

                for node in &chunk_result.vector_results {
                    let acc = chunk_map
                        .entry(node.chunk_id.clone())
                        .or_insert_with(|| ChunkAccumulator::new(node.clone()));
                    acc.vector_scores.push(node.score * chunk_weight);
                    acc.matched_by.insert(chunk_text.clone());
                }

                for node in &chunk_result.graph_results {
                    let acc = chunk_map
                        .entry(node.chunk_id.clone())
                        .or_insert_with(|| ChunkAccumulator::new(node.clone()));
                    acc.graph_scores.push(node.score * chunk_weight);
                    acc.matched_by.insert(chunk_text.clone());
                }
            }

            let total_query_chunks = parallel_result.chunks_searched.max(1) as f64;
            let mut legacy_scored: Vec<ScoredChunk> = vec![];

            for (chunk_id, acc) in chunk_map {
                let best_vector = acc.vector_scores.iter().cloned().fold(0.0, f64::max);
                let best_graph = acc.graph_scores.iter().cloned().fold(0.0, f64::max);
                let cross_boost = acc.matched_by.len() as f64 / total_query_chunks;
                let pr = pagerank_scores.get(&chunk_id).cloned().unwrap_or(0.0);

                let final_score = (best_vector * self.vector_weight
                    + best_graph * self.graph_weight
                    + cross_boost * self.cross_chunk_weight)
                    * (1.0 + pr * self.fusion_engine.pagerank_boost_weight);

                legacy_scored.push(ScoredChunk {
                    chunk_id,
                    content: acc.node.content,
                    final_score,
                    vector_score: best_vector,
                    graph_score: best_graph,
                    cross_chunk_boost: cross_boost,
                    pagerank_score: pr,
                    chunk_type: acc.node.chunk_type,
                    source_id: acc.node.source_id.clone(),
                    document_id: acc.node.document_id,
                    metadata: acc.node.metadata,
                    matched_by_chunks: acc.matched_by.into_iter().collect(),
                    depth: acc.node.depth,
                });
            }

            legacy_scored.sort_by(|a, b| {
                b.final_score
                    .partial_cmp(&a.final_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            legacy_scored.truncate(limit);
            legacy_scored
        };

        let completion = self.check_completion(&scored);
        let mut unique_sources_set = HashSet::new();
        for c in &scored {
            if !c.source_id.is_empty() {
                unique_sources_set.insert(c.source_id.clone());
            }
        }
        let unique_sources = unique_sources_set.len();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let total_results = scored.len();
        let guidance = self.build_guidance(&parallel_result, &scored, unique_sources);

        info!(
            "results_aggregated total={} unique_sources={} completion={} algo={} time_ms={:.2}",
            total_results, unique_sources, completion, self.fusion_algorithm, elapsed
        );

        AggregatedResult {
            chunks: scored,
            total_results,
            unique_sources,
            vector_matches: parallel_result.total_vector_hits,
            graph_matches: parallel_result.total_graph_hits,
            completion_reached: completion,
            aggregation_time_ms: elapsed,
            guidance,
        }
    }

    fn check_completion(&self, chunks: &[ScoredChunk]) -> bool {
        if chunks.len() < self.min_chunks_for_completion {
            return false;
        }
        if chunks.len() >= self.max_results {
            return true;
        }
        let sum: f64 = chunks.iter().map(|c| c.final_score).sum();
        let avg = sum / chunks.len().max(1) as f64;
        avg >= self.min_avg_score
    }

    /// Compute research guidance: how much of the question is backed by
    /// evidence, and what the caller should do next.
    ///
    /// `completeness` blends two honest signals:
    ///   * sub-query coverage — did every part of the decomposed question match?
    ///   * evidence depth    — is it corroborated by 2+ independent sources?
    ///
    /// `status` is "ready_to_answer" only when there are no remaining gaps, so
    /// an agent cannot mistake a partially-covered question for a finished one.
    fn build_guidance(
        &self,
        parallel_result: &ParallelSearchResult,
        scored: &[ScoredChunk],
        unique_sources: usize,
    ) -> ResearchGuidance {
        let coverage: Vec<SubQueryCoverage> = parallel_result
            .chunk_results
            .iter()
            .map(|cr| SubQueryCoverage {
                sub_query: cr.query_chunk.text.clone(),
                intent: cr.query_chunk.intent.clone(),
                hits: cr.vector_results.len() + cr.graph_results.len(),
                errored: cr.error.is_some(),
            })
            .collect();

        let total = coverage.len();
        if total == 0 {
            return ResearchGuidance::empty();
        }

        // Keywords are injected as "entity_lookup" chunks by the HTTP layer, so
        // they can be echoed back verbatim as follow-up search arguments.
        let entity_keywords: Vec<String> = parallel_result
            .chunk_results
            .iter()
            .filter(|cr| cr.query_chunk.intent == "entity_lookup")
            .map(|cr| cr.query_chunk.text.clone())
            .collect();

        let covered = coverage
            .iter()
            .filter(|c| !c.errored && c.hits > 0)
            .count();
        let uncovered: Vec<String> = coverage
            .iter()
            .filter(|c| !c.errored && c.hits == 0)
            .map(|c| c.sub_query.clone())
            .collect();

        let coverage_ratio = covered as f64 / total as f64;
        // Two independent sources is the bar for corroborated evidence.
        let evidence_depth = (unique_sources as f64 / 2.0).min(1.0);
        let completeness = if scored.is_empty() {
            0.0
        } else {
            0.7 * coverage_ratio + 0.3 * evidence_depth
        };

        let mut next_actions: Vec<NextAction> = Vec::new();

        // Gap 1: a sub-question produced no evidence at all.
        for sub in uncovered.iter().take(3) {
            let keywords = if entity_keywords.is_empty() {
                vec![sub.clone()]
            } else {
                entity_keywords.clone()
            };
            next_actions.push(NextAction {
                tool: "search_knowledge".to_string(),
                intent: sub.clone(),
                keywords,
                why: "sub-query produced no evidence — retry with tighter terms \
                      (file, function, class or error-message names) before answering"
                    .to_string(),
            });
        }

        // Gap 2: everything matched, but from a single source — uncorroborated.
        if next_actions.is_empty() && !scored.is_empty() && unique_sources < 2 {
            let keywords = if entity_keywords.is_empty() {
                coverage.iter().map(|c| c.sub_query.clone()).take(3).collect()
            } else {
                entity_keywords.clone()
            };
            next_actions.push(NextAction {
                tool: "search_knowledge".to_string(),
                intent: "cross-check this question from an independent angle".to_string(),
                keywords,
                why: "only one source supports the evidence — corroborate before \
                      treating the claim as settled"
                    .to_string(),
            });
        }

        let status = if scored.is_empty() {
            "not_covered"
        } else if next_actions.is_empty() && completeness >= 0.8 {
            "ready_to_answer"
        } else if completeness >= 0.5 {
            "partial"
        } else {
            "needs_more_evidence"
        };

        info!(
            "research_guidance status={} completeness={:.2} covered={}/{} actions={}",
            status,
            completeness,
            covered,
            total,
            next_actions.len()
        );

        ResearchGuidance {
            completeness: (completeness * 100.0).round() / 100.0,
            status: status.to_string(),
            covered_sub_queries: covered,
            total_sub_queries: total,
            uncovered_sub_queries: uncovered,
            sub_query_coverage: coverage,
            next_actions,
        }
    }
}

struct ChunkAccumulator {
    node: SearchResult,
    vector_scores: Vec<f64>,
    graph_scores: Vec<f64>,
    matched_by: HashSet<String>,
}

impl ChunkAccumulator {
    fn new(node: SearchResult) -> Self {
        Self {
            node,
            vector_scores: vec![],
            graph_scores: vec![],
            matched_by: HashSet::new(),
        }
    }
}
