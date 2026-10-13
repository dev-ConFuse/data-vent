use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::infra::Config;
use crate::services::intelligent_retriever::SearchResult;

#[allow(dead_code)]
#[derive(Default)]
pub struct CacheMetrics {
    pub embedding_hits: AtomicU64,
    pub embedding_misses: AtomicU64,
    pub vector_hits: AtomicU64,
    pub vector_misses: AtomicU64,
    pub algorithm_hits: AtomicU64,
    pub algorithm_misses: AtomicU64,
    pub result_hits: AtomicU64,
    pub result_misses: AtomicU64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VectorCacheKey {
    pub graph_name: String,
    pub vector_hash: u64,
    pub limit: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AlgorithmCacheKey {
    pub graph_name: String,
    pub algo_name: String,
    pub params_hash: u64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResultCacheKey {
    pub graph_name: String,
    pub query_hash: u64,
    pub limit: usize,
    pub options_hash: u64,
}

#[derive(Clone)]
pub struct MultiLevelCache {
    embedding_cache: Arc<Mutex<LruCache<String, (Vec<f64>, Instant)>>>,
    embedding_ttl: Duration,

    vector_cache: Arc<Mutex<LruCache<VectorCacheKey, (Vec<SearchResult>, Instant)>>>,
    vector_ttl: Duration,

    #[allow(dead_code)]
    algorithm_cache: Arc<Mutex<LruCache<AlgorithmCacheKey, (serde_json::Value, Instant)>>>,
    #[allow(dead_code)]
    algorithm_ttl: Duration,

    #[allow(dead_code)]
    result_cache: Arc<Mutex<LruCache<ResultCacheKey, (serde_json::Value, Instant)>>>,
    #[allow(dead_code)]
    result_ttl: Duration,

    pub metrics: Arc<CacheMetrics>,
}

impl MultiLevelCache {
    pub fn new(config: &Config) -> Self {
        let emb_cap = NonZeroUsize::new(config.cache_embedding_capacity).unwrap_or(NonZeroUsize::new(1000).expect("literal capacity is non-zero"));
        let vec_cap = NonZeroUsize::new(config.cache_vector_capacity).unwrap_or(NonZeroUsize::new(1000).expect("literal capacity is non-zero"));
        let algo_cap = NonZeroUsize::new(config.cache_algorithm_capacity).unwrap_or(NonZeroUsize::new(500).expect("literal capacity is non-zero"));
        let res_cap = NonZeroUsize::new(config.cache_result_capacity).unwrap_or(NonZeroUsize::new(500).expect("literal capacity is non-zero"));

        Self {
            embedding_cache: Arc::new(Mutex::new(LruCache::new(emb_cap))),
            embedding_ttl: Duration::from_secs(config.cache_embedding_ttl_secs),

            vector_cache: Arc::new(Mutex::new(LruCache::new(vec_cap))),
            vector_ttl: Duration::from_secs(config.cache_vector_ttl_secs),

            algorithm_cache: Arc::new(Mutex::new(LruCache::new(algo_cap))),
            algorithm_ttl: Duration::from_secs(config.cache_algorithm_ttl_secs),

            result_cache: Arc::new(Mutex::new(LruCache::new(res_cap))),
            result_ttl: Duration::from_secs(config.cache_result_ttl_secs),

            metrics: Arc::new(CacheMetrics::default()),
        }
    }

    pub fn hash_embedding(vec: &[f64]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for &val in vec {
            hasher.write_u64(val.to_bits());
        }
        hasher.finish()
    }

    #[allow(dead_code)]
    pub fn hash_string(s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    // --- Tier 1: Query Embeddings ---

    pub async fn get_embedding(&self, text: &str) -> Option<Vec<f64>> {
        let mut cache = self.embedding_cache.lock().await;
        if let Some((vec, timestamp)) = cache.get(text) {
            if timestamp.elapsed() < self.embedding_ttl {
                self.metrics.embedding_hits.fetch_add(1, Ordering::Relaxed);
                return Some(vec.clone());
            }
        }
        self.metrics.embedding_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub async fn put_embedding(&self, text: String, embedding: Vec<f64>) {
        if !embedding.is_empty() {
            let mut cache = self.embedding_cache.lock().await;
            cache.put(text, (embedding, Instant::now()));
        }
    }

    // --- Tier 2: Vector Search Results ---

    pub async fn get_vector_results(&self, key: &VectorCacheKey) -> Option<Vec<SearchResult>> {
        let mut cache = self.vector_cache.lock().await;
        if let Some((results, timestamp)) = cache.get(key) {
            if timestamp.elapsed() < self.vector_ttl {
                self.metrics.vector_hits.fetch_add(1, Ordering::Relaxed);
                return Some(results.clone());
            }
        }
        self.metrics.vector_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub async fn put_vector_results(&self, key: VectorCacheKey, results: Vec<SearchResult>) {
        let mut cache = self.vector_cache.lock().await;
        cache.put(key, (results, Instant::now()));
    }

    // --- Tier 3: Graph Algorithm Computations ---

    #[allow(dead_code)]
    pub async fn get_algorithm_result(&self, key: &AlgorithmCacheKey) -> Option<serde_json::Value> {
        let mut cache = self.algorithm_cache.lock().await;
        if let Some((val, timestamp)) = cache.get(key) {
            if timestamp.elapsed() < self.algorithm_ttl {
                self.metrics.algorithm_hits.fetch_add(1, Ordering::Relaxed);
                return Some(val.clone());
            }
        }
        self.metrics.algorithm_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    #[allow(dead_code)]
    pub async fn put_algorithm_result(&self, key: AlgorithmCacheKey, val: serde_json::Value) {
        let mut cache = self.algorithm_cache.lock().await;
        cache.put(key, (val, Instant::now()));
    }

    // --- Tier 4: Aggregated Responses ---

    #[allow(dead_code)]
    pub async fn get_result(&self, key: &ResultCacheKey) -> Option<serde_json::Value> {
        let mut cache = self.result_cache.lock().await;
        if let Some((val, timestamp)) = cache.get(key) {
            if timestamp.elapsed() < self.result_ttl {
                self.metrics.result_hits.fetch_add(1, Ordering::Relaxed);
                return Some(val.clone());
            }
        }
        self.metrics.result_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    #[allow(dead_code)]
    pub async fn put_result(&self, key: ResultCacheKey, val: serde_json::Value) {
        let mut cache = self.result_cache.lock().await;
        cache.put(key, (val, Instant::now()));
    }

    // --- Cache Invalidation ---

    pub async fn invalidate_graph(&self, graph_name: &str) {
        {
            let mut v_cache = self.vector_cache.lock().await;
            v_cache.clear();
        }
        {
            let mut a_cache = self.algorithm_cache.lock().await;
            a_cache.clear();
        }
        {
            let mut r_cache = self.result_cache.lock().await;
            r_cache.clear();
        }
        tracing::info!("Invalidated caches for graph: {}", graph_name);
    }

    #[allow(dead_code)]
    pub async fn warm_frequent_queries(
        &self,
        queries: Vec<String>,
        retriever: &crate::services::intelligent_retriever::IntelligentRetriever,
        graph_name: &str,
    ) -> anyhow::Result<usize> {
        let mut warmed = 0;

        for query in queries {
            let embedding = retriever.vectorize_query(&query).await;
            if !embedding.is_empty() {
                let vector_key = VectorCacheKey {
                    graph_name: graph_name.to_string(),
                    vector_hash: Self::hash_embedding(&embedding),
                    limit: 10,
                };

                // Pre-fetch results
                let results = retriever.vector_search(graph_name, &embedding, 10).await;
                self.put_vector_results(vector_key, results).await;
                warmed += 1;
            }
        }

        tracing::info!("Cache warming completed: {} queries pre-fetched", warmed);
        Ok(warmed)
    }

    pub async fn get_cache_stats(&self) -> CacheStats {
        let embedding_hit_rate = self.calculate_hit_rate(
            self.metrics.embedding_hits.load(Ordering::Relaxed),
            self.metrics.embedding_misses.load(Ordering::Relaxed),
        );

        let vector_hit_rate = self.calculate_hit_rate(
            self.metrics.vector_hits.load(Ordering::Relaxed),
            self.metrics.vector_misses.load(Ordering::Relaxed),
        );

        let algorithm_hit_rate = self.calculate_hit_rate(
            self.metrics.algorithm_hits.load(Ordering::Relaxed),
            self.metrics.algorithm_misses.load(Ordering::Relaxed),
        );

        let result_hit_rate = self.calculate_hit_rate(
            self.metrics.result_hits.load(Ordering::Relaxed),
            self.metrics.result_misses.load(Ordering::Relaxed),
        );

        CacheStats {
            embedding_hit_rate,
            vector_hit_rate,
            algorithm_hit_rate,
            result_hit_rate,
        }
    }

    fn calculate_hit_rate(&self, hits: u64, misses: u64) -> f64 {
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            (hits as f64) / (total as f64)
        }
    }

    #[allow(dead_code)]
    pub async fn auto_tune_cache_sizes(&self, target_hit_rate: f64) {
        let stats = self.get_cache_stats().await;

        if stats.vector_hit_rate < target_hit_rate {
            let mut v_cache = self.vector_cache.lock().await;
            let current_cap = v_cache.cap().get();
            let new_cap = NonZeroUsize::new(current_cap * 3 / 2).unwrap_or(v_cache.cap());
            v_cache.resize(new_cap);
            tracing::info!(
                "Auto-tuning: Increased vector cache size from {} to {} (hit rate: {:.2}%)",
                current_cap,
                new_cap.get(),
                stats.vector_hit_rate * 100.0
            );
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    pub embedding_hit_rate: f64,
    pub vector_hit_rate: f64,
    pub algorithm_hit_rate: f64,
    pub result_hit_rate: f64,
}



