use envconfig::Envconfig;

/// data-vent configuration. Every field is env-only (no in-code defaults —
/// see `.cline/rules/env-policy.md`); `Option` fields are optional-by-design
/// (absence disables the feature).
#[allow(dead_code)]
#[derive(Envconfig, Debug, Clone)]
pub struct Config {
    #[envconfig(from = "DATA_VENT_PORT")]
    pub app_port: u16,
    
    #[envconfig(from = "HOST")]
    pub host: String,
    
    // FalkorDB
    #[envconfig(from = "FALKORDB_HOST")]
    pub falkordb_host: String,
    
    #[envconfig(from = "FALKORDB_PORT")]
    pub falkordb_port: u16,
    
    #[envconfig(from = "FALKORDB_USERNAME")]
    pub falkordb_username: String,
    
    #[envconfig(from = "FALKORDB_PASSWORD")]
    pub falkordb_password: Option<String>,
    
    #[envconfig(from = "FALKORDB_DATABASE")]
    pub falkordb_database: u16,
    
    #[envconfig(from = "FALKORDB_GRAPH_NAME")]
    pub falkordb_graph_name: String,
    
    #[envconfig(from = "FALKORDB_VECTOR_DIMENSION")]
    pub falkordb_vector_dimension: u16,
    
    #[envconfig(from = "FALKORDB_SIMILARITY_THRESHOLD")]
    pub falkordb_similarity_threshold: f64,
    
    #[envconfig(from = "FALKORDB_MAX_RESULTS")]
    pub falkordb_max_results: u32,
    
    #[envconfig(from = "FALKORDB_USE_TLS")]
    pub falkordb_use_tls: bool,
    
    // Downstream Services
    #[envconfig(from = "NVIDIA_NIM_API_KEY")]
    pub nvidia_nim_api_key: Option<String>,
    
    #[envconfig(from = "NVIDIA_NIM_BASE_URL")]
    pub nvidia_nim_base_url: String,
    
    #[envconfig(from = "DEFAULT_EMBEDDING_MODEL")]
    pub default_embedding_model: String,
    
    // Retrieval Pipeline
    #[envconfig(from = "PIPELINE_MAX_QUERY_CHUNKS")]
    pub pipeline_max_query_chunks: usize,
    
    #[envconfig(from = "PIPELINE_PER_CHUNK_TIMEOUT")]
    pub pipeline_per_chunk_timeout: f64,
    
    #[envconfig(from = "PIPELINE_VECTOR_TOP_K")]
    pub pipeline_vector_top_k: usize,
    
    #[envconfig(from = "PIPELINE_DFS_DEPTH")]
    pub pipeline_dfs_depth: usize,
    
    #[envconfig(from = "PIPELINE_DFS_MIN_RELEVANCE")]
    pub pipeline_dfs_min_relevance: f64,
    
    #[envconfig(from = "PIPELINE_DFS_MAX_RESULTS")]
    pub pipeline_dfs_max_results: usize,
    
    #[envconfig(from = "PIPELINE_MAX_TOTAL_RESULTS")]
    pub pipeline_max_total_results: usize,
    
    #[envconfig(from = "PIPELINE_VECTOR_WEIGHT")]
    pub pipeline_vector_weight: f64,
    
    #[envconfig(from = "PIPELINE_GRAPH_WEIGHT")]
    pub pipeline_graph_weight: f64,
    
    #[envconfig(from = "PIPELINE_CROSS_CHUNK_WEIGHT")]
    pub pipeline_cross_chunk_weight: f64,

    // FalkorDB Graph Algorithms
    #[envconfig(from = "ENABLE_BFS")]
    pub enable_bfs: bool,

    #[envconfig(from = "ENABLE_PAGERANK")]
    pub enable_pagerank: bool,

    #[envconfig(from = "ENABLE_BETWEENNESS")]
    pub enable_betweenness: bool,

    #[envconfig(from = "ENABLE_WCC")]
    pub enable_wcc: bool,

    #[envconfig(from = "ENABLE_SPPATHS")]
    pub enable_sppaths: bool,

    #[envconfig(from = "PAGERANK_BOOST_WEIGHT")]
    pub pagerank_boost_weight: f64,

    // HNSW Configuration
    #[envconfig(from = "HNSW_M")]
    pub hnsw_m: u32,

    #[envconfig(from = "HNSW_EF_CONSTRUCTION")]
    pub hnsw_ef_construction: u32,

    #[envconfig(from = "HNSW_EF_RUNTIME")]
    pub hnsw_ef_runtime: u32,

    #[envconfig(from = "HNSW_SIMILARITY_FUNCTION")]
    pub hnsw_similarity_function: String,

    #[envconfig(from = "HNSW_MODE")]
    pub hnsw_mode: String,

    // Multi-Level Caching
    #[envconfig(from = "CACHE_EMBEDDING_CAPACITY")]
    pub cache_embedding_capacity: usize,

    #[envconfig(from = "CACHE_EMBEDDING_TTL_SECS")]
    pub cache_embedding_ttl_secs: u64,

    #[envconfig(from = "CACHE_VECTOR_CAPACITY")]
    pub cache_vector_capacity: usize,

    #[envconfig(from = "CACHE_VECTOR_TTL_SECS")]
    pub cache_vector_ttl_secs: u64,

    #[envconfig(from = "CACHE_ALGORITHM_CAPACITY")]
    pub cache_algorithm_capacity: usize,

    #[envconfig(from = "CACHE_ALGORITHM_TTL_SECS")]
    pub cache_algorithm_ttl_secs: u64,

    #[envconfig(from = "CACHE_RESULT_CAPACITY")]
    pub cache_result_capacity: usize,

    #[envconfig(from = "CACHE_RESULT_TTL_SECS")]
    pub cache_result_ttl_secs: u64,

    // Advanced Result Fusion
    #[envconfig(from = "FUSION_ALGORITHM")]
    pub fusion_algorithm: String,

    #[envconfig(from = "WRRF_K")]
    pub wrrf_k: f64,
    
    #[envconfig(from = "LOG_LEVEL")]
    pub log_level: String,
}
