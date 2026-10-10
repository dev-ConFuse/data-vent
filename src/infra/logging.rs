use serde::Serialize;
use tracing::info;

#[allow(dead_code)]
#[derive(Serialize, Debug, Clone)]
pub struct PerformanceLog {
    pub operation: String,
    pub duration_ms: f64,
    pub success: bool,
    pub cache_hit: bool,
    pub result_count: usize,
    pub metadata: serde_json::Value,
}

#[allow(dead_code)]
pub fn log_performance(log: PerformanceLog) {
    info!(
        operation = %log.operation,
        duration_ms = log.duration_ms,
        success = log.success,
        cache_hit = log.cache_hit,
        result_count = log.result_count,
        metadata = %serde_json::to_string(&log.metadata).unwrap_or_default(),
        "PERFORMANCE_LOG"
    );
}

#[allow(dead_code)]
pub fn log_retrieval_performance(
    operation: &str,
    duration_ms: f64,
    success: bool,
    cache_hit: bool,
    result_count: usize,
) {
    log_performance(PerformanceLog {
        operation: operation.to_string(),
        duration_ms,
        success,
        cache_hit,
        result_count,
        metadata: serde_json::json!({}),
    });
}


