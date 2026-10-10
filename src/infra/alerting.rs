use tracing::{error, warn};

#[allow(dead_code)]
pub struct PerformanceAlerts {
    pub max_latency_ms: f64,
    pub min_cache_hit_rate: f64,
    pub max_error_rate: f64,
}

#[allow(dead_code)]
impl PerformanceAlerts {
    pub fn new(max_latency_ms: f64, min_cache_hit_rate: f64, max_error_rate: f64) -> Self {
        Self {
            max_latency_ms,
            min_cache_hit_rate,
            max_error_rate,
        }
    }

    pub fn check_and_alert(
        &self,
        operation: &str,
        latency_ms: f64,
        cache_hit_rate: f64,
        error_rate: f64,
    ) {
        if latency_ms > self.max_latency_ms {
            warn!(
                "PERFORMANCE ALERT: {} latency {:.2}ms exceeds threshold {:.2}ms",
                operation, latency_ms, self.max_latency_ms
            );
        }

        if cache_hit_rate < self.min_cache_hit_rate {
            warn!(
                "PERFORMANCE ALERT: {} cache hit rate {:.2}% below threshold {:.2}%",
                operation,
                cache_hit_rate * 100.0,
                self.min_cache_hit_rate * 100.0
            );
        }

        if error_rate > self.max_error_rate {
            error!(
                "PERFORMANCE ALERT: {} error rate {:.2}% exceeds threshold {:.2}%",
                operation,
                error_rate * 100.0,
                self.max_error_rate * 100.0
            );
        }
    }
}

impl Default for PerformanceAlerts {
    fn default() -> Self {
        Self {
            max_latency_ms: 100.0,
            min_cache_hit_rate: 0.6,
            max_error_rate: 0.05,
        }
    }
}


