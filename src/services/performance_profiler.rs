use std::time::Instant;

#[allow(dead_code)]
pub struct PerformanceProfiler {
    measurements: Vec<(String, f64)>,
}

#[allow(dead_code)]
impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    pub fn measure<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64() * 1000.0;
        self.measurements.push((name.to_string(), duration));
        tracing::info!("PERF: {} took {:.2}ms", name, duration);
        result
    }

    pub async fn measure_async<F, Fut, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = Instant::now();
        let result = f().await;
        let duration = start.elapsed().as_secs_f64() * 1000.0;
        self.measurements.push((name.to_string(), duration));
        tracing::info!("PERF: {} took {:.2}ms", name, duration);
        result
    }

    pub fn get_measurements(&self) -> &[(String, f64)] {
        &self.measurements
    }

    pub fn total_duration_ms(&self) -> f64 {
        self.measurements.iter().map(|(_, d)| *d).sum()
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}


