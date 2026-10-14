use serde::Serialize;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
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

// Per-request observability: correlation IDs and structured access logs.
//
// Why: every HTTP request produces exactly one access record classified by
// outcome — `success` (2xx), `redirect` (3xx + target), `client_error` (4xx)
// or `server_error` (5xx) — so failures and redirects are as visible as
// successes. The `x-correlation-id` header is honoured from upstream callers
// (or generated) and echoed on the response, so one request can be traced
// across every service it touches.

use std::time::Instant;

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use tracing::{error, warn, Instrument};
use uuid::Uuid;

/// Correlation-ID header honoured from upstream callers and echoed back.
pub const CORRELATION_HEADER: &str = "x-correlation-id";

/// Read a required env var, failing fast and naming it (env-policy).
fn required_env(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_else(|_| panic!("Missing required environment variable: {name}"))
}

/// Redact secret-looking query values so access logs never leak credentials.
fn sanitize_query(query: Option<&str>) -> Option<String> {
    query.map(|raw| {
        raw.split('&')
            .map(|pair| {
                let key = pair.split('=').next().unwrap_or("");
                if ["token", "key", "secret", "password", "auth"]
                    .iter()
                    .any(|needle| key.to_ascii_lowercase().contains(needle))
                {
                    format!("{key}=[redacted]")
                } else {
                    pair.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("&")
    })
}

/// Initialise structured tracing from env (env-policy: no baked-in fallback).
///
/// `RUST_LOG` (full EnvFilter directives) overrides `LOG_LEVEL`; `LOG_FORMAT`
/// selects `json` (machine-parseable) or `console` (human-readable) output.
pub fn init_tracing() {
    let filter = std::env::var("RUST_LOG")
        .or_else(|_| std::env::var("LOG_LEVEL"))
        .unwrap_or_else(|_| {
            panic!("Missing required environment variable: LOG_LEVEL (or RUST_LOG)")
        });
    let format = required_env("LOG_FORMAT");
    let env_filter = tracing_subscriber::EnvFilter::new(filter);
    if format == "json" {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .flatten_event(true)
                    .with_current_span(true)
                    .with_span_list(false),
            )
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().with_target(true))
            .init();
    }
}

/// Access-log middleware: timing + outcome classification for every request.
pub async fn access_log_middleware(mut request: Request, next: Next) -> Response {
    let correlation_id = request
        .headers()
        .get(CORRELATION_HEADER)
        .or_else(|| request.headers().get("x-request-id"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    if let Ok(val) = correlation_id.parse() {
        request.headers_mut().insert(CORRELATION_HEADER, val);
    }

    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let query = sanitize_query(request.uri().query());
    let user_agent = request
        .headers()
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    let span = tracing::info_span!(
        "request",
        correlation_id = %correlation_id,
        method = %method,
        path = %path,
    );
    let start = Instant::now();
    info!(query = ?query, user_agent = ?user_agent, "request started");
    let mut response = next.run(request).instrument(span).await;
    let duration_ms = start.elapsed().as_millis() as u64;
    let status = response.status().as_u16();
    if let Ok(val) = correlation_id.parse() {
        response.headers_mut().insert(CORRELATION_HEADER, val);
    }

    match status {
        200..=299 => info!(status, duration_ms, outcome = "success", "request completed"),
        300..=399 => warn!(
            status,
            duration_ms,
            outcome = "redirect",
            location = ?response.headers().get("location"),
            "request redirected"
        ),
        400..=499 => warn!(status, duration_ms, outcome = "client_error", "request failed"),
        _ => error!(status, duration_ms, outcome = "server_error", "request failed"),
    }
    response
}


