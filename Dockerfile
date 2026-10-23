# ==============================================================================
# Data Vent Service - Dockerfile
# ==============================================================================
# Multi-stage build for Rust retrieval engine
# Port: 3002 (HTTP)
# ==============================================================================

# Stage 1: Rust builder
FROM rust:1.94-slim-bookworm AS rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Tolerate flaky links when talking to crates.io: retry hard, never abort a
# transfer just because it is momentarily slow, and use the git CLI fallback.
ENV CARGO_NET_RETRY=10 \
    CARGO_HTTP_TIMEOUT=120 \
    CARGO_HTTP_LOW_SPEED_LIMIT=1 \
    CARGO_NET_GIT_FETCH_WITH_CLI=true

# ---------------------------------------------------------------------------
# Dependency caching layer
# ---------------------------------------------------------------------------
COPY Cargo.toml Cargo.lock* ./

# Dummy crate for dependency caching. Cargo.toml declares BOTH a lib
# (src/lib.rs) and a bin (src/main.rs), plus an explicit [[bench]] target, so
# all three must exist or `cargo build` aborts before fetching a single crate.
# Do not mask failures here: a failed warm-up used to be cached as an empty
# registry (via `|| true`), which forced every later build to re-download all
# dependencies over a flaky link.
RUN mkdir -p src benches && \
    echo 'fn main() {}' > src/main.rs && \
    echo '' > src/lib.rs && \
    echo '' > benches/hnsw_benchmark.rs

RUN cargo fetch
RUN cargo build --release

# ---------------------------------------------------------------------------
# Real build
# ---------------------------------------------------------------------------
RUN rm -rf src/*
COPY src/ ./src/
# Bench targets declared in Cargo.toml must exist for the manifest to load
COPY benches/ ./benches/

# Force Cargo to rebuild the actual binary
RUN cargo clean -p data-vent 2>/dev/null; true
RUN find src -type f -exec touch {} + && cargo build --release

# ==============================================================================
# Stage 2: Runtime image
# ==============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dumb-init \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
WORKDIR /app
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app appuser

# Copy the compiled Rust binary
COPY --from=rust-builder --chown=appuser:appgroup /app/target/release/data-vent /usr/local/bin/data-vent

# Ensure correct permissions
RUN chown -R appuser:appgroup /app

# PORT is supplied at runtime via .map.env / .secret.env (see .cline/rules/env-policy.md)

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 3002

# Use dumb-init as PID 1 for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

CMD ["data-vent"]

