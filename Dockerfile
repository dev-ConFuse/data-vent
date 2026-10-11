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

# ---------------------------------------------------------------------------
# Dependency caching layer
# ---------------------------------------------------------------------------
COPY Cargo.toml Cargo.lock* ./

# Create dummy src for dependency caching
RUN mkdir -p src && \
    echo 'fn main() {}' > src/main.rs && \
    cargo build --release 2>/dev/null || true

# ---------------------------------------------------------------------------
# Real build
# ---------------------------------------------------------------------------
RUN rm -rf src/*
COPY src/ ./src/

# Force Cargo to rebuild the actual binary
RUN cargo clean -p data-vent 2>/dev/null; true
RUN cargo build --release

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

