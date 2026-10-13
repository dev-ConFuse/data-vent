# Cloud Deployment Guide - ConFuse Retrieval Engine

This guide details the prerequisites, container build steps, Kubernetes cloud manifests, auto-scaling, and health validation for deploying `data-vent`.

## Prerequisites

1. **FalkorDB Instance**:
   - FalkorDB cloud instance or self-hosted Redis with FalkorDB module.
   - Credentials configured in Kubernetes secrets (`falkordb-secret`).
2. **NVIDIA NIM API Access**:
   - Access token with permission to call `nvidia/nemotron-3-embed-1b`.
   - Configured in Kubernetes secrets (`nvidia-secret`).
3. **Kubernetes Cluster**:
   - AWS EKS, GCP GKE, or Azure AKS with HorizontalPodAutoscaler support.

---

## Container Build

Build and tag the production Docker container:
```bash
docker build -t data-vent:0.3.0 -f Dockerfile .
```

---

## Deployment Manifests

The `deployment/` directory includes cloud-native specifications:

1. **`deployment/cloud-config.yaml`**:
   - Declares 3 replicas by default.
   - Resource requests: `500m` CPU, `1Gi` RAM; limits: `2000m` CPU, `2Gi` RAM.
   - Liveness probe targeting `/health` every 10s.
   - Readiness probe targeting `/health` every 5s.
   - ClusterIP service binding port 80 to port 3002.

2. **`deployment/autoscaling.yaml`**:
   - HorizontalPodAutoscaler targeting minimum 2 and maximum 10 pods.
   - Scaling thresholds: 70% CPU utilization, 80% Memory utilization.
   - Immediate scale-up behavior (0s stabilization) and 300s scale-down stabilization to prevent flapping.

Apply manifests:
```bash
kubectl apply -f deployment/cloud-config.yaml
kubectl apply -f deployment/autoscaling.yaml
```

---

## Health & Performance Validation

Verify that all pods are healthy and metrics are accessible:
```bash
# Health Check
curl http://localhost:3002/health

# Real-time Metrics & Cache Hit Rates
curl http://localhost:3002/metrics

# Runtime HNSW Tuning
curl -X POST http://localhost:3002/api/v1/hnsw/update \
  -H "Content-Type: application/json" \
  -d '{"mode": "balanced"}'
```
