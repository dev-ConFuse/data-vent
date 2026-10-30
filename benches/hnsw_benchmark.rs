use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use data_vent::services::vector_search::HNSWConfig;

fn hnsw_configuration_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_configurations");

    for mode in ["low_latency", "balanced", "high_recall"].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(mode), mode, |b, &mode| {
            b.iter(|| {
                let config = HNSWConfig::from_mode(mode);
                black_box(config)
            })
        });
    }

    group.finish();
}

fn vector_search_cypher_generation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_cypher_generation");

    let modes = ["low_latency", "balanced", "high_recall"];
    for mode in modes.iter() {
        let config = HNSWConfig::from_mode(mode);
        group.bench_with_input(BenchmarkId::from_parameter(mode), &config, |b, cfg| {
            b.iter(|| {
                let cypher = format!(
                    "CREATE VECTOR INDEX FOR (c:Vector_Chunk) ON (c.embeddings) \
                     OPTIONS {{dimension: 2048, similarityFunction: '{}', M: {}, efConstruction: {}, efRuntime: {}}}",
                    cfg.m, cfg.ef_construction, cfg.ef_runtime, cfg.similarity_function
                );
                black_box(cypher)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    hnsw_configuration_benchmark,
    vector_search_cypher_generation_benchmark
);
criterion_main!(benches);
