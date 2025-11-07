//! # Vector Operations Benchmarks
//!
//! Benchmarks for vector operations including similarity calculations,
//! normalization, and distance metrics.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use phenix_db::core::vector::Vector;

fn benchmark_vector_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_creation");
    
    for size in [128, 256, 512, 1024, 2048].iter() {
        group.bench_with_input(BenchmarkId::new("new", size), size, |b, &size| {
            let dimensions: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            b.iter(|| {
                Vector::new(black_box(dimensions.clone()))
            });
        });
        
        group.bench_with_input(BenchmarkId::new("new_normalized", size), size, |b, &size| {
            let dimensions: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            b.iter(|| {
                Vector::new_normalized(black_box(dimensions.clone()))
            });
        });
    }
    
    group.finish();
}

fn benchmark_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");
    
    let v1 = Vector::new((0..1024).map(|i| i as f32 * 0.1).collect());
    let v2 = Vector::new((0..1024).map(|i| (i + 100) as f32 * 0.1).collect());
    
    group.bench_function("cosine_similarity", |b| {
        b.iter(|| {
            v1.cosine_similarity(black_box(&v2)).unwrap()
        });
    });
    
    group.bench_function("dot_product", |b| {
        b.iter(|| {
            v1.dot_product(black_box(&v2))
        });
    });
    
    group.bench_function("euclidean_distance", |b| {
        b.iter(|| {
            v1.euclidean_distance(black_box(&v2)).unwrap()
        });
    });
    
    group.bench_function("manhattan_distance", |b| {
        b.iter(|| {
            v1.manhattan_distance(black_box(&v2)).unwrap()
        });
    });
    
    group.finish();
}

fn benchmark_vector_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_normalization");
    
    for size in [128, 256, 512, 1024, 2048].iter() {
        group.bench_with_input(BenchmarkId::new("normalize", size), size, |b, &size| {
            b.iter_batched(
                || {
                    let dimensions: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
                    Vector::new(dimensions)
                },
                |mut vector| {
                    vector.normalize();
                    vector
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_vector_creation,
    benchmark_vector_operations,
    benchmark_vector_normalization
);
criterion_main!(benches);