//! # Entity Operations Benchmarks
//!
//! Benchmarks for unified entity operations including creation, validation,
//! and serialization/deserialization.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use phenix_db::core::{
    entity::{Entity, EntityBuilder},
    vector::Vector,
    edges::Edge,
};
use serde_json::json;

fn benchmark_entity_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_creation");
    
    group.bench_function("empty_entity", |b| {
        b.iter(|| {
            Entity::new_random()
        });
    });
    
    group.bench_function("entity_with_vector", |b| {
        let vector = Vector::new((0..384).map(|i| i as f32 * 0.1).collect());
        b.iter(|| {
            Entity::builder()
                .with_vector(black_box(vector.dimensions.clone()))
                .build()
        });
    });
    
    group.bench_function("entity_with_metadata", |b| {
        let metadata = json!({
            "title": "Test Document",
            "category": "benchmark",
            "tags": ["test", "performance"],
            "score": 95.5,
            "nested": {
                "field1": "value1",
                "field2": 42
            }
        });
        b.iter(|| {
            Entity::builder()
                .with_metadata(black_box(metadata.clone()))
                .build()
        });
    });
    
    group.bench_function("full_entity", |b| {
        let vector = Vector::new((0..384).map(|i| i as f32 * 0.1).collect());
        let metadata = json!({
            "title": "Full Entity",
            "category": "complete",
            "score": 88.7
        });
        b.iter(|| {
            Entity::builder()
                .with_vector(black_box(vector.dimensions.clone()))
                .with_metadata(black_box(metadata.clone()))
                .with_edge("related_to", black_box(Entity::new_random().id), 0.8)
                .build()
        });
    });
    
    group.finish();
}

fn benchmark_entity_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_validation");
    
    let simple_entity = Entity::builder()
        .with_vector((0..256).map(|i| i as f32 * 0.1).collect())
        .build();
    
    let complex_entity = Entity::builder()
        .with_vector((0..1024).map(|i| i as f32 * 0.1).collect())
        .with_metadata(json!({
            "title": "Complex Entity",
            "nested": {
                "deep": {
                    "field": "value"
                }
            },
            "array": [1, 2, 3, 4, 5]
        }))
        .with_edge("edge1", Entity::new_random().id, 0.9)
        .with_edge("edge2", Entity::new_random().id, 0.7)
        .build();
    
    group.bench_function("simple_entity", |b| {
        b.iter(|| {
            black_box(&simple_entity).validate().unwrap()
        });
    });
    
    group.bench_function("complex_entity", |b| {
        b.iter(|| {
            black_box(&complex_entity).validate().unwrap()
        });
    });
    
    group.finish();
}

fn benchmark_entity_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_serialization");
    
    let entity = Entity::builder()
        .with_vector((0..512).map(|i| i as f32 * 0.1).collect())
        .with_metadata(json!({
            "title": "Serialization Test",
            "data": {
                "field1": "value1",
                "field2": 42,
                "array": [1, 2, 3, 4, 5]
            }
        }))
        .build();
    
    group.bench_function("serialize_json", |b| {
        b.iter(|| {
            serde_json::to_string(black_box(&entity)).unwrap()
        });
    });
    
    let serialized = serde_json::to_string(&entity).unwrap();
    group.bench_function("deserialize_json", |b| {
        b.iter(|| {
            serde_json::from_str::<Entity>(black_box(&serialized)).unwrap()
        });
    });
    
    group.finish();
}

fn benchmark_edge_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_operations");
    
    let source_id = Entity::new_random().id;
    let target_id = Entity::new_random().id;
    
    group.bench_function("edge_creation", |b| {
        b.iter(|| {
            Edge::new(
                black_box(source_id),
                black_box(target_id),
                black_box("test_edge".to_string()),
                black_box(0.8),
            )
        });
    });
    
    let edge = Edge::new(source_id, target_id, "test_edge".to_string(), 0.8);
    group.bench_function("edge_validation", |b| {
        b.iter(|| {
            black_box(&edge).validate().unwrap()
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_entity_creation,
    benchmark_entity_validation,
    benchmark_entity_serialization,
    benchmark_edge_operations
);
criterion_main!(benches);