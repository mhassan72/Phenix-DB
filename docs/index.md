# Phenix DB Documentation

Welcome to the Phenix DB documentation. Phenix DB is a unified vector + document + graph database implemented in Rust, designed for sub-millisecond hybrid queries across billions of entities.

## Current Status

**⚠️ Development Phase**: Phenix DB is currently in active development. The core interfaces and data structures have been implemented, but many features are still being built.

### Completed Components
- ✅ Unified Entity data model (vector + metadata + edges)
- ✅ Core trait interfaces (PhenixDBAPI, EntityManager, UnifiedQueryPlanner, StorageTier)
- ✅ MVCC and transaction management structures
- ✅ Error handling hierarchy with recovery strategies
- ✅ Vector operations with SIMD optimizations
- ✅ Graph edge management and traversal
- ✅ Unified query language and planning
- ✅ Comprehensive test suite

### In Development
- 🚧 Storage layer implementation (hot/cold tiers)
- 🚧 Vector indexing (HNSW/IVF-PQ)
- 🚧 Shard management and distribution
- 🚧 API layer (gRPC/REST)
- 🚧 Security and encryption
- 🚧 Observability and monitoring

## Table of Contents

### Getting Started
- [Quick Start Guide](getting-started.md) 🚧
- [Installation](installation.md) 🚧
- [Configuration](configuration.md) 🚧

### Architecture
- [System Overview](architecture/overview.md) 🚧
- [Unified Data Model](architecture/data-model.md) 🚧
- [Storage Architecture](architecture/storage-architecture.md) 🚧
- [Query Planning](architecture/query-planning.md) 🚧
- [Security Model](architecture/security-model.md) 🚧
- [Scaling Strategy](architecture/scaling-strategy.md) 🚧

### API Reference
- [gRPC API](api/grpc-reference.md) 🚧
- [REST API](api/rest-reference.md) 🚧
- [Unified Queries](api/unified-queries.md) 🚧
- [SDK Examples](api/sdk-examples/) 🚧

### Development
- [Development Setup](development/getting-started.md) 🚧
- [Code Organization](development/code-organization.md) ✅
- [Testing Guide](development/testing-guide.md) 🚧
- [Performance Tuning](development/performance-tuning.md) 🚧
- [Debugging Guide](development/debugging-guide.md) 🚧

### Deployment
- [Kubernetes Deployment](deployment/kubernetes.md) 🚧
- [Docker Deployment](deployment/docker.md) 🚧
- [Configuration Management](deployment/configuration.md) 🚧
- [Monitoring Setup](deployment/monitoring.md) 🚧
- [Backup & Recovery](deployment/backup-recovery.md) 🚧

### Security
- [Encryption](security/encryption.md) 🚧
- [Authentication](security/authentication.md) 🚧
- [Tenant Isolation](security/tenant-isolation.md) 🚧
- [Compliance](security/compliance.md) 🚧

### Tutorials
- [Creating Your First Entity](tutorials/first-entity.md) 🚧
- [Building Hybrid Queries](tutorials/hybrid-queries.md) 🚧
- [Scaling Your Deployment](tutorials/scaling-deployment.md) 🚧
- [Migration Guide](tutorials/migration-guide.md) 🚧

## Key Features

### Unified Data Model
- **Single Transactional Surface**: Vector + metadata + edges in one ACID transaction
- **First-Class Entities**: Embeddings, documents, and relationships as unified entities
- **MVCC Support**: Multi-version concurrency control across all data types

### Performance & Scale
- **Sub-millisecond Queries**: Hybrid queries combining vector similarity, metadata filtering, and graph traversal
- **100B+ Entity Scale**: Intelligent hot/cold tiering with 70%+ compression
- **Horizontal Scaling**: Entity-aware sharding with automatic rebalancing

### Developer Experience
- **Multiple APIs**: gRPC and REST interfaces with comprehensive SDKs
- **Rich Query Language**: Unified queries across vectors, metadata, and graphs
- **Memory Safety**: Built with Rust for zero-cost abstractions and compile-time safety

### Enterprise Ready
- **ACID Compliance**: Full transactional guarantees with distributed two-phase commit
- **Multi-tenant**: Per-tenant encryption and isolation
- **Cloud Native**: Kubernetes-first with Docker Swarm and single-container options

## Quick Example

```rust
use phenix_db::{PhenixDB, Entity, UnifiedQuery, Vector};
use phenix_db::core::traits::PhenixDBAPI;
use serde_json::json;

// Create unified entity
let entity = Entity::builder()
    .with_vector(vec![0.1; 384])
    .with_metadata(json!({"title": "Document", "category": "AI"}))
    .with_edge("related_to", other_entity_id, 0.8)
    .build();

// Insert with ACID guarantees
let entity_id = db.insert_entity(entity).await?;

// Hybrid query
let query = UnifiedQuery::builder()
    .vector_similarity(query_vector, 10)
    .metadata_filter(metadata_query)
    .build();

let results = db.query(query).await?;
```

## Community & Support

- **GitHub**: [phenix-db/phenix-db](https://github.com/phenix-db/phenix-db)
- **Documentation**: This site
- **Issues**: [GitHub Issues](https://github.com/phenix-db/phenix-db/issues)
- **Discussions**: [GitHub Discussions](https://github.com/phenix-db/phenix-db/discussions)

## License

Phenix DB is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.