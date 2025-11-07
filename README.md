# Phenix DB

A unified vector + document + graph database implemented in Rust, designed for sub-millisecond hybrid queries across billions of entities.

[![Build Status](https://github.com/phenix-db/phenix-db/workflows/CI/badge.svg)](https://github.com/phenix-db/phenix-db/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)

## 🚀 What is Phenix DB?

Phenix DB treats embeddings as first-class citizens while keeping metadata and relationships fully transactional and queryable. Unlike pure vector indexes or document stores with add-on vectors, Phenix DB provides:

- **Single Transactional Surface**: Vector + metadata + edges in one ACID transaction
- **Sub-millisecond Queries**: Hybrid queries combining vector similarity, metadata filtering, and graph traversal
- **100B+ Scale**: Intelligent hot/cold tiering with 70%+ compression
- **Memory Safety**: Built with Rust for zero-cost abstractions and compile-time safety

## ⚠️ Development Status

**Currently in active development** - Core interfaces and data structures are complete, but many features are still being implemented. See [docs/TODO.md](docs/TODO.md) for detailed progress.

### ✅ Completed
- Unified Entity data model (vector + metadata + edges)
- Core trait interfaces and error handling
- MVCC and transaction management
- Vector operations with distance calculations
- Comprehensive test suite (49 unit tests)

### 🚧 In Progress
- Storage layer implementation
- Vector indexing (HNSW/IVF-PQ)
- API layer (gRPC/REST)
- Distributed sharding

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   gRPC/REST     │    │   Manager       │    │   Worker        │
│   API Layer     │───▶│   Layer         │───▶│   Nodes         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │                        │
                               ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Shard         │    │   Storage       │
                       │   Manager       │    │   Tiers         │
                       └─────────────────┘    └─────────────────┘
```

### Key Components

- **Unified Entity**: First-class data structure containing optional vector, metadata (JSONB), and edges
- **Multi-Tier Storage**: Hot (RAM/NVMe) and cold (object storage) tiers with automatic promotion/demotion
- **Hybrid Indexing**: HNSW/IVF-PQ for vectors, B-trees for metadata, adjacency lists for graphs
- **Distributed Sharding**: Entity-aware sharding with consistent hashing and automatic rebalancing
- **ACID Transactions**: Full transactional guarantees with MVCC and distributed two-phase commit

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ 
- Cargo

### Installation

```bash
git clone https://github.com/phenix-db/phenix-db.git
cd phenix-db
cargo build --release
```

### Basic Usage

```rust
use phenix_db::{PhenixDB, Entity, UnifiedQuery, Vector};
use phenix_db::core::traits::PhenixDBAPI;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize database
    let mut db = PhenixDB::builder()
        .with_config_file("phenix.toml")
        .build()
        .await?;

    // Create unified entity with vector, metadata, and edges
    let entity = Entity::builder()
        .with_vector(vec![0.1; 384]) // 384-dimensional vector
        .with_metadata(json!({"title": "Sample Document", "category": "AI"}))
        .with_edge("related_to", Entity::new_random().id, 0.8)
        .build();

    // Insert entity with ACID guarantees
    let entity_id = db.insert_entity(entity).await?;

    // Perform unified query
    let query_vector = Vector::new(vec![0.1; 384]);
    let query = UnifiedQuery::builder()
        .vector_similarity(query_vector, 10)
        .metadata_equals("category", json!("AI"))
        .build();

    let results = db.query(query).await?;
    println!("Found {} matching entities", results.entities.len());

    Ok(())
}
```

### Running Tests

```bash
# Run all tests
cargo test

# Run benchmarks
cargo bench

# Check code
cargo check
cargo clippy
```

## 📊 Performance Targets

- **Query Latency**: Sub-millisecond for hot tier access
- **Hybrid Queries**: <5ms for complex vector+metadata+graph operations
- **Scale**: 100B+ entities with horizontal sharding
- **Throughput**: 100K+ entities/second ingestion per shard

## 🛠️ Development

### Project Structure

```
phenix-db/
├── src/
│   ├── core/           # Unified data model and interfaces
│   ├── storage/        # Multi-tiered storage layer
│   ├── index/          # Vector and metadata indexing
│   ├── shard/          # Distribution and sharding
│   ├── api/            # gRPC and REST interfaces
│   └── bin/            # Server and CLI binaries
├── tests/              # Integration tests
├── benches/            # Performance benchmarks
└── docs/               # Documentation
```

### Contributing

1. Check [docs/TODO.md](docs/TODO.md) for current development priorities
2. Read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines
3. Look at [GitHub Issues](https://github.com/phenix-db/phenix-db/issues) for specific tasks
4. Join [GitHub Discussions](https://github.com/phenix-db/phenix-db/discussions) for questions

### Development Setup

```bash
# Clone repository
git clone https://github.com/phenix-db/phenix-db.git
cd phenix-db

# Install dependencies
cargo build

# Run tests
cargo test

# Start development server
cargo run --bin phenix-db-server

# Use CLI tools
cargo run --bin phenix-db-cli -- --help
```

## 📚 Documentation

- **[Documentation Index](docs/index.md)**: Complete documentation overview
- **[Development TODO](docs/TODO.md)**: Current development status and priorities
- **[Code Organization](docs/development/code-organization.md)**: Module structure and patterns
- **[API Reference](docs/api/)**: gRPC and REST API documentation (coming soon)
- **[Deployment Guide](docs/deployment/)**: Kubernetes and Docker deployment (coming soon)

## 🔒 Security

Phenix DB is designed with security as a first-class concern:

- **Memory Safety**: Rust's ownership system prevents common vulnerabilities
- **Encryption**: AES-GCM and ChaCha20-Poly1305 for data at rest and in transit
- **Multi-tenant**: Per-tenant encryption keys and data isolation
- **Authentication**: JWT and API key support with RBAC
- **Audit Logging**: Comprehensive audit trails for compliance

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t phenix-db .

# Run container
docker run -p 8080:8080 -p 9090:9090 phenix-db
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=phenix-db
```

### Single Binary

```bash
# Run single-node deployment
./target/release/phenix-db-server --config phenix.toml
```

## 🤝 Community

- **GitHub**: [phenix-db/phenix-db](https://github.com/phenix-db/phenix-db)
- **Issues**: [Report bugs and request features](https://github.com/phenix-db/phenix-db/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/phenix-db/phenix-db/discussions)
- **Documentation**: [Complete docs](docs/index.md)

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Phenix DB builds upon research and open-source projects in vector databases, distributed systems, and Rust ecosystem:

- **HNSW**: Hierarchical Navigable Small World graphs for vector similarity
- **IVF-PQ**: Inverted File with Product Quantization for memory efficiency
- **Raft**: Consensus algorithm for distributed coordination
- **RocksDB**: Embedded key-value storage engine
- **Tokio**: Async runtime for Rust

---

**Status**: Active Development | **License**: Apache 2.0 | **Language**: Rust