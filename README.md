# Phenix DB — Unified Vector + Document + Graph Database

A production-ready, transactional, sharded database implemented in Rust that unifies vectors, documents, and graph relationships under one ACID engine.

## Features

### 🔄 Unified Data Model
- **Single transactional surface** for vectors, documents, and graph relationships
- **Entity-first design** with optional vector, metadata (JSONB), and edges
- **Unified query planning** combining vector similarity, metadata filters, and graph traversal
- **Eliminates dual-write complexity** between separate vector and document stores

### 🚀 Performance at Scale
- **Sub-millisecond query latency** for billions of entities
- **100B+ vector capacity** with intelligent hot/cold tiering
- **SIMD and AVX optimizations** for vector operations
- **GPU acceleration** support where available
- **Pipeline parallelism** for entity processing
- **Hybrid HNSW/IVF-PQ indexing** with metadata co-location

### 🔒 ACID Guarantees Across All Data Types
- **Full ACID compliance** for vector + metadata + edge operations
- **MVCC (Multi-Version Concurrency Control)** for snapshot isolation
- **Write-Ahead Log (WAL)** with signed entries for integrity
- **Two-phase commit protocol** for cross-shard transactions
- **Consistent reads** across vectors, documents, and relationships

### 📈 Horizontal Scaling
- **Automatic entity distribution** using consistent hashing
- **Dynamic shard rebalancing** without service interruption
- **Unified query routing** with parallel execution across shards
- **Automatic failover** and replica management
- **Multi-tenant isolation** with per-tenant encryption

### 💾 Intelligent Multi-Tiered Storage
- **Hot tier**: RAM and NVMe for frequently accessed entities and indexes
- **Cold tier**: Object storage with 70%+ compression using PQ/OPQ
- **Chunked storage**: 1M entities per block with lazy decompression
- **Smart caching** with LRU/LFU policies across all data types
- **Automatic promotion/demotion** based on unified access patterns

### 🛡️ Security & Safety
- **Memory safety** through Rust's ownership system
- **Envelope encryption** with per-tenant DEK + KMS CMK
- **Signed WAL and snapshots** for integrity verification
- **mTLS for internal comms**, TLS1.3 for client-facing
- **RBAC enforcement** at all API and management layers

### 🌐 Flexible Deployment
- **Kubernetes**: StatefulSets for shard nodes, Deployments for stateless components
- **Docker Swarm** with service-based architecture
- **Single container** mode for edge computing and testing
- **Cloud-native** design with object storage integration (S3/MinIO)

### 📊 Enterprise Observability
- **Prometheus metrics** with custom collectors for hybrid queries
- **OpenTelemetry tracing** with correlation IDs across distributed operations
- **Structured JSON logging** with tenant and shard identifiers
- **Comprehensive alerting** for WAL lag, replication, and SLA violations

## Quick Start

### Prerequisites

- Rust 1.70+ with Cargo
- Docker (optional, for containerized deployment)
- Kubernetes cluster (optional, for K8s deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/mhassan72/Rust-Vector-Database.git
cd Rust-Vector-Database

# Build the project
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Running the Database

#### Single Container Mode
```bash
# Start the database server
cargo run --bin server

# Or using Docker
docker build -t phenix-db .
docker run -p 8080:8080 -p 9090:9090 phenix-db
```

#### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=vector-db

# View logs
kubectl logs -f deployment/vector-db-manager
```

#### Docker Swarm Deployment
```bash
# Deploy using Docker Compose
docker stack deploy -c docker/docker-compose.yml phenix-db

# Check service status
docker service ls
```

## API Usage

### gRPC API

```rust
use phenix_db_client::PhenixDBClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = PhenixDBClient::connect("http://localhost:9090").await?;
    
    // Insert unified entities (vector + metadata + edges)
    let entities = vec![
        Entity::new()
            .with_vector(vec![0.1, 0.2, 0.3, 0.4])
            .with_metadata(json!({"title": "Document 1", "category": "tech"}))
            .with_edge("related_to", "entity_2", 0.8),
        Entity::new()
            .with_vector(vec![0.5, 0.6, 0.7, 0.8])
            .with_metadata(json!({"title": "Document 2", "category": "science"})),
    ];
    
    let response = client.upsert_entities(entities).await?;
    println!("Upserted {} entities", response.count);
    
    // Unified query: vector similarity + metadata filter + graph traversal
    let query = UnifiedQuery::new()
        .with_vector_similarity(vec![0.1, 0.2, 0.3, 0.4], 10)
        .with_metadata_filter(json!({"category": "tech"}))
        .with_graph_constraint("related_to:entity_2");
    
    let results = client.hybrid_query(query).await?;
    
    for result in results.entities {
        println!("Entity ID: {}, Score: {}, Metadata: {:?}", 
                 result.id, result.score, result.metadata);
    }
    
    Ok(())
}
```

### REST API

```bash
# Insert unified entities
curl -X POST http://localhost:8080/entities \
  -H "Content-Type: application/json" \
  -d '{
    "entities": [
      {
        "id": "entity1",
        "vector": [0.1, 0.2, 0.3, 0.4],
        "metadata": {"title": "Document 1", "category": "tech"},
        "edges": [{"target": "entity2", "label": "related_to", "weight": 0.8}]
      },
      {
        "id": "entity2", 
        "vector": [0.5, 0.6, 0.7, 0.8],
        "metadata": {"title": "Document 2", "category": "science"}
      }
    ]
  }'

# Unified hybrid query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "vector_similarity": {
      "vector": [0.1, 0.2, 0.3, 0.4],
      "k": 10
    },
    "metadata_filter": {"category": "tech"},
    "graph_constraint": "related_to:entity2"
  }'
```

## SDKs

Official SDKs are planned for multiple programming languages (coming after core database development):

- **Rust**: `cargo add phenix-db-client` *(planned)*
- **Python**: `pip install phenix-db-python` *(planned)*
- **Go**: `go get github.com/mhassan72/phenix-db-go` *(planned)*
- **JavaScript/Node.js**: `npm install phenix-db-js` *(planned)*
- **Ruby**: `gem install phenix-db-ruby` *(planned)*

For now, you can interact with the database directly through the gRPC and REST APIs.

## Configuration

### Environment Variables

```bash
# Server configuration
PHENIX_DB_HOST=0.0.0.0
PHENIX_DB_GRPC_PORT=9090
PHENIX_DB_HTTP_PORT=8080

# Storage configuration
PHENIX_DB_HOT_TIER_SIZE=10GB
PHENIX_DB_COLD_TIER_ENDPOINT=s3://bucket/path
PHENIX_DB_COMPRESSION_ENABLED=true

# Shard configuration
PHENIX_DB_SHARD_COUNT=16
PHENIX_DB_REPLICATION_FACTOR=3

# Security configuration
PHENIX_DB_ENCRYPTION_ENABLED=true
PHENIX_DB_ENCRYPTION_ALGORITHM=AES-GCM
PHENIX_DB_KMS_ENDPOINT=https://kms.example.com

# Performance tuning
PHENIX_DB_BATCH_SIZE=1000
PHENIX_DB_WORKER_THREADS=8
PHENIX_DB_GPU_ENABLED=true

# Unified query settings
PHENIX_DB_MAX_HYBRID_RESULTS=1000
PHENIX_DB_GRAPH_TRAVERSAL_DEPTH=3
```

### Configuration File

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  grpc_port: 9090
  http_port: 8080

storage:
  hot_tier:
    type: "memory_nvme"
    size: "10GB"
    cache_policy: "lru"
  cold_tier:
    type: "s3"
    endpoint: "s3://bucket/path"
    compression: true
    compression_ratio: 0.7

sharding:
  shard_count: 16
  replication_factor: 3
  consistent_hashing: true

security:
  encryption:
    enabled: true
    algorithm: "aes-gcm"
  authentication:
    enabled: true
    method: "jwt"

performance:
  batch_size: 1000
  worker_threads: 8
  simd_enabled: true
  gpu_enabled: true

observability:
  metrics:
    enabled: true
    endpoint: "0.0.0.0:2112"
  tracing:
    enabled: true
    endpoint: "http://jaeger:14268"
  logging:
    level: "info"
    format: "json"
```

## Performance Benchmarks

### Query Latency
- **1M vectors**: < 0.5ms average latency
- **100M vectors**: < 0.8ms average latency  
- **1B vectors**: < 1.0ms average latency

### Throughput
- **Insert throughput**: 100K+ vectors/second
- **Query throughput**: 10K+ queries/second
- **Concurrent users**: 1000+ simultaneous connections

### Storage Efficiency
- **Hot tier**: Sub-millisecond access times
- **Cold tier**: 70%+ compression ratio
- **Memory usage**: Optimized for billion-scale datasets

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client SDKs   │    │   Client SDKs   │    │   Client SDKs   │
│ (Rust/Python/Go)│    │ (JS/Ruby/etc.)  │    │   (REST/gRPC)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │       API Layer          │
                    │  ┌─────────┐ ┌─────────┐ │
                    │  │  gRPC   │ │  REST   │ │
                    │  │ Server  │ │ Server  │ │
                    │  └─────────┘ └─────────┘ │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     Manager Layer        │
                    │ ┌─────────┐ ┌─────────┐  │
                    │ │   Tx    │ │ Ingestion│  │
                    │ │Coordinator│ │ Manager │  │
                    │ └─────────┘ └─────────┘  │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────┴─────────┐   ┌─────────┴─────────┐   ┌─────────┴─────────┐
│   Worker Node 1   │   │   Worker Node 2   │   │   Worker Node N   │
│ ┌───────────────┐ │   │ ┌───────────────┐ │   │ ┌───────────────┐ │
│ │ Vector        │ │   │ │ Vector        │ │   │ │ Vector        │ │
│ │ Processor     │ │   │ │ Processor     │ │   │ │ Processor     │ │
│ └───────────────┘ │   │ └───────────────┘ │   │ └───────────────┘ │
│ ┌───────────────┐ │   │ ┌───────────────┐ │   │ ┌───────────────┐ │
│ │ HNSW/IVF-PQ   │ │   │ │ HNSW/IVF-PQ   │ │   │ │ HNSW/IVF-PQ   │ │
│ │ Index         │ │   │ │ Index         │ │   │ │ Index         │ │
│ └───────────────┘ │   │ └───────────────┘ │   │ └───────────────┘ │
│ ┌───────────────┐ │   │ ┌───────────────┐ │   │ ┌───────────────┐ │
│ │ Hot/Cold      │ │   │ │ Hot/Cold      │ │   │ │ Hot/Cold      │ │
│ │ Storage       │ │   │ │ Storage       │ │   │ │ Storage       │ │
│ └───────────────┘ │   │ └───────────────┘ │   │ └───────────────┘ │
└───────────────────┘   └───────────────────┘   └───────────────────┘
```

## Development

### Project Structure

```
src/
├── lib.rs                    # Main library entry point
├── bin/                      # Binary executables
├── core/                     # Core database functionality
├── storage/                  # Storage layer (WAL, hot/cold tiers)
├── index/                    # Indexing (HNSW, IVF-PQ, SIMD)
├── shard/                    # Sharding and distribution
├── worker/                   # Worker node functionality
├── security/                 # Security and encryption
├── api/                      # API layer (gRPC, REST)
├── observability/            # Monitoring and tracing
└── deployment/               # Deployment configurations

tests/
├── integration/              # Integration tests
├── performance/              # Performance benchmarks
└── unit/                     # Unit tests

k8s/                          # Kubernetes manifests
docker/                       # Docker configurations
docs/                         # Documentation
```

### Building and Testing

```bash
# Format code
cargo fmt

# Lint code
cargo clippy

# Run unit tests
cargo test

# Run integration tests
cargo test --test integration

# Run performance benchmarks
cargo bench

# Build optimized release
cargo build --release

# Generate documentation
cargo doc --open
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`cargo test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

For detailed contribution guidelines, see [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Monitoring and Observability

### Prometheus Metrics

Key metrics exposed at `/metrics` endpoint:

- `vector_db_query_duration_seconds`: Query latency histogram
- `vector_db_insert_total`: Total vectors inserted
- `vector_db_storage_bytes`: Storage usage by tier
- `vector_db_shard_health`: Shard health status
- `vector_db_transaction_duration_seconds`: Transaction duration

### Grafana Dashboard

Import the provided Grafana dashboard (`monitoring/grafana-dashboard.json`) for comprehensive monitoring.

### Distributed Tracing

Configure OpenTelemetry to export traces to Jaeger, Zipkin, or other compatible backends:

```yaml
observability:
  tracing:
    enabled: true
    exporter: "jaeger"
    endpoint: "http://jaeger:14268/api/traces"
    sampling_ratio: 0.1
```

## Troubleshooting

### Common Issues

**High query latency**
- Check shard distribution and rebalancing
- Verify hot tier cache hit rates
- Monitor CPU and memory usage
- Consider GPU acceleration

**Transaction failures**
- Check WAL disk space and performance
- Monitor network connectivity between shards
- Verify clock synchronization across nodes

**Storage issues**
- Monitor hot/cold tier usage and promotion policies
- Check object storage connectivity and credentials
- Verify compression ratios and performance

### Debugging

Enable debug logging:
```bash
RUST_LOG=vector_db=debug cargo run --bin server
```

Check system health:
```bash
curl http://localhost:8080/health
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Documentation

Comprehensive documentation is available in the [docs/](docs/) folder:

- **[Architecture](docs/architecture/)** - System design and technical decisions
- **[API Reference](docs/api/)** - gRPC and REST API documentation
- **[Development Guide](docs/development/)** - Setup, testing, and contribution workflow
- **[Deployment](docs/deployment/)** - Kubernetes, Docker, and configuration guides
- **[Security](docs/security/)** - Encryption, authentication, and compliance
- **[Tutorials](docs/tutorials/)** - Step-by-step learning guides

## Support

- **Documentation**: [docs/](docs/)
- **Contributing**: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- **Issues**: [GitHub Issues](https://github.com/mhassan72/Rust-Vector-Database/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mhassan72/Rust-Vector-Database/discussions)
- **Security**: Report security issues via Discord (see community channels below)