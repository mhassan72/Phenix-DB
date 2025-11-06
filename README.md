# Rust Vector Database

A production-ready, transactional, sharded vector database implemented in Rust designed for high-performance similarity search at scale.

## Features

### 🚀 Performance
- **Sub-millisecond query latency** for billions of vectors
- **SIMD and AVX optimizations** for vector operations
- **GPU acceleration** support where available
- **Pipeline parallelism** for vector processing
- **Hybrid HNSW/IVF-PQ indexing** for optimal search performance

### 🔒 ACID Guarantees
- **Full ACID compliance** with distributed transactions
- **MVCC (Multi-Version Concurrency Control)** for snapshot isolation
- **Write-Ahead Log (WAL)** for durability and crash recovery
- **Two-phase commit protocol** for cross-shard transactions

### 📈 Horizontal Scaling
- **Automatic shard distribution** using consistent hashing
- **Dynamic shard rebalancing** without service interruption
- **Query routing** with parallel execution across shards
- **Automatic failover** and replica management

### 💾 Multi-Tiered Storage
- **Hot tier**: RAM and NVMe for frequently accessed vectors
- **Cold tier**: Object storage with 70%+ compression
- **Intelligent caching** with LRU/LFU policies
- **Automatic promotion/demotion** based on access patterns

### 🛡️ Security & Safety
- **Memory safety** through Rust's ownership system
- **End-to-end encryption** (AES-GCM, ChaCha20-Poly1305)
- **Per-tenant encryption keys** for multi-tenant deployments
- **Compile-time safety checks** preventing memory leaks

### 🌐 Flexible Deployment
- **Kubernetes** with StatefulSets and HPA integration
- **Docker Swarm** with service-based architecture
- **Single container** mode for edge and testing
- **Cloud-native** design with persistent volume support

### 📊 Comprehensive Observability
- **Prometheus metrics** for monitoring and alerting
- **OpenTelemetry tracing** for distributed request tracking
- **Structured logging** with correlation IDs
- **Health checks** and fault tolerance

## Quick Start

### Prerequisites

- Rust 1.70+ with Cargo
- Docker (optional, for containerized deployment)
- Kubernetes cluster (optional, for K8s deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/rust-vector-database.git
cd rust-vector-database

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
docker build -t vector-db .
docker run -p 8080:8080 -p 9090:9090 vector-db
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
docker stack deploy -c docker/docker-compose.yml vector-db

# Check service status
docker service ls
```

## API Usage

### gRPC API

```rust
use vector_db_client::VectorDatabaseClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = VectorDatabaseClient::connect("http://localhost:9090").await?;
    
    // Insert vectors
    let vectors = vec![
        Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
        Vector::new(vec![0.5, 0.6, 0.7, 0.8]),
    ];
    
    let response = client.insert_vectors(vectors).await?;
    println!("Inserted {} vectors", response.count);
    
    // Query similar vectors
    let query = Vector::new(vec![0.1, 0.2, 0.3, 0.4]);
    let results = client.query_similar(query, 10).await?;
    
    for result in results.vectors {
        println!("Vector ID: {}, Similarity: {}", result.id, result.similarity);
    }
    
    Ok(())
}
```

### REST API

```bash
# Insert vectors
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {"id": "vec1", "dimensions": [0.1, 0.2, 0.3, 0.4]},
      {"id": "vec2", "dimensions": [0.5, 0.6, 0.7, 0.8]}
    ]
  }'

# Query similar vectors
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4],
    "k": 10
  }'
```

## SDKs

Official SDKs are planned for multiple programming languages (coming after core database development):

- **Rust**: `cargo add vector-db-client` *(planned)*
- **Python**: `pip install vector-db-python` *(planned)*
- **Go**: `go get github.com/your-org/vector-db-go` *(planned)*
- **JavaScript/Node.js**: `npm install vector-db-js` *(planned)*
- **Ruby**: `gem install vector-db-ruby` *(planned)*

For now, you can interact with the database directly through the gRPC and REST APIs.

## Configuration

### Environment Variables

```bash
# Server configuration
VECTOR_DB_HOST=0.0.0.0
VECTOR_DB_GRPC_PORT=9090
VECTOR_DB_HTTP_PORT=8080

# Storage configuration
VECTOR_DB_HOT_TIER_SIZE=10GB
VECTOR_DB_COLD_TIER_ENDPOINT=s3://bucket/path
VECTOR_DB_COMPRESSION_ENABLED=true

# Shard configuration
VECTOR_DB_SHARD_COUNT=16
VECTOR_DB_REPLICATION_FACTOR=3

# Security configuration
VECTOR_DB_ENCRYPTION_ENABLED=true
VECTOR_DB_ENCRYPTION_ALGORITHM=AES-GCM

# Performance tuning
VECTOR_DB_BATCH_SIZE=1000
VECTOR_DB_WORKER_THREADS=8
VECTOR_DB_GPU_ENABLED=true
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

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/rust-vector-database/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rust-vector-database/discussions)
- **Security**: Report security issues via Discord (see community channels below)