# Code Organization

This document describes the code organization and module structure of Phenix DB, following the unified vector + document + graph database architecture.

## Project Structure Overview

```
phenix-db/
├── src/
│   ├── lib.rs                    # Main library entry point
│   ├── bin/                      # Binary executables
│   │   ├── server.rs            # Main Phenix DB server binary
│   │   └── cli.rs               # CLI tools and utilities
│   ├── core/                     # Core unified database functionality
│   ├── storage/                  # Multi-tiered storage layer
│   ├── index/                    # Unified indexing and search
│   ├── shard/                    # Sharding and distribution
│   ├── worker/                   # Worker node functionality
│   ├── security/                 # Security and encryption
│   ├── api/                      # API layer
│   ├── observability/            # Monitoring and tracing
│   └── deployment/               # Deployment configurations
├── tests/                        # Integration and system tests
├── benches/                      # Performance benchmarks
├── docs/                         # Documentation
└── k8s/                          # Kubernetes manifests
```

## Core Module (`src/core/`)

The core module contains the fundamental data structures and interfaces for Phenix DB's unified data model.

### Module Structure

```rust
src/core/
├── mod.rs              # Module exports and re-exports
├── entity.rs           # Unified Entity (vector + metadata + edges)
├── vector.rs           # Vector operations and distance functions
├── metadata.rs         # JSONB metadata handling and indexing
├── edges.rs            # Graph edge management and traversal
├── transaction.rs      # ACID transaction logic
├── mvcc.rs             # Multi-version concurrency control
├── query.rs            # Unified query language and planning
├── traits.rs           # Shared abstractions and interfaces
└── error.rs            # Error handling hierarchy
```

### Key Components

#### Entity (`entity.rs`)
The unified Entity is the first-class citizen in Phenix DB:

```rust
pub struct Entity {
    pub id: EntityId,
    pub vector: Option<Vector>,        // Dense float32 array
    pub metadata: Option<JSONB>,       // Structured attributes
    pub edges: Option<Vec<Edge>>,      // Graph relationships
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub version: MVCCVersion,
    pub tenant_id: Option<String>,
}
```

**Design Principles:**
- Optional components allow flexible entity types
- MVCC versioning for concurrent access
- Tenant isolation for multi-tenant deployments
- Builder pattern for ergonomic construction

#### Vector (`vector.rs`)
Dense embeddings with performance optimizations:

```rust
pub struct Vector {
    pub dimensions: Vec<f32>,
    pub dimension_count: usize,
    pub norm: f32,
    pub compression_ratio: Option<f32>,
    pub encryption_algorithm: Option<EncryptionAlgorithm>,
    pub is_normalized: bool,
}
```

**Features:**
- SIMD-optimized distance calculations
- Multiple distance metrics (cosine, euclidean, manhattan)
- Normalization and validation
- Compression and encryption metadata

#### Edges (`edges.rs`)
Graph relationships between entities:

```rust
pub struct Edge {
    pub id: EdgeId,
    pub source_id: EntityId,
    pub target_id: EntityId,
    pub label: String,
    pub weight: f32,
    pub metadata: Option<JSONB>,
    // ... timestamps and versioning
}
```

**Capabilities:**
- Weighted, labeled relationships
- Rich metadata for complex relationships
- Bidirectional traversal support
- MVCC versioning

#### Traits (`traits.rs`)
Core abstractions defining system interfaces:

```rust
#[async_trait]
pub trait PhenixDBAPI: Send + Sync {
    async fn insert_entity(&mut self, entity: Entity) -> Result<EntityId>;
    async fn query(&self, query: UnifiedQuery) -> Result<QueryResult>;
    // ... other database operations
}

#[async_trait]
pub trait StorageTier: Send + Sync {
    async fn store_entity(&mut self, entity: Entity) -> Result<StorageLocation>;
    async fn retrieve_entity(&self, location: StorageLocation) -> Result<Option<Entity>>;
    // ... tier management operations
}
```

## Storage Module (`src/storage/`)

Multi-tiered storage system with intelligent caching and compression.

### Planned Structure

```rust
src/storage/
├── mod.rs              # Storage layer coordination
├── wal.rs              # Write-ahead log with signed entries
├── hot_tier.rs         # RAM/NVMe storage for entities and indexes
├── cold_tier.rs        # Object storage with chunked compression
├── compression.rs      # PQ, OPQ, quantization algorithms
└── tiering.rs          # Hot/cold promotion and demotion logic
```

### Design Principles

- **Hot Tier**: RAM and NVMe for frequently accessed entities
- **Cold Tier**: Object storage with compression for archival
- **Automatic Tiering**: LRU/LFU policies for promotion/demotion
- **Compression**: 70%+ storage reduction for cold tier
- **WAL**: Structured logging for durability and recovery

## Index Module (`src/index/`)

Unified indexing for vectors, metadata, and graph relationships.

### Planned Structure

```rust
src/index/
├── mod.rs              # Index coordination and management
├── hnsw.rs             # HNSW implementation for vectors
├── ivf_pq.rs           # IVF-PQ implementation with compression
├── metadata_index.rs   # B-tree and inverted indexes for metadata
├── graph_index.rs      # Adjacency lists and graph traversal
├── simd.rs             # SIMD optimizations for vector ops
└── unified_planner.rs  # Query planner for hybrid queries
```

### Index Types

- **HNSW**: Hierarchical Navigable Small World for vector similarity
- **IVF-PQ**: Inverted File with Product Quantization for memory efficiency
- **Metadata Indexes**: B-trees and inverted indexes for structured queries
- **Graph Indexes**: Adjacency lists with sampling for large-degree nodes

## API Module (`src/api/`)

External interfaces for client applications and SDK integration.

### Planned Structure

```rust
src/api/
├── mod.rs              # API coordination and main PhenixDB struct
├── grpc.rs             # gRPC interface for high-performance access
├── rest.rs             # REST API for web integration
├── protocol.rs         # Protocol definitions and versioning
└── unified_query.rs    # Unified query language and parsing
```

### API Design

- **gRPC**: High-performance binary protocol for production use
- **REST**: HTTP/JSON for web applications and development
- **Unified Queries**: Single query language across all interfaces
- **Authentication**: JWT and API key support
- **Rate Limiting**: Per-client quotas and throttling

## Naming Conventions

### Files and Modules
- **snake_case**: All file names and module names
- **Descriptive**: Clear indication of module purpose
- **Hierarchical**: Logical grouping of related functionality

### Rust Code
- **PascalCase**: Structs, enums, and traits (`Entity`, `PhenixDBAPI`)
- **snake_case**: Functions, variables, and modules (`insert_entity`, `query_vector`)
- **SCREAMING_SNAKE_CASE**: Constants (`MAX_VECTOR_DIM`, `DEFAULT_SHARD_SIZE`)

### Examples
```rust
// Good naming examples
pub struct Entity { ... }                    // PascalCase for types
pub trait PhenixDBAPI { ... }               // PascalCase for traits
pub async fn insert_entity(...) { ... }     // snake_case for functions
pub const MAX_VECTOR_DIM: usize = 4096;    // SCREAMING_SNAKE_CASE for constants

// Module organization
mod unified_planner;                         // snake_case module names
use crate::core::entity::Entity;            // Clear import paths
```

## Composition Patterns

### Unified Data Model
The core principle is composition over inheritance:

```rust
// Entity composes vector, metadata, and edges
pub struct Entity {
    pub vector: Option<Vector>,      // Optional vector component
    pub metadata: Option<JSONB>,     // Optional metadata component
    pub edges: Option<Vec<Edge>>,    // Optional graph component
}

// Traits define behavior, structs define data
trait VectorOperations {
    fn cosine_similarity(&self, other: &Vector) -> f32;
}

impl VectorOperations for Vector { ... }
```

### Async-First Design
All I/O operations use async/await:

```rust
#[async_trait]
pub trait EntityManager: Send + Sync {
    async fn create_entity(&mut self, entity: Entity) -> Result<EntityId>;
    async fn read_entity(&self, id: EntityId) -> Result<Option<Entity>>;
}
```

### Error Handling
Comprehensive error types with recovery strategies:

```rust
#[derive(Debug, Error)]
pub enum PhenixDBError {
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    
    #[error("Transaction error: {0}")]
    Transaction(#[from] TransactionError),
    // ... other error types
}

impl PhenixDBError {
    pub fn is_retryable(&self) -> bool { ... }
    pub fn retry_delay_ms(&self) -> Option<u64> { ... }
}
```

## Security Boundaries

### Tenant Isolation
Multi-tenant support enforced at module boundaries:

```rust
pub trait TenantScoped {
    fn tenant_id(&self) -> TenantId;
    fn validate_access(&self, operation: Operation) -> Result<(), AuthError>;
}

// All operations check tenant boundaries
impl EntityManager for ConcreteManager {
    async fn read_entity(&self, id: EntityId, tenant: TenantId) -> Result<Option<Entity>> {
        self.validate_tenant_access(id, tenant)?;
        // ... actual read operation
    }
}
```

### Encryption Integration
Encryption at every component boundary:

```rust
trait EncryptedStorage {
    async fn store_encrypted(&mut self, data: &[u8], key: &EncryptionKey) -> Result<StorageLocation>;
    async fn retrieve_decrypted(&self, location: StorageLocation, key: &EncryptionKey) -> Result<Vec<u8>>;
}
```

## Testing Organization

### Unit Tests
Co-located with implementation:

```rust
// In entity.rs
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entity_creation() { ... }
    
    #[test]
    fn test_entity_validation() { ... }
}
```

### Integration Tests
Separate test directory:

```
tests/
├── integration/        # Cross-module integration tests
├── performance/        # Performance benchmarks and load tests
├── security/           # Security and penetration tests
└── chaos/              # Chaos engineering and fault injection
```

### Benchmarks
Performance-focused tests:

```rust
// In benches/entity_operations.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_entity_creation(c: &mut Criterion) {
    c.bench_function("entity_creation", |b| {
        b.iter(|| Entity::new_random())
    });
}
```

## Development Workflow

### Module Development Order
1. **Core**: Data structures and interfaces
2. **Storage**: Persistence and caching
3. **Index**: Search and retrieval
4. **Shard**: Distribution and scaling
5. **API**: External interfaces
6. **Security**: Authentication and encryption
7. **Observability**: Monitoring and debugging

### Code Review Guidelines
- **Interface First**: Define traits before implementations
- **Test Coverage**: Unit tests for all public functions
- **Documentation**: Rustdoc for all public APIs
- **Performance**: Benchmark critical paths
- **Security**: Review for tenant isolation and data protection

This organization ensures maintainable, scalable, and secure code while supporting the unified vector + document + graph database architecture.