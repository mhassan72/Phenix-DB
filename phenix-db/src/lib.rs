//! # Phenix DB - Unified Vector + Document + Graph Database
//!
//! Phenix DB is a production-ready, unified vector + document + graph database designed for 
//! sub-millisecond hybrid queries across billions of entities. The system treats embeddings 
//! as first-class citizens while keeping metadata and relationships fully transactional and queryable.
//!
//! ## Key Features
//! - **Unified Data Model**: Single transactional surface for vectors, metadata (JSONB), and graph relationships
//! - **ACID Compliance**: Full transactional guarantees across all data types with MVCC and distributed two-phase commit
//! - **Performance First**: Sub-millisecond hybrid query latency through SIMD optimizations and intelligent caching
//! - **Horizontal Scalability**: Entity-aware sharding with consistent hashing and automatic rebalancing
//! - **Multi-Tier Storage**: Intelligent hot/cold tier management with 70%+ compression for 100B+ entity scale
//!
//! ## Architecture Overview
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   gRPC/REST     │    │   Manager       │    │   Worker        │
//! │   API Layer     │───▶│   Layer         │───▶│   Nodes         │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!                                │                        │
//!                                ▼                        ▼
//!                        ┌─────────────────┐    ┌─────────────────┐
//!                        │   Shard         │    │   Storage       │
//!                        │   Manager       │    │   Tiers         │
//!                        └─────────────────┘    └─────────────────┘
//! ```
//!
//! ## Quick Start
//! ```rust,no_run
//! use phenix_db::{PhenixDB, Entity, UnifiedQuery, Vector};
//! use phenix_db::core::traits::PhenixDBAPI;
//! use phenix_db::core::metadata::MetadataQuery;
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize database
//!     let mut db = PhenixDB::builder()
//!         .with_config_file("phenix.toml")
//!         .build()
//!         .await?;
//!
//!     // Create unified entity with vector, metadata, and edges
//!     let entity = Entity::builder()
//!         .with_vector(vec![0.1; 384]) // 384-dimensional vector
//!         .with_metadata(json!({"title": "Sample Document", "category": "AI"}))
//!         .with_edge("related_to", Entity::new_random().id, 0.8)
//!         .build();
//!
//!     // Insert entity with ACID guarantees
//!     let entity_id = db.insert_entity(entity).await?;
//!
//!     // Perform unified query with vector similarity and metadata filtering
//!     let query_vector = Vector::new(vec![0.1; 384]);
//!     let metadata_filter = MetadataQuery::Equals {
//!         field: "category".to_string(),
//!         value: json!("AI"),
//!     };
//!     
//!     let query = UnifiedQuery::builder()
//!         .vector_similarity(query_vector, 10)
//!         .metadata_filter(metadata_filter)
//!         .build();
//!
//!     let results = db.query(query).await?;
//!     println!("Found {} matching entities", results.entities.len());
//!
//!     Ok(())
//! }
//! ```

pub mod core;
pub mod storage;
pub mod index;
pub mod shard;
pub mod worker;
pub mod security;
pub mod api;
pub mod observability;
pub mod deployment;

// Re-export core types for convenience
pub use core::{
    entity::{Entity, EntityBuilder, EntityId},
    edges::Edge,
    traits::{PhenixDBAPI, EntityManager, UnifiedQueryPlanner, StorageTier},
    transaction::{Transaction, TransactionId, TransactionStatus},
    vector::{Vector, VectorId},
    query::{UnifiedQuery, UnifiedQueryBuilder, QueryResult},
    error::{PhenixDBError, Result},
};

// Re-export main database interface
pub use api::PhenixDB;

/// Current version of Phenix DB
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration values
pub mod defaults {
    use std::time::Duration;

    /// Default vector dimensions
    pub const DEFAULT_VECTOR_DIMENSIONS: usize = 384;
    
    /// Maximum supported vector dimensions
    pub const MAX_VECTOR_DIMENSIONS: usize = 4096;
    
    /// Minimum supported vector dimensions
    pub const MIN_VECTOR_DIMENSIONS: usize = 128;
    
    /// Default query timeout
    pub const DEFAULT_QUERY_TIMEOUT: Duration = Duration::from_millis(1000);
    
    /// Default batch size for entity operations
    pub const DEFAULT_BATCH_SIZE: usize = 1000;
    
    /// Default hot tier cache size (in MB)
    pub const DEFAULT_HOT_TIER_SIZE_MB: usize = 1024;
}