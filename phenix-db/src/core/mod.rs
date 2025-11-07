//! # Core Module - Unified Database Functionality
//!
//! This module contains the core unified database functionality for Phenix DB.
//! It provides the fundamental data structures and interfaces for managing
//! vectors, metadata, and graph relationships as first-class unified entities.
//!
//! ## Key Components
//! - **Entity**: Unified data structure containing optional vector, metadata (JSONB), and edges
//! - **Transaction**: ACID transaction management with MVCC support
//! - **Traits**: Core abstractions for database operations and storage tiers
//! - **Error Handling**: Comprehensive error types with recovery strategies
//!
//! ## Architecture Integration
//! ```text
//! ┌─────────────────┐
//! │     Entity      │ ◄─── Unified first-class citizen
//! │  ┌───────────┐  │
//! │  │  Vector   │  │ ◄─── Dense float32 embeddings
//! │  └───────────┘  │
//! │  ┌───────────┐  │
//! │  │ Metadata  │  │ ◄─── JSONB structured data
//! │  └───────────┘  │
//! │  ┌───────────┐  │
//! │  │   Edges   │  │ ◄─── Graph relationships
//! │  └───────────┘  │
//! └─────────────────┘
//! ```
//!
//! ## Performance Characteristics
//! - **Entity Operations**: Sub-millisecond for hot tier access
//! - **Transaction Latency**: ~1-5ms for distributed ACID operations
//! - **Memory Usage**: Optimized for 100B+ entity scale with tiered storage
//!
//! ## Security Considerations
//! - All entities support per-tenant encryption with envelope encryption
//! - MVCC provides snapshot isolation for concurrent access
//! - Audit logging for all entity lifecycle events

pub mod entity;
pub mod vector;
pub mod metadata;
pub mod edges;
pub mod transaction;
pub mod mvcc;
pub mod query;
pub mod traits;
pub mod error;

// Re-export commonly used types
pub use entity::{Entity, EntityBuilder, EntityId};
pub use vector::{Vector, VectorId};
pub use edges::{Edge, EdgeId};
pub use transaction::{Transaction, TransactionId, TransactionStatus};
pub use mvcc::{MVCCVersion, Snapshot, VersionInfo};
pub use query::{UnifiedQuery, UnifiedQueryBuilder, QueryResult};
pub use traits::{PhenixDBAPI, EntityManager, UnifiedQueryPlanner, StorageTier};
pub use error::{PhenixDBError, Result};