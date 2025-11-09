//! Core data structures and types for Phenix-DB
//!
//! This module contains the fundamental building blocks of the memory substrate:
//! - Entity: Unified data structure containing vector + metadata + edges
//! - Vector: High-dimensional embeddings
//! - Edge: Probabilistic relationships between entities
//! - Error types and result handling
//! - Configuration management

pub mod config;
pub mod entity;
pub mod error;
pub mod metadata;
pub mod mvcc;
pub mod query;
pub mod traits;
pub mod transaction;
pub mod types;
pub mod vector;
pub mod edges;

// Re-export commonly used types
pub use entity::{Entity, AccessStatistics, CompressionMetadata};
pub use error::{MemorySubstrateError, Result};
pub use types::{EntityId, NodeId, ShardId, ClusterId, TransactionId, Timestamp};
pub use vector::Vector;
pub use edges::{Edge, ProbabilisticEdge};
pub use metadata::Metadata;
pub use mvcc::{MVCCVersion, EntityVersion};
pub use query::{CognitiveQuery, QueryType, QueryOptions};
pub use transaction::Transaction;
pub use config::Config;

/// Memory tier enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MemoryTier {
    /// Hot tier: RAM/NVMe, <1ms access time
    Hot,
    /// Warm tier: NVMe/SSD, 1-10ms access time
    Warm,
    /// Cold tier: Object storage, 10-100ms access time, compressed
    Cold,
}

impl MemoryTier {
    /// Get the expected latency range for this tier
    pub fn latency_ms(&self) -> (f64, f64) {
        match self {
            MemoryTier::Hot => (0.0, 1.0),
            MemoryTier::Warm => (1.0, 10.0),
            MemoryTier::Cold => (10.0, 100.0),
        }
    }
}
