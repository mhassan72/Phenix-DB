// Core module - Fundamental data structures and types for Phenix-DB
//
// This module contains the core cognitive memory functionality including:
// - Entity: Unified data structure (vector + metadata + edges)
// - Vector: Vector operations and distance functions
// - Edges: Probabilistic edge management with PGM fields
// - Types: Core type aliases (EntityId, NodeId, ShardId, ClusterId)
// - Traits: Shared abstractions for memory substrate components

pub mod entity;
pub mod vector;
pub mod edges;
pub mod types;
pub mod error;
pub mod traits;
pub mod metadata;
pub mod transaction;
pub mod mvcc;
pub mod query;
pub mod config;

// Re-export commonly used types
pub use entity::{Entity, MemoryTier, AccessStatistics};
pub use vector::Vector;
pub use edges::Edge;
pub use types::{EntityId, NodeId, ShardId, ClusterId};
pub use error::{Result, MemorySubstrateError};
pub use config::PhenixConfig;
