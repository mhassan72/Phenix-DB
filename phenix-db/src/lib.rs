//! # Phenix-DB: Mathematical Memory Substrate
//!
//! Phenix-DB is a self-organizing, learning, and adaptive cognitive memory system
//! grounded in centuries of mathematical wisdom. It transforms static data storage
//! into living memory that learns, compresses, and self-reorganizes.
//!
//! ## Core Philosophy
//!
//! Five mathematical pillars:
//! - **Recursion** (Al-Samawal, von Neumann): Memory references itself to learn
//! - **Probability** (Kolmogorov, De Moivre): Adaptive retrieval, not absolute
//! - **Optimization** (Bellman, Kantorovich): Paths evolve as system learns
//! - **Geometry** (Khayyam, Tusi, Euler): Semantic meaning in curved space
//! - **Compression** (Ramanujan, Gauss): Dense storage without distortion
//!
//! ## Architecture
//!
//! The system is organized into layers:
//! - **Core**: Fundamental data structures and operations
//! - **Mathematical**: Pure mathematical operations and algorithms
//! - **Memory**: Memory substrate components (RPI, PGM, KCE, etc.)
//! - **Storage**: Hierarchical storage tiers (hot/warm/cold)
//! - **Distributed**: Distributed consciousness and coordination
//! - **Concurrency**: Lock-free concurrency and MVCC
//! - **Learning**: Adaptive learning and optimization
//! - **Security**: Encryption and integrity verification
//! - **API**: External interfaces (gRPC, REST)
//! - **Observability**: Metrics, tracing, and logging

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

// Core modules
pub mod core;

// Mathematical foundation modules
pub mod mathematical;

// Memory substrate components
pub mod memory;

// Storage layer
pub mod storage;

// Indexing and search
pub mod index;

// Distributed consciousness
pub mod distributed;

// Lock-free concurrency
pub mod concurrency;

// Adaptive learning
pub mod learning;

// Security and encryption
pub mod security;

// API layer
pub mod api;

// Observability
pub mod observability;

// Re-export commonly used types
pub use core::{Entity, EntityId, Vector, Edge, MemoryTier};
pub use memory::{
    RecursivePolynomialIndex,
    ProbabilisticGraphMemory,
    BellmanOptimizer,
    KolmogorovCompressionEngine,
};

/// Result type alias for Phenix-DB operations
pub type Result<T> = std::result::Result<T, core::error::MemorySubstrateError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: &str = concat!(
    "Phenix-DB v",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("CARGO_PKG_REPOSITORY"),
    ")"
);
