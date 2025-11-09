//! Entity data structure - the first-class citizen of Phenix-DB

use super::{EntityId, MemoryTier, Timestamp, Vector, Edge};
use serde::{Deserialize, Serialize};

/// Unified entity containing vector + metadata + edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier
    pub id: EntityId,
    
    /// Optional vector embedding
    pub vector: Option<Vector>,
    
    /// Optional metadata (JSONB)
    pub metadata: Option<serde_json::Value>,
    
    /// Optional edges to other entities
    pub edges: Option<Vec<Edge>>,
    
    /// Creation timestamp
    pub created_at: Timestamp,
    
    /// Last update timestamp
    pub updated_at: Timestamp,
    
    /// MVCC version
    pub version: u64,
    
    /// Current memory tier
    pub tier: MemoryTier,
    
    /// Polynomial embedding coefficients (for RPI)
    pub polynomial_embedding: Option<Vec<f64>>,
    
    /// Access statistics for learning
    pub access_statistics: AccessStatistics,
    
    /// Compression metadata
    pub compression_metadata: Option<CompressionMetadata>,
}

/// Access statistics for adaptive learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessStatistics {
    /// Total number of accesses
    pub total_accesses: u64,
    
    /// Last access timestamp
    pub last_access: Timestamp,
    
    /// Access frequency (accesses per hour)
    pub access_frequency: f32,
    
    /// Entities frequently co-accessed with this one
    pub co_access_entities: Vec<EntityId>,
}

impl Default for AccessStatistics {
    fn default() -> Self {
        Self {
            total_accesses: 0,
            last_access: Timestamp::now(),
            access_frequency: 0.0,
            co_access_entities: Vec::new(),
        }
    }
}

/// Compression metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    /// Compression method used
    pub method: CompressionMethod,
    
    /// Original size in bytes
    pub original_size: usize,
    
    /// Compressed size in bytes
    pub compressed_size: usize,
    
    /// Compression ratio (compressed/original)
    pub compression_ratio: f32,
    
    /// Dictionary references (if using pattern dictionary)
    pub dictionary_refs: Vec<u64>,
}

/// Compression methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// Ramanujan series encoding
    RamanujanSeries { degree: usize },
    /// Pattern dictionary compression
    PatternDictionary,
    /// Hybrid approach
    Hybrid,
}

impl Entity {
    /// Create a new entity with the given ID
    pub fn new(id: EntityId) -> Self {
        let now = Timestamp::now();
        Self {
            id,
            vector: None,
            metadata: None,
            edges: None,
            created_at: now,
            updated_at: now,
            version: 0,
            tier: MemoryTier::Hot,
            polynomial_embedding: None,
            access_statistics: AccessStatistics::default(),
            compression_metadata: None,
        }
    }

    /// Record an access to this entity
    pub fn record_access(&mut self) {
        self.access_statistics.total_accesses += 1;
        self.access_statistics.last_access = Timestamp::now();
        // TODO: Update access frequency calculation
    }
}
