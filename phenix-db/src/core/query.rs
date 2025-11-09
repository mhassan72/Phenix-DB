//! Query data structures

use super::{EntityId, MemoryTier, Vector};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Cognitive query structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveQuery {
    /// Query type
    pub query_type: QueryType,
    
    /// Optional vector query
    pub vector_query: Option<VectorQuery>,
    
    /// Optional metadata filter
    pub metadata_filter: Option<super::metadata::MetadataFilter>,
    
    /// Optional graph traversal
    pub graph_traversal: Option<GraphTraversal>,
    
    /// Query options
    pub options: QueryOptions,
}

/// Query type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    /// Vector similarity search
    VectorSimilarity,
    /// Metadata search
    MetadataSearch,
    /// Graph traversal
    GraphTraversal,
    /// Hybrid query combining multiple types
    Hybrid,
}

/// Vector similarity query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQuery {
    /// Query vector
    pub vector: Vector,
    
    /// Number of results (k)
    pub k: usize,
    
    /// Distance metric
    pub distance_metric: DistanceMetric,
    
    /// Tier preference
    pub tier_preference: Option<MemoryTier>,
}

/// Distance metric enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Curved space distance (non-Euclidean)
    Curved,
}

/// Graph traversal query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraversal {
    /// Starting entities
    pub start_entities: Vec<EntityId>,
    
    /// Maximum traversal depth
    pub max_depth: usize,
    
    /// Optional edge filter
    pub edge_filter: Option<EdgeFilter>,
    
    /// Use probabilistic weights
    pub use_probabilistic_weights: bool,
}

/// Edge filter for graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeFilter {
    /// Edge labels to include
    pub labels: Option<Vec<String>>,
    
    /// Minimum weight threshold
    pub min_weight: Option<f32>,
    
    /// Minimum probability threshold
    pub min_probability: Option<f32>,
}

/// Query execution options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptions {
    /// Query timeout
    pub timeout: Duration,
    
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    
    /// Use learning cache for predictions
    pub use_learning_cache: bool,
    
    /// Return query execution plan
    pub explain: bool,
}

/// Consistency level for distributed queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Strong consistency
    Strong,
    /// Eventual consistency
    Eventual,
    /// Local consistency
    Local,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            consistency_level: ConsistencyLevel::Eventual,
            use_learning_cache: true,
            explain: false,
        }
    }
}
