//! Shared trait definitions

use super::{Entity, EntityId, Result};
use async_trait::async_trait;

/// Trait for adaptive learning components
#[async_trait]
pub trait AdaptiveLearning: Send + Sync {
    /// Observe an access event
    async fn observe_access(&mut self, entity_id: EntityId, timestamp: super::Timestamp);
    
    /// Predict next access time
    async fn predict_next_access(&self, entity_id: EntityId) -> Option<super::Timestamp>;
    
    /// Adjust parameters based on feedback
    async fn adjust_parameters(&mut self, feedback: LearningFeedback) -> Result<()>;
    
    /// Get current prediction accuracy
    fn get_accuracy(&self) -> f32;
}

/// Learning feedback for parameter adjustment
#[derive(Debug, Clone)]
pub struct LearningFeedback {
    /// Prediction was correct
    pub correct: bool,
    
    /// Actual access time
    pub actual_time: super::Timestamp,
    
    /// Predicted access time
    pub predicted_time: super::Timestamp,
}

/// Trait for entropy-aware storage components
#[async_trait]
pub trait EntropyAwareStorage: Send + Sync {
    /// Compute entropy for a shard
    async fn compute_entropy(&self, shard_id: super::ShardId) -> Result<f64>;
    
    /// Check if reorganization is needed
    async fn should_reorganize(&self, shard_id: super::ShardId) -> Result<bool>;
    
    /// Optimize storage layout
    async fn optimize_layout(&mut self, shard_id: super::ShardId) -> Result<()>;
}

/// Trait for polynomial index operations
#[async_trait]
pub trait PolynomialIndex: Send + Sync {
    /// Insert an entity
    async fn insert(&mut self, entity: &Entity) -> Result<Vec<f64>>;
    
    /// Search for k nearest neighbors
    async fn search(&self, query: &super::Vector, k: usize) -> Result<Vec<EntityId>>;
    
    /// Update an entity
    async fn update(&mut self, entity_id: EntityId, entity: &Entity) -> Result<()>;
    
    /// Delete an entity
    async fn delete(&mut self, entity_id: EntityId) -> Result<()>;
}

/// Trait for probabilistic graph operations
#[async_trait]
pub trait ProbabilisticGraph: Send + Sync {
    /// Record an access
    async fn record_access(&self, entity_id: EntityId, timestamp: super::Timestamp);
    
    /// Update edge weights
    async fn update_edge_weights(&mut self) -> Result<()>;
    
    /// Prune low-probability edges
    async fn prune_low_probability_edges(&mut self) -> Result<usize>;
    
    /// Traverse the graph
    async fn traverse(&self, start: EntityId, depth: usize) -> Result<Vec<EntityId>>;
}
