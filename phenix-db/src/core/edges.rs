//! Edge data structures for graph relationships

use super::{EntityId, Timestamp};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Basic edge between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source entity ID
    pub source_id: EntityId,
    
    /// Target entity ID
    pub target_id: EntityId,
    
    /// Edge label/type
    pub label: String,
    
    /// Edge weight
    pub weight: f32,
    
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Probabilistic edge for PGM (Probabilistic Graph Memory)
#[derive(Debug)]
pub struct ProbabilisticEdge {
    /// Source entity ID
    pub source_id: EntityId,
    
    /// Target entity ID
    pub target_id: EntityId,
    
    /// Edge label/type
    pub label: String,
    
    /// Raw weight
    pub weight: f32,
    
    /// Normalized probability (0.0-1.0)
    pub probability: f32,
    
    /// Lock-free access counter
    pub access_count: AtomicU64,
    
    /// Last accessed timestamp (atomic)
    pub last_accessed: AtomicU64,
    
    /// Co-access time window in milliseconds
    pub co_access_window_ms: u64,
}

impl ProbabilisticEdge {
    /// Create a new probabilistic edge
    pub fn new(
        source_id: EntityId,
        target_id: EntityId,
        label: String,
        initial_weight: f32,
    ) -> Self {
        Self {
            source_id,
            target_id,
            label,
            weight: initial_weight,
            probability: initial_weight,
            access_count: AtomicU64::new(0),
            last_accessed: AtomicU64::new(Timestamp::now().as_micros()),
            co_access_window_ms: 100, // Default 100ms window
        }
    }

    /// Record an access to this edge (lock-free)
    pub fn record_access(&self) {
        self.access_count.fetch_add(1, Ordering::SeqCst);
        self.last_accessed
            .store(Timestamp::now().as_micros(), Ordering::SeqCst);
    }

    /// Get the current access count
    pub fn get_access_count(&self) -> u64 {
        self.access_count.load(Ordering::SeqCst)
    }

    /// Get the last accessed timestamp
    pub fn get_last_accessed(&self) -> Timestamp {
        Timestamp::from_micros(self.last_accessed.load(Ordering::SeqCst))
    }

    /// Update probability (should be called during weight normalization)
    pub fn update_probability(&mut self, new_probability: f32) {
        self.probability = new_probability.clamp(0.0, 1.0);
    }

    /// Update weight based on co-access (learning algorithm)
    pub fn update_weight(&mut self, learning_rate: f32, co_accessed: bool) {
        if co_accessed {
            self.weight = (self.weight + learning_rate).min(1.0);
        } else {
            // Decay weight slightly if not co-accessed
            self.weight = (self.weight - learning_rate * 0.1).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probabilistic_edge_creation() {
        let source = EntityId::new();
        let target = EntityId::new();
        let edge = ProbabilisticEdge::new(source, target, "relates_to".to_string(), 0.5);
        
        assert_eq!(edge.weight, 0.5);
        assert_eq!(edge.probability, 0.5);
        assert_eq!(edge.get_access_count(), 0);
    }

    #[test]
    fn test_edge_access_recording() {
        let source = EntityId::new();
        let target = EntityId::new();
        let edge = ProbabilisticEdge::new(source, target, "relates_to".to_string(), 0.5);
        
        edge.record_access();
        edge.record_access();
        
        assert_eq!(edge.get_access_count(), 2);
    }

    #[test]
    fn test_weight_update() {
        let source = EntityId::new();
        let target = EntityId::new();
        let mut edge = ProbabilisticEdge::new(source, target, "relates_to".to_string(), 0.5);
        
        edge.update_weight(0.1, true);
        assert!((edge.weight - 0.6).abs() < 0.0001);
        
        edge.update_weight(0.1, false);
        assert!((edge.weight - 0.59).abs() < 0.0001);
    }
}
