//! # Graph Edge Management and Traversal
//!
//! This module implements graph edge data structures and operations for Phenix DB.
//! Edges represent relationships between entities and support weighted connections
//! with optional metadata for complex graph operations.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::core::{
    entity::{EntityId, JSONB},
    mvcc::MVCCVersion,
    error::{PhenixDBError, Result},
};

/// Unique identifier for edges in the database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub Uuid);

impl EdgeId {
    /// Generate a new random EdgeId
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create EdgeId from string representation
    pub fn from_string(s: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(s)
            .map_err(|e| PhenixDBError::InvalidEdgeId(format!("Invalid UUID format: {}", e)))?;
        Ok(Self(uuid))
    }

    /// Get string representation of EdgeId
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl Default for EdgeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Graph Edge for relationships between entities
///
/// The Edge represents a directed relationship between two entities in the graph.
/// Edges support labels, weights, and optional metadata for rich relationship
/// modeling. All edges are versioned for MVCC support and can be queried
/// efficiently for graph traversal operations.
///
/// # Design Principles
/// - **Directed Relationships**: Edges have explicit source and target entities
/// - **Weighted Connections**: Numeric weights for relationship strength
/// - **Rich Metadata**: Optional JSONB metadata for complex relationships
/// - **MVCC Support**: Full versioning for concurrent access and transactions
///
/// # Examples
/// ```rust
/// use phenix_db::core::edges::Edge;
/// use phenix_db::core::entity::EntityId;
/// use serde_json::json;
///
/// let source_id = EntityId::new();
/// let target_id = EntityId::new();
///
/// // Create a simple labeled edge
/// let edge = Edge::new(source_id, target_id, "follows".to_string(), 1.0);
///
/// // Create an edge with metadata
/// let mut edge_with_metadata = Edge::new(
///     source_id, 
///     target_id, 
///     "related_to".to_string(), 
///     0.8
/// );
/// edge_with_metadata.set_metadata(json!({
///     "relationship_type": "strong",
///     "created_by": "system"
/// }));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier for this edge
    pub id: EdgeId,
    
    /// Source entity ID
    pub source_id: EntityId,
    
    /// Target entity ID
    pub target_id: EntityId,
    
    /// Edge label describing the relationship type
    pub label: String,
    
    /// Numeric weight representing relationship strength (0.0 to 1.0)
    pub weight: f32,
    
    /// Optional metadata for rich relationship information
    pub metadata: Option<JSONB>,
    
    /// Edge creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Edge last update timestamp
    pub updated_at: DateTime<Utc>,
    
    /// MVCC version for concurrency control
    pub version: MVCCVersion,
}

impl Edge {
    /// Create a new Edge with the given parameters
    pub fn new(source_id: EntityId, target_id: EntityId, label: String, weight: f32) -> Self {
        let now = Utc::now();
        Self {
            id: EdgeId::new(),
            source_id,
            target_id,
            label,
            weight: weight.clamp(0.0, 1.0), // Ensure weight is in valid range
            metadata: None,
            created_at: now,
            updated_at: now,
            version: MVCCVersion::new(),
        }
    }

    /// Create a new Edge with metadata
    pub fn new_with_metadata(
        source_id: EntityId,
        target_id: EntityId,
        label: String,
        weight: f32,
        metadata: JSONB,
    ) -> Self {
        let mut edge = Self::new(source_id, target_id, label, weight);
        edge.metadata = Some(metadata);
        edge
    }

    /// Get the edge label
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the edge weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Set the edge weight (clamped to 0.0-1.0 range)
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }

    /// Check if edge has metadata
    pub fn has_metadata(&self) -> bool {
        self.metadata.is_some()
    }

    /// Set edge metadata
    pub fn set_metadata(&mut self, metadata: JSONB) {
        self.metadata = Some(metadata);
        self.updated_at = Utc::now();
    }

    /// Remove edge metadata
    pub fn remove_metadata(&mut self) {
        if self.metadata.is_some() {
            self.metadata = None;
            self.updated_at = Utc::now();
        }
    }

    /// Update edge label
    pub fn set_label(&mut self, label: String) {
        if self.label != label {
            self.label = label;
            self.updated_at = Utc::now();
        }
    }

    /// Check if this edge connects the given entities (in either direction)
    pub fn connects(&self, entity1: EntityId, entity2: EntityId) -> bool {
        (self.source_id == entity1 && self.target_id == entity2) ||
        (self.source_id == entity2 && self.target_id == entity1)
    }

    /// Check if this edge is directed from source to target
    pub fn is_directed_from_to(&self, source: EntityId, target: EntityId) -> bool {
        self.source_id == source && self.target_id == target
    }

    /// Get the opposite entity ID given one end of the edge
    pub fn get_opposite_entity(&self, entity_id: EntityId) -> Option<EntityId> {
        if self.source_id == entity_id {
            Some(self.target_id)
        } else if self.target_id == entity_id {
            Some(self.source_id)
        } else {
            None
        }
    }

    /// Reverse the direction of the edge
    pub fn reverse(&mut self) {
        std::mem::swap(&mut self.source_id, &mut self.target_id);
        self.updated_at = Utc::now();
    }

    /// Create a reversed copy of the edge
    pub fn reversed(&self) -> Self {
        let mut reversed = self.clone();
        reversed.id = EdgeId::new(); // New ID for the reversed edge
        reversed.reverse();
        reversed
    }

    /// Validate edge constraints
    pub fn validate(&self) -> Result<()> {
        // Check that source and target are different
        if self.source_id == self.target_id {
            return Err(PhenixDBError::ValidationError {
                message: "Edge source and target cannot be the same entity".to_string()
            });
        }

        // Check weight range
        if !(0.0..=1.0).contains(&self.weight) {
            return Err(PhenixDBError::ValidationError {
                message: format!("Edge weight {} must be between 0.0 and 1.0", self.weight)
            });
        }

        // Check label is not empty
        if self.label.trim().is_empty() {
            return Err(PhenixDBError::ValidationError {
                message: "Edge label cannot be empty".to_string()
            });
        }

        // Validate metadata size if present
        if let Some(ref metadata) = self.metadata {
            let serialized = serde_json::to_string(metadata)
                .map_err(|e| PhenixDBError::SerializationError { message: format!("Failed to serialize edge metadata: {}", e) })?;
            
            const MAX_EDGE_METADATA_SIZE: usize = 64 * 1024; // 64KB limit
            if serialized.len() > MAX_EDGE_METADATA_SIZE {
                return Err(PhenixDBError::ValidationError {
                    message: format!("Edge metadata size {} exceeds maximum allowed size {}", 
                           serialized.len(), MAX_EDGE_METADATA_SIZE)
                });
            }
        }

        Ok(())
    }

    /// Calculate approximate memory usage of this edge
    pub fn memory_usage(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        size += self.label.len();
        
        if let Some(ref metadata) = self.metadata {
            // Approximate JSON size
            size += serde_json::to_string(metadata).map_or(0, |s| s.len());
        }
        
        size
    }
}

/// Edge query and filtering utilities
impl Edge {
    /// Check if edge matches the given label
    pub fn matches_label(&self, label: &str) -> bool {
        self.label == label
    }

    /// Check if edge weight is above threshold
    pub fn weight_above(&self, threshold: f32) -> bool {
        self.weight > threshold
    }

    /// Check if edge weight is below threshold
    pub fn weight_below(&self, threshold: f32) -> bool {
        self.weight < threshold
    }

    /// Check if edge weight is within range
    pub fn weight_in_range(&self, min: f32, max: f32) -> bool {
        self.weight >= min && self.weight <= max
    }

    /// Check if edge has metadata field with specific value
    pub fn has_metadata_field(&self, field: &str, value: &serde_json::Value) -> bool {
        if let Some(ref metadata) = self.metadata {
            metadata.get(field) == Some(value)
        } else {
            false
        }
    }
}

/// Builder pattern for constructing Edge instances
#[derive(Debug, Default)]
pub struct EdgeBuilder {
    source_id: Option<EntityId>,
    target_id: Option<EntityId>,
    label: Option<String>,
    weight: Option<f32>,
    metadata: Option<JSONB>,
}

impl EdgeBuilder {
    /// Create a new EdgeBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source entity ID
    pub fn source(mut self, source_id: EntityId) -> Self {
        self.source_id = Some(source_id);
        self
    }

    /// Set the target entity ID
    pub fn target(mut self, target_id: EntityId) -> Self {
        self.target_id = Some(target_id);
        self
    }

    /// Set the edge label
    pub fn label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the edge weight
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = Some(weight);
        self
    }

    /// Set the edge metadata
    pub fn metadata(mut self, metadata: JSONB) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Build the Edge instance
    pub fn build(self) -> Result<Edge> {
        let source_id = self.source_id.ok_or_else(|| {
            PhenixDBError::ValidationError { message: "Edge source_id is required".to_string() }
        })?;

        let target_id = self.target_id.ok_or_else(|| {
            PhenixDBError::ValidationError { message: "Edge target_id is required".to_string() }
        })?;

        let label = self.label.ok_or_else(|| {
            PhenixDBError::ValidationError { message: "Edge label is required".to_string() }
        })?;

        let weight = self.weight.unwrap_or(1.0);

        let mut edge = if let Some(metadata) = self.metadata {
            Edge::new_with_metadata(source_id, target_id, label, weight, metadata)
        } else {
            Edge::new(source_id, target_id, label, weight)
        };

        edge.validate()?;
        Ok(edge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_edge_creation() {
        let source_id = EntityId::new();
        let target_id = EntityId::new();
        let edge = Edge::new(source_id, target_id, "follows".to_string(), 0.8);

        assert_eq!(edge.source_id, source_id);
        assert_eq!(edge.target_id, target_id);
        assert_eq!(edge.label, "follows");
        assert_eq!(edge.weight, 0.8);
        assert!(!edge.has_metadata());
    }

    #[test]
    fn test_edge_with_metadata() {
        let source_id = EntityId::new();
        let target_id = EntityId::new();
        let metadata = json!({"type": "strong", "confidence": 0.9});
        
        let edge = Edge::new_with_metadata(
            source_id, 
            target_id, 
            "related".to_string(), 
            0.9, 
            metadata.clone()
        );

        assert!(edge.has_metadata());
        assert_eq!(edge.metadata, Some(metadata));
    }

    #[test]
    fn test_edge_weight_clamping() {
        let source_id = EntityId::new();
        let target_id = EntityId::new();
        
        let edge1 = Edge::new(source_id, target_id, "test".to_string(), -0.5);
        assert_eq!(edge1.weight, 0.0);
        
        let edge2 = Edge::new(source_id, target_id, "test".to_string(), 1.5);
        assert_eq!(edge2.weight, 1.0);
    }

    #[test]
    fn test_edge_connections() {
        let entity1 = EntityId::new();
        let entity2 = EntityId::new();
        let entity3 = EntityId::new();
        
        let edge = Edge::new(entity1, entity2, "connects".to_string(), 1.0);
        
        assert!(edge.connects(entity1, entity2));
        assert!(edge.connects(entity2, entity1));
        assert!(!edge.connects(entity1, entity3));
        
        assert!(edge.is_directed_from_to(entity1, entity2));
        assert!(!edge.is_directed_from_to(entity2, entity1));
    }

    #[test]
    fn test_edge_reversal() {
        let source_id = EntityId::new();
        let target_id = EntityId::new();
        let mut edge = Edge::new(source_id, target_id, "points_to".to_string(), 0.7);
        
        edge.reverse();
        assert_eq!(edge.source_id, target_id);
        assert_eq!(edge.target_id, source_id);
    }

    #[test]
    fn test_edge_validation() {
        let entity_id = EntityId::new();
        
        // Self-loop should fail
        let self_edge = Edge::new(entity_id, entity_id, "self".to_string(), 1.0);
        assert!(self_edge.validate().is_err());
        
        // Valid edge should pass
        let valid_edge = Edge::new(EntityId::new(), EntityId::new(), "valid".to_string(), 0.5);
        assert!(valid_edge.validate().is_ok());
        
        // Empty label should fail
        let empty_label_edge = Edge::new(EntityId::new(), EntityId::new(), "".to_string(), 0.5);
        assert!(empty_label_edge.validate().is_err());
    }

    #[test]
    fn test_edge_builder() {
        let source_id = EntityId::new();
        let target_id = EntityId::new();
        
        let edge = EdgeBuilder::new()
            .source(source_id)
            .target(target_id)
            .label("built")
            .weight(0.6)
            .metadata(json!({"builder": true}))
            .build()
            .unwrap();

        assert_eq!(edge.source_id, source_id);
        assert_eq!(edge.target_id, target_id);
        assert_eq!(edge.label, "built");
        assert_eq!(edge.weight, 0.6);
        assert!(edge.has_metadata());
    }

    #[test]
    fn test_edge_filtering() {
        let edge = Edge::new_with_metadata(
            EntityId::new(),
            EntityId::new(),
            "test_label".to_string(),
            0.7,
            json!({"category": "important"})
        );

        assert!(edge.matches_label("test_label"));
        assert!(!edge.matches_label("other_label"));
        assert!(edge.weight_above(0.5));
        assert!(!edge.weight_above(0.8));
        assert!(edge.weight_in_range(0.6, 0.8));
        assert!(edge.has_metadata_field("category", &json!("important")));
    }
}