//! # Unified Entity Data Structure
//!
//! This module implements the core Entity data structure that serves as the first-class
//! citizen in Phenix DB. An Entity can contain an optional vector, metadata (JSONB),
//! and edges for unified storage and querying across vector similarity, document
//! filtering, and graph traversal operations.

use serde::{Deserialize, Serialize};
// use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::core::{
    vector::Vector,
    edges::Edge,
    mvcc::MVCCVersion,
    error::{PhenixDBError, Result},
};

/// Unique identifier for entities in the database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub Uuid);

impl EntityId {
    /// Generate a new random EntityId
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create EntityId from string representation
    pub fn from_string(s: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(s)
            .map_err(|e| PhenixDBError::InvalidEntityId(format!("Invalid UUID format: {}", e)))?;
        Ok(Self(uuid))
    }

    /// Get string representation of EntityId
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl Default for EntityId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// JSONB metadata type for structured document data
pub type JSONB = serde_json::Value;

/// Unified Entity - first-class citizen containing vector, metadata, and edges
///
/// The Entity represents the core data structure in Phenix DB, designed to unify
/// vector embeddings, structured metadata, and graph relationships in a single
/// transactional unit. This eliminates the complexity of managing separate
/// databases and ensures ACID guarantees across all data types.
///
/// # Design Principles
/// - **Optional Components**: Each component (vector, metadata, edges) is optional
/// - **MVCC Support**: Full versioning for concurrent access and transactions
/// - **Tenant Isolation**: Built-in support for multi-tenant deployments
/// - **Performance Optimized**: Efficient serialization and memory layout
///
/// # Examples
/// ```rust
/// use phenix_db::core::entity::{Entity, EntityBuilder};
/// use serde_json::json;
///
/// // Create entity with all components
/// let entity = Entity::builder()
///     .with_vector(vec![0.1, 0.2, 0.3, 0.4])
///     .with_metadata(json!({"title": "Document", "category": "AI"}))
///     .with_edge("related_to", "other_entity_id", 0.8)
///     .build();
///
/// // Create vector-only entity
/// let vector_entity = Entity::builder()
///     .with_vector(vec![0.5, 0.6, 0.7, 0.8])
///     .build();
///
/// // Create document-only entity
/// let doc_entity = Entity::builder()
///     .with_metadata(json!({"content": "Text content", "tags": ["important"]}))
///     .build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier for this entity
    pub id: EntityId,
    
    /// Optional dense vector embedding (float32 array)
    pub vector: Option<Vector>,
    
    /// Optional structured metadata as JSONB
    pub metadata: Option<JSONB>,
    
    /// Optional graph edges for relationships
    pub edges: Option<Vec<Edge>>,
    
    /// Entity creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Entity last update timestamp
    pub updated_at: DateTime<Utc>,
    
    /// MVCC version for concurrency control
    pub version: MVCCVersion,
    
    /// Optional tenant identifier for multi-tenant isolation
    pub tenant_id: Option<String>,
}

impl Entity {
    /// Create a new Entity with the given ID
    pub fn new(id: EntityId) -> Self {
        let now = Utc::now();
        Self {
            id,
            vector: None,
            metadata: None,
            edges: None,
            created_at: now,
            updated_at: now,
            version: MVCCVersion::new(),
            tenant_id: None,
        }
    }

    /// Create a new Entity with a random ID
    pub fn new_random() -> Self {
        Self::new(EntityId::new())
    }

    /// Create a builder for constructing entities
    pub fn builder() -> EntityBuilder {
        EntityBuilder::new()
    }

    /// Check if entity has a vector component
    pub fn has_vector(&self) -> bool {
        self.vector.is_some()
    }

    /// Check if entity has metadata component
    pub fn has_metadata(&self) -> bool {
        self.metadata.is_some()
    }

    /// Check if entity has edges component
    pub fn has_edges(&self) -> bool {
        self.edges.as_ref().map_or(false, |edges| !edges.is_empty())
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.as_ref().map_or(0, |edges| edges.len())
    }

    /// Add an edge to this entity
    pub fn add_edge(&mut self, edge: Edge) {
        match &mut self.edges {
            Some(edges) => edges.push(edge),
            None => self.edges = Some(vec![edge]),
        }
        self.updated_at = Utc::now();
    }

    /// Remove edges with the given target entity ID
    pub fn remove_edges_to(&mut self, target_id: EntityId) -> usize {
        if let Some(edges) = &mut self.edges {
            let initial_len = edges.len();
            edges.retain(|edge| edge.target_id != target_id);
            let removed_count = initial_len - edges.len();
            
            if removed_count > 0 {
                self.updated_at = Utc::now();
                
                // Remove edges vector if empty
                if edges.is_empty() {
                    self.edges = None;
                }
            }
            
            removed_count
        } else {
            0
        }
    }

    /// Update the entity's metadata
    pub fn update_metadata(&mut self, metadata: JSONB) {
        self.metadata = Some(metadata);
        self.updated_at = Utc::now();
    }

    /// Update the entity's vector
    pub fn update_vector(&mut self, vector: Vector) {
        self.vector = Some(vector);
        self.updated_at = Utc::now();
    }

    /// Set the tenant ID for multi-tenant isolation
    pub fn set_tenant_id(&mut self, tenant_id: String) {
        self.tenant_id = Some(tenant_id);
    }

    /// Validate entity constraints
    pub fn validate(&self) -> Result<()> {
        // Validate vector dimensions if present
        if let Some(ref vector) = self.vector {
            vector.validate()?;
        }

        // Validate edges if present
        if let Some(ref edges) = self.edges {
            for edge in edges {
                edge.validate()?;
            }
        }

        // Validate metadata size (prevent extremely large documents)
        if let Some(ref metadata) = self.metadata {
            let serialized = serde_json::to_string(metadata)
                .map_err(|e| PhenixDBError::SerializationError { message: format!("Failed to serialize metadata: {}", e) })?;
            
            const MAX_METADATA_SIZE: usize = 1024 * 1024; // 1MB limit
            if serialized.len() > MAX_METADATA_SIZE {
                return Err(PhenixDBError::ValidationError {
                    message: format!("Metadata size {} exceeds maximum allowed size {}", 
                           serialized.len(), MAX_METADATA_SIZE)
                });
            }
        }

        Ok(())
    }

    /// Calculate approximate memory usage of this entity
    pub fn memory_usage(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        
        if let Some(ref vector) = self.vector {
            size += vector.memory_usage();
        }
        
        if let Some(ref metadata) = self.metadata {
            // Approximate JSON size
            size += serde_json::to_string(metadata).map_or(0, |s| s.len());
        }
        
        if let Some(ref edges) = self.edges {
            size += edges.iter().map(|edge| edge.memory_usage()).sum::<usize>();
        }
        
        if let Some(ref tenant_id) = self.tenant_id {
            size += tenant_id.len();
        }
        
        size
    }
}

/// Builder pattern for constructing Entity instances
///
/// The EntityBuilder provides a fluent interface for creating entities with
/// optional components. This ensures proper initialization and validation
/// while maintaining ergonomic usage patterns.
///
/// # Examples
/// ```rust
/// use phenix_db::core::entity::EntityBuilder;
/// use serde_json::json;
///
/// let entity = EntityBuilder::new()
///     .with_id("custom-id")
///     .with_vector(vec![0.1, 0.2, 0.3])
///     .with_metadata(json!({"type": "document"}))
///     .with_tenant_id("tenant-123")
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct EntityBuilder {
    id: Option<EntityId>,
    vector: Option<Vector>,
    metadata: Option<JSONB>,
    edges: Option<Vec<Edge>>,
    tenant_id: Option<String>,
}

impl EntityBuilder {
    /// Create a new EntityBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the entity ID (generates random ID if not set)
    pub fn with_id<T: Into<EntityId>>(mut self, id: T) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the entity ID from string
    pub fn with_id_string(mut self, id: &str) -> Result<Self> {
        self.id = Some(EntityId::from_string(id)?);
        Ok(self)
    }

    /// Add a vector component
    pub fn with_vector(mut self, dimensions: Vec<f32>) -> Self {
        self.vector = Some(Vector::new(dimensions));
        self
    }

    /// Add a metadata component
    pub fn with_metadata(mut self, metadata: JSONB) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add an edge
    pub fn with_edge<T: Into<EntityId>>(mut self, label: &str, target_id: T, weight: f32) -> Self {
        let edge = Edge::new(
            self.id.unwrap_or_default(), // Will be set properly in build()
            target_id.into(),
            label.to_string(),
            weight,
        );
        
        match &mut self.edges {
            Some(edges) => edges.push(edge),
            None => self.edges = Some(vec![edge]),
        }
        self
    }

    /// Add multiple edges
    pub fn with_edges(mut self, edges: Vec<Edge>) -> Self {
        self.edges = Some(edges);
        self
    }

    /// Set tenant ID for multi-tenant isolation
    pub fn with_tenant_id(mut self, tenant_id: &str) -> Self {
        self.tenant_id = Some(tenant_id.to_string());
        self
    }

    /// Build the Entity instance
    pub fn build(self) -> Entity {
        let id = self.id.unwrap_or_else(EntityId::new);
        let now = Utc::now();

        // Fix edge source IDs if they were set before the entity ID was known
        let edges = self.edges.map(|mut edges| {
            for edge in &mut edges {
                if edge.source_id.0.is_nil() {
                    edge.source_id = id;
                }
            }
            edges
        });

        Entity {
            id,
            vector: self.vector,
            metadata: self.metadata,
            edges,
            created_at: now,
            updated_at: now,
            version: MVCCVersion::new(),
            tenant_id: self.tenant_id,
        }
    }
}

impl From<&str> for EntityId {
    fn from(s: &str) -> Self {
        EntityId::from_string(s).unwrap_or_else(|_| EntityId::new())
    }
}

impl From<String> for EntityId {
    fn from(s: String) -> Self {
        EntityId::from_string(&s).unwrap_or_else(|_| EntityId::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new_random();
        assert!(entity.vector.is_none());
        assert!(entity.metadata.is_none());
        assert!(entity.edges.is_none());
        assert!(!entity.has_vector());
        assert!(!entity.has_metadata());
        assert!(!entity.has_edges());
    }

    #[test]
    fn test_entity_builder() {
        let entity = Entity::builder()
            .with_vector(vec![0.1, 0.2, 0.3])
            .with_metadata(json!({"title": "Test"}))
            .with_edge("related", EntityId::new(), 0.8)
            .build();

        assert!(entity.has_vector());
        assert!(entity.has_metadata());
        assert!(entity.has_edges());
        assert_eq!(entity.edge_count(), 1);
    }

    #[test]
    fn test_entity_validation() {
        let entity = Entity::builder()
            .with_vector(vec![0.1; 256]) // Use 256 dimensions to meet minimum requirement
            .build();

        assert!(entity.validate().is_ok());
    }

    #[test]
    fn test_entity_id_conversion() {
        let id_str = "550e8400-e29b-41d4-a716-446655440000";
        let entity_id = EntityId::from_string(id_str).unwrap();
        assert_eq!(entity_id.to_string(), id_str);
    }

    #[test]
    fn test_edge_management() {
        let mut entity = Entity::new_random();
        let target_id = EntityId::new();
        
        let edge = Edge::new(entity.id, target_id, "test".to_string(), 1.0);
        entity.add_edge(edge);
        
        assert_eq!(entity.edge_count(), 1);
        
        let removed = entity.remove_edges_to(target_id);
        assert_eq!(removed, 1);
        assert_eq!(entity.edge_count(), 0);
    }
}