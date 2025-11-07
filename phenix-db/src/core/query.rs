//! # Unified Query Language and Planning
//!
//! This module implements the unified query language for Phenix DB that combines
//! vector similarity search, metadata filtering, and graph traversal in single operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::core::{
    entity::{Entity, EntityId},
    vector::Vector,
    metadata::{JSONB, MetadataQuery},
    // edges::EdgeId,
    error::{PhenixDBError, Result},
};

/// Unified query combining vector similarity, metadata filtering, and graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedQuery {
    /// Vector similarity search component
    pub vector_similarity: Option<VectorQuery>,
    
    /// Metadata filtering component
    pub metadata_filter: Option<MetadataQuery>,
    
    /// Graph traversal component
    pub graph_constraint: Option<GraphQuery>,
    
    /// Maximum number of results to return
    pub limit: Option<usize>,
    
    /// Minimum similarity threshold for vector results
    pub similarity_threshold: Option<f32>,
    
    /// Query timeout in milliseconds
    pub timeout_ms: Option<u64>,
    
    /// Whether to include entity vectors in results
    pub include_vectors: bool,
    
    /// Whether to include entity metadata in results
    pub include_metadata: bool,
    
    /// Whether to include entity edges in results
    pub include_edges: bool,
}

impl Default for UnifiedQuery {
    fn default() -> Self {
        Self {
            vector_similarity: None,
            metadata_filter: None,
            graph_constraint: None,
            limit: Some(10),
            similarity_threshold: None,
            timeout_ms: Some(1000),
            include_vectors: true,
            include_metadata: true,
            include_edges: true,
        }
    }
}

/// Vector similarity query component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQuery {
    /// Query vector for similarity search
    pub query_vector: Vector,
    
    /// Number of nearest neighbors to find
    pub k: usize,
    
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    
    /// Optional vector field filter (for entities with multiple vectors)
    pub vector_field: Option<String>,
}

/// Distance metrics for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (default for normalized vectors)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Dot product similarity
    DotProduct,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

/// Graph traversal query component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQuery {
    /// Starting entity IDs for traversal
    pub start_entities: Vec<EntityId>,
    
    /// Edge labels to follow (empty means all labels)
    pub edge_labels: Vec<String>,
    
    /// Maximum traversal depth
    pub max_depth: usize,
    
    /// Traversal direction
    pub direction: TraversalDirection,
    
    /// Minimum edge weight threshold
    pub min_edge_weight: Option<f32>,
    
    /// Maximum number of entities to visit
    pub max_entities: Option<usize>,
}

/// Graph traversal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraversalDirection {
    /// Follow outgoing edges only
    Outgoing,
    /// Follow incoming edges only
    Incoming,
    /// Follow edges in both directions
    Both,
}

impl Default for TraversalDirection {
    fn default() -> Self {
        Self::Outgoing
    }
}

/// Query result containing scored entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Matching entities with scores
    pub entities: Vec<ScoredEntity>,
    
    /// Total number of entities examined
    pub total_examined: usize,
    
    /// Query execution time in milliseconds
    pub execution_time_ms: u64,
    
    /// Whether the query timed out
    pub timed_out: bool,
    
    /// Query statistics
    pub statistics: QueryStatistics,
}

/// Entity with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredEntity {
    /// The entity
    pub entity: Entity,
    
    /// Overall relevance score (0.0 to 1.0)
    pub score: f32,
    
    /// Component scores
    pub component_scores: ComponentScores,
    
    /// Explanation of how the score was calculated
    pub explanation: Option<String>,
}

/// Component scores for different query parts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentScores {
    /// Vector similarity score (if vector query was used)
    pub vector_score: Option<f32>,
    
    /// Metadata match score (if metadata filter was used)
    pub metadata_score: Option<f32>,
    
    /// Graph relevance score (if graph query was used)
    pub graph_score: Option<f32>,
}

/// Query execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatistics {
    /// Number of entities matching vector similarity
    pub vector_matches: usize,
    
    /// Number of entities matching metadata filter
    pub metadata_matches: usize,
    
    /// Number of entities found through graph traversal
    pub graph_matches: usize,
    
    /// Number of shards queried
    pub shards_queried: usize,
    
    /// Index operations performed
    pub index_operations: HashMap<String, usize>,
}

/// Builder for constructing unified queries
#[derive(Debug, Default)]
pub struct UnifiedQueryBuilder {
    vector_similarity: Option<VectorQuery>,
    metadata_filter: Option<MetadataQuery>,
    graph_constraint: Option<GraphQuery>,
    limit: Option<usize>,
    similarity_threshold: Option<f32>,
    timeout_ms: Option<u64>,
    include_vectors: bool,
    include_metadata: bool,
    include_edges: bool,
}

impl UnifiedQueryBuilder {
    /// Create new query builder
    pub fn new() -> Self {
        Self {
            include_vectors: true,
            include_metadata: true,
            include_edges: true,
            ..Default::default()
        }
    }

    /// Add vector similarity search
    pub fn vector_similarity(mut self, query_vector: Vector, k: usize) -> Self {
        self.vector_similarity = Some(VectorQuery {
            query_vector,
            k,
            distance_metric: DistanceMetric::default(),
            vector_field: None,
        });
        self
    }

    /// Add vector similarity search with custom distance metric
    pub fn vector_similarity_with_metric(
        mut self,
        query_vector: Vector,
        k: usize,
        distance_metric: DistanceMetric,
    ) -> Self {
        self.vector_similarity = Some(VectorQuery {
            query_vector,
            k,
            distance_metric,
            vector_field: None,
        });
        self
    }

    /// Add metadata filter
    pub fn metadata_filter(mut self, filter: MetadataQuery) -> Self {
        self.metadata_filter = Some(filter);
        self
    }

    /// Add simple metadata equals filter
    pub fn metadata_equals(mut self, field: &str, value: JSONB) -> Self {
        self.metadata_filter = Some(MetadataQuery::Equals {
            field: field.to_string(),
            value,
        });
        self
    }

    /// Add graph traversal constraint
    pub fn graph_constraint(mut self, constraint: GraphQuery) -> Self {
        self.graph_constraint = Some(constraint);
        self
    }

    /// Add simple graph traversal from entity
    pub fn graph_from_entity(mut self, entity_id: EntityId, max_depth: usize) -> Self {
        self.graph_constraint = Some(GraphQuery {
            start_entities: vec![entity_id],
            edge_labels: Vec::new(),
            max_depth,
            direction: TraversalDirection::default(),
            min_edge_weight: None,
            max_entities: None,
        });
        self
    }

    /// Set result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set similarity threshold
    pub fn similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = Some(threshold.clamp(0.0, 1.0));
        self
    }

    /// Set query timeout
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Set whether to include vectors in results
    pub fn include_vectors(mut self, include: bool) -> Self {
        self.include_vectors = include;
        self
    }

    /// Set whether to include metadata in results
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Set whether to include edges in results
    pub fn include_edges(mut self, include: bool) -> Self {
        self.include_edges = include;
        self
    }

    /// Build the unified query
    pub fn build(self) -> UnifiedQuery {
        UnifiedQuery {
            vector_similarity: self.vector_similarity,
            metadata_filter: self.metadata_filter,
            graph_constraint: self.graph_constraint,
            limit: self.limit,
            similarity_threshold: self.similarity_threshold,
            timeout_ms: self.timeout_ms,
            include_vectors: self.include_vectors,
            include_metadata: self.include_metadata,
            include_edges: self.include_edges,
        }
    }
}

impl UnifiedQuery {
    /// Create a new query builder
    pub fn builder() -> UnifiedQueryBuilder {
        UnifiedQueryBuilder::new()
    }

    /// Validate the query
    pub fn validate(&self) -> Result<()> {
        // At least one query component must be specified
        if self.vector_similarity.is_none() 
            && self.metadata_filter.is_none() 
            && self.graph_constraint.is_none() {
            return Err(PhenixDBError::QueryError {
                message: "Query must specify at least one component (vector, metadata, or graph)".to_string(),
            });
        }

        // Validate vector query if present
        if let Some(ref vector_query) = self.vector_similarity {
            if vector_query.k == 0 {
                return Err(PhenixDBError::QueryError {
                    message: "Vector query k must be greater than 0".to_string(),
                });
            }
            
            vector_query.query_vector.validate()?;
        }

        // Validate graph query if present
        if let Some(ref graph_query) = self.graph_constraint {
            if graph_query.start_entities.is_empty() {
                return Err(PhenixDBError::QueryError {
                    message: "Graph query must specify at least one start entity".to_string(),
                });
            }
            
            if graph_query.max_depth == 0 {
                return Err(PhenixDBError::QueryError {
                    message: "Graph query max_depth must be greater than 0".to_string(),
                });
            }
        }

        // Validate limit
        if let Some(limit) = self.limit {
            if limit == 0 {
                return Err(PhenixDBError::QueryError {
                    message: "Query limit must be greater than 0".to_string(),
                });
            }
        }

        // Validate similarity threshold
        if let Some(threshold) = self.similarity_threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(PhenixDBError::QueryError {
                    message: "Similarity threshold must be between 0.0 and 1.0".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Check if query has vector component
    pub fn has_vector_component(&self) -> bool {
        self.vector_similarity.is_some()
    }

    /// Check if query has metadata component
    pub fn has_metadata_component(&self) -> bool {
        self.metadata_filter.is_some()
    }

    /// Check if query has graph component
    pub fn has_graph_component(&self) -> bool {
        self.graph_constraint.is_some()
    }

    /// Check if query is hybrid (multiple components)
    pub fn is_hybrid(&self) -> bool {
        let component_count = [
            self.has_vector_component(),
            self.has_metadata_component(),
            self.has_graph_component(),
        ].iter().filter(|&&x| x).count();
        
        component_count > 1
    }

    /// Get estimated complexity score
    pub fn complexity_score(&self) -> f32 {
        let mut score = 0.0;

        // Vector similarity complexity
        if let Some(ref vector_query) = self.vector_similarity {
            score += vector_query.k as f32 * 0.1;
        }

        // Metadata filter complexity (simplified)
        if self.metadata_filter.is_some() {
            score += 1.0;
        }

        // Graph traversal complexity
        if let Some(ref graph_query) = self.graph_constraint {
            score += (graph_query.max_depth as f32).powi(2) * graph_query.start_entities.len() as f32 * 0.1;
        }

        // Hybrid query penalty
        if self.is_hybrid() {
            score *= 1.5;
        }

        score
    }
}

impl ScoredEntity {
    /// Create new scored entity
    pub fn new(entity: Entity, score: f32) -> Self {
        Self {
            entity,
            score: score.clamp(0.0, 1.0),
            component_scores: ComponentScores {
                vector_score: None,
                metadata_score: None,
                graph_score: None,
            },
            explanation: None,
        }
    }

    /// Set vector similarity score
    pub fn with_vector_score(mut self, score: f32) -> Self {
        self.component_scores.vector_score = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Set metadata match score
    pub fn with_metadata_score(mut self, score: f32) -> Self {
        self.component_scores.metadata_score = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Set graph relevance score
    pub fn with_graph_score(mut self, score: f32) -> Self {
        self.component_scores.graph_score = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Set score explanation
    pub fn with_explanation<S: Into<String>>(mut self, explanation: S) -> Self {
        self.explanation = Some(explanation.into());
        self
    }

    /// Calculate combined score from components
    pub fn calculate_combined_score(&mut self, weights: &ScoreWeights) {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        if let Some(vector_score) = self.component_scores.vector_score {
            total_score += vector_score * weights.vector_weight;
            total_weight += weights.vector_weight;
        }

        if let Some(metadata_score) = self.component_scores.metadata_score {
            total_score += metadata_score * weights.metadata_weight;
            total_weight += weights.metadata_weight;
        }

        if let Some(graph_score) = self.component_scores.graph_score {
            total_score += graph_score * weights.graph_weight;
            total_weight += weights.graph_weight;
        }

        if total_weight > 0.0 {
            self.score = (total_score / total_weight).clamp(0.0, 1.0);
        }
    }
}

/// Weights for combining component scores
#[derive(Debug, Clone)]
pub struct ScoreWeights {
    pub vector_weight: f32,
    pub metadata_weight: f32,
    pub graph_weight: f32,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            vector_weight: 0.5,
            metadata_weight: 0.3,
            graph_weight: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::Vector;
    use serde_json::json;

    #[test]
    fn test_unified_query_builder() {
        let query_vector = Vector::new(vec![0.1, 0.2, 0.3]);
        
        let query = UnifiedQuery::builder()
            .vector_similarity(query_vector, 10)
            .metadata_equals("category", json!("document"))
            .limit(20)
            .similarity_threshold(0.8)
            .build();

        assert!(query.has_vector_component());
        assert!(query.has_metadata_component());
        assert!(!query.has_graph_component());
        assert!(query.is_hybrid());
        assert_eq!(query.limit, Some(20));
        assert_eq!(query.similarity_threshold, Some(0.8));
    }

    #[test]
    fn test_query_validation() {
        // Valid query
        let valid_query = UnifiedQuery::builder()
            .vector_similarity(Vector::new(vec![0.1; 256]), 5)
            .build();
        assert!(valid_query.validate().is_ok());

        // Invalid query - no components
        let invalid_query = UnifiedQuery::default();
        assert!(invalid_query.validate().is_err());

        // Invalid query - zero k
        let invalid_k_query = UnifiedQuery::builder()
            .vector_similarity(Vector::new(vec![0.1; 256]), 0)
            .build();
        assert!(invalid_k_query.validate().is_err());
    }

    #[test]
    fn test_scored_entity() {
        let entity = crate::core::entity::Entity::new_random();
        let mut scored_entity = ScoredEntity::new(entity, 0.0)
            .with_vector_score(0.9)
            .with_metadata_score(0.7)
            .with_graph_score(0.8);

        let weights = ScoreWeights::default();
        scored_entity.calculate_combined_score(&weights);

        // Should be weighted average: 0.9*0.5 + 0.7*0.3 + 0.8*0.2 = 0.82
        assert!((scored_entity.score - 0.82).abs() < 0.01);
    }

    #[test]
    fn test_query_complexity() {
        let simple_query = UnifiedQuery::builder()
            .vector_similarity(Vector::new(vec![0.1; 256]), 5)
            .build();
        
        let complex_query = UnifiedQuery::builder()
            .vector_similarity(Vector::new(vec![0.1; 256]), 100)
            .metadata_equals("category", json!("document"))
            .graph_from_entity(crate::core::entity::EntityId::new(), 3)
            .build();

        assert!(complex_query.complexity_score() > simple_query.complexity_score());
    }

    #[test]
    fn test_distance_metrics() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
        
        let vector_query = VectorQuery {
            query_vector: Vector::new(vec![0.1, 0.2, 0.3]),
            k: 10,
            distance_metric: DistanceMetric::Euclidean,
            vector_field: None,
        };
        
        assert_eq!(vector_query.distance_metric, DistanceMetric::Euclidean);
    }
}