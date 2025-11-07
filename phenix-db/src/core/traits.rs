//! # Core Trait Interfaces
//!
//! This module defines the core trait interfaces for Phenix DB components,
//! providing abstractions for database operations, storage tiers, and
//! unified query planning across vector, metadata, and graph operations.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::core::{
    entity::{Entity, EntityId},
    vector::{Vector, VectorId},
    edges::{Edge, EdgeId},
    transaction::TransactionId,
    mvcc::{MVCCVersion, Snapshot},
    query::{UnifiedQuery, QueryResult},
    error::Result,
};

/// Main database API interface for unified operations
///
/// This trait defines the primary interface for Phenix DB, providing
/// unified operations across vectors, metadata, and graph relationships
/// with ACID transaction guarantees.
#[async_trait]
pub trait PhenixDBAPI: Send + Sync {
    /// Insert a new entity with optional vector, metadata, and edges
    async fn insert_entity(&mut self, entity: Entity) -> Result<EntityId>;

    /// Insert multiple entities in a batch operation
    async fn insert_entities(&mut self, entities: Vec<Entity>) -> Result<Vec<EntityId>>;

    /// Update an existing entity
    async fn update_entity(&mut self, entity: Entity) -> Result<MVCCVersion>;

    /// Get entity by ID with optional snapshot for consistent reads
    async fn get_entity(&self, id: EntityId, snapshot: Option<&Snapshot>) -> Result<Option<Entity>>;

    /// Get multiple entities by IDs
    async fn get_entities(&self, ids: Vec<EntityId>, snapshot: Option<&Snapshot>) -> Result<Vec<Option<Entity>>>;

    /// Delete entity by ID
    async fn delete_entity(&mut self, id: EntityId) -> Result<bool>;

    /// Execute unified query combining vector similarity, metadata filtering, and graph traversal
    async fn query(&self, query: UnifiedQuery) -> Result<QueryResult>;

    /// Begin a new transaction
    async fn begin_transaction(&mut self) -> Result<TransactionId>;

    /// Commit a transaction
    async fn commit_transaction(&mut self, tx_id: TransactionId) -> Result<()>;

    /// Rollback a transaction
    async fn rollback_transaction(&mut self, tx_id: TransactionId) -> Result<()>;

    /// Execute operations within a transaction context
    async fn with_transaction<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(&mut dyn EntityManager) -> Result<R> + Send,
        R: Send;

    /// Get database statistics
    async fn get_statistics(&self) -> Result<DatabaseStatistics>;

    /// Perform database maintenance operations
    async fn maintenance(&mut self) -> Result<MaintenanceResult>;
}

/// Entity management interface for CRUD operations
///
/// This trait provides lower-level entity management operations
/// that can be used within transactions or for direct entity manipulation.
#[async_trait]
pub trait EntityManager: Send + Sync {
    /// Create a new entity
    async fn create_entity(&mut self, entity: Entity, tx_id: Option<TransactionId>) -> Result<EntityId>;

    /// Read entity by ID
    async fn read_entity(&self, id: EntityId, snapshot: Option<&Snapshot>) -> Result<Option<Entity>>;

    /// Update existing entity
    async fn update_entity(&mut self, entity: Entity, tx_id: Option<TransactionId>) -> Result<MVCCVersion>;

    /// Delete entity
    async fn delete_entity(&mut self, id: EntityId, tx_id: Option<TransactionId>) -> Result<bool>;

    /// Check if entity exists
    async fn entity_exists(&self, id: EntityId, snapshot: Option<&Snapshot>) -> Result<bool>;

    /// Get entity version
    async fn get_entity_version(&self, id: EntityId) -> Result<Option<MVCCVersion>>;

    /// List entities with pagination
    async fn list_entities(&self, offset: usize, limit: usize, snapshot: Option<&Snapshot>) -> Result<Vec<Entity>>;

    /// Count total entities
    async fn count_entities(&self, snapshot: Option<&Snapshot>) -> Result<usize>;

    /// Add edge between entities
    async fn add_edge(&mut self, edge: Edge, tx_id: Option<TransactionId>) -> Result<EdgeId>;

    /// Remove edge between entities
    async fn remove_edge(&mut self, edge_id: EdgeId, tx_id: Option<TransactionId>) -> Result<bool>;

    /// Get edges for entity
    async fn get_entity_edges(&self, entity_id: EntityId, snapshot: Option<&Snapshot>) -> Result<Vec<Edge>>;
}

/// Unified query planner interface
///
/// This trait defines the interface for planning and executing unified queries
/// that combine vector similarity, metadata filtering, and graph traversal.
#[async_trait]
pub trait UnifiedQueryPlanner: Send + Sync {
    /// Plan query execution strategy
    async fn plan_query(&self, query: &UnifiedQuery) -> Result<QueryPlan>;

    /// Execute planned query
    async fn execute_query(&self, plan: QueryPlan) -> Result<QueryResult>;

    /// Execute query with automatic planning
    async fn execute_unified_query(&self, query: UnifiedQuery) -> Result<QueryResult> {
        let plan = self.plan_query(&query).await?;
        self.execute_query(plan).await
    }

    /// Optimize query for better performance
    async fn optimize_query(&self, query: UnifiedQuery) -> Result<UnifiedQuery>;

    /// Get query execution statistics
    async fn get_query_statistics(&self) -> Result<QueryPlannerStatistics>;

    /// Estimate query cost
    async fn estimate_query_cost(&self, query: &UnifiedQuery) -> Result<QueryCost>;
}

/// Storage tier interface for multi-tiered storage
///
/// This trait defines the interface for different storage tiers (hot/cold)
/// with automatic promotion and demotion based on access patterns.
#[async_trait]
pub trait StorageTier: Send + Sync {
    /// Store entity in this tier
    async fn store_entity(&mut self, entity: Entity) -> Result<StorageLocation>;

    /// Retrieve entity from this tier
    async fn retrieve_entity(&self, location: StorageLocation) -> Result<Option<Entity>>;

    /// Delete entity from this tier
    async fn delete_entity(&mut self, location: StorageLocation) -> Result<bool>;

    /// Check if entity exists in this tier
    async fn contains_entity(&self, location: StorageLocation) -> Result<bool>;

    /// Move entity to another tier
    async fn migrate_entity(&mut self, location: StorageLocation, target_tier: &mut dyn StorageTier) -> Result<StorageLocation>;

    /// Get tier statistics
    async fn get_tier_statistics(&self) -> Result<TierStatistics>;

    /// Perform tier maintenance (compression, cleanup, etc.)
    async fn maintenance(&mut self) -> Result<TierMaintenanceResult>;

    /// Get tier type identifier
    fn tier_type(&self) -> TierType;

    /// Get tier capacity information
    async fn get_capacity(&self) -> Result<TierCapacity>;
}

/// Vector index interface for similarity search
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// Insert vector into index
    async fn insert_vector(&mut self, vector_id: VectorId, vector: Vector) -> Result<()>;

    /// Search for similar vectors
    async fn search_similar(&self, query_vector: Vector, k: usize, threshold: Option<f32>) -> Result<Vec<VectorSearchResult>>;

    /// Update vector in index
    async fn update_vector(&mut self, vector_id: VectorId, vector: Vector) -> Result<()>;

    /// Remove vector from index
    async fn remove_vector(&mut self, vector_id: VectorId) -> Result<bool>;

    /// Get index statistics
    async fn get_index_statistics(&self) -> Result<IndexStatistics>;

    /// Rebuild index
    async fn rebuild_index(&mut self) -> Result<()>;

    /// Get index type
    fn index_type(&self) -> IndexType;
}

/// Graph index interface for relationship traversal
#[async_trait]
pub trait GraphIndex: Send + Sync {
    /// Add edge to graph index
    async fn add_edge(&mut self, edge: Edge) -> Result<()>;

    /// Remove edge from graph index
    async fn remove_edge(&mut self, edge_id: EdgeId) -> Result<bool>;

    /// Find neighbors of entity
    async fn get_neighbors(&self, entity_id: EntityId, max_depth: usize) -> Result<Vec<EntityId>>;

    /// Traverse graph from starting entities
    async fn traverse_graph(&self, start_entities: Vec<EntityId>, max_depth: usize, edge_labels: Vec<String>) -> Result<Vec<GraphTraversalResult>>;

    /// Get shortest path between entities
    async fn shortest_path(&self, from: EntityId, to: EntityId, max_depth: usize) -> Result<Option<Vec<EntityId>>>;

    /// Get graph statistics
    async fn get_graph_statistics(&self) -> Result<GraphStatistics>;
}

// Supporting types and structures

/// Query execution plan
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub query_id: String,
    pub execution_steps: Vec<ExecutionStep>,
    pub estimated_cost: QueryCost,
    pub parallelization_strategy: ParallelizationStrategy,
}

/// Query execution step
#[derive(Debug, Clone)]
pub enum ExecutionStep {
    VectorSearch { index_name: String, k: usize },
    MetadataFilter { filter_expression: String },
    GraphTraversal { start_entities: Vec<EntityId>, max_depth: usize },
    ResultMerging { strategy: MergingStrategy },
    Scoring { weights: HashMap<String, f32> },
}

/// Query cost estimation
#[derive(Debug, Clone)]
pub struct QueryCost {
    pub estimated_time_ms: u64,
    pub estimated_memory_mb: u64,
    pub estimated_io_operations: u64,
    pub complexity_score: f32,
}

/// Parallelization strategy
#[derive(Debug, Clone)]
pub enum ParallelizationStrategy {
    Sequential,
    Parallel { max_threads: usize },
    Distributed { shard_count: usize },
}

/// Result merging strategy
#[derive(Debug, Clone)]
pub enum MergingStrategy {
    Union,
    Intersection,
    WeightedCombination { weights: HashMap<String, f32> },
}

/// Storage location identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StorageLocation {
    pub tier: TierType,
    pub path: String,
    pub offset: Option<u64>,
    pub size: Option<u64>,
}

/// Storage tier types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TierType {
    Hot,    // RAM/NVMe
    Warm,   // SSD
    Cold,   // Object storage
}

/// Vector search result
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub vector_id: VectorId,
    pub entity_id: EntityId,
    pub similarity_score: f32,
    pub distance: f32,
}

/// Graph traversal result
#[derive(Debug, Clone)]
pub struct GraphTraversalResult {
    pub entity_id: EntityId,
    pub path: Vec<EdgeId>,
    pub depth: usize,
    pub total_weight: f32,
}

/// Index types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    HNSW,
    IVFPQ,
    BruteForce,
    Hybrid,
}

// Statistics structures

/// Database-wide statistics
#[derive(Debug, Clone)]
pub struct DatabaseStatistics {
    pub total_entities: usize,
    pub total_vectors: usize,
    pub total_edges: usize,
    pub storage_usage: StorageUsage,
    pub query_performance: QueryPerformanceStats,
    pub transaction_stats: TransactionStats,
}

/// Storage usage statistics
#[derive(Debug, Clone)]
pub struct StorageUsage {
    pub hot_tier_usage: TierStatistics,
    pub cold_tier_usage: TierStatistics,
    pub total_size_bytes: u64,
    pub compression_ratio: f32,
}

/// Tier-specific statistics
#[derive(Debug, Clone)]
pub struct TierStatistics {
    pub entity_count: usize,
    pub size_bytes: u64,
    pub access_frequency: f32,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
}

/// Tier capacity information
#[derive(Debug, Clone)]
pub struct TierCapacity {
    pub total_capacity_bytes: u64,
    pub used_capacity_bytes: u64,
    pub available_capacity_bytes: u64,
    pub utilization_percentage: f32,
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    pub index_type: IndexType,
    pub vector_count: usize,
    pub index_size_bytes: u64,
    pub build_time_ms: u64,
    pub average_search_time_ms: f32,
    pub memory_usage_bytes: u64,
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f32,
    pub max_degree: usize,
    pub connected_components: usize,
}

/// Query planner statistics
#[derive(Debug, Clone)]
pub struct QueryPlannerStatistics {
    pub total_queries_planned: usize,
    pub average_planning_time_ms: f32,
    pub cache_hit_rate: f32,
    pub optimization_success_rate: f32,
}

/// Query performance statistics
#[derive(Debug, Clone)]
pub struct QueryPerformanceStats {
    pub total_queries: usize,
    pub average_latency_ms: f32,
    pub p95_latency_ms: f32,
    pub p99_latency_ms: f32,
    pub throughput_qps: f32,
}

/// Transaction statistics
#[derive(Debug, Clone)]
pub struct TransactionStats {
    pub total_transactions: usize,
    pub committed_transactions: usize,
    pub aborted_transactions: usize,
    pub average_duration_ms: f32,
    pub deadlock_count: usize,
}

/// Maintenance operation results
#[derive(Debug, Clone)]
pub struct MaintenanceResult {
    pub operations_performed: Vec<String>,
    pub entities_processed: usize,
    pub storage_reclaimed_bytes: u64,
    pub duration_ms: u64,
    pub errors: Vec<String>,
}

/// Tier maintenance results
#[derive(Debug, Clone)]
pub struct TierMaintenanceResult {
    pub entities_compressed: usize,
    pub entities_migrated: usize,
    pub storage_reclaimed_bytes: u64,
    pub duration_ms: u64,
}

// Trait implementations for common operations
impl StorageLocation {
    pub fn new(tier: TierType, path: String) -> Self {
        Self {
            tier,
            path,
            offset: None,
            size: None,
        }
    }

    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }
}

impl std::fmt::Display for TierType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TierType::Hot => write!(f, "hot"),
            TierType::Warm => write!(f, "warm"),
            TierType::Cold => write!(f, "cold"),
        }
    }
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexType::HNSW => write!(f, "HNSW"),
            IndexType::IVFPQ => write!(f, "IVF-PQ"),
            IndexType::BruteForce => write!(f, "BruteForce"),
            IndexType::Hybrid => write!(f, "Hybrid"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_location() {
        let location = StorageLocation::new(TierType::Hot, "/path/to/entity".to_string())
            .with_offset(1024)
            .with_size(512);

        assert_eq!(location.tier, TierType::Hot);
        assert_eq!(location.path, "/path/to/entity");
        assert_eq!(location.offset, Some(1024));
        assert_eq!(location.size, Some(512));
    }

    #[test]
    fn test_tier_type_display() {
        assert_eq!(TierType::Hot.to_string(), "hot");
        assert_eq!(TierType::Warm.to_string(), "warm");
        assert_eq!(TierType::Cold.to_string(), "cold");
    }

    #[test]
    fn test_index_type_display() {
        assert_eq!(IndexType::HNSW.to_string(), "HNSW");
        assert_eq!(IndexType::IVFPQ.to_string(), "IVF-PQ");
        assert_eq!(IndexType::BruteForce.to_string(), "BruteForce");
        assert_eq!(IndexType::Hybrid.to_string(), "Hybrid");
    }
}