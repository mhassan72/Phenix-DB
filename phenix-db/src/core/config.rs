// Configuration management for Phenix-DB
//
// This module provides configuration structures for all mathematical parameters
// and system settings. Configuration can be loaded from:
// - TOML configuration files (phenix-db.toml)
// - Environment variables (for deployment flexibility)
// - Programmatic defaults
//
// All configurations include validation to prevent invalid states.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Configuration error types
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Configuration file not found: {0}")]
    FileNotFound(String),
    
    #[error("Failed to parse configuration: {0}")]
    ParseError(String),
    
    #[error("Invalid configuration: {0}")]
    ValidationError(String),
    
    #[error("Environment variable error: {0}")]
    EnvError(String),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

/// Main configuration structure for Phenix-DB
///
/// Contains all mathematical parameters and system settings.
/// Requirements: 10.4, 12.4
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhenixConfig {
    /// Polynomial index configuration
    pub polynomial: PolynomialConfig,
    
    /// Probabilistic graph memory configuration
    pub pgm: PGMConfig,
    
    /// Bellman optimizer configuration
    pub bellman: BellmanConfig,
    
    /// Compression engine configuration
    pub compression: CompressionConfig,
    
    /// Adaptive learning configuration
    pub learning: LearningConfig,
    
    /// Memory tiering configuration
    pub tiering: TieringConfig,
    
    /// Distributed consciousness configuration
    pub distributed: DistributedConfig,
}

impl PhenixConfig {
    /// Load configuration from TOML file
    ///
    /// # Arguments
    /// * `path` - Path to TOML configuration file
    ///
    /// # Returns
    /// * Parsed and validated configuration
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        
        let contents = std::fs::read_to_string(path)
            .map_err(|_| ConfigError::FileNotFound(path_str.clone()))?;
        
        let config: Self = toml::from_str(&contents)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;
        
        config.validate()?;
        
        Ok(config)
    }
    
    /// Load configuration from environment variables
    ///
    /// Environment variables override file-based configuration.
    /// Prefix: PHENIX_DB_
    pub fn from_env() -> Result<Self> {
        envy::prefixed("PHENIX_DB_")
            .from_env()
            .map_err(|e| ConfigError::EnvError(e.to_string()))
    }
    
    /// Create configuration with sensible defaults
    pub fn default() -> Self {
        Self {
            polynomial: PolynomialConfig::default(),
            pgm: PGMConfig::default(),
            bellman: BellmanConfig::default(),
            compression: CompressionConfig::default(),
            learning: LearningConfig::default(),
            tiering: TieringConfig::default(),
            distributed: DistributedConfig::default(),
        }
    }
    
    /// Validate configuration parameters
    ///
    /// Ensures all parameters are within valid ranges:
    /// - Probabilities in [0, 1]
    /// - Positive thresholds
    /// - Valid time durations
    pub fn validate(&self) -> Result<()> {
        self.polynomial.validate()?;
        self.pgm.validate()?;
        self.bellman.validate()?;
        self.compression.validate()?;
        self.learning.validate()?;
        self.tiering.validate()?;
        self.distributed.validate()?;
        
        Ok(())
    }
}

/// Polynomial index configuration
///
/// Controls Recursive Polynomial Index (RPI) behavior.
/// Requirement: 1.1, 1.3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialConfig {
    /// Polynomial degree (default: 5)
    /// Higher degrees provide more precision but increase computation cost
    pub degree: usize,
    
    /// Branching factor for polynomial tree (default: 16)
    /// Controls tree width vs depth tradeoff
    pub branching_factor: usize,
    
    /// Maximum tree depth (default: 20)
    pub max_depth: usize,
    
    /// Node capacity before splitting (default: 100)
    pub node_capacity: usize,
    
    /// Precision tolerance for polynomial evaluation (default: 0.00001)
    /// Maximum acceptable error in polynomial computations
    pub precision_tolerance: f64,
}

impl PolynomialConfig {
    pub fn default() -> Self {
        Self {
            degree: 5,
            branching_factor: 16,
            max_depth: 20,
            node_capacity: 100,
            precision_tolerance: 0.00001,
        }
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.degree == 0 {
            return Err(ConfigError::ValidationError(
                "Polynomial degree must be positive".to_string()
            ));
        }
        
        if self.branching_factor < 2 {
            return Err(ConfigError::ValidationError(
                "Branching factor must be at least 2".to_string()
            ));
        }
        
        if self.precision_tolerance <= 0.0 || self.precision_tolerance >= 1.0 {
            return Err(ConfigError::ValidationError(
                "Precision tolerance must be in (0, 1)".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Probabilistic Graph Memory configuration
///
/// Controls PGM edge weight learning and pruning.
/// Requirement: 2.1, 2.2, 2.3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PGMConfig {
    /// Learning rate for edge weight updates (default: 0.1)
    /// Controls how quickly edge weights adapt to access patterns
    pub learning_rate: f32,
    
    /// Pruning threshold for low-probability edges (default: 0.01)
    /// Edges below this probability are removed after inactivity period
    pub pruning_threshold: f32,
    
    /// Co-access time window in milliseconds (default: 100)
    /// Entities accessed within this window are considered co-accessed
    pub co_access_window_ms: u64,
    
    /// Normalization interval in seconds (default: 10)
    /// How often to renormalize probability distributions
    pub normalization_interval_secs: u64,
    
    /// Inactivity period for pruning in days (default: 30)
    /// Edges inactive for this period are candidates for pruning
    pub inactivity_period_days: u64,
    
    /// Probability sum tolerance (default: 0.001)
    /// Maximum acceptable deviation from Σp = 1.0
    pub probability_tolerance: f32,
}

impl PGMConfig {
    pub fn default() -> Self {
        Self {
            learning_rate: 0.1,
            pruning_threshold: 0.01,
            co_access_window_ms: 100,
            normalization_interval_secs: 10,
            inactivity_period_days: 30,
            probability_tolerance: 0.001,
        }
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(ConfigError::ValidationError(
                "Learning rate must be in (0, 1]".to_string()
            ));
        }
        
        if self.pruning_threshold < 0.0 || self.pruning_threshold > 1.0 {
            return Err(ConfigError::ValidationError(
                "Pruning threshold must be in [0, 1]".to_string()
            ));
        }
        
        if self.probability_tolerance <= 0.0 || self.probability_tolerance >= 1.0 {
            return Err(ConfigError::ValidationError(
                "Probability tolerance must be in (0, 1)".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Bellman optimizer configuration
///
/// Controls dynamic path optimization and restructuring.
/// Requirement: 3.1, 3.2, 3.3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BellmanConfig {
    /// Observation window size (default: 1000 queries)
    /// Number of queries to observe before computing optimal paths
    pub observation_window: usize,
    
    /// Restructure threshold multiplier (default: 1.5)
    /// Trigger restructuring when cost exceeds this multiple of minimum
    pub restructure_threshold: f64,
    
    /// Update interval in seconds (default: 10)
    /// How often to recompute optimal paths
    pub update_interval_secs: u64,
    
    /// Restructure timeout in seconds (default: 60)
    /// Maximum time allowed for restructuring operations
    pub restructure_timeout_secs: u64,
    
    /// Discount factor for Bellman equation (default: 0.95)
    /// Controls weight of future costs vs immediate costs
    pub discount_factor: f64,
    
    /// Cost weights for optimization
    pub cost_weights: CostWeights,
}

impl BellmanConfig {
    pub fn default() -> Self {
        Self {
            observation_window: 1000,
            restructure_threshold: 1.5,
            update_interval_secs: 10,
            restructure_timeout_secs: 60,
            discount_factor: 0.95,
            cost_weights: CostWeights::default(),
        }
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.observation_window == 0 {
            return Err(ConfigError::ValidationError(
                "Observation window must be positive".to_string()
            ));
        }
        
        if self.restructure_threshold <= 1.0 {
            return Err(ConfigError::ValidationError(
                "Restructure threshold must be > 1.0".to_string()
            ));
        }
        
        if self.discount_factor <= 0.0 || self.discount_factor >= 1.0 {
            return Err(ConfigError::ValidationError(
                "Discount factor must be in (0, 1)".to_string()
            ));
        }
        
        self.cost_weights.validate()?;
        
        Ok(())
    }
}

/// Cost weights for Bellman optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostWeights {
    /// Weight for latency cost (default: 1.0)
    pub latency: f64,
    
    /// Weight for memory access cost (default: 0.5)
    pub memory: f64,
    
    /// Weight for disk I/O cost (default: 2.0)
    pub io: f64,
}

impl CostWeights {
    pub fn default() -> Self {
        Self {
            latency: 1.0,
            memory: 0.5,
            io: 2.0,
        }
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.latency < 0.0 || self.memory < 0.0 || self.io < 0.0 {
            return Err(ConfigError::ValidationError(
                "Cost weights must be non-negative".to_string()
            ));
        }
        
        Ok(())
    }
}


/// Compression engine configuration
///
/// Controls Kolmogorov Compression Engine (KCE) behavior.
/// Requirement: 4.1, 4.2, 4.3, 4.4, 4.5
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Minimum pattern frequency for dictionary (default: 100 entities)
    /// Patterns must appear in at least this many entities to be added to dictionary
    pub min_pattern_frequency: usize,
    
    /// Target compression ratio (default: 0.8 = 80%)
    /// Desired ratio of compressed_size / original_size
    pub target_compression_ratio: f32,
    
    /// Maximum decompression time in milliseconds (default: 5)
    /// Decompression must complete within this time
    pub max_decompression_time_ms: u64,
    
    /// Ramanujan series degree (default: 10)
    /// Degree of Ramanujan series for compression
    pub ramanujan_degree: usize,
    
    /// Dictionary size limit (default: 10000 patterns)
    /// Maximum number of patterns in shared dictionary
    pub dictionary_size_limit: usize,
    
    /// Minimum information density (default: 0.85 bits/byte)
    /// Minimum acceptable information density after compression
    pub min_information_density: f32,
}

impl CompressionConfig {
    pub fn default() -> Self {
        Self {
            min_pattern_frequency: 100,
            target_compression_ratio: 0.8,
            max_decompression_time_ms: 5,
            ramanujan_degree: 10,
            dictionary_size_limit: 10000,
            min_information_density: 0.85,
        }
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.target_compression_ratio <= 0.0 || self.target_compression_ratio > 1.0 {
            return Err(ConfigError::ValidationError(
                "Target compression ratio must be in (0, 1]".to_string()
            ));
        }
        
        if self.min_information_density <= 0.0 || self.min_information_density > 1.0 {
            return Err(ConfigError::ValidationError(
                "Minimum information density must be in (0, 1]".to_string()
            ));
        }
        
        if self.ramanujan_degree == 0 {
            return Err(ConfigError::ValidationError(
                "Ramanujan degree must be positive".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Adaptive learning configuration
///
/// Controls learning engine behavior and convergence.
/// Requirement: 16.1, 16.3, 16.4, 16.5
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Learning rate (default: 0.1)
    /// Controls how quickly the system adapts to new patterns
    pub learning_rate: f32,
    
    /// Convergence threshold (default: 0.001)
    /// Learning is considered converged when error < threshold
    pub convergence_threshold: f32,
    
    /// Sample window size (default: 1000 queries)
    /// Number of queries to observe for learning
    pub sample_window: usize,
    
    /// Minimum prediction accuracy (default: 0.8 = 80%)
    /// Target accuracy for access predictions
    pub min_prediction_accuracy: f32,
    
    /// Maximum CPU overhead (default: 0.05 = 5%)
    /// Learning should consume at most this fraction of CPU
    pub max_cpu_overhead: f32,
    
    /// Feedback interval in seconds (default: 60)
    /// How often to adjust parameters based on feedback
    pub feedback_interval_secs: u64,
}

impl LearningConfig {
    pub fn default() -> Self {
        Self {
            learning_rate: 0.1,
            convergence_threshold: 0.001,
            sample_window: 1000,
            min_prediction_accuracy: 0.8,
            max_cpu_overhead: 0.05,
            feedback_interval_secs: 60,
        }
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(ConfigError::ValidationError(
                "Learning rate must be in (0, 1]".to_string()
            ));
        }
        
        if self.convergence_threshold <= 0.0 {
            return Err(ConfigError::ValidationError(
                "Convergence threshold must be positive".to_string()
            ));
        }
        
        if self.min_prediction_accuracy < 0.0 || self.min_prediction_accuracy > 1.0 {
            return Err(ConfigError::ValidationError(
                "Minimum prediction accuracy must be in [0, 1]".to_string()
            ));
        }
        
        if self.max_cpu_overhead < 0.0 || self.max_cpu_overhead > 1.0 {
            return Err(ConfigError::ValidationError(
                "Maximum CPU overhead must be in [0, 1]".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Memory tiering configuration
///
/// Controls hierarchical memory tier behavior.
/// Requirement: 15.1, 15.3, 15.4, 15.5
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringConfig {
    /// Hot tier promotion threshold (accesses/hour, default: 100.0)
    /// Entities exceeding this frequency are promoted to hot tier
    pub hot_promotion_threshold: f32,
    
    /// Warm tier promotion threshold (accesses/hour, default: 10.0)
    /// Entities exceeding this frequency are promoted to warm tier
    pub warm_promotion_threshold: f32,
    
    /// Hot tier demotion threshold (accesses/hour, default: 1.0)
    /// Entities below this frequency are demoted from hot tier
    pub hot_demotion_threshold: f32,
    
    /// Warm tier demotion threshold (accesses/hour, default: 0.1)
    /// Entities below this frequency are demoted from warm tier
    pub warm_demotion_threshold: f32,
    
    /// Hot tier demotion period in hours (default: 24)
    /// Time below threshold before demotion from hot tier
    pub hot_demotion_period_hours: u64,
    
    /// Warm tier demotion period in hours (default: 168 = 7 days)
    /// Time below threshold before demotion from warm tier
    pub warm_demotion_period_hours: u64,
    
    /// Hot tier size percentage (default: 0.1 = 10%)
    /// Target percentage of data in hot tier
    pub hot_tier_size_pct: f32,
    
    /// Warm tier size percentage (default: 0.3 = 30%)
    /// Target percentage of data in warm tier
    pub warm_tier_size_pct: f32,
}

impl TieringConfig {
    pub fn default() -> Self {
        Self {
            hot_promotion_threshold: 100.0,
            warm_promotion_threshold: 10.0,
            hot_demotion_threshold: 1.0,
            warm_demotion_threshold: 0.1,
            hot_demotion_period_hours: 24,
            warm_demotion_period_hours: 168,
            hot_tier_size_pct: 0.1,
            warm_tier_size_pct: 0.3,
        }
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.hot_promotion_threshold <= self.warm_promotion_threshold {
            return Err(ConfigError::ValidationError(
                "Hot promotion threshold must be > warm promotion threshold".to_string()
            ));
        }
        
        if self.warm_promotion_threshold <= 0.0 {
            return Err(ConfigError::ValidationError(
                "Warm promotion threshold must be positive".to_string()
            ));
        }
        
        if self.hot_demotion_threshold >= self.hot_promotion_threshold {
            return Err(ConfigError::ValidationError(
                "Hot demotion threshold must be < hot promotion threshold".to_string()
            ));
        }
        
        if self.hot_tier_size_pct + self.warm_tier_size_pct >= 1.0 {
            return Err(ConfigError::ValidationError(
                "Hot + warm tier sizes must be < 100%".to_string()
            ));
        }
        
        if self.hot_tier_size_pct <= 0.0 || self.warm_tier_size_pct <= 0.0 {
            return Err(ConfigError::ValidationError(
                "Tier size percentages must be positive".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Distributed consciousness configuration
///
/// Controls distributed system behavior and consensus.
/// Requirement: 7.1, 7.2, 7.3, 7.4, 7.5
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Global awareness percentage (default: 0.1 = 10%)
    /// Percentage of global entity distribution each node samples
    pub global_awareness_pct: f32,
    
    /// Cluster join timeout in seconds (default: 30)
    /// Maximum time allowed for node to join cluster
    pub cluster_join_timeout_secs: u64,
    
    /// Routing accuracy target (default: 0.9 = 90%)
    /// Target accuracy for query routing decisions
    pub routing_accuracy_target: f32,
    
    /// Entropy convergence epsilon (default: 0.001)
    /// Consensus achieved when |H_i - H_j| < epsilon
    pub entropy_convergence_epsilon: f64,
    
    /// Heartbeat interval in milliseconds (default: 500)
    /// How often nodes send heartbeats
    pub heartbeat_interval_ms: u64,
    
    /// Node failure timeout in milliseconds (default: 1500)
    /// Time without heartbeat before node considered failed
    pub node_failure_timeout_ms: u64,
    
    /// Cross-datacenter latency threshold in milliseconds (default: 100)
    /// Latency above this triggers routing optimization
    pub cross_dc_latency_threshold_ms: u64,
    
    /// Minimum replicas (default: 3)
    /// Minimum number of replicas for each entity
    pub min_replicas: usize,
    
    /// Maximum replicas (default: 10)
    /// Maximum number of replicas for each entity
    pub max_replicas: usize,
    
    /// Consistency level (default: EventualConsistency)
    pub consistency_level: ConsistencyLevel,
}

impl DistributedConfig {
    pub fn default() -> Self {
        Self {
            global_awareness_pct: 0.1,
            cluster_join_timeout_secs: 30,
            routing_accuracy_target: 0.9,
            entropy_convergence_epsilon: 0.001,
            heartbeat_interval_ms: 500,
            node_failure_timeout_ms: 1500,
            cross_dc_latency_threshold_ms: 100,
            min_replicas: 3,
            max_replicas: 10,
            consistency_level: ConsistencyLevel::EventualConsistency,
        }
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.global_awareness_pct <= 0.0 || self.global_awareness_pct > 1.0 {
            return Err(ConfigError::ValidationError(
                "Global awareness percentage must be in (0, 1]".to_string()
            ));
        }
        
        if self.routing_accuracy_target < 0.0 || self.routing_accuracy_target > 1.0 {
            return Err(ConfigError::ValidationError(
                "Routing accuracy target must be in [0, 1]".to_string()
            ));
        }
        
        if self.entropy_convergence_epsilon <= 0.0 {
            return Err(ConfigError::ValidationError(
                "Entropy convergence epsilon must be positive".to_string()
            ));
        }
        
        if self.min_replicas == 0 {
            return Err(ConfigError::ValidationError(
                "Minimum replicas must be at least 1".to_string()
            ));
        }
        
        if self.max_replicas < self.min_replicas {
            return Err(ConfigError::ValidationError(
                "Maximum replicas must be >= minimum replicas".to_string()
            ));
        }
        
        if self.node_failure_timeout_ms <= self.heartbeat_interval_ms {
            return Err(ConfigError::ValidationError(
                "Node failure timeout must be > heartbeat interval".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Consistency level for distributed operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Strong consistency (linearizable)
    Strong,
    
    /// Eventual consistency (default)
    /// Converges within 1 second
    EventualConsistency,
    
    /// Causal consistency
    Causal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = PhenixConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_polynomial_config_validation() {
        let mut config = PolynomialConfig::default();
        assert!(config.validate().is_ok());
        
        // Invalid degree
        config.degree = 0;
        assert!(config.validate().is_err());
        
        // Invalid branching factor
        config.degree = 5;
        config.branching_factor = 1;
        assert!(config.validate().is_err());
        
        // Invalid precision tolerance
        config.branching_factor = 16;
        config.precision_tolerance = 0.0;
        assert!(config.validate().is_err());
    }


    #[test]
    fn test_pgm_config_validation() {
        let mut config = PGMConfig::default();
        assert!(config.validate().is_ok());
        
        // Invalid learning rate
        config.learning_rate = 0.0;
        assert!(config.validate().is_err());
        
        config.learning_rate = 1.5;
        assert!(config.validate().is_err());
        
        // Invalid pruning threshold
        config.learning_rate = 0.1;
        config.pruning_threshold = -0.1;
        assert!(config.validate().is_err());
        
        config.pruning_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_bellman_config_validation() {
        let mut config = BellmanConfig::default();
        assert!(config.validate().is_ok());
        
        // Invalid observation window
        config.observation_window = 0;
        assert!(config.validate().is_err());
        
        // Invalid restructure threshold
        config.observation_window = 1000;
        config.restructure_threshold = 0.5;
        assert!(config.validate().is_err());
        
        // Invalid discount factor
        config.restructure_threshold = 1.5;
        config.discount_factor = 0.0;
        assert!(config.validate().is_err());
        
        config.discount_factor = 1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_compression_config_validation() {
        let mut config = CompressionConfig::default();
        assert!(config.validate().is_ok());
        
        // Invalid compression ratio
        config.target_compression_ratio = 0.0;
        assert!(config.validate().is_err());
        
        config.target_compression_ratio = 1.5;
        assert!(config.validate().is_err());
        
        // Invalid information density
        config.target_compression_ratio = 0.8;
        config.min_information_density = 0.0;
        assert!(config.validate().is_err());
    }


    #[test]
    fn test_learning_config_validation() {
        let mut config = LearningConfig::default();
        assert!(config.validate().is_ok());
        
        // Invalid learning rate
        config.learning_rate = 0.0;
        assert!(config.validate().is_err());
        
        // Invalid prediction accuracy
        config.learning_rate = 0.1;
        config.min_prediction_accuracy = 1.5;
        assert!(config.validate().is_err());
        
        // Invalid CPU overhead
        config.min_prediction_accuracy = 0.8;
        config.max_cpu_overhead = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_tiering_config_validation() {
        let mut config = TieringConfig::default();
        assert!(config.validate().is_ok());
        
        // Hot threshold must be > warm threshold
        config.hot_promotion_threshold = 5.0;
        config.warm_promotion_threshold = 10.0;
        assert!(config.validate().is_err());
        
        // Tier sizes must sum to < 100%
        config.hot_promotion_threshold = 100.0;
        config.warm_promotion_threshold = 10.0;
        config.hot_tier_size_pct = 0.6;
        config.warm_tier_size_pct = 0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_distributed_config_validation() {
        let mut config = DistributedConfig::default();
        assert!(config.validate().is_ok());
        
        // Invalid awareness percentage
        config.global_awareness_pct = 0.0;
        assert!(config.validate().is_err());
        
        config.global_awareness_pct = 1.5;
        assert!(config.validate().is_err());
        
        // Invalid replica counts
        config.global_awareness_pct = 0.1;
        config.min_replicas = 0;
        assert!(config.validate().is_err());
        
        config.min_replicas = 5;
        config.max_replicas = 3;
        assert!(config.validate().is_err());
    }


    #[test]
    fn test_config_serialization() {
        let config = PhenixConfig::default();
        
        // Test TOML serialization
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: PhenixConfig = toml::from_str(&toml_str).unwrap();
        
        assert_eq!(config.polynomial.degree, deserialized.polynomial.degree);
        assert_eq!(config.pgm.learning_rate, deserialized.pgm.learning_rate);
        assert_eq!(config.bellman.discount_factor, deserialized.bellman.discount_factor);
    }
}
