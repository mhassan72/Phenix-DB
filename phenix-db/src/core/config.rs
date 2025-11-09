//! Configuration management for Phenix-DB

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Polynomial index configuration
    pub polynomial_index: PolynomialConfig,
    
    /// Probabilistic graph memory configuration
    pub pgm: PGMConfig,
    
    /// Bellman optimizer configuration
    pub bellman: BellmanConfig,
    
    /// Compression engine configuration
    pub compression: CompressionConfig,
    
    /// Learning engine configuration
    pub learning: LearningConfig,
    
    /// Tiering configuration
    pub tiering: TieringConfig,
    
    /// Distributed system configuration
    pub distributed: DistributedConfig,
}

/// Polynomial index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialConfig {
    /// Polynomial degree (default: 5)
    pub degree: usize,
    
    /// Branching factor for tree (default: 16)
    pub branching_factor: usize,
    
    /// Precision tolerance (default: 0.00001)
    pub precision_tolerance: f64,
}

/// PGM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PGMConfig {
    /// Learning rate (default: 0.1)
    pub learning_rate: f32,
    
    /// Pruning threshold (default: 0.01)
    pub pruning_threshold: f32,
    
    /// Co-access window in milliseconds (default: 100)
    pub co_access_window_ms: u64,
    
    /// Normalization interval in seconds (default: 10)
    pub normalization_interval_sec: u64,
}

/// Bellman optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BellmanConfig {
    /// Observation window size (default: 1000)
    pub observation_window: usize,
    
    /// Restructure threshold multiplier (default: 1.5)
    pub restructure_threshold: f64,
    
    /// Update interval in seconds (default: 10)
    pub update_interval_sec: u64,
    
    /// Discount factor (default: 0.95)
    pub discount_factor: f64,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Target compression ratio (default: 0.8)
    pub target_ratio: f32,
    
    /// Maximum decompression time in milliseconds (default: 5)
    pub max_decompression_ms: u64,
    
    /// Minimum pattern frequency for dictionary (default: 100)
    pub min_pattern_frequency: usize,
    
    /// Ramanujan series degree (default: 10)
    pub ramanujan_series_degree: usize,
}

/// Learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Convergence threshold (default: 0.001)
    pub convergence_threshold: f32,
    
    /// Sample window size (default: 1000)
    pub sample_window: usize,
    
    /// Minimum accuracy (default: 0.75)
    pub min_accuracy: f32,
    
    /// Maximum CPU overhead (default: 0.05)
    pub max_cpu_overhead: f32,
}

/// Tiering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringConfig {
    /// Hot tier promotion threshold (accesses/hour, default: 100.0)
    pub hot_promotion_threshold: f32,
    
    /// Warm tier promotion threshold (accesses/hour, default: 10.0)
    pub warm_promotion_threshold: f32,
    
    /// Hot tier demotion threshold (accesses/hour, default: 1.0)
    pub hot_demotion_threshold: f32,
    
    /// Warm tier demotion threshold (accesses/hour, default: 0.1)
    pub warm_demotion_threshold: f32,
    
    /// Hot tier demotion period (default: 24 hours)
    pub hot_demotion_period: Duration,
    
    /// Warm tier demotion period (default: 7 days)
    pub warm_demotion_period: Duration,
}

/// Distributed system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Minimum replicas (default: 3)
    pub min_replicas: usize,
    
    /// Maximum replicas (default: 10)
    pub max_replicas: usize,
    
    /// Awareness percentage (default: 0.1)
    pub awareness_percentage: f32,
    
    /// Consensus timeout in seconds (default: 5)
    pub consensus_timeout_sec: u64,
    
    /// Heartbeat interval in milliseconds (default: 500)
    pub heartbeat_interval_ms: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            polynomial_index: PolynomialConfig {
                degree: 5,
                branching_factor: 16,
                precision_tolerance: 0.00001,
            },
            pgm: PGMConfig {
                learning_rate: 0.1,
                pruning_threshold: 0.01,
                co_access_window_ms: 100,
                normalization_interval_sec: 10,
            },
            bellman: BellmanConfig {
                observation_window: 1000,
                restructure_threshold: 1.5,
                update_interval_sec: 10,
                discount_factor: 0.95,
            },
            compression: CompressionConfig {
                target_ratio: 0.8,
                max_decompression_ms: 5,
                min_pattern_frequency: 100,
                ramanujan_series_degree: 10,
            },
            learning: LearningConfig {
                convergence_threshold: 0.001,
                sample_window: 1000,
                min_accuracy: 0.75,
                max_cpu_overhead: 0.05,
            },
            tiering: TieringConfig {
                hot_promotion_threshold: 100.0,
                warm_promotion_threshold: 10.0,
                hot_demotion_threshold: 1.0,
                warm_demotion_threshold: 0.1,
                hot_demotion_period: Duration::from_secs(24 * 3600),
                warm_demotion_period: Duration::from_secs(7 * 24 * 3600),
            },
            distributed: DistributedConfig {
                min_replicas: 3,
                max_replicas: 10,
                awareness_percentage: 0.1,
                consensus_timeout_sec: 5,
                heartbeat_interval_ms: 500,
            },
        }
    }
}
