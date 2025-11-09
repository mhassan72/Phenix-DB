//! Error types for Phenix-DB memory substrate

use thiserror::Error;

/// Main error type for memory substrate operations
#[derive(Debug, Error)]
pub enum MemorySubstrateError {
    /// Polynomial evaluation failed
    #[error("Polynomial evaluation failed: {0}")]
    PolynomialError(String),

    /// Probabilistic graph operation failed
    #[error("Probabilistic graph operation failed: {0}")]
    GraphError(String),

    /// Compression operation failed
    #[error("Compression failed: {0}")]
    CompressionError(String),

    /// Consensus not achieved
    #[error("Consensus not achieved: {0}")]
    ConsensusError(String),

    /// Memory tier operation failed
    #[error("Tier operation failed: {0}")]
    TierError(String),

    /// Learning algorithm failed
    #[error("Learning algorithm failed: {0}")]
    LearningError(String),

    /// Concurrency conflict detected
    #[error("Concurrency conflict: {0}")]
    ConcurrencyError(String),

    /// Mathematical invariant violated
    #[error("Mathematical invariant violated: {0}")]
    InvariantViolation(String),

    /// Entity not found
    #[error("Entity not found: {0}")]
    EntityNotFound(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    /// Storage operation failed
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Network operation failed
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Serialization/deserialization failed
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Encryption/decryption failed
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    /// Transaction failed
    #[error("Transaction error: {0}")]
    TransactionError(String),

    /// Generic internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type alias for memory substrate operations
pub type Result<T> = std::result::Result<T, MemorySubstrateError>;

impl MemorySubstrateError {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            MemorySubstrateError::ConcurrencyError(_)
                | MemorySubstrateError::NetworkError(_)
                | MemorySubstrateError::ConsensusError(_)
        )
    }

    /// Get error recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            MemorySubstrateError::PolynomialError(_) => RecoveryStrategy::RetryWithLowerDegree,
            MemorySubstrateError::GraphError(_) => RecoveryStrategy::UseStaticEdges,
            MemorySubstrateError::CompressionError(_) => RecoveryStrategy::StoreUncompressed,
            MemorySubstrateError::ConsensusError(_) => RecoveryStrategy::UseLocalDecision,
            MemorySubstrateError::TierError(_) => RecoveryStrategy::KeepInCurrentTier,
            MemorySubstrateError::LearningError(_) => RecoveryStrategy::UseDefaultParameters,
            MemorySubstrateError::ConcurrencyError(_) => RecoveryStrategy::ExponentialBackoff,
            _ => RecoveryStrategy::Fail,
        }
    }
}

/// Error recovery strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Retry with lower polynomial degree
    RetryWithLowerDegree,
    /// Use static edges instead of probabilistic
    UseStaticEdges,
    /// Store data uncompressed
    StoreUncompressed,
    /// Use local decision without consensus
    UseLocalDecision,
    /// Keep data in current tier
    KeepInCurrentTier,
    /// Use default parameters
    UseDefaultParameters,
    /// Apply exponential backoff and retry
    ExponentialBackoff,
    /// Fail the operation
    Fail,
}
