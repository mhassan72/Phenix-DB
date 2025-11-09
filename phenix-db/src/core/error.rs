// Error types for Phenix-DB core module
//
// Comprehensive error handling framework with recovery strategies,
// context tracking, and distributed tracing support.
//
// Requirements: 10.1 (Mathematical Foundation), 13.2 (Observability)

use std::fmt;
use thiserror::Error;
use uuid::Uuid;

/// Result type alias for core operations
pub type Result<T> = std::result::Result<T, MemorySubstrateError>;

/// Result type alias for polynomial operations
pub type PolynomialResult<T> = std::result::Result<T, PolynomialError>;

/// Result type alias for graph operations
pub type GraphResult<T> = std::result::Result<T, GraphError>;

/// Result type alias for compression operations
pub type CompressionResult<T> = std::result::Result<T, CompressionError>;

/// Result type alias for consensus operations
pub type ConsensusResult<T> = std::result::Result<T, ConsensusError>;

/// Result type alias for tier operations
pub type TierResult<T> = std::result::Result<T, TierError>;

/// Result type alias for learning operations
pub type LearningResult<T> = std::result::Result<T, LearningError>;

/// Result type alias for concurrency operations
pub type ConcurrencyResult<T> = std::result::Result<T, ConcurrencyError>;

/// Correlation ID for distributed tracing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CorrelationId(Uuid);

impl CorrelationId {
    /// Create a new correlation ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the UUID value
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for CorrelationId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CorrelationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Error context for distributed tracing and debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Correlation ID for distributed tracing
    pub correlation_id: CorrelationId,
    /// Component where the error occurred
    pub component: String,
    /// Operation that failed
    pub operation: String,
    /// Additional context information
    pub details: Vec<(String, String)>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(component: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            correlation_id: CorrelationId::new(),
            component: component.into(),
            operation: operation.into(),
            details: Vec::new(),
        }
    }

    /// Add a detail to the context
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.push((key.into(), value.into()));
        self
    }

    /// Set the correlation ID
    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = correlation_id;
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}::{} ",
            self.correlation_id, self.component, self.operation
        )?;
        if !self.details.is_empty() {
            write!(f, "(")?;
            for (i, (k, v)) in self.details.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}={}", k, v)?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

/// Recovery strategy for errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Retry the operation
    Retry,
    /// Fallback to alternative approach
    Fallback,
    /// Skip and continue
    Skip,
    /// Abort the operation
    Abort,
    /// Propagate to caller
    Propagate,
}

/// Main error type for the memory substrate
#[derive(Debug, Error)]
pub enum MemorySubstrateError {
    /// Polynomial operation errors
    #[error("Polynomial error: {error}")]
    Polynomial {
        error: PolynomialError,
        context: Option<ErrorContext>,
    },

    /// Probabilistic graph operation errors
    #[error("Graph error: {error}")]
    Graph {
        error: GraphError,
        context: Option<ErrorContext>,
    },

    /// Compression operation errors
    #[error("Compression error: {error}")]
    Compression {
        error: CompressionError,
        context: Option<ErrorContext>,
    },

    /// Consensus operation errors
    #[error("Consensus error: {error}")]
    Consensus {
        error: ConsensusError,
        context: Option<ErrorContext>,
    },

    /// Memory tier operation errors
    #[error("Tier error: {error}")]
    Tier {
        error: TierError,
        context: Option<ErrorContext>,
    },

    /// Learning algorithm errors
    #[error("Learning error: {error}")]
    Learning {
        error: LearningError,
        context: Option<ErrorContext>,
    },

    /// Concurrency control errors
    #[error("Concurrency error: {error}")]
    Concurrency {
        error: ConcurrencyError,
        context: Option<ErrorContext>,
    },

    /// Mathematical invariant violations
    #[error("Invariant violation: {message}")]
    InvariantViolation {
        message: String,
        context: ErrorContext,
    },

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Internal errors
    #[error("Internal error: {0}")]
    Internal(String),
}

// Implement From traits manually for better error context handling
impl From<PolynomialError> for MemorySubstrateError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial {
            error,
            context: None,
        }
    }
}

impl From<GraphError> for MemorySubstrateError {
    fn from(error: GraphError) -> Self {
        Self::Graph {
            error,
            context: None,
        }
    }
}

impl From<CompressionError> for MemorySubstrateError {
    fn from(error: CompressionError) -> Self {
        Self::Compression {
            error,
            context: None,
        }
    }
}

impl From<ConsensusError> for MemorySubstrateError {
    fn from(error: ConsensusError) -> Self {
        Self::Consensus {
            error,
            context: None,
        }
    }
}

impl From<TierError> for MemorySubstrateError {
    fn from(error: TierError) -> Self {
        Self::Tier {
            error,
            context: None,
        }
    }
}

impl From<LearningError> for MemorySubstrateError {
    fn from(error: LearningError) -> Self {
        Self::Learning {
            error,
            context: None,
        }
    }
}

impl From<ConcurrencyError> for MemorySubstrateError {
    fn from(error: ConcurrencyError) -> Self {
        Self::Concurrency {
            error,
            context: None,
        }
    }
}

impl MemorySubstrateError {
    /// Add context to the error
    pub fn with_context(self, context: ErrorContext) -> Self {
        match self {
            Self::Polynomial { error, .. } => Self::Polynomial {
                error,
                context: Some(context),
            },
            Self::Graph { error, .. } => Self::Graph {
                error,
                context: Some(context),
            },
            Self::Compression { error, .. } => Self::Compression {
                error,
                context: Some(context),
            },
            Self::Consensus { error, .. } => Self::Consensus {
                error,
                context: Some(context),
            },
            Self::Tier { error, .. } => Self::Tier {
                error,
                context: Some(context),
            },
            Self::Learning { error, .. } => Self::Learning {
                error,
                context: Some(context),
            },
            Self::Concurrency { error, .. } => Self::Concurrency {
                error,
                context: Some(context),
            },
            other => other,
        }
    }

    /// Get the correlation ID if available
    pub fn correlation_id(&self) -> Option<CorrelationId> {
        match self {
            Self::Polynomial { context, .. }
            | Self::Graph { context, .. }
            | Self::Compression { context, .. }
            | Self::Consensus { context, .. }
            | Self::Tier { context, .. }
            | Self::Learning { context, .. }
            | Self::Concurrency { context, .. } => {
                context.as_ref().map(|c| c.correlation_id)
            }
            Self::InvariantViolation { context, .. } => Some(context.correlation_id),
            _ => None,
        }
    }

    /// Get the recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::Polynomial { error, .. } => error.recovery_strategy(),
            Self::Graph { error, .. } => error.recovery_strategy(),
            Self::Compression { error, .. } => error.recovery_strategy(),
            Self::Consensus { error, .. } => error.recovery_strategy(),
            Self::Tier { error, .. } => error.recovery_strategy(),
            Self::Learning { error, .. } => error.recovery_strategy(),
            Self::Concurrency { error, .. } => error.recovery_strategy(),
            Self::InvariantViolation { .. } => RecoveryStrategy::Abort,
            Self::Io(_) => RecoveryStrategy::Retry,
            Self::Serialization(_) => RecoveryStrategy::Abort,
            Self::Configuration(_) => RecoveryStrategy::Abort,
            Self::Internal(_) => RecoveryStrategy::Abort,
        }
    }
}

/// Polynomial operation errors
#[derive(Debug, Error, Clone)]
pub enum PolynomialError {
    /// Coefficient computation failed
    #[error("Failed to compute polynomial coefficients: {reason}")]
    CoefficientComputationFailed { reason: String },

    /// Evaluation failed
    #[error("Polynomial evaluation failed at x={x}: {reason}")]
    EvaluationFailed { x: f64, reason: String },

    /// Degree exceeds maximum
    #[error("Polynomial degree {degree} exceeds maximum {max_degree}")]
    DegreeExceeded { degree: usize, max_degree: usize },

    /// Numerical instability detected
    #[error("Numerical instability detected: {reason}")]
    NumericalInstability { reason: String },

    /// Precision tolerance violated
    #[error("Precision tolerance violated: error={error}, tolerance={tolerance}")]
    PrecisionViolation { error: f64, tolerance: f64 },

    /// Invalid polynomial structure
    #[error("Invalid polynomial structure: {reason}")]
    InvalidStructure { reason: String },
}

impl PolynomialError {
    /// Get the recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::CoefficientComputationFailed { .. } => RecoveryStrategy::Retry,
            Self::EvaluationFailed { .. } => RecoveryStrategy::Fallback,
            Self::DegreeExceeded { .. } => RecoveryStrategy::Abort,
            Self::NumericalInstability { .. } => RecoveryStrategy::Fallback,
            Self::PrecisionViolation { .. } => RecoveryStrategy::Retry,
            Self::InvalidStructure { .. } => RecoveryStrategy::Abort,
        }
    }
}

/// Probabilistic graph operation errors
#[derive(Debug, Error, Clone)]
pub enum GraphError {
    /// Edge not found
    #[error("Edge not found: source={source_id}, target={target_id}")]
    EdgeNotFound { source_id: String, target_id: String },

    /// Probability distribution invalid
    #[error("Probability distribution invalid: sum={sum}, expected=1.0, tolerance={tolerance}")]
    InvalidProbabilityDistribution {
        sum: f64,
        tolerance: f64,
    },

    /// Weight update failed
    #[error("Weight update failed: {reason}")]
    WeightUpdateFailed { reason: String },

    /// Traversal failed
    #[error("Graph traversal failed at depth {depth}: {reason}")]
    TraversalFailed { depth: usize, reason: String },

    /// Cycle detected
    #[error("Cycle detected in graph traversal")]
    CycleDetected,

    /// Co-access detection failed
    #[error("Co-access detection failed: {reason}")]
    CoAccessDetectionFailed { reason: String },
}

impl GraphError {
    /// Get the recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::EdgeNotFound { .. } => RecoveryStrategy::Skip,
            Self::InvalidProbabilityDistribution { .. } => RecoveryStrategy::Abort,
            Self::WeightUpdateFailed { .. } => RecoveryStrategy::Retry,
            Self::TraversalFailed { .. } => RecoveryStrategy::Fallback,
            Self::CycleDetected => RecoveryStrategy::Skip,
            Self::CoAccessDetectionFailed { .. } => RecoveryStrategy::Skip,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_id_creation() {
        let id1 = CorrelationId::new();
        let id2 = CorrelationId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new("RPI", "insert")
            .with_detail("entity_id", "123")
            .with_detail("degree", "5");
        
        assert_eq!(context.component, "RPI");
        assert_eq!(context.operation, "insert");
        assert_eq!(context.details.len(), 2);
    }

    #[test]
    fn test_polynomial_error_recovery_strategy() {
        let error = PolynomialError::EvaluationFailed {
            x: 1.0,
            reason: "overflow".to_string(),
        };
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Fallback);
    }

    #[test]
    fn test_graph_error_recovery_strategy() {
        let error = GraphError::EdgeNotFound {
            source_id: "a".to_string(),
            target_id: "b".to_string(),
        };
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Skip);
    }

    #[test]
    fn test_compression_error_recovery_strategy() {
        let error = CompressionError::DecompressionFailed {
            reason: "corrupted data".to_string(),
        };
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Abort);
    }

    #[test]
    fn test_consensus_error_recovery_strategy() {
        let error = ConsensusError::NotAchieved { attempts: 10 };
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Retry);
    }

    #[test]
    fn test_tier_error_recovery_strategy() {
        let error = TierError::CapacityExceeded {
            tier: "hot".to_string(),
            used: 1000,
            capacity: 900,
        };
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Fallback);
    }

    #[test]
    fn test_learning_error_recovery_strategy() {
        let error = LearningError::ConvergenceFailed { iterations: 1000 };
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Retry);
    }

    #[test]
    fn test_concurrency_error_recovery_strategy() {
        let error = ConcurrencyError::DeadlockDetected { transactions: 3 };
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Abort);
    }

    #[test]
    fn test_memory_substrate_error_with_context() {
        let poly_error = PolynomialError::DegreeExceeded {
            degree: 10,
            max_degree: 5,
        };
        let context = ErrorContext::new("RPI", "insert");
        let error = MemorySubstrateError::from(poly_error).with_context(context.clone());
        
        assert!(error.correlation_id().is_some());
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Abort);
    }

    #[test]
    fn test_error_conversion() {
        let poly_error = PolynomialError::NumericalInstability {
            reason: "overflow".to_string(),
        };
        let substrate_error: MemorySubstrateError = poly_error.into();
        
        match substrate_error {
            MemorySubstrateError::Polynomial { error, context } => {
                assert!(matches!(error, PolynomialError::NumericalInstability { .. }));
                assert!(context.is_none());
            }
            _ => panic!("Expected Polynomial error"),
        }
    }

    #[test]
    fn test_invariant_violation() {
        let context = ErrorContext::new("PGM", "normalize_probabilities");
        let error = MemorySubstrateError::InvariantViolation {
            message: "Probability sum != 1.0".to_string(),
            context,
        };
        
        assert!(error.correlation_id().is_some());
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Abort);
    }
}

/// Compression operation errors
#[derive(Debug, Error, Clone)]
pub enum CompressionError {
    /// Compression failed
    #[error("Compression failed: {reason}")]
    CompressionFailed { reason: String },

    /// Decompression failed
    #[error("Decompression failed: {reason}")]
    DecompressionFailed { reason: String },

    /// Compression ratio not met
    #[error("Compression ratio {actual} does not meet target {target}")]
    RatioNotMet { actual: f64, target: f64 },

    /// Decompression time exceeded
    #[error("Decompression time {actual_ms}ms exceeds limit {limit_ms}ms")]
    DecompressionTimeExceeded { actual_ms: u64, limit_ms: u64 },

    /// Fidelity loss detected
    #[error("Fidelity loss detected: {reason}")]
    FidelityLoss { reason: String },

    /// Pattern dictionary error
    #[error("Pattern dictionary error: {reason}")]
    DictionaryError { reason: String },

    /// Entropy calculation failed
    #[error("Entropy calculation failed: {reason}")]
    EntropyCalculationFailed { reason: String },
}

impl CompressionError {
    /// Get the recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::CompressionFailed { .. } => RecoveryStrategy::Fallback,
            Self::DecompressionFailed { .. } => RecoveryStrategy::Abort,
            Self::RatioNotMet { .. } => RecoveryStrategy::Skip,
            Self::DecompressionTimeExceeded { .. } => RecoveryStrategy::Fallback,
            Self::FidelityLoss { .. } => RecoveryStrategy::Abort,
            Self::DictionaryError { .. } => RecoveryStrategy::Retry,
            Self::EntropyCalculationFailed { .. } => RecoveryStrategy::Skip,
        }
    }
}

/// Consensus operation errors
#[derive(Debug, Error, Clone)]
pub enum ConsensusError {
    /// Consensus not achieved
    #[error("Consensus not achieved after {attempts} attempts")]
    NotAchieved { attempts: usize },

    /// Entropy convergence failed
    #[error("Entropy convergence failed: delta={delta}, threshold={threshold}")]
    EntropyConvergenceFailed { delta: f64, threshold: f64 },

    /// Quorum not reached
    #[error("Quorum not reached: {participants}/{required} participants")]
    QuorumNotReached {
        participants: usize,
        required: usize,
    },

    /// Node communication failed
    #[error("Node communication failed: {reason}")]
    CommunicationFailed { reason: String },

    /// State synchronization failed
    #[error("State synchronization failed: {reason}")]
    SynchronizationFailed { reason: String },
}

impl ConsensusError {
    /// Get the recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::NotAchieved { .. } => RecoveryStrategy::Retry,
            Self::EntropyConvergenceFailed { .. } => RecoveryStrategy::Retry,
            Self::QuorumNotReached { .. } => RecoveryStrategy::Retry,
            Self::CommunicationFailed { .. } => RecoveryStrategy::Retry,
            Self::SynchronizationFailed { .. } => RecoveryStrategy::Retry,
        }
    }
}

/// Memory tier operation errors
#[derive(Debug, Error, Clone)]
pub enum TierError {
    /// Promotion failed
    #[error("Tier promotion failed from {from} to {to}: {reason}")]
    PromotionFailed {
        from: String,
        to: String,
        reason: String,
    },

    /// Demotion failed
    #[error("Tier demotion failed from {from} to {to}: {reason}")]
    DemotionFailed {
        from: String,
        to: String,
        reason: String,
    },

    /// Tier not available
    #[error("Tier {tier} not available: {reason}")]
    TierNotAvailable { tier: String, reason: String },

    /// Capacity exceeded
    #[error("Tier {tier} capacity exceeded: {used}/{capacity}")]
    CapacityExceeded {
        tier: String,
        used: u64,
        capacity: u64,
    },

    /// Access latency exceeded
    #[error("Access latency {actual_ms}ms exceeds tier limit {limit_ms}ms")]
    LatencyExceeded { actual_ms: u64, limit_ms: u64 },
}

impl TierError {
    /// Get the recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::PromotionFailed { .. } => RecoveryStrategy::Retry,
            Self::DemotionFailed { .. } => RecoveryStrategy::Retry,
            Self::TierNotAvailable { .. } => RecoveryStrategy::Fallback,
            Self::CapacityExceeded { .. } => RecoveryStrategy::Fallback,
            Self::LatencyExceeded { .. } => RecoveryStrategy::Skip,
        }
    }
}

/// Learning algorithm errors
#[derive(Debug, Error, Clone)]
pub enum LearningError {
    /// Convergence failed
    #[error("Learning convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    /// Prediction accuracy too low
    #[error("Prediction accuracy {accuracy} below threshold {threshold}")]
    AccuracyTooLow { accuracy: f64, threshold: f64 },

    /// Sample size insufficient
    #[error("Sample size {actual} insufficient, need {required}")]
    InsufficientSamples { actual: usize, required: usize },

    /// Model update failed
    #[error("Model update failed: {reason}")]
    ModelUpdateFailed { reason: String },

    /// Pattern recognition failed
    #[error("Pattern recognition failed: {reason}")]
    PatternRecognitionFailed { reason: String },

    /// Feedback loop error
    #[error("Feedback loop error: {reason}")]
    FeedbackLoopError { reason: String },
}

impl LearningError {
    /// Get the recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::ConvergenceFailed { .. } => RecoveryStrategy::Retry,
            Self::AccuracyTooLow { .. } => RecoveryStrategy::Skip,
            Self::InsufficientSamples { .. } => RecoveryStrategy::Skip,
            Self::ModelUpdateFailed { .. } => RecoveryStrategy::Retry,
            Self::PatternRecognitionFailed { .. } => RecoveryStrategy::Skip,
            Self::FeedbackLoopError { .. } => RecoveryStrategy::Retry,
        }
    }
}

/// Concurrency control errors
#[derive(Debug, Error, Clone)]
pub enum ConcurrencyError {
    /// Transaction conflict
    #[error("Transaction conflict detected: {reason}")]
    TransactionConflict { reason: String },

    /// Lock acquisition failed
    #[error("Lock acquisition failed after {attempts} attempts")]
    LockAcquisitionFailed { attempts: usize },

    /// Deadlock detected
    #[error("Deadlock detected involving {transactions} transactions")]
    DeadlockDetected { transactions: usize },

    /// Version conflict
    #[error("Version conflict: expected={expected}, actual={actual}")]
    VersionConflict { expected: u64, actual: u64 },

    /// Snapshot isolation violation
    #[error("Snapshot isolation violation: {reason}")]
    SnapshotIsolationViolation { reason: String },

    /// Retry limit exceeded
    #[error("Retry limit exceeded: {attempts} attempts")]
    RetryLimitExceeded { attempts: usize },
}

impl ConcurrencyError {
    /// Get the recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::TransactionConflict { .. } => RecoveryStrategy::Retry,
            Self::LockAcquisitionFailed { .. } => RecoveryStrategy::Retry,
            Self::DeadlockDetected { .. } => RecoveryStrategy::Abort,
            Self::VersionConflict { .. } => RecoveryStrategy::Retry,
            Self::SnapshotIsolationViolation { .. } => RecoveryStrategy::Abort,
            Self::RetryLimitExceeded { .. } => RecoveryStrategy::Abort,
        }
    }
}
