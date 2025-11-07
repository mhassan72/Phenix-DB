//! # Error Handling Hierarchy
//!
//! This module defines the comprehensive error types and recovery strategies
//! for Phenix DB. The error hierarchy is designed to provide detailed context
//! for debugging while enabling appropriate recovery strategies at different
//! system levels.

use thiserror::Error;

/// Result type alias for Phenix DB operations
pub type Result<T> = std::result::Result<T, PhenixDBError>;

/// Top-level error types for Phenix DB operations
///
/// The error hierarchy is designed to provide specific error types with
/// recovery strategies while maintaining clear categorization for different
/// system components. Each error type includes detailed context and
/// suggestions for resolution.
#[derive(Debug, Error)]
pub enum PhenixDBError {
    // === Storage Errors ===
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    // === Transaction Errors ===
    #[error("Transaction error: {0}")]
    Transaction(#[from] TransactionError),

    // === Shard Errors ===
    #[error("Shard error: {0}")]
    Shard(#[from] ShardError),

    // === Index Errors ===
    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    // === API Errors ===
    #[error("API error: {0}")]
    API(#[from] APIError),

    // === Security Errors ===
    #[error("Security error: {0}")]
    Security(#[from] SecurityError),

    // === Configuration Errors ===
    #[error("Configuration error: {0}")]
    Configuration(#[from] ConfigError),

    // === Validation Errors ===
    #[error("Validation error: {message}")]
    ValidationError { message: String },

    // === Serialization Errors ===
    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    // === Entity-specific Errors ===
    #[error("Invalid entity ID: {0}")]
    InvalidEntityId(String),

    #[error("Entity not found: {id}")]
    EntityNotFound { id: String },

    // === Vector-specific Errors ===
    #[error("Invalid vector ID: {0}")]
    InvalidVectorId(String),

    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },

    // === Edge-specific Errors ===
    #[error("Invalid edge ID: {0}")]
    InvalidEdgeId(String),

    #[error("Edge not found: {id}")]
    EdgeNotFound { id: String },

    // === Query Errors ===
    #[error("Query error: {message}")]
    QueryError { message: String },

    #[error("Query timeout after {timeout_ms}ms")]
    QueryTimeout { timeout_ms: u64 },

    // === Network Errors ===
    #[error("Network error: {message}")]
    NetworkError { message: String },

    // === Resource Errors ===
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    // === Internal Errors ===
    #[error("Internal error: {message}")]
    InternalError { message: String },
}

/// Storage-specific error types with recovery strategies
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Hot tier full, attempting cold tier promotion")]
    HotTierFull,

    #[error("Cold tier unavailable, retrying with backoff")]
    ColdTierUnavailable,

    #[error("Data corruption detected in entity {entity_id}")]
    DataCorruption { entity_id: String },

    #[error("Encryption key not found for tenant {tenant_id}")]
    EncryptionKeyMissing { tenant_id: String },

    #[error("Compression failed: {reason}")]
    CompressionFailed { reason: String },

    #[error("Decompression failed: {reason}")]
    DecompressionFailed { reason: String },

    #[error("Storage backend error: {message}")]
    BackendError { message: String },

    #[error("Insufficient storage space: required {required_bytes} bytes, available {available_bytes} bytes")]
    InsufficientSpace { required_bytes: u64, available_bytes: u64 },

    #[error("Storage tier migration failed: {reason}")]
    TierMigrationFailed { reason: String },

    #[error("WAL write failed: {reason}")]
    WALWriteFailed { reason: String },

    #[error("WAL replay failed: {reason}")]
    WALReplayFailed { reason: String },
}

/// Transaction-specific error types
#[derive(Debug, Error)]
pub enum TransactionError {
    #[error("Transaction {tx_id} not found")]
    TransactionNotFound { tx_id: String },

    #[error("Transaction {tx_id} already committed")]
    AlreadyCommitted { tx_id: String },

    #[error("Transaction {tx_id} already aborted")]
    AlreadyAborted { tx_id: String },

    #[error("Transaction {tx_id} timed out after {timeout_ms}ms")]
    TransactionTimeout { tx_id: String, timeout_ms: u64 },

    #[error("Deadlock detected involving transactions: {tx_ids:?}")]
    DeadlockDetected { tx_ids: Vec<String> },

    #[error("Conflict detected: {message}")]
    ConflictDetected { message: String },

    #[error("Two-phase commit failed in prepare phase: {reason}")]
    PrephaseFailed { reason: String },

    #[error("Two-phase commit failed in commit phase: {reason}")]
    CommitPhaseFailed { reason: String },

    #[error("MVCC version conflict: expected {expected}, got {actual}")]
    VersionConflict { expected: u64, actual: u64 },

    #[error("Snapshot isolation violation: {message}")]
    SnapshotIsolationViolation { message: String },
}

/// Shard management error types
#[derive(Debug, Error)]
pub enum ShardError {
    #[error("Shard {shard_id} not found")]
    ShardNotFound { shard_id: String },

    #[error("Shard {shard_id} is unavailable")]
    ShardUnavailable { shard_id: String },

    #[error("Shard rebalancing failed: {reason}")]
    RebalancingFailed { reason: String },

    #[error("Consistent hashing error: {message}")]
    ConsistentHashingError { message: String },

    #[error("Shard migration failed from {from_shard} to {to_shard}: {reason}")]
    MigrationFailed { from_shard: String, to_shard: String, reason: String },

    #[error("Replica synchronization failed for shard {shard_id}: {reason}")]
    ReplicaSyncFailed { shard_id: String, reason: String },

    #[error("Shard capacity exceeded: {shard_id}")]
    CapacityExceeded { shard_id: String },
}

/// Index operation error types
#[derive(Debug, Error)]
pub enum IndexError {
    #[error("Index {index_name} not found")]
    IndexNotFound { index_name: String },

    #[error("Index build failed for {index_name}: {reason}")]
    IndexBuildFailed { index_name: String, reason: String },

    #[error("Index corruption detected in {index_name}")]
    IndexCorruption { index_name: String },

    #[error("HNSW index error: {message}")]
    HNSWError { message: String },

    #[error("IVF-PQ index error: {message}")]
    IVFPQError { message: String },

    #[error("Metadata index error: {message}")]
    MetadataIndexError { message: String },

    #[error("Graph index error: {message}")]
    GraphIndexError { message: String },

    #[error("Index search failed: {reason}")]
    SearchFailed { reason: String },

    #[error("Index update failed: {reason}")]
    UpdateFailed { reason: String },
}

/// API layer error types
#[derive(Debug, Error)]
pub enum APIError {
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },

    #[error("Authentication failed: {reason}")]
    AuthenticationFailed { reason: String },

    #[error("Authorization failed: insufficient permissions for {operation}")]
    AuthorizationFailed { operation: String },

    #[error("Rate limit exceeded: {limit} requests per {window}")]
    RateLimitExceeded { limit: u32, window: String },

    #[error("Request too large: {size} bytes exceeds limit of {limit} bytes")]
    RequestTooLarge { size: usize, limit: usize },

    #[error("Unsupported API version: {version}")]
    UnsupportedVersion { version: String },

    #[error("gRPC error: {message}")]
    GrpcError { message: String },

    #[error("REST API error: {status_code} - {message}")]
    RestError { status_code: u16, message: String },

    #[error("Protocol buffer error: {message}")]
    ProtocolBufferError { message: String },
}

/// Security-related error types
#[derive(Debug, Error)]
pub enum SecurityError {
    #[error("Encryption failed: {reason}")]
    EncryptionFailed { reason: String },

    #[error("Decryption failed: {reason}")]
    DecryptionFailed { reason: String },

    #[error("Key management error: {message}")]
    KeyManagementError { message: String },

    #[error("Certificate validation failed: {reason}")]
    CertificateValidationFailed { reason: String },

    #[error("Tenant isolation violation: {message}")]
    TenantIsolationViolation { message: String },

    #[error("Audit log write failed: {reason}")]
    AuditLogFailed { reason: String },

    #[error("Security policy violation: {policy}")]
    PolicyViolation { policy: String },
}

/// Configuration error types
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },

    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    #[error("Missing required configuration: {key}")]
    MissingRequired { key: String },

    #[error("Configuration validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("Environment variable error: {message}")]
    EnvironmentError { message: String },
}

// Convenience constructors for common error patterns
impl PhenixDBError {
    /// Create a validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::ValidationError { message: message.into() }
    }

    /// Create a serialization error
    pub fn serialization<S: Into<String>>(message: S) -> Self {
        Self::SerializationError { message: message.into() }
    }

    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::InternalError { message: message.into() }
    }

    /// Create a query error
    pub fn query<S: Into<String>>(message: S) -> Self {
        Self::QueryError { message: message.into() }
    }

    /// Create a network error
    pub fn network<S: Into<String>>(message: S) -> Self {
        Self::NetworkError { message: message.into() }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted<S: Into<String>>(resource: S) -> Self {
        Self::ResourceExhausted { resource: resource.into() }
    }
}

// Error recovery and classification traits
impl PhenixDBError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            // Retryable storage errors
            PhenixDBError::Storage(StorageError::ColdTierUnavailable) => true,
            PhenixDBError::Storage(StorageError::HotTierFull) => true,
            
            // Retryable transaction errors
            PhenixDBError::Transaction(TransactionError::ConflictDetected { .. }) => true,
            PhenixDBError::Transaction(TransactionError::TransactionTimeout { .. }) => true,
            
            // Retryable shard errors
            PhenixDBError::Shard(ShardError::ShardUnavailable { .. }) => true,
            PhenixDBError::Shard(ShardError::ReplicaSyncFailed { .. }) => true,
            
            // Retryable network errors
            PhenixDBError::NetworkError { .. } => true,
            PhenixDBError::QueryTimeout { .. } => true,
            
            // Non-retryable errors
            _ => false,
        }
    }

    /// Check if the error indicates a permanent failure
    pub fn is_permanent(&self) -> bool {
        match self {
            // Permanent validation errors
            PhenixDBError::ValidationError { .. } => true,
            PhenixDBError::DimensionMismatch { .. } => true,
            
            // Permanent security errors
            PhenixDBError::Security(SecurityError::TenantIsolationViolation { .. }) => true,
            PhenixDBError::Security(SecurityError::PolicyViolation { .. }) => true,
            
            // Permanent API errors
            PhenixDBError::API(APIError::AuthorizationFailed { .. }) => true,
            PhenixDBError::API(APIError::UnsupportedVersion { .. }) => true,
            
            // Permanent configuration errors
            PhenixDBError::Configuration(_) => true,
            
            _ => false,
        }
    }

    /// Get suggested retry delay in milliseconds
    pub fn retry_delay_ms(&self) -> Option<u64> {
        if !self.is_retryable() {
            return None;
        }

        match self {
            // Short delays for transient issues
            PhenixDBError::Transaction(TransactionError::ConflictDetected { .. }) => Some(10),
            PhenixDBError::Storage(StorageError::HotTierFull) => Some(100),
            
            // Medium delays for resource issues
            PhenixDBError::Shard(ShardError::ShardUnavailable { .. }) => Some(1000),
            PhenixDBError::NetworkError { .. } => Some(1000),
            
            // Longer delays for infrastructure issues
            PhenixDBError::Storage(StorageError::ColdTierUnavailable) => Some(5000),
            
            _ => Some(1000), // Default 1 second
        }
    }

    /// Get error category for metrics and monitoring
    pub fn category(&self) -> &'static str {
        match self {
            PhenixDBError::Storage(_) => "storage",
            PhenixDBError::Transaction(_) => "transaction",
            PhenixDBError::Shard(_) => "shard",
            PhenixDBError::Index(_) => "index",
            PhenixDBError::API(_) => "api",
            PhenixDBError::Security(_) => "security",
            PhenixDBError::Configuration(_) => "configuration",
            PhenixDBError::ValidationError { .. } => "validation",
            PhenixDBError::SerializationError { .. } => "serialization",
            PhenixDBError::QueryError { .. } => "query",
            PhenixDBError::NetworkError { .. } => "network",
            _ => "other",
        }
    }
}

// Standard library error conversions
impl From<std::io::Error> for PhenixDBError {
    fn from(err: std::io::Error) -> Self {
        PhenixDBError::Storage(StorageError::BackendError {
            message: err.to_string(),
        })
    }
}

impl From<serde_json::Error> for PhenixDBError {
    fn from(err: serde_json::Error) -> Self {
        PhenixDBError::SerializationError {
            message: err.to_string(),
        }
    }
}

impl From<uuid::Error> for PhenixDBError {
    fn from(err: uuid::Error) -> Self {
        PhenixDBError::ValidationError {
            message: format!("UUID error: {}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_retryability() {
        let retryable_error = PhenixDBError::Storage(StorageError::ColdTierUnavailable);
        assert!(retryable_error.is_retryable());
        assert!(!retryable_error.is_permanent());

        let permanent_error = PhenixDBError::ValidationError {
            message: "Invalid input".to_string(),
        };
        assert!(!permanent_error.is_retryable());
        assert!(permanent_error.is_permanent());
    }

    #[test]
    fn test_error_categories() {
        let storage_error = PhenixDBError::Storage(StorageError::HotTierFull);
        assert_eq!(storage_error.category(), "storage");

        let api_error = PhenixDBError::API(APIError::RateLimitExceeded {
            limit: 100,
            window: "minute".to_string(),
        });
        assert_eq!(api_error.category(), "api");
    }

    #[test]
    fn test_retry_delays() {
        let conflict_error = PhenixDBError::Transaction(TransactionError::ConflictDetected {
            message: "Version conflict".to_string(),
        });
        assert_eq!(conflict_error.retry_delay_ms(), Some(10));

        let permanent_error = PhenixDBError::ValidationError {
            message: "Invalid".to_string(),
        };
        assert_eq!(permanent_error.retry_delay_ms(), None);
    }

    #[test]
    fn test_convenience_constructors() {
        let validation_error = PhenixDBError::validation("Test validation error");
        match validation_error {
            PhenixDBError::ValidationError { message } => {
                assert_eq!(message, "Test validation error");
            }
            _ => panic!("Expected ValidationError"),
        }
    }
}