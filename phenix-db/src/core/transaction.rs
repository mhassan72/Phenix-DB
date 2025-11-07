//! # Transaction Management
//!
//! This module implements ACID transaction support for Phenix DB, providing
//! distributed transaction coordination with two-phase commit protocol
//! and MVCC integration for unified entity operations.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::core::{
    entity::EntityId,
    mvcc::{MVCCVersion, Snapshot},
    error::{PhenixDBError, TransactionError, Result},
};

/// Unique identifier for transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransactionId(pub Uuid);

impl TransactionId {
    /// Generate a new random TransactionId
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create TransactionId from string representation
    pub fn from_string(s: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(s)
            .map_err(|e| PhenixDBError::Transaction(TransactionError::TransactionNotFound {
                tx_id: format!("Invalid UUID format: {}", e),
            }))?;
        Ok(Self(uuid))
    }

    /// Get string representation of TransactionId
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl Default for TransactionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Transaction isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsolationLevel {
    /// Read uncommitted (lowest isolation)
    ReadUncommitted,
    /// Read committed
    ReadCommitted,
    /// Repeatable read
    RepeatableRead,
    /// Serializable (highest isolation)
    Serializable,
}

impl Default for IsolationLevel {
    fn default() -> Self {
        Self::RepeatableRead
    }
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionStatus {
    /// Transaction is active and accepting operations
    Active,
    /// Transaction is preparing to commit (2PC prepare phase)
    Preparing,
    /// Transaction is prepared and waiting for commit decision
    Prepared,
    /// Transaction is committing
    Committing,
    /// Transaction has been committed successfully
    Committed,
    /// Transaction is aborting
    Aborting,
    /// Transaction has been aborted
    Aborted,
    /// Transaction timed out
    TimedOut,
}

impl TransactionStatus {
    /// Check if transaction is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Committed | Self::Aborted | Self::TimedOut)
    }

    /// Check if transaction is active (can accept operations)
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Check if transaction can be committed
    pub fn can_commit(&self) -> bool {
        matches!(self, Self::Active | Self::Prepared)
    }

    /// Check if transaction can be aborted
    pub fn can_abort(&self) -> bool {
        !self.is_terminal()
    }
}

/// Shard identifier for distributed transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId(pub u32);

impl ShardId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shard_{}", self.0)
    }
}

/// Transaction operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionOperation {
    /// Insert new entity
    Insert {
        entity_id: EntityId,
        shard_id: ShardId,
    },
    /// Update existing entity
    Update {
        entity_id: EntityId,
        old_version: MVCCVersion,
        new_version: MVCCVersion,
        shard_id: ShardId,
    },
    /// Delete entity
    Delete {
        entity_id: EntityId,
        version: MVCCVersion,
        shard_id: ShardId,
    },
}

impl TransactionOperation {
    /// Get the entity ID involved in this operation
    pub fn entity_id(&self) -> EntityId {
        match self {
            Self::Insert { entity_id, .. } => *entity_id,
            Self::Update { entity_id, .. } => *entity_id,
            Self::Delete { entity_id, .. } => *entity_id,
        }
    }

    /// Get the shard ID for this operation
    pub fn shard_id(&self) -> ShardId {
        match self {
            Self::Insert { shard_id, .. } => *shard_id,
            Self::Update { shard_id, .. } => *shard_id,
            Self::Delete { shard_id, .. } => *shard_id,
        }
    }
}

/// Transaction context for ACID operations
///
/// The Transaction represents a unit of work that maintains ACID properties
/// across unified entity operations (vector + metadata + edges). Transactions
/// support distributed operations across multiple shards with two-phase commit
/// protocol and MVCC snapshot isolation.
#[derive(Debug, Clone)]
pub struct Transaction {
    /// Unique transaction identifier
    pub id: TransactionId,
    
    /// Transaction start timestamp
    pub start_timestamp: DateTime<Utc>,
    
    /// Isolation level for this transaction
    pub isolation_level: IsolationLevel,
    
    /// Current transaction status
    pub status: TransactionStatus,
    
    /// Shards participating in this transaction
    pub shard_participants: HashSet<ShardId>,
    
    /// Operations performed in this transaction
    pub operations: Vec<TransactionOperation>,
    
    /// MVCC snapshot for consistent reads
    pub snapshot: Option<Snapshot>,
    
    /// Transaction timeout
    pub timeout: std::time::Duration,
    
    /// Optional tenant ID for multi-tenant isolation
    pub tenant_id: Option<String>,
}

impl Transaction {
    /// Create a new transaction
    pub fn new(isolation_level: IsolationLevel) -> Self {
        Self {
            id: TransactionId::new(),
            start_timestamp: Utc::now(),
            isolation_level,
            status: TransactionStatus::Active,
            shard_participants: HashSet::new(),
            operations: Vec::new(),
            snapshot: None,
            timeout: std::time::Duration::from_secs(30), // Default 30 second timeout
            tenant_id: None,
        }
    }

    /// Create a new transaction with default isolation level
    pub fn new_default() -> Self {
        Self::new(IsolationLevel::default())
    }

    /// Set transaction timeout
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set tenant ID for multi-tenant isolation
    pub fn with_tenant_id(mut self, tenant_id: String) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }

    /// Add operation to transaction
    pub fn add_operation(&mut self, operation: TransactionOperation) -> Result<()> {
        if !self.status.is_active() {
            return Err(PhenixDBError::Transaction(TransactionError::AlreadyCommitted {
                tx_id: self.id.to_string(),
            }));
        }

        // Add shard to participants
        self.shard_participants.insert(operation.shard_id());
        
        // Add operation to list
        self.operations.push(operation);
        
        Ok(())
    }

    /// Set MVCC snapshot for consistent reads
    pub fn set_snapshot(&mut self, snapshot: Snapshot) {
        self.snapshot = Some(snapshot);
    }

    /// Get MVCC snapshot
    pub fn get_snapshot(&self) -> Option<&Snapshot> {
        self.snapshot.as_ref()
    }

    /// Check if transaction has timed out
    pub fn is_timed_out(&self) -> bool {
        let elapsed = Utc::now().signed_duration_since(self.start_timestamp);
        elapsed.to_std().unwrap_or_default() > self.timeout
    }

    /// Get participating shards
    pub fn get_participating_shards(&self) -> &HashSet<ShardId> {
        &self.shard_participants
    }

    /// Check if transaction is distributed (multiple shards)
    pub fn is_distributed(&self) -> bool {
        self.shard_participants.len() > 1
    }

    /// Get operations for specific shard
    pub fn get_shard_operations(&self, shard_id: ShardId) -> Vec<&TransactionOperation> {
        self.operations
            .iter()
            .filter(|op| op.shard_id() == shard_id)
            .collect()
    }

    /// Transition to new status
    pub fn transition_to(&mut self, new_status: TransactionStatus) -> Result<()> {
        match (self.status, new_status) {
            // Valid transitions
            (TransactionStatus::Active, TransactionStatus::Preparing) => {},
            (TransactionStatus::Preparing, TransactionStatus::Prepared) => {},
            (TransactionStatus::Preparing, TransactionStatus::Aborting) => {},
            (TransactionStatus::Prepared, TransactionStatus::Committing) => {},
            (TransactionStatus::Prepared, TransactionStatus::Aborting) => {},
            (TransactionStatus::Committing, TransactionStatus::Committed) => {},
            (TransactionStatus::Aborting, TransactionStatus::Aborted) => {},
            (_, TransactionStatus::TimedOut) if self.is_timed_out() => {},
            
            // Invalid transitions
            _ => {
                return Err(PhenixDBError::Transaction(TransactionError::ConflictDetected {
                    message: format!("Invalid status transition from {:?} to {:?}", self.status, new_status),
                }));
            }
        }

        self.status = new_status;
        Ok(())
    }

    /// Calculate transaction duration
    pub fn duration(&self) -> std::time::Duration {
        let elapsed = Utc::now().signed_duration_since(self.start_timestamp);
        elapsed.to_std().unwrap_or_default()
    }

    /// Get transaction statistics
    pub fn get_statistics(&self) -> TransactionStatistics {
        let mut entity_count = HashSet::new();
        let mut insert_count = 0;
        let mut update_count = 0;
        let mut delete_count = 0;

        for operation in &self.operations {
            entity_count.insert(operation.entity_id());
            match operation {
                TransactionOperation::Insert { .. } => insert_count += 1,
                TransactionOperation::Update { .. } => update_count += 1,
                TransactionOperation::Delete { .. } => delete_count += 1,
            }
        }

        TransactionStatistics {
            operation_count: self.operations.len(),
            entity_count: entity_count.len(),
            shard_count: self.shard_participants.len(),
            insert_count,
            update_count,
            delete_count,
            duration: self.duration(),
            is_distributed: self.is_distributed(),
        }
    }
}

/// Transaction statistics for monitoring
#[derive(Debug, Clone)]
pub struct TransactionStatistics {
    pub operation_count: usize,
    pub entity_count: usize,
    pub shard_count: usize,
    pub insert_count: usize,
    pub update_count: usize,
    pub delete_count: usize,
    pub duration: std::time::Duration,
    pub is_distributed: bool,
}

/// Distributed transaction coordinator
#[derive(Debug)]
pub struct TransactionCoordinator {
    /// Active transactions
    active_transactions: HashMap<TransactionId, Transaction>,
    
    /// Transaction timeout checker
    timeout_checker: std::time::Instant,
}

impl TransactionCoordinator {
    /// Create new transaction coordinator
    pub fn new() -> Self {
        Self {
            active_transactions: HashMap::new(),
            timeout_checker: std::time::Instant::now(),
        }
    }

    /// Begin new transaction
    pub fn begin_transaction(&mut self, isolation_level: IsolationLevel) -> TransactionId {
        let transaction = Transaction::new(isolation_level);
        let tx_id = transaction.id;
        self.active_transactions.insert(tx_id, transaction);
        tx_id
    }

    /// Get transaction by ID
    pub fn get_transaction(&self, tx_id: TransactionId) -> Option<&Transaction> {
        self.active_transactions.get(&tx_id)
    }

    /// Get mutable transaction by ID
    pub fn get_transaction_mut(&mut self, tx_id: TransactionId) -> Option<&mut Transaction> {
        self.active_transactions.get_mut(&tx_id)
    }

    /// Add operation to transaction
    pub fn add_operation(&mut self, tx_id: TransactionId, operation: TransactionOperation) -> Result<()> {
        let transaction = self.active_transactions
            .get_mut(&tx_id)
            .ok_or_else(|| PhenixDBError::Transaction(TransactionError::TransactionNotFound {
                tx_id: tx_id.to_string(),
            }))?;

        transaction.add_operation(operation)
    }

    /// Prepare transaction (2PC prepare phase)
    pub fn prepare_transaction(&mut self, tx_id: TransactionId) -> Result<()> {
        let transaction = self.active_transactions
            .get_mut(&tx_id)
            .ok_or_else(|| PhenixDBError::Transaction(TransactionError::TransactionNotFound {
                tx_id: tx_id.to_string(),
            }))?;

        if transaction.is_timed_out() {
            transaction.transition_to(TransactionStatus::TimedOut)?;
            return Err(PhenixDBError::Transaction(TransactionError::TransactionTimeout {
                tx_id: tx_id.to_string(),
                timeout_ms: transaction.timeout.as_millis() as u64,
            }));
        }

        transaction.transition_to(TransactionStatus::Preparing)?;
        // TODO: Implement actual prepare logic with shard coordination
        transaction.transition_to(TransactionStatus::Prepared)?;
        
        Ok(())
    }

    /// Commit transaction
    pub fn commit_transaction(&mut self, tx_id: TransactionId) -> Result<()> {
        let transaction = self.active_transactions
            .get_mut(&tx_id)
            .ok_or_else(|| PhenixDBError::Transaction(TransactionError::TransactionNotFound {
                tx_id: tx_id.to_string(),
            }))?;

        if !transaction.status.can_commit() {
            return Err(PhenixDBError::Transaction(TransactionError::ConflictDetected {
                message: format!("Transaction {} cannot be committed in status {:?}", tx_id, transaction.status),
            }));
        }

        transaction.transition_to(TransactionStatus::Committing)?;
        // TODO: Implement actual commit logic with shard coordination
        transaction.transition_to(TransactionStatus::Committed)?;
        
        // Remove from active transactions
        self.active_transactions.remove(&tx_id);
        
        Ok(())
    }

    /// Abort transaction
    pub fn abort_transaction(&mut self, tx_id: TransactionId) -> Result<()> {
        let transaction = self.active_transactions
            .get_mut(&tx_id)
            .ok_or_else(|| PhenixDBError::Transaction(TransactionError::TransactionNotFound {
                tx_id: tx_id.to_string(),
            }))?;

        if !transaction.status.can_abort() {
            return Err(PhenixDBError::Transaction(TransactionError::AlreadyCommitted {
                tx_id: tx_id.to_string(),
            }));
        }

        transaction.transition_to(TransactionStatus::Aborting)?;
        // TODO: Implement actual abort logic with shard coordination
        transaction.transition_to(TransactionStatus::Aborted)?;
        
        // Remove from active transactions
        self.active_transactions.remove(&tx_id);
        
        Ok(())
    }

    /// Check for timed out transactions
    pub fn check_timeouts(&mut self) -> Vec<TransactionId> {
        let mut timed_out = Vec::new();
        
        for (&tx_id, transaction) in &mut self.active_transactions {
            if transaction.is_timed_out() && !transaction.status.is_terminal() {
                let _ = transaction.transition_to(TransactionStatus::TimedOut);
                timed_out.push(tx_id);
            }
        }

        // Remove timed out transactions
        for tx_id in &timed_out {
            self.active_transactions.remove(tx_id);
        }

        timed_out
    }

    /// Get active transaction count
    pub fn active_transaction_count(&self) -> usize {
        self.active_transactions.len()
    }

    /// Get all active transaction IDs
    pub fn get_active_transaction_ids(&self) -> Vec<TransactionId> {
        self.active_transactions.keys().copied().collect()
    }
}

impl Default for TransactionCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_creation() {
        let tx = Transaction::new_default();
        assert_eq!(tx.status, TransactionStatus::Active);
        assert_eq!(tx.isolation_level, IsolationLevel::RepeatableRead);
        assert!(tx.operations.is_empty());
        assert!(tx.shard_participants.is_empty());
    }

    #[test]
    fn test_transaction_operations() {
        let mut tx = Transaction::new_default();
        let entity_id = EntityId::new();
        let shard_id = ShardId::new(1);
        
        let operation = TransactionOperation::Insert { entity_id, shard_id };
        tx.add_operation(operation).unwrap();
        
        assert_eq!(tx.operations.len(), 1);
        assert!(tx.shard_participants.contains(&shard_id));
    }

    #[test]
    fn test_transaction_status_transitions() {
        let mut tx = Transaction::new_default();
        
        // Valid transition
        tx.transition_to(TransactionStatus::Preparing).unwrap();
        assert_eq!(tx.status, TransactionStatus::Preparing);
        
        // Invalid transition
        let result = tx.transition_to(TransactionStatus::Active);
        assert!(result.is_err());
    }

    #[test]
    fn test_transaction_coordinator() {
        let mut coordinator = TransactionCoordinator::new();
        
        // Begin transaction
        let tx_id = coordinator.begin_transaction(IsolationLevel::ReadCommitted);
        assert_eq!(coordinator.active_transaction_count(), 1);
        
        // Add operation
        let operation = TransactionOperation::Insert {
            entity_id: EntityId::new(),
            shard_id: ShardId::new(1),
        };
        coordinator.add_operation(tx_id, operation).unwrap();
        
        // Commit transaction
        coordinator.prepare_transaction(tx_id).unwrap();
        coordinator.commit_transaction(tx_id).unwrap();
        assert_eq!(coordinator.active_transaction_count(), 0);
    }

    #[test]
    fn test_transaction_timeout() {
        let tx = Transaction::new(IsolationLevel::ReadCommitted)
            .with_timeout(std::time::Duration::from_millis(1));
        
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(tx.is_timed_out());
    }

    #[test]
    fn test_distributed_transaction() {
        let mut tx = Transaction::new_default();
        
        let op1 = TransactionOperation::Insert {
            entity_id: EntityId::new(),
            shard_id: ShardId::new(1),
        };
        let op2 = TransactionOperation::Update {
            entity_id: EntityId::new(),
            old_version: MVCCVersion::new(),
            new_version: MVCCVersion::from_u64(2),
            shard_id: ShardId::new(2),
        };
        
        tx.add_operation(op1).unwrap();
        tx.add_operation(op2).unwrap();
        
        assert!(tx.is_distributed());
        assert_eq!(tx.shard_participants.len(), 2);
    }
}