//! Transaction management

use super::{Timestamp, TransactionId};
use serde::{Deserialize, Serialize};

/// Transaction state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Transaction ID
    pub id: TransactionId,
    
    /// Transaction state
    pub state: TransactionState,
    
    /// Start timestamp
    pub start_time: Timestamp,
    
    /// Commit timestamp (if committed)
    pub commit_time: Option<Timestamp>,
}

/// Transaction state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionState {
    /// Transaction is active
    Active,
    /// Transaction is preparing to commit
    Preparing,
    /// Transaction is committed
    Committed,
    /// Transaction is aborted
    Aborted,
}

impl Transaction {
    /// Create a new transaction
    pub fn new(id: TransactionId) -> Self {
        Self {
            id,
            state: TransactionState::Active,
            start_time: Timestamp::now(),
            commit_time: None,
        }
    }

    /// Check if transaction is active
    pub fn is_active(&self) -> bool {
        self.state == TransactionState::Active
    }

    /// Check if transaction is committed
    pub fn is_committed(&self) -> bool {
        self.state == TransactionState::Committed
    }
}
