//! Multi-Version Concurrency Control (MVCC)

use super::{Entity, EntityId, Timestamp, TransactionId};
use serde::{Deserialize, Serialize};

/// MVCC version number
pub type MVCCVersion = u64;

/// Entity version for MVCC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityVersion {
    /// Entity ID
    pub entity_id: EntityId,
    
    /// Version number
    pub version: MVCCVersion,
    
    /// Entity data
    pub data: Entity,
    
    /// Transaction that created this version
    pub created_by: TransactionId,
    
    /// Transactions that can see this version
    pub visible_to: Vec<TransactionId>,
    
    /// Creation timestamp
    pub timestamp: Timestamp,
}

impl EntityVersion {
    /// Create a new entity version
    pub fn new(
        entity_id: EntityId,
        version: MVCCVersion,
        data: Entity,
        created_by: TransactionId,
    ) -> Self {
        Self {
            entity_id,
            version,
            data,
            created_by,
            visible_to: Vec::new(),
            timestamp: Timestamp::now(),
        }
    }

    /// Check if this version is visible to a transaction
    pub fn is_visible_to(&self, tx_id: TransactionId) -> bool {
        self.visible_to.contains(&tx_id) || self.created_by == tx_id
    }
}
