//! # Multi-Version Concurrency Control (MVCC)
//!
//! This module implements MVCC support for Phenix DB, providing snapshot isolation
//! and version management for concurrent transactions across unified entities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

use crate::core::{
    entity::EntityId,
    transaction::TransactionId,
    error::{PhenixDBError, Result},
};

/// MVCC version number for entity versioning
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MVCCVersion(pub u64);

impl MVCCVersion {
    /// Create a new version starting from 1
    pub fn new() -> Self {
        Self(1)
    }

    /// Create version from u64
    pub fn from_u64(version: u64) -> Self {
        Self(version)
    }

    /// Get the version number
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Increment version and return new version
    pub fn increment(&self) -> Self {
        Self(self.0 + 1)
    }

    /// Check if this version is newer than another
    pub fn is_newer_than(&self, other: &MVCCVersion) -> bool {
        self.0 > other.0
    }
}

impl Default for MVCCVersion {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for MVCCVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Timestamp type for MVCC operations
pub type Timestamp = DateTime<Utc>;

/// Version information for entity history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Entity ID this version belongs to
    pub entity_id: EntityId,
    
    /// Version number
    pub version: MVCCVersion,
    
    /// Transaction that created this version
    pub created_by_tx: TransactionId,
    
    /// Timestamp when version was created
    pub created_at: Timestamp,
    
    /// Transaction that deleted this version (if any)
    pub deleted_by_tx: Option<TransactionId>,
    
    /// Timestamp when version was deleted (if any)
    pub deleted_at: Option<Timestamp>,
    
    /// Whether this version is visible to new transactions
    pub is_visible: bool,
}

impl VersionInfo {
    /// Create new version info
    pub fn new(entity_id: EntityId, version: MVCCVersion, tx_id: TransactionId) -> Self {
        Self {
            entity_id,
            version,
            created_by_tx: tx_id,
            created_at: Utc::now(),
            deleted_by_tx: None,
            deleted_at: None,
            is_visible: true,
        }
    }

    /// Mark version as deleted
    pub fn mark_deleted(&mut self, tx_id: TransactionId) {
        self.deleted_by_tx = Some(tx_id);
        self.deleted_at = Some(Utc::now());
        self.is_visible = false;
    }

    /// Check if version is active (not deleted)
    pub fn is_active(&self) -> bool {
        self.deleted_at.is_none()
    }

    /// Check if version was created by transaction
    pub fn created_by(&self, tx_id: TransactionId) -> bool {
        self.created_by_tx == tx_id
    }

    /// Check if version was deleted by transaction
    pub fn deleted_by(&self, tx_id: TransactionId) -> bool {
        self.deleted_by_tx == Some(tx_id)
    }
}

/// Snapshot for consistent reads across transactions
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Snapshot timestamp
    pub timestamp: Timestamp,
    
    /// Transaction ID that created this snapshot
    pub transaction_id: TransactionId,
    
    /// Visible versions for each entity at snapshot time
    pub visible_versions: HashMap<EntityId, MVCCVersion>,
    
    /// Active transactions at snapshot time
    pub active_transactions: Vec<TransactionId>,
}

impl Snapshot {
    /// Create new snapshot
    pub fn new(tx_id: TransactionId) -> Self {
        Self {
            timestamp: Utc::now(),
            transaction_id: tx_id,
            visible_versions: HashMap::new(),
            active_transactions: Vec::new(),
        }
    }

    /// Check if entity version is visible in this snapshot
    pub fn is_version_visible(&self, entity_id: EntityId, version: MVCCVersion) -> bool {
        if let Some(&visible_version) = self.visible_versions.get(&entity_id) {
            version <= visible_version
        } else {
            false
        }
    }

    /// Add visible version for entity
    pub fn add_visible_version(&mut self, entity_id: EntityId, version: MVCCVersion) {
        self.visible_versions.insert(entity_id, version);
    }

    /// Get visible version for entity
    pub fn get_visible_version(&self, entity_id: EntityId) -> Option<MVCCVersion> {
        self.visible_versions.get(&entity_id).copied()
    }

    /// Check if transaction was active at snapshot time
    pub fn was_transaction_active(&self, tx_id: TransactionId) -> bool {
        self.active_transactions.contains(&tx_id)
    }

    /// Add active transaction
    pub fn add_active_transaction(&mut self, tx_id: TransactionId) {
        if !self.active_transactions.contains(&tx_id) {
            self.active_transactions.push(tx_id);
        }
    }
}

/// MVCC engine for managing entity versions and snapshots
#[derive(Debug)]
pub struct MVCCEngine {
    /// Version history for all entities
    version_history: HashMap<EntityId, Vec<VersionInfo>>,
    
    /// Current version for each entity
    current_versions: HashMap<EntityId, MVCCVersion>,
    
    /// Active snapshots
    active_snapshots: HashMap<TransactionId, Snapshot>,
}

impl MVCCEngine {
    /// Create new MVCC engine
    pub fn new() -> Self {
        Self {
            version_history: HashMap::new(),
            current_versions: HashMap::new(),
            active_snapshots: HashMap::new(),
        }
    }

    /// Create new version for entity
    pub fn create_version(&mut self, entity_id: EntityId, tx_id: TransactionId) -> Result<MVCCVersion> {
        let new_version = if let Some(&current_version) = self.current_versions.get(&entity_id) {
            current_version.increment()
        } else {
            MVCCVersion::new()
        };

        let version_info = VersionInfo::new(entity_id, new_version, tx_id);
        
        // Add to version history
        self.version_history
            .entry(entity_id)
            .or_insert_with(Vec::new)
            .push(version_info);

        // Update current version
        self.current_versions.insert(entity_id, new_version);

        Ok(new_version)
    }

    /// Mark entity version as deleted
    pub fn delete_version(&mut self, entity_id: EntityId, version: MVCCVersion, tx_id: TransactionId) -> Result<()> {
        if let Some(versions) = self.version_history.get_mut(&entity_id) {
            if let Some(version_info) = versions.iter_mut().find(|v| v.version == version) {
                version_info.mark_deleted(tx_id);
                return Ok(());
            }
        }

        Err(PhenixDBError::EntityNotFound {
            id: format!("{}@{}", entity_id, version),
        })
    }

    /// Create snapshot for transaction
    pub fn create_snapshot(&mut self, tx_id: TransactionId) -> Snapshot {
        let mut snapshot = Snapshot::new(tx_id);

        // Add visible versions for all entities
        for (&entity_id, &current_version) in &self.current_versions {
            snapshot.add_visible_version(entity_id, current_version);
        }

        // Add active transactions
        for &active_tx_id in self.active_snapshots.keys() {
            if active_tx_id != tx_id {
                snapshot.add_active_transaction(active_tx_id);
            }
        }

        self.active_snapshots.insert(tx_id, snapshot.clone());
        snapshot
    }

    /// Get snapshot for transaction
    pub fn get_snapshot(&self, tx_id: TransactionId) -> Option<&Snapshot> {
        self.active_snapshots.get(&tx_id)
    }

    /// Remove snapshot for completed transaction
    pub fn remove_snapshot(&mut self, tx_id: TransactionId) {
        self.active_snapshots.remove(&tx_id);
    }

    /// Get current version for entity
    pub fn get_current_version(&self, entity_id: EntityId) -> Option<MVCCVersion> {
        self.current_versions.get(&entity_id).copied()
    }

    /// Get version history for entity
    pub fn get_version_history(&self, entity_id: EntityId) -> Option<&Vec<VersionInfo>> {
        self.version_history.get(&entity_id)
    }

    /// Check if version is visible to transaction
    pub fn is_version_visible(&self, entity_id: EntityId, version: MVCCVersion, tx_id: TransactionId) -> bool {
        if let Some(snapshot) = self.get_snapshot(tx_id) {
            snapshot.is_version_visible(entity_id, version)
        } else {
            // If no snapshot, check against current version
            if let Some(&current_version) = self.current_versions.get(&entity_id) {
                version <= current_version
            } else {
                false
            }
        }
    }

    /// Garbage collect old versions
    pub fn garbage_collect(&mut self, before_timestamp: Timestamp) -> usize {
        let mut collected_count = 0;

        for versions in self.version_history.values_mut() {
            let initial_len = versions.len();
            
            // Keep only versions that are:
            // 1. Still active (not deleted)
            // 2. Deleted after the GC timestamp
            // 3. Referenced by active snapshots
            versions.retain(|version_info| {
                if version_info.is_active() {
                    return true;
                }

                if let Some(deleted_at) = version_info.deleted_at {
                    if deleted_at > before_timestamp {
                        return true;
                    }
                }

                // Check if any active snapshot references this version
                for snapshot in self.active_snapshots.values() {
                    if let Some(visible_version) = snapshot.get_visible_version(version_info.entity_id) {
                        if version_info.version <= visible_version {
                            return true;
                        }
                    }
                }

                false
            });

            collected_count += initial_len - versions.len();
        }

        collected_count
    }

    /// Get statistics about MVCC state
    pub fn get_statistics(&self) -> MVCCStatistics {
        let total_versions = self.version_history.values().map(|v| v.len()).sum();
        let active_versions = self.version_history
            .values()
            .flat_map(|versions| versions.iter())
            .filter(|v| v.is_active())
            .count();

        MVCCStatistics {
            total_entities: self.current_versions.len(),
            total_versions,
            active_versions,
            deleted_versions: total_versions - active_versions,
            active_snapshots: self.active_snapshots.len(),
        }
    }
}

impl Default for MVCCEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about MVCC engine state
#[derive(Debug, Clone)]
pub struct MVCCStatistics {
    pub total_entities: usize,
    pub total_versions: usize,
    pub active_versions: usize,
    pub deleted_versions: usize,
    pub active_snapshots: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transaction::TransactionId;

    #[test]
    fn test_mvcc_version() {
        let v1 = MVCCVersion::new();
        let v2 = v1.increment();
        
        assert_eq!(v1.as_u64(), 1);
        assert_eq!(v2.as_u64(), 2);
        assert!(v2.is_newer_than(&v1));
    }

    #[test]
    fn test_version_info() {
        let entity_id = EntityId::new();
        let tx_id = TransactionId::new();
        let version = MVCCVersion::new();
        
        let mut version_info = VersionInfo::new(entity_id, version, tx_id);
        assert!(version_info.is_active());
        assert!(version_info.created_by(tx_id));
        
        let delete_tx_id = TransactionId::new();
        version_info.mark_deleted(delete_tx_id);
        assert!(!version_info.is_active());
        assert!(version_info.deleted_by(delete_tx_id));
    }

    #[test]
    fn test_snapshot() {
        let tx_id = TransactionId::new();
        let mut snapshot = Snapshot::new(tx_id);
        
        let entity_id = EntityId::new();
        let version = MVCCVersion::from_u64(5);
        
        snapshot.add_visible_version(entity_id, version);
        assert!(snapshot.is_version_visible(entity_id, MVCCVersion::from_u64(3)));
        assert!(snapshot.is_version_visible(entity_id, version));
        assert!(!snapshot.is_version_visible(entity_id, MVCCVersion::from_u64(6)));
    }

    #[test]
    fn test_mvcc_engine() {
        let mut engine = MVCCEngine::new();
        let entity_id = EntityId::new();
        let tx_id = TransactionId::new();
        
        // Create first version
        let v1 = engine.create_version(entity_id, tx_id).unwrap();
        assert_eq!(v1, MVCCVersion::new());
        assert_eq!(engine.get_current_version(entity_id), Some(v1));
        
        // Create second version
        let v2 = engine.create_version(entity_id, tx_id).unwrap();
        assert_eq!(v2, v1.increment());
        assert_eq!(engine.get_current_version(entity_id), Some(v2));
        
        // Create snapshot
        let snapshot = engine.create_snapshot(tx_id);
        assert!(snapshot.is_version_visible(entity_id, v2));
    }

    #[test]
    fn test_garbage_collection() {
        let mut engine = MVCCEngine::new();
        let entity_id = EntityId::new();
        let tx_id = TransactionId::new();
        
        // Create and delete some versions
        let v1 = engine.create_version(entity_id, tx_id).unwrap();
        let v2 = engine.create_version(entity_id, tx_id).unwrap();
        
        engine.delete_version(entity_id, v1, tx_id).unwrap();
        
        // GC should collect the deleted version
        let gc_timestamp = Utc::now();
        let collected = engine.garbage_collect(gc_timestamp);
        assert_eq!(collected, 1);
    }
}