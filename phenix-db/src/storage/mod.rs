//! Hierarchical storage layer
//!
//! Three-tier memory architecture:
//! - Hot tier: RAM/NVMe (<1ms)
//! - Warm tier: NVMe/SSD (1-10ms)
//! - Cold tier: Object storage (10-100ms)

pub mod cold_tier;
pub mod hot_tier;
pub mod persistence;
pub mod tiering;
pub mod wal;
pub mod warm_tier;
