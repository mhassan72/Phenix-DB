//! Memory substrate components
//!
//! Core memory substrate implementations:
//! - RPI: Recursive Polynomial Index
//! - PGM: Probabilistic Graph Memory
//! - Bellman Optimizer: Dynamic path optimization
//! - KCE: Kolmogorov Compression Engine
//! - VNR: Von Neumann Redundancy Fabric
//! - Entropy Monitor: Information density optimization

pub mod bellman_optimizer;
pub mod cognitive_cache;
pub mod entropy_monitor;
pub mod kce;
pub mod pgm;
pub mod rpi;
pub mod vnr;

// Re-export main types
pub use bellman_optimizer::BellmanOptimizer;
pub use kce::KolmogorovCompressionEngine;
pub use pgm::ProbabilisticGraphMemory;
pub use rpi::RecursivePolynomialIndex;
pub use vnr::VonNeumannRedundancyFabric;
pub use entropy_monitor::EntropyMonitor;
