# Code Organization

This document describes the organization and structure of the Phenix-DB codebase, explaining how mathematical principles are mapped to code modules and how components interact.

## Overview

Phenix-DB is organized around seven core mathematical modules, each implementing specific mathematical principles from historical mathematicians. The codebase follows a layered architecture where each layer builds upon the mathematical foundations of the previous layers.

---

## Directory Structure

```
Phenix-DB/
├── src/
│   ├── lib.rs                          # Main library entry point
│   ├── bin/                            # Binary executables
│   │   ├── server.rs                  # Phenix-DB server
│   │   └── cli.rs                     # CLI tools
│   ├── core/                           # Core functionality
│   │   ├── mod.rs
│   │   ├── entity.rs                  # Unified Entity (vector + metadata + edges)
│   │   ├── vector.rs                  # Vector operations
│   │   ├── metadata.rs                # JSONB metadata
│   │   ├── edges.rs                   # Probabilistic edges
│   │   ├── transaction.rs             # ACID transactions
│   │   ├── mvcc.rs                    # Multi-version concurrency control
│   │   ├── query.rs                   # Query structures
│   │   ├── error.rs                   # Error types
│   │   └── traits.rs                  # Shared abstractions
│   ├── mathematical/                   # Mathematical foundation modules
│   │   ├── mod.rs
│   │   ├── polynomial.rs              # Al-Karaji polynomial operations
│   │   ├── geometry.rs                # Khayyam/Tusi non-Euclidean geometry
│   │   ├── probability.rs             # Kolmogorov probability theory
│   │   ├── optimization.rs            # Bellman/Kantorovich optimization
│   │   ├── compression.rs             # Ramanujan/Gauss compression
│   │   ├── entropy.rs                 # Shannon/Ibn al-Haytham entropy
│   │   └── learning.rs                # Valiant PAC learning
│   ├── memory/                         # Memory substrate components
│   │   ├── mod.rs
│   │   ├── rpi.rs                     # Recursive Polynomial Index
│   │   ├── pgm.rs                     # Probabilistic Graph Memory
│   │   ├── bellman_optimizer.rs      # Bellman path optimizer
│   │   ├── kce.rs                     # Kolmogorov Compression Engine
│   │   ├── vnr.rs                     # Von Neumann Redundancy Fabric
│   │   ├── entropy_monitor.rs        # Entropy monitoring
│   │   └── cognitive_cache.rs        # Adaptive learning cache
│   ├── storage/                        # Hierarchical storage layer
│   │   ├── mod.rs
│   │   ├── wal.rs                     # Write-ahead log
│   │   ├── hot_tier.rs                # RAM/NVMe hot tier
│   │   ├── warm_tier.rs               # NVMe/SSD warm tier
│   │   ├── cold_tier.rs               # Object storage cold tier
│   │   ├── tiering.rs                 # Adaptive tier management
│   │   └── persistence.rs             # Durable storage primitives
│   ├── index/                          # Indexing and search
│   │   ├── mod.rs
│   │   ├── polynomial_tree.rs         # Recursive polynomial tree
│   │   ├── probabilistic_graph.rs     # Erdős-style graph index
│   │   ├── vector_transform.rs        # Cayley/Khayyam transformations
│   │   ├── simd.rs                    # SIMD optimizations
│   │   └── gpu.rs                     # GPU acceleration
│   ├── distributed/                    # Distributed consciousness
│   │   ├── mod.rs
│   │   ├── consciousness.rs           # Distributed coordinator
│   │   ├── node.rs                    # Memory node
│   │   ├── consensus.rs               # Entropy-driven consensus
│   │   ├── replication.rs             # Probabilistic replication
│   │   ├── router.rs                  # Intelligent routing
│   │   └── rebalancer.rs              # Dynamic rebalancing
│   ├── concurrency/                    # Lock-free concurrency
│   │   ├── mod.rs
│   │   ├── lockfree.rs                # Lock-free data structures
│   │   ├── atomic_ops.rs              # Atomic operations
│   │   ├── mvcc_engine.rs             # MVCC snapshot isolation
│   │   └── transaction_coordinator.rs # Transaction coordination
│   ├── learning/                       # Adaptive learning
│   │   ├── mod.rs
│   │   ├── access_predictor.rs        # Kolmogorov access prediction
│   │   ├── pattern_learner.rs         # Pattern recognition
│   │   ├── semantic_drift.rs          # Semantic drift detection
│   │   └── feedback_loop.rs           # Ibn al-Haytham feedback
│   ├── security/                       # Security and encryption
│   │   ├── mod.rs
│   │   ├── encryption.rs              # AES-GCM, ChaCha20-Poly1305
│   │   ├── homomorphic.rs             # Homomorphic encryption
│   │   ├── zero_knowledge.rs          # Zero-knowledge proofs
│   │   ├── auth.rs                    # Authentication and RBAC
│   │   ├── kms.rs                     # Key Management Service
│   │   └── audit.rs                   # Audit logging
│   ├── api/                            # API layer
│   │   ├── mod.rs
│   │   ├── grpc.rs                    # gRPC interface
│   │   ├── rest.rs                    # REST API
│   │   ├── protocol.rs                # Protocol definitions
│   │   └── cognitive_query.rs         # Cognitive query language
│   └── observability/                  # Monitoring and tracing
│       ├── mod.rs
│       ├── metrics.rs                 # Prometheus metrics
│       ├── tracing.rs                 # OpenTelemetry tracing
│       ├── logging.rs                 # Structured logging
│       └── mathematical_monitors.rs   # Mathematical health monitors
├── tests/
│   ├── integration/                    # Integration tests
│   ├── mathematical/                   # Mathematical correctness tests
│   ├── performance/                    # Performance benchmarks
│   ├── cognitive/                      # Cognitive behavior tests
│   └── security/                       # Security tests
├── benches/                            # Performance benchmarks
├── docs/                               # Documentation
│   ├── reasoning/                      # Mathematical reasoning docs
│   ├── architecture/                   # Architecture documentation
│   ├── api/                            # API documentation
│   └── development/                    # Development guides
└── k8s/                                # Kubernetes deployment configs
```

---

## Module Mapping: Mathematics to Code

### Module 1: Vector Transformation Engine
**Location:** `src/index/vector_transform.rs`, `src/mathematical/geometry.rs`

**Mathematicians:** Khayyam, Tusi, Cayley, Euler

**Key Components:**
- `SphericalManifold`: Non-Euclidean geometry transformations
- `CayleyTransform`: Matrix algebra for stable transformations
- `EulerianNeighborhood`: Graph-based vector traversal

**See:** [docs/reasoning/01-vector-transformation-engine.md](../reasoning/01-vector-transformation-engine.md)

---

### Module 2: Indexing & Retrieval Core
**Location:** `src/index/`, `src/memory/rpi.rs`

**Mathematicians:** Al-Khwarizmi, Al-Karaji, Gauss, Erdős, Knuth

**Key Components:**
- `AlgorithmicSearchPipeline`: Deterministic search execution
- `RecursivePolynomialIndex`: O(log n) hierarchical indexing
- `GaussianShardRouter`: Modular arithmetic for shard placement
- `ErdosGraphIndex`: Probabilistic graph-based ANN

**See:** [docs/reasoning/02-indexing-retrieval-core.md](../reasoning/02-indexing-retrieval-core.md)

---

### Module 3: Hierarchical Memory System
**Location:** `src/storage/`, `src/memory/bellman_optimizer.rs`

**Mathematicians:** Al-Samawal, von Neumann, Bellman, Kantorovich

**Key Components:**
- `RecursiveMemoryHierarchy`: Three-tier recursive structure
- `VonNeumannMemorySystem`: Hardware-aligned tiers
- `BellmanMemoryOptimizer`: Dynamic programming for paths
- `KantorovichResourceAllocator`: Linear optimization

**See:** [docs/reasoning/03-hierarchical-memory-system.md](../reasoning/03-hierarchical-memory-system.md)

---

### Module 4: Compression & Storage Efficiency
**Location:** `src/memory/kce.rs`, `src/mathematical/compression.rs`

**Mathematicians:** Al-Biruni, Ramanujan, Gauss

**Key Components:**
- `PrecisionPreservingCompressor`: Al-Biruni calibration
- `RamanujanSeriesCompressor`: Infinite series encoding
- `GaussianQuantizer`: Optimal quantization

**See:** [docs/reasoning/04-compression-storage-efficiency.md](../reasoning/04-compression-storage-efficiency.md)

---

### Module 5: Adaptive Learning & Optimization
**Location:** `src/learning/`, `src/mathematical/learning.rs`

**Mathematicians:** Ibn al-Haytham, Kolmogorov, De Moivre, Valiant

**Key Components:**
- `ExperimentalFeedbackSystem`: Closed-loop optimization
- `KolmogorovPredictor`: Probabilistic access prediction
- `PACLearner`: Valiant's learning framework

**See:** [docs/reasoning/05-adaptive-learning-optimization.md](../reasoning/05-adaptive-learning-optimization.md)

---

### Module 6: Retrieval & Path Optimization
**Location:** `src/memory/bellman_optimizer.rs`, `src/index/`

**Mathematicians:** Euler, Bellman, Knuth

**Key Components:**
- `EulerianPathFinder`: Minimal traversal routes
- `BellmanShortestPath`: Guaranteed minimum latency
- `KnuthBalancedTree`: Depth-optimal structures

---

### Module 7: Self-Organizing Memory Intelligence
**Location:** `src/learning/`, `src/distributed/consciousness.rs`

**Mathematicians:** Ibn Sina, von Neumann, Kolmogorov, Valiant

**Key Components:**
- `CognitiveOrganizer`: Semantic abstraction
- `SelfReplicatingNode`: Autonomous reorganization
- `EntropyPruner`: Redundancy elimination
- `LearnabilityBounds`: Stable learning thresholds

---

## Coding Patterns

### Mathematical Correctness

Every mathematical operation must include:

1. **Citation**: Reference to mathematician and principle
2. **Formula**: Mathematical formula in comments
3. **Proof**: Link to correctness proof or reference
4. **Test**: Unit test validating mathematical properties

Example:
```rust
/// Evaluate polynomial using Al-Karaji's recursive method
/// Formula: P(x) = a₀ + x(a₁ + x(a₂ + ... + x(aₙ)))
/// Reference: Al-Karaji, "Al-Fakhri" (1010 CE)
/// Complexity: O(n) for degree n polynomial
pub fn evaluate_polynomial_alkaraji(coefficients: &[f64], x: f64) -> f64 {
    // Horner's method (equivalent to Al-Karaji's recursion)
    coefficients.iter()
        .rev()
        .fold(0.0, |acc, &coeff| acc * x + coeff)
}

#[test]
fn test_polynomial_evaluation_correctness() {
    // Test: P(x) = 2x² + 3x + 1 at x = 2
    // Expected: 2(4) + 3(2) + 1 = 15
    let coeffs = vec![1.0, 3.0, 2.0];  // [a₀, a₁, a₂]
    let result = evaluate_polynomial_alkaraji(&coeffs, 2.0);
    assert!((result - 15.0).abs() < 1e-10);
}
```

### Probability Operations

All probability operations must maintain invariants:

```rust
/// Update edge probability using Kolmogorov axioms
/// Invariant: 0 ≤ P(E) ≤ 1
/// Invariant: Σ P(Eᵢ) = 1 for partition {Eᵢ}
pub fn update_probability(&mut self, co_accessed: bool) {
    if co_accessed {
        self.probability = (self.probability + 0.1).min(1.0);
    } else {
        self.probability = (self.probability - 0.01).max(0.0);
    }
    
    // Verify Kolmogorov axioms
    debug_assert!(self.probability >= 0.0 && self.probability <= 1.0);
}

#[test]
fn test_probability_distribution_sums_to_one() {
    let edges = create_test_edges();
    let sum: f32 = edges.iter().map(|e| e.probability).sum();
    
    // Kolmogorov axiom: probabilities sum to 1
    assert!((sum - 1.0).abs() < 0.001);
}
```

### Recursive Structures

Follow Al-Samawal's recursive patterns:

```rust
/// Recursive memory tier following Al-Samawal's principles
pub struct MemoryTier {
    storage: Box<dyn Storage>,
    next_tier: Option<Box<MemoryTier>>,
}

impl MemoryTier {
    /// Recursive retrieval: check current, then recurse
    pub fn get_recursive(&self, key: &EntityId) -> Option<Entity> {
        // Base case
        if let Some(entity) = self.storage.get(key) {
            return Some(entity);
        }
        
        // Recursive case
        self.next_tier.as_ref()?.get_recursive(key)
    }
}
```

---

## Component Interaction

### Data Flow

```
Query → Vector Transform → Index Lookup → Memory Tier → Compression → Storage
  ↓           ↓                ↓              ↓             ↓           ↓
Cayley    Khayyam/Tusi    Al-Karaji/Erdős  Al-Samawal   Ramanujan   Gauss
```

### Learning Loop

```
Access Pattern → Kolmogorov Prediction → Bellman Optimization → Tier Placement
       ↓                                                              ↓
Ibn al-Haytham Feedback ← Performance Measurement ← Actual Access ←─┘
```

---

## Testing Strategy

### Mathematical Correctness Tests
**Location:** `tests/mathematical/`

Test mathematical properties, not just functionality:

```rust
#[test]
fn test_bellman_optimality() {
    let optimizer = BellmanOptimizer::new();
    let path = optimizer.compute_optimal_path(start, goal);
    
    // Verify Bellman optimality: no shorter path exists
    let cost = compute_path_cost(&path);
    let any_path_cost = compute_any_valid_path_cost(start, goal);
    assert!(cost <= any_path_cost);
}
```

### Integration Tests
**Location:** `tests/integration/`

Test component interactions:

```rust
#[test]
fn test_end_to_end_cognitive_query() {
    let db = PhenixDB::new();
    
    // Insert with RPI
    db.insert(entity);
    
    // Query with learning
    let results = db.cognitive_query(query);
    
    // Verify all modules worked together
    assert!(results.used_rpi);
    assert!(results.used_bellman_optimization);
    assert!(results.learned_from_access);
}
```

---

## Performance Profiling

### Metrics to Track

Each module exposes Prometheus metrics:

```rust
// RPI metrics
phenix_db_polynomial_evaluation_seconds
phenix_db_rpi_tree_depth
phenix_db_rpi_leaf_size

// Compression metrics
phenix_db_compression_ratio
phenix_db_decompression_seconds
phenix_db_compression_error

// Learning metrics
phenix_db_prediction_accuracy
phenix_db_cache_hit_rate
phenix_db_learning_convergence_rate
```

---

## Contributing Guidelines

When adding new code:

1. **Identify the mathematical principle** it implements
2. **Document the mathematician** and their contribution
3. **Include the formula** in comments
4. **Write correctness tests** that validate mathematical properties
5. **Add performance benchmarks** if applicable
6. **Update this document** with the new component

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

---

## References

- [Mathematical Reasoning Documentation](../reasoning/README.md)
- [Contributing Guide](../../CONTRIBUTING.md)
- [Development TODO](../TODO.md)

---

**"Code is mathematics made executable."**

Every line of Phenix-DB code traces back to centuries of proven mathematical principles.
