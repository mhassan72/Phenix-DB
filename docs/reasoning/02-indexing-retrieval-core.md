# Module 2: Indexing & Retrieval Core

## Overview

The Indexing & Retrieval Core is the computational heart of Phenix-DB, responsible for organizing trillions of vectors into searchable structures and executing queries with sub-millisecond latency. This module combines algorithmic rigor with probabilistic efficiency to achieve both deterministic correctness and practical scalability.

---

## Mathematical Foundations

### 🔹 Al-Khwarizmi
**Contribution:** Algorithmic Logic & Structured Computation

#### Historical Context
- **Muhammad ibn Musa al-Khwarizmi (780-850)**: Persian mathematician whose name gave us the word "algorithm"
- Wrote *Al-Kitab al-Mukhtasar fi Hisab al-Jabr wal-Muqabala* (The Compendious Book on Calculation by Completion and Balancing)
- Established systematic approaches to solving equations

#### Application in Phenix-DB

**Problem:** Vector search must be deterministic, traceable, and reproducible across distributed nodes.

**Solution:** Structured computation pipelines with ordered execution guarantees.

**Implementation:**
```rust
/// Al-Khwarizmi inspired deterministic search pipeline
pub struct AlgorithmicSearchPipeline {
    stages: Vec<SearchStage>,
    execution_log: Vec<ExecutionStep>,
}

impl AlgorithmicSearchPipeline {
    /// Execute search with complete traceability
    /// Every step is logged and reproducible
    pub fn execute(&mut self, query: &Query) -> SearchResult {
        let mut context = SearchContext::new(query);
        
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            let step_start = Instant::now();
            
            // Execute stage with deterministic ordering
            let stage_result = stage.execute(&mut context);
            
            // Log execution for reproducibility
            self.execution_log.push(ExecutionStep {
                stage_index: stage_idx,
                stage_name: stage.name(),
                duration: step_start.elapsed(),
                input_state: context.snapshot(),
                output_state: stage_result.snapshot(),
                deterministic_hash: stage_result.compute_hash(),
            });
            
            context.apply_result(stage_result);
        }
        
        context.finalize()
    }
    
    /// Verify execution correctness using Al-Khwarizmi's systematic approach
    pub fn verify_execution(&self, expected_hash: u64) -> bool {
        let computed_hash = self.execution_log.iter()
            .fold(0u64, |acc, step| acc ^ step.deterministic_hash);
        computed_hash == expected_hash
    }
}

/// Search stages follow Al-Khwarizmi's structured problem-solving
pub enum SearchStage {
    /// 1. Completion: Fill in missing query parameters
    QueryCompletion(CompletionStrategy),
    
    /// 2. Balancing: Normalize and balance query vectors
    QueryBalancing(BalancingStrategy),
    
    /// 3. Reduction: Reduce search space using filters
    SpaceReduction(ReductionStrategy),
    
    /// 4. Traversal: Navigate index structure
    IndexTraversal(TraversalStrategy),
    
    /// 5. Ranking: Order results by relevance
    ResultRanking(RankingStrategy),
}
```

**Benefits:**
- **Determinism**: Same query always produces same results
- **Traceability**: Every step is logged and auditable
- **Reproducibility**: Execution can be replayed for debugging
- **Correctness**: Systematic approach reduces errors

**Example Use Case:**
When debugging why a particular vector wasn't returned, the execution log shows exactly which stage filtered it out and why, following Al-Khwarizmi's principle of showing your work.

---

### 🔹 Thābit ibn Qurra & Al-Karaji
**Contribution:** Polynomial & Recursive Equations

#### Historical Context
- **Thābit ibn Qurra (826-901)**: Iraqi mathematician who advanced algebra and number theory
- **Al-Karaji (953-1029)**: Persian mathematician who developed polynomial algebra and recursive methods
- Pioneered recursive formulas and polynomial manipulation

#### Application in Phenix-DB

**Problem:** Need hierarchical index structures that support O(log n) search with predictable traversal.

**Solution:** Recursive Polynomial Index (RPI) using polynomial embeddings and recursive descent.

**Implementation:**
```rust
/// Recursive Polynomial Index inspired by Al-Karaji
pub struct RecursivePolynomialIndex {
    root: PolynomialNode,
    degree: usize,
    depth: usize,
}

/// Each node stores data as polynomial coefficients
pub struct PolynomialNode {
    /// Polynomial representation: P(x) = Σ aᵢ * xⁱ
    coefficients: Vec<f64>,
    
    /// Recursive children (Al-Karaji's recursive structure)
    children: Vec<Box<PolynomialNode>>,
    
    /// Entities stored at this node
    entities: Vec<EntityId>,
    
    /// Polynomial bounds for pruning
    min_value: f64,
    max_value: f64,
}

impl RecursivePolynomialIndex {
    /// Insert using recursive polynomial evaluation
    pub fn insert(&mut self, vector: &[f64], entity_id: EntityId) {
        // Convert vector to polynomial coefficients (Al-Karaji method)
        let poly = self.vectorize_to_polynomial(vector);
        
        // Recursively descend to appropriate leaf
        self.recursive_insert(&mut self.root, poly, entity_id, 0);
    }
    
    /// Recursive insertion following Al-Karaji's recursive patterns
    fn recursive_insert(
        &mut self,
        node: &mut PolynomialNode,
        poly: Polynomial,
        entity_id: EntityId,
        depth: usize,
    ) {
        if depth >= self.depth {
            // Leaf node: store entity
            node.entities.push(entity_id);
            return;
        }
        
        // Evaluate polynomial at this level
        let eval_point = self.compute_eval_point(depth);
        let value = poly.evaluate(eval_point);
        
        // Select child using Thābit ibn Qurra's number theory
        let child_idx = self.select_child_index(value, node.children.len());
        
        // Recurse to child
        self.recursive_insert(
            &mut node.children[child_idx],
            poly,
            entity_id,
            depth + 1,
        );
    }
    
    /// Search using recursive polynomial matching
    pub fn search(&self, query: &[f64], k: usize) -> Vec<EntityId> {
        let query_poly = self.vectorize_to_polynomial(query);
        let mut results = Vec::new();
        
        self.recursive_search(&self.root, &query_poly, k, &mut results, 0);
        
        results
    }
    
    /// Recursive search with polynomial pruning
    fn recursive_search(
        &self,
        node: &PolynomialNode,
        query_poly: &Polynomial,
        k: usize,
        results: &mut Vec<EntityId>,
        depth: usize,
    ) {
        if depth >= self.depth {
            // Leaf node: add entities to results
            results.extend(&node.entities);
            return;
        }
        
        // Evaluate query polynomial at this level
        let eval_point = self.compute_eval_point(depth);
        let query_value = query_poly.evaluate(eval_point);
        
        // Prune children using polynomial bounds (Al-Karaji optimization)
        for (idx, child) in node.children.iter().enumerate() {
            if self.should_explore_child(query_value, child) {
                self.recursive_search(child, query_poly, k, results, depth + 1);
            }
        }
    }
    
    /// Convert vector to polynomial using Al-Karaji's methods
    fn vectorize_to_polynomial(&self, vector: &[f64]) -> Polynomial {
        // Use recursive formula: P_n(x) = a_n + x * P_{n-1}(x)
        let mut coefficients = Vec::with_capacity(self.degree);
        
        for i in 0..self.degree {
            let coeff = vector.iter()
                .enumerate()
                .map(|(j, &v)| v * (j as f64).powi(i as i32))
                .sum::<f64>() / vector.len() as f64;
            coefficients.push(coeff);
        }
        
        Polynomial::new(coefficients)
    }
}
```

**Benefits:**
- **Logarithmic Search**: O(log n) traversal through recursive structure
- **Polynomial Pruning**: Bounds checking eliminates entire subtrees
- **Predictable Performance**: Depth-bounded recursion guarantees
- **Semantic Clustering**: Similar vectors have similar polynomial representations

**Example Use Case:**
A 1 billion vector index with depth 10 requires only 10 polynomial evaluations to reach any leaf, compared to millions of distance calculations in flat indexes.

---

### 🔹 Carl Friedrich Gauss
**Contribution:** Modular Arithmetic & Gaussian Elimination

#### Historical Context
- **Carl Friedrich Gauss (1777-1855)**: German mathematician, "Prince of Mathematicians"
- Developed modular arithmetic (*Disquisitiones Arithmeticae*, 1801)
- Invented Gaussian elimination for solving linear systems

#### Application in Phenix-DB

**Problem:** Need fast, deterministic data placement across distributed shards with minimal collisions.

**Solution:** Modular hashing and linear reduction for optimal shard assignment.

**Implementation:**
```rust
/// Gaussian modular hashing for shard placement
pub struct GaussianShardRouter {
    shard_count: usize,
    modulus: u64,
    hash_matrix: Matrix,
}

impl GaussianShardRouter {
    /// Compute shard assignment using modular arithmetic
    /// Gauss's theorem: Every integer has unique representation mod p
    pub fn route_to_shard(&self, entity_id: EntityId) -> ShardId {
        // Apply Gaussian modular reduction
        let hash = self.compute_hash(entity_id);
        let shard = (hash % self.modulus as u64) as usize % self.shard_count;
        
        ShardId(shard)
    }
    
    /// Compute hash using Gaussian elimination principles
    fn compute_hash(&self, entity_id: EntityId) -> u64 {
        // Convert entity ID to vector
        let id_vector = entity_id.to_vector();
        
        // Apply hash matrix (Gaussian elimination for dimension reduction)
        let reduced = self.hash_matrix.multiply_vector(&id_vector);
        
        // Combine using modular arithmetic
        reduced.iter()
            .enumerate()
            .map(|(i, &v)| {
                let scaled = (v * 1000.0) as u64;
                scaled.wrapping_mul(self.prime_at(i))
            })
            .fold(0u64, |acc, x| acc.wrapping_add(x))
    }
    
    /// Rebalance shards using Gaussian elimination
    /// Solves: Ax = b where A is load matrix, x is migration plan
    pub fn rebalance_shards(&self, current_loads: &[usize]) -> MigrationPlan {
        let target_load = current_loads.iter().sum::<usize>() / self.shard_count;
        
        // Build linear system: each shard should reach target load
        let mut system = LinearSystem::new(self.shard_count);
        for (i, &load) in current_loads.iter().enumerate() {
            system.add_equation(i, load as f64, target_load as f64);
        }
        
        // Solve using Gaussian elimination
        let solution = system.gaussian_eliminate();
        
        // Convert solution to migration plan
        self.solution_to_migration_plan(solution)
    }
}

/// Gaussian elimination for linear systems
pub struct LinearSystem {
    matrix: Matrix,
    constants: Vec<f64>,
}

impl LinearSystem {
    /// Solve Ax = b using Gaussian elimination
    pub fn gaussian_eliminate(&self) -> Vec<f64> {
        let mut augmented = self.matrix.augment(&self.constants);
        
        // Forward elimination
        for pivot in 0..augmented.rows() {
            // Find pivot
            let max_row = self.find_max_pivot(pivot, &augmented);
            augmented.swap_rows(pivot, max_row);
            
            // Eliminate below pivot
            for row in (pivot + 1)..augmented.rows() {
                let factor = augmented[(row, pivot)] / augmented[(pivot, pivot)];
                for col in pivot..augmented.cols() {
                    let value = augmented[(row, col)] - factor * augmented[(pivot, col)];
                    augmented[(row, col)] = value;
                }
            }
        }
        
        // Back substitution
        self.back_substitute(&augmented)
    }
}
```

**Benefits:**
- **Uniform Distribution**: Modular arithmetic ensures even shard distribution
- **Deterministic Placement**: Same entity always routes to same shard
- **Efficient Rebalancing**: Gaussian elimination finds optimal migration
- **Collision Minimization**: Prime moduli reduce hash collisions

**Example Use Case:**
When adding new shards, Gaussian elimination computes the minimal set of entities to migrate to achieve perfect load balance.

---

### 🔹 Paul Erdős
**Contribution:** Random Graph Theory

#### Historical Context
- **Paul Erdős (1913-1996)**: Hungarian mathematician, most prolific in history
- Co-developed random graph theory with Alfréd Rényi
- Proved existence of graphs with specific properties using probabilistic methods

#### Application in Phenix-DB

**Problem:** Need scalable graph-based indexes that work with billions of vectors.

**Solution:** Probabilistic graph structures (HNSW, small-world networks) with Erdős-Rényi properties.

**Implementation:**
```rust
/// Hierarchical Navigable Small World (HNSW) graph
/// Based on Erdős-Rényi random graph theory
pub struct ErdosGraphIndex {
    layers: Vec<GraphLayer>,
    max_connections: usize,
    level_multiplier: f64,
}

pub struct GraphLayer {
    nodes: HashMap<EntityId, GraphNode>,
    edge_probability: f64,  // Erdős-Rényi parameter
}

pub struct GraphNode {
    entity_id: EntityId,
    vector: Vec<f64>,
    neighbors: Vec<(EntityId, f64)>,  // (neighbor_id, distance)
}

impl ErdosGraphIndex {
    /// Insert node using Erdős random graph principles
    pub fn insert(&mut self, entity_id: EntityId, vector: Vec<f64>) {
        // Determine layer using exponential distribution (Erdős-Rényi)
        let layer = self.select_layer_erdos();
        
        // Insert into all layers up to selected layer
        for l in 0..=layer {
            self.insert_at_layer(entity_id, &vector, l);
        }
    }
    
    /// Select layer using Erdős exponential distribution
    fn select_layer_erdos(&self) -> usize {
        let uniform: f64 = rand::random();
        let level = (-uniform.ln() * self.level_multiplier) as usize;
        level.min(self.layers.len() - 1)
    }
    
    /// Insert at specific layer with probabilistic connections
    fn insert_at_layer(&mut self, entity_id: EntityId, vector: &[f64], layer: usize) {
        let layer_graph = &mut self.layers[layer];
        
        // Find entry point (highest layer node)
        let entry_point = self.find_entry_point(layer);
        
        // Search for nearest neighbors using greedy traversal
        let candidates = self.search_layer(entry_point, vector, self.max_connections, layer);
        
        // Connect to neighbors with probability based on Erdős-Rényi model
        let mut neighbors = Vec::new();
        for (candidate_id, distance) in candidates {
            // Erdős-Rényi: connect with probability p
            let connection_prob = self.compute_connection_probability(distance, layer);
            if rand::random::<f64>() < connection_prob {
                neighbors.push((candidate_id, distance));
            }
        }
        
        // Prune connections to maintain small-world property
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbors.truncate(self.max_connections);
        
        // Insert node
        layer_graph.nodes.insert(entity_id, GraphNode {
            entity_id,
            vector: vector.to_vec(),
            neighbors,
        });
        
        // Update reverse connections (maintain symmetry)
        self.update_reverse_connections(entity_id, &neighbors, layer);
    }
    
    /// Compute connection probability using Erdős-Rényi formula
    /// p = c * ln(n) / n for connectivity threshold
    fn compute_connection_probability(&self, distance: f64, layer: usize) -> f64 {
        let n = self.layers[layer].nodes.len() as f64;
        let c = 2.0;  // Connectivity constant
        
        let base_prob = c * n.ln() / n;
        
        // Adjust by distance (closer = higher probability)
        let distance_factor = (-distance * distance).exp();
        
        (base_prob * distance_factor).min(1.0)
    }
    
    /// Search using Erdős small-world property
    /// Expected path length: O(log n)
    pub fn search(&self, query: &[f64], k: usize) -> Vec<EntityId> {
        let mut current_layer = self.layers.len() - 1;
        let mut entry_point = self.find_entry_point(current_layer);
        
        // Traverse layers from top to bottom
        while current_layer > 0 {
            let nearest = self.search_layer(entry_point, query, 1, current_layer);
            entry_point = nearest[0].0;
            current_layer -= 1;
        }
        
        // Final search at layer 0
        let results = self.search_layer(entry_point, query, k, 0);
        results.into_iter().map(|(id, _)| id).collect()
    }
}
```

**Benefits:**
- **Logarithmic Search**: O(log n) expected path length (Erdős small-world property)
- **Scalability**: Handles billions of vectors with constant-time insertions
- **Robustness**: Probabilistic connections provide multiple paths
- **Efficiency**: Sparse connectivity (O(n log n) edges) vs dense graphs (O(n²))

**Example Use Case:**
Searching 1 billion vectors requires only ~30 hops on average (log₂(10⁹) ≈ 30), compared to millions of comparisons in brute force.

---

### 🔹 Donald Knuth
**Contribution:** Algorithmic Optimization & Data Structures

#### Historical Context
- **Donald Knuth (1938-present)**: American computer scientist, author of *The Art of Computer Programming*
- Developed analysis of algorithms as a discipline
- Created TeX typesetting system and literate programming

#### Application in Phenix-DB

**Problem:** Need optimal data structures and algorithms with proven performance bounds.

**Solution:** Apply Knuth's principles of algorithmic analysis and optimization.

**Implementation:**
```rust
/// Knuth-optimized balanced tree for indexing
/// Based on B-tree principles from TAOCP Volume 3
pub struct KnuthBalancedTree {
    root: TreeNode,
    order: usize,  // Knuth's B-tree order
    height: usize,
}

pub struct TreeNode {
    keys: Vec<f64>,
    children: Vec<Box<TreeNode>>,
    entities: Vec<EntityId>,
    is_leaf: bool,
}

impl KnuthBalancedTree {
    /// Insert with Knuth's B-tree balancing
    /// Maintains: ⌈m/2⌉ ≤ keys ≤ m-1 (Knuth's invariant)
    pub fn insert(&mut self, key: f64, entity_id: EntityId) {
        if self.root.is_full(self.order) {
            // Split root (Knuth's root splitting algorithm)
            let new_root = self.split_root();
            self.root = new_root;
            self.height += 1;
        }
        
        self.insert_non_full(&mut self.root, key, entity_id);
    }
    
    /// Knuth's algorithm for insertion into non-full node
    fn insert_non_full(&mut self, node: &mut TreeNode, key: f64, entity_id: EntityId) {
        if node.is_leaf {
            // Insert into sorted position (Knuth's binary insertion)
            let pos = node.keys.binary_search_by(|k| {
                k.partial_cmp(&key).unwrap()
            }).unwrap_or_else(|e| e);
            
            node.keys.insert(pos, key);
            node.entities.insert(pos, entity_id);
        } else {
            // Find child to descend into
            let child_idx = self.find_child_index(node, key);
            
            if node.children[child_idx].is_full(self.order) {
                // Split child before descending (Knuth's preemptive splitting)
                self.split_child(node, child_idx);
                
                // Recompute child index after split
                let child_idx = self.find_child_index(node, key);
            }
            
            self.insert_non_full(&mut node.children[child_idx], key, entity_id);
        }
    }
    
    /// Search with Knuth's optimal comparison strategy
    /// Minimizes comparisons using binary search
    pub fn search(&self, key: f64) -> Option<EntityId> {
        self.search_node(&self.root, key)
    }
    
    fn search_node(&self, node: &TreeNode, key: f64) -> Option<EntityId> {
        // Knuth's binary search within node
        match node.keys.binary_search_by(|k| k.partial_cmp(&key).unwrap()) {
            Ok(idx) => Some(node.entities[idx]),
            Err(idx) => {
                if node.is_leaf {
                    None
                } else {
                    self.search_node(&node.children[idx], key)
                }
            }
        }
    }
    
    /// Analyze performance using Knuth's methods
    pub fn analyze_performance(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            height: self.height,
            // Knuth's theorem: height ≤ log_m(n)
            theoretical_max_height: (self.count_entities() as f64)
                .log(self.order as f64)
                .ceil() as usize,
            average_node_fill: self.compute_average_fill(),
            // Knuth's optimal fill: ~69% for random insertions
            optimal_fill: 0.69,
        }
    }
}
```

**Benefits:**
- **Proven Bounds**: All operations have mathematically proven complexity
- **Optimal Performance**: Algorithms are asymptotically optimal
- **Code Quality**: Knuth's literate programming principles ensure clarity
- **Profiling**: Built-in performance analysis

**Example Use Case:**
B-tree guarantees O(log n) search with minimal disk I/O, proven optimal by Knuth's analysis.

---

## Integration in Phenix-DB Architecture

### Component Interaction

```
┌─────────────────────────────────────────────────────────┐
│         Indexing & Retrieval Core                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Query Input                                             │
│         ↓                                                │
│  [Al-Khwarizmi Pipeline] → Structured Execution         │
│         ↓                                                │
│  [Gaussian Shard Router] → Distributed Placement        │
│         ↓                                                │
│  [RPI / Erdős Graph] → Index Traversal                  │
│         ↓                                                │
│  [Knuth Balanced Tree] → Final Ranking                  │
│         ↓                                                │
│  Search Results                                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Performance Characteristics

| Operation | Complexity | Mathematician |
|-----------|-----------|---------------|
| Insert | O(log n) | Al-Karaji, Knuth |
| Search | O(log n) | Erdős, Knuth |
| Shard Route | O(1) | Gauss |
| Rebalance | O(n²) | Gauss |
| Pipeline Execute | O(k log n) | Al-Khwarizmi |

---

## Testing & Validation

```rust
#[test]
fn test_algorithmic_determinism() {
    let mut pipeline = AlgorithmicSearchPipeline::new();
    let query = create_test_query();
    
    let result1 = pipeline.execute(&query);
    let result2 = pipeline.execute(&query);
    
    // Al-Khwarizmi principle: same input → same output
    assert_eq!(result1.hash(), result2.hash());
}

#[test]
fn test_polynomial_index_logarithmic() {
    let mut index = RecursivePolynomialIndex::new(10, 1000000);
    
    // Insert 1M vectors
    for i in 0..1000000 {
        index.insert(&random_vector(128), EntityId(i));
    }
    
    // Search should take ~log(1M) = 20 steps
    let steps = index.search_with_step_count(&random_vector(128), 10);
    assert!(steps <= 25);  // Allow some overhead
}

#[test]
fn test_erdos_small_world_property() {
    let mut graph = ErdosGraphIndex::new();
    
    // Insert 10K nodes
    for i in 0..10000 {
        graph.insert(EntityId(i), random_vector(128));
    }
    
    // Average path length should be O(log n) ≈ 13
    let avg_path_length = graph.compute_average_path_length();
    assert!(avg_path_length < 20.0);
}
```

---

**Next Module**: [Hierarchical Memory System](03-hierarchical-memory-system.md)
