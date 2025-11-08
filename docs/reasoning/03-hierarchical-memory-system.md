# Module 3: Hierarchical Memory System

## Overview

The Hierarchical Memory System implements a three-tier memory architecture (hot/warm/cold) that automatically optimizes data placement based on access patterns. This module combines recursive computation, dynamic programming, and linear optimization to achieve sub-millisecond access for frequently used data while maintaining cost-effective storage for rarely accessed vectors.

---

## Mathematical Foundations

### 🔹 Al-Samawal
**Contribution:** Recursive Computation

#### Historical Context
- **Al-Samawal al-Maghribi (1130-1180)**: Iraqi mathematician and astronomer
- Developed recursive methods for polynomial division and root extraction
- Wrote *Al-Bahir fi'l-jabr* (The Brilliant in Algebra)
- Extended Al-Karaji's work on recursive algorithms

#### Application in Phenix-DB

**Problem:** Need memory hierarchy where each tier can recursively reference lower tiers without circular dependencies.

**Solution:** Recursive memory nodes where each tier maintains references to the next tier, forming a directed acyclic graph.

**Implementation:**
```rust
/// Al-Samawal inspired recursive memory hierarchy
pub struct RecursiveMemoryHierarchy {
    hot_tier: MemoryTier,
    warm_tier: MemoryTier,
    cold_tier: MemoryTier,
}

pub struct MemoryTier {
    name: TierName,
    storage: Box<dyn Storage>,
    next_tier: Option<Box<MemoryTier>>,
    access_latency: Duration,
    capacity: usize,
    current_size: AtomicUsize,
}

impl MemoryTier {
    /// Recursive retrieval following Al-Samawal's pattern
    /// If not found in current tier, recurse to next tier
    pub fn get_recursive(&self, key: &EntityId) -> Option<Entity> {
        // Base case: check current tier
        if let Some(entity) = self.storage.get(key) {
            return Some(entity);
        }
        
        // Recursive case: check next tier
        if let Some(ref next) = self.next_tier {
            if let Some(entity) = next.get_recursive(key) {
                // Promote to current tier (Al-Samawal's optimization)
                self.promote_from_lower(key, &entity);
                return Some(entity);
            }
        }
        
        None
    }
    
    /// Recursive insertion with overflow handling
    pub fn insert_recursive(&mut self, key: EntityId, entity: Entity) -> Result<(), MemoryError> {
        // Check if current tier has capacity
        if self.has_capacity() {
            self.storage.insert(key, entity)?;
            self.current_size.fetch_add(1, Ordering::SeqCst);
            return Ok(());
        }
        
        // Current tier full: evict and recurse
        if let Some(ref mut next) = self.next_tier {
            // Evict least recently used to next tier (recursive)
            let (evict_key, evict_entity) = self.select_eviction_candidate();
            next.insert_recursive(evict_key, evict_entity)?;
            
            // Now insert new entity in current tier
            self.storage.insert(key, entity)?;
            Ok(())
        } else {
            // No next tier: this is cold tier, must have space
            Err(MemoryError::OutOfSpace)
        }
    }
    
    /// Recursive tier statistics computation
    pub fn compute_stats_recursive(&self) -> TierStats {
        let mut stats = TierStats {
            tier_name: self.name,
            size: self.current_size.load(Ordering::SeqCst),
            capacity: self.capacity,
            access_latency: self.access_latency,
            hit_rate: self.compute_hit_rate(),
            subtier_stats: None,
        };
        
        // Recurse to next tier
        if let Some(ref next) = self.next_tier {
            stats.subtier_stats = Some(Box::new(next.compute_stats_recursive()));
        }
        
        stats
    }
}

/// Recursive memory access pattern
impl RecursiveMemoryHierarchy {
    /// Get entity with automatic tier promotion
    pub fn get(&self, key: &EntityId) -> Option<Entity> {
        // Start recursive search from hot tier
        self.hot_tier.get_recursive(key)
    }
    
    /// Recursive tier reorganization (Al-Samawal's self-optimization)
    pub fn reorganize_recursive(&mut self) {
        self.reorganize_tier(&mut self.hot_tier);
    }
    
    fn reorganize_tier(&mut self, tier: &mut MemoryTier) {
        // Reorganize current tier based on access patterns
        tier.reorganize_internal();
        
        // Recursively reorganize next tier
        if let Some(ref mut next) = tier.next_tier {
            self.reorganize_tier(next);
        }
    }
}
```

**Benefits:**
- **Automatic Promotion**: Frequently accessed data moves up tiers
- **Overflow Handling**: Recursive eviction to lower tiers
- **Unified Interface**: Same API regardless of tier depth
- **Self-Optimization**: Recursive reorganization improves over time

**Example Use Case:**
When accessing a cold-tier entity, it's automatically promoted to hot tier, and the least-used hot-tier entity is recursively pushed down through warm to cold.

---

### 🔹 John von Neumann
**Contribution:** Memory Hierarchy & Computational Architecture

#### Historical Context
- **John von Neumann (1903-1957)**: Hungarian-American mathematician and physicist
- Designed the von Neumann architecture (stored-program computer)
- Established principles of memory hierarchy (registers, cache, RAM, storage)
- Pioneered self-replicating automata theory

#### Application in Phenix-DB

**Problem:** Need physical memory tier alignment with CPU cache, DRAM, and SSD for optimal performance.

**Solution:** Map logical tiers to physical hardware following von Neumann architecture principles.

**Implementation:**
```rust
/// Von Neumann architecture-aligned memory tiers
pub struct VonNeumannMemorySystem {
    /// Hot tier: CPU cache + RAM (L1/L2/L3 + DRAM)
    hot_tier: HotTierStorage,
    
    /// Warm tier: NVMe SSD (persistent, fast)
    warm_tier: WarmTierStorage,
    
    /// Cold tier: Object storage (S3, distributed)
    cold_tier: ColdTierStorage,
    
    /// Von Neumann's instruction pipeline for parallel access
    access_pipeline: InstructionPipeline,
}

/// Hot tier: RAM-based with CPU cache optimization
pub struct HotTierStorage {
    /// Cache-line aligned storage (64 bytes)
    cache_aligned_data: Vec<CacheAlignedEntity>,
    
    /// SIMD-optimized vector operations
    simd_engine: SIMDEngine,
    
    /// Prefetch hints for CPU
    prefetch_queue: VecDeque<EntityId>,
}

impl HotTierStorage {
    /// Store with cache-line alignment (von Neumann optimization)
    pub fn store(&mut self, entity: Entity) {
        // Align to 64-byte cache lines
        let aligned = self.align_to_cache_line(entity);
        
        // Use SIMD for parallel writes
        self.simd_engine.parallel_write(&aligned);
        
        // Prefetch related entities
        self.prefetch_related(&aligned);
    }
    
    /// Retrieve with CPU cache optimization
    pub fn retrieve(&self, key: &EntityId) -> Option<Entity> {
        // Check if in CPU cache (via prefetch)
        if self.is_prefetched(key) {
            return self.fast_retrieve(key);
        }
        
        // Trigger prefetch for next access
        self.prefetch_queue.push_back(*key);
        
        self.standard_retrieve(key)
    }
    
    /// Von Neumann's instruction-level parallelism
    pub fn parallel_retrieve(&self, keys: &[EntityId]) -> Vec<Option<Entity>> {
        // Use SIMD to retrieve multiple entities in parallel
        self.simd_engine.parallel_read(keys)
    }
}

/// Warm tier: NVMe SSD with direct memory access
pub struct WarmTierStorage {
    /// Memory-mapped file for zero-copy access
    mmap: MemoryMappedFile,
    
    /// Direct I/O for bypassing OS cache
    direct_io: DirectIOEngine,
    
    /// Async I/O for non-blocking operations
    async_io: AsyncIOEngine,
}

impl WarmTierStorage {
    /// Store with direct memory access (von Neumann DMA)
    pub async fn store(&mut self, entity: Entity) -> Result<(), IOError> {
        // Use DMA for zero-copy write to NVMe
        self.direct_io.dma_write(&entity).await
    }
    
    /// Retrieve with memory mapping
    pub fn retrieve(&self, key: &EntityId) -> Option<Entity> {
        // Zero-copy read via memory mapping
        self.mmap.read_at_offset(self.compute_offset(key))
    }
}

/// Cold tier: Object storage with compression
pub struct ColdTierStorage {
    /// S3-compatible object storage
    object_store: ObjectStore,
    
    /// Compression engine
    compressor: CompressionEngine,
    
    /// Batch operations for efficiency
    batch_queue: BatchQueue,
}

impl ColdTierStorage {
    /// Store with compression and batching
    pub async fn store(&mut self, entity: Entity) -> Result<(), IOError> {
        // Compress entity
        let compressed = self.compressor.compress(&entity);
        
        // Add to batch queue
        self.batch_queue.add(compressed);
        
        // Flush batch if full
        if self.batch_queue.is_full() {
            self.flush_batch().await?;
        }
        
        Ok(())
    }
}

/// Von Neumann instruction pipeline for memory operations
pub struct InstructionPipeline {
    stages: Vec<PipelineStage>,
}

impl InstructionPipeline {
    /// Execute memory operation with pipelining
    pub fn execute(&mut self, operation: MemoryOperation) -> PipelinedResult {
        // Stage 1: Fetch (determine tier)
        let tier = self.stages[0].determine_tier(&operation);
        
        // Stage 2: Decode (prepare access)
        let access_plan = self.stages[1].prepare_access(tier, &operation);
        
        // Stage 3: Execute (perform I/O)
        let result = self.stages[2].execute_io(&access_plan);
        
        // Stage 4: Write-back (update caches)
        self.stages[3].update_caches(&result);
        
        result
    }
}
```

**Benefits:**
- **Hardware Alignment**: Tiers map directly to physical hardware
- **Cache Optimization**: Cache-line alignment reduces CPU stalls
- **Parallel Access**: SIMD and instruction pipelining for throughput
- **Zero-Copy**: Memory mapping and DMA eliminate data copying

**Example Use Case:**
Hot-tier entities are stored in cache-aligned 64-byte blocks, allowing CPU to fetch entire cache lines in one operation, achieving <1ms access times.

---

### 🔹 Richard Bellman
**Contribution:** Dynamic Programming

#### Historical Context
- **Richard Bellman (1920-1984)**: American mathematician
- Invented dynamic programming in the 1950s
- Developed Bellman equation for optimal control
- Applied to shortest path, resource allocation, and optimization problems

#### Application in Phenix-DB

**Problem:** Need to find optimal retrieval paths through memory tiers with minimal cost.

**Solution:** Use Bellman's dynamic programming to compute optimal access sequences.

**Implementation:**
```rust
/// Bellman optimizer for memory access paths
pub struct BellmanMemoryOptimizer {
    /// Cost matrix: cost[tier][operation]
    cost_matrix: Matrix,
    
    /// Value function: V(state) = optimal cost from state
    value_function: HashMap<MemoryState, f64>,
    
    /// Policy: π(state) = optimal action from state
    policy: HashMap<MemoryState, MemoryAction>,
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct MemoryState {
    current_tier: TierName,
    entity_location: Option<TierName>,
    access_history: Vec<EntityId>,
}

pub enum MemoryAction {
    AccessHot,
    AccessWarm,
    AccessCold,
    PromoteToHot,
    DemoteToWarm,
    DemoteToCold,
}

impl BellmanMemoryOptimizer {
    /// Compute optimal access path using Bellman equation
    /// V(s) = min_a [cost(s,a) + γ * V(s')]
    pub fn compute_optimal_path(&mut self, start: MemoryState, goal: EntityId) -> Vec<MemoryAction> {
        // Initialize value function
        self.initialize_value_function();
        
        // Value iteration (Bellman's algorithm)
        loop {
            let mut delta = 0.0;
            
            for state in self.all_states() {
                let old_value = self.value_function[&state];
                
                // Bellman update
                let new_value = self.bellman_update(&state);
                
                self.value_function.insert(state.clone(), new_value);
                delta = delta.max((old_value - new_value).abs());
            }
            
            // Converged?
            if delta < 1e-6 {
                break;
            }
        }
        
        // Extract optimal policy
        self.extract_policy(start, goal)
    }
    
    /// Bellman equation update
    fn bellman_update(&self, state: &MemoryState) -> f64 {
        let mut min_cost = f64::INFINITY;
        
        for action in self.possible_actions(state) {
            // Immediate cost
            let immediate_cost = self.cost(state, &action);
            
            // Next state
            let next_state = self.transition(state, &action);
            
            // Future cost (discounted)
            let future_cost = 0.9 * self.value_function.get(&next_state).unwrap_or(&0.0);
            
            // Total cost
            let total_cost = immediate_cost + future_cost;
            
            if total_cost < min_cost {
                min_cost = total_cost;
            }
        }
        
        min_cost
    }
    
    /// Cost function: latency + energy + monetary cost
    fn cost(&self, state: &MemoryState, action: &MemoryAction) -> f64 {
        match action {
            MemoryAction::AccessHot => 0.001,      // 1ms
            MemoryAction::AccessWarm => 0.005,     // 5ms
            MemoryAction::AccessCold => 0.050,     // 50ms
            MemoryAction::PromoteToHot => 0.010,   // 10ms + energy
            MemoryAction::DemoteToWarm => 0.002,   // 2ms
            MemoryAction::DemoteToCold => 0.001,   // 1ms (async)
        }
    }
    
    /// Trigger restructuring when cost exceeds threshold
    /// Bellman principle: restructure if cost > 1.5x optimal
    pub fn should_restructure(&self, current_cost: f64, optimal_cost: f64) -> bool {
        current_cost > 1.5 * optimal_cost
    }
    
    /// Restructure memory tiers using Bellman optimization
    pub fn restructure(&mut self, access_patterns: &[AccessPattern]) {
        // Recompute optimal policy based on new access patterns
        self.update_cost_matrix(access_patterns);
        
        // Recompute value function
        self.compute_optimal_path(
            MemoryState::initial(),
            EntityId::default(),
        );
        
        // Apply new policy
        self.apply_policy();
    }
}

/// Bellman-optimized tier manager
pub struct BellmanTierManager {
    optimizer: BellmanMemoryOptimizer,
    access_log: Vec<AccessRecord>,
}

impl BellmanTierManager {
    /// Decide tier placement using Bellman optimization
    pub fn decide_placement(&mut self, entity_id: EntityId) -> TierName {
        // Analyze access patterns
        let access_freq = self.compute_access_frequency(entity_id);
        let access_recency = self.compute_access_recency(entity_id);
        
        // Compute expected future accesses (Bellman prediction)
        let expected_accesses = self.predict_future_accesses(entity_id);
        
        // Compute cost for each tier
        let hot_cost = self.optimizer.compute_tier_cost(TierName::Hot, expected_accesses);
        let warm_cost = self.optimizer.compute_tier_cost(TierName::Warm, expected_accesses);
        let cold_cost = self.optimizer.compute_tier_cost(TierName::Cold, expected_accesses);
        
        // Select tier with minimum cost (Bellman optimality)
        if hot_cost < warm_cost && hot_cost < cold_cost {
            TierName::Hot
        } else if warm_cost < cold_cost {
            TierName::Warm
        } else {
            TierName::Cold
        }
    }
}
```

**Benefits:**
- **Optimal Paths**: Bellman equation guarantees minimum-cost access sequences
- **Adaptive**: Recomputes optimal policy as access patterns change
- **Predictive**: Anticipates future accesses for proactive tier placement
- **Cost-Aware**: Balances latency, energy, and monetary costs

**Example Use Case:**
When an entity is accessed frequently, Bellman optimizer computes that promoting to hot tier will save 100ms over next 1000 accesses, justifying the 10ms promotion cost.

---

### 🔹 Leonid Kantorovich
**Contribution:** Linear Optimization

#### Historical Context
- **Leonid Kantorovich (1912-1986)**: Soviet mathematician and economist
- Developed linear programming independently of Dantzig
- Nobel Prize in Economics (1975) for optimal resource allocation
- Applied mathematics to economic planning and transportation problems

#### Application in Phenix-DB

**Problem:** Need to balance storage space and retrieval speed under limited compute/memory budgets.

**Solution:** Linear optimization to allocate resources optimally across tiers.

**Implementation:**
```rust
/// Kantorovich linear optimizer for resource allocation
pub struct KantorovichResourceAllocator {
    /// Decision variables: x[i] = entities in tier i
    tier_allocations: Vec<f64>,
    
    /// Constraints: capacity, budget, latency
    constraints: Vec<LinearConstraint>,
    
    /// Objective: minimize cost or maximize throughput
    objective: ObjectiveFunction,
}

pub struct LinearConstraint {
    coefficients: Vec<f64>,
    bound: f64,
    constraint_type: ConstraintType,
}

pub enum ConstraintType {
    LessThanOrEqual,
    GreaterThanOrEqual,
    Equal,
}

impl KantorovichResourceAllocator {
    /// Solve linear program using Kantorovich's method
    /// Minimize: c^T x
    /// Subject to: Ax ≤ b, x ≥ 0
    pub fn optimize(&mut self) -> AllocationPlan {
        // Formulate linear program
        let lp = self.formulate_linear_program();
        
        // Solve using simplex method (Kantorovich-Dantzig)
        let solution = self.solve_simplex(lp);
        
        // Convert solution to allocation plan
        self.solution_to_plan(solution)
    }
    
    /// Formulate resource allocation as linear program
    fn formulate_linear_program(&self) -> LinearProgram {
        // Variables: x_hot, x_warm, x_cold (entities per tier)
        let variables = vec!["x_hot", "x_warm", "x_cold"];
        
        // Objective: minimize total cost
        // cost = c_hot * x_hot + c_warm * x_warm + c_cold * x_cold
        let objective = ObjectiveFunction {
            coefficients: vec![
                self.cost_per_entity(TierName::Hot),
                self.cost_per_entity(TierName::Warm),
                self.cost_per_entity(TierName::Cold),
            ],
            sense: OptimizationSense::Minimize,
        };
        
        // Constraint 1: Total entities
        // x_hot + x_warm + x_cold = total_entities
        let total_constraint = LinearConstraint {
            coefficients: vec![1.0, 1.0, 1.0],
            bound: self.total_entities as f64,
            constraint_type: ConstraintType::Equal,
        };
        
        // Constraint 2: Hot tier capacity
        // x_hot ≤ hot_capacity
        let hot_capacity_constraint = LinearConstraint {
            coefficients: vec![1.0, 0.0, 0.0],
            bound: self.hot_capacity as f64,
            constraint_type: ConstraintType::LessThanOrEqual,
        };
        
        // Constraint 3: Latency requirement
        // latency_hot * x_hot + latency_warm * x_warm + latency_cold * x_cold ≤ max_latency
        let latency_constraint = LinearConstraint {
            coefficients: vec![0.001, 0.005, 0.050],
            bound: self.max_average_latency,
            constraint_type: ConstraintType::LessThanOrEqual,
        };
        
        // Constraint 4: Budget
        // cost_hot * x_hot + cost_warm * x_warm + cost_cold * x_cold ≤ budget
        let budget_constraint = LinearConstraint {
            coefficients: vec![10.0, 1.0, 0.1],  // $/entity/month
            bound: self.monthly_budget,
            constraint_type: ConstraintType::LessThanOrEqual,
        };
        
        LinearProgram {
            variables,
            objective,
            constraints: vec![
                total_constraint,
                hot_capacity_constraint,
                latency_constraint,
                budget_constraint,
            ],
        }
    }
    
    /// Solve using simplex method
    fn solve_simplex(&self, lp: LinearProgram) -> Solution {
        // Initialize simplex tableau
        let mut tableau = self.initialize_tableau(&lp);
        
        // Kantorovich's simplex iterations
        loop {
            // Find entering variable (most negative reduced cost)
            let entering = self.find_entering_variable(&tableau);
            if entering.is_none() {
                break;  // Optimal solution found
            }
            
            // Find leaving variable (minimum ratio test)
            let leaving = self.find_leaving_variable(&tableau, entering.unwrap());
            
            // Pivot
            self.pivot(&mut tableau, entering.unwrap(), leaving);
        }
        
        // Extract solution
        self.extract_solution(&tableau)
    }
    
    /// Balance storage space vs retrieval speed
    /// Kantorovich's transportation problem formulation
    pub fn balance_space_speed(&mut self, space_weight: f64, speed_weight: f64) -> AllocationPlan {
        // Multi-objective optimization
        // Minimize: w1 * space_cost + w2 * latency_cost
        
        self.objective = ObjectiveFunction {
            coefficients: vec![
                space_weight * self.space_cost(TierName::Hot) + speed_weight * self.latency_cost(TierName::Hot),
                space_weight * self.space_cost(TierName::Warm) + speed_weight * self.latency_cost(TierName::Warm),
                space_weight * self.space_cost(TierName::Cold) + speed_weight * self.latency_cost(TierName::Cold),
            ],
            sense: OptimizationSense::Minimize,
        };
        
        self.optimize()
    }
}
```

**Benefits:**
- **Optimal Allocation**: Kantorovich's method guarantees optimal resource distribution
- **Multi-Objective**: Balances competing goals (cost, latency, capacity)
- **Constraint Satisfaction**: Respects hard limits (budget, capacity)
- **Scalable**: Linear programming scales to thousands of variables

**Example Use Case:**
Given $1000/month budget and 10ms average latency requirement, Kantorovich optimizer determines optimal split: 10% hot tier, 30% warm tier, 60% cold tier.

---

## Integration in Phenix-DB Architecture

### Component Interaction

```
┌─────────────────────────────────────────────────────────┐
│         Hierarchical Memory System                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Access Request                                          │
│         ↓                                                │
│  [Al-Samawal Recursive Lookup] → Tier Traversal        │
│         ↓                                                │
│  [Von Neumann Hardware Mapping] → Physical Access       │
│         ↓                                                │
│  [Bellman Path Optimization] → Optimal Route            │
│         ↓                                                │
│  [Kantorovich Resource Allocation] → Tier Placement     │
│         ↓                                                │
│  Entity Retrieved                                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Performance Characteristics

| Tier | Latency | Capacity | Cost/GB/Month | Mathematician |
|------|---------|----------|---------------|---------------|
| Hot | <1ms | 10GB | $100 | Von Neumann |
| Warm | 1-10ms | 1TB | $10 | Von Neumann |
| Cold | 10-100ms | ∞ | $1 | Von Neumann |

| Operation | Complexity | Mathematician |
|-----------|-----------|---------------|
| Recursive Lookup | O(log t) | Al-Samawal |
| Path Optimization | O(t²) | Bellman |
| Resource Allocation | O(t³) | Kantorovich |

---

## Testing & Validation

```rust
#[test]
fn test_recursive_tier_traversal() {
    let mut hierarchy = RecursiveMemoryHierarchy::new();
    
    // Insert in cold tier
    hierarchy.cold_tier.insert(EntityId(1), create_entity());
    
    // Access should recursively find and promote
    let entity = hierarchy.get(&EntityId(1)).unwrap();
    
    // Should now be in hot tier
    assert!(hierarchy.hot_tier.contains(&EntityId(1)));
}

#[test]
fn test_bellman_optimal_path() {
    let mut optimizer = BellmanMemoryOptimizer::new();
    
    let path = optimizer.compute_optimal_path(
        MemoryState::initial(),
        EntityId(1),
    );
    
    // Verify path is optimal (minimum cost)
    let cost = optimizer.compute_path_cost(&path);
    assert!(cost <= optimizer.compute_any_path_cost() * 1.01);
}

#[test]
fn test_kantorovich_resource_allocation() {
    let mut allocator = KantorovichResourceAllocator::new();
    
    let plan = allocator.optimize();
    
    // Verify constraints satisfied
    assert!(plan.total_cost() <= allocator.monthly_budget);
    assert!(plan.average_latency() <= allocator.max_average_latency);
    assert!(plan.hot_tier_size() <= allocator.hot_capacity);
}
```

---

**Next Module**: [Compression & Storage Efficiency](04-compression-storage-efficiency.md)
