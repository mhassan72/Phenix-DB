# Contributing to Phenix DB

Thank you for your interest in contributing to Phenix DB! This document provides guidelines and information for contributors to our unified vector + document + graph database.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Security Guidelines](#security-guidelines)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful, inclusive, and constructive in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- **Rust 1.70+** with Cargo
- **Git** for version control
- **Docker** (optional, for testing deployments)
- **Kubernetes** (optional, for K8s testing)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/Rust-Vector-Database.git
   cd Rust-Vector-Database
   ```

2. **Install Dependencies**
   ```bash
   # Install Rust toolchain components
   rustup component add clippy rustfmt
   
   # Install development tools
   cargo install cargo-watch cargo-audit cargo-outdated
   ```

3. **Build and Test**
   ```bash
   # Build the project
   cargo build
   
   # Run tests
   cargo test
   
   # Run clippy for linting
   cargo clippy
   
   # Format code
   cargo fmt
   ```

4. **Set up Pre-commit Hooks** (optional but recommended)
   ```bash
   # Install pre-commit
   pip install pre-commit
   
   # Install hooks
   pre-commit install
   ```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions to Phenix DB:

- **Unified data model improvements** - Enhance Entity, vector, metadata, and edge handling
- **Query optimization** - Improve hybrid vector + metadata + graph query performance
- **Storage enhancements** - Optimize hot/cold tiering and compression algorithms
- **Security features** - Strengthen encryption, KMS integration, and RBAC
- **Shard management** - Improve distributed operations and rebalancing
- **API development** - Enhance gRPC/REST interfaces and SDK generation
- **Performance improvements** - Optimize SIMD operations and GPU acceleration
- **Documentation** - Improve docs, examples, and architectural guides
- **Tests** - Add comprehensive test coverage for unified operations
- **Bug fixes** - Fix issues across vector, document, and graph functionality

### Before You Start

1. **Check existing issues** - Look for related issues or discussions
2. **Create an issue** - For significant changes, create an issue first
3. **Discuss your approach** - Get feedback before implementing large features
4. **Check the roadmap** - Ensure your contribution aligns with project goals

### Issue Guidelines

When creating issues:

- **Use clear, descriptive titles**
- **Provide detailed descriptions** with steps to reproduce (for bugs)
- **Include system information** (OS, Rust version, etc.)
- **Add relevant labels** (bug, enhancement, documentation, etc.)
- **Reference related issues** if applicable

### Feature Requests

For new features:

- **Explain the use case** - Why is this feature needed?
- **Describe the solution** - What should the feature do?
- **Consider alternatives** - Are there other ways to solve this?
- **Assess impact** - How does this affect performance, API, etc.?

## Pull Request Process

### 1. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 2. Make Changes

- Follow the [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Guidelines

Use conventional commit messages:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```
feat(entity): add unified Entity upsert with ACID guarantees

fix(query): resolve hybrid query result aggregation bug

docs(api): update unified query examples with graph traversal

test(storage): add benchmarks for hot/cold tier promotion

perf(index): optimize SIMD operations for vector similarity

security(kms): implement envelope encryption with key rotation
```

### 4. Submit Pull Request

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create pull request** on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Description of changes made
   - Testing performed
   - Breaking changes (if any)

3. **Address review feedback** promptly and professionally

### 5. Pull Request Checklist

Before submitting:

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Tests added for new functionality
- [ ] All tests pass locally
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages follow conventional format
- [ ] Branch is up to date with main

## Coding Standards

### Rust Style Guidelines

Follow standard Rust conventions:

- **Use `rustfmt`** for consistent formatting
- **Follow Rust naming conventions**:
  - `snake_case` for functions, variables, modules
  - `PascalCase` for types, structs, enums
  - `SCREAMING_SNAKE_CASE` for constants
- **Use meaningful names** that clearly express intent
- **Prefer composition over inheritance** (traits over structs)
- **Use `Result<T, E>` for error handling**
- **Avoid `unwrap()` and `expect()` in production code**

### Code Organization for Phenix DB

- **Unified operations** - Group vector, metadata, and edge operations together
- **Entity-first design** - Structure code around the unified Entity model
- **Security boundaries** - Enforce tenant isolation and encryption at module boundaries
- **Shard awareness** - All components must handle distributed operations gracefully
- **Hot path optimization** - Separate performance-critical vector operations
- **Clear interfaces** - Well-documented traits for unified storage and querying
- **MVCC integration** - All data modifications must be version-aware

### Error Handling

```rust
// Good: Unified entity operations with proper error handling
async fn upsert_entity(entity: &Entity, tenant_id: TenantId) -> Result<EntityId, PhenixError> {
    // Validate entity components
    entity.validate()?;
    
    // Ensure tenant isolation
    self.validate_tenant_access(tenant_id)?;
    
    // Perform unified upsert with ACID guarantees
    self.storage.upsert_with_transaction(entity, tenant_id).await
}

// Bad: Separate operations without ACID guarantees
fn insert_vector_and_metadata(vector: Vector, metadata: JSONB) -> VectorId {
    // Don't split unified operations
    let vector_id = self.vector_store.insert(vector).unwrap();
    self.metadata_store.insert(vector_id, metadata).unwrap(); // Race condition!
    vector_id
}
```

### Performance Guidelines for Phenix DB

- **Unified query optimization** - Optimize across vector similarity, metadata filters, and graph traversal
- **Hot path separation** - Isolate vector operations in performance-critical modules
- **SIMD and GPU acceleration** - Leverage hardware acceleration for vector computations
- **Smart caching** - Implement intelligent hot/cold tier decisions across all data types
- **Batch operations** - Process entities in batches for optimal throughput
- **Memory layout optimization** - Consider cache efficiency for unified Entity storage
- **Zero-copy operations** - Minimize data copying in vector processing pipelines
- **Profile hybrid queries** - Measure performance across all query components

## Testing Guidelines

### Test Categories

1. **Unit Tests** - Test individual functions and components
2. **Integration Tests** - Test component interactions
3. **Performance Tests** - Validate performance requirements
4. **End-to-End Tests** - Test complete workflows

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entity_validation() {
        let entity = Entity::new()
            .with_vector(vec![3.0, 4.0])
            .with_metadata(json!({"title": "Test Doc"}))
            .with_edge("related_to", "entity_2", 0.8);
        
        assert!(entity.validate().is_ok());
        assert_eq!(entity.vector.as_ref().unwrap().magnitude(), 5.0);
    }
    
    #[tokio::test]
    async fn test_unified_upsert() {
        let database = setup_test_database().await;
        let entity = create_test_entity();
        
        let result = database.upsert_entity(entity, TenantId::test()).await;
        assert!(result.is_ok());
        
        // Verify ACID guarantees - all components should be stored together
        let retrieved = database.get_entity(result.unwrap(), Snapshot::latest()).await.unwrap();
        assert!(retrieved.vector.is_some());
        assert!(retrieved.metadata.is_some());
        assert!(!retrieved.edges.as_ref().unwrap().is_empty());
    }
    
    #[tokio::test]
    async fn test_hybrid_query() {
        let database = setup_test_database_with_entities().await;
        
        let query = UnifiedQuery::new()
            .with_vector_similarity(vec![0.1, 0.2], 5)
            .with_metadata_filter(json!({"category": "tech"}))
            .with_graph_constraint("related_to:entity_2");
        
        let results = database.hybrid_query(query).await.unwrap();
        
        // Verify results satisfy all constraints
        for result in results {
            assert!(result.similarity_score > 0.0);
            assert_eq!(result.entity.metadata.as_ref().unwrap()["category"], "tech");
        }
    }
}
```

### Test Guidelines for Phenix DB

- **Test unified operations** - Verify ACID guarantees across vector, metadata, and edges
- **Test hybrid queries** - Validate combined vector similarity, metadata filtering, and graph traversal
- **Test tenant isolation** - Ensure proper security boundaries and data separation
- **Test shard operations** - Verify distributed query routing and result aggregation
- **Test hot/cold tiering** - Validate promotion/demotion logic and compression
- **Test failure scenarios** - WAL replay, node failures, and transaction rollbacks
- **Use descriptive test names** that explain the unified operation being tested
- **Mock external dependencies** (KMS, object storage) in unit tests
- **Test performance requirements** - Sub-millisecond latency for hybrid queries

### Performance Testing

```rust
#[cfg(test)]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn benchmark_vector_search(c: &mut Criterion) {
        let index = setup_large_index();
        let query = test_query_vector();
        
        c.bench_function("vector_search_1m", |b| {
            b.iter(|| {
                black_box(index.search(black_box(&query), 10))
            })
        });
    }
    
    criterion_group!(benches, benchmark_vector_search);
    criterion_main!(benches);
}
```

## Documentation

### Code Documentation

- **Document public APIs** with comprehensive rustdoc comments
- **Include examples** in documentation
- **Document error conditions** and edge cases
- **Keep documentation up to date** with code changes

```rust
/// Executes a unified hybrid query combining vector similarity, metadata filtering, and graph traversal.
///
/// # Arguments
///
/// * `query` - The unified query specification containing vector, metadata, and graph constraints
/// * `tenant_id` - Tenant identifier for security isolation
/// * `snapshot` - MVCC snapshot for consistent reads
///
/// # Returns
///
/// Returns a `Result` containing a vector of `ScoredEntity` objects
/// ordered by combined similarity score (highest first).
///
/// # Errors
///
/// Returns `QueryError` if:
/// * The query vector has invalid dimensions
/// * Metadata filter syntax is invalid
/// * Graph constraint references non-existent entities
/// * Tenant access is denied
/// * MVCC snapshot is invalid
///
/// # Examples
///
/// ```
/// use phenix_db::{PhenixDB, UnifiedQuery, TenantId};
///
/// let db = PhenixDB::new().await?;
/// let query = UnifiedQuery::new()
///     .with_vector_similarity(vec![0.1, 0.2, 0.3], 10)
///     .with_metadata_filter(json!({"category": "tech"}))
///     .with_graph_constraint("related_to:entity_123");
/// 
/// let results = db.hybrid_query(query, TenantId::from("tenant1"), Snapshot::latest()).await?;
///
/// for result in results {
///     println!("Entity {}: score {}, metadata: {:?}", 
///              result.entity.id, result.score, result.entity.metadata);
/// }
/// ```
pub async fn hybrid_query(
    &self,
    query: UnifiedQuery,
    tenant_id: TenantId,
    snapshot: Snapshot,
) -> Result<Vec<ScoredEntity>, QueryError> {
    // Implementation with unified query planning
}
```

### README and Guides

- **Keep README up to date** with latest features
- **Provide clear examples** for common use cases
- **Document configuration options** thoroughly
- **Include troubleshooting guides** for common issues

## Performance Considerations

### Benchmarking

Always benchmark performance-critical changes:

```bash
# Run benchmarks
cargo bench

# Compare with baseline
cargo bench -- --save-baseline main
git checkout feature-branch
cargo bench -- --baseline main
```

### Memory Management

- **Use `Box<dyn Trait>` for trait objects** when needed
- **Prefer `Rc<RefCell<T>>` over `Arc<Mutex<T>>`** for single-threaded scenarios
- **Use `Arc<T>` for immutable shared data**
- **Consider memory pools** for frequent allocations
- **Profile memory usage** with tools like `valgrind` or `heaptrack`

### Concurrency

- **Use `tokio` for async operations**
- **Prefer message passing** over shared state
- **Use `RwLock` instead of `Mutex`** when appropriate
- **Avoid blocking operations** in async contexts
- **Consider work-stealing** for CPU-intensive tasks

## Security Guidelines

### Secure Coding Practices for Phenix DB

- **Tenant isolation** - Always validate tenant access for all operations
- **Envelope encryption** - Use per-tenant DEK with KMS CMK for all data
- **Input validation** - Sanitize vectors, metadata, and graph constraints at API boundaries
- **Constant-time operations** - Avoid timing attacks in cryptographic and comparison operations
- **Secure key management** - Integrate with KMS for key lifecycle and rotation
- **Audit logging** - Log all security-relevant operations with tenant context
- **RBAC enforcement** - Check permissions at every API and management layer
- **Signed operations** - Use signed WAL entries and snapshots for integrity

### Cryptography for Phenix DB

- **Envelope encryption pattern** - DEK for data encryption, CMK for key encryption
- **Use established libraries** (ring, rustls, aws-kms, etc.)
- **AES-GCM or ChaCha20-Poly1305** for vector and metadata encryption
- **Key rotation support** - Implement seamless key rotation without downtime
- **KMS integration** - Support multiple KMS providers (AWS KMS, HashiCorp Vault, etc.)
- **Signed WAL entries** - Ensure integrity of transaction logs
- **mTLS for internal communication** - Secure shard-to-shard communication
- **Audit all cryptographic operations** - Comprehensive security review required

### Memory Safety

- **Leverage Rust's ownership system**
- **Avoid `unsafe` code** unless absolutely necessary
- **Audit all `unsafe` blocks** thoroughly
- **Use tools like `miri`** to detect undefined behavior
- **Consider fuzzing** for input validation

## Community

### Communication Channels

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and discussions
- **Discord** - Real-time chat and security issue reporting

### Getting Help

- **Search existing issues** before asking questions
- **Provide minimal reproducible examples**
- **Include relevant system information**
- **Be patient and respectful** when asking for help

### Mentorship

New contributors are welcome! If you're new to:

- **Rust** - We can help you learn Rust-specific patterns
- **Vector databases** - We can explain domain concepts
- **Open source** - We can guide you through the contribution process
- **Performance optimization** - We can share profiling and optimization techniques

### Recognition

Contributors to Phenix DB are recognized through:

- **Contributor list** in README and project documentation
- **Release notes** highlighting significant unified database improvements
- **GitHub contributor graphs** and statistics
- **Community shout-outs** for exceptional contributions to the unified architecture
- **Technical blog posts** featuring major architectural contributions
- **Conference presentations** acknowledging key contributors

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** - Incompatible API changes
- **MINOR** - New functionality (backward compatible)
- **PATCH** - Bug fixes (backward compatible)

### Release Checklist

For maintainers preparing releases:

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release notes
6. Tag release
7. Publish to crates.io
8. Update Docker images
9. Announce release

## License

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to Phenix DB! Your efforts help make this unified vector + document + graph database better for everyone. Together, we're building the future of transactional, scalable, and secure data storage for AI/ML applications.