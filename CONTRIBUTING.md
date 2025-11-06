# Contributing to Rust Vector Database

Thank you for your interest in contributing to the Rust Vector Database! This document provides guidelines and information for contributors.

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
   git clone https://github.com/your-username/rust-vector-database.git
   cd rust-vector-database
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

We welcome various types of contributions:

- **Bug fixes** - Fix issues and improve stability
- **Feature implementations** - Add new functionality
- **Performance improvements** - Optimize existing code
- **Documentation** - Improve docs, examples, and guides
- **Tests** - Add test coverage and improve test quality
- **Refactoring** - Improve code structure and maintainability

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
feat(api): add gRPC streaming support for batch operations

fix(storage): resolve memory leak in hot tier cache

docs(readme): update installation instructions

test(index): add benchmarks for HNSW search performance
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

### Code Organization

- **Keep functions small** and focused on single responsibility
- **Use modules** to organize related functionality
- **Define clear interfaces** with well-documented traits
- **Separate concerns** between layers (API, business logic, storage)
- **Use dependency injection** for testability

### Error Handling

```rust
// Good: Use proper error types
fn process_vector(vector: &Vector) -> Result<ProcessedVector, ProcessingError> {
    // Implementation
}

// Bad: Use generic errors or panics
fn process_vector(vector: &Vector) -> ProcessedVector {
    // Don't panic in production code
    vector.validate().unwrap()
}
```

### Performance Guidelines

- **Use SIMD operations** where applicable
- **Minimize allocations** in hot paths
- **Use zero-copy operations** when possible
- **Profile before optimizing** - measure, don't guess
- **Consider memory layout** for cache efficiency
- **Use appropriate data structures** for the use case

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
    fn test_vector_normalization() {
        let input = Vector::new(vec![3.0, 4.0]);
        let result = normalize_vector(&input).unwrap();
        
        assert_eq!(result.magnitude(), 1.0);
        assert_eq!(result.dimensions(), &[0.6, 0.8]);
    }
    
    #[tokio::test]
    async fn test_async_operation() {
        let database = setup_test_database().await;
        let result = database.insert_vector(test_vector()).await;
        
        assert!(result.is_ok());
    }
}
```

### Test Guidelines

- **Test both success and failure cases**
- **Use descriptive test names** that explain what is being tested
- **Keep tests independent** - no shared state between tests
- **Use test fixtures** for common setup
- **Mock external dependencies** in unit tests
- **Test edge cases** and boundary conditions

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
/// Searches for the k most similar vectors to the query vector.
///
/// # Arguments
///
/// * `query` - The query vector to search for
/// * `k` - The number of similar vectors to return
/// * `filter` - Optional filter to apply to results
///
/// # Returns
///
/// Returns a `Result` containing a vector of `SimilarityResult` objects
/// ordered by similarity score (highest first).
///
/// # Errors
///
/// Returns `SearchError` if:
/// * The query vector has invalid dimensions
/// * The index is corrupted or unavailable
/// * The value of k exceeds the maximum allowed
///
/// # Examples
///
/// ```
/// use vector_db::{VectorIndex, Vector};
///
/// let index = VectorIndex::new();
/// let query = Vector::new(vec![0.1, 0.2, 0.3]);
/// let results = index.search(&query, 10, None)?;
///
/// for result in results {
///     println!("Vector {}: similarity {}", result.id, result.similarity);
/// }
/// ```
pub async fn search(
    &self,
    query: &Vector,
    k: usize,
    filter: Option<Filter>,
) -> Result<Vec<SimilarityResult>, SearchError> {
    // Implementation
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

### Secure Coding Practices

- **Validate all inputs** at API boundaries
- **Use constant-time comparisons** for cryptographic operations
- **Avoid timing attacks** in security-sensitive code
- **Use secure random number generation**
- **Handle secrets securely** (no logging, proper cleanup)

### Cryptography

- **Use established libraries** (ring, rustls, etc.)
- **Don't implement crypto primitives** yourself
- **Use appropriate key sizes** and algorithms
- **Implement proper key rotation**
- **Audit cryptographic code** carefully

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

Contributors are recognized through:

- **Contributor list** in README
- **Release notes** mentioning significant contributions
- **GitHub contributor graphs** and statistics
- **Community shout-outs** for exceptional contributions

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

Thank you for contributing to the Rust Vector Database! Your efforts help make this project better for everyone.