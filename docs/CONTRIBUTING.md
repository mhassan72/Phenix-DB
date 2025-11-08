# Contributing to Phenix-DB

Thank you for your interest in contributing to Phenix-DB! This project aims to build a mathematically grounded, adaptive, distributed memory database that transforms static storage into cognitive memory.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Philosophy](#development-philosophy)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

---

## Code of Conduct

Phenix-DB is committed to fostering an open, welcoming, and inclusive community. We expect all contributors to:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on what is best for the community and the project
- Show empathy towards other community members
- Provide and accept constructive feedback gracefully

---

## Getting Started

### Prerequisites

- **Rust**: Latest stable version (1.70+)
- **Git**: For version control
- **Basic understanding**: Linear algebra, probability theory, or distributed systems (helpful but not required)

### Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Phenix-DB.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `cargo test`
6. Commit and push your changes
7. Open a pull request

---

## Development Philosophy

Phenix-DB is built on five mathematical pillars. Every contribution should align with these principles:

### Core Principles

1. **Recursion** — Memory references itself to learn and optimize
2. **Probability** — Adaptive retrieval based on patterns, not deterministic lookups
3. **Optimization** — Paths evolve as the system learns (Bellman optimization)
4. **Geometry** — Semantic meaning preserved in non-Euclidean space
5. **Compression** — Dense storage without information loss (Kolmogorov complexity)

### Design Patterns

- **Recursive composition**: Components reference and optimize each other
- **Probabilistic coordination**: Entropy-driven consensus instead of deterministic
- **Mathematical correctness**: Unit tests validate mathematical properties
- **Adaptive behavior**: System learns and evolves over time
- **Semantic awareness**: Operations preserve meaning, not just data
- **Distributed intelligence**: Each node contributes to global consciousness

### Architectural Principles

- **OOP**: Object-Oriented modularity for clarity and reusability
- **SRP**: Single Responsibility — each component does one thing well
- **DRY**: Don't Repeat Yourself — abstract common logic
- **KISS**: Keep It Simple, but mathematically rigorous
- **SOLID**: Scalable and replaceable architecture

---

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

#### 1. Code Contributions
- Implement core mathematical modules (RPI, PGM, KCE, Bellman Optimizer)
- Add new features aligned with the roadmap
- Optimize performance (SIMD, GPU acceleration)
- Fix bugs and improve error handling

#### 2. Documentation
- Improve API documentation
- Write tutorials and guides
- Document mathematical derivations
- Create examples and use cases

#### 3. Testing
- Write unit tests for mathematical correctness
- Create integration tests
- Develop performance benchmarks
- Conduct chaos engineering tests

#### 4. Research & Design
- Propose new mathematical approaches
- Design distributed algorithms
- Research compression techniques
- Explore learning algorithms

#### 5. Community Support
- Answer questions in discussions
- Review pull requests
- Help newcomers get started
- Share use cases and feedback

---

## Development Setup

### 1. Install Dependencies

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository
git clone https://github.com/mhassan72/Phenix-DB.git
cd Phenix-DB

# Build the project
cargo build

# Run tests
cargo test
```

### 2. Project Structure

```
src/
├── core/           # Core entity and transaction logic
├── mathematical/   # Mathematical algorithms (polynomial, probability, etc.)
├── memory/         # Memory substrate (RPI, PGM, KCE, Bellman)
├── storage/        # Three-tier storage (hot/warm/cold)
├── distributed/    # Distributed consciousness and consensus
├── concurrency/    # Lock-free concurrency and MVCC
├── learning/       # Adaptive learning system
├── security/       # Encryption and authentication
└── api/            # gRPC and REST interfaces
```

### 3. Running Locally

```bash
# Run the server
cargo run --bin server

# Run CLI tools
cargo run --bin cli -- --help

# Run with features
cargo run --features "simd,gpu"
```

---

## Coding Standards

### Rust Style

- Follow idiomatic Rust patterns
- Use the type system for safety
- Prefer immutability where possible
- Use `Result` and `Option` for error handling
- Run `cargo fmt` before committing
- Run `cargo clippy` and fix warnings

### Naming Conventions

- **Modules**: `snake_case` (e.g., `polynomial_tree.rs`)
- **Structs**: `PascalCase` (e.g., `RecursivePolynomialIndex`)
- **Functions**: `snake_case` (e.g., `evaluate_polynomial`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_POLYNOMIAL_DEGREE`)
- **Traits**: Descriptive with behavior (e.g., `AdaptiveLearning`)

### Mathematical Code

- **Document derivations**: Include references to mathematical principles
- **Cite formulas**: Comment with the source (e.g., "Bellman equation, Kantorovich 1939")
- **Validate properties**: Write tests that verify mathematical correctness
- **Use clear notation**: Match variable names to mathematical notation when possible

Example:
```rust
/// Evaluates polynomial using Al-Karaji recursive pattern
/// P(x) = Σ aᵢ * xⁱ for i = 0 to n
fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().enumerate()
        .fold(0.0, |acc, (i, &coeff)| acc + coeff * x.powi(i as i32))
}
```

### Probability and Statistics

- Probabilities must be in range [0.0, 1.0]
- Probability distributions must sum to 1.0 (within tolerance 0.001)
- Use atomic operations for concurrent probability updates
- Implement decay for unused edges

Example:
```rust
/// Updates edge probability based on co-access pattern
fn update_probability(&mut self, co_accessed: bool) {
    if co_accessed {
        self.probability = (self.probability + 0.1).min(1.0);
    } else {
        self.probability = (self.probability - 0.01).max(0.0);
    }
}
```

### Concurrency

- Use lock-free atomics with `SeqCst` ordering
- Implement MVCC for snapshot isolation
- Never block readers during writes
- Use probabilistic quorum, not strict majority

---

## Testing Requirements

### Test Categories

#### 1. Mathematical Correctness Tests
Located in `tests/mathematical/`

- Verify polynomial evaluation produces correct results
- Ensure probability distributions sum to 1.0
- Validate entropy calculations stay in [0.0, 1.0]
- Test compression is lossless (round-trip)

Example:
```rust
#[test]
fn test_probability_distribution_sums_to_one() {
    let edges = create_test_edges();
    let sum: f32 = edges.iter().map(|e| e.probability).sum();
    assert!((sum - 1.0).abs() < 0.001, "Probabilities must sum to 1.0");
}
```

#### 2. Integration Tests
Located in `tests/integration/`

- End-to-end workflows
- Multi-component interactions
- Distributed scenarios

#### 3. Performance Benchmarks
Located in `benches/`

- Polynomial evaluation speed
- Compression ratio and latency
- Query throughput
- Learning convergence rate

#### 4. Cognitive Behavior Tests
Located in `tests/cognitive/`

- Adaptive learning behavior
- Access pattern prediction
- Self-reorganization triggers

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test category
cargo test --test mathematical

# Run benchmarks
cargo bench

# Run with coverage
cargo tarpaulin --out Html
```

---

## Pull Request Process

### Before Submitting

1. **Create an issue** describing the problem or feature
2. **Discuss the approach** with maintainers if it's a significant change
3. **Write tests** that validate your changes
4. **Update documentation** if you're changing APIs or behavior
5. **Run all tests** and ensure they pass
6. **Format code**: `cargo fmt`
7. **Check lints**: `cargo clippy`

### PR Guidelines

- **Title**: Clear and descriptive (e.g., "Implement Bellman optimizer for path selection")
- **Description**: Explain what, why, and how
  - What problem does this solve?
  - What approach did you take?
  - Are there any trade-offs or limitations?
- **Link to issue**: Reference related issues
- **Tests**: Include test results or benchmark comparisons
- **Breaking changes**: Clearly mark and explain any breaking changes

### Review Process

1. **Automated checks**: CI must pass (tests, lints, formatting)
2. **Mathematical review**: For mathematical components, a reviewer will verify correctness
3. **Code review**: At least one maintainer approval required
4. **Discussion**: Address feedback and questions
5. **Merge**: Once approved, maintainers will merge your PR

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Detailed explanation if needed

Fixes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Examples:
- `feat(rpi): implement recursive polynomial index`
- `fix(pgm): correct probability normalization`
- `docs(api): add examples for query interface`
- `test(bellman): add convergence tests`

---

## Community

### Communication Channels

- **GitHub Discussions**: For questions, ideas, and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions and reviews

### Getting Help

- Check existing documentation in `docs/`
- Search GitHub issues for similar questions
- Ask in GitHub Discussions
- Review the project documentation and guides

### Recognition

We value all contributions! Contributors will be:
- Listed in the project's contributors page
- Acknowledged in release notes for significant contributions
- Invited to join the core team for sustained, high-quality contributions

---

## Ethical Directive

This project is open-source for the advancement of human knowledge. We are building transparent, auditable, and fair data systems. Phenix-DB will never hide its memory structure or bias. It is designed to serve collective intelligence, not replace it.

### Security

- Never commit credentials or secrets
- Report security vulnerabilities privately to maintainers
- Use reproducible builds
- Verify third-party dependencies

### Licensing

By contributing to Phenix-DB, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

## Final Motto

**"True intelligence begins with memory."**

Phenix-DB is humanity's step toward computational remembrance. A system born from mathematics — for understanding, not storage.

Thank you for contributing to this vision!

---

## Questions?

If you have questions about contributing, please:
1. Check the documentation in `docs/`
2. Ask in GitHub Discussions
3. Open an issue for clarification

We're here to help you contribute successfully!
