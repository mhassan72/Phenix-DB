# Phenix-DB: Mathematical Memory Substrate

A self-organizing, learning, and adaptive cognitive memory system grounded in centuries of mathematical wisdom.

## Overview

Phenix-DB transforms static data storage into living memory that learns, compresses, and self-reorganizes. It is not a traditional database—it is a cognitive memory system.

## Core Philosophy

Five mathematical pillars:

1. **Recursion** (Al-Samawal, von Neumann): Memory references itself to learn
2. **Probability** (Kolmogorov, De Moivre): Adaptive retrieval, not absolute
3. **Optimization** (Bellman, Kantorovich): Paths evolve as system learns
4. **Geometry** (Khayyam, Tusi, Euler): Semantic meaning in curved space
5. **Compression** (Ramanujan, Gauss): Dense storage without distortion

## Architecture

### Core Components

- **RPI (Recursive Polynomial Index)**: O(log n) hierarchical recall through polynomial embeddings
- **PGM (Probabilistic Graph Memory)**: Relationships that evolve based on access patterns
- **Bellman Optimizer**: Dynamic path optimization using dynamic programming
- **KCE (Kolmogorov Compression Engine)**: 70-90% storage reduction through mathematical compression
- **VNR (Von Neumann Redundancy Fabric)**: Self-healing fault tolerance
- **Entropy Monitor**: Information density optimization and stagnation prevention

### Three-Tier Memory

- **Hot tier**: RAM/NVMe, <1ms access time
- **Warm tier**: NVMe/SSD, 1-10ms access time
- **Cold tier**: Object storage, 10-100ms access time, compressed

## Building

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Build with all features
cargo build --release --all-features

# Build with specific features
cargo build --release --features simd,gpu
```

## Features

- `simd`: SIMD optimizations (AVX-512, AVX2, SSE)
- `gpu`: GPU acceleration (CUDA, OpenCL)
- `cuda`: CUDA-specific GPU acceleration
- `opencl`: OpenCL-specific GPU acceleration
- `homomorphic`: Homomorphic encryption support
- `full`: Enable all features

## Running

```bash
# Start the server
cargo run --release --bin phenix-server

# Use the CLI
cargo run --release --bin phenix-cli -- --help
```

## Configuration

Configuration is managed through TOML files and environment variables. See `phenix-db.toml.example` for configuration options.

## Development Status

**Current Phase**: Foundation (Phase 1)

The project structure has been initialized with:
- ✅ Cargo workspace configuration
- ✅ Module directory structure
- ✅ Core data types and error handling
- ✅ Feature flags for optional components
- ⏳ Component implementations (in progress)


## License

MIT OR Apache-2.0

## Contributing

See [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for contribution guidelines.
