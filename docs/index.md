# Phenix DB Documentation

Welcome to the comprehensive documentation for Phenix DB, the unified vector + document + graph database.

## Quick Navigation

### 🚀 Getting Started
- [Project Overview](../README.md) - Main project information and quick start
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project
- [Development Setup](development/getting-started.md) - Local development environment

### 🏗️ Architecture
- [System Overview](architecture/overview.md) - High-level system architecture
- [Unified Data Model](architecture/data-model.md) - Entity model with vectors, metadata, and edges
- [Storage Architecture](architecture/storage-architecture.md) - Hot/cold tiering and persistence
- [Query Planning](architecture/query-planning.md) - Unified query execution and optimization
- [Security Model](architecture/security-model.md) - Encryption, auth, and tenant isolation
- [Scaling Strategy](architecture/scaling-strategy.md) - Sharding, replication, and growth patterns

### 📡 API Reference
- [gRPC Reference](api/grpc-reference.md) - gRPC service definitions and usage
- [REST Reference](api/rest-reference.md) - REST API endpoints and examples
- [Unified Queries](api/unified-queries.md) - Hybrid query syntax and examples
- [SDK Examples](api/sdk-examples/) - Language-specific SDK usage guides

### 💻 Development
- [Code Organization](development/code-organization.md) - Module structure and boundaries
- [Testing Guide](development/testing-guide.md) - Testing strategies and frameworks
- [Performance Tuning](development/performance-tuning.md) - Optimization techniques and profiling
- [Debugging Guide](development/debugging-guide.md) - Common issues and troubleshooting

### 🚀 Deployment
- [Kubernetes Deployment](deployment/kubernetes.md) - K8s deployment and configuration
- [Docker Deployment](deployment/docker.md) - Container deployment options
- [Configuration Reference](deployment/configuration.md) - Environment variables and settings
- [Monitoring Setup](deployment/monitoring.md) - Observability setup and dashboards
- [Backup & Recovery](deployment/backup-recovery.md) - Data protection and disaster recovery

### 🔒 Security
- [Encryption](security/encryption.md) - Envelope encryption and key management
- [Authentication](security/authentication.md) - Auth mechanisms and RBAC
- [Tenant Isolation](security/tenant-isolation.md) - Multi-tenant security model
- [Compliance](security/compliance.md) - Audit logging and regulatory requirements

### 📚 Tutorials
- [Your First Entity](tutorials/first-entity.md) - Creating your first unified entity
- [Hybrid Queries](tutorials/hybrid-queries.md) - Building complex vector+metadata+graph queries
- [Scaling Your Deployment](tutorials/scaling-deployment.md) - Growing from single-node to cluster
- [Migration Guide](tutorials/migration-guide.md) - Migrating from other vector databases

## Documentation Standards

This documentation follows established standards to ensure consistency, accuracy, and developer-friendliness.

### Key Principles
- **Developer-first**: Written for developers who need to understand, modify, or debug the code
- **Context-aware**: Explains not just what the code does, but why it was designed that way
- **Unified perspective**: Documents vector, metadata, and graph operations as integrated features
- **Practical examples**: Includes working code snippets that demonstrate real usage patterns

### File Organization
- **README.md**: Only .md file in project root - contains project overview and quick start
- **All other documentation**: Organized in this docs/ folder with logical subdirectories
- **Living documentation**: Updated alongside code changes to maintain accuracy

## Contributing to Documentation

Documentation improvements are always welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for:

- How to write effective documentation
- Documentation review process
- Style guidelines and templates
- Testing documentation changes

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/mhassan72/Rust-Vector-Database/issues) for bugs and feature requests
- **Discussions**: [GitHub Discussions](https://github.com/mhassan72/Rust-Vector-Database/discussions) for questions and ideas
- **Security**: Report security issues via Discord (see main README for community channels)

---

*This documentation is maintained by the Phenix DB community and updated with each release.*