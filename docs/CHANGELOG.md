# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CHANGELOG.md for tracking project changes
- Organized scripts into subdirectories (production/, evaluation/, training/, analysis/)
- scripts/README.md documenting all utility scripts

### Changed

- Updated .gitignore for better coverage of generated artifacts
- Scripts reorganized for better maintainability

### Removed

- Stale tracked artifacts (bandit reports, one-time audit docs)
- 2.9GB of untracked experiment data and legacy virtual environments
- Experiment results from root directory

### Performance

- Repository size reduced by ~3GB through cleanup

## [1.0.0] - 2026-02-15

### Added

- Native macOS desktop app with Tauri and Svelte
- Unix socket IPC for sub-5ms latency
- Speculative decoding with LFM 350M draft model
- KV cache quantization (8-bit) for memory optimization
- Contact hover cards in desktop UI
- Turbo Mode for faster generation

### Changed

- Optimized iMessage data fetching to eliminate N+1 queries
- Reduced LLM memory limit to 1GB for 8GB systems
- Improved semantic indexing with float16 precision
- Fixed relationship classification logic

### Fixed

- Backfill crash when processing large datasets
- Stuck chat input issue in desktop app
- Memory info display accuracy
- Vec_messages timestamp type mismatch
- RelationshipGraph rendering timing issue

### Performance

- N+1 query patterns eliminated in message fetching (30x faster)
- Memory management optimized for 8GB systems
- Semantic indexing upgraded to float16
- Sub-5ms latency for direct SQLite access

## [0.9.0] - 2026-01-15

### Added

- V4 fact extraction pipeline with instruction-based prompts
- Two-pass LLM self-correction architecture
- Turn-based message grouping for better context
- ChatML prompt templates for extraction
- LFM-0.7b as default extraction model

### Changed

- Migrated from NLI cross-encoder to two-pass LLM approach
- Consolidated prompt governance into single policy document
- Improved attribution accuracy in group chats

### Removed

- GLiNER as primary extraction method (now benchmark-only)
- Legacy NLI entailment stage

## [0.8.0] - 2025-12-01

### Added

- SQLite-vec integration for vector search
- Int8 and binary quantization support
- Per-contact partitioned search
- Topic-based chunking for RAG
- MLX embedding service with GPU acceleration

### Changed

- Migrated from external vector DB to sqlite-vec
- Improved search latency from 50ms to 3ms
- Reduced embedding storage by 75% with quantization

## [0.7.0] - 2025-11-01

### Added

- MLX-based local inference on Apple Silicon
- Intent classification pipeline
- Memory-aware model loading
- Three-tier mode system (FULL/LITE/MINIMAL)
- Circuit breaker pattern for graceful degradation

### Changed

- Default model changed to LFM 2.5 1.2B Inst

ruct (4-bit)

- Memory controller with automatic mode selection
- Prometheus-compatible metrics endpoint

### Security

- Read-only access to iMessage database
- Local-first architecture with no cloud data transmission
- Full disk access permission validation

## [0.5.0] - 2025-09-01

### Added

- Initial iMessage integration
- FastAPI REST backend
- CLI with interactive chat, search, and export
- Conversation summaries
- Basic reply suggestions

### Infrastructure

- Project structure with layered architecture
- Test suite with pytest
- CI/CD with GitHub Actions
- Development tooling (ruff, mypy, pre-commit hooks)
