# Documentation Index

> **Last Updated:** 2026-02-16

## Core

| Doc | Purpose |
|-----|---------|
| [HOW_IT_WORKS.md](./HOW_IT_WORKS.md) | End-to-end system overview, message flow, services |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | V2 technical architecture, implementation phases |
| [SCHEMA.md](./SCHEMA.md) | Database schema (iMessage, JARVIS, vector search) |
| [REPLY_PIPELINE_GUIDE.md](./REPLY_PIPELINE_GUIDE.md) | Reply generation: classification, context, prompts, RAG |

## Design

[`docs/design/`](./design/) contains design docs:
- [OVERVIEW.md](./design/OVERVIEW.md) — Architecture summary
- [DECISIONS.md](./design/DECISIONS.md) — Key decisions and rationale
- [PIPELINE.md](./design/PIPELINE.md) — Classification & routing pipeline
- [EMBEDDINGS.md](./design/EMBEDDINGS.md) — Embedding strategy
- [CONTACT_PROFILES.md](./design/CONTACT_PROFILES.md) — Contact profile system
- [TOPIC_SEGMENTATION.md](./design/TOPIC_SEGMENTATION.md) — Topic boundary detection
- [TEXT_NORMALIZATION.md](./design/TEXT_NORMALIZATION.md) — Text preprocessing
- [METRICS.md](./design/METRICS.md) — Metrics and observability design
- [FEEDBACK.md](./design/FEEDBACK.md) — User feedback mechanism
- [V2_ARCHITECTURE.md](./design/V2_ARCHITECTURE.md) — V2 architecture details
- [fact_extraction_strategy.md](./design/fact_extraction_strategy.md) — Fact extraction design

## Research

[`docs/research/`](./research/) contains research findings:
- [prompt_experiments.md](./research/prompt_experiments.md) — Prompt engineering learnings

## Standards & Guidelines

| Doc | Purpose |
|-----|---------|
| [PERFORMANCE.md](./PERFORMANCE.md) | **Consolidated**: Performance rules, N+1 prevention, optimizations, model memory management |
| [TESTING_GUIDELINES.md](./TESTING_GUIDELINES.md) | Test patterns, fixtures, mock strategies, pytest conventions |
| [PROMPT_MODEL_GOVERNANCE_POLICY.md](./PROMPT_MODEL_GOVERNANCE_POLICY.md) | Prompt/model versioning, evaluation gates, change management |
| [SECURITY.md](./SECURITY.md) | Security guidelines and best practices |
| [FACADE_MIGRATION.md](./FACADE_MIGRATION.md) | Canonical import mapping after facade retirement |

## Operations

| Doc | Purpose |
|-----|---------|
| [RUNBOOK.md](./RUNBOOK.md) | On-call procedures, alert responses, recovery steps |
| [RELIABILITY_FRAMEWORK.md](./RELIABILITY_FRAMEWORK.md) | Offline mode, degradation policies, resilience testing |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common issues and solutions |

## Reference

| Doc | Purpose |
|-----|---------|
| [COMPONENT_CATALOG.md](./COMPONENT_CATALOG.md) | Svelte component reference (41 components) |
| [EXTRACTOR_BAKEOFF.md](./EXTRACTOR_BAKEOFF.md) | Fact extraction approach comparison |
| [fact_extraction_review.md](./fact_extraction_review.md) | Fact extraction analysis and findings |
| [CLEANUP_SUMMARY.md](./CLEANUP_SUMMARY.md) | Codebase cleanup documentation (2026-02-15) |

## Archived

[`docs/archived/`](./archived/) contains historical/completed documents:
- Performance: `PERFORMANCE_OPTIMIZATIONS.md`, `PERFORMANCE_RULES.md`, `REUSE_AND_LOAD_UNLOAD_AUDIT.md`
- Roadmaps: `FACT_KG_ROADMAP.md`, `OBSERVABILITY_ROADMAP.md`, `REPOSITORY_MODERNIZATION_ROADMAP.md`, `DEV_PRODUCTIVITY_PLAN.md`
- Audit Reports: `SQL_OPTIMIZATION_REPORT.md`, `DATABASE_QUERY_AUDIT.md`, `API_ERROR_HANDLING_STANDARD.md`
- Migration: `V4_MIGRATION_REPORT.md`
- Release: `RELEASE_READINESS_CHECKLIST.md`

## Runtime

- **CLI**: `jarvis --help`
- **API**: `http://localhost:8742/docs`
