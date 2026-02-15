# Documentation Index

> **Last Updated:** 2026-02-13

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

## Standards & Guidelines

| Doc | Purpose |
|-----|---------|
| [API_ERROR_HANDLING_STANDARD.md](./API_ERROR_HANDLING_STANDARD.md) | Error taxonomy, codes, retry semantics, payload schemas |
| [PERFORMANCE_RULES.md](./PERFORMANCE_RULES.md) | N+1 prevention, latency thresholds, performance testing |
| [TESTING_GUIDELINES.md](./TESTING_GUIDELINES.md) | Test patterns, fixtures, mock strategies, pytest conventions |
| [PROMPT_MODEL_GOVERNANCE_POLICY.md](./PROMPT_MODEL_GOVERNANCE_POLICY.md) | Prompt/model versioning, evaluation gates, change management |

## Operations

| Doc | Purpose |
|-----|---------|
| [RUNBOOK.md](./RUNBOOK.md) | On-call procedures, alert responses, recovery steps |
| [RELIABILITY_FRAMEWORK.md](./RELIABILITY_FRAMEWORK.md) | Offline mode, degradation policies, resilience testing |
| [OBSERVABILITY_ROADMAP.md](./OBSERVABILITY_ROADMAP.md) | Metrics, logging, tracing, SLOs, dashboards |
| [RELEASE_READINESS_CHECKLIST.md](./RELEASE_READINESS_CHECKLIST.md) | Pre-release verification gates |

## Roadmaps & Plans

| Doc | Purpose |
|-----|---------|
| [REPOSITORY_MODERNIZATION_ROADMAP.md](./REPOSITORY_MODERNIZATION_ROADMAP.md) | 90-day complexity reduction plan |
| [DEV_PRODUCTIVITY_PLAN.md](./DEV_PRODUCTIVITY_PLAN.md) | Developer experience improvements |
| [DEV_PRODUCTIVITY_PLAN.md](./DEV_PRODUCTIVITY_PLAN.md) | Developer experience improvements |
| [FACT_KG_ROADMAP.md](./FACT_KG_ROADMAP.md) | Fact extraction & knowledge graph roadmap |

## Reference

| Doc | Purpose |
|-----|---------|
| [V4_MIGRATION_REPORT.md](./V4_MIGRATION_REPORT.md) | V4 Fact Extraction design decisions, findings, and lessons learned |
| [COMPONENT_CATALOG.md](./COMPONENT_CATALOG.md) | Svelte component reference (41 components) |
| [EXTRACTOR_BAKEOFF.md](./EXTRACTOR_BAKEOFF.md) | Fact extraction approach comparison |
| [SQL_OPTIMIZATION_REPORT.md](./SQL_OPTIMIZATION_REPORT.md) | SQL query optimization findings |

## Runtime

- **CLI**: `jarvis --help`
- **API**: `http://localhost:8742/docs`
