# JARVIS AI Assistant

Local-first AI assistant for iMessage workflows on macOS.

## 60-Second Overview

JARVIS is an end-to-end product system I built to test whether small local models can deliver practical reply drafting with strong latency and privacy constraints.

It combines:
- local message retrieval (read-only)
- retrieval-augmented generation
- intent routing and reliability gates
- desktop app + API + observability

## Results (Measured)

| Metric | Result |
| --- | --- |
| End-to-end draft pipeline | ~300ms |
| Mobilization classifier | 12ms p50 |
| Vector retrieval (`sqlite-vec`) | 3ms p50 |
| Generation latency | ~180ms/token |
| Mean draft latency | 0.42s |
| P95 draft latency | 1.15s |
| Retrieval hit@5 | 0.88 |
| Hallucination gate pass | 96.2% |

## Technical Decisions

Shipped:
- Local-first inference over cloud-first for privacy and predictable latency.
- MLX-native runtime for better Apple Silicon memory control.
- Template-first + generation fallback to reduce cost and hallucination risk.
- Socket-based desktop IPC for lower local overhead.
- RAG + few-shot over heavy fine-tuning for maintainability.

Tried and rejected:
- Direct response retrieval from nearest neighbor.
- Pure embedding-only classification.
- Single global confidence threshold.
- Polling-based update flow.
- Heavy per-user fine-tuning for personalization.

## Architecture (Simplified)

```text
+----------------------+      +----------------------+      +----------------------+
| iMessage Integrator  | ---> | Core Pipeline        | ---> | Output Interfaces    |
| (local read-only)    |      | intent/retrieval/gen |      | CLI / API / Desktop  |
+----------------------+      +----------+-----------+      +----------+-----------+
                                         |                             |
                                         v                             v
                               +----------------------+      +----------------------+
                               | SQLite + Vector Index|      | Metrics + Health     |
                               | context + search     |      | tracing + reliability|
                               +----------------------+      +----------------------+
```

## For Employers

- Portfolio summary: `PORTFOLIO.md`
- Showcase brief: `docs/SHOWCASE.md`
- System walkthrough: `docs/HOW_IT_WORKS.md`
- Architecture details: `docs/ARCHITECTURE.md`
- Design decisions: `docs/design/DECISIONS.md`
- Performance notes: `docs/PERFORMANCE.md`

## Quick Start

```bash
git clone <repo-url>
cd jarvis-ai-assistant
cp .env.example .env
make setup
make verify
jarvis serve
```

## Privacy Note

This public repository is sanitized for sharing: no personal exports, no local logs, no secret material.

