# Portfolio Summary: JARVIS AI Assistant

## Elevator Pitch

JARVIS is a local-first AI assistant for iMessage workflows on macOS.
It prioritizes privacy, low latency, and practical product behavior over flashy demos.

## What I Built

- End-to-end pipeline: ingestion → retrieval → generation → delivery
- Multi-interface product: CLI, FastAPI backend, and Tauri/Svelte desktop app
- Vector retrieval and indexing path over SQLite + `sqlite-vec`
- Reliability and health signals for runtime monitoring
- Structured evaluation workflows for quality and latency

## Technical Outcomes (Measured)

| Metric | Value | Note |
| --- | --- | --- |
| End-to-end draft pipeline | ~300ms | Target <500ms |
| Mobilization classifier | 12ms p50 | Target <50ms |
| Vector retrieval | 3ms p50 | sqlite-vec path |
| Generation | ~180ms/token | Warm-path behavior |
| Representative mean draft latency | 0.42s | Eval snapshot |
| Representative p95 latency | 1.15s | Eval snapshot |
| Retrieval hit@5 | 0.88 | Eval snapshot |
| Hallucination gate pass | 96.2% | Eval snapshot |

## Performance Improvements Delivered

| Change | Before | After/Impact |
| --- | --- | --- |
| Conversation startup query | 1400ms | Sub-100ms via query redesign |
| Message load path | 500ms/page | Reduced with better SQL strategy |
| Fact extraction | 1 segment/call | 5 segments/call (5x throughput) |
| Vector inserts | Per-item writes | Batched transaction (~3x faster) |
| Index memory footprint | Baseline FAISS | 3.8x compression, ~92% recall |

## Design Decisions

- **Local-first over cloud-first** — Chosen for privacy guarantees and predictable latency
- **MLX-native runtime over wrapper tools** — Chosen for tighter memory control on Apple Silicon
- **Template-first with generation fallback** — Chosen to reduce latency and guard low-value generations
- **Socket IPC for desktop path** — Chosen for lower overhead than HTTP in high-frequency local calls
- **RAG + few-shot over user-level fine-tuning** — Chosen for maintainability and safer behavior under changing user style

## Failed Experiments I Learned From

- **Direct response retrieval** — Failed because similar incoming texts required different replies across contacts
- **Pure embedding classification** — Failed to hit quality bar consistently without structural/rule signals
- **Single global confidence threshold** — Failed due to class-specific calibration differences
- **Polling-based update flow** — Failed from latency/race issues; replaced by watcher + push
- **Heavy fine-tuning for personalization** — Failed cost/maintenance tradeoff and style-drift concerns

## Tech Stack

- Python, FastAPI, SQLite, `sqlite-vec`
- MLX (Apple Silicon local model runtime)
- Svelte + Tauri + TypeScript
- Pytest, benchmark/eval pipelines, structured observability

## Where To Look In Code

- Core pipeline: `jarvis/`
- API: `api/`
- Desktop client: `desktop/`
- Benchmarks/evals: `evals/`, `benchmarks/`
- Architecture docs: `docs/ARCHITECTURE.md`, `docs/HOW_IT_WORKS.md`
