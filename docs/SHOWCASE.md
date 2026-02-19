# JARVIS Showcase Brief

## Elevator Pitch

JARVIS is a local-first AI assistant for iMessage workflows on macOS.
It prioritizes privacy, low latency, and practical product behavior over flashy demos.

## Product Scope

- Read-only local iMessage ingestion
- Intent-aware reply/summarize/search pipeline
- Retrieval-augmented generation (RAG) over conversation history
- Desktop app (Tauri/Svelte) + FastAPI + socket interfaces
- Reliability and observability built into the serving path

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
| Conversation startup query | 1400ms | Sub-100ms target via query redesign |
| Message load path | 500ms/page | Reduced with better SQL/data access strategy |
| Fact extraction | 1 segment/model call | 5 segments/model call (5x throughput) |
| Vector inserts | Per-item writes | Batched transaction (~3x faster) |
| Index memory footprint | Baseline FAISS flat | 3.8x compression with ~92% recall |

## Design Decisions

- **Local-first over cloud-first**
  - Chosen for privacy guarantees and predictable latency.
- **MLX-native runtime over wrapper tools**
  - Chosen for tighter memory control and Apple Silicon optimization.
- **Template-first with generation fallback**
  - Chosen to reduce latency and guard against low-value generations.
- **Socket IPC for desktop path**
  - Chosen for lower overhead than HTTP in high-frequency local calls.
- **RAG + few-shot over user-level fine-tuning**
  - Chosen for maintainability and safer behavior under changing user style.

## Experiments That Failed (And Why)

- **Direct response retrieval**
  - Failed because similar incoming texts required different replies across contacts.
- **Pure embedding classification**
  - Failed to hit quality bar consistently without structural/rule signals.
- **Single global confidence threshold**
  - Failed due to class-specific calibration differences.
- **Polling-based update flow**
  - Failed from latency/race issues; replaced by watcher + push.
- **Heavy fine-tuning for personalization**
  - Failed cost/maintenance tradeoff and style-drift concerns.

## Security & Privacy Posture

- Sensitive local exports and logs are excluded from public repo state.
- `.env`-based secret management with public `.env.example` template.
- Repository history rewritten to remove previously tracked sensitive artifacts.

## Where To Look In Code

- Core pipeline: `jarvis/`
- API: `api/`
- Desktop client: `desktop/`
- Benchmarks/evals: `evals/`, `benchmarks/`
- Architecture docs: `docs/HOW_IT_WORKS.md`, `docs/ARCHITECTURE.md`

