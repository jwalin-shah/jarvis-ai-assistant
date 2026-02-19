# JARVIS AI Assistant

Local-first AI assistant for iMessage workflows on macOS.

This repository is intentionally cleaned for public sharing: no personal exports, no local logs, and no secret material.

## Why I Built This

I built JARVIS to explore a practical AI system that can:
- run locally on Apple Silicon
- keep private messages on-device
- generate useful draft replies and summaries from real conversation context
- expose clear reliability and performance signals

## What It Does

- Reads local iMessage data (read-only)
- Classifies user intent (reply, summarize, search, export)
- Retrieves relevant context from recent message history
- Generates draft responses with local or configured model providers
- Serves a desktop app + API with observability and health endpoints

## How It Works

1. Ingestion: iMessage data is read from local sources via integration adapters.
2. Processing: text normalization, feature extraction, and intent/category routing.
3. Retrieval: semantic + lexical lookup over indexed conversation context.
4. Generation: response draft is composed with prompt templates and safety gates.
5. Delivery: output is returned through CLI, REST API, or desktop socket stream.

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

## Resume-Friendly Project Structure

```text
jarvis-ai-assistant/
├── jarvis/              # Core assistant logic
├── api/                 # FastAPI service layer
├── desktop/             # Tauri + Svelte desktop app
├── models/              # Model loading and routing utilities
├── integrations/        # iMessage/calendar connectors
├── contracts/           # Interface contracts and protocol types
├── tests/               # Unit/integration/security tests
├── docs/                # Architecture, runbooks, design notes
├── scripts/             # Utilities and eval runners
└── evals/               # Benchmark/evaluation framework
```

## Quick Start

```bash
git clone <repo-url>
cd jarvis-ai-assistant
cp .env.example .env
make setup
make verify
```

Run locally:

```bash
jarvis chat
jarvis search-messages "dinner"
jarvis serve
```

## Example Output

```text
User: "tell mom i will be there in 20"
Assistant draft: "omw, there in 20"
Intent: reply
Latency: 420ms
```

## Example Evaluation Snapshot

These are representative metrics from local benchmark runs (not tied to personal message content):

| Metric | Value |
| --- | --- |
| Mean draft latency | 0.42s |
| P95 draft latency | 1.15s |
| Retrieval hit@5 | 0.88 |
| Hallucination gate pass | 96.2% |

## Results By The Numbers

System and benchmark highlights from local runs on Apple Silicon:

| Area | Result | Context |
| --- | --- | --- |
| End-to-end reply pipeline | ~300ms | Target: <500ms |
| Mobilization classification | 12ms p50 | Target: <50ms |
| Vector retrieval (`sqlite-vec`) | 3ms p50 | Target: <50ms |
| Generation latency | ~180ms/token | Target: <2s total |
| Startup query regression fixed | 1400ms -> <100ms target | Removed correlated subqueries / N+1 patterns |
| Batched fact extraction | 5x throughput | 5 segments per model call instead of 1 |
| Vector insert path | ~3x faster | Transaction + batch insert strategy |
| Compressed vector index | 3.8x smaller | With ~92% recall tradeoff |

Source docs:
- `docs/HOW_IT_WORKS.md`
- `docs/PERFORMANCE.md`

## Design Decisions And Lessons

What we shipped:
- Local-first architecture instead of cloud inference (privacy and offline reliability).
- MLX-native model runtime instead of wrappers (better memory control on Apple Silicon).
- Template-first + generation fallback (lower latency and less hallucination risk).
- Unix sockets for desktop IPC instead of HTTP polling (faster local communication).
- RAG + few-shot instead of fine-tuning (simpler updates, lower hallucination risk).

What we tried that did not work:
- Direct response retrieval from nearest match:
  Same prompts from different contacts needed different replies; context was missing.
- Pure embedding-only classification:
  Quality was not stable enough for production behavior.
- One global confidence threshold:
  Per-class behavior differed too much, causing class-specific misrouting.
- HTTP polling for updates:
  Added latency and race conditions; moved to watcher + push flow.
- Heavy personalization via fine-tuning:
  Higher maintenance cost and weaker adaptability to evolving user style.

See detailed decision log: `docs/design/DECISIONS.md`
See VC-ready summary: `docs/SHOWCASE.md`

## Documentation

- `docs/HOW_IT_WORKS.md`
- `docs/ARCHITECTURE.md`
- `docs/RUNBOOK.md`
- `docs/TESTING_GUIDELINES.md`

## Privacy and Security Notes

- Personal exports, logs, and local training artifacts are intentionally excluded from this public version.
- Use `.env` for secrets and never commit credential files.
- See `docs/SECURITY.md` for security guidance.
