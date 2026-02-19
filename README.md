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

## Documentation

- `docs/HOW_IT_WORKS.md`
- `docs/ARCHITECTURE.md`
- `docs/RUNBOOK.md`
- `docs/TESTING_GUIDELINES.md`

## Privacy and Security Notes

- Personal exports, logs, and local training artifacts are intentionally excluded from this public version.
- Use `.env` for secrets and never commit credential files.
- See `docs/SECURITY.md` for security guidance.
