# Portfolio Summary: JARVIS AI Assistant

## Project Goal

Build a local-first assistant that can generate useful iMessage draft replies with low latency, privacy safeguards, and production-style observability.

## What I Built

- End-to-end pipeline: ingestion -> retrieval -> generation -> delivery.
- Multi-interface product: CLI, FastAPI backend, and Tauri/Svelte desktop app.
- Vector retrieval and indexing path over SQLite + `sqlite-vec`.
- Reliability and health signals for runtime monitoring.
- Structured evaluation workflows for quality and latency.

## Engineering Impact

- Reduced end-to-end draft path to ~300ms.
- Achieved 3ms p50 vector retrieval and 12ms p50 mobilization routing.
- Improved extraction throughput by batching model calls (5x).
- Improved insert/index path with batched transactional writes (~3x).
- Resolved major startup/query regressions (N+1/correlated query patterns).

## Tradeoffs and Decisions

- Chose local inference over cloud APIs for privacy and deterministic behavior.
- Chose MLX-native runtime for tighter memory control on Apple Silicon.
- Chose template-first + generation fallback for latency and reliability.
- Chose RAG + few-shot over fine-tuning to avoid heavy per-user maintenance.

## Failed Experiments I Learned From

- Direct nearest-response replay failed on contact-specific context.
- Pure embedding-only classification failed consistency requirements.
- Global confidence threshold failed class-specific calibration needs.
- Polling-based message update flow failed latency and race-condition goals.

## Tech Stack

- Python, FastAPI, SQLite, `sqlite-vec`
- MLX (Apple Silicon local model runtime)
- Svelte + Tauri + TypeScript
- Pytest, benchmark/eval pipelines, structured observability

## Links

- Main README: `README.md`
- Showcase brief: `docs/SHOWCASE.md`
- Architecture: `docs/ARCHITECTURE.md`
- How it works: `docs/HOW_IT_WORKS.md`
- Design records: `docs/design/DECISIONS.md`
