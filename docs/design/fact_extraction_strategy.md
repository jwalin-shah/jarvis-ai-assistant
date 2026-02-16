# Fact Extraction Strategy & Architecture

## Overview

This document outlines the design decisions, experiment results, and final architecture for the `InstructionFactExtractor` pipeline in Jarvis.

## Core Design Decisions

### 1. Two-Pass Extraction w/ Single-Segment Constraint

We use a two-pass approach but constrained to **single-segment processing** (one conversation chunk at a time):

- **Pass 1 (Extraction)**: identifying raw claims using a strict JSONL system prompt.
- **Pass 2 (Structuring)**: converting claims into S|P|O (Subject | Predicate | Object) triples.
- **Constraint**: Multi-segment batching (processing multiple conversations in one prompt) was removed because the 0.7b model often hallucinated segment IDs or mixed up contexts.

### 2. Prompt Engineering Findings (Bakeoff v3)

We tested 10+ prompt variants. Key findings:

- **JSON > Plaintext**: The 0.7b model follows JSON schemas better than plaintext templates.
- **No Output Priming**: Priming the model with `{"` actually _hurt_ performance by breaking its internal chain-of-thought or causing it to hallucinate.
- **Combined Prompt**: The `combined_v3` prompt (explicit JSONL instructions + negative constraints) performed best (75% grounding).

### 3. Safety & Grounding Mechanisms

To achieve 100% grounding and empty-input safety, we implemented:

- **Empty Guard**: `if not text.strip(): return []`. Prevented the model's memorized hallucination ("Sarah reads book...").
- **Pass-2 Short Circuit**: If Pass-1 returns `NONE`, we skip Pass-2 entirely. This prevents the model from inventing new facts during the structuring phase.
- **Grounding Filter**: Post-processing check that rejects any fact where `value` words do not appear in the source text (< 60% overlap).

### 4. Heuristic Confidence Scoring

Since the model generates free-text predicates (e.g., `is_planning_to_visit`), we map them to confidence scores heuristically:

- **1.0 (High)**: Identity/Work (`works_at`, `is_family_of`, `lives_in`)
- **0.8 (Medium)**: Preferences (`likes`, `loves`, `hates`, `has_hobby`)
- **0.6 (Low)**: Schedule/Availability (`is_free`, `is_busy`)
- **0.7 (Default)**: Generic facts (`has_fact`, `mentioned`)

## Future Improvements

- **Adaptive Batching**: We currently use fixed-size sliding windows (default size=25, overlap=5 in backfill). We should make windows turn-aware so they don't split mid-turn or mid-sentence.
- **Entity Context**: Injecting known contact details (from profile) into the prompt would help resolve pronouns ("she", "they") and relationships.
- **Embedding-Based Confidence**: Replace keyword heuristics with a lightweight classifier or embedding similarity check for more accurate confidence scoring.
