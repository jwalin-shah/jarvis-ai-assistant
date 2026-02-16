# V4 Fact Extraction Migration Report

> **Date:** 2026-02-13
> **Status:** Implemented & Deployed

This document details the transition to the V4 Fact Extraction pipeline, documenting the design decisions, failed attempts, and final architecture.

## 1. Motivation

The previous extraction pipeline (GLiNER + Rule-Based) suffered from several key limitations:

- **Context Loss**: Extracting from single messages often missed the "who" (attribution) and "when" (temporal status).
- **Precision Issues**: GLiNER would often tag conversational filler (e.g., "I love this era") as preferences ("likes era").
- **Identity Confusion**: The system struggled to distinguish between the user ("Me") and the contact, especially in group chats or when names weren't explicitly mentioned.

## 2. Key Architectural Decisions

### 2.1 Turn-Based Extraction

- **Problem**: Fixed-window segmentation (e.g., 50 messages) often split active conversations or grouped unrelated topics.
- **Solution**: We implemented **Turn-Based Extraction**, which groups consecutive messages from the same sender into a single "turn". This preserves the natural flow of conversation and provides the LLM with coherent context.
- **Result**: significantly improved extraction of complex facts that span multiple short messages.

### 2.2 Instruction-Based Extraction (LFM-0.7b)

- **Approach**: Instead of relying solely on NER tags (GLiNER), we now use a fine-tuned LFM-0.7b model with natural language instructions (ChatML format).
- **Prompt Strategy**:
  - _System_: "Extract 3-5 PERSONAL facts... STRICT RULES: ONLY extract facts about the PEOPLE."
  - _User_: Provides the chat turns.
- **Findings**: This approach is far more resilient to slang and conversational nuance than rigid NER labels.

### 2.3 Address Book Integration

- **Problem**: The system often defaulted to "Contact" or "Unknown" for names.
- **Solution**: We integrated directly with the macOS Address Book (via `chat.db` handles) to resolve:
  - **Contact Name**: Real name from the user's contacts.
  - **User Identity**: "Me" is resolved to the user's actual name (e.g., "Jwalin Shah"), allowing for correct third-person attribution.
- **Impact**: Eliminates "Me" vs "You" ambiguity in extracted facts.

### 2.4 Two-Pass LLM Self-Correction (The "BS Filter")

- **Problem**: The LLM would sometimes hallucinate or extract conversational metaphors as facts (e.g., "I'm dead" -> "Health condition: dead").
- **Solution**: We implemented **Two-Pass Self-Correction** using the same LFM-0.7b model:
  - _Pass 1_: Raw extraction with ChatML prompts
  - _Pass 2_: Self-correction pass that reviews and filters extracted facts
- **Filtering**:
  - Removes commentary markers ("removing", "keeping", "verified facts", etc.)
  - Strips conversational filler and metaphors
  - Only keeps facts in proper "- [Name]: [fact]" format
- **Success Story**: Accurately rejects "I love this era" (metaphor) while keeping "I love sushi" (preference).

> **Note:** An earlier design used a separate NLI (Natural Language Inference) cross-encoder for verification. This was replaced with the two-pass LLM approach for better performance and simpler architecture while maintaining accuracy.

## 3. Failed Attempts & Lessons Learned

### 3.1 Pure Segmentation

- **Attempt**: We tried using semantic topic segmentation to chunk conversations before extraction.
- **Failure**: The segments were often too coarse or too fine, leading to context fragmentation.
- **Pivot**: Turn-based grouping proved to be a more robust and lightweight proxy for "topic" in the context of fact extraction.

### 3.2 GLiNER-Only Pipeline

- **Attempt**: Relying 100% on GLiNER for candidate generation.
- **Failure**: High recall but low precision. It generated too many noise candidates that clogged the downstream filters.
- **Pivot**: Using GLiNER only as a signal/feature, but relying on LFM-0.7b for the primary extraction logic.

### 3.3 NLI Cross-Encoder Verification (Deprecated)

- **Attempt**: Using a separate DeBERTa-v3 cross-encoder for entailment verification.
- **Failure**: Added complexity and memory overhead without significant accuracy gains over two-pass LLM.
- **Pivot**: Replaced with two-pass LLM self-correction in the same model session.

## 4. Final Pipeline Flow

1.  **Ingest**: `ChatDBWatcher` detects new messages.
2.  **Group**: Messages are grouped into Turns (User/Contact).
3.  **Resolve**: Identities are resolved via Address Book.
4.  **Extract (Pass 1)**: LFM-0.7b generates candidate facts (JSON/Bullets).
5.  **Verify (Pass 2)**: Same model reviews and filters candidates.
6.  **Store**: Validated facts are saved to `contact_facts` with attribution.

## 5. Active Components

| Component             | File                                       | Purpose                      | Status           |
| --------------------- | ------------------------------------------ | ---------------------------- | ---------------- |
| Instruction Extractor | `jarvis/contacts/instruction_extractor.py` | Two-pass LLM extraction      | ✅ **Active/V4** |
| Watcher               | `jarvis/watcher.py`                        | Real-time extraction trigger | ✅ **Active**    |
| Worker                | `jarvis/tasks/worker.py`                   | Background extraction tasks  | ✅ **Active**    |
| Backfill Script       | `scripts/backfill_v4_final.py`             | Batch backfill with V4       | ✅ **Active**    |

## 6. Deprecated Components

| Component           | File                                     | Purpose                  | Status            |
| ------------------- | ---------------------------------------- | ------------------------ | ----------------- |
| Candidate Extractor | `jarvis/contacts/candidate_extractor.py` | GLiNER + NLI pipeline    | ⚠️ **Deprecated** |
| Old Backfill        | `scripts/backfill_contact_facts.py`      | Uses candidate_extractor | ⚠️ **Deprecated** |
| Segment Extractor   | `jarvis/contacts/segment_extractor.py`   | GLiNER bridge            | ⚠️ **Deprecated** |

> **Migration Note:** Use `scripts/backfill_v4_final.py` for new backfills. The old `backfill_contact_facts.py` uses the deprecated GLiNER+NLI pipeline.

## 7. Future Work

- **Temporal Resolution**: Better handling of "used to" vs "currently".
- **Conflict Resolution**: improved logic for merging contradictory facts (e.g., "moved to NY" vs "lives in SF").
