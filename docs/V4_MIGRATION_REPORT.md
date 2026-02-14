# V4 Fact Extraction Migration Report

> **Date:** 2026-02-13
> **Status:** Implemented & Deployed

This document details the transition to the V4 Fact Extraction pipeline, documenting the design decisions, failed attempts, and final architecture.

## 1. Motivation

The previous extraction pipeline (GLiNER + Rule-Based) suffered from several key limitations:
*   **Context Loss**: Extracting from single messages often missed the "who" (attribution) and "when" (temporal status).
*   **Precision Issues**: GLiNER would often tag conversational filler (e.g., "I love this era") as preferences ("likes era").
*   **Identity Confusion**: The system struggled to distinguish between the user ("Me") and the contact, especially in group chats or when names weren't explicitly mentioned.

## 2. Key Architectural Decisions

### 2.1 Turn-Based Extraction
*   **Problem**: Fixed-window segmentation (e.g., 50 messages) often split active conversations or grouped unrelated topics.
*   **Solution**: We implemented **Turn-Based Extraction**, which groups consecutive messages from the same sender into a single "turn". This preserves the natural flow of conversation and provides the LLM with coherent context.
*   **Result**: significantly improved extraction of complex facts that span multiple short messages.

### 2.2 Instruction-Based Extraction (LFM-0.7b)
*   **Approach**: Instead of relying solely on NER tags (GLiNER), we now use a fine-tuned LFM-0.7b model with natural language instructions (ChatML format).
*   **Prompt Strategy**:
    *   *System*: "Extract 3-5 PERSONAL facts... STRICT RULES: ONLY extract facts about the PEOPLE."
    *   *User*: Provides the chat turns.
*   **Findings**: This approach is far more resilient to slang and conversational nuance than rigid NER labels.

### 2.3 Address Book Integration
*   **Problem**: The system often defaulted to "Contact" or "Unknown" for names.
*   **Solution**: We integrated directly with the macOS Address Book (via `chat.db` handles) to resolve:
    *   **Contact Name**: Real name from the user's contacts.
    *   **User Identity**: "Me" is resolved to the user's actual name (e.g., "Jwalin Shah"), allowing for correct third-person attribution.
*   **Impact**: Eliminates "Me" vs "You" ambiguity in extracted facts.

### 2.4 NLI Verification (The "BS Filter")
*   **Problem**: The LLM would sometimes hallucinate or extract conversational metaphors as facts (e.g., "I'm dead" -> "Health condition: dead").
*   **Solution**: We implemented a **Natural Language Inference (NLI)** stage using a cross-encoder.
    *   *Hypothesis*: "{Subject} {Predicate} {Value}"
    *   *Premise*: The source message(s).
*   **Thresholds**:
    *   High confidence (>0.7): Accepted immediately.
    *   Medium confidence (0.3-0.7): Verified against NLI.
    *   Low confidence (<0.3): Rejected.
*   **Success Story**: Accurately rejects "I love this era" (metaphor) while keeping "I love sushi" (preference).

## 3. Failed Attempts & Lessons Learned

### 3.1 Pure Segmentation
*   **Attempt**: We tried using semantic topic segmentation to chunk conversations before extraction.
*   **Failure**: The segments were often too coarse or too fine, leading to context fragmentation.
*   **Pivot**: Turn-based grouping proved to be a more robust and lightweight proxy for "topic" in the context of fact extraction.

### 3.2 GLiNER-Only Pipeline
*   **Attempt**: Relying 100% on GLiNER for candidate generation.
*   **Failure**: High recall but low precision. It generated too many noise candidates that clogged the downstream filters.
*   **Pivot**: Using GLiNER only as a signal/feature, but relying on LFM-0.7b for the primary extraction logic.

## 4. Final Pipeline Flow

1.  **Ingest**: `ChatDBWatcher` detects new messages.
2.  **Group**: Messages are grouped into Turns (User/Contact).
3.  **Resolve**: Identities are resolved via Address Book.
4.  **Extract**: LFM-0.7b generates candidate facts (JSON/Bullets).
5.  **Verify**: NLI model checks candidates against source text.
6.  **Store**: Validated facts are saved to `contact_facts` with `relationship_reasoning`.

## 5. Future Work

*   **Temporal Resolution**: Better handling of "used to" vs "currently".
*   **Conflict Resolution**: improved logic for merging contradictory facts (e.g., "moved to NY" vs "lives in SF").
