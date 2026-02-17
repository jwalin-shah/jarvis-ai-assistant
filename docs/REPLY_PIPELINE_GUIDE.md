# Reply Pipeline Architecture Guide

> **Last Updated:** 2026-02-17  
> **Status:** Production - RAG-Enhanced / KV-Cache Optimized

---

## Overview

This guide covers the architecture for generating text message replies that match your personal style while remaining fast and reliable.

**Core Philosophy:** "Just Texting." The model is strictly constrained to act as a person texting from their phone, never as an AI assistant.

---

## Architecture Components

### 1. Unified Reply Service (`jarvis/reply_service.py`)

The `ReplyService` orchestrates the entire generation flow. It uses a **stateless, request-isolated** model where every message gets a fresh `MessageContext` to prevent state leakage between conversations.

### 2. Context Isolation & Security

To ensure that personal data (like names or facts) from one chat doesn't "trip" into another:
- **Fresh Context:** A new `MessageContext` object is instantiated for every RPC call.
- **Metadata Sanitization:** The `metadata` dictionary is populated fresh from the database per request.
- **Stateless RPC:** The backend does not maintain session state; each `chat_id` is treated as a unique, independent entry point.

### 3. Prompt Engineering & Performance

#### The "Just Texting" Universal Prompt
Research showed that category-specific prompts (e.g., "They asked a question") actually hurt quality. We now use a **Universal Prompt** that anchors the model in a human persona.

**Universal System Prefix (Constant):**
```
You are texting from your phone. Reply naturally. 
Be brief (1-2 sentences), casual, like a real person.
```
*Note: This prefix is kept 100% constant to enable KV-Cache reuse, dropping generation time from ~3s to <450ms.*

#### RAG & Fact Anchoring
To prevent hallucinations (like the "Neuropathy" issue), the model is "anchored" with:
- **Facts:** Relevant personal knowledge retrieved from `vec_facts`.
- **Examples:** Past similar exchanges retrieved via Hybrid Search (Semantic + Keyword).
- **Personalization:** Explicitly stating the recipient's name in the instruction field.

---

## Performance & Resource Management

### Disabled Background Tasks
To prevent 90s timeouts and GPU resource contention:
- **Fact Extraction:** Background fact extraction during focus/chat events is **disabled**. Extraction is now an offline/batch process.
- **Graph Fetching:** Blocking relationship graph construction is disabled in the critical path.

### Performance Targets
| Operation | Target | Actual (Optimized) |
|-----------|--------|-------------------|
| KV-Cached Gen | <500ms | **~420ms** |
| Context Search| <100ms | **~60ms** |
| Total Pipeline| <1s    | **~600ms** |

---

## Optimization Path: DSPy & MIPROv2

We use DSPy for **offline optimization** only. This gives us high-quality instructions without the runtime cost.

**The Workflow:**
1.  **Define Signature:** `Context + History + Name -> Reply`.
2.  **Gold Dataset:** Curate 50-100 of your best real-world text replies.
3.  **Optimize:** Run `MIPROv2` to find the best instructions and few-shot examples.
4.  **Deploy:** Paste the optimized instruction strings into `jarvis/prompts/constants.py`.

---

## Troubleshooting "Tripping"

If the AI starts acting like an assistant:
1. **Check Logs:** Inspect `final_prompt` in `reply_logs` (metadata field).
2. **Verify Names:** Ensure `contact_name` is correctly resolved in the prompt.
3. **Instruction Check:** Ensure the `{instruction}` placeholder in `SIMPLE_REPLY_PROMPT` is populated.

---

## Related Documents

- [Categorization Ablation Findings](./research/CATEGORIZATION_ABLATION_FINDINGS.md) - Why we use a universal prompt.
- [LLM Judge Evaluation](./research/LLM_JUDGE_EVALUATION.md) - How we measure quality.
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Deep dive into system internals.
