# Reply Generation Improvement Plan

**Created**: 2026-01-29
**Updated**: 2026-01-29 (Added research findings + revised strategy)
**Mode**: Draft assistance (generate suggestions for human review)
**Primary Goal**: 70% "send as-is or minor edit" rate on human evaluation

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Research Findings & Reality Check](#2-research-findings--reality-check)
3. [Pipeline Deep Dive](#3-pipeline-deep-dive)
4. [Evaluation Strategy (Revised)](#4-evaluation-strategy-revised)
5. [Fine-Tuning Roadmap (Two-Stage)](#5-fine-tuning-roadmap-two-stage)
6. [Implementation Priority](#6-implementation-priority)
7. [Success Metrics](#7-success-metrics)
8. [References](#8-references)

---

## 1. Current State Assessment

### 1.1 What We Have

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **Contact Profiler** | ✅ Complete | `scripts/profile_contacts.py` | 277 contacts profiled with LLM classification |
| **Embedding Store** | ✅ Mature | `core/embeddings/store.py` | FAISS + BM25 hybrid, disk persistence |
| **Style Analyzer** | ✅ Good | `core/generation/style_analyzer.py` | Per-chat style detection |
| **Reply Generator** | ⚠️ Needs Integration | `core/generation/reply_generator.py` | Missing profile integration |
| **Smart Prompter V2** | ✅ Better | `core/generation/smart_prompter_v2.py` | Hybrid retrieval (60% input + 40% response style) |
| **Few-shot Retriever V2** | ✅ Better | `core/generation/fewshot_retriever_v2.py` | Response embeddings |
| **Golden Test Set** | ✅ Exists | `results/test_set/test_data.jsonl` | 200 examples |
| **Model (LFM2-2.6B-Exp)** | ✅ Best | `core/models/registry.py` | 88%+ IFEval, 72.5% win rate |

### 1.2 Current Performance

| Metric | V1 Prompter | V2 Prompter | Target |
|--------|-------------|-------------|--------|
| Avg Semantic Similarity | 0.606 | 0.622 | **0.75** |
| Win Rate vs Baseline | 72.5% | 43% vs V1 | N/A |
| Perfect Matches (>0.9) | ~15% | ~15% | **30%** |
| Poor Results (<0.4) | ~10% | ~10% | **<5%** |

### 1.3 Contact Profile Distribution

```
Total Profiled: 277
├── Friend: 145 (53%)
├── Group: 69 (25%)
├── Best Friend: 45 (16%)
├── Family: 15 (5%)
└── Professional: 3 (1%)

Activity:
├── Daily: 8
├── Frequent: 14
├── Occasional: 35
└── Inactive: 220
```

---

## 2. Research Findings & Reality Check

### 2.1 The Semantic Similarity Ceiling

Research (arXiv:2509.14543, Sept 2025) shows LLMs fundamentally struggle with informal writing style:

> "LLMs excel in formal and structured genres (news, email), but struggle with informal, highly idiosyncratic creative domains (blogs, forums), even with multiple demonstrations."

**Key findings:**
- Authorship verification drops from 95-97% (news) to 19-21% (blogs/forums)
- AI-generated text remains "readily detectable" even with few-shot examples
- "Including more writing examples affects metrics very little" - diminishing returns beyond 5

**Implication**: The 0.75 semantic similarity target may be unrealistic for casual iMessage text. The 0.62 we're getting might be closer to the ceiling than expected.

### 2.2 StyleTunedLM: The State of the Art (Sept 2024)

[StyleTunedLM](https://arxiv.org/html/2409.04574v1) achieved 87.9% author attribution (vs 68% few-shot) using:

1. **Named Entity Masking**: Mask attention on proper nouns during training so model learns *style patterns* not *content memorization*
2. **LoRA Adapter Merging**: Train style adapter + instruction adapter separately, merge via weight concatenation
3. **Next-token prediction**: Simple objective on raw author text, not instruction tuning

**Hyperparameters:**
- Learning rate: 5e-5
- Epochs: 3
- Batch size: 4
- Max tokens: 256

### 2.3 What Successful Projects Do Differently

**Moltbot/Clawdbot** (68K GitHub stars):
- **Doesn't try to perfectly mimic** - aims for "useful proxy"
- **Human-in-loop** - drafts for approval, not auto-send
- **Proactive** - suggests what you should reply to, not just how

**Poke** ($15M seed, iMessage-native):
- Focuses on **what needs to be said** (reminders, logistics) not style
- Calendar/email integration for context
- Actionable nudges over perfect replies

### 2.4 Revised Success Definition

**Old**: 0.75 semantic similarity to gold responses
**New**: 70% "send as-is or minor edit" rate on human evaluation

| Tier | Definition | Target |
|------|------------|--------|
| **Tier 1** | Send as-is | 30-40% |
| **Tier 2** | Minor edit needed | 30-40% |
| **Tier 3** | Major rewrite needed | 20-30% |
| **Tier 4** | Unusable | <10% |

---

## 3. Pipeline Deep Dive

### 3.1 Current Flow (with status)

```
REQUEST: Generate reply for chat_id + last_message
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 0: Coherence Filter                                    │
│ Status: ⚠️ NEEDS WORK                                        │
│ Problem: Simple heuristics miss semantic topic breaks       │
│ Fix: Use embedding similarity to detect topic boundaries    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 0.5: Template Match (Fast Path)                        │
│ Status: ✅ GOOD                                              │
│ Timing: ~2-5ms                                              │
│ Note: Skips LLM for common patterns (threshold 0.75)        │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Style Analysis                                      │
│ Status: ✅ GOOD                                              │
│ Cached: Per chat_id (in-memory)                             │
│ Metrics: word_count, emoji, caps, abbreviations, punct      │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Context Analysis                                    │
│ Status: ⚠️ NEEDS WORK                                        │
│ Problem: Pure rule-based, no ML, misses nuance              │
│ Fix: Use LLM-based intent classification (like profiler)    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Past Reply Retrieval                                │
│ Status: ⚠️ SIMILARITY SCORING NEEDS WORK                     │
│ Current: FAISS + time-weighting                             │
│ Problem: Generic embedding model, no style matching         │
│ Fix: See Section 3                                          │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Personal Template Match                             │
│ Status: ✅ GOOD                                              │
│ Note: Smart shortcut when past replies are consistent       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Contact Profile Retrieval                           │
│ Status: ✅ NEWLY COMPLETE - NEEDS INTEGRATION                │
│ Data: relationship, temporal, reciprocity, topics, themes   │
│ Fix: Wire into prompt builder                               │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Build Prompt                                        │
│ Status: ⚠️ NEEDS WORK                                        │
│ Problem: Hardcoded format, not model-specific               │
│ Problem: Doesn't use contact profile                        │
│ Fix: Model-specific templates + profile injection           │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: LLM Generation                                      │
│ Status: ✅ GOOD (with LFM2-2.6B-Exp)                         │
│ Timing: ~200-800ms                                          │
│ Future: Fine-tune for even better results                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 8-9: Post-Processing + Repetition Filter               │
│ Status: ✅ GOOD                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Diagram

```
                                    iMessage DB (chat.db)
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │   MessageReader       │
                               │   (read-only access)  │
                               └───────────┬───────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
          ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
          │ Embedding Store │    │ Contact Profiler│    │ Reply Generator │
          │ (FAISS + SQLite)│    │ (LLM classify)  │    │ (orchestrator)  │
          └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
                   │                      │                      │
                   │              ┌───────┴───────┐              │
                   │              │               │              │
                   │              ▼               │              │
                   │    ┌─────────────────┐      │              │
                   │    │ Profile Cache   │      │              │
                   │    │ (JSON file)     │◄─────┘              │
                   │    └────────┬────────┘                     │
                   │             │                              │
                   └─────────────┼──────────────────────────────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │ Smart Prompter  │
                       │ V2 (hybrid)     │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Model Loader    │
                       │ (LFM2-2.6B-Exp) │
                       └────────┬────────┘
                                │
                                ▼
                          Generated Reply
```

---

## 4. Evaluation Strategy (Revised)

### 4.1 Human Evaluation Pipeline (NEW - Priority)

```bash
# Run human evaluation
python scripts/human_eval.py \
    --test-set results/test_set/test_data.jsonl \
    --model lfm2-2.6b-exp \
    --samples 50 \
    --output results/evaluation/human_eval.json
```

**Scoring rubric:**
| Score | Label | Definition |
|-------|-------|------------|
| 4 | Send as-is | Would send immediately without changes |
| 3 | Minor edit | Small tweak (word choice, punctuation) |
| 2 | Major edit | Right idea, needs significant rewriting |
| 1 | Unusable | Wrong tone, content, or completely off |

**Target distribution:**
- Score 4: 30-40%
- Score 3: 30-40%
- Score 2: 20-30%
- Score 1: <10%

### 4.2 Current Test Set

- **Location**: `results/test_set/test_data.jsonl`
- **Size**: 200 examples
- **Format**: `{contact, input (their msg), gold (your actual reply)}`
- **Stratification**: By relationship type and contact cluster

### 4.3 Automated Metrics (Secondary)

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| **Semantic Similarity** | 0.62 | 0.65-0.68 | Ceiling ~0.68 for informal text |
| **Style Score** | 0.87 | 0.90 | Length + punct + markers match |
| **Exact Match Rate** | ~5% | 10% | Realistic for casual text |
| **Poor Results (<0.4)** | ~10% | <5% | Critical failures |

**Note**: Research shows semantic similarity plateaus at 0.65-0.68 for informal text. Human evaluation is the primary success metric.

### 4.4 Evaluation Script Improvements

```bash
# Current evaluation
python scripts/evaluate_smart_prompts.py

# Proposed: Multi-metric evaluation
python scripts/evaluate_replies.py \
    --test-set results/test_set/test_data.jsonl \
    --model lfm2-2.6b-exp \
    --metrics semantic,style,exact,combined \
    --output results/evaluation/latest.json
```

### 4.5 Error Analysis Categories

| Category | Description | Action |
|----------|-------------|--------|
| **Wrong Intent** | Answers question when should acknowledge | Improve context analysis |
| **Wrong Style** | Too formal/casual for relationship | Use contact profile |
| **Too Generic** | "Sounds good" when specific reply needed | Better few-shot retrieval |
| **Hallucination** | Mentions things not in context | Reduce temperature |
| **Too Long/Short** | Length mismatch with user's style | Style constraints |
| **Missing Info** | Doesn't include necessary details | Better context window |

### 4.6 Golden Test Set Expansion

Current: 200 examples
Target: 500+ examples

**Stratification:**
- By relationship type (family, friend, professional)
- By message type (question, statement, reaction, logistics)
- By your response type (short, medium, long)
- By time of day (morning, afternoon, evening)

---

## 5. Fine-Tuning Roadmap (Two-Stage)

### 5.1 Architecture: Base Voice + Relationship Adapters

```
┌────────────────────────────────────────────────────────────┐
│  Base Model (LFM2-2.6B-Exp)                                │
└─────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│  Stage 1: "Your Voice" LoRA                                │
│  - Trained on ALL your messages (10K+ pairs)               │
│  - Learns: lowercase, abbreviations, emoji, length, etc.   │
│  - Uses named entity masking (StyleTunedLM technique)      │
│  - This IS you, baseline for everyone                      │
└─────────────────────────┬──────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
     ┌───────────┐ ┌───────────┐ ┌───────────┐
     │ Family    │ │ Friends   │ │ Default   │
     │ Adapter   │ │ Adapter   │ │ (none)    │
     │ (+warmth) │ │ (+casual) │ │           │
     └───────────┘ └───────────┘ └───────────┘
```

**Why two-stage?**
- Base adapter captures YOUR general patterns (most of the signal)
- Per-relationship adapters are small tweaks, not full retraining
- Only need 3-5 relationship adapters, not 277 per-contact adapters

### 5.2 Stage 1: Base Voice Training

```python
# Extract ALL your replies for base voice training
training_data = []
for contact in contacts:
    messages = get_messages(contact.chat_id)
    for i, msg in enumerate(messages):
        if not msg.is_from_me and i+1 < len(messages):
            next_msg = messages[i+1]
            if next_msg.is_from_me:
                training_data.append({
                    "input": msg.text,
                    "response": next_msg.text,
                    # NO contact/relationship - this learns YOUR style globally
                })

# Target: 10,000+ pairs from ALL contacts
```

**Named Entity Masking (StyleTunedLM technique):**
- During training, mask attention on proper nouns (names, places)
- Model learns style patterns, not content memorization
- Prevents overfitting to specific conversations

**Hyperparameters (from StyleTunedLM):**
- Learning rate: 5e-5
- Epochs: 3
- Batch size: 4
- LoRA rank: 16-32
- Max tokens: 256

### 5.3 Stage 2: Relationship Adapters (Optional)

| Adapter | Train On | Learns |
|---------|----------|--------|
| **Family** | Messages to parents/siblings | +warmth, formal names, more detail |
| **Close Friends** | Messages to best friends | +casual, inside jokes, abbreviations |
| **Professional** | Work contacts | +formal, proper punctuation |

These are **small LoRA adapters stacked on the base** - merge using StyleTunedLM's weight concatenation.

### 5.4 DPO Training Data Format

```python
# Direct Preference Optimization pairs
{
    "prompt": "them: want to grab dinner tomorrow?\nme:",
    "chosen": "yeah down",           # Your actual reply
    "rejected": "Yes, I would love to have dinner with you tomorrow!"  # Generic
}
```

**Generating "rejected" examples:**
1. Use base model (pre-fine-tune) to generate responses
2. Or use generic formal responses
3. Or use other people's style (from public datasets)

### 5.5 Fine-Tuning Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Collection                                          │
│    - Extract ALL (their_msg, your_reply) pairs              │
│    - Filter: min 3 chars, no reactions, no system msgs      │
│    - Apply named entity masking to training data            │
│    - Split: 80% train, 10% val, 10% test                   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Stage 1: Base Voice LoRA                                │
│    - Train on ALL messages (no relationship labels)         │
│    - Objective: next-token prediction                       │
│    - LoRA rank: 16-32, lr: 5e-5, epochs: 3                 │
│    - Output: base_voice_adapter.safetensors                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Stage 2: DPO Preference Tuning                          │
│    - Generate "rejected" examples with base model           │
│    - Train DPO on preference pairs                          │
│    - Steer away from generic, toward YOUR style             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. (Optional) Relationship Adapters                         │
│    - Train small LoRA per relationship type                 │
│    - Merge with base adapter at inference time              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Human Evaluation                                         │
│    - 50 sample A/B test: fine-tuned vs base                │
│    - Score: send as-is / minor edit / major edit / unusable│
│    - Target: 70% in top two tiers                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Priority

### Phase 1: Quick Wins (1-2 days)

| Task | Impact | Effort | File |
|------|--------|--------|------|
| 1.1 Integrate contact profiles into prompt | High | Low | `reply_generator.py` |
| 1.2 Add exact-match bonus for short msgs | Medium | Low | `store.py` |
| 1.3 Add style score to evaluation | Medium | Low | `evaluate_*.py` |

### Phase 2: Retrieval Improvements (3-5 days)

| Task | Impact | Effort | File |
|------|--------|--------|------|
| 2.1 Implement multi-metric scoring | High | Medium | `store.py`, `similarity.py` |
| 2.2 Add response style matching | High | Medium | `fewshot_retriever_v2.py` |
| 2.3 Dynamic BM25/vector weights | Medium | Low | `store.py` |

### Phase 3: Evaluation Infrastructure (2-3 days)

| Task | Impact | Effort | File |
|------|--------|--------|------|
| 3.1 Expand test set to 500 examples | High | Medium | `scripts/create_test_set.py` |
| 3.2 Build error analysis tool | Medium | Medium | `scripts/analyze_errors.py` |
| 3.3 Add automated regression tests | Medium | Medium | `tests/test_generation.py` |

### Phase 4: Model Fine-Tuning (1-2 weeks)

| Task | Impact | Effort | File |
|------|--------|--------|------|
| 4.1 Collect training data (10K pairs) | High | Medium | `scripts/collect_training_data.py` |
| 4.2 Generate DPO preference pairs | High | Medium | `scripts/generate_preferences.py` |
| 4.3 Fine-tune with LoRA + DPO | Very High | High | New fine-tuning scripts |
| 4.4 Evaluate and deploy | High | Medium | Evaluation + registry update |

---

## 7. Success Metrics

### 7.1 Primary Metrics (Human Evaluation)

| Metric | Current | Phase 1 | Phase 2 | Phase 4 |
|--------|---------|---------|---------|---------|
| **Send as-is rate** | ~15% | 20% | 30% | **40%** |
| **Minor edit rate** | ~25% | 30% | 35% | **35%** |
| **Major rewrite rate** | ~40% | 35% | 25% | **20%** |
| **Unusable rate** | ~20% | 15% | 10% | **<5%** |
| **Combined acceptable (tiers 1+2)** | ~40% | 50% | 65% | **75%** |

### 7.2 Secondary Metrics (Automated)

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Semantic Similarity | 0.62 | 0.65-0.70 | Ceiling ~0.68 for informal text |
| Style Score | 0.87 | 0.90 | Length + punct + markers |
| Functional Correctness | N/A | 95% | Dates, times, logistics right |
| Generation Latency (p50) | ~400ms | <500ms | Keep fast |
| Generation Latency (p99) | ~1000ms | <1500ms | Acceptable |

### 7.3 Qualitative Goals

- [ ] Replies "sound like me" in blind test (6/10 correct attribution - realistic target)
- [ ] Relationship-appropriate tone (formal for professional, casual for friends)
- [ ] **No hallucinations about commitments** (critical failure mode)
- [ ] Handles logistics correctly (times, places, confirmations)
- [ ] User edits suggestions, not rewrites from scratch

---

## Appendix: Key Files Reference

### Core Pipeline
- `core/generation/reply_generator.py` - Main orchestrator
- `core/generation/prompts.py` - Prompt templates
- `core/generation/style_analyzer.py` - Style detection
- `core/generation/context_analyzer.py` - Intent/context

### Embedding & Retrieval
- `core/embeddings/store.py` - FAISS + hybrid search
- `core/embeddings/similarity.py` - Similarity functions
- `core/generation/fewshot_retriever_v2.py` - Few-shot with response matching

### Smart Prompting
- `core/generation/smart_prompter.py` - V1 prompter
- `core/generation/smart_prompter_v2.py` - V2 with hybrid retrieval

### Evaluation
- `scripts/evaluate_smart_prompts.py` - Main evaluation
- `scripts/evaluate_v2_prompts.py` - V1 vs V2 comparison
- `results/test_set/test_data.jsonl` - Golden test set

### Contact Profiling
- `scripts/profile_contacts.py` - Comprehensive profiler
- `results/profiles/contact_profiles.json` - Profile cache

### Models
- `core/models/loader.py` - MLX model loading
- `core/models/registry.py` - Available models

---

## 8. References

### Research Papers

1. **StyleTunedLM** (Sept 2024) - [arXiv:2409.04574](https://arxiv.org/html/2409.04574v1)
   - Key technique: Named entity masking + LoRA adapter merging
   - Result: 87.9% author attribution vs 68% few-shot

2. **"Catch Me If You Can? Not Yet"** (Sept 2025) - [arXiv:2509.14543](https://arxiv.org/html/2509.14543v1)
   - Finding: LLMs struggle with informal, idiosyncratic writing styles
   - Implication: Semantic similarity ceiling ~0.65-0.68 for casual text

### Practical Implementations

3. **Moltbot/Clawdbot** - [TechCrunch overview](https://techcrunch.com/2026/01/27/everything-you-need-to-know-about-viral-personal-ai-assistant-clawdbot-now-moltbot/)
   - Philosophy: Useful proxy > perfect mimic
   - Key: Human-in-loop, proactive features

4. **Poke** ($15M seed) - [Launch announcement](https://techfundingnews.com/poke-launches-15m-seed-imessage-ai-assistant/)
   - Focus: What to say (context) over how to say it (style)
   - Integration: Calendar, email, actionable nudges

### Fine-Tuning Guides

5. **DPO Fine-tuning** - [Cerebras blog](https://www.cerebras.ai/blog/fine-tuning-language-models-using-direct-preference-optimization)
6. **Unsloth LoRA Guide** - [Documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
