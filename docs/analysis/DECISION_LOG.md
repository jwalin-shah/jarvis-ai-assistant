# Decision Log

**Last Updated**: 2026-02-01

This document tracks decisions that need to be made, are pending, or have been made.

---

## Pending Decisions

### ~~D1: Default Model Selection~~ → RESOLVED (see M12)

---

### ~~D2: Cloud Fallback Implementation~~ → REJECTED

**Decision**: No cloud fallback. JARVIS is local-first only.

**Rationale**: Privacy is core to the product. Cloud defeats the purpose.

---

### ~~D3: API Router Test Coverage~~ → RESOLVED

**Status**: Router tests exist:
- `tests/unit/test_router.py` (1769 lines)
- `tests/integration/test_api*.py` (4 files)

---

### D4: iMessageSender Deprecation

**Context**: CLAUDE.md marks it as experimental/unreliable. Apple's AppleScript restrictions make it fragile.

**Options**:
| Option | Pros | Cons |
|--------|------|------|
| Remove entirely | Clean codebase | Lose write capability |
| Keep but hidden | Available if needed | Technical debt |
| Invest in fixing | Full functionality | High effort, may break with updates |

**Recommendation**: Keep but mark deprecated, remove in v2.

**Blocked by**: None

---

### D5: Template Count Expansion

**Context**: Currently ~75 templates. Design doc suggests 50-100 is target. Coverage benchmark needed to validate.

**Options**:
| Option | Effort | Impact |
|--------|--------|--------|
| Keep current ~75 | None | Unknown coverage |
| Expand to 100+ | 1-2 weeks | May improve coverage |
| Mine from real queries | 1 week | Data-driven expansion |

**Recommendation**: Run coverage benchmark first, then decide.

**Blocked by**: Coverage benchmark results

---

## Made Decisions

### M1: No Fine-Tuning

**Decision**: Use RAG and few-shot prompting only, no fine-tuning.

**Rationale**: Gekhman et al. (EMNLP 2024) shows fine-tuning on new knowledge increases hallucinations.

**Consequences**: Style matching is less precise but hallucination risk is not increased.

**Status**: IMPLEMENTED (no fine-tuning code in codebase)

---

### M2: Three-Tier Memory Modes

**Decision**: Support FULL/LITE/MINIMAL modes with automatic detection.

**Rationale**: 8GB is aspirational but math doesn't work for full functionality.

**Consequences**: More complex code paths but viable across hardware range.

**Status**: IMPLEMENTED (`core/memory/controller.py`)

---

### M3: Template-First Architecture

**Decision**: Match requests to templates first, generate only when no match.

**Rationale**: Generation is expensive (memory, latency) and risky (hallucination).

**Consequences**: Better latency and quality for common cases. Requires template coverage investment.

**Status**: IMPLEMENTED (`models/templates.py`)

---

### M4: Read-Only iMessage Access

**Decision**: Read-only access with schema detection and fallback.

**Rationale**: chat.db access is fragile; Apple changes schema between releases.

**Consequences**: Cannot write to iMessage (acceptable for v1). Resilient to schema changes.

**Status**: IMPLEMENTED (`integrations/imessage/reader.py`)

---

### M5: MLX Over Ollama

**Decision**: Use MLX framework directly instead of Ollama wrapper.

**Rationale**: Better memory control on Apple Silicon, explicit Metal cache clearing.

**Consequences**: More code complexity but precise memory management.

**Status**: IMPLEMENTED (`models/loader.py`)

---

### M6: Sentence-Transformers for Embeddings

**Decision**: Use all-MiniLM-L6-v2 instead of heavier models like GLiNER.

**Rationale**: ~100MB vs 800MB-1.2GB footprint.

**Consequences**: Sufficient for template matching with much lower memory.

**Status**: IMPLEMENTED (`models/templates.py`)

---

### M7: Unified Error Hierarchy

**Decision**: Create centralized error hierarchy with error codes.

**Rationale**: Consistent error handling across CLI and API.

**Consequences**: All errors map to appropriate HTTP status codes.

**Status**: IMPLEMENTED (`jarvis/errors.py`, `api/errors.py`)

---

### M8: Centralized Prompt Registry

**Decision**: Single source of truth for all prompts in `jarvis/prompts.py`.

**Rationale**: Prevents prompt drift and enables versioning.

**Consequences**: All prompts must be defined in one place.

**Status**: IMPLEMENTED (`jarvis/prompts.py`)

---

### M9: SVM over k-NN for Classifiers

**Decision**: Use SVM classifiers instead of k-NN (DA classifier) for both trigger and response classification.

**Rationale**:
- SVM achieves 82% macro F1 vs ~70% for k-NN
- Faster inference (single model forward pass vs k-nearest search)
- Per-class thresholds enable precision/recall tuning

**Consequences**:
- Removed DA classifier code and dependencies
- Centroids still used for structural hint verification (loaded from cache)
- Training scripts: `scripts/train_trigger_classifier.py`, `scripts/train_response_classifier.py`

**Status**: IMPLEMENTED (`jarvis/trigger_classifier.py`, `jarvis/response_classifier.py`)

---

### M10: K-means over HDBSCAN for Topic Clustering

**Decision**: Use K-means (sklearn) for topic clustering in embedding profiles, remove HDBSCAN from core dependencies.

**Rationale**:
- Topics are informational only (not critical to reply generation)
- K-means is simpler and faster
- HDBSCAN adds heavy dependency for minimal benefit
- Fixed cluster count (5) is acceptable for topic discovery

**Consequences**:
- HDBSCAN moved to optional `[benchmarks]` dependency
- Removed `jarvis/cluster.py` (HDBSCAN response clustering)
- Topic clustering in `jarvis/embedding_profile.py` uses K-means

**Status**: IMPLEMENTED

---

### M11: Hybrid Classifier Architecture

**Decision**: Three-layer hybrid approach: Structural patterns → Centroid verification → SVM fallback.

**Rationale**:
- Structural patterns (regex) catch high-confidence cases instantly (~11%)
- Centroid verification adds semantic check without full classification
- SVM handles ambiguous cases with 82% accuracy

**Consequences**: Fast for common cases, accurate for edge cases.

**Status**: IMPLEMENTED (`jarvis/response_classifier.py`, `jarvis/trigger_classifier.py`)

---

### M12: Default Model: LFM-2.5-1.2B-Instruct-4bit

**Decision**: Use LFM-2.5-1.2B-Instruct-4bit as default model.

**Rationale**:
- Small footprint (~1GB) works on 8GB RAM
- 4-bit quantization balances quality vs memory
- Sufficient quality for reply suggestions

**Consequences**: Users with more RAM can select larger models via config.

**Status**: IMPLEMENTED (default in `CLAUDE.md`, `models/registry.py`)

---

## Decision Template

```markdown
### Dxx: [Title]

**Context**: [Why this decision is needed]

**Options**:
| Option | Pros | Cons |
|--------|------|------|
| Option 1 | ... | ... |
| Option 2 | ... | ... |

**Recommendation**: [What we suggest]

**Blocked by**: [What info/decision is needed first]
```
