# Multi-Agent Codebase Review - Prioritized Findings

## Priority 1: High Impact (Core Routing & Classification)

| # | Issue | File:Line | Assigned To | Rationale |
|---|-------|-----------|-------------|-----------|
| 1 | Context-dependent check after FAISS (wasted latency) | `router.py:806-838` | **Codex** | Architecture change, needs careful refactoring |
| 2 | Intent classifier underutilized | `router.py:767-779` | **Claude** | Requires understanding full routing semantics |
| 3 | Insufficient recall for rare response types | `retrieval.py:228` | **Gemini** | Already proposed fix in review |
| 4 | Coherence scoring too late | `router.py:987-994` | **Codex** | Performance optimization |

## Priority 2: Medium Impact (Classifier Accuracy)

| # | Issue | File:Line | Assigned To | Rationale |
|---|-------|-----------|-------------|-----------|
| 5 | Pattern ordering conflicts (ACK vs REACT) | `response_classifier_v2.py:350+` | **Kimi** | Pattern analysis & reordering |
| 6 | No diversity filtering in retrieval | `retrieval.py:245` | **Gemini** | Proposed fix in review |
| 7 | Hardcoded cosine/entity weights | `topic_discovery.py:102` | **OpenCode** | Proposed fix in review |
| 8 | Token-level entity matching false positives | `topic_discovery.py:195-196` | **OpenCode** | Entity handling expertise |

## Priority 3: Medium Impact (Topic Segmentation)

| # | Issue | File:Line | Assigned To | Rationale |
|---|-------|-----------|-------------|-----------|
| 9 | Fixed min_cluster_size=5 | `topic_discovery.py:454` | **OpenCode** | Already has adaptive solution |
| 10 | No TF-IDF for topic keywords | `topic_discovery.py:641-673` | **Gemini** | NLP/IR expertise |
| 11 | No topic merging | `topic_discovery.py:580-590` | **OpenCode** | Clustering expertise |
| 12 | Inconsistent temporal decay | `index.py:440` vs `retrieval.py:60` | **Gemini** | Retrieval scope |

## Priority 4: Code Quality (Cleanup)

| # | Issue | File:Line | Assigned To | Rationale |
|---|-------|-----------|-------------|-----------|
| 13 | Unused `RouteResult` dataclass | `router.py:213-233` | **Kimi** | Dead code removal |
| 14 | Unused `IndexNotAvailableError` | `router.py:202-206` | **Kimi** | Dead code removal |
| 15 | Duplicate `_REFERENCE_WORDS` | `router.py:138-153` | **Kimi** | Consolidation |
| 16 | Redundant `if thread` checks | `router.py:534-537` | **Kimi** | Cleanup |

---

## Agent Workload Summary

| Agent | Tasks | Focus |
|-------|-------|-------|
| **Codex** | #1, #4 | Router architecture & performance |
| **Claude** | #2 | Intent integration |
| **Gemini** | #3, #6, #10, #12 | Retrieval & NLP |
| **OpenCode** | #7, #8, #9, #11 | Topic segmentation |
| **Kimi** | #5, #13-16 | Patterns & cleanup |
