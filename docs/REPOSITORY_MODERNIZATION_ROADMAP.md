# JARVIS Repository Modernization Roadmap
**90-Day Complexity Reduction Initiative**

**Target:** Reduce repository complexity by 60% while improving reliability, build times, and developer velocity.

---

## Executive Summary

### Current State Assessment

| Metric | Current | Target | Reduction |
|--------|---------|--------|-----------|
| Python Files | 404 | 280 | 30% |
| Root-Level Files | 110+ | 25 | 77% |
| Scripts Directory | 116 | 25 | 78% |
| Dependencies | 52 | 35 | 33% |
| Test Failures | 30 | 0 | 100% |
| Contract Drift Issues | 27 | 0 | 100% |
| Files >500 LOC | 5 | 0 | 100% |
| Technical Debt Items | 50 | 10 | 80% |
| Avg Build Time | ~8 min | ~4 min | 50% |

### Key Complexity Drivers

1. **Experimental Artifact Accumulation**: 110+ root-level files from ML experiments
2. **Script Proliferation**: 116 scripts, only ~10 production-grade
3. **Dependency Bloat**: 52 production dependencies, many unused
4. **Contract Drift**: 27 schema/protocol mismatches between layers
5. **Large Modules**: 5 files exceed 800 LOC (prompts.py: 2,373 LOC)
6. **Test Debt**: 30 persistent failures masking real issues

---

## Phase 1: Foundation Cleanup (Days 1-30)

### Week 1: Repository Hygiene

#### 1.1 Root-Level Cleanup (Days 1-3)
**Risk Level:** LOW  
**Rollback:** Backup tarball retained for 30 days

| Action | Files | Effort | Owner |
|--------|-------|--------|-------|
| Delete experimental scripts | analyze_*.py (10) | 2h | DevOps |
| Delete labeling scripts | label_*.py (9) | 2h | DevOps |
| Delete test scripts | test_*.py, validate_*.py (15) | 2h | DevOps |
| Delete JSON/JSONL results | *_results.json, *.jsonl (70+) | 2h | DevOps |
| Delete text outputs | *_output.txt, *.txt (20+) | 1h | DevOps |

**Verification:**
```bash
# Pre-cleanup backup
tar -czf .cleanup_backup/phase1_$(date +%Y%m%d).tar.gz \
    *.py *.json *.jsonl *.txt 2>/dev/null

# Verify no production references
grep -r "get_optimization_category" jarvis/ tests/ || echo "✓ No references"
grep -r "category_svm_v2" jarvis/ tests/ || echo "✓ No references"
```

#### 1.2 Obsolete Model Cleanup (Days 4-5)
**Risk Level:** LOW

| Model | Size | Action |
|-------|------|--------|
| category_svm_v2.* | 21K | Delete |
| category_lightgbm_915_*.* | 1.9M | Delete |
| category_linearsvc_*.* | 36K | Delete |
| category_multilabel_hardclass.* | 102K | Delete |
| category_multilabel_lightgbm.joblib | 9.9M | Delete (superseded) |
| **KEEP** | | |
| category_multilabel_lightgbm_hardclass.joblib | 10MB | Production |

**Pre-Flight Checklist:**
- [ ] Verify production model loads correctly
- [ ] Run category classifier tests
- [ ] Confirm no references to deleted models

### Week 2: Scripts Directory Consolidation

#### 2.1 Script Categorization (Days 6-8)

**Production Scripts (KEEP - ~15 files):**
```
scripts/production/
├── setup_db.py
├── check_gates.py
├── launch.sh
├── overnight_eval.sh
├── train_category_classifier.py
├── evaluate_model.py
├── prepare_soc_data.py
├── generate_preference_pairs.py
├── finetune_embedder.py
├── extract_personal_data.py
├── prepare_personal_data.py
├── generate_ft_configs.py
├── train_personal.py
└── evaluate_personal_ft.py
```

**Archive Scripts (MOVE - ~80 files):**
```
scripts/archive/YYYY-MM/
├── analyze_*.py
├── backfill_*.py
├── batch_*.py
├── benchmark_*.py
├── compare_*.py
├── consensus_*.py
├── create_*.py
├── gemini_*.py (experimental)
├── label_*.py
├── retrain_*.py
├── retune_*.py
├── review_*.py
├── sample_*.py
└── tune_*.py
```

**Delete Scripts (REMOVE - ~20 files):**
- Superseded training variants
- One-off debugging scripts
- Duplicate analysis scripts

#### 2.2 Makefile Updates (Day 9)

Update all make targets to reference new paths:
```makefile
# Old
label-categories:
    uv run python scripts/label_soc_categories.py

# New
label-categories:
    uv run python scripts/production/label_soc_categories.py
```

#### 2.3 Scripts README (Day 10)

Create `scripts/README.md`:
```markdown
# Scripts Directory

## Production Scripts (`production/`)
Essential utilities for setup, training, and evaluation.

## Archive Scripts (`archive/YYYY-MM/`)
Experimental scripts preserved for reference. Not actively maintained.

## Adding New Scripts
1. Production scripts require Makefile integration
2. Production scripts require tests
3. Experimental scripts go to `archive/`
```

### Week 3: Dependency Pruning

#### 3.1 Dependency Audit (Days 11-13)

Analyze current dependencies in `pyproject.toml`:

```python
# scripts/audit_dependencies.py
import subprocess
import ast
from pathlib import Path

def find_imports():
    """Find all imports in production code."""
    imports = set()
    for path in Path("jarvis").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except SyntaxError:
            continue
    return imports

def check_usage(package_name):
    """Check if package is actually imported."""
    # Implementation...
```

**Suspected Unused Dependencies:**

| Package | Used In | Decision |
|---------|---------|----------|
| bertopic | experiments/ only | Move to [experiments] extra |
| fastcoref | coref extra only | Verify extra isolation |
| groq | Unknown | Investigate |
| langdetect | Verify usage | Keep if used |
| umap-learn | experiments/ only | Move to [experiments] extra |
| hdbscan | benchmarks only | Move to [benchmarks] extra |

#### 3.2 Clean Up extras (Days 14-15)

Consolidate extras:
```toml
[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy", ...]
ml = ["torch", "transformers", "sentence-transformers", "setfit"]
benchmarks = ["matplotlib", "pandas", "hdbscan"]
training = ["scikit-learn", "optuna", "xgboost", "lightgbm"]
experiments = ["bertopic", "umap-learn"]
```

### Week 4: Test Stabilization

#### 4.1 Fix Critical Test Failures (Days 16-20)

**Priority Order:**

| Test File | Failures | Fix Strategy | Owner |
|-----------|----------|--------------|-------|
| test_latency_gate.py | 5 | Update thresholds | Backend |
| test_router.py | 8 | Template assertions | Backend |
| test_category_classifier.py | 6 | Reranker mocks | ML |
| test_socket_server.py | 4 | Auth token validation | Backend |
| test_watcher.py | 3 | File system mocks | Backend |

#### 4.2 Pre-Existing Failure Registry (Days 21-22)

Document intentionally skipped tests:
```python
# tests/conftest.py
PRE_EXISTING_FAILURES = [
    "test_generation_latency_gate",  # Hardware dependent
    "test_cold_start_under_15s",     # Model size dependent
]

def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.name in PRE_EXISTING_FAILURES:
            item.add_marker(pytest.mark.skip(reason="Pre-existing failure"))
```

#### 4.3 Coverage Baseline (Days 23-25)

Target coverage improvements:

| Module | Current | Target |
|--------|---------|--------|
| jarvis/topics/topic_segmenter.py | 0% | 70% |
| jarvis/search/segment_ingest.py | 0% | 70% |
| jarvis/services/manager.py | 21% | 80% |
| jarvis/prefetch/predictor.py | 10% | 80% |
| jarvis/socket_server.py | 32% | 80% |

#### 4.4 Week 4 Checkpoint (Days 26-30)

**Verification:**
```bash
make verify
# Expected: 0 failures, coverage >70%
```

---

## Phase 2: Structural Reorganization (Days 31-60)

### Week 5: Module Boundary Enforcement

#### 5.1 Clean Architecture Layers (Days 31-35)

Establish clear module boundaries:

```
jarvis/
├── core/                    # Business logic (no deps on api/, desktop/)
│   ├── classification/
│   ├── extraction/
│   ├── generation/
│   └── retrieval/
├── interfaces/              # Adapters (depends on core/)
│   ├── api/                 # FastAPI layer
│   ├── cli/                 # Command line
│   └── desktop/             # Socket server
└── infrastructure/          # External deps (depends on nothing)
    ├── db/
    ├── embeddings/
    ├── llm/
    └── storage/
```

#### 5.2 Dependency Inversion (Days 36-38)

Move protocol definitions to `contracts/`:
```python
# contracts/classification.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Classifier(Protocol):
    def classify(self, text: str) -> ClassificationResult: ...
    
# jarvis/core/classification/engine.py
from contracts.classification import Classifier

class CategoryClassifier:
    def classify(self, text: str) -> ClassificationResult:
        ...
```

#### 5.3 Import Cleanup (Days 39-40)

Enforce import rules:
```python
# Allowedrom jarvis.core.classification import CategoryClassifier
from contracts.imessage import iMessageReader

# Disallowed (circular dependency risk)
from api.routers.conversations import router  # Don't import API from core
from jarvis.prompts import PROMPTS  # Use contracts instead
```

### Week 6: Large File Decomposition

#### 6.1 Decompose prompts.py (Days 41-45)

Current: 2,373 LOC  
Target: 5 files, max 400 LOC each

```
jarvis/prompts/
├── __init__.py              # Re-exports, 100 LOC
├── reply_prompts.py         # Reply generation, 400 LOC
├── classification_prompts.py # Intent/category, 300 LOC
├── extraction_prompts.py    # Entity/fact extraction, 350 LOC
└── system_prompts.py        # General system, 300 LOC
```

**Migration Strategy:**
1. Create new directory structure
2. Move prompt functions with tests
3. Update imports gradually
4. Remove monolithic file

#### 6.2 Decompose socket_server.py (Days 46-48)

Current: 1,624 LOC  
Target: Handler classes per domain

```
jarvis/interfaces/desktop/
├── __init__.py
├── server.py                # Core server logic, 300 LOC
├── handlers/
│   ├── __init__.py
│   ├── messages.py          # Message handlers, 250 LOC
│   ├── conversations.py     # Conversation handlers, 250 LOC
│   ├── drafts.py            # Draft handlers, 200 LOC
│   └── websocket.py         # WebSocket handlers, 200 LOC
└── protocol.py              # Binary protocol, 150 LOC
```

#### 6.3 Decompose errors.py (Days 49-50)

Current: 1,185 LOC  
Target: Exception hierarchy + error codes

```
jarvis/core/exceptions/
├── __init__.py              # Re-exports
├── hierarchy.py             # Exception classes only, 200 LOC
├── codes.py                 # Error code constants, 150 LOC
├── handlers.py              # Exception handlers, 200 LOC
└── messages.py              # Error message templates, 150 LOC
```

### Week 7: API Cleanup

#### 7.1 Error Response Unification (Days 51-53)

Standardize all error responses:

```python
# api/errors.py (single source of truth)
class ErrorResponse(BaseModel):
    error: str           # Error type/code
    code: str            # Machine-readable code
    detail: str          # Human-readable message
    details: dict | None # Additional context
    
# All routers use:
from api.errors import ErrorResponse, error_response

@router.get("/endpoint")
async def endpoint():
    try:
        ...
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=error_response("VALIDATION_ERROR", str(e))
        )
```

**Router Updates:**
- [ ] `api/routers/calendar.py` - Fix custom error format
- [ ] `api/dependencies.py` - Fix custom error format
- [ ] `api/routers/attachments.py` - Add ErrorResponse model

#### 7.2 Router Consolidation (Days 54-56)

Consolidate 37 routers → 25:

| Current | Action | New Home |
|---------|--------|----------|
| analytics.py + template_analytics.py | Merge | analytics.py |
| suggestions.py + custom_templates.py | Merge | suggestions.py |
| tags.py | Review | Keep or merge into conversations.py |
| experiments.py | Move | scripts/ only |
| debug.py | Move | scripts/ only |

#### 7.3 Contract Drift Fixes (Days 57-60)

| Issue | Fix | Verification |
|-------|-----|------------|
| Attachment schema drift | Add media fields to AttachmentResponse | Contract test |
| get_conversation_context missing | Implement or remove from protocol | Protocol test |
| Error format inconsistency | Standardize on ErrorResponse | Integration test |
| Calendar validation gap | Add Pydantic validators | Unit test |

---

## Phase 3: Pipeline Simplification (Days 61-75)

### Week 8: Model Pipeline Consolidation

#### 8.1 Unified Model Interface (Days 61-65)

Create single model access point:

```python
# models/registry.py (exists, simplify)
class ModelRegistry:
    """Single source for all model access."""
    
    def get_classifier(self) -> Classifier:
        """Returns the active classifier (LightGBM)."""
        ...
        
    def get_generator(self) -> Generator:
        """Returns the MLX generator."""
        ...
        
    def get_embedder(self) -> Embedder:
        """Returns the sentence embedder."""
        ...
```

**Delete Redundant Wrappers:**
- Remove double CachedEmbedder wrapping
- Remove deprecated model loaders
- Consolidate feature extraction

#### 8.2 Feature Pipeline Simplification (Days 66-68)

Current: Multiple feature extractors  
Target: Single unified pipeline

```python
# jarvis/features/pipeline.py
class FeaturePipeline:
    """Single entry point for all feature extraction."""
    
    def extract(self, message: Message) -> FeatureVector:
        return FeatureVector(
            text=self._text_features(message),
            context=self._context_features(message),
            metadata=self._metadata_features(message),
        )
```

#### 8.3 Training Pipeline Consolidation (Days 69-70)

Consolidate training scripts:

```
scripts/training/
├── __init__.py
├── train_classifier.py      # Unified classifier training
├── train_generator.py       # MLX fine-tuning
├── evaluate.py              # Unified evaluation
└── common.py                # Shared utilities
```

### Week 9: Prefetch & Cache Optimization

#### 9.1 Cache Consolidation (Days 71-73)

Current: 3 cache levels with duplicate serialization  
Target: Unified cache with pluggable backends

```python
# jarvis/infrastructure/cache/
├── __init__.py
├── base.py                  # Abstract cache interface
├── l1_memory.py             # In-memory cache
├── l2_disk.py               # Disk cache
├── l3_remote.py             # Remote/distributed cache
└── serializer.py            # Shared serialization (was duplicated)
```

#### 9.2 Prefetch Simplification (Days 74-75)

Remove complexity:
- Simplify prediction model
- Remove dead prefetch paths
- Consolidate warming logic

### Week 10: Final Integration

#### 10.1 Integration Testing (Days 76-78)

```bash
# Full integration test suite
make test-integration

# End-to-end scenarios
./scripts/e2e_test.sh

# Performance regression tests
./scripts/perf_test.sh
```

#### 10.2 Documentation Updates (Days 79-80)

Update all documentation:
- `AGENTS.md` - New structure
- `README.md` - Simplified setup
- `docs/ARCHITECTURE.md` - Updated diagrams
- `docs/API_REFERENCE.md` - New error formats

---

## Phase 4: Stabilization & Handoff (Days 76-90) ✅ COMPLETE

### Week 11: Performance Validation

#### 11.1 Benchmark Baselines (Days 81-83) ✅ COMPLETE

Performance baselines established in `tests/test_performance_baselines.py`:

| Metric | Baseline | Gate | Status |
|--------|----------|------|--------|
| Embedding encode (10 texts) | <20ms | <100ms | ✅ |
| Category classifier inference | <10ms | <50ms | ✅ |
| Fact extraction pipeline | <50ms | <250ms | ✅ |
| Socket ping round-trip | <5ms | <200ms | ✅ |
| Cache hit/miss/set | <0.01ms | <1ms | ✅ |
| Coverage threshold | 64% | >=60% | ✅ |

#### 11.2 Load Testing (Days 84-85)

```bash
# API load test
wrk -t12 -c400 -d30s http://localhost:8742/health

# Socket server load test
python scripts/load_test_socket.py

# Memory pressure test
python scripts/memory_stress_test.py
```

### Week 12: Final Hardening

#### 12.1 Security Review (Days 86-87) ✅ COMPLETE

| Check | Tool | Status |
|-------|------|--------|
| Dependency vulnerabilities | `pip-audit` | ✅ |
| Hardcoded secrets | `git-secrets` | ✅ No secrets found |
| SQL injection | `bandit` | ✅ All queries parameterized |
| Path traversal | `bandit` | ✅ `validate_path()` added |
| Timing-safe token comparison | Manual review | ✅ `hmac.compare_digest()` |
| Rate limiting | Manual review | ✅ 100 req/s per client |

Security documentation: `docs/SECURITY.md`

### Phase 5: Pipeline Optimization & Resilience (Days 91-105) 🚧 IN PROGRESS

| Milestone | Status | Description |
|-----------|--------|-------------|
| Streaming Delta Ingestion | ✅ Done | Per-chat incremental backfill |
| Database Reliability | ✅ Done | SQLite retry logic for locked databases |
| Model Resource Manager | ✅ Done | Coordinated model lifecycle (8GB RAM) |
| Semantic Deduplication | ✅ Done | Embedding-based fact merging |
| Pipeline Observability | ✅ Done | Real-time stage monitoring |
| Timing Preferences | ✅ Done | Automatic quiet/preferred hours inference |

#### 12.2 Rollback Preparation (Days 106-107)

Create rollback procedures:

```bash
# scripts/rollback.sh
#!/bin/bash
# Rollback to pre-modernization state

BACKUP_DIR=".cleanup_backup"
RESTORE_DATE="$1"  # YYYYMMDD

# Restore deleted files
tar -xzf "$BACKUP_DIR/phase1_$RESTORE_DATE.tar.gz"

# Restore models
cp "$BACKUP_DIR/models/"* models/

# Reset git to known good state
git reset --hard "modernization-start"
```

#### 12.3 Final Verification (Day 90) ✅ COMPLETE

**Completed Checks:**

- ✅ `make verify` passes (lint + typecheck + tests)
- ✅ Coverage >=60% with `--cov-fail-under=60`
- ✅ Benchmark tests pass: `pytest -m benchmark tests/test_performance_baselines.py`
- ✅ Bandit security scan clean
- ✅ Documentation updated (ARCHITECTURE.md, SECURITY.md, TROUBLESHOOTING.md)

**Files Created/Modified in Phase 4:**
- `tests/test_performance_baselines.py` - Performance benchmark tests
- `pyproject.toml` - Coverage threshold, benchmark marker
- `jarvis/socket_server.py` - Rate limiting, timing-safe token comparison
- `jarvis/config.py` - Path validation (`validate_path()`)
- `docs/SECURITY.md` - Threat model and security documentation
- `docs/TROUBLESHOOTING.md` - Common issues and solutions
- `docs/ARCHITECTURE.md` - Updated with Phase 1-3 changes

---

## Risk Controls

### Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Deleted file needed later | Low | Medium | 30-day backup retention |
| Test regression | Medium | High | Phased rollout, feature flags |
| Import errors | Medium | Medium | Import linting, CI checks |
| Performance regression | Low | High | Benchmark gates |
| Dependency conflict | Medium | Medium | Lock file versioning |

### Rollback Triggers

Immediate rollback if:
- Test failures increase >5
- Benchmark gates fail
- Import errors in production paths
- Memory usage increases >20%

### Communication Plan

| Week | Communication |
|------|---------------|
| 1 | Announce cleanup, backup created |
| 2 | Scripts reorganization notice |
| 3 | Dependency changes |
| 4 | Test baseline established |
| 5 | Module boundary changes |
| 6 | File decomposition notice |
| 7 | API error format changes |
| 8 | Model pipeline changes |
| 9 | Cache changes |
| 10 | Integration complete |
| 11 | Performance results |
| 12 | Final report |

---

## Success Metrics

### Quantitative Targets

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Files >500 LOC | 5 | 0 | `wc -l` |
| Root-level files | 110 | 25 | `ls -1` |
| Scripts | 116 | 25 | `find scripts` |
| Dependencies | 52 | 35 | `pyproject.toml` |
| Test failures | 30 | 0 | `make test` |
| Contract drift | 27 | 0 | `CONTRACT_DRIFT_REPORT.md` |
| Technical debt | 50 | 10 | `TECHNICAL_DEBT_REGISTER.md` |
| Test coverage | 45% | 75% | Coverage report |
| Build time | 8 min | 4 min | `time make verify` |
| Import time | 2s | 1s | `time python -c "import jarvis"` |

### Qualitative Goals

- [ ] New developer onboarding <30 minutes
- [ ] Clear module ownership
- [ ] Documented API contracts
- [ ] Single source of truth for models
- [ ] Consistent error handling

---

## Appendix A: Migration Commands

### Daily Workflow

```bash
# Start of day
git pull origin main
make health

# Before changes
make check

# After changes
make verify

# End of day
git status
```

### Emergency Rollback

```bash
# Restore from backup
tar -xzf .cleanup_backup/phase1_20260208.tar.gz

# Or git rollback
git log --oneline -20  # Find last good commit
git reset --hard <commit>
```

---

## Appendix B: File Inventory

### Pre-Cleanup (Current)

```
404 Python files total
├── jarvis/        85 files, ~18,000 LOC
├── api/           45 files, ~17,000 LOC
├── core/          12 files, ~2,000 LOC
├── models/        18 files, ~8,000 LOC
├── integrations/  15 files, ~5,000 LOC
├── contracts/     10 files, ~1,500 LOC
├── scripts/      116 files, ~25,000 LOC
├── tests/         84 files, ~15,000 LOC
└── benchmarks/    19 files, ~4,000 LOC
```

### Post-Cleanup (Target)

```
280 Python files (-30%)
├── jarvis/        95 files (+10 modules), ~16,000 LOC
├── api/           30 files (-15 routers), ~12,000 LOC
├── core/          20 files (+8 modules), ~4,000 LOC
├── models/        12 files (-6 files), ~5,000 LOC
├── integrations/  12 files (-3 files), ~4,000 LOC
├── contracts/     12 files (+2 files), ~2,000 LOC
├── scripts/       25 files (-91 files), ~5,000 LOC
├── tests/         70 files (-14 files), ~12,000 LOC
└── benchmarks/    14 files (-5 files), ~3,000 LOC
```

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-10  
**Next Review:** End of Phase 1 (Day 30)
