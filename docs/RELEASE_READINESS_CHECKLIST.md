# JARVIS Release Readiness Checklist

> **Version:** 1.0  
> **Effective Date:** 2026-02-10  
> **Target Environment:** macOS Apple Silicon, 8GB RAM minimum  
> **Classification:** Release Blocker Document

---

## Overview

This checklist defines the exhaustive criteria for releasing JARVIS AI Assistant. All items must be evaluated and signed off before any release to production. The checklist covers code quality, testing, performance, documentation, migrations, observability, and incident readiness.

**Release Sign-Off Required From:**
- [ ] Engineering Lead
- [ ] QA Lead
- [ ] Security Lead
- [ ] Release Manager

---

## 1. CODE QUALITY GATES

### 1.1 Static Analysis

| # | Check | Command | Pass Criteria | Status |
|---|-------|---------|---------------|--------|
| 1.1.1 | Format Check | `make format-check` | 0 ruff format violations | ⬜ |
| 1.1.2 | Lint Check | `make lint` | 0 ruff violations (E,F,I,N,W,UP) | ⬜ |
| 1.1.3 | Type Safety | `make typecheck` | 0 mypy errors (strict mode) | ⬜ |
| 1.1.4 | Security Scan | `uv run bandit -r jarvis/ api/ core/` | 0 high/critical issues | ⬜ |
| 1.1.5 | Import Sort | `uv run ruff check --select I` | 0 import order violations | ⬜ |
| 1.1.6 | Debug Statements | `grep -rn 'breakpoint()\|import pdb\|IPython' jarvis/ api/ core/` | 0 debug statements | ⬜ |
| 1.1.7 | Dead Code | `uv run ruff check . --select=F401` | ≤5 unused imports (legacy OK) | ⬜ |
| 1.1.8 | TODO/FIXME Audit | `grep -rn 'TODO\|FIXME\|XXX\|HACK' jarvis/ core/ models/` | <100 total, none >30 days old | ⬜ |

**Rubric:**
- ✅ **PASS:** All checks pass with ≤5 exceptions (documented in `TECHNICAL_DEBT_REGISTER.md`)
- ⚠️ **CONDITIONAL:** 6-10 exceptions with Engineering Lead approval
- ❌ **FAIL:** >10 exceptions OR any high/critical security issues

### 1.2 Code Review Compliance

| # | Check | Criteria | Status |
|---|-------|----------|--------|
| 1.2.1 | PR Review | All changes reviewed by ≥1 engineer | ⬜ |
| 1.2.2 | Core Changes | Changes to `core/`, `models/` reviewed by ≥2 engineers | ⬜ |
| 1.2.3 | API Contracts | Contract changes reviewed by API owner | ⬜ |
| 1.2.4 | Breaking Changes | Documented in CHANGELOG with migration guide | ⬜ |
| 1.2.5 | Large Changes | >800 lines has 2+ reviewers | ⬜ |

### 1.3 Dependencies & Build

| # | Check | Command | Pass Criteria | Status |
|---|-------|---------|---------------|--------|
| 1.3.1 | Lock File | `uv lock` | No changes (fresh) | ⬜ |
| 1.3.2 | Build Verification | `uv build` | Clean wheel build | ⬜ |
| 1.3.3 | Install Test | `uv pip install dist/*.whl` | Successful installation | ⬜ |
| 1.3.4 | CLI Functionality | `jarvis --help` | Returns help text | ⬜ |
| 1.3.5 | Vulnerable Dependencies | `uv pip audit` OR `pip-audit` | 0 known CVEs in dependencies | ⬜ |

---

## 2. TEST VALIDATION

### 2.1 Unit Tests

| # | Check | Command | Pass Criteria | Status |
|---|-------|---------|---------------|--------|
| 2.1.1 | Unit Test Suite | `make test-unit` | ≥95% pass rate* | ⬜ |
| 2.1.2 | Contract Tests | `pytest tests/unit/test_contracts.py -v` | 100% pass | ⬜ |
| 2.1.3 | Test Duration | `make test` | <30s per test average | ⬜ |
| 2.1.4 | Memory Per Test | `pytest --memray` (if available) | <500MB peak per test | ⬜ |
| 2.1.5 | New Test Coverage | `pytest --cov` | New code has ≥70% coverage | ⬜ |

\* *Pre-existing failures tracked in `benchmarks/baseline.json` (currently 8)*

### 2.2 Integration Tests

| # | Check | Command | Pass Criteria | Status |
|---|-------|---------|---------------|--------|
| 2.2.1 | Integration Suite | `make test-integration` | 100% pass | ⬜ |
| 2.2.2 | Boundary Tests | `pytest tests/integration/test_boundary_contracts.py` | 100% pass | ⬜ |
| 2.2.3 | API Contract Tests | `pytest tests/integration/test_api_contracts.py` | 100% pass | ⬜ |
| 2.2.4 | DB Migration Tests | `pytest tests/integration/test_migrations.py` | 100% pass | ⬜ |

### 2.3 Hardware Tests (Apple Silicon Required)

| # | Check | Command | Pass Criteria | Status |
|---|-------|---------|---------------|--------|
| 2.3.1 | MLX Compatibility | `pytest tests/hardware/ -v` | 100% pass | ⬜ |
| 2.3.2 | Model Loading | `jarvis benchmark memory` | Completes without OOM | ⬜ |
| 2.3.3 | Generation Tests | `pytest tests/hardware/test_generation.py` | 100% pass | ⬜ |

### 2.4 Coverage Thresholds

| Component | Minimum | Target | Actual | Status |
|-----------|---------|--------|--------|--------|
| `jarvis/` | 60% | 70% | ___% | ⬜ |
| `api/` | 70% | 75% | ___% | ⬜ |
| `core/` | 60% | 70% | ___% | ⬜ |
| `contracts/` | 85% | 90% | ___% | ⬜ |
| `models/` | 50% | 60% | ___% | ⬜ |

**Rubric:**
- ✅ **PASS:** All unit tests ≥95% pass, integration tests 100% pass, coverage meets minimums
- ⚠️ **CONDITIONAL:** Unit tests 90-95% pass with documented exceptions
- ❌ **FAIL:** <90% unit test pass OR any integration test failures OR coverage <minimum -10%

---

## 3. PERFORMANCE VALIDATION

### 3.1 Latency Benchmarks

| Operation | p50 Target | p99 Target | Max | Actual p50 | Actual p99 | Status |
|-----------|------------|------------|-----|------------|------------|--------|
| Embedding (single) | <100ms | <200ms | Hard | ___ms | ___ms | ⬜ |
| Embedding (batch 100) | <500ms | <1000ms | Soft | ___ms | ___ms | ⬜ |
| Generation (warm) | <500ms | <2000ms | Hard | ___ms | ___ms | ⬜ |
| Generation (cold) | <5000ms | <10000ms | Soft | ___ms | ___ms | ⬜ |
| Classification | <50ms | <100ms | Hard | ___ms | ___ms | ⬜ |
| API Response | <200ms | <500ms | Hard | ___ms | ___ms | ⬜ |

**Command:** `pytest tests/integration/test_latency_gate.py -v`

### 3.2 Memory Benchmarks

| Operation | Peak Limit | Working Set | Actual Peak | Actual Working | Status |
|-----------|------------|-------------|-------------|----------------|--------|
| Batch Embedding (100) | <1GB | <500MB | ___GB | ___GB | ⬜ |
| Generation (512 tokens) | <4GB | <2GB | ___GB | ___GB | ⬜ |
| Full Pipeline | <6GB | <3GB | ___GB | ___GB | ⬜ |
| Startup | <2GB | <1GB | ___GB | ___GB | ⬜ |

**Command:** `uv run python -m benchmarks.memory.run --output results/memory.json`

### 3.3 Throughput Benchmarks

| Metric | Target | Minimum | Actual | Status |
|--------|--------|---------|--------|--------|
| Messages/sec (classification) | >20 | >15 | ___ | ⬜ |
| Embeddings/sec | >100 | >80 | ___ | ⬜ |
| Tokens/sec (generation) | >50 | >40 | ___ | ⬜ |
| API RPS | >10 | >8 | ___ | ⬜ |

### 3.4 Quality Benchmarks

| Metric | Target | Minimum | Actual | Status |
|--------|--------|---------|--------|--------|
| HHEM Score (hallucination) | ≥0.5 | ≥0.4 | ___ | ⬜ |
| Prompt Pass Rate | ≥80% | ≥75% | ___% | ⬜ |
| RAG Retrieval Relevance | ≥70% @ top-5 | ≥60% | ___% | ⬜ |
| RAG Answer Accuracy | ≥65% | ≥55% | ___% | ⬜ |

**Commands:**
```bash
# Hallucination benchmark
uv run python -m benchmarks.hallucination.run --output results/hhem.json

# Prompt evaluation
make eval

# RAG evaluation
make eval-rag
make eval-rag-relevance
```

### 3.5 Validation Gate Status

| Gate | Metric | Pass | Conditional | Fail | Actual | Status |
|------|--------|------|-------------|------|--------|--------|
| G1 | Model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB | ___GB | ⬜ |
| G2 | Mean HHEM score | ≥0.5 | 0.4-0.5 | <0.4 | ___ | ⬜ |
| G3 | Warm-start latency | <3s | 3-5s | >5s | ___s | ⬜ |
| G4 | Cold-start latency | <15s | 15-20s | >20s | ___s | ⬜ |

**Command:** `uv run python scripts/check_gates.py results/latest/`

**Rubric:**
- ✅ **PASS:** All hard gates pass, soft gates within 10% of target
- ⚠️ **CONDITIONAL:** 1 soft gate failing with documented impact and mitigation
- ❌ **FAIL:** Any hard gate failing OR >1 soft gate failing OR memory >7GB

---

## 4. DOCUMENTATION

### 4.1 User Documentation

| # | Document | Check | Status |
|---|----------|-------|--------|
| 4.1.1 | README.md | Updated for new features | ⬜ |
| 4.1.2 | CHANGELOG.md | Version entry with changes | ⬜ |
| 4.1.3 | docs/CLI_GUIDE.md | CLI changes documented | ⬜ |
| 4.1.4 | docs/API_REFERENCE.md | API changes documented | ⬜ |
| 4.1.5 | desktop/README.md | Desktop changes documented | ⬜ |
| 4.1.6 | Migration Guide | Breaking changes have migration steps | ⬜ |

### 4.2 Developer Documentation

| # | Document | Check | Status |
|---|----------|-------|--------|
| 4.2.1 | AGENTS.md | Updated for new conventions | ⬜ |
| 4.2.2 | docs/ARCHITECTURE.md | Architecture changes reflected | ⬜ |
| 4.2.3 | CODEOWNERS | Updated for new modules | ⬜ |
| 4.2.4 | API Contracts | Protocol interfaces documented | ⬜ |
| 4.2.5 | TECHNICAL_DEBT_REGISTER.md | New debt documented | ⬜ |

### 4.3 Code Documentation

| # | Check | Criteria | Status |
|---|-------|----------|--------|
| 4.3.1 | Public APIs | All public functions have docstrings | ⬜ |
| 4.3.2 | Complex Logic | Non-obvious code has comments | ⬜ |
| 4.3.3 | Type Hints | All public functions typed | ⬜ |
| 4.3.4 | Examples | New features have usage examples | ⬜ |

**Rubric:**
- ✅ **PASS:** All docs updated, user-facing changes have examples
- ⚠️ **CONDITIONAL:** Minor doc gaps with follow-up ticket
- ❌ **FAIL:** Missing API documentation OR breaking changes without migration guide

---

## 5. MIGRATIONS & DATA INTEGRITY

### 5.1 Database Schema

| # | Check | Command/Method | Pass Criteria | Status |
|---|-------|----------------|---------------|--------|
| 5.1.1 | Schema Version | Check `jarvis.db.schema` | Current version is latest | ⬜ |
| 5.1.2 | Migration Test | Fresh install → current | Completes without error | ⬜ |
| 5.1.3 | Rollback Test | Current → previous version | Rollback script works | ⬜ |
| 5.1.4 | Data Integrity | `PRAGMA integrity_check` on test DB | `ok` returned | ⬜ |
| 5.1.5 | Index Validation | Query plan analysis | Indexes used appropriately | ⬜ |

### 5.2 Configuration Migration

| # | Check | Pass Criteria | Status |
|---|-------|---------------|--------|
| 5.2.1 | Config Version | Config version incremented if changed | ⬜ |
| 5.2.2 | Auto-Migration | Old configs auto-upgrade | ⬜ |
| 5.2.3 | Default Values | New settings have sensible defaults | ⬜ |
| 5.2.4 | Validation | Invalid configs rejected with clear errors | ⬜ |

### 5.3 Model Artifacts

| # | Check | Location | Status |
|---|-------|----------|--------|
| 5.3.1 | Trigger Classifier | `~/.jarvis/trigger_classifier_model/` | ⬜ |
| 5.3.2 | Response Classifier | `~/.jarvis/response_classifier_model/` | ⬜ |
| 5.3.3 | Embedding Model | `BAAI/bge-small-en-v1.5` or compatible | ⬜ |
| 5.3.4 | Category Classifier | SVM model in `models/` or `~/.jarvis/` | ⬜ |
| 5.3.5 | MLX Models | Specified models downloadable | ⬜ |

**Rubric:**
- ✅ **PASS:** All migrations tested forward/backward, data integrity verified
- ⚠️ **CONDITIONAL:** Minor schema changes with documented rollback plan
- ❌ **FAIL:** Untested migrations OR breaking schema changes without migration

---

## 6. OBSERVABILITY

### 6.1 Logging

| # | Check | Criteria | Status |
|---|-------|----------|--------|
| 6.1.1 | Structured Logging | JSON format in production | ⬜ |
| 6.1.2 | Log Levels | Appropriate levels (DEBUG/INFO/WARNING/ERROR) | ⬜ |
| 6.1.3 | Sensitive Data | No PII in logs (hashed IDs) | ⬜ |
| 6.1.4 | Error Context | Errors have sufficient context | ⬜ |
| 6.1.5 | Performance Logs | Key operations have timing logs | ⬜ |

### 6.2 Metrics

| # | Metric | Endpoint | Status |
|---|--------|----------|--------|
| 6.2.1 | API Metrics | `/metrics` exposes request counts/latency | ⬜ |
| 6.2.2 | Model Metrics | `/metrics` exposes memory/tokens per second | ⬜ |
| 6.2.3 | Health Metrics | `/metrics` exposes health status | ⬜ |
| 6.2.4 | Custom Metrics | New features have relevant metrics | ⬜ |
| 6.2.5 | Prometheus Format | Metrics valid Prometheus format | ⬜ |

### 6.3 Health Endpoints

| # | Endpoint | Check | Status |
|---|----------|-------|--------|
| 6.3.1 | `/health` | Returns 200 with component status | ⬜ |
| 6.3.2 | `/health/ready` | Readiness probe works | ⬜ |
| 6.3.3 | `/health/live` | Liveness probe works | ⬜ |
| 6.3.4 | `/health/detailed` | Detailed diagnostics available | ⬜ |
| 6.3.5 | Circuit Status | `/circuits` shows breaker states | ⬜ |

### 6.4 Alerting (If Applicable)

| # | Check | Criteria | Status |
|---|-------|----------|--------|
| 6.4.1 | Critical Alerts | P0/P1 alerts defined | ⬜ |
| 6.4.2 | Alert Thresholds | Thresholds documented | ⬜ |
| 6.4.3 | Runbooks | Alert response procedures documented | ⬜ |

**Rubric:**
- ✅ **PASS:** All metrics available, health endpoints functional, no sensitive data in logs
- ⚠️ **CONDITIONAL:** Minor gaps with follow-up tickets
- ❌ **FAIL:** Missing health endpoints OR sensitive data in logs OR no error context

---

## 7. INCIDENT READINESS

### 7.1 Rollback Plan

| # | Check | Criteria | Status |
|---|-------|----------|--------|
| 7.1.1 | Version Tag | Previous stable version tagged | ⬜ |
| 7.1.2 | Database Compatibility | Rollback version compatible with schema | ⬜ |
| 7.1.3 | Config Compatibility | Config backward compatible | ⬜ |
| 7.1.4 | Rollback Procedure | Documented rollback steps | ⬜ |
| 7.1.5 | Rollback Tested | Rollback verified in staging | ⬜ |

### 7.2 Failure Mode Testing

| # | Scenario | Test Method | Status |
|---|----------|-------------|--------|
| 7.2.1 | Model Load Failure | Kill model process during request | ⬜ |
| 7.2.2 | Memory Pressure | Simulate high memory (90%+) | ⬜ |
| 7.2.3 | iMessage Access Lost | Revoke Full Disk Access | ⬜ |
| 7.2.4 | Database Lock | Lock chat.db during query | ⬜ |
| 7.2.5 | Circuit Breaker | Force failures to trigger CB | ⬜ |
| 7.2.6 | Rate Limiting | Exceed request limits | ⬜ |

### 7.3 Recovery Procedures

| # | Procedure | Documentation | Status |
|---|-----------|---------------|--------|
| 7.3.1 | Force Model Reload | `docs/RELIABILITY_ENGINEERING_PLAN.md` §8.3 | ⬜ |
| 7.3.2 | Reset Circuit Breakers | `docs/RELIABILITY_ENGINEERING_PLAN.md` §8.3 | ⬜ |
| 7.3.3 | Clear Task Queue | `docs/RELIABILITY_ENGINEERING_PLAN.md` §8.3 | ⬜ |
| 7.3.4 | Disaster Recovery | `docs/RELIABILITY_ENGINEERING_PLAN.md` §8.4 | ⬜ |

### 7.4 Graceful Degradation

| # | Feature | Healthy | Degraded | Failed Fallback | Status |
|---|---------|---------|----------|-----------------|---------|
| 7.4.1 | AI Drafts | Full LLM | Shorter context | Template replies | ⬜ |
| 7.4.2 | Summaries | Full LLM | Bullet extraction | "Unable to summarize" | ⬜ |
| 7.4.3 | Smart Replies | Context-aware | Pattern matching | Generic suggestions | ⬜ |
| 7.4.4 | Semantic Search | Vector + FTS | FTS only | Recent messages only | ⬜ |

### 7.5 Support Resources

| # | Resource | Check | Status |
|---|----------|-------|--------|
| 7.5.1 | On-Call Contact | Contact info current | ⬜ |
| 7.5.2 | Escalation Path | P0/P1 escalation defined | ⬜ |
| 7.5.3 | Issue Tracker | Template for release issues | ⬜ |
| 7.5.4 | Communication Plan | Team notification process | ⬜ |

**Rubric:**
- ✅ **PASS:** Rollback tested, all failure modes handled, recovery procedures documented
- ⚠️ **CONDITIONAL:** 1-2 untested scenarios with documented workarounds
- ❌ **FAIL:** No rollback plan OR untested rollback OR missing recovery procedures

---

## 8. DESKTOP APPLICATION (If Applicable)

### 8.1 Build Verification

| # | Check | Command | Pass Criteria | Status |
|---|-------|---------|---------------|--------|
| 8.1.1 | Dev Build | `make desktop-build` | Builds successfully | ⬜ |
| 8.1.2 | Production Build | `cd desktop && npm run tauri build` | Creates .app bundle | ⬜ |
| 8.1.3 | Code Signing | Check signing | Signed (or documented skip) | ⬜ |
| 8.1.4 | Notarization | Check notarization | Notarized (or documented skip) | ⬜ |

### 8.2 E2E Tests

| # | Check | Command | Pass Criteria | Status |
|---|-------|---------|---------------|--------|
| 8.2.1 | E2E Suite | `cd desktop && npm run test:e2e` | ≥90% pass | ⬜ |
| 8.2.2 | Browser Parity | `npm run test:parity:browser` | Passes on Chromium + WebKit | ⬜ |
| 8.2.3 | Critical Paths | App launch, conversation list, AI draft | 100% pass | ⬜ |

### 8.3 Desktop Integration

| # | Check | Criteria | Status |
|---|-------|----------|--------|
| 8.3.1 | Socket Communication | Unix socket IPC functional | ⬜ |
| 8.3.2 | Direct DB Access | SQLite reads functional | ⬜ |
| 8.3.3 | Real-time Updates | Push notifications work | ⬜ |
| 8.3.4 | Keyboard Shortcuts | Cmd+D, Cmd+S work | ⬜ |

**Rubric:**
- ✅ **PASS:** Desktop builds, E2E tests ≥90% pass, all integrations functional
- ⚠️ **CONDITIONAL:** Minor E2E failures with documented bugs
- ❌ **FAIL:** Build fails OR critical E2E tests fail

---

## 9. SECURITY & PRIVACY

### 9.1 Security Checks

| # | Check | Method | Pass Criteria | Status |
|---|-------|--------|---------------|--------|
| 9.1.1 | Secret Scan | `trufflehog` | 0 verified secrets | ⬜ |
| 9.1.2 | Dependency Audit | `pip-audit` | 0 critical/high CVEs | ⬜ |
| 9.1.3 | Static Analysis | `bandit` | 0 high/critical issues | ⬜ |
| 9.1.4 | Permission Check | Full Disk Access handling | Graceful degradation | ⬜ |
| 9.1.5 | Input Validation | Fuzz testing | No crashes/panics | ⬜ |

### 9.2 Privacy Compliance

| # | Check | Criteria | Status |
|---|-------|----------|--------|
| 9.2.1 | Local-First | No cloud data transmission | ⬜ |
| 9.2.2 | Data Minimization | Only necessary data collected | ⬜ |
| 9.2.3 | PII Handling | PII hashed/anonymized in logs | ⬜ |
| 9.2.4 | User Consent | Permissions requested appropriately | ⬜ |

**Rubric:**
- ✅ **PASS:** All security scans clean, privacy requirements met
- ⚠️ **CONDITIONAL:** Minor issues with immediate fix plan
- ❌ **FAIL:** Any critical security issue OR privacy violation

---

## 10. FINAL VERIFICATION

### 10.1 Release Build

| # | Check | Command | Status |
|---|-------|---------|--------|
| 10.1.1 | Full Verification | `make verify` | ⬜ |
| 10.1.2 | Health Check | `make health` | ⬜ |
| 10.1.3 | API Startup | `make api-dev` starts cleanly | ⬜ |
| 10.1.4 | CLI Smoke Test | `jarvis health` returns valid | ⬜ |
| 10.1.5 | Desktop Smoke Test | App launches and shows UI | ⬜ |

### 10.2 Version Checklist

| # | Check | Status |
|---|-------|--------|
| 10.2.1 | Version bump in `pyproject.toml` | ⬜ |
| 10.2.2 | CHANGELOG.md updated | ⬜ |
| 10.2.3 | Git tag created | ⬜ |
| 10.2.4 | Release notes drafted | ⬜ |

### 10.3 Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Engineering Lead | ________ | ____________ | ________ |
| QA Lead | ________ | ____________ | ________ |
| Security Lead | ________ | ____________ | ________ |
| Release Manager | ________ | ____________ | ________ |

---

## QUICK REFERENCE: AUTOMATED COMMANDS

Run these commands to validate the release:

```bash
# Complete verification pipeline
make verify                          # Lint + typecheck + test
make health                          # Project health summary

# Performance validation
uv run python -m benchmarks.memory.run --output results/memory.json
uv run python -m benchmarks.hallucination.run --output results/hhem.json
uv run python -m benchmarks.latency.run --output results/latency.json
uv run python scripts/check_gates.py results/latest/

# Security
uv run bandit -r jarvis/ api/ core/ -f screen
trufflehog git file://. --only-verified

# Desktop
make desktop-build

# Full smoke test
make api-dev &                         # Start API
curl http://localhost:8742/health      # Health check
jarvis health                          # CLI health
```

---

## APPENDIX: PASS/FAIL SUMMARY MATRIX

| Category | Weight | Score | Status |
|----------|--------|-------|--------|
| Code Quality | 20% | ___% | ⬜ |
| Tests | 20% | ___% | ⬜ |
| Performance | 20% | ___% | ⬜ |
| Documentation | 10% | ___% | ⬜ |
| Migrations | 10% | ___% | ⬜ |
| Observability | 10% | ___% | ⬜ |
| Incident Readiness | 10% | ___% | ⬜ |
| **TOTAL** | **100%** | **___%** | ⬜ |

### Release Decision

| Criteria | Threshold | Actual | Status |
|----------|-----------|--------|--------|
| Overall Score | ≥90% | ___% | ⬜ |
| Critical Gates | 0 failures | ___ | ⬜ |
| Security Issues | 0 critical | ___ | ⬜ |

**Final Decision:** ⬜ **APPROVED** / ⬜ **CONDITIONAL** / ⬜ **REJECTED**

**Conditions (if applicable):**
```
1. _______________________________________________________________
2. _______________________________________________________________
3. _______________________________________________________________
```

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-10 | Engineering | Initial release checklist |

---

**Questions or Issues:**
- File an issue with `[RELEASE]` prefix
- Contact: Engineering Lead
