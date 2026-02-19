# JARVIS Prompt and Model Governance Policy

**Version:** 1.0.0  
**Effective Date:** 2026-02-10  
**Owner:** ML Engineering Team  
**Review Cycle:** Monthly for active development, Quarterly for stable releases  
**Related Documents:** [QUALITY_GATES_POLICY.md](../QUALITY_GATES_POLICY.md), [AGENTS.md](../AGENTS.md)

---

## 1. OVERVIEW

This document establishes the governance framework for managing prompts, model configurations, and ML artifacts in the JARVIS AI Assistant. It ensures controlled, traceable, and reversible changes to all AI/ML components that affect user-facing behavior.

**Scope:**

- All prompt templates in `jarvis/prompts/`
- Few-shot examples and style configurations
- Model weights and configurations in `models/`
- Classification thresholds and calibration parameters
- Embedding model selections and configurations

**Target Environment:**

- Apple Silicon (M1/M2/M3), 8GB RAM minimum
- Local-first, privacy-preserving inference
- MLX-based generation with sub-second latency targets

---

## 2. PROMPT VERSIONING

### 2.1 Version Numbering

Prompts follow **Semantic Versioning** (MAJOR.MINOR.PATCH):

| Level     | Trigger                                                                          | Example           |
| --------- | -------------------------------------------------------------------------------- | ----------------- |
| **MAJOR** | Breaking change to output format, new required parameters, removed functionality | `1.0.0` → `2.0.0` |
| **MINOR** | New features, additional optional parameters, new prompt templates               | `1.0.0` → `1.1.0` |
| **PATCH** | Bug fixes, wording improvements, example updates, threshold tuning               | `1.0.0` → `1.0.1` |

### 2.2 Source of Truth

```python
# jarvis/prompts/ - Single source of truth
PROMPT_VERSION = "1.0.0"
PROMPT_LAST_UPDATED = "2026-01-26"
```

All prompt changes must update these constants. The version must match the format defined in `PromptMetadata`:

```python
@dataclass
class PromptMetadata:
    name: str
    version: str = PROMPT_VERSION
    last_updated: str = PROMPT_LAST_UPDATED
    description: str = ""
```

### 2.3 Version Registry

A version registry is maintained in `evals/prompt_versions.json`:

```json
{
  "registry_version": "1.0",
  "prompts": [
    {
      "name": "reply_generation",
      "current_version": "1.0.0",
      "min_compatible": "1.0.0",
      "deprecated": false,
      "changelog": [
        {
          "version": "1.0.0",
          "date": "2026-01-26",
          "changes": ["Initial stable release"],
          "author": "eng-team",
          "commit": "abc123d"
        }
      ]
    }
  ]
}
```

### 2.4 Template Versioning

Each `PromptTemplate` must include version metadata:

```python
REPLY_PROMPT = PromptTemplate(
    name="reply_generation",
    system_message="...",
    template="...",
    max_output_tokens=25,  # Optimized for brief texting
)

# Metadata tracked separately
REPLY_PROMPT_METADATA = PromptMetadata(
    name="reply_generation",
    version="1.0.0",
    last_updated="2026-01-26",
    description="Core reply generation template with RAG support"
)
```

### 2.5 Change Classification

| Change Type                 | Version Bump | Approval Required  | Testing Required       |
| --------------------------- | ------------ | ------------------ | ---------------------- |
| Fix typo in system message  | PATCH        | 1 reviewer         | Unit tests             |
| Add new few-shot examples   | PATCH        | 1 reviewer         | Eval pipeline          |
| Modify tone detection logic | MINOR        | 2 reviewers        | Full eval + A/B        |
| New prompt template         | MINOR        | 2 reviewers        | Full eval + benchmarks |
| Remove/deprecate template   | MAJOR        | Eng Lead + Product | Migration plan         |
| Change output format/schema | MAJOR        | Eng Lead + Product | Contract tests         |

---

## 3. EVALUATION GATES

### 3.1 Pre-Merge Gates

All prompt changes must pass automated evaluation gates before merge:

#### 3.1.1 Static Analysis Gates

| Gate             | Tool/Method                | Threshold                       | Enforcement |
| ---------------- | -------------------------- | ------------------------------- | ----------- |
| Version Updated  | `grep PROMPT_VERSION`      | Must change if template changes | Hard block  |
| Date Updated     | `grep PROMPT_LAST_UPDATED` | Must match commit date          | Hard block  |
| Template Syntax  | Python `str.format()`      | No missing placeholders         | Hard block  |
| Token Estimation | `estimate_tokens()`        | <1500 tokens for prompts        | Hard block  |
| JSON Validity    | `json.loads()`             | All example JSON valid          | Hard block  |

#### 3.1.2 Functional Evaluation Gates

| Gate                    | Command                             | Threshold         | Max Time |
| ----------------------- | ----------------------------------- | ----------------- | -------- |
| Prompt Render Test      | `pytest tests/unit/test_prompts.py` | 100% pass         | 30s      |
| Category Classification | `evals/eval_pipeline.py`            | ≥90% accuracy     | 60s      |
| Anti-AI Detection       | `evals/eval_pipeline.py`            | ≥90% clean        | 30s      |
| Style Match Score       | LLM judge eval                      | ≥7.0/10 average   | 120s     |
| Response Length         | `evals/eval_pipeline.py`            | <80 chars average | 30s      |

#### 3.1.3 Quality Benchmark Gates

| Metric            | Threshold     | Test File                | Enforcement |
| ----------------- | ------------- | ------------------------ | ----------- |
| Prompt Pass Rate  | ≥80%          | `evals/promptfoo.yaml`   | Hard block  |
| Anti-AI Score     | ≥90%          | promptfoo anti-AI checks | Hard block  |
| Style Match       | ≥75%          | LLM rubric evaluation    | Soft gate   |
| Brevity Score     | <80 chars avg | JavaScript assertions    | Soft gate   |
| Category Accuracy | ≥85%          | `evals/eval_pipeline.py` | Hard block  |

### 3.2 A/B Testing Gates (For Major Changes)

For MINOR and MAJOR version bumps, shadow A/B testing is required:

```python
# Example: A/B testing configuration
AB_TEST_CONFIG = {
    "test_id": "prompt_v1_1_0_reply",
    "control_version": "1.0.0",
    "treatment_version": "1.1.0",
    "traffic_split": 0.1,  # 10% to treatment
    "metrics": [
        "response_acceptance_rate",
        "generation_latency_ms",
        "anti_ai_violation_rate",
        "user_edit_rate"
    ],
    "min_sample_size": 500,
    "duration_hours": 48,
    "success_criteria": {
        "response_acceptance_rate": ">= control - 5%",
        "generation_latency_ms": "<= control + 10%",
        "anti_ai_violation_rate": "<= control"
    }
}
```

### 3.3 Evaluation Dataset Management

| Dataset           | Location                       | Update Frequency | Approval      |
| ----------------- | ------------------------------ | ---------------- | ------------- |
| Gold eval set     | `evals/eval_dataset.jsonl`     | Per release      | Eng Lead      |
| Regression tests  | `evals/regression_tests.jsonl` | Continuous       | 2 reviewers   |
| Few-shot examples | `jarvis/prompts/`              | As needed        | 1 reviewer    |
| A/B test results  | `evals/ab_tests/`              | Per experiment   | Product + Eng |

### 3.4 Evaluation Pipeline

The standard evaluation pipeline (`evals/eval_pipeline.py`) must pass before any prompt change:

```bash
# Run full evaluation pipeline
uv run python evals/eval_pipeline.py --judge --similarity

# Gates enforced:
# 1. Category accuracy ≥ 85%
# 2. Anti-AI violations < 10%
# 3. Judge score ≥ 7.0/10 average
# 4. Latency p95 < 2000ms
```

---

## 4. ANTI-REGRESSION CHECKS

### 4.1 Baseline Establishment

Before any prompt change, establish or update baselines:

```bash
# Create baseline for current version
python evals/eval_pipeline.py --save-baseline results/prompt_baseline_v1.0.0.json
```

Baseline includes:

- Per-category accuracy rates
- Anti-AI violation rates
- Average response lengths
- Latency percentiles (p50, p95, p99)
- LLM judge scores
- Example outputs for 20 representative inputs

### 4.2 Regression Detection Matrix

| Metric              | Warning Threshold | Block Threshold | Action              |
| ------------------- | ----------------- | --------------- | ------------------- |
| Category Accuracy   | -5%               | -10%            | Investigate / Block |
| Anti-AI Violations  | +5% absolute      | +10% absolute   | Investigate / Block |
| Avg Response Length | +20%              | +50%            | Review / Block      |
| Latency p95         | +10%              | +25%            | Optimize / Block    |
| LLM Judge Score     | -0.5 points       | -1.0 points     | Review / Block      |
| Output Similarity\* | <0.85             | <0.70           | Review / Block      |

\*Cosine similarity between old and new outputs on same inputs

### 4.3 Automated Regression Testing

```python
# tests/unit/test_prompt_regression.py
REGRESSION_TESTS = [
    {
        "name": "casual_acknowledgment",
        "input": {
            "context": ["[10:00] John: Want to grab lunch?"],
            "last_message": "Want to grab lunch?",
            "tone": "casual"
        },
        "assertions": [
            "len(output) < 50",
            "'I would be happy' not in output.lower()",
            "'?' in output or output.endswith('!')"
        ]
    }
]
```

### 4.4 Output Diff Reporting

For every prompt change, generate a diff report:

```bash
# Generate output comparison
python scripts/compare_prompt_versions.py \
    --old-version 1.0.0 \
    --new-version 1.1.0 \
    --test-cases evals/regression_tests.jsonl \
    --output results/prompt_diff_v1.0.0_to_v1.1.0.md
```

Diff report must be attached to PR for MINOR/MAJOR changes.

### 4.5 Few-Shot Example Stability

Few-shot examples are critical for output quality. Changes require:

1. **Addition**: Add new examples with justification
2. **Modification**: Show before/after output changes
3. **Removal**: Requires 2 approvals + impact analysis

```python
# Example stability check
@dataclass
class FewShotExample:
    context: str
    output: str
    tone: Literal["casual", "professional"] = "casual"
    added_in_version: str = "1.0.0"  # Track when added
    stability_score: float = 0.0  # Measured consistency
```

---

## 5. ROLLBACK STRATEGY

### 5.1 Rollback Triggers

| Trigger  | Condition                           | Response Time      |
| -------- | ----------------------------------- | ------------------ |
| Critical | >20% increase in anti-AI violations | Immediate (<5 min) |
| Critical | Category accuracy drops below 70%   | Immediate (<5 min) |
| High     | Judge score drops by >1.5 points    | <30 min            |
| High     | Latency p95 increases >50%          | <30 min            |
| Medium   | Style match score drops >10%        | <4 hours           |
| Low      | Cosmetic output differences         | Next release       |

### 5.2 Rollback Mechanisms

#### 5.2.1 Code Rollback (Recommended)

```bash
# Revert to previous version
git revert HEAD  # Single commit
git revert HEAD~3..HEAD  # Range of commits

# Or checkout previous version
git checkout v1.0.0 -- jarvis/prompts/

# Update version constant
git commit -m "rollback: Revert prompts to v1.0.0 due to regression"
```

#### 5.2.2 Feature Flag Rollback (For production)

```python
# config/prompt_versions.yaml
prompt_versions:
  reply_generation:
    active_version: "1.0.0"  # Quick rollback by changing this
    available_versions:
      - "1.0.0"
      - "1.1.0"  # Flagged off if issues detected
```

#### 5.2.3 Emergency Hotfix

For critical issues requiring immediate fix:

```bash
# 1. Create hotfix branch from last stable tag
git checkout -b hotfix/prompt-v1.0.1 v1.0.0

# 2. Apply minimal fix
# Edit jarvis/prompts/

# 3. Fast-track review (1 approver for critical fixes)
git commit -m "hotfix: [EMERGENCY] Fix anti-AI regression in v1.1.0"

# 4. Merge and tag
git tag prompt-v1.0.1
```

### 5.3 Rollback Verification

After any rollback, verify:

| Check             | Command                               | Threshold                |
| ----------------- | ------------------------------------- | ------------------------ |
| Version correct   | `grep PROMPT_VERSION jarvis/prompts/` | Matches target           |
| Tests pass        | `make test`                           | 100% of regression tests |
| Baseline restored | `evals/eval_pipeline.py`              | Within 2% of baseline    |
| No data loss      | Check user feedback DB                | No missing records       |

### 5.4 Rollback Decision Tree

```
Issue Detected
    │
    ├─> Critical (user-facing breakage)?
    │   ├─> YES → Immediate rollback + incident
    │   └─> NO
    │
    ├─> Can fix forward in <30 min?
    │   ├─> YES → Hotfix branch
    │   └─> NO → Rollback + planned fix
    │
    └─> Isolated to specific category?
        ├─> YES → Disable category-specific prompt
        └─> NO → Full rollback
```

---

## 6. CHANGE-MANAGEMENT WORKFLOW

### 6.1 Change Classification

| Type            | Examples                       | Workflow     | SLA    |
| --------------- | ------------------------------ | ------------ | ------ |
| **Trivial**     | Typo fixes, comment updates    | Fast-track   | 1 hour |
| **Routine**     | New examples, threshold tuning | Standard     | 1 day  |
| **Substantial** | New templates, major rewrites  | Full process | 3 days |
| **Emergency**   | Hotfixes for production issues | Emergency    | 30 min |

### 6.2 Standard Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   DRAFT     │────▶│   REVIEW    │────▶│   TEST      │
│  (Local)    │     │  (PR Open)  │     │  (CI/CD)    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                         ┌──────────────────────┘
                         ▼
                ┌─────────────┐     ┌─────────────┐
                │   A/B TEST  │────▶│   RELEASE   │
                │  (If MINOR+ │     │  (Deploy)   │
                │   or MAJOR) │     │             │
                └─────────────┘     └─────────────┘
```

### 6.3 Phase Details

#### Phase 1: Draft (Local Development)

```bash
# 1. Create feature branch
git checkout -b feat/prompt-category-coverage

# 2. Make changes with version bump
# Edit jarvis/prompts/
# Update PROMPT_VERSION and PROMPT_LAST_UPDATED

# 3. Run local validation
make verify
python evals/eval_pipeline.py

# 4. Commit with conventional format
git commit -m "feat(prompts): Add planning category examples for v1.1.0

- Add 3 new few-shot examples for planning threads
- Improve context handling for group chats
- Bump version: 1.0.0 -> 1.1.0

Evaluation: +5% category accuracy, no regression"
```

**Checklist:**

- [ ] Version constants updated
- [ ] Local tests pass
- [ ] Evaluation shows improvement or parity
- [ ] Commit message includes evaluation summary

#### Phase 2: Review (Pull Request)

**PR Template for Prompt Changes:**

````markdown
## Prompt Change Summary

| Field                   | Value                            |
| ----------------------- | -------------------------------- |
| **Type**                | PATCH / MINOR / MAJOR            |
| **Version**             | 1.0.0 → 1.1.0                    |
| **Affected Templates**  | reply_generation, threaded_reply |
| **Categories Impacted** | planning, logistics              |

## Changes

- Added 3 new few-shot examples for planning threads
- Improved context truncation logic

## Evaluation Results

| Metric            | Before   | After    | Delta   |
| ----------------- | -------- | -------- | ------- |
| Category Accuracy | 87%      | 92%      | +5% ✅  |
| Anti-AI Clean     | 94%      | 95%      | +1% ✅  |
| Avg Length        | 42 chars | 45 chars | +3 ⚠️   |
| Latency p95       | 180ms    | 175ms    | -5ms ✅ |

## Testing

- [ ] Unit tests pass
- [ ] Regression tests pass
- [ ] A/B test configured (if MINOR/MAJOR)
- [ ] Rollback plan documented

## Rollback Plan

Revert to commit `abc123`:

```bash
git revert HEAD
git push
```
````

````

#### Phase 3: Test (CI/CD)

Automated gates run in sequence:

```yaml
# .github/workflows/prompt-ci.yml
jobs:
  validate:
    steps:
      - name: Version Check
        run: python scripts/check_prompt_version.py

      - name: Static Analysis
        run: |
          python -c "from jarvis.prompts import PROMPT_VERSION; print(f'Version: {PROMPT_VERSION}')"
          python scripts/validate_prompt_templates.py

      - name: Unit Tests
        run: pytest tests/unit/test_prompts.py -v

      - name: Regression Tests
        run: pytest tests/unit/test_prompt_regression.py -v

      - name: Evaluation Pipeline
        run: |
          python evals/eval_pipeline.py --save-results results/eval_${{ github.sha }}.json
          python scripts/check_regression.py --baseline results/prompt_baseline.json --current results/eval_${{ github.sha }}.json

      - name: A/B Test Setup (if MINOR/MAJOR)
        if: contains(github.event.pull_request.labels.*.name, 'prompt-minor') || contains(github.event.pull_request.labels.*.name, 'prompt-major')
        run: python scripts/setup_ab_test.py --pr ${{ github.event.number }}
````

#### Phase 4: A/B Testing (If Required)

For MINOR and MAJOR changes:

```python
# scripts/ab_test_runner.py
AB_TEST_DURATION = timedelta(hours=48)
MIN_SAMPLE_SIZE = 500

# Metrics tracked:
METRICS = {
    "primary": ["response_acceptance_rate", "anti_ai_violation_rate"],
    "secondary": ["generation_latency_ms", "user_edit_rate"],
    "guardrails": ["error_rate", "timeout_rate"]
}

# Success criteria:
SUCCESS_CRITERIA = {
    "response_acceptance_rate": lambda c, t: t >= c - 0.05,
    "anti_ai_violation_rate": lambda c, t: t <= c + 0.02,
    "generation_latency_ms": lambda c, t: t <= c * 1.10,
}
```

**A/B Test Report Template:**

```markdown
## A/B Test Results: prompt_v1.1.0

| Metric          | Control (v1.0.0) | Treatment (v1.1.0) | Delta | Pass? |
| --------------- | ---------------- | ------------------ | ----- | ----- |
| Acceptance Rate | 78%              | 81%                | +3%   | ✅    |
| Anti-AI Rate    | 6%               | 4%                 | -2%   | ✅    |
| Latency p95     | 185ms            | 178ms              | -4%   | ✅    |

**Decision:** PROMOTE v1.1.0 to 100% traffic
```

#### Phase 5: Release (Deploy)

```bash
# 1. Tag release
git tag prompt-v1.1.0
git push origin prompt-v1.1.0

# 2. Update changelog
cat >> docs/CHANGELOG.md << 'EOF'
## [1.1.0] - 2026-02-10
### Added
- Planning category few-shot examples
- Group chat context handling

### Changed
- Improved context truncation

### Evaluation
- Category accuracy: 87% → 92%
- Anti-AI clean: 94% → 95%
EOF

# 3. Deploy
make deploy

# 4. Post-deploy verification
python scripts/verify_deployment.py --version 1.1.0
```

### 6.4 Emergency Workflow

For critical production issues:

```
┌─────────────────────────────────────────────────────────┐
│  INCIDENT DETECTED                                      │
│  - Anti-AI violations spiked to 35%                     │
│  - Response acceptance dropped to 45%                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  1. IMMEDIATE ROLLBACK (< 5 min)                        │
│     git revert HEAD && git push                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  2. VERIFY ROLLBACK (< 10 min)                          │
│     - Check metrics return to baseline                  │
│     - Confirm user reports stop                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  3. INCIDENT DOCUMENT (< 1 hour)                        │
│     - Create incident doc                               │
│     - Notify team                                       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  4. ROOT CAUSE ANALYSIS (< 4 hours)                     │
│     - Reproduce in staging                              │
│     - Identify fix                                      │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  5. HOTFIX OR PLANNED FIX                               │
│     - If urgent: hotfix branch                          │
│     - If not: standard workflow                         │
└─────────────────────────────────────────────────────────┘
```

---

## 7. MODEL GOVERNANCE

### 7.1 Model Versioning

Models are versioned independently from prompts:

```
models/
├── category_svm_v2.pkl              # Model file
├── category_svm_v2_metadata.json    # Version metadata
└── category_svm_v2_card.md          # Model card
```

**Metadata Format:**

```json
{
  "model_name": "category_svm_rbf",
  "version": "2.0.0",
  "created_at": "2026-01-15T10:00:00Z",
  "training_data": {
    "source": "training_data/labeled_v2.jsonl",
    "samples": 15000,
    "date_range": "2025-10-01 to 2026-01-10"
  },
  "performance": {
    "accuracy": 0.89,
    "f1_macro": 0.87,
    "eval_dataset": "evals/category_eval_v2.jsonl"
  },
  "prompt_compatibility": {
    "min_prompt_version": "1.0.0",
    "max_prompt_version": "1.9.9"
  }
}
```

### 7.2 Model Gates

| Gate          | Check          | Threshold  | Enforcement |
| ------------- | -------------- | ---------- | ----------- |
| Accuracy      | Validation set | ≥85%       | Hard block  |
| Latency       | Inference time | <50ms p99  | Hard block  |
| Size          | Model file     | <100MB     | Soft gate   |
| Compatibility | Prompt version | Compatible | Hard block  |

### 7.3 Threshold Management

Classification thresholds are treated as code:

```python
# models/category_thresholds.yaml
thresholds:
  acknowledge: 0.45
  closing: 0.50
  question: 0.40
  request: 0.55
  emotion: 0.60
  statement: 0.35

version: "2026-01-15"
calibration_dataset: "evals/calibration_v2.jsonl"
```

Changes to thresholds require:

1. Calibration on held-out dataset
2. Evaluation on full eval set
3. Approval from 2 reviewers

---

## 8. AUDIT AND COMPLIANCE

### 8.1 Audit Trail

All changes are logged to `logs/prompt_audit.log`:

```json
{
  "timestamp": "2026-02-10T14:30:00Z",
  "event": "prompt_change",
  "version": "1.0.0",
  "author": "developer@example.com",
  "commit": "abc123def456",
  "change_type": "minor",
  "templates_modified": ["reply_generation"],
  "evaluation_results": {
    "category_accuracy": 0.92,
    "anti_ai_rate": 0.05
  }
}
```

### 8.2 Compliance Checks

| Check                | Frequency | Owner    |
| -------------------- | --------- | -------- |
| Version alignment    | Per PR    | CI/CD    |
| Evaluation freshness | Weekly    | ML Eng   |
| Baseline currency    | Monthly   | ML Eng   |
| Audit log review     | Quarterly | Security |

### 8.3 Required Records

| Record             | Location                     | Retention |
| ------------------ | ---------------------------- | --------- |
| Prompt versions    | `evals/prompt_versions.json` | Permanent |
| Evaluation results | `results/eval_*.json`        | 1 year    |
| A/B test results   | `evals/ab_tests/`            | 2 years   |
| Audit logs         | `logs/prompt_audit.log`      | 3 years   |

---

## 9. ROLES AND RESPONSIBILITIES

| Role                 | Responsibilities                                | Escalation       |
| -------------------- | ----------------------------------------------- | ---------------- |
| **ML Engineer**      | Implement changes, run evaluations, write tests | Engineering Lead |
| **Reviewer**         | Review PRs, verify evaluations, check baselines | Engineering Lead |
| **Engineering Lead** | Approve MINOR/MAJOR changes, handle exceptions  | CTO              |
| **Product Manager**  | Approve A/B tests, review user-facing changes   | Engineering Lead |
| **Release Manager**  | Coordinate releases, verify rollback plans      | CTO              |
| **On-Call Engineer** | Handle incidents, execute rollbacks             | Engineering Lead |

---

## 10. APPENDIX

### 10.1 Quick Reference

```bash
# Check current prompt version
python -c "from jarvis.prompts import PROMPT_VERSION; print(PROMPT_VERSION)"

# Run evaluation
python evals/eval_pipeline.py --judge --similarity

# Compare versions
python scripts/compare_prompt_versions.py --old 1.0.0 --new 1.1.0

# Create baseline
python evals/eval_pipeline.py --save-baseline results/baseline_v1.0.0.json

# Check for regression
python scripts/check_regression.py --baseline results/baseline.json

# Emergency rollback
git revert HEAD && git push
```

### 10.2 Decision Matrix

| Situation            | Version Bump | Tests                      | A/B Test | Approval           |
| -------------------- | ------------ | -------------------------- | -------- | ------------------ |
| Fix typo             | PATCH        | Unit                       | No       | 1 reviewer         |
| New examples         | PATCH        | Full eval                  | No       | 1 reviewer         |
| New template         | MINOR        | Full eval                  | Yes      | 2 reviewers        |
| Remove template      | MAJOR        | Full eval + migration      | Yes      | Eng Lead + Product |
| Output format change | MAJOR        | Contract tests + full eval | Yes      | Eng Lead + Product |
| Emergency fix        | PATCH        | Minimal                    | No       | Post-hoc review    |

### 10.3 Glossary

| Term                  | Definition                                                  |
| --------------------- | ----------------------------------------------------------- |
| **Prompt Template**   | A structured string with placeholders for dynamic content   |
| **Few-Shot Example**  | Example input/output pairs provided in the prompt           |
| **Anti-AI Violation** | Output containing phrases that sound artificially generated |
| **Category Accuracy** | Percentage of correct intent classifications                |
| **A/B Test**          | Controlled experiment comparing two versions                |
| **Baseline**          | Established performance metrics for comparison              |
| **Regression**        | Degradation in performance compared to baseline             |

### 10.4 Changelog

| Version | Date       | Changes                   |
| ------- | ---------- | ------------------------- |
| 1.0.0   | 2026-02-10 | Initial governance policy |

---

**Questions or exceptions:** Contact the ML Engineering Team or file an issue with `[PROMPT-GOVERNANCE]` prefix.
