## Prompt Change Summary

| Field | Value |
|-------|-------|
| **Type** | PATCH / MINOR / MAJOR |
| **Version** | 1.0.0 → X.Y.Z |
| **Affected Templates** | <!-- e.g., reply_generation, threaded_reply --> |
| **Categories Impacted** | <!-- e.g., planning, logistics, emotional_support --> |

## Description

<!-- Describe what changed and why -->

## Changes Made

- [ ] <!-- List specific changes -->
- [ ] <!-- List specific changes -->
- [ ] <!-- List specific changes -->

## Version Updates

- [ ] `PROMPT_VERSION` updated: `1.0.0` → `X.Y.Z`
- [ ] `PROMPT_LAST_UPDATED` updated to today
- [ ] `evals/prompt_versions.json` updated (if applicable)

## Evaluation Results

| Metric | Before | After | Delta | Status |
|--------|--------|-------|-------|--------|
| Category Accuracy | XX% | XX% | +/-X% | ✅/⚠️/❌ |
| Anti-AI Clean Rate | XX% | XX% | +/-X% | ✅/⚠️/❌ |
| Avg Response Length | XX chars | XX chars | +/-X | ✅/⚠️/❌ |
| Latency p95 | XXms | XXms | +/-X% | ✅/⚠️/❌ |
| Judge Score | X.X/10 | X.X/10 | +/-X.X | ✅/⚠️/❌ |

<!-- Attach evaluation output or link to CI artifacts -->

## Testing

- [ ] Unit tests pass (`pytest tests/unit/test_prompts.py`)
- [ ] Regression tests pass
- [ ] Evaluation pipeline run (`python evals/eval_pipeline.py`)
- [ ] No regressions detected (`python scripts/check_regression.py`)
- [ ] Token limits verified (< 1500 tokens)

## A/B Testing

<!-- Required for MINOR and MAJOR changes -->

- [ ] A/B test configured
- [ ] Success criteria defined
- [ ] Minimum sample size: ___
- [ ] Test duration: ___ hours

## Rollback Plan

In case of issues:

```bash
# Revert to previous version
git revert HEAD
git push

# Or checkout previous version
git checkout v1.0.0 -- jarvis/prompts.py
git commit -m "rollback: Revert prompts to v1.0.0"
```

## Checklist

- [ ] I have read the [Prompt Governance Policy](docs/PROMPT_MODEL_GOVERNANCE_POLICY.md)
- [ ] My changes follow the coding standards
- [ ] I have added appropriate tests
- [ ] I have updated documentation
- [ ] I have considered backwards compatibility
- [ ] Breaking changes are documented (for MAJOR)

## Related Issues

<!-- Link to related issues -->

## Screenshots / Examples

<!-- Show before/after outputs if applicable -->

---

**For Reviewers:**

Please verify:
1. Version bump is appropriate for the change type
2. Evaluation results show no significant regression
3. Rollback plan is clear and tested
4. For MINOR/MAJOR: A/B test plan is adequate
