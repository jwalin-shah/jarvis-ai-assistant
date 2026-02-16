# JARVIS Code Ownership and Review Policy

**Version:** 1.0  
**Effective Date:** 2026-02-10  
**Owner:** Engineering Team  
**Review Cycle:** Quarterly

---

## 1. PURPOSE AND SCOPE

This document establishes comprehensive guidelines for code ownership, reviewer assignment, review SLAs, escalation procedures, and pull request quality standards for the JARVIS AI Assistant repository.

**Scope:** All code changes to `main` branch via Pull Requests  
**Target:** 8GB RAM Apple Silicon local-first AI assistant  
**Related Documents:**

- `QUALITY_GATES_POLICY.md` - Quality gates and thresholds
- `AGENTS.md` - Development guidelines and conventions
- `CODEOWNERS` - Automated reviewer assignment
- `CONTRIBUTING.md` - Contribution guidelines

---

## 2. CODE OWNERSHIP MODEL

### 2.1 Ownership Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    TECHNICAL OWNERSHIP                       │
├─────────────────────────────────────────────────────────────┤
│  Global Fallback    │  * @backend-lead                      │
│  Primary Owners     │  Area-specific teams/leads            │
│  Secondary Owners   │  Cross-domain reviewers               │
│  Consulted          │  Domain experts (advisory only)       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Domain Ownership Matrix

| Domain              | Primary Owner   | Secondary Owners              | Escalation          |
| ------------------- | --------------- | ----------------------------- | ------------------- |
| **Core Backend**    | `@backend-lead` | `@devops`                     | `@architect`        |
| **ML/Models**       | `@ml-lead`      | `@backend-lead`, `@data-lead` | `@architect`        |
| **Data Pipeline**   | `@data-lead`    | `@ml-lead`                    | `@backend-lead`     |
| **API Layer**       | `@backend-lead` | `@frontend-lead`              | `@architect`        |
| **Desktop App**     | `@desktop-lead` | `@frontend-lead`              | `@backend-lead`     |
| **Infrastructure**  | `@devops`       | `@backend-lead`               | `@architect`        |
| **Quality/Testing** | `@qa-lead`      | `@backend-lead`, `@ml-lead`   | `@engineering-lead` |
| **Documentation**   | `@tech-writer`  | `@backend-lead`               | `@engineering-lead` |

### 2.3 Code Ownership by Path

See `.github/CODEOWNERS` for the complete, authoritative mapping. Key patterns:

```
# Critical Infrastructure (requires 2 reviewers)
jarvis/db/          @data-lead @backend-lead
jarvis/contracts/   @architect @backend-lead
models/             @ml-lead @backend-lead

# API & Client (cross-domain)
api/                @backend-lead @frontend-lead

# Desktop (Tauri + Svelte)
desktop/src-tauri/  @desktop-lead @backend-lead
desktop/src/        @frontend-lead @desktop-lead

# Quality Gates
tests/integration/  @qa-lead @backend-lead
benchmarks/         @ml-lead @qa-lead
```

---

## 3. REVIEWER ROUTING

### 3.1 Automatic Assignment Rules

| Change Characteristic               | Reviewer Assignment                | Rationale     |
| ----------------------------------- | ---------------------------------- | ------------- |
| Single-domain change                | 1 primary owner                    | Efficiency    |
| Cross-domain change                 | 1 from each domain                 | Coverage      |
| Critical path (`db/`, `contracts/`) | 2 reviewers minimum                | Safety        |
| >400 lines changed                  | +1 reviewer                        | Thoroughness  |
| >800 lines changed                  | +2 reviewers, split recommended    | Manageability |
| Breaking API change                 | `@architect` + domain owners       | Governance    |
| Security-sensitive                  | `@security-lead` + owner           | Compliance    |
| Release PR                          | `@release-manager` + domain owners | Control       |

### 3.2 Reviewer Selection Algorithm

```python
def assign_reviewers(pr):
    """
    Determine required reviewers for a PR.
    """
    reviewers = set()

    # 1. Get CODEOWNERS matches
    codeowners = get_codeowners(pr.changed_files)

    # 2. Add primary owners
    for owner in codeowners.primary:
        reviewers.add(owner)

    # 3. Add secondary for critical paths
    if pr.touches_critical_path():
        for owner in codeowners.secondary[:1]:  # First secondary
            reviewers.add(owner)

    # 4. Scale with change size
    if pr.lines_changed > 400:
        reviewers.add(codeowners.secondary[0])
    if pr.lines_changed > 800:
        reviewers.add(codeowners.secondary[1] if len(codeowners.secondary) > 1 else "@backend-lead")

    # 5. Special cases
    if pr.has_tag("breaking"):
        reviewers.add("@architect")
    if pr.has_tag("security"):
        reviewers.add("@security-lead")

    return list(reviewers)
```

### 3.3 Review Request Protocol

```
┌────────────────────────────────────────────────────────────────┐
│                    REVIEW REQUEST FLOW                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Author opens PR                                            │
│     └─→ CI runs automatically                                  │
│                                                                │
│  2. Author runs: make verify                                   │
│     └─→ All local gates must pass                              │
│                                                                │
│  3. Author fills PR template                                   │
│     └─→ Include: what, why, testing, risks                     │
│                                                                │
│  4. GitHub assigns reviewers per CODEOWNERS                    │
│     └─→ Author may add explicit reviewers                      │
│                                                                │
│  5. Reviewers notified (GitHub + Slack #code-reviews)          │
│                                                                │
│  6. Review cycle begins (see Section 4)                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.4 Review Delegation

When a requested reviewer is unavailable:

| Scenario                | Action                           | Timeframe |
| ----------------------- | -------------------------------- | --------- |
| Out of office           | Auto-reassign to secondary owner | Immediate |
| No response (24h)       | Ping in Slack #code-reviews      | 24h       |
| Still no response (48h) | Escalate to domain lead          | 48h       |
| Domain lead unavailable | Escalate to `@backend-lead`      | 72h       |

---

## 4. REVIEW SLA TARGETS

### 4.1 Response Time SLAs

| PR Priority     | First Response | Full Review | Re-review  |
| --------------- | -------------- | ----------- | ---------- |
| **P0 (Hotfix)** | 1 hour         | 2 hours     | 30 minutes |
| **P1 (Urgent)** | 4 hours        | 8 hours     | 2 hours    |
| **P2 (Normal)** | 24 hours       | 48 hours    | 24 hours   |
| **P3 (Low)**    | 48 hours       | 72 hours    | 24 hours   |

### 4.2 Priority Classification

| Priority | Criteria                                         | Label         |
| -------- | ------------------------------------------------ | ------------- |
| P0       | Security fix, production outage, data loss       | `priority/p0` |
| P1       | Feature blocking release, performance regression | `priority/p1` |
| P2       | Regular feature work, bug fixes (default)        | `priority/p2` |
| P3       | Refactoring, documentation, tech debt            | `priority/p3` |

### 4.3 Size-Based Adjustments

| Lines Changed | SLA Multiplier  | Recommendation                     |
| ------------- | --------------- | ---------------------------------- |
| <100 lines    | 1.0x (standard) | Ideal size                         |
| 100-400 lines | 1.0x            | Standard review                    |
| 400-800 lines | 1.5x            | Consider splitting                 |
| >800 lines    | 2.0x            | Must split or schedule deep review |

### 4.4 SLA Monitoring

```
┌────────────────────────────────────────────────────────────────┐
│                    SLA MONITORING DASHBOARD                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Metric                    │ Target  │ Alert │ Critical      │
│  ─────────────────────────────────────────────────────────────│
│  Avg first response time   │ <24h    │ >36h  │ >48h          │
│  Avg full review time      │ <48h    │ >60h  │ >72h          │
│  PRs >72h without review   │ 0       │ >3    │ >5            │
│  PRs >7 days open          │ 0       │ >5    │ >10           │
│  Reviewer load imbalance   │ <2x     │ >2.5x │ >3x           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.5 SLA Exceptions

SLAs may be extended with explicit communication:

| Exception                    | Extension | Communication       |
| ---------------------------- | --------- | ------------------- |
| Weekend/holiday              | +48h      | Auto-applied        |
| Large PR (>800 lines)        | 2x SLA    | Author note in PR   |
| Complex algorithmic change   | +24h      | Reviewer request    |
| Awaiting external dependency | Hold      | Comment with ticket |

---

## 5. REVIEW QUALITY STANDARDS

### 5.1 Reviewer Checklist

Every reviewer must verify:

#### Code Correctness

- [ ] Logic is correct and handles edge cases
- [ ] Error handling is appropriate
- [ ] No obvious security vulnerabilities
- [ ] Thread safety considered (if applicable)

#### Code Quality

- [ ] Follows project conventions (AGENTS.md)
- [ ] Naming is clear and consistent
- [ ] Functions are reasonably sized (<50 lines preferred)
- [ ] No code duplication (DRY principle)

#### Testing

- [ ] New code has tests
- [ ] Tests cover edge cases
- [ ] Existing tests still pass
- [ ] No flaky tests introduced

#### Performance

- [ ] Memory usage is reasonable (8GB constraint)
- [ ] No obvious N+1 queries or unbatched operations
- [ ] Vectorized operations preferred over loops
- [ ] Lazy imports for heavy modules

#### Documentation

- [ ] Complex logic has comments
- [ ] Public APIs have docstrings
- [ ] User-facing changes documented
- [ ] ADR created for architectural changes

### 5.2 Review Comment Severity

| Severity       | Indicator | Action Required                   |
| -------------- | --------- | --------------------------------- |
| **Blocking**   | 🔴        | Must address before merge         |
| **Suggestion** | 🟡        | Address or respond with rationale |
| **Nitpick**    | 🟢        | Optional, author discretion       |
| **Praise**     | ⭐        | Positive reinforcement            |

### 5.3 Approval Requirements

| PR Type              | Required Approvals | Required Roles                    |
| -------------------- | ------------------ | --------------------------------- |
| Documentation only   | 1                  | Any domain owner                  |
| Single-domain change | 1                  | Domain primary owner              |
| Cross-domain change  | 2                  | One from each domain              |
| Critical path change | 2                  | Primary + secondary               |
| API breaking change  | 2                  | Domain owner + `@architect`       |
| Security change      | 2                  | Domain owner + `@security-lead`   |
| Release PR           | 2                  | `@release-manager` + domain owner |

---

## 6. ESCALATION PATHS

### 6.1 Review Stalemate Resolution

```
┌────────────────────────────────────────────────────────────────┐
│                    ESCALATION PATHWAY                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Level 0: Reviewer Request Changes                             │
│     │                                                          │
│     ▼                                                          │
│  Level 1: Author ↔ Reviewer Discussion (48h max)               │
│     │  └─→ Resolve through conversation                        │
│     │                                                          │
│     ▼ (unresolved)                                             │
│  Level 2: Domain Lead Mediation                                │
│     │  └─→ @<domain>-lead reviews both positions               │
│     │                                                          │
│     ▼ (unresolved)                                             │
│  Level 3: Architect Decision                                   │
│     │  └─→ @architect makes final technical decision           │
│     │                                                          │
│     ▼ (unresolved)                                             │
│  Level 4: Engineering Lead Arbitration                         │
│        └─→ @engineering-lead decides, documents precedent      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 Emergency Escalation Triggers

| Trigger                      | Action                        | Contact          |
| ---------------------------- | ----------------------------- | ---------------- |
| Security vulnerability found | Immediate halt, security team | `@security-lead` |
| Test regression in `main`    | Block merges, investigate     | `@qa-lead`       |
| Performance regression >50%  | Performance review            | `@backend-lead`  |
| Contract drift detected      | Architecture review           | `@architect`     |
| CI/CD failure >1 hour        | Infrastructure review         | `@devops`        |

### 6.3 Escalation Communication Template

```markdown
## Escalation Request

**PR:** #<number>
**Escalated By:** @<username>
**Level:** 2 (Domain Lead)
**Reason:** <brief description>

### Summary of Disagreement

<What is the technical disagreement>

### Positions

- **Author:** <position>
- **Reviewer:** <position>

### Attempted Resolution

<What has been tried>

### Requested Outcome

<What decision is needed>
```

---

## 7. PR QUALITY CHECKLIST STANDARDS

### 7.1 Pre-Submission Checklist

Authors must complete before requesting review:

```markdown
## Pre-Review Checklist

### Local Verification

- [ ] `make format` - Code formatted
- [ ] `make lint` - No lint errors
- [ ] `make typecheck` - Type checking passes
- [ ] `make test` - All tests pass (or pre-existing failures only)
- [ ] `make verify` - Full verification passes

### Change Quality

- [ ] PR is focused (single concern)
- [ ] Changes are necessary and minimal
- [ ] No debug code or print statements
- [ ] No secrets or credentials in code

### Testing

- [ ] New code has unit tests
- [ ] Edge cases are covered
- [ ] Integration tests updated (if needed)
- [ ] Manual testing performed (if UI changes)

### Documentation

- [ ] Code comments for complex logic
- [ ] Docstrings for public APIs
- [ ] README updated (if needed)
- [ ] CHANGELOG entry (if user-facing)

### Review Readiness

- [ ] PR description is complete
- [ ] Linked issues referenced
- [ ] Screenshots attached (for UI changes)
- [ ] Breaking changes documented
```

### 7.2 PR Description Template

```markdown
## Summary

<!-- One-line summary of the change -->

## Motivation

<!-- Why is this change needed? Link to issues. -->

## Changes

<!-- Bullet points of what changed -->

-
-
-

## Testing

<!-- How was this tested? -->

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist

- [ ] `make verify` passes
- [ ] Documentation updated
- [ ] Breaking changes documented (if any)

## Screenshots / Logs

<!-- If applicable -->

## Breaking Changes

<!-- List any breaking changes or "None" -->

## Related Issues

<!-- Link to related issues: Fixes #123, Relates to #456 -->
```

### 7.3 Change Size Guidelines

| Size   | Lines   | Review Time | Best For          |
| ------ | ------- | ----------- | ----------------- |
| **XS** | <50     | 5-10 min    | Bug fixes, typos  |
| **S**  | 50-150  | 15-30 min   | Small features    |
| **M**  | 150-400 | 30-60 min   | Standard features |
| **L**  | 400-800 | 1-2 hours   | Large features    |
| **XL** | >800    | 2+ hours    | Should split      |

**Recommendation:** Most PRs should be S or M size.

### 7.4 PR Title Conventions

```
<type>(<scope>): <description>

Types:
  feat     - New feature
  fix      - Bug fix
  docs     - Documentation only
  refactor - Code refactoring
  perf     - Performance improvement
  test     - Test changes
  chore    - Build/tooling changes

Examples:
  feat(classifier): add adaptive threshold tuning
  fix(db): resolve connection pool exhaustion
  docs(api): update websocket documentation
  refactor(search): extract embedding cache
```

### 7.5 Merge Requirements Matrix

| Branch      | Required Checks    | Min Approvals | Special Requirements |
| ----------- | ------------------ | ------------- | -------------------- |
| `main`      | All CI checks      | 2             | No direct commits    |
| `release/*` | All CI + manual QA | 2             | `@release-manager`   |
| `hotfix/*`  | Critical tests     | 1             | Post-merge review    |
| `feature/*` | All CI checks      | 1             | Standard process     |

---

## 8. CONFLICT RESOLUTION

### 8.1 Technical Disagreements

When reviewers disagree on technical approach:

1. **Discuss First** - Async discussion in PR comments
2. **Sync if Needed** - Video call for complex disagreements
3. **Domain Lead Decides** - If still unresolved
4. **Document Decision** - Add comment explaining outcome

### 8.2 Style Disagreements

Style disagreements default to:

1. Existing codebase patterns
2. AGENTS.md conventions
3. Tool configurations (ruff, mypy)
4. Domain lead preference (if not covered above)

### 8.3 Scope Creep Prevention

If reviewers request changes beyond PR scope:

| Approach      | When to Use                       |
| ------------- | --------------------------------- |
| Address in PR | Related to changes, small effort  |
| Follow-up PR  | Unrelated, or large effort        |
| Out of Scope  | Clearly unrelated - create ticket |

**Process:**

1. Author responds with proposed approach
2. Reviewer approves approach
3. Create tracking ticket for follow-up
4. Document decision in PR

---

## 9. REVIEW METRICS AND IMPROVEMENT

### 9.1 Key Metrics

| Metric              | Target      | Measurement        |
| ------------------- | ----------- | ------------------ |
| Review Turnaround   | <48h median | GitHub API         |
| PR Lead Time        | <5 days     | GitHub API         |
| Review Rounds       | <2 median   | GitHub API         |
| Defect Escape Rate  | <5%         | Post-merge bugs    |
| Review Load Balance | <2:1 ratio  | Reviews per person |

### 9.2 Review Quality Indicators

Positive indicators:

- Constructive, specific feedback
- Educational comments (explaining why)
- Alternative approaches suggested
- Praise for good patterns

Negative indicators:

- Vague "fix this" comments
- Personal criticism
- Nits without context
- Blocking without explanation

### 9.3 Continuous Improvement

Quarterly review process:

1. Analyze metrics dashboard
2. Survey team on review experience
3. Identify bottlenecks
4. Update this policy
5. Share learnings in engineering sync

---

## 10. SPECIAL CASES

### 10.1 Emergency/Hotfix Process

```
┌────────────────────────────────────────────────────────────────┐
│                    HOTFIX REVIEW PROCESS                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Branch from main: hotfix/<description>                     │
│                                                                │
│  2. Make minimal fix + regression test                         │
│                                                                │
│  3. PR with [HOTFIX] prefix + P0 label                         │
│                                                                │
│  4. 1 reviewer required (domain owner)                         │
│     └─→ If owner unavailable, any senior engineer              │
│                                                                │
│  5. Post-merge review within 24h                               │
│                                                                │
│  6. Incident document created                                  │
│                                                                │
│  7. Follow-up PR for comprehensive fix                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 10.2 Dependency Updates

| Dependency Type  | Review Requirement | Auto-merge      |
| ---------------- | ------------------ | --------------- |
| Security patch   | 1 reviewer         | After CI passes |
| Minor version    | 1 reviewer         | After 24h       |
| Major version    | 2 reviewers        | Never           |
| Internal package | Standard process   | Never           |

### 10.3 Documentation-Only Changes

- 1 reviewer required (any domain owner)
- `make verify` not required (but CI checks must pass)
- Can be fast-tracked if trivial

### 10.4 Generated Code

| Type                            | Review Approach           |
| ------------------------------- | ------------------------- |
| Auto-generated (protobuf, etc.) | Spot-check + CI           |
| Migration scripts               | Full review               |
| ML model artifacts              | Artifact review + metrics |

---

## 11. ROLES AND RESPONSIBILITIES

### 11.1 Author Responsibilities

- Write clear PR descriptions
- Keep changes focused and reasonably sized
- Respond to feedback promptly
- Run `make verify` before requesting review
- Address or respond to all comments
- Merge only after approval

### 11.2 Reviewer Responsibilities

- Review within SLA timeframe
- Provide actionable, constructive feedback
- Distinguish blocking vs. non-blocking comments
- Approve when satisfied, request changes when not
- Escalate when stuck

### 11.3 Lead Responsibilities

| Lead             | Responsibility                                    |
| ---------------- | ------------------------------------------------- |
| Domain Leads     | Ensure timely reviews in domain, mediate disputes |
| `@backend-lead`  | Overall review process health, final arbiter      |
| `@architect`     | Technical direction, breaking change approval     |
| `@qa-lead`       | Test quality, coverage gates                      |
| `@security-lead` | Security review, vulnerability response           |

---

## 12. APPENDIX

### 12.1 Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│                    CODE REVIEW QUICK REF                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  BEFORE REQUESTING REVIEW:                                     │
│    make verify                                                 │
│    Fill PR template                                            │
│    < 400 lines preferred                                       │
│                                                                │
│  REVIEW SLAs:                                                  │
│    P0: 1h response, 2h review                                  │
│    P1: 4h response, 8h review                                  │
│    P2: 24h response, 48h review                                │
│                                                                │
│  APPROVALS NEEDED:                                             │
│    Normal: 1 domain owner                                      │
│    Critical: 2 reviewers                                       │
│    Breaking: +@architect                                       │
│                                                                │
│  ESCALATION:                                                   │
│    Stalemate → Domain Lead → Architect → Eng Lead              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 12.2 Related Files

| File                       | Purpose                       |
| -------------------------- | ----------------------------- |
| `.github/CODEOWNERS`       | Automatic reviewer assignment |
| `.github/workflows/ci.yml` | CI/CD enforcement             |
| `QUALITY_GATES_POLICY.md`  | Quality thresholds            |
| `AGENTS.md`                | Coding conventions            |
| `CONTRIBUTING.md`          | Contribution guide            |

### 12.3 Changelog

| Version | Date       | Changes                 |
| ------- | ---------- | ----------------------- |
| 1.0     | 2026-02-10 | Initial policy document |

---

## 13. ACKNOWLEDGMENTS

This policy is enforced through:

- GitHub CODEOWNERS for automatic assignment
- GitHub branch protection rules
- CI/CD pipeline gates
- Pre-commit hooks
- Team norms and culture

**Questions or feedback:** Contact `@backend-lead` or file an issue with `[REVIEW-POLICY]` prefix.
