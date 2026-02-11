# Post-Mortem: [Incident Title]

**Incident ID:** INC-YYYY-MM-DD-XXX  
**Date:** [Start date]  
**Duration:** [X minutes/hours]  
**Severity:** SEV-X  
**Reporter:** @username  

## Summary

[2-3 sentence summary of what happened]

## Timeline

| Time (UTC) | Event | Owner |
|------------|-------|-------|
| HH:MM | Issue detected via [monitoring/user report] | @detector |
| HH:MM | Incident declared, IC assigned | @oncall |
| HH:MM | [Key event] | @owner |
| HH:MM | Issue resolved | @resolver |

## Root Cause Analysis

### What happened?
[Detailed technical description]

### Why did it happen?
[Contributing factors]

### How did we detect it?
[Detection method and timing]

### Why didn't we catch it earlier?
[Gap in testing/monitoring]

## Impact Assessment

- **Users affected:** [Number or percentage]
- **Features impacted:** [List]
- **Data affected:** [None/Read-only/Corruption/Permanent loss]
- **Performance impact:** [If applicable]

## Detection & Response

### Detection
- **Detection method:** [Monitoring alert/User report/Other]
- **Time to detect:** [X minutes]
- **Alert quality:** [Was alert clear and actionable?]

### Response
- **Time to acknowledge:** [X minutes]
- **Time to diagnose:** [X minutes]
- **Time to mitigate:** [X minutes]
- **Time to resolve:** [X minutes]

## Lessons Learned

### What went well
1. [Item 1]
2. [Item 2]

### What could have gone better
1. [Item 1]
2. [Item 2]

### What went wrong
1. [Item 1]
2. [Item 2]

## Action Items

| ID | Action | Owner | Due Date | Priority | Status |
|----|--------|-------|----------|----------|--------|
| 1 | [Specific, actionable item] | @username | YYYY-MM-DD | P0/P1/P2 | Not started |
| 2 | [Specific, actionable item] | @username | YYYY-MM-DD | P0/P1/P2 | Not started |

## Prevention

### How do we prevent this from happening again?
[Technical or process changes]

### How do we detect this faster next time?
[Monitoring/alerting improvements]

### How do we respond faster next time?
[Runbook/process improvements]

## Attachments

- [Incident Slack channel: #inc-YYYY-MM-DD-xxx]()
- [Related PRs/commits]()
- [Log excerpts]()
- [Dashboards/metrics screenshots]()

---

## Review

**Post-Mortem Review Date:** YYYY-MM-DD  
**Attendees:** @names  
**Approved by:** @name  

**Action Item Follow-up:**
- [ ] All P0 items completed
- [ ] All P1 items scheduled
- [ ] Monitoring improvements deployed
- [ ] Runbook updates published
