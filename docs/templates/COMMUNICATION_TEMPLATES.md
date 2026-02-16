# JARVIS Incident Communication Templates

Quick templates for incident communications. Copy, customize, and send.

---

## Initial Incident Declaration (Internal)

**Channel:** `#jarvis-incidents`

```
🚨 INCIDENT DECLARED 🚨

ID: INC-YYYY-MM-DD-XXX
Severity: SEV-1/2/3/4
Status: Investigating

Summary: [One-line description of what's happening]

Impact: [What is affected and how many users]
Started: [Timestamp]
Detected: [How detected - monitoring alert, user report, etc.]

Actions Taken:
- [Action 1]
- [Action 2]

Next Update: [Timestamp + 30 min for SEV-1, + 1 hour for SEV-2]

IC: @username
Tech Lead: @username
```

---

## Status Update (Ongoing)

**Channel:** Incident-specific channel

```
⏱️ INCIDENT UPDATE - INC-YYYY-MM-DD-XXX

Status: [Investigating/Identified/Monitoring/Resolved]
Duration: [X minutes since start]

What we know:
- [Finding 1]
- [Finding 2]

What we're doing:
- [Action 1]
- [Action 2]

ETA for resolution: [Time estimate or "TBD"]

Next update: [When you'll post next]
```

---

## Resolution Notification (Internal)

**Channel:** `#jarvis-incidents` + Incident channel

```
✅ INCIDENT RESOLVED - INC-YYYY-MM-DD-XXX

Status: RESOLVED
Duration: [X minutes/hours]
Resolved at: [Timestamp]

Summary:
[What happened and how it was fixed]

Impact:
- Users affected: [Number/percentage]
- Features impacted: [List]
- Data impact: [None/Read-only/Corruption]

Next Steps:
- Post-mortem scheduled: [Date/time]
- Action items: [Brief description or link]

Thank you to everyone who helped resolve this incident.
```

---

## User Notification (SEV-1/2)

**Subject:** [JARVIS] Service Issue - We're Working On It

```
Hi [User/Team],

We're currently experiencing an issue with JARVIS that is affecting [description of impact].

What you may be experiencing:
- [Symptom 1]
- [Symptom 2]

What we're doing:
Our engineering team is actively working to resolve this issue. We've identified [the cause/are investigating] and expect [resolution ETA].

We'll update you every [30 minutes/hour] until this is resolved.

Thank you for your patience.

JARVIS Engineering Team
```

---

## User Notification (Resolved)

**Subject:** [JARVIS] Service Restored - Issue Resolved

```
Hi [User/Team],

The issue affecting JARVIS has been resolved as of [time].

What happened:
[Brief, non-technical description]

What we did:
[Brief description of resolution]

All services are now operating normally. If you continue to experience any issues, please [contact method].

We'll share a post-mortem with additional details within [48 hours/1 week].

Thank you for your patience.

JARVIS Engineering Team
```

---

## Executive Summary (SEV-1)

**To:** Leadership team  
**Subject:** [INCIDENT] JARVIS SEV-1 - Executive Summary

```
INCIDENT SUMMARY

Incident ID: INC-YYYY-MM-DD-XXX
Time: [Start time] - [Resolution time] ([Duration])
Severity: SEV-1

BUSINESS IMPACT
- Users affected: [Number/percentage]
- Features affected: [List]
- Data impact: [None/Read-only/Corruption]
- Revenue impact: [If applicable]

TECHNICAL SUMMARY
[2-3 sentence technical description]

RESOLUTION
[How the issue was resolved]

NEXT STEPS
- Post-mortem: [Scheduled time]
- Action items: [Number] items identified
- Prevention: [Brief description]

COMMUNICATION
- Internal status page updated: [Yes/No]
- User notification sent: [Yes/No]
- External communication: [If applicable]

CONTACT
Incident Commander: [Name] ([Contact])
```

---

## Post-Mortem Announcement

**Channel:** `#jarvis-engineering`

```
📋 POST-MORTEM PUBLISHED

Incident: INC-YYYY-MM-DD-XXX
Title: [Brief description]
Severity: SEV-X
Duration: [X minutes/hours]

Link: [URL to post-mortem document]

Key Takeaways:
1. [Key finding 1]
2. [Key finding 2]

Action Items: [Number] items ([Number] P0, [Number] P1)

Review Meeting: [Date/time] ([Calendar link])

Please review the document before the meeting and add any questions or comments.
```

---

## Stakeholder Update (Regular Cadence)

**To:** Product, Support, Leadership  
**Subject:** [JARVIS] Weekly Incident Report - [Date Range]

```
INCIDENT SUMMARY - [Date Range]

Total Incidents: [Number]
- SEV-1: [Number]
- SEV-2: [Number]
- SEV-3: [Number]
- SEV-4: [Number]

DETAILS

1. INC-YYYY-MM-DD-XXX - [Title] (SEV-X)
   - Duration: [X minutes]
   - Impact: [Brief description]
   - Status: [Resolved/Ongoing]
   - Link: [Post-mortem URL]

[Repeat for each incident]

TRENDS
[Week-over-week comparison, recurring issues, etc.]

UPCOMING PREVENTION WORK
- [Item 1]
- [Item 2]

Questions? Contact the on-call engineer: @oncall
```

---

## Feature Degradation Notice

**Channel:** `#jarvis-users` or in-app notification

```
⚠️ FEATURE DEGRADATION NOTICE

We're currently experiencing degraded performance for [feature name].

Impact:
- [Description of degraded behavior]
- Expected performance: [X seconds]
- Current performance: [Y seconds]

Workaround:
[If there's a workaround, describe it here]

We're actively working on a fix and expect resolution by [ETA].

Thank you for your patience.
```

---

## Scheduled Maintenance Notice

**Channel:** `#jarvis-users` + Email  
**Subject:** [JARVIS] Scheduled Maintenance - [Date/Time]

```
SCHEDULED MAINTENANCE

When: [Date] at [Time] ([Timezone]) - Duration: [X minutes]
What: [Brief description of maintenance]
Impact: [What will be unavailable/degraded]

During this maintenance window:
- [Service/feature 1] will be unavailable
- [Service/feature 2] may be slower than normal

Please save any work and avoid using [affected features] during this window.

We'll send an update when maintenance is complete.

Questions? Contact [support channel].
```

---

## False Alarm / All Clear

**Channel:** `#jarvis-incidents`

```
✅ FALSE ALARM - INC-YYYY-MM-DD-XXX

After investigation, the alert that triggered this incident was determined to be a false alarm.

What happened:
[Explanation of why the alert fired incorrectly]

Actions taken:
- [Fixed/adjusted monitoring rule]
- [Updated threshold from X to Y]

Incident is closed. No user impact.
```

---

_Customize these templates as needed. Remove sections that don't apply and add details specific to your incident._
