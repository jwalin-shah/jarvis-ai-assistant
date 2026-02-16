# Template Cleanup And Personalization Notes

## Scope
- Consolidate template cleanup outcomes and personalization guidance into one source.
- Keep production template system docs in `docs/TEMPLATE_SYSTEM.md`.

## Cleanup Outcomes
- Removed assistant-capability templates from reply-template matching.
- Rewrote overly formal/business-style templates to casual texting style.
- Added missing high-frequency casual patterns (agreement, decline, flexibility, status updates).
- Fixed known bad-match edge cases from evaluation feedback.

## Personalization Principles
- Keep **pattern matching** general and broadly reusable.
- Personalize **response text** to match user voice (tone, slang, brevity).
- Avoid contact-specific names, inside jokes, or one-off references in defaults.

## Recommended Personalization Workflow
1. Mine frequent outgoing short replies from local history.
2. Compare frequent replies against existing template responses.
3. Update response strings to match user voice while preserving generic trigger patterns.
4. Re-run template evaluation and spot-check low-score examples.

## Evaluation And Validation
- Validate with semantic template evals under `evals/`.
- Track:
  - Match rate
  - Quality score
  - Bad-match count
- Only promote changes after regression check on representative samples.

## File Hygiene
- Do not keep generated backups or ad-hoc experiment outputs in production commits.
- Keep exploratory writeups under `docs/research/`.
