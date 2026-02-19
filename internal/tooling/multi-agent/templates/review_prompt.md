# Code Review: Lane {SOURCE_LANE} changes reviewed by Lane {REVIEWER_LANE}

You are reviewing code changes from **Lane {SOURCE_LANE} ({SOURCE_LABEL})**.
You are reviewing as **Lane {REVIEWER_LANE} ({REVIEWER_LABEL})**.

## Lane Ownership Reference

| Lane    | Owned Paths                                                                                                   |
| ------- | ------------------------------------------------------------------------------------------------------------- |
| A (App) | `desktop/`, `api/`, `jarvis/router.py`, `jarvis/prompts.py`, `jarvis/retrieval/`                              |
| B (ML)  | `models/`, `jarvis/classifiers/`, `jarvis/extractors/`, `jarvis/graph/`, `scripts/train*`, `scripts/extract*` |
| C (QA)  | `tests/`, `benchmarks/`, `evals/`                                                                             |
| Shared  | `jarvis/contracts/pipeline.py`                                                                                |

## The Diff

```diff
{DIFF_CONTENT}
```

## Review Criteria

1. **Ownership boundaries**: Does this change stay within Lane {SOURCE_LANE}'s owned paths? Flag any files outside their ownership.
2. **Contract changes**: If `jarvis/contracts/pipeline.py` was modified, are the changes backward-compatible? Do they require coordination?
3. **Code quality**: Is the code clean, well-structured, and following project conventions?
4. **Impact on YOUR lane**: Will this change break or interfere with your lane's work?
5. **Test coverage**: Are changes adequately tested (or do they need Lane C to add tests)?
6. **Performance**: Any obvious performance concerns (N+1 queries, unbatched ops, memory issues)?

## Your Response

Start your response with EXACTLY one of these on the first line:

- `APPROVE: <one-line reason>`
- `REJECT: <one-line reason>`

Then provide detailed feedback below. If rejecting, be specific about what needs to change.
