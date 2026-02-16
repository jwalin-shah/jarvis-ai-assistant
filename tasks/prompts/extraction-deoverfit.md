# Fix Extraction Overfitting

You are working in the extraction-opt worktree at `/Users/jwalinshah/projects/jarvis-ai-assistant-extraction-opt`.

## Problem

The extraction optimization work achieved F1=0.976 but is heavily overfit:

1. **109 hardcoded keywords** in `_rule_based_boost()` that are specific to the goldset user's messages (dallas, meditation, 5k, biryani, facebook, etc.)
2. **Goldset reduced 22%** - 70 spans removed from the original 312, inflating recall
3. **No cross-validation** - `--limit 100` always tests the same first 100 records with no shuffle

Real generalizable F1 is estimated at ~0.65-0.70.

## Your Task

Strip the overfitting and measure real performance. Do this in phases:

### Phase 1: Measure True Baseline

1. Run the current extraction eval on the **ORIGINAL** goldset (`candidate_gold_merged_r4.json`), full 796 records (not --limit 100)
2. Record: P, R, F1, per-label breakdown
3. This is the "inflated" baseline

### Phase 2: Strip Hardcoded Keywords

1. In `scripts/eval_llm_extraction.py`, remove or disable `_rule_based_boost()` entirely
2. Remove all hardcoded keyword lists: `_FAMILY_WORDS`, `_HEALTH_KEYWORDS`, known orgs, known foods, known activities, known locations, "dallas", etc.
3. Keep ONLY generalizable logic:
   - The LLM extraction itself (prompts, few-shot examples are OK)
   - Post-processing that doesn't depend on specific entity values (e.g., "reject pronouns as subjects" is fine, "boost if text contains 'dallas'" is not)
   - `_is_transient_family_mention()` IF it uses linguistic patterns (not hardcoded names)
   - Label normalization/mapping (generic, not entity-specific)
4. Re-run eval on ORIGINAL goldset, full 796 records
5. Record: P, R, F1 - this is the "clean" baseline

### Phase 3: Add Cross-Validation

1. Add `--shuffle --seed 42` option to the eval script
2. Implement 5-fold cross-validation: split goldset into 5 folds, report mean and std of F1
3. Run 5-fold CV on the ORIGINAL goldset with the cleaned (no keyword boost) pipeline
4. Report mean F1 +/- std

### Phase 4: Targeted, Generalizable Improvements

If clean F1 < 0.60, try these (ONE at a time, measure each):

1. Better few-shot examples (diverse, not goldset-specific)
2. Better system prompt (emphasize "lasting personal facts" vs transient mentions)
3. Context window (include prev/next messages for disambiguation)
4. Two-pass: first classify if message has extractable facts, then extract
5. Post-processing: linguistic rules that generalize (e.g., "my X" family pattern is OK because it's a linguistic structure, not a keyword list)

### What's OK to Keep (Generalizable Rules)

- "my <family_relationship>" pattern (linguistic structure, not keyword)
- Pronoun rejection ("me", "you", "it" as subjects)
- Short span filtering (<2 chars)
- Label normalization (org -> employer)
- Transient mention detection based on linguistic cues (not entity matching)

### What MUST Go (Overfit Rules)

- Any hardcoded entity: "dallas", "facebook", "biryani", "meditation", "5k"
- Known orgs/schools list
- Known foods list
- Known activities list
- Known locations dict
- Any rule that checks for a specific string value from the goldset

## Constraints

- **ALWAYS eval on original goldset** (`candidate_gold_merged_r4.json`), never v5.x variants
- **8GB RAM** - one model at a time
- Use `uv run python` for all commands
- Commit after each phase with clear message: `extraction-deoverfit: phase N - <description> F1=X.XXX`
- Update `tasks/extraction-opt-status.md` after each phase

## Success Criteria

- Clean F1 >= 0.50 on original goldset (full 796 records) without any hardcoded keywords
- 5-fold CV results reported
- All goldset-specific keywords removed from eval script
