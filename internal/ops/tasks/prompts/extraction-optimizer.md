# Fact Extraction Optimizer

You are an autonomous agent optimizing a personal fact extraction pipeline for iMessage conversations. Your goal is to maximize F1 score on a gold evaluation set.

## Problem Definition

JARVIS extracts structured personal facts (family members, locations, jobs, hobbies, food preferences, health conditions) from iMessage text. The current pipeline uses GLiNER NER + rule-based patterns. We want to improve or replace it with LLM-based extraction using our already-loaded LFM-1.2B model.

**Current best F1**: 0.368 (constrained_categories strategy, lfm-1.2b-instruct)
**GLiNER baseline F1**: 0.273
**Target**: Maximize F1. Stretch goal 0.90, realistic goal 0.65+ for a local 1.2B model.

## Key Files

| File                                                               | Purpose                                                             |
| ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| `scripts/eval_llm_extraction.py`                                   | Main eval script. Runs LLM extraction on goldset, computes P/R/F1   |
| `scripts/eval_shared.py`                                           | Shared evaluation utilities (spans_match, label aliases)            |
| `training_data/gliner_goldset/candidate_gold_merged_r4.json`       | Gold evaluation set (796 records, 312 positive spans)               |
| `training_data/gliner_goldset/candidate_gold_merged_r4_clean.json` | Cleaned version of goldset                                          |
| `jarvis/contacts/fact_extractor.py`                                | Production fact extraction pipeline                                 |
| `models/loader.py`                                                 | MLX model loader                                                    |
| `tasks/extraction-opt-status.md`                                   | **YOUR STATUS FILE** - read this first, update after each iteration |

## Gold Set Format

Each record in the goldset JSON array:

```json
{
  "sample_id": "r2_fact_gs_0363",
  "message_id": 502741,
  "message_text": "Issk my brother bakes and I just eat whatever he makes",
  "is_from_me": true,
  "context_prev": "502910|me|previous msg text || ...",
  "context_next": "502794|me|next msg text || ...",
  "gold_keep": 1,
  "gold_fact_type": "hobby",
  "gold_subject": "brother",
  "gold_notes": "speaker's brother bakes; speaker eats whatever brother makes",
  "slice": "positive", // or "random_negative", "hard_negative", "near_miss"
  "expected_candidates": [
    { "span_text": "brother", "span_label": "family_member", "fact_type": "relationship.family" },
    { "span_text": "bakes", "span_label": "activity", "fact_type": "preference.activity" }
  ],
  "source_slice": "backfill"
}
```

Slices: `positive` (has facts), `random_negative` (no facts), `hard_negative` (looks like facts but isn't), `near_miss` (borderline).

## Evaluation Command

Run evaluation and check results:

```bash
uv run python scripts/eval_llm_extraction.py --gold training_data/gliner_goldset/candidate_gold_merged_r4.json --limit 100
```

Use `--limit 100` for fast iteration (100 messages ~2 min), full goldset (796 messages) for final measurement.

To see latest metrics:

```bash
cat results/llm_extraction/lfm2-extract_metrics.json | python3 -c "import json,sys; m=json.load(sys.stdin); print(f'P={m[\"overall\"][\"precision\"]:.3f} R={m[\"overall\"][\"recall\"]:.3f} F1={m[\"overall\"][\"f1\"]:.3f}')"
```

## What You CAN Modify

1. **Prompts** in `scripts/eval_llm_extraction.py`:
   - `EXTRACTION_SCHEMA` - the JSON schema
   - `EXTRACT_SYSTEM_PROMPT` - system prompt for Extract models
   - `INSTRUCT_USER_PROMPT` - user prompt for Instruct models
   - Add new prompt variants / strategies

2. **Extraction logic** in `scripts/eval_llm_extraction.py`:
   - `json_to_spans()` - how parsed JSON maps to span predictions
   - `parse_llm_json()` - JSON parsing from LLM output
   - Add new model configs to `MODEL_CONFIGS`
   - Add pre/post-processing steps

3. **Evaluation matching** in `scripts/eval_shared.py`:
   - `DEFAULT_LABEL_ALIASES` - label aliasing for fuzzy matching
   - `spans_match()` - matching criteria (but be careful not to inflate metrics artificially)

4. **Create new scripts** if needed for specific experiments.

5. **Gold set quality**: If you find systematic errors in the goldset (wrong labels, missing spans), you may fix them in a NEW copy (never overwrite the original). Document all goldset changes.

6. **Build a new goldset from scratch** (see Goldset Creation section below).

## What You CANNOT Modify

1. **Do NOT change the evaluation metric** (P/R/F1 computed by `compute_metrics()`)
2. **Do NOT modify the original goldset** (`candidate_gold_merged_r4.json`) - always create new files
3. **Do NOT use external API calls** (everything runs locally with MLX)
4. **Do NOT download new models** (use what's available: lfm2-350m-extract, lfm2-1.2b-extract, lfm2.5-1.2b-instruct)
5. **Do NOT modify production code** (`jarvis/contacts/fact_extractor.py`) until an approach proves > 0.60 F1

## Goldset Creation (HIGH PRIORITY)

The current goldset has known quality issues that cap achievable F1:

- Only 192/796 records have `expected_candidates` (the rest are negatives with no spans)
- Only 312 total gold spans across 13 label types (very sparse for some: friend_name=3, person_name=2)
- Label inconsistencies: `gold_keep` has mixed int/string types (1 vs "1")
- Some `expected_candidates` have span_text not found in the message_text
- Activity/hobby labels are underrepresented (52 spans) despite being common in chat

**You are encouraged to build a better goldset.** Here's how:

### Existing Tools

- `scripts/build_fact_goldset.py` - Samples messages from iMessage chat.db (stratified: random, likely-fact-bearing, hard negatives). Outputs JSONL/CSV/manifest. Run: `uv run python scripts/build_fact_goldset.py --total 400`
- `scripts/clean_goldset.py` - Cleans existing goldset (removes phantom spans, deduplicates entities)
- `scripts/merge_goldsets.py` - Merges multiple goldset rounds

### Building a New Goldset

You can create a new goldset by writing a script that:

1. **Samples diverse messages** from the existing goldset messages OR from `training_data/gliner_goldset/sampled_messages.json` (250 raw messages)
2. **Uses the LLM itself to label** - Use the LFM model to extract facts, then use a second pass or self-verification to validate
3. **Cross-validates** - Run multiple extraction strategies, keep spans where 2+ strategies agree
4. **Balances label distribution** - Ensure adequate representation across all fact types
5. **Includes proper negatives** - Messages that look like they have facts but don't (hard negatives)

### Goldset Quality Checklist

When creating a new goldset, ensure:

- [ ] Every `expected_candidates[].span_text` appears in `message_text` (case-insensitive substring match is OK)
- [ ] No duplicate spans within a record
- [ ] Labels use the canonical set: `family_member`, `activity`, `health_condition`, `job_role`, `org`, `place`, `food_item`, `current_location`, `future_location`, `past_location`, `friend_name`, `person_name`
- [ ] `fact_type` follows the hierarchy: `relationship.family`, `preference.activity`, `health.condition`, `health.allergy`, `work.employer`, `work.job_title`, `personal.school`, `preference.food_like`, `preference.food_dislike`, `location.current`, `location.future`, `location.past`
- [ ] Mix of slices: ~40% positive, ~10% hard_negative, ~5% near_miss, ~45% random_negative
- [ ] At least 10 examples per label type for statistical significance
- [ ] `gold_keep` is consistent (int: 1 or 0)

### Goldset Files

- **Original (read-only)**: `training_data/gliner_goldset/candidate_gold_merged_r4.json`
- **New goldsets**: Save to `training_data/gliner_goldset/goldset_v5_<description>.json`
- **Always eval on BOTH** old and new goldsets to track progress on each
- **Document** all goldset changes in the status file

### Eval with Custom Goldset

```bash
uv run python scripts/eval_llm_extraction.py --gold training_data/gliner_goldset/goldset_v5_improved.json --limit 100
```

## Prioritized Techniques to Try

### Tier 1: Quick Wins (try first)

1. **Better prompts**: More specific instructions, few-shot examples from the goldset, chain-of-thought
2. **Schema refinement**: Simplify or restructure the extraction schema to match what the model actually outputs
3. **Post-processing**: Better JSON parsing, span normalization, label mapping
4. **Temperature/sampling**: Test temp=0.0 vs 0.1 vs 0.3, top_p, repetition_penalty
5. **Two-pass extraction**: First pass detects if message has facts, second pass extracts them

### Tier 2: Structural Changes

6. **Constrained generation**: Force JSON output structure using logit constraints
7. **Label-specific prompts**: Run separate focused prompts for each fact type
8. **Context injection**: Include context_prev/context_next in the prompt
9. **NuExtract-style formatting**: Follow NuExtract paper's exact template format
10. **Ensemble**: Combine GLiNER + LLM predictions

### Tier 3: Advanced / Research-Based

11. **DSPy optimization**: Use DSPy framework to auto-optimize prompts (web search for latest techniques)
12. **Self-consistency**: Generate N times, take majority vote
13. **Error analysis driven**: Categorize failure modes, target specific fix for each
14. **Goldset augmentation**: Identify goldset quality issues (missing labels, wrong types)
15. **Hybrid pipeline**: Use LLM for some categories, rules for others

## Web Research

Use WebSearch to research:

- "NER extraction prompting techniques 2025"
- "DSPy structured extraction optimization"
- "few-shot NER small language models"
- "constrained JSON generation MLX"
- "NuExtract prompt format"
- Any technique you think might help

## Status File Protocol

At the START of each iteration:

1. Read `tasks/extraction-opt-status.md`
2. Check what's been tried, what worked, what didn't

At the END of each iteration:

1. Update the status file with:
   - What strategy you tried this iteration
   - The measured F1 (with --limit used)
   - Whether it improved over the previous best
   - What you plan to try next
2. If F1 improved, commit the changes: `git add -A && git commit -m "extraction-opt: <strategy> F1=<score>"`

### Status File Format

```markdown
## STATUS: IN_PROGRESS

## Current Best

- **F1**: 0.368
- **Strategy**: constrained_categories
- **Commit**: <hash>

## Iteration Log

### Iteration N - <strategy name>

- **F1**: <measured>
- **Limit**: 100 or full
- **Changes**: brief description
- **Result**: improved/regression/no change
- **Notes**: observations, error analysis

## Error Analysis

- <category>: <observation>

## Next Steps

1. <planned strategy>
```

## Important Constraints

- **8GB RAM**: Only one model loaded at a time. Unload before loading another.
- **MLX GPU not thread-safe**: Never run concurrent GPU ops.
- **Use `uv run`** for all Python commands.
- **Use `--limit 100`** for fast iteration, full goldset only for final measurements.
- **Commit improvements only**: Don't commit regressions.
- **One strategy per iteration**: Don't try to change everything at once.
- **Measure, don't guess**: Always run the eval script and report actual numbers.

## Starting Point

If this is your first iteration:

1. Read the current eval script thoroughly
2. Run baseline: `uv run python scripts/eval_llm_extraction.py --limit 100`
3. Analyze per-label breakdown to find biggest gaps
4. Start with Tier 1 techniques (prompts, schema, post-processing)
5. Create the status file with initial plan
