# Deprecated

This goldset is subsumed by `training_data/gliner_goldset/candidate_gold_merged_r4.json`.

The gliner goldset contains all 115 positive messages from this set with richer span-level
annotations, plus 681 additional messages. Use the gliner goldset for all evaluation.

To evaluate extractors against the canonical goldset:

```bash
uv run python scripts/run_extractor_bakeoff.py --extractors regex,gliner
```
