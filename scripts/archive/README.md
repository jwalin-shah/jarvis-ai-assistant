# Archived Scripts

These scripts were archived on 2025-02-12 because they are no longer actively used. They are one-off experiments, superseded versions, or completed migrations.

## What's Here

| Script | Reason Archived |
|--------|----------------|
| `extraction_bakeoff.py` | Superseded by `run_extractor_bakeoff.py` (v2) |
| `extract_facts.py` | Superseded by `extract_facts_batched.py` |
| `extract_and_validate_facts.py` | Superseded by `eval_extraction.py` |
| `cleanup_legacy_facts.py` | One-time migration, completed |
| `clean_goldset_phantoms.py` | One-time cleanup, completed |
| `sample_messages.py` | Redundant with `sample_for_labeling.py` |
| `explore_model.py` | Interactive dev tool, not part of pipeline |
| `filter_quality_pairs.py` | One-off quality filtering |
| `chat_llm.py` | Interactive dev tool |
| `run_extraction_chat.py` | Minimal test script |
| `smoke_test_entailment.py` | Manual testing script |
| `test_conversation_extraction.py` | Specific conversation test |
| `test_freeform.py` | Quick test script |

## Recovery

All scripts preserve full git history. To recover:

```bash
# Move back to scripts/
mv scripts/archive/script_name.py scripts/
```
