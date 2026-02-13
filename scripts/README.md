# JARVIS Scripts

Utility scripts for development, benchmarking, and automation.

## Core Pipeline (Topic Segments)

- `backfill_segments.py` - Migrates historical iMessage data to the new topic-based segment system.
- `export_eval_segments.py` - Exports segmented conversation data for fact extraction evaluation.
- `backfill_contact_facts.py` - Runs fact extraction on existing contacts to populate the knowledge graph.

## Evaluation & Benchmarking

- `eval_entailment_gate.py` - Evaluates the impact of the NLI entailment gate on fact extraction.
- `eval_extraction.py` - Comprehensive fact extraction evaluation against goldsets.
- `eval_combined_extraction.py` - Evaluates the merged GLiNER + spaCy extraction pipeline.
- `eval_classifiers.py` - Benchmarks category and mobilization classifiers.
- `generate_report.py` - Aggregates benchmark results into human-readable reports.

## Model Training & Preparation

- `train_personal.py` - Orchestrates training of style-aware models.
- `prepare_gliner_training.py` - Prepares datasets for GLiNER fine-tuning.
- `prepare_mobilization_training.py` - Prepares datasets for mobilization classifier training.
- `generate_preference_pairs.py` - Uses Gemini to generate ORPO preference pairs for RLHF.

## Development Helpers

- `ner_server.py` - Standalone spaCy NER service for local development.
- `db_maintenance.py` - Database optimization and integrity checks (Archived).
- `check_regression.py` - Runs fast regression tests on key models.

---

## Legacy Archive

Older scripts related to the legacy "pairs" system (trigger/response pairs) have been moved to `scripts/archive/`.
These include:
- `filter_quality_pairs.py`
- `prepare_personal_data.py`
- `train_category_svm.py`
- Legacy extraction bakeoffs and goldset cleaning scripts.

---

## Adding New Scripts

When adding scripts:
1. Add shebang line: `#!/usr/bin/env python3`
2. Make executable: `chmod +x scripts/your_script.py`
3. Add docstring with usage examples
4. Update this README
