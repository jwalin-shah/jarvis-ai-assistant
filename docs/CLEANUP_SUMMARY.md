# Codebase Cleanup Summary

**Date:** 2026-02-15  
**Cleanup Scope:** Documentation, Results, Scripts

---

## Files Removed

### Results Directory (24 files removed)
**Timestamped Experiments (5 directories):**
- `results/experiment_20260130_113712/`
- `results/experiment_20260130_113802/`
- `results/experiment_20260130_113845/`
- `results/experiment_20260130_114053/`
- `results/experiment_20260130_114216/`

**Superseded Diagnosis Files (4 files):**
- `results/diagnosis_before.txt`
- `results/diagnosis_after.txt`
- `results/diagnosis_test.txt`
- `results/diagnosis_nogate.txt`
- **Kept:** `diagnosis_fixed_verbose.txt` (most comprehensive)

**Raw Dumps & Caches (4 files, ~2.5MB):**
- `results/gliner_candidates_dump.json` (1.3MB)
- `results/raw_pool_cache.json` (924KB)
- `results/json_vs_summary_dump.txt` (88KB)
- `results/facts_5contacts.txt` (175KB)

**Intermediate Eval Files (3 files):**
- `results/eval_200.json`
- `results/freeform_test.json`
- `results/small_nli_benchmark.json`

### Scripts Directory (33 files removed)
**One-Off Test Scripts (15 files):**
- Person-specific: `test_robert.py`, `test_mateo_extraction.py`, `find_lavanya.py`, `find_shanay.py`
- Model-specific: `test_07b_extraction.py`, `test_1_2b_knowledge.py`
- Prompt experiments: `test_chatml_extraction.py`, `test_extract_temp0.py`, `test_fewshot.py`, `test_prompt_styles.py`
- Superseded: `test_identity_labels.py`, `test_labeler.py`, `test_no_rag_reply.py`, `test_real_reply.py`, `test_summary_generation.py`

**Debug Scripts (3 files):**
- `debug_350m.py`
- `debug_extraction.py`
- `debug_nli.py`

**Personal/Specific Scripts (7 files):**
- `generate_radhika_bio.py`
- `reextract_radhika.py`, `reextract_radhika_limited.py`, `reextract_radhika_sliding.py`
- `extract_personal_data.py`, `evaluate_personal_ft.py`, `train_personal.py`

**Obsolete Extraction Scripts (8 files):**
- `extract_facts_350m_chunks.py`, `extract_facts_350m_constrained.py`
- `extract_facts_batched.py`, `extract_hybrid_facts.py`, `extract_candidates.py`
- `eval_extraction.py`, `eval_pass1_claims.py`, `evaluate_saved_facts.py`

---

## Files Organized

### Results Directory
**Created `results/published/` for final results:**
- Moved: `batch_eval_latest.json`
- Moved: `extraction_reality_check.md`
- Moved: `350m_knowledge_graph_summary.md`
- Moved: `trigger_classifier_comparison_report.md`

---

## Findings Preserved

All deleted experiment results have their findings documented in:
- `docs/EXTRACTOR_BAKEOFF.md` - Extractor comparison results
- `docs/fact_extraction_review.md` - Fact extraction analysis
- `docs/research/prompt_experiments.md` - Prompt engineering learnings
- `experiments/DAILYDIALOG_EXPERIMENTS.md` - Dialog dataset experiments

---

## Impact

- **Files Removed:** 57 files (~3MB disk space)
- **Directories Removed:** 5 experiment directories
- **Files Organized:** 4 files moved to `results/published/`
- **Findings Preserved:** All key learnings documented in markdown files

---

## Next Steps (Not Yet Completed)

### Documentation Consolidation
- Merge overlapping architecture docs (ARCHITECTURE.md, COMPONENT_CATALOG.md, HOW_IT_WORKS.md)
- Consolidate performance docs
- Consolidate roadmaps
- Archive large audit reports

### Scripts Organization
- Create `scripts/archived/` for rarely-used scripts
- Create `scripts/utils/` for reusable utilities
- Move appropriate scripts to these directories

### Verification
- Run `make test` to ensure no broken imports
- Check for broken documentation links
