# Codebase Cleanup Audit
**Generated**: 2026-02-08
**Context**: After LightGBM model integration, audit experimental artifacts

---

## Summary

- **Root-level experimental files**: 110+ files (~50MB disk usage)
- **Experimental model files**: 20+ models (~50MB disk usage)
- **Scripts directory**: 116 scripts (many experimental)
- **Production model**: `models/category_multilabel_lightgbm_hardclass.joblib` (10MB) ✅ KEEP

---

## Category 1: Root-Level Experimental Files (DELETE)

### Analysis Scripts (DELETE ALL - 24 files)
```
analyze_errors.py
analyze_quote_fix_impact.py
analyze_set2_errors.py
analyze_validation_errors.py
detailed_error_analysis.py
extract_set5_errors.py
review_errors.py
verify_validation_independence.py
compare_models.py
research_spacy_features.py
```

### Labeling Scripts (DELETE ALL - 9 files)
```
label_all_sets.py
label_dialog_simple.py
label_set2_with_llm.py
llm_label_dialog_clean.py
llm_label_dialog_set.py
manual_review_set2.py
sample_dialog_datasets.py
sample_multiple_sets.py
sample_new_validation_set.py
```

### Test Scripts (DELETE ALL - 6 files)
```
test_conservative_fixes.py
test_lightgbm_smoke.py
test_lol_category.py
test_quote_fix.py
inspect_model.py
validate_all_five_sets.py
validate_both_sets.py
validate_dialog_set.py
```

### Experiment Results/Data (DELETE ALL - 70+ files)
```
# JSON results
ablation_results.json
class_weight_test_results.json
class_weight_tuning_results.json
gemini_disagreement_analysis.json
learning_curve_results.json
lightgbm_tuning_results.json
linearsvc_comparison_results.json
threshold_tuning_results.json
gold_standard_150.json
human_labels_progress.json
improved_prompt_results.json
llm_pilot_results.json
llm_test_results.json
manual_labeling_*.json
model_comparison_results.json
production_validation*.json
qwen_test_results.json
simple_prompt_results.json
true_gold_standard_150.json

# JSONL datasets
dialog_validation_labeled.jsonl
dialog_validation_set.jsonl
gemini_label_review.jsonl
gemini_reviews*.jsonl
llm_category_labels.jsonl
set4_validation_results.jsonl
validation_*.jsonl (15+ files)

# Search result dumps (16 files)
c_search_results_*.json
rbf_search_results_*.json

# Text outputs
dailydialog_memory_trace.txt
debate_prompt*.txt
dry_run_output.txt
final_experiments_output.txt
imessage_threshold_tuning.txt
manual_review_corrections.txt
quote_fix_validation_results.txt
retrain_zeroed_output.txt
set2_review_summary.txt
test_category_results.txt
threshold_comparison_full.txt
train_*_output.txt (4 files)
tune_*_output.txt (2 files)
true_manual_review_20.txt
validation_*.txt (6+ files)
verify_results.txt
```

### Documentation Artifacts (CONSIDER ARCHIVING)
**These contain useful experiment notes - consider moving to `docs/experiments/`**
```
CONSENSUS_LABELING_GUIDE.md
CONTEXT_BACKFILL_RESULTS.md
CONTEXT_STRATEGY_ANALYSIS.md
debate_summary.md
FINAL_HUMAN_LABELS.md
FINAL_VERDICT.md
```

---

## Category 2: Obsolete Model Files (DELETE)

### Keep (Production)
- ✅ `models/category_multilabel_lightgbm_hardclass.joblib` (10MB) - **CURRENT PRODUCTION**
- ✅ `models/category_multilabel_lightgbm_hardclass_metadata.json`
- ✅ `models/category_multilabel_lightgbm_hardclass_optimal_thresholds.json`

### Archive/Delete (Superseded - 20+ files, ~43MB)
```
# Old SVM model (superseded)
category_svm_v2.joblib (21K)
category_svm_v2_metadata.json

# Experimental LightGBM variants
category_lightgbm_915_zeroed_context.joblib (1.9M)
category_lightgbm_no_context.joblib (1.9M)
category_lightgbm_imessage_thresholds.json
lightgbm_category_final.joblib (6.0M)
lightgbm_category_final.json

# Experimental multilabel variants
category_multilabel_hardclass.joblib (102K)
category_multilabel_hardclass_metadata.json
category_multilabel_hardclass_optimal_thresholds.json
category_multilabel_lightgbm.joblib (9.9M)
category_multilabel_lightgbm_metadata.json
category_multilabel_logistic.joblib (59K)
category_multilabel_logistic_metadata.json
category_multilabel_svm.joblib (59K)
category_multilabel_metadata.json
category_multilabel_optimal_thresholds.json

# Experimental LinearSVC variants
category_linearsvc_best.joblib (36K)
category_linearsvc_tuned.joblib (31K)
category_linearsvc_unbalanced.joblib (31K)
category_linearsvc_imessage_thresholds.json
```

---

## Category 3: Scripts Directory (116 files)

### Production Scripts (KEEP - ~10 files)
```
scripts/setup_db.py
scripts/train_category_classifier.py  # if still used
scripts/evaluate_model.py  # if still used
# ... identify which are still referenced in Makefile/docs
```

### Experimental Scripts (ARCHIVE/DELETE - ~100+ files)
```
scripts/add_multilabel_features.py
scripts/add_personachat_data.py
scripts/analyze_*.py (10+ files)
scripts/backfill_validation_context*.py (3 files)
scripts/batch_*.py
scripts/benchmark_*.py
scripts/compare_*.py
scripts/consensus_*.py
scripts/create_*.py
scripts/derive_*.py
scripts/evaluate_*.py (multiple)
scripts/fair_threshold_comparison.py
scripts/final_experiments.py
scripts/gemini_*.py
scripts/label_*.py
scripts/prepare_*.py
scripts/retrain_*.py
scripts/retune_*.py
scripts/review_*.py
scripts/sample_*.py
scripts/test_*.py
scripts/train_*.py (experimental variants)
scripts/tune_*.py (multiple)
scripts/validate_*.py
... (90+ more)
```

---

## Category 4: Dead Code in Production

### jarvis/prompts.py - Dead function (DELETE)
- `get_optimization_category()` - not called anywhere (verified with grep)
- **Action**: Remove function

**Note**: `jarvis/classifiers/adaptive_thresholds.py` was initially flagged but actually exists in `jarvis/eval/adaptive_thresholds.py` with tests - NOT dead code.

---

## Category 5: .gitignore Updates (ADD)

```gitignore
# Experiment artifacts (root level)
*.jsonl
*_results.json
*_output.txt
*_tuning_results.json
*search_results_*.json
validation_*.json
validation_*.jsonl
manual_*.json
human_labels_*.json
llm_*.json
gemini_*.json

# Test scripts (root level)
test_*.py
validate_*.py
analyze_*.py
label_*.py
sample_*.py
compare_*.py
inspect_*.py

# Training outputs
train_*_output.txt
tune_*_output.txt
retrain_*_output.txt

# Experiment documentation (if moving to docs/experiments/)
CONSENSUS_*.md
FINAL_*.md
*_ANALYSIS.md
*_RESULTS.md
```

---

## Recommended Actions

### Phase 1: Safe Deletions (Immediate)
1. Delete all root-level experimental files (~110 files)
2. Delete obsolete model files except production model (~20 files, ~43MB)
3. Remove `get_optimization_category()` from `jarvis/prompts.py`
4. Update `.gitignore` with experiment patterns

**Disk space reclaimed**: ~100MB

### Phase 2: Scripts Audit (Next session)
1. Identify which scripts are still used in production
2. Move production scripts to clearly named directory
3. Archive or delete experimental scripts

### Phase 3: Documentation Archive (Optional)
1. Create `docs/experiments/` directory
2. Move experiment markdown files there for reference
3. Create `docs/experiments/README.md` summarizing findings

---

## Verification Before Deletion

Run these checks before executing cleanup:
```bash
# 1. Verify no calls to get_optimization_category
grep -r "get_optimization_category" jarvis/ tests/

# 2. Verify no imports of deleted models
grep -r "category_svm_v2\|category_lightgbm_915\|category_linearsvc" jarvis/ tests/

# 3. Check git status to confirm what's tracked
git status --short

# 4. Verify current production model
grep -r "category_multilabel_lightgbm_hardclass" jarvis/
```

---

## Execute Cleanup Script

```bash
#!/bin/bash
# cleanup.sh - Execute Phase 1 cleanup

set -e

echo "=== Phase 1: Safe Deletions ==="

# Backup first
mkdir -p .cleanup_backup
tar -czf .cleanup_backup/pre_cleanup_$(date +%Y%m%d_%H%M%S).tar.gz \
    *.py *.json *.jsonl *.txt *.md models/ 2>/dev/null || true

# Delete root-level experimental files
rm -f analyze_*.py label_*.py test_*.py validate_*.py compare_*.py \
    sample_*.py inspect_*.py manual_review_*.py research_*.py \
    extract_*.py review_*.py detailed_*.py verify_*.py

rm -f *.jsonl *_results.json *_output.txt *_tuning_results.json \
    *search_results_*.json validation_*.json manual_*.json \
    human_labels_*.json llm_*.json gemini_*.json gold_standard*.json \
    improved_*.json production_*.json qwen_*.json simple_*.json \
    true_*.json ablation*.json class_weight*.json learning*.json

rm -f train_*_output.txt tune_*_output.txt retrain_*_output.txt \
    threshold_*.txt imessage_*.txt quote_fix*.txt set2_*.txt \
    test_*.txt true_*.txt verify_*.txt debate_*.txt dry_run*.txt \
    final_*.txt dailydialog*.txt manual_review*.txt

# Delete obsolete models (KEEP: category_multilabel_lightgbm_hardclass.* files)
cd models/
rm -f category_svm_v2.* category_lightgbm_915_*.* category_lightgbm_no_context.* \
    lightgbm_category_final.* category_multilabel_hardclass.* \
    category_multilabel_logistic.* category_multilabel_svm.* \
    category_multilabel_metadata.json category_multilabel_optimal_thresholds.json \
    category_linearsvc_*.* category_multilabel_lightgbm.joblib \
    category_multilabel_lightgbm_metadata.json
cd ..

echo "✓ Phase 1 complete"
echo "Backup saved to .cleanup_backup/"
echo "Disk space reclaimed: $(du -sh .cleanup_backup | cut -f1)"
```
