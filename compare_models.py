#!/usr/bin/env python3
"""Compare old vs new model predictions on validation set."""
import json
import sys

sys.path.insert(0, '.')
from jarvis.classifiers.category_classifier import classify_category

# Load the old validation results
with open('validation_results.jsonl') as f:
    old_results = [json.loads(line) for line in f]

print('Comparing OLD vs NEW model on all 100 examples:')
print('='*80)

changed = 0
new_agrees = 0
old_agrees = 0

for i, r in enumerate(old_results, 1):
    # Run new model
    new_result = classify_category(r['text'], context=r['context'])
    old_pred = r['svm_prediction']
    new_pred = new_result.category
    llm_pred = r['llm_prediction']

    old_match = old_pred == llm_pred
    new_match = new_pred == llm_pred

    if old_match:
        old_agrees += 1
    if new_match:
        new_agrees += 1

    if old_pred != new_pred:
        changed += 1
        match_old = '✓' if old_match else '✗'
        match_new = '✓' if new_match else '✗'

        print(f'{i}. "{r["text"][:60]}"')
        print(f'   OLD: {old_pred:12} {match_old}  NEW: {new_pred:12} {match_new}  LLM: {llm_pred}')
        if not old_match and new_match:
            print(f'   ✅ FIXED!')
        elif old_match and not new_match:
            print(f'   ❌ BROKE!')
        print()

print('='*80)
print(f'Total examples: {len(old_results)}')
print(f'OLD model agreement: {old_agrees}/100 ({old_agrees}%)')
print(f'NEW model agreement: {new_agrees}/100 ({new_agrees}%)')
print(f'Changed predictions: {changed}')
print(f'Improvement: {new_agrees - old_agrees:+d} percentage points')
