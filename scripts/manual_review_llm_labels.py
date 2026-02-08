#!/usr/bin/env python3
"""Manual review: Compare LLM vs heuristic labels on sample for quality check.

Shows disagreements between LLM and heuristics for manual assessment.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    results_path = PROJECT_ROOT / "llm_pilot_results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run validate_llm_categories.py first.")
        return 1

    # For now, just show key statistics and recommend next steps
    with results_path.open() as f:
        results = json.load(f)

    print("=" * 60)
    print("LLM Labeling Pilot Results Analysis")
    print("=" * 60)
    print()
    print(f"Accuracy vs heuristics: {results['accuracy']:.1%}")
    print()
    print("⚠️  WARNING: This accuracy measures agreement with weak")
    print("   supervision heuristics, NOT ground truth accuracy!")
    print()
    print("The heuristic labels themselves are only ~68% accurate,")
    print("so low agreement could mean:")
    print("  1. LLM is wrong (bad)")
    print("  2. Heuristics are wrong (good!)")
    print("  3. Examples are genuinely ambiguous")
    print()
    print("=" * 60)
    print("Recommendations:")
    print("=" * 60)
    print()
    print("Option 1: Skip pilot validation, proceed with full labeling")
    print("  - Label 15-17k ambiguous examples with LLM")
    print("  - Retrain category classifier")
    print("  - Evaluate on test set (real ground truth)")
    print("  - If F1 improves from 68.5% → 80%+, LLM labels are good!")
    print()
    print("Option 2: Manual spot-check (20-50 examples)")
    print("  - Manually review LLM vs heuristic disagreements")
    print("  - Assess if LLM labels seem reasonable")
    print("  - Proceed if quality looks good")
    print()
    print("Option 3: Use different model")
    print("  - Try zai-glm-4.7 instead of gpt-oss-120b")
    print("  - May have better instruction following")
    print()
    print("=" * 60)
    print()
    print("Recommended: Option 1 (full labeling + downstream eval)")
    print()
    print("Run full labeling with:")
    print("  uv run python scripts/llm_category_labeler.py --max-examples 5000")
    print()
    print("Then retrain and check if test F1 improves!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
