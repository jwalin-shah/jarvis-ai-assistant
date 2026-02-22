
# Fix E501 in internal/archive/evals/evaluate_optimized_settings.py
    print(f"   Score: {score_improvement:+.2f} points")
    print(
        f"   Length: -{length_reduction:.0f} chars "
        f"({length_reduction / baseline_results['avg_length'] * 100:.0f}% shorter)"
    )
