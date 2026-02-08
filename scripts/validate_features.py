#!/usr/bin/env python3
"""Validate feature extraction before training.

Verifies:
1. Feature dimensions match expected (26 + 69 + 8 = 103 non-BERT)
2. Feature extraction is consistent across calls
3. Binary features are 0/1
4. No NaN or Inf values
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.features import CategoryFeatureExtractor, FeatureConfig


def test_feature_dimensions():
    """Verify feature dimensions match FeatureConfig."""
    print("Testing feature dimensions...", flush=True)
    extractor = CategoryFeatureExtractor()

    text = "Can you help me with this?"
    context = ["Hey", "What's up?"]

    # Test individual feature groups
    hand_crafted = extractor.extract_hand_crafted(text, context)
    assert len(hand_crafted) == FeatureConfig.HAND_CRAFTED_DIM, \
        f"Hand-crafted features: expected {FeatureConfig.HAND_CRAFTED_DIM}, got {len(hand_crafted)}"

    spacy_feats = extractor.extract_spacy_features(text)
    assert len(spacy_feats) == FeatureConfig.SPACY_DIM, \
        f"SpaCy features: expected {FeatureConfig.SPACY_DIM}, got {len(spacy_feats)}"

    new_hand_crafted = extractor.extract_new_hand_crafted(text)
    assert len(new_hand_crafted) == FeatureConfig.NEW_HAND_CRAFTED_DIM, \
        f"New hand-crafted features: expected {FeatureConfig.NEW_HAND_CRAFTED_DIM}, got {len(new_hand_crafted)}"

    # Test combined features
    all_features = extractor.extract_all(text, context)
    assert len(all_features) == FeatureConfig.TOTAL_NON_BERT, \
        f"Total non-BERT features: expected {FeatureConfig.TOTAL_NON_BERT}, got {len(all_features)}"

    print(f"✓ Feature dimensions correct:", flush=True)
    print(f"  Hand-crafted: {len(hand_crafted)}", flush=True)
    print(f"  SpaCy: {len(spacy_feats)}", flush=True)
    print(f"  New hand-crafted: {len(new_hand_crafted)}", flush=True)
    print(f"  Total non-BERT: {len(all_features)}", flush=True)


def test_consistency():
    """Verify feature extraction is consistent across calls."""
    print("\nTesting consistency...", flush=True)
    extractor = CategoryFeatureExtractor()

    text = "What's going on?"
    context = ["Hi"]

    # Extract twice
    features1 = extractor.extract_all(text, context)
    features2 = extractor.extract_all(text, context)

    assert np.allclose(features1, features2), "Features are not consistent across calls"
    print("✓ Feature extraction is consistent", flush=True)


def test_no_nan_or_inf():
    """Verify no NaN or Inf values in features."""
    print("\nTesting for NaN/Inf values...", flush=True)
    extractor = CategoryFeatureExtractor()

    test_cases = [
        ("", []),  # Empty text
        ("hi", []),  # Single word
        ("Can you help me with this project deadline?", ["Hey", "What's up?"]),  # Long text
        ("😭😭😭", []),  # Emoji only
        ("lol", []),  # Abbreviation
    ]

    for text, context in test_cases:
        features = extractor.extract_all(text, context)
        assert not np.isnan(features).any(), f"NaN values found in features for: {text}"
        assert not np.isinf(features).any(), f"Inf values found in features for: {text}"

    print("✓ No NaN or Inf values found", flush=True)


def test_binary_features():
    """Verify binary features are 0 or 1."""
    print("\nTesting binary features...", flush=True)
    extractor = CategoryFeatureExtractor()

    text = "Can you help me?"
    context = []

    features = extractor.extract_all(text, context)

    # Mobilization one-hots (indices 5-11 in hand-crafted = indices 5-11 in all)
    mobilization_one_hots = features[5:12]
    for i, val in enumerate(mobilization_one_hots):
        assert val in (0.0, 1.0), f"Mobilization one-hot {i} is not binary: {val}"

    print("✓ Binary features are 0 or 1", flush=True)


def test_scaling_indices():
    """Verify scaling index ranges are correct."""
    print("\nTesting scaling indices...", flush=True)

    bert_indices, binary_indices, scale_indices = FeatureConfig.get_scaling_indices()

    # Verify ranges
    assert len(bert_indices) == FeatureConfig.BERT_DIM
    assert len(binary_indices) == 7  # Mobilization one-hots
    assert len(scale_indices) == FeatureConfig.TOTAL_NON_BERT - 7  # All non-BERT except mobilization

    # Verify no overlap
    all_indices = set(bert_indices) | set(binary_indices) | set(scale_indices)
    assert len(all_indices) == FeatureConfig.TOTAL_DIM, \
        f"Index overlap detected: {len(all_indices)} != {FeatureConfig.TOTAL_DIM}"

    print("✓ Scaling indices are correct", flush=True)
    print(f"  BERT indices: {len(bert_indices)}", flush=True)
    print(f"  Binary indices: {len(binary_indices)}", flush=True)
    print(f"  Scale indices: {len(scale_indices)}", flush=True)


def main():
    print("="*70, flush=True)
    print("FEATURE EXTRACTION VALIDATION", flush=True)
    print("="*70, flush=True)

    try:
        test_feature_dimensions()
        test_consistency()
        test_no_nan_or_inf()
        test_binary_features()
        test_scaling_indices()

        print("\n" + "="*70, flush=True)
        print("✓ ALL TESTS PASSED", flush=True)
        print("="*70, flush=True)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
