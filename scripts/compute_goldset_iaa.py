"""Compute Inter-Annotator Agreement and merge annotations from 3 annotators.

Computes span-level agreement metrics (Cohen's kappa pairwise, Fleiss' kappa
for all 3), then produces a merged goldset using majority vote (2/3 agree).

Usage:
    uv run python scripts/compute_goldset_iaa.py \
        --annotator1 training_data/goldset_v6/annotator_a.json \
        --annotator2 training_data/goldset_v6/annotator_b.json \
        --annotator3 training_data/goldset_v6/annotator_c.json \
        --output-dir training_data/goldset_v6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

# Import span matching from eval_shared (same directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from eval_shared import DEFAULT_LABEL_ALIASES, spans_match

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotator_file(path: str) -> list[dict]:
    """Load an annotator JSON file and return list of samples."""
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples from {os.path.basename(path)}", flush=True)
    return data


def index_by_sample_id(data: list[dict]) -> dict[str, dict]:
    """Index samples by sample_id for fast lookup."""
    return {s["sample_id"]: s for s in data}


# ---------------------------------------------------------------------------
# Span matching between annotators
# ---------------------------------------------------------------------------

def find_matching_span(
    span: dict,
    candidate_spans: list[dict],
    label_aliases: dict[str, set[str]] | None = None,
) -> dict | None:
    """Find a matching span in candidate_spans for the given span."""
    for cand in candidate_spans:
        if spans_match(
            span["span_text"], span["span_label"],
            cand["span_text"], cand["span_label"],
            label_aliases=label_aliases,
        ):
            return cand
    return None


def build_span_universe(
    spans_a: list[dict],
    spans_b: list[dict],
    label_aliases: dict[str, set[str]] | None = None,
) -> list[tuple[str, str]]:
    """Build the universe of unique spans across two annotators.

    Returns a list of (canonical_text, canonical_label) tuples representing
    every distinct span mentioned by either annotator.
    """
    universe: list[tuple[str, str]] = []
    matched_b_indices: set[int] = set()

    for sa in spans_a:
        key = (sa["span_text"].lower().strip(), sa["span_label"])
        if key not in universe:
            universe.append(key)
        # Track which B spans are matched to A
        for i, sb in enumerate(spans_b):
            if i not in matched_b_indices and spans_match(
                sa["span_text"], sa["span_label"],
                sb["span_text"], sb["span_label"],
                label_aliases=label_aliases,
            ):
                matched_b_indices.add(i)

    # Add unmatched B spans
    for i, sb in enumerate(spans_b):
        if i not in matched_b_indices:
            key = (sb["span_text"].lower().strip(), sb["span_label"])
            if key not in universe:
                universe.append(key)

    return universe


def build_span_universe_three(
    spans_list: list[list[dict]],
    label_aliases: dict[str, set[str]] | None = None,
) -> list[tuple[str, str, list[dict]]]:
    """Build the universe of unique spans across 3 annotators.

    Returns list of (canonical_text, canonical_label, [representative_span_per_annotator_or_None]).
    Each entry tracks which annotators marked that span.
    """
    # Collect all spans with their annotator index
    all_spans: list[tuple[dict, int]] = []
    for idx, spans in enumerate(spans_list):
        for s in spans:
            all_spans.append((s, idx))

    # Cluster spans that match each other
    clusters: list[list[tuple[dict, int]]] = []
    assigned: set[int] = set()

    for i, (si, ai) in enumerate(all_spans):
        if i in assigned:
            continue
        cluster = [(si, ai)]
        assigned.add(i)
        for j, (sj, aj) in enumerate(all_spans):
            if j in assigned:
                continue
            if spans_match(
                si["span_text"], si["span_label"],
                sj["span_text"], sj["span_label"],
                label_aliases=label_aliases,
            ):
                cluster.append((sj, aj))
                assigned.add(j)
        clusters.append(cluster)

    # Build universe entries
    universe: list[tuple[str, str, list[dict | None]]] = []
    for cluster in clusters:
        # Use the first span as canonical
        canonical = cluster[0][0]
        text = canonical["span_text"]
        label = canonical["span_label"]
        # Track per-annotator presence
        per_annotator: list[dict | None] = [None, None, None]
        for span, ann_idx in cluster:
            if per_annotator[ann_idx] is None:
                per_annotator[ann_idx] = span
        universe.append((text, label, per_annotator))

    return universe


# ---------------------------------------------------------------------------
# Cohen's kappa (pairwise, binary per-span)
# ---------------------------------------------------------------------------

def cohens_kappa_binary(ratings_a: list[int], ratings_b: list[int]) -> float:
    """Compute Cohen's kappa for two raters with binary labels (0/1).

    Args:
        ratings_a: List of 0/1 ratings from annotator A.
        ratings_b: List of 0/1 ratings from annotator B.

    Returns:
        Cohen's kappa coefficient.
    """
    n = len(ratings_a)
    if n == 0:
        return 0.0

    # Confusion matrix counts
    n11 = sum(1 for a, b in zip(ratings_a, ratings_b) if a == 1 and b == 1)
    n00 = sum(1 for a, b in zip(ratings_a, ratings_b) if a == 0 and b == 0)
    n10 = sum(1 for a, b in zip(ratings_a, ratings_b) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(ratings_a, ratings_b) if a == 0 and b == 1)

    po = (n11 + n00) / n  # observed agreement
    # Expected agreement
    pa1 = (n11 + n10) / n
    pb1 = (n11 + n01) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def compute_pairwise_kappa(
    ann_data: list[dict[str, dict]],
    label_aliases: dict[str, set[str]] | None = None,
) -> list[tuple[str, str, float]]:
    """Compute Cohen's kappa for each pair of annotators.

    Args:
        ann_data: List of 3 dicts mapping sample_id -> sample.

    Returns:
        List of (name_a, name_b, kappa) tuples.
    """
    names = ["ann1", "ann2", "ann3"]
    results = []

    for (i, name_a), (j, name_b) in combinations(enumerate(names), 2):
        all_ratings_a: list[int] = []
        all_ratings_b: list[int] = []

        # Only compare samples both annotators have
        common_ids = set(ann_data[i].keys()) & set(ann_data[j].keys())

        for sid in sorted(common_ids):
            sa = ann_data[i][sid]
            sb = ann_data[j][sid]
            spans_a = sa.get("expected_candidates", [])
            spans_b = sb.get("expected_candidates", [])

            # Build span universe for this sample pair
            universe = build_span_universe(spans_a, spans_b, label_aliases)

            for text, label in universe:
                # Did annotator A mark this span?
                present_a = 1 if find_matching_span(
                    {"span_text": text, "span_label": label},
                    spans_a, label_aliases,
                ) else 0
                present_b = 1 if find_matching_span(
                    {"span_text": text, "span_label": label},
                    spans_b, label_aliases,
                ) else 0
                all_ratings_a.append(present_a)
                all_ratings_b.append(present_b)

        kappa = cohens_kappa_binary(all_ratings_a, all_ratings_b)
        results.append((name_a, name_b, kappa))
        print(
            f"  Cohen's kappa ({name_a} vs {name_b}): {kappa:.4f} "
            f"({len(all_ratings_a)} span decisions across {len(common_ids)} samples)",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# Fleiss' kappa (all 3 annotators)
# ---------------------------------------------------------------------------

def fleiss_kappa(
    ann_data: list[dict[str, dict]],
    label_aliases: dict[str, set[str]] | None = None,
) -> float:
    """Compute Fleiss' kappa for 3 annotators on binary span-level agreement.

    For each unique span across all 3 annotators, we count how many annotators
    marked that span as present. This gives us an N x 2 matrix (present/absent)
    which we use for Fleiss' kappa.

    Fleiss' kappa formula:
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    where:
        P_bar = mean of per-item agreement proportions
        P_e_bar = sum of squared category proportions (chance agreement)
    """
    n_raters = 3  # k = number of raters

    # Collect per-span rating vectors: each entry is [n_present, n_absent]
    rating_matrix: list[list[int]] = []  # N x 2 (present, absent)

    common_ids = set(ann_data[0].keys()) & set(ann_data[1].keys()) & set(ann_data[2].keys())

    for sid in sorted(common_ids):
        spans_list = [
            ann_data[a][sid].get("expected_candidates", [])
            for a in range(3)
        ]
        universe = build_span_universe_three(spans_list, label_aliases)

        for _text, _label, per_annotator in universe:
            n_present = sum(1 for p in per_annotator if p is not None)
            n_absent = n_raters - n_present
            rating_matrix.append([n_present, n_absent])

    n_items = len(rating_matrix)
    if n_items == 0:
        return 0.0

    # P_i for each item: proportion of agreeing pairs
    # P_i = (1 / (k*(k-1))) * sum_j(n_ij^2) - k)  ... simplified for binary
    p_items: list[float] = []
    for row in rating_matrix:
        sum_sq = sum(r * r for r in row)
        p_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1))
        p_items.append(p_i)

    p_bar = sum(p_items) / n_items

    # P_e_bar: sum of squared proportions per category
    # p_j = (1 / (N * k)) * sum_i(n_ij)
    total_ratings = n_items * n_raters
    p_present = sum(row[0] for row in rating_matrix) / total_ratings
    p_absent = sum(row[1] for row in rating_matrix) / total_ratings
    p_e_bar = p_present ** 2 + p_absent ** 2

    if p_e_bar == 1.0:
        return 1.0

    kappa = (p_bar - p_e_bar) / (1 - p_e_bar)
    return kappa


# ---------------------------------------------------------------------------
# Majority-vote merging
# ---------------------------------------------------------------------------

def merge_annotations(
    ann_data: list[dict[str, dict]],
    label_aliases: dict[str, set[str]] | None = None,
) -> list[dict]:
    """Merge 3 annotator files using majority vote.

    - 3/3 agree: high confidence
    - 2/3 agree: medium confidence (majority vote)
    - 1/3 (all disagree): flagged for review, still included with low confidence

    Returns merged samples sorted by sample_id.
    """
    # Use union of all sample IDs, prefer samples present in all 3
    all_ids: set[str] = set()
    for ad in ann_data:
        all_ids.update(ad.keys())

    merged: list[dict] = []

    for sid in sorted(all_ids):
        # Get the sample from whichever annotator has it (prefer first)
        base_sample = None
        for ad in ann_data:
            if sid in ad:
                base_sample = ad[sid]
                break
        if base_sample is None:
            continue

        # Collect span lists from each annotator that has this sample
        spans_list: list[list[dict]] = []
        annotator_present: list[int] = []
        for idx, ad in enumerate(ann_data):
            if sid in ad:
                spans_list.append(ad[sid].get("expected_candidates", []))
                annotator_present.append(idx)
            else:
                spans_list.append([])

        n_annotators = len(annotator_present)

        # Build span universe across all available annotators
        universe = build_span_universe_three(spans_list, label_aliases)

        merged_candidates: list[dict] = []
        any_review_needed = False

        for text, label, per_annotator in universe:
            agreement = sum(1 for p in per_annotator if p is not None)

            if agreement >= 2:
                # Majority vote or unanimous: accept
                if agreement == 3:
                    confidence = "high"
                else:
                    confidence = "medium"
                merged_candidates.append({
                    "span_text": text,
                    "span_label": label,
                    "agreement": agreement,
                    "confidence": confidence,
                })
            else:
                # Only 1 annotator marked this span
                if n_annotators >= 3:
                    # All 3 had a chance, only 1 marked it: flag for review
                    any_review_needed = True
                    merged_candidates.append({
                        "span_text": text,
                        "span_label": label,
                        "agreement": agreement,
                        "confidence": "low",
                    })
                else:
                    # Fewer than 3 annotators had this sample; still include
                    merged_candidates.append({
                        "span_text": text,
                        "span_label": label,
                        "agreement": agreement,
                        "confidence": "low",
                    })
                    any_review_needed = True

        # Check for complete disagreement: each annotator has different spans,
        # none overlap. This is the "all 3 disagree" case.
        if n_annotators >= 3 and all(
            c["agreement"] == 1 for c in merged_candidates
        ) and len(merged_candidates) > 0:
            any_review_needed = True

        # Filter to only majority-accepted spans for the final expected_candidates,
        # but keep low-confidence spans in a separate field for review
        accepted = [c for c in merged_candidates if c["agreement"] >= 2]
        review_spans = [c for c in merged_candidates if c["agreement"] < 2]

        out = {
            "sample_id": sid,
            "message_id": base_sample.get("message_id"),
            "message_text": base_sample.get("message_text", ""),
            "is_from_me": base_sample.get("is_from_me", False),
            "context_prev": base_sample.get("context_prev", ""),
            "context_next": base_sample.get("context_next", ""),
            "slice": base_sample.get("slice", "unknown"),
            "expected_candidates": accepted + review_spans,
            "review_needed": any_review_needed,
        }

        merged.append(out)

    return merged


# ---------------------------------------------------------------------------
# Stratified train/dev/test split
# ---------------------------------------------------------------------------

def stratified_split(
    samples: list[dict],
    train_frac: float = 0.6,
    dev_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test stratified by slice and has_candidates.

    Stratification key: (slice, has_any_candidates)
    Uses manual stratification to avoid sklearn dependency.
    """
    import random

    rng = random.Random(seed)

    # Build stratification key
    strata: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for s in samples:
        has_cands = len([
            c for c in s.get("expected_candidates", [])
            if c.get("agreement", 0) >= 2
        ]) > 0
        key = (s.get("slice", "unknown"), has_cands)
        strata[key].append(s)

    train, dev, test = [], [], []

    for key, group in sorted(strata.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_frac))
        n_dev = max(1, round(n * dev_frac)) if n > 1 else 0
        # Ensure we don't exceed total
        if n_train + n_dev >= n:
            n_dev = max(0, n - n_train)
        n_test = n - n_train - n_dev

        train.extend(group[:n_train])
        dev.extend(group[n_train:n_train + n_dev])
        test.extend(group[n_train + n_dev:])

        print(
            f"  Stratum {key}: {n} samples -> "
            f"train={n_train}, dev={n_dev}, test={n_test}",
            flush=True,
        )

    # Shuffle within splits for good measure
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    return train, dev, test


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(merged: list[dict], kappas: list[tuple[str, str, float]], fk: float) -> None:
    """Print a summary report of IAA metrics and merged dataset stats."""
    print("\n" + "=" * 60, flush=True)
    print("INTER-ANNOTATOR AGREEMENT REPORT", flush=True)
    print("=" * 60, flush=True)

    print("\nPairwise Cohen's Kappa:", flush=True)
    for name_a, name_b, k in kappas:
        interpretation = (
            "almost perfect" if k > 0.8 else
            "substantial" if k > 0.6 else
            "moderate" if k > 0.4 else
            "fair" if k > 0.2 else
            "slight" if k > 0.0 else
            "poor"
        )
        print(f"  {name_a} vs {name_b}: {k:.4f} ({interpretation})", flush=True)

    avg_kappa = sum(k for _, _, k in kappas) / len(kappas) if kappas else 0.0
    print(f"  Average pairwise: {avg_kappa:.4f}", flush=True)

    interpretation = (
        "almost perfect" if fk > 0.8 else
        "substantial" if fk > 0.6 else
        "moderate" if fk > 0.4 else
        "fair" if fk > 0.2 else
        "slight" if fk > 0.0 else
        "poor"
    )
    print(f"\nFleiss' Kappa (all 3): {fk:.4f} ({interpretation})", flush=True)

    # Merged dataset stats
    total = len(merged)
    n_review = sum(1 for s in merged if s["review_needed"])
    total_spans = sum(len(s["expected_candidates"]) for s in merged)
    high = sum(
        1 for s in merged
        for c in s["expected_candidates"]
        if c.get("confidence") == "high"
    )
    medium = sum(
        1 for s in merged
        for c in s["expected_candidates"]
        if c.get("confidence") == "medium"
    )
    low = sum(
        1 for s in merged
        for c in s["expected_candidates"]
        if c.get("confidence") == "low"
    )

    print("\nMerged Dataset:", flush=True)
    print(f"  Total samples: {total}", flush=True)
    if total:
        pct = 100 * n_review / total
        print(f"  Needs review:  {n_review} ({pct:.1f}%)", flush=True)
    else:
        print("", flush=True)
    print(f"  Total spans:   {total_spans}", flush=True)
    print(f"    High (3/3):  {high}", flush=True)
    print(f"    Medium (2/3): {medium}", flush=True)
    print(f"    Low (1/3):   {low}", flush=True)

    # Slice distribution
    slice_counts: Counter[str] = Counter(s.get("slice", "unknown") for s in merged)
    print("\n  Slice distribution:", flush=True)
    for sl, cnt in sorted(slice_counts.items()):
        print(f"    {sl}: {cnt}", flush=True)

    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute IAA and merge 3 annotator label files into a goldset.",
    )
    parser.add_argument(
        "--annotator1", required=True,
        help="Path to annotator 1 JSON file",
    )
    parser.add_argument(
        "--annotator2", required=True,
        help="Path to annotator 2 JSON file",
    )
    parser.add_argument(
        "--annotator3", required=True,
        help="Path to annotator 3 JSON file",
    )
    parser.add_argument(
        "--output-dir", default="training_data/goldset_v6",
        help="Output directory for merged goldset and splits (default: training_data/goldset_v6)",
    )
    parser.add_argument(
        "--no-aliases", action="store_true",
        help="Disable label aliasing (require exact label match)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/dev/test split",
    )
    args = parser.parse_args()

    label_aliases = None if args.no_aliases else DEFAULT_LABEL_ALIASES

    # Load annotator files
    print("Loading annotator files...", flush=True)
    ann1 = load_annotator_file(args.annotator1)
    ann2 = load_annotator_file(args.annotator2)
    ann3 = load_annotator_file(args.annotator3)

    # Index by sample_id
    ann_indexed = [
        index_by_sample_id(ann1),
        index_by_sample_id(ann2),
        index_by_sample_id(ann3),
    ]

    # Compute pairwise Cohen's kappa
    print("\nComputing pairwise Cohen's kappa...", flush=True)
    kappas = compute_pairwise_kappa(ann_indexed, label_aliases)

    # Compute Fleiss' kappa
    print("\nComputing Fleiss' kappa...", flush=True)
    fk = fleiss_kappa(ann_indexed, label_aliases)
    print(f"  Fleiss' kappa: {fk:.4f}", flush=True)

    # Merge annotations
    print("\nMerging annotations with majority vote...", flush=True)
    merged = merge_annotations(ann_indexed, label_aliases)
    print(f"  Merged {len(merged)} samples", flush=True)

    # Print summary
    print_summary(merged, kappas, fk)

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)

    merged_path = os.path.join(args.output_dir, "goldset_v6_merged.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nWrote merged goldset: {merged_path} ({len(merged)} samples)", flush=True)

    # Stratified split
    print("\nSplitting into train/dev/test (60/20/20)...", flush=True)
    train, dev, test = stratified_split(merged, seed=args.seed)

    for name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        path = os.path.join(args.output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {path}: {len(split_data)} samples", flush=True)

    print(
        f"\nDone. Total: {len(train)} train / {len(dev)} dev / {len(test)} test",
        flush=True,
    )


if __name__ == "__main__":
    main()
