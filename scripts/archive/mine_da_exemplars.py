#!/usr/bin/env python3
"""Mine dialogue act exemplars from high-purity clusters.

Problem: The DA classifier has too many STATEMENT exemplars (87%),
causing most responses to be classified as STATEMENT.

Solution: Extract additional exemplars from high-purity clusters
identified in our data analysis to improve classification of
AGREE, DECLINE, DEFER, QUESTION, etc.

High-purity clusters identified (from FROM_SCRATCH_PLAN):
- AGREE: clusters 7, 9, 74 (~430 exemplars)
- ACKNOWLEDGE: clusters 19, 11, 55, 38 (~600 exemplars)
- QUESTION: clusters 45, 80, 73 (~450 exemplars)
- REACT_POSITIVE: clusters 32, 12, 160, 18 (~500 exemplars)
- ANSWER: cluster 213 (~200 exemplars)

Also mine from structural patterns for DECLINE and DEFER which
are rare in clusters.

Usage:
    uv run python -m scripts.mine_da_exemplars --analyze      # Show cluster purity
    uv run python -m scripts.mine_da_exemplars --extract      # Extract exemplars
    uv run python -m scripts.mine_da_exemplars --validate     # Validate with structural
    uv run python -m scripts.mine_da_exemplars --output DIR   # Save to directory
"""

import argparse
import json
import re
from pathlib import Path

# Cluster mappings: DA type -> list of high-purity cluster IDs
# These were identified from cluster analysis in FROM_SCRATCH_PLAN
HIGH_PURITY_CLUSTERS = {
    "AGREE": [7, 9, 74],
    "ACKNOWLEDGE": [19, 11, 55, 38],
    "QUESTION": [45, 80, 73],
    "REACT_POSITIVE": [32, 12, 160, 18],
    "ANSWER": [213],
}

# Structural patterns for validation and mining rare types
# These are HIGH PRECISION patterns that indicate a DA type
STRUCTURAL_VALIDATORS = {
    "AGREE": [
        r"^(yes|yeah|yep|yup|yea|ya)[\s!.]*$",
        r"^(sure|definitely|absolutely|of course)[\s!.]*$",
        r"^(i'm down|im down|down)[\s!.]*$",
        r"^(sounds good|sounds great|let's do it)[\s!.]*$",
        r"^(count me in|i'm in|works for me)[\s!.]*$",
        r"^(for sure|100%|bet|deal)[\s!.]*$",
    ],
    "DECLINE": [
        r"^(no|nope|nah|naw)[\s!.]*$",
        r"^(can't|cannot|cant)[\s!.,]*",
        r"^(i can't|i cannot|won't be able)",
        r"^(sorry.*can't|sorry.*cannot)",
        r"^(not (today|tonight|this time|gonna work))",
        r"^(i('m| am) (busy|not free))",
        r"^(i'll pass|hard pass|pass)[\s!.]*$",
        r"^(rain check)",
    ],
    "DEFER": [
        r"^(maybe|possibly|perhaps)[\s!.]*$",
        r"^(let me (check|see|think|get back))",
        r"^(i'll (see|check|let you know|think))",
        r"^(not sure|unsure)[\s!.,]*$",
        r"^(depends|it depends)[\s!.]*$",
        r"^(we'll see|might|could be)",
        r"^(gotta see|have to see|need to check)",
    ],
    "ACKNOWLEDGE": [
        r"^(ok|okay|k|kk)[\s!.]*$",
        r"^(got it|gotcha|alright|cool)[\s!.]*$",
        r"^(noted|understood|copy|roger)[\s!.]*$",
        r"^(no worries|no problem|np)[\s!.]*$",
    ],
    "QUESTION": [
        r"\?[\s]*$",  # Ends with ?
        r"^(what|when|where|who|why|how|which)\b",
    ],
    "REACT_POSITIVE": [
        r"^(congrats|congratulations)",
        r"^(that's (awesome|amazing|great))",
        r"^(so (happy|excited|proud))",
        r"^(omg|no way|yay|woohoo)[\s!.]*$",
        r"^(nice|sick|dope|fire|lit)[\s!.]*$",
    ],
    "REACT_SYMPATHY": [
        r"^(i'm sorry|im sorry|so sorry)",
        r"^(that (sucks|stinks|is rough))",
        r"^(here for you|thinking of you)",
    ],
}

# Compile patterns
_COMPILED_VALIDATORS = {}
for da_type, patterns in STRUCTURAL_VALIDATORS.items():
    _COMPILED_VALIDATORS[da_type] = [
        re.compile(p, re.IGNORECASE) for p in patterns
    ]


def validate_with_structural(text: str, expected_da: str) -> bool:
    """Check if text matches structural patterns for expected DA type.

    Args:
        text: Response text to validate.
        expected_da: Expected DA type.

    Returns:
        True if text matches structural patterns for expected_da.
    """
    if expected_da not in _COMPILED_VALIDATORS:
        return True  # No validator, assume valid

    text_clean = text.strip().lower()
    for pattern in _COMPILED_VALIDATORS[expected_da]:
        if pattern.search(text_clean):
            return True
    return False


def mine_from_structural(texts: list[str], da_type: str) -> list[str]:
    """Mine exemplars that match structural patterns for a DA type.

    Args:
        texts: List of response texts to search.
        da_type: DA type to mine for.

    Returns:
        List of texts that match structural patterns.
    """
    if da_type not in _COMPILED_VALIDATORS:
        return []

    matches = []
    for text in texts:
        text_clean = text.strip().lower()
        for pattern in _COMPILED_VALIDATORS[da_type]:
            if pattern.search(text_clean):
                matches.append(text)
                break
    return matches


def analyze_cluster_purity():
    """Analyze purity of clusters by checking structural pattern matches."""
    from jarvis.db import get_db

    print("=" * 70)
    print("CLUSTER PURITY ANALYSIS")
    print("=" * 70)

    db = get_db()
    db.init_schema()

    # Get all pairs with cluster assignments
    pairs = db.get_all_pairs(min_quality=0.0)
    pairs_with_cluster = [p for p in pairs if p.cluster_id is not None and p.cluster_id >= 0]

    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Pairs with cluster: {len(pairs_with_cluster)}")

    # Group by cluster
    clusters: dict[int, list] = {}
    for p in pairs_with_cluster:
        if p.cluster_id not in clusters:
            clusters[p.cluster_id] = []
        clusters[p.cluster_id].append(p)

    print(f"Number of clusters: {len(clusters)}")

    # For each target DA type, find clusters with high purity
    print("\n" + "-" * 70)
    print("DA TYPE PURITY BY CLUSTER")
    print("-" * 70)

    for da_type, patterns in _COMPILED_VALIDATORS.items():
        print(f"\n{da_type}:")
        cluster_purity = []

        for cluster_id, cluster_pairs in clusters.items():
            if len(cluster_pairs) < 5:
                continue

            # Count matches
            matches = sum(
                1 for p in cluster_pairs
                if any(pat.search(p.response_text.strip().lower()) for pat in patterns)
            )
            purity = matches / len(cluster_pairs)

            if purity >= 0.3:  # At least 30% purity
                cluster_purity.append((cluster_id, len(cluster_pairs), matches, purity))

        # Sort by purity
        cluster_purity.sort(key=lambda x: x[3], reverse=True)

        for cluster_id, size, matches, purity in cluster_purity[:5]:
            print(f"  Cluster {cluster_id:4d}: {matches:3d}/{size:3d} ({purity:.0%} purity)")
            # Show sample
            sample = [p.response_text for p in clusters[cluster_id][:3]]
            for s in sample:
                print(f"    - {s[:60]}")


def extract_exemplars_from_clusters(output_dir: Path | None = None):
    """Extract exemplars from high-purity clusters.

    Args:
        output_dir: Directory to save exemplars. Defaults to ~/.jarvis/da_exemplars/
    """
    from jarvis.db import get_db

    print("=" * 70)
    print("EXTRACTING EXEMPLARS FROM CLUSTERS")
    print("=" * 70)

    if output_dir is None:
        output_dir = Path.home() / ".jarvis" / "da_exemplars"
    output_dir.mkdir(parents=True, exist_ok=True)

    db = get_db()
    db.init_schema()

    # Get all pairs with cluster assignments
    pairs = db.get_all_pairs(min_quality=0.0)
    pairs_with_cluster = [p for p in pairs if p.cluster_id is not None and p.cluster_id >= 0]

    # Group by cluster
    clusters: dict[int, list] = {}
    for p in pairs_with_cluster:
        if p.cluster_id not in clusters:
            clusters[p.cluster_id] = []
        clusters[p.cluster_id].append(p)

    # Extract from configured high-purity clusters
    exemplars: dict[str, list[str]] = {}

    for da_type, cluster_ids in HIGH_PURITY_CLUSTERS.items():
        exemplars[da_type] = []
        for cluster_id in cluster_ids:
            if cluster_id in clusters:
                for p in clusters[cluster_id]:
                    # Validate with structural if available
                    if validate_with_structural(p.response_text, da_type):
                        exemplars[da_type].append(p.response_text)
                    elif da_type not in _COMPILED_VALIDATORS:
                        # No validator, include all
                        exemplars[da_type].append(p.response_text)

        # Deduplicate
        exemplars[da_type] = list(set(exemplars[da_type]))
        print(f"{da_type}: {len(exemplars[da_type])} exemplars from {len(cluster_ids)} clusters")

    # Also mine DECLINE and DEFER from structural patterns (rare in clusters)
    all_responses = [p.response_text for p in pairs]
    for da_type in ["DECLINE", "DEFER"]:
        if da_type not in exemplars:
            exemplars[da_type] = []

        mined = mine_from_structural(all_responses, da_type)
        existing = set(exemplars[da_type])
        new_mined = [m for m in mined if m not in existing]
        exemplars[da_type].extend(new_mined)
        exemplars[da_type] = list(set(exemplars[da_type]))
        total = len(exemplars[da_type])
        print(f"{da_type}: +{len(new_mined)} from structural mining, total {total}")

    # Save
    for da_type, texts in exemplars.items():
        output_file = output_dir / f"{da_type.lower()}_exemplars.json"
        with open(output_file, "w") as f:
            json.dump(texts, f, indent=2)
        print(f"  Saved {len(texts)} exemplars to {output_file}")

    # Also save combined file
    combined_file = output_dir / "all_mined_exemplars.json"
    with open(combined_file, "w") as f:
        json.dump(exemplars, f, indent=2)
    print(f"\nSaved combined exemplars to {combined_file}")

    return exemplars


def validate_extracted_exemplars(output_dir: Path | None = None):
    """Validate extracted exemplars against structural patterns.

    Args:
        output_dir: Directory containing exemplars.
    """
    if output_dir is None:
        output_dir = Path.home() / ".jarvis" / "da_exemplars"

    print("=" * 70)
    print("VALIDATING EXTRACTED EXEMPLARS")
    print("=" * 70)

    combined_file = output_dir / "all_mined_exemplars.json"
    if not combined_file.exists():
        print(f"No exemplars found at {combined_file}")
        print("Run --extract first.")
        return

    with open(combined_file) as f:
        exemplars = json.load(f)

    for da_type, texts in exemplars.items():
        if da_type not in _COMPILED_VALIDATORS:
            print(f"\n{da_type}: No structural validator, skipping")
            continue

        valid_count = sum(1 for t in texts if validate_with_structural(t, da_type))
        purity = valid_count / len(texts) if texts else 0

        print(f"\n{da_type}: {valid_count}/{len(texts)} pass structural validation ({purity:.0%})")

        # Show failures
        failures = [t for t in texts if not validate_with_structural(t, da_type)][:5]
        if failures:
            print("  Sample failures:")
            for f in failures:
                print(f"    - {f[:60]}")


def show_current_da_distribution():
    """Show current DA distribution in the database."""
    from jarvis.db import get_db

    print("=" * 70)
    print("CURRENT DA DISTRIBUTION")
    print("=" * 70)

    db = get_db()
    db.init_schema()

    dist = db.get_da_distribution()

    print("\nResponse DA Types:")
    total = sum(dist["response_da"].values())
    for da_type, count in sorted(dist["response_da"].items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {da_type:20} {count:6} ({pct:5.1f}%)")

    print(f"\nTotal classified: {dist['total_classified']}")

    # Cross-tabulation
    print("\n" + "-" * 70)
    print("TRIGGER -> RESPONSE CROSS-TAB")
    print("-" * 70)

    cross_tab = db.get_da_cross_tabulation()
    for trigger_da, response_counts in sorted(cross_tab.items()):
        print(f"\n{trigger_da}:")
        for response_da, count in list(response_counts.items())[:5]:
            print(f"  -> {response_da:20} {count:4}")


def main():
    parser = argparse.ArgumentParser(description="Mine DA exemplars from clusters")
    parser.add_argument("--analyze", action="store_true", help="Analyze cluster purity")
    parser.add_argument("--extract", action="store_true", help="Extract exemplars from clusters")
    parser.add_argument("--validate", action="store_true", help="Validate extracted exemplars")
    parser.add_argument("--distribution", action="store_true", help="Show current DA distribution")
    parser.add_argument("--output", type=Path, help="Output directory for exemplars")

    args = parser.parse_args()

    if args.distribution:
        show_current_da_distribution()

    if args.analyze:
        analyze_cluster_purity()

    if args.extract:
        extract_exemplars_from_clusters(args.output)

    if args.validate:
        validate_extracted_exemplars(args.output)

    if not any([args.analyze, args.extract, args.validate, args.distribution]):
        parser.print_help()


if __name__ == "__main__":
    main()
