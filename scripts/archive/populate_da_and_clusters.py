#!/usr/bin/env python3
"""Populate DA classifications and cluster assignments in the database.

Runs DA classifier on all pairs and saves results.
Also loads HDBSCAN cluster assignments if available.

Usage:
    uv run python -m scripts.populate_da_and_clusters
    uv run python -m scripts.populate_da_and_clusters --batch-size 1000
    uv run python -m scripts.populate_da_and_clusters --clusters-only  # Just load clusters
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def populate_da_classifications(batch_size: int = 500) -> dict:
    """Classify all pairs and save to database."""
    from jarvis.db import get_db
    from scripts.build_da_classifier import DialogueActClassifier

    db = get_db()

    # Load classifiers
    logger.info("Loading DA classifiers...")
    trigger_clf = DialogueActClassifier("trigger")
    response_clf = DialogueActClassifier("response")

    # Get all pairs
    logger.info("Loading pairs from database...")
    pairs = db.get_training_pairs(min_quality=0.0)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Process in batches
    total_updated = 0
    start_time = time.time()

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]

        # Classify batch
        triggers = [p.trigger_text for p in batch]
        responses = [p.response_text for p in batch]

        trigger_results = trigger_clf.classify_batch(triggers)
        response_results = response_clf.classify_batch(responses)

        # Prepare updates
        updates = [
            (
                p.id,
                tr.label,
                tr.confidence,
                rr.label,
                rr.confidence,
            )
            for p, tr, rr in zip(batch, trigger_results, response_results)
        ]

        # Save to database
        updated = db.update_da_classifications(updates)
        total_updated += updated

        progress = min(i + batch_size, len(pairs))
        elapsed = time.time() - start_time
        rate = progress / elapsed if elapsed > 0 else 0
        logger.info(f"Classified {progress}/{len(pairs)} ({rate:.0f} pairs/sec)")

    elapsed = time.time() - start_time
    logger.info(f"Done! Updated {total_updated} pairs in {elapsed:.1f}s")

    # Get distribution
    dist = db.get_da_distribution()

    return {
        "total_pairs": len(pairs),
        "total_updated": total_updated,
        "elapsed_seconds": elapsed,
        "trigger_da_distribution": dist["trigger_da"],
        "response_da_distribution": dist["response_da"],
    }


def populate_cluster_assignments() -> dict:
    """Load HDBSCAN cluster assignments and save to database."""
    from jarvis.db import get_db

    db = get_db()
    cluster_dir = Path.home() / ".jarvis" / "response_clusters"
    labels_path = cluster_dir / "hdbscan_labels.npy"

    if not labels_path.exists():
        logger.warning(f"No cluster labels found at {labels_path}")
        logger.info("Run: uv run python -m scripts.cluster_response_types --save")
        return {"error": "No cluster labels found"}

    # Load labels
    labels = np.load(labels_path)
    logger.info(f"Loaded {len(labels)} cluster labels")

    # Get pairs (in same order as clustering was done)
    pairs = db.get_training_pairs(min_quality=0.0)

    if len(pairs) != len(labels):
        logger.warning(f"Mismatch: {len(pairs)} pairs vs {len(labels)} labels")
        logger.info("Cluster labels may be stale. Re-run clustering.")
        return {"error": f"Mismatch: {len(pairs)} pairs vs {len(labels)} labels"}

    # Prepare assignments
    assignments = [(p.id, int(label)) for p, label in zip(pairs, labels)]

    # Save to database
    updated = db.update_cluster_assignments(assignments)

    # Count clusters
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = sum(1 for label in labels if label == -1)

    logger.info(f"Updated {updated} pairs with cluster assignments")
    logger.info(f"Clusters: {n_clusters}, Noise: {n_noise} ({100*n_noise/len(labels):.1f}%)")

    return {
        "total_pairs": len(pairs),
        "total_updated": updated,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }


def print_analysis():
    """Print DA vs cluster analysis."""
    from collections import Counter

    from jarvis.db import get_db

    db = get_db()
    pairs = db.get_training_pairs(min_quality=0.0)

    # Check how many have DA and cluster
    has_da = sum(1 for p in pairs if p.response_da_type is not None)
    has_cluster = sum(1 for p in pairs if p.cluster_id is not None)

    print(f"\n{'='*70}")
    print("DA AND CLUSTER COVERAGE")
    print(f"{'='*70}")
    print(f"Total pairs: {len(pairs)}")
    print(f"With DA classification: {has_da} ({100*has_da/len(pairs):.1f}%)")
    print(f"With cluster assignment: {has_cluster} ({100*has_cluster/len(pairs):.1f}%)")

    if has_da > 0 and has_cluster > 0:
        # Cross-tabulation
        print(f"\n{'='*70}")
        print("CLUSTER vs DA CROSS-TAB (Response types)")
        print(f"{'='*70}")

        cluster_da = {}
        for p in pairs:
            if p.cluster_id is not None and p.response_da_type is not None:
                if p.cluster_id not in cluster_da:
                    cluster_da[p.cluster_id] = Counter()
                cluster_da[p.cluster_id][p.response_da_type] += 1

        # Sort by cluster size
        sorted_clusters = sorted(
            cluster_da.keys(), key=lambda x: sum(cluster_da[x].values()), reverse=True
        )
        for cluster_id in sorted_clusters:
            if cluster_id == -1:
                continue
            dist = cluster_da[cluster_id]
            total = sum(dist.values())
            dominant, dom_count = dist.most_common(1)[0]
            purity = dom_count / total

            print(f"\n  Cluster {cluster_id} (n={total}, purity={purity:.0%})")
            for da, count in dist.most_common(3):
                print(f"    {da:20} {count:5} ({100*count/total:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Populate DA and cluster fields")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for classification")
    parser.add_argument("--da-only", action="store_true", help="Only run DA classification")
    parser.add_argument(
        "--clusters-only", action="store_true", help="Only load cluster assignments"
    )
    parser.add_argument("--analyze", action="store_true", help="Print analysis after populating")
    args = parser.parse_args()

    if args.clusters_only:
        result = populate_cluster_assignments()
        print(f"\nResult: {result}")
    elif args.da_only:
        result = populate_da_classifications(batch_size=args.batch_size)
        print(f"\nResult: {result}")
    else:
        # Do both
        print("\n" + "="*70)
        print("STEP 1: DA Classification")
        print("="*70)
        da_result = populate_da_classifications(batch_size=args.batch_size)

        print("\n" + "="*70)
        print("STEP 2: Cluster Assignments")
        print("="*70)
        cluster_result = populate_cluster_assignments()

        print(f"\n\nDA Result: {da_result}")
        print(f"Cluster Result: {cluster_result}")

    if args.analyze:
        print_analysis()


if __name__ == "__main__":
    main()
