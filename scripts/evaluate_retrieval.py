#!/usr/bin/env python3
"""Evaluate retrieval quality on holdout set.

Uses batched encoding and FAISS search for efficiency.

Usage:
    uv run python -m scripts.evaluate_retrieval
    uv run python -m scripts.evaluate_retrieval --limit 1000
    uv run python -m scripts.evaluate_retrieval --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result for a single holdout pair."""

    pair_id: int
    trigger: str
    expected_response: str
    top_match_trigger: str | None
    top_match_response: str | None
    top_score: float
    top_k_scores: list[float]
    is_exact_match: bool  # Response matches exactly
    is_semantic_match: bool  # Similar meaning (score > threshold)


def batch_encode(texts: list[str], batch_size: int = 500) -> np.ndarray:
    """Encode texts in batches, showing progress."""
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = embedder.encode(batch, normalize=True)
        all_embeddings.append(embeddings)

        progress = min(i + batch_size, len(texts))
        if progress % 2000 == 0 or progress == len(texts):
            logger.info(f"  Encoded {progress}/{len(texts)} ({100 * progress / len(texts):.1f}%)")

    return np.vstack(all_embeddings).astype(np.float32)


def batch_search(
    query_embeddings: np.ndarray,
    index,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Search FAISS index with batched queries."""
    # FAISS search supports batch queries natively
    scores, indices = index.search(query_embeddings, k)
    return scores, indices


def run_evaluation(
    limit: int | None = None,
    threshold: float = 0.5,
    k: int = 5,
    batch_size: int = 500,
    measure_response_quality: bool = True,
) -> dict:
    """Run evaluation on holdout set.

    Args:
        limit: Max holdout pairs to evaluate (None = all)
        threshold: Score threshold for "semantic match"
        k: Number of results to retrieve per query
        batch_size: Batch size for encoding
        measure_response_quality: Also measure if retrieved responses are appropriate

    Returns:
        Evaluation metrics dict
    """
    import faiss

    from jarvis.db import get_db
    from jarvis.index import JARVIS_DIR

    db = get_db()

    # Get holdout pairs
    logger.info("Loading holdout pairs...")
    holdout_pairs = db.get_holdout_pairs()

    if not holdout_pairs:
        return {"error": "No holdout pairs found. Run train/test split first."}

    if limit:
        holdout_pairs = holdout_pairs[:limit]

    logger.info(f"Evaluating {len(holdout_pairs)} holdout pairs")

    # Load FAISS index
    logger.info("Loading FAISS index...")
    active_index = db.get_active_index()
    if not active_index:
        return {"error": "No active FAISS index. Run build_training_index first."}

    index_path = JARVIS_DIR / active_index.index_path
    index = faiss.read_index(str(index_path))
    logger.info(f"  Index has {index.ntotal} vectors")

    # Get training pairs for lookup
    logger.info("Loading training pairs for response lookup...")
    training_pairs = db.get_training_pairs()

    # Build faiss_id -> pair mapping
    # We need to get the embedding mappings
    pair_by_faiss_id = {}
    for pair in training_pairs:
        emb = db.get_embedding_by_pair(pair.id)
        if emb:
            pair_by_faiss_id[emb.faiss_id] = pair

    logger.info(f"  Mapped {len(pair_by_faiss_id)} pairs to FAISS IDs")

    # Batch encode all holdout triggers
    logger.info("Encoding holdout triggers...")
    start_time = time.time()
    triggers = [p.trigger_text for p in holdout_pairs]
    query_embeddings = batch_encode(triggers, batch_size=batch_size)
    encode_time = time.time() - start_time
    logger.info(f"  Encoded in {encode_time:.1f}s ({len(triggers) / encode_time:.0f} texts/sec)")

    # Batch search
    logger.info("Searching FAISS index...")
    start_time = time.time()
    scores, indices = batch_search(query_embeddings, index, k=k)
    search_time = time.time() - start_time
    logger.info(f"  Searched in {search_time:.2f}s ({len(triggers) / search_time:.0f} queries/sec)")

    # Analyze results
    logger.info("Analyzing results...")
    results: list[EvalResult] = []
    exact_matches = 0
    semantic_matches = 0
    score_buckets = {
        "0.9+": 0,
        "0.8-0.9": 0,
        "0.7-0.8": 0,
        "0.6-0.7": 0,
        "0.5-0.6": 0,
        "<0.5": 0,
    }
    all_top_scores = []

    for i, pair in enumerate(holdout_pairs):
        top_score = float(scores[i][0]) if indices[i][0] >= 0 else 0.0
        top_k_scores = [float(s) for s in scores[i] if s > 0]
        all_top_scores.append(top_score)

        # Get matched pair
        top_faiss_id = int(indices[i][0]) if indices[i][0] >= 0 else -1
        matched_pair = pair_by_faiss_id.get(top_faiss_id)

        # Check for exact match (same response text)
        is_exact = False
        if matched_pair:
            matched_resp = matched_pair.response_text.strip().lower()
            expected_resp = pair.response_text.strip().lower()
            is_exact = matched_resp == expected_resp

        is_semantic = top_score >= threshold

        if is_exact:
            exact_matches += 1
        if is_semantic:
            semantic_matches += 1

        # Score bucket
        if top_score >= 0.9:
            score_buckets["0.9+"] += 1
        elif top_score >= 0.8:
            score_buckets["0.8-0.9"] += 1
        elif top_score >= 0.7:
            score_buckets["0.7-0.8"] += 1
        elif top_score >= 0.6:
            score_buckets["0.6-0.7"] += 1
        elif top_score >= 0.5:
            score_buckets["0.5-0.6"] += 1
        else:
            score_buckets["<0.5"] += 1

        results.append(
            EvalResult(
                pair_id=pair.id,
                trigger=pair.trigger_text,
                expected_response=pair.response_text,
                top_match_trigger=matched_pair.trigger_text if matched_pair else None,
                top_match_response=matched_pair.response_text if matched_pair else None,
                top_score=top_score,
                top_k_scores=top_k_scores,
                is_exact_match=is_exact,
                is_semantic_match=is_semantic,
            )
        )

    # Compute metrics
    n = len(results)
    metrics = {
        "total_evaluated": n,
        "threshold": threshold,
        "exact_match_rate": exact_matches / n if n else 0,
        "semantic_match_rate": semantic_matches / n if n else 0,
        "mean_top_score": float(np.mean(all_top_scores)),
        "median_top_score": float(np.median(all_top_scores)),
        "p90_top_score": float(np.percentile(all_top_scores, 90)),
        "p10_top_score": float(np.percentile(all_top_scores, 10)),
        "score_distribution": score_buckets,
        "encode_time_sec": encode_time,
        "search_time_sec": search_time,
        "total_time_sec": encode_time + search_time,
    }

    # Measure response quality: is the retrieved response appropriate?
    if measure_response_quality:
        logger.info("Measuring response quality (expected vs retrieved)...")

        # Collect response pairs
        expected_responses = []
        retrieved_responses = []
        valid_indices = []

        for i, r in enumerate(results):
            if r.top_match_response:
                expected_responses.append(r.expected_response)
                retrieved_responses.append(r.top_match_response)
                valid_indices.append(i)

        if expected_responses:
            # Batch encode both sets
            logger.info(f"  Encoding {len(expected_responses)} response pairs...")
            all_responses = expected_responses + retrieved_responses
            response_embeddings = batch_encode(all_responses, batch_size=batch_size)

            # Split back
            n_responses = len(expected_responses)
            expected_emb = response_embeddings[:n_responses]
            retrieved_emb = response_embeddings[n_responses:]

            # Compute pairwise cosine similarity (dot product since normalized)
            response_similarities = np.sum(expected_emb * retrieved_emb, axis=1)

            # Compute metrics by trigger score bucket
            response_sim_by_trigger_bucket = {
                "0.9+": [],
                "0.8-0.9": [],
                "0.7-0.8": [],
                "0.6-0.7": [],
                "<0.6": [],
            }

            for i, sim in zip(valid_indices, response_similarities):
                trigger_score = results[i].top_score
                if trigger_score >= 0.9:
                    response_sim_by_trigger_bucket["0.9+"].append(sim)
                elif trigger_score >= 0.8:
                    response_sim_by_trigger_bucket["0.8-0.9"].append(sim)
                elif trigger_score >= 0.7:
                    response_sim_by_trigger_bucket["0.7-0.8"].append(sim)
                elif trigger_score >= 0.6:
                    response_sim_by_trigger_bucket["0.6-0.7"].append(sim)
                else:
                    response_sim_by_trigger_bucket["<0.6"].append(sim)

            # Store response quality metrics
            metrics["response_quality"] = {
                "mean_response_similarity": float(np.mean(response_similarities)),
                "median_response_similarity": float(np.median(response_similarities)),
                "p90_response_similarity": float(np.percentile(response_similarities, 90)),
                "p10_response_similarity": float(np.percentile(response_similarities, 10)),
                "by_trigger_bucket": {
                    bucket: {
                        "count": len(sims),
                        "mean": float(np.mean(sims)) if sims else 0,
                        "median": float(np.median(sims)) if sims else 0,
                    }
                    for bucket, sims in response_sim_by_trigger_bucket.items()
                },
            }

            # Add response similarity to individual results
            for i, sim in zip(valid_indices, response_similarities):
                results[i] = EvalResult(
                    pair_id=results[i].pair_id,
                    trigger=results[i].trigger,
                    expected_response=results[i].expected_response,
                    top_match_trigger=results[i].top_match_trigger,
                    top_match_response=results[i].top_match_response,
                    top_score=results[i].top_score,
                    top_k_scores=results[i].top_k_scores,
                    is_exact_match=results[i].is_exact_match,
                    is_semantic_match=results[i].is_semantic_match,
                )

            mean_sim = metrics["response_quality"]["mean_response_similarity"]
            logger.info(f"  Mean response similarity: {mean_sim:.3f}")

    return {
        "metrics": metrics,
        "results": results,
    }


def print_report(eval_output: dict) -> None:
    """Print evaluation report."""
    if "error" in eval_output:
        logger.error(eval_output["error"])
        return

    metrics = eval_output["metrics"]
    results = eval_output["results"]

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 60)

    print(f"\nPairs evaluated: {metrics['total_evaluated']}")
    print(f"Threshold: {metrics['threshold']}")

    print("\n--- Accuracy ---")
    print(f"Exact match rate:    {metrics['exact_match_rate']:.1%}")
    sem_rate = metrics["semantic_match_rate"]
    print(f"Semantic match rate: {sem_rate:.1%} (score >= {metrics['threshold']})")

    print("\n--- Score Statistics ---")
    print(f"Mean top score:   {metrics['mean_top_score']:.3f}")
    print(f"Median top score: {metrics['median_top_score']:.3f}")
    print(f"90th percentile:  {metrics['p90_top_score']:.3f}")
    print(f"10th percentile:  {metrics['p10_top_score']:.3f}")

    print("\n--- Trigger Score Distribution ---")
    for bucket, count in metrics["score_distribution"].items():
        pct = 100 * count / metrics["total_evaluated"]
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>8}: {count:5d} ({pct:5.1f}%) {bar}")

    # Response quality metrics
    if "response_quality" in metrics:
        rq = metrics["response_quality"]
        print("\n--- Response Quality (is retrieved response appropriate?) ---")
        print(f"Mean response similarity:   {rq['mean_response_similarity']:.3f}")
        print(f"Median response similarity: {rq['median_response_similarity']:.3f}")
        print(f"90th percentile:            {rq['p90_response_similarity']:.3f}")
        print(f"10th percentile:            {rq['p10_response_similarity']:.3f}")

        print("\n--- Response Similarity by Trigger Score ---")
        print("  (Does higher trigger match → better response match?)")
        for bucket, stats in rq["by_trigger_bucket"].items():
            if stats["count"] > 0:
                print(
                    f"  Trigger {bucket:>8}: response_sim={stats['mean']:.3f} (n={stats['count']})"
                )

    print("\n--- Performance ---")
    print(f"Encode time: {metrics['encode_time_sec']:.1f}s")
    print(f"Search time: {metrics['search_time_sec']:.2f}s")
    print(f"Total time:  {metrics['total_time_sec']:.1f}s")

    # Show some examples
    print("\n--- Example Matches (high score) ---")
    high_score = [r for r in results if r.top_score >= 0.9][:3]
    for r in high_score:
        print(f'\n  Query: "{r.trigger[:60]}..."')
        match_t = r.top_match_trigger[:60] if r.top_match_trigger else "N/A"
        print(f'  Match: "{match_t}..." [{r.top_score:.3f}]')
        print(f'  Expected: "{r.expected_response[:40]}..."')
        match_r = r.top_match_response[:40] if r.top_match_response else "N/A"
        print(f'  Got:      "{match_r}..."')

    print("\n--- Example Matches (low score) ---")
    low_score = [r for r in results if r.top_score < 0.5][:3]
    for r in low_score:
        print(f'\n  Query: "{r.trigger[:60]}..."')
        match_t = r.top_match_trigger[:60] if r.top_match_trigger else "N/A"
        print(f'  Match: "{match_t}..." [{r.top_score:.3f}]')

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on holdout set")
    parser.add_argument("--limit", type=int, help="Max pairs to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Semantic match threshold")
    parser.add_argument("--k", type=int, default=5, help="Top-k results to retrieve")
    parser.add_argument("--batch-size", type=int, default=500, help="Encoding batch size")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    eval_output = run_evaluation(
        limit=args.limit,
        threshold=args.threshold,
        k=args.k,
        batch_size=args.batch_size,
    )

    print_report(eval_output)

    if args.output:
        # Save metrics (not full results - too large)
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(eval_output["metrics"], f, indent=2)
        logger.info(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
