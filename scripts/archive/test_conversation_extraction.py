#!/usr/bin/env python3
"""Test local model fact extraction on the Radhika conversation.

Loads conversation chunks, runs them through local MLX models, and compares
extracted facts against annotator ground truth (Gemini + Kimi + Claude consensus).

Usage:
    uv run python scripts/test_conversation_extraction.py [--models lfm-1.2b qwen-3b]
    uv run python scripts/test_conversation_extraction.py --dry-run  # consensus only, no model
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "training_data" / "conversation_facts"
RESULTS_DIR = ROOT / "results" / "conversation_extraction"

# --- Reaction patterns to filter out ---
REACTION_PATTERNS = re.compile(
    r"^(Loved|Liked|Disliked|Laughed at|Emphasized|Questioned)\s+\u201c", re.IGNORECASE
)
# Attachment-only messages (object replacement character)
ATTACHMENT_ONLY = re.compile(r"^\ufffc$")
# Very short filler (single emoji, "lol", etc.)
FILLER_PATTERN = re.compile(r"^(lol|lmao|haha|ok|yep|ya|yeah|nah|omg|bruh|fr|rip)\s*$", re.I)


# ---- Data structures ----

@dataclass
class Message:
    id: int
    speaker: str
    text: str
    date: str


@dataclass
class Fact:
    about: str
    category: str
    fact: str
    confidence: str
    source_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "about": self.about,
            "category": self.category,
            "fact": self.fact,
            "confidence": self.confidence,
        }


@dataclass
class ExtractionResult:
    model_id: str
    chunks_processed: int
    chunks_parsed: int
    total_facts: int
    facts: list[dict]
    time_per_chunk_ms: list[float]
    tokens_per_second: list[float]
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    matched_ground_truth: list[str] = field(default_factory=list)
    missed_ground_truth: list[str] = field(default_factory=list)


# ---- Step 1: Load and chunk conversation ----

def load_conversation(path: Path) -> list[Message]:
    """Load conversation from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [Message(**m) for m in data]


def is_substantive(msg: Message) -> bool:
    """Filter out reactions, attachments, and filler."""
    text = msg.text.strip()
    if not text:
        return False
    if REACTION_PATTERNS.match(text):
        return False
    if ATTACHMENT_ONLY.match(text):
        return False
    if FILLER_PATTERN.match(text):
        return False
    if len(text) < 3:
        return False
    return True


def chunk_conversation(
    messages: list[Message], chunk_size: int = 20
) -> list[list[Message]]:
    """Split substantive messages into chunks of ~chunk_size."""
    substantive = [m for m in messages if is_substantive(m)]
    chunks = []
    for i in range(0, len(substantive), chunk_size):
        chunks.append(substantive[i : i + chunk_size])
    return chunks


def format_chunk(chunk: list[Message]) -> str:
    """Format a chunk of messages for the prompt."""
    lines = []
    for m in chunk:
        lines.append(f"[{m.date}] {m.speaker}: {m.text}")
    return "\n".join(lines)


# ---- Step 2: Ground truth consensus ----

def load_annotator_facts(path: Path) -> list[Fact]:
    """Load facts from an annotator JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [
        Fact(
            about=d["about"],
            category=d["category"],
            fact=d["fact"],
            confidence=d["confidence"],
            source_messages=d.get("source_messages", []),
        )
        for d in data
    ]


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_about(about: str) -> str:
    """Normalize the 'about' field for matching."""
    about = about.lower().strip()
    # Map variants to canonical names
    aliases = {
        "jwalin's masi": "masi",
        "masi": "masi",
        "runis": "runi",
        "rumi": "runi",
        "faltu": "faltu",
    }
    return aliases.get(about, about)


def word_overlap(a: str, b: str) -> float:
    """Compute Jaccard word overlap between two normalized strings."""
    words_a = set(normalize_text(a).split())
    words_b = set(normalize_text(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def facts_match(a: Fact, b: Fact, threshold: float = 0.35) -> bool:
    """Check if two facts refer to the same information."""
    # Must be about the same person
    if normalize_about(a.about) != normalize_about(b.about):
        return False
    # Word overlap in fact description
    return word_overlap(a.fact, b.fact) >= threshold


def build_consensus(
    annotators: dict[str, list[Fact]], min_agreement: int = 2
) -> list[Fact]:
    """Build consensus facts that appear in >= min_agreement annotators.

    For each fact in any annotator, check how many other annotators have
    a matching fact. Keep the version with the most detail (longest description).
    """
    all_facts: list[tuple[str, Fact]] = []
    for name, facts in annotators.items():
        for f in facts:
            all_facts.append((name, f))

    # Group matching facts across annotators
    clusters: list[list[tuple[str, Fact]]] = []
    used = set()

    for i, (name_i, fact_i) in enumerate(all_facts):
        if i in used:
            continue
        cluster = [(name_i, fact_i)]
        used.add(i)
        for j, (name_j, fact_j) in enumerate(all_facts):
            if j in used:
                continue
            if name_j == name_i:
                continue  # same annotator
            if facts_match(fact_i, fact_j):
                cluster.append((name_j, fact_j))
                used.add(j)
        clusters.append(cluster)

    # Keep clusters with >= min_agreement unique annotators
    consensus = []
    for cluster in clusters:
        unique_annotators = {name for name, _ in cluster}
        if len(unique_annotators) >= min_agreement:
            # Pick the most detailed version (longest fact text)
            best = max(cluster, key=lambda x: len(x[1].fact))
            consensus.append(best[1])

    return consensus


# ---- Step 3: Extraction prompt ----

SYSTEM_PROMPT = """You are extracting personal facts from a text conversation between Jwalin and Radhika.

A "fact" is anything you'd remember about a person after reading their messages - relationships, where they live, their job, hobbies, health conditions, etc.

Extract facts about ALL people mentioned (Jwalin, Radhika, and anyone else referenced).

Categories: relationship, location, work, education, health, preference, life_event, contact_info, physical, other

Rules:
- Skip iMessage reactions (Loved "...", Liked "...", etc.)
- Extract facts even from casual/joking messages if real info is there
- Use natural language for fact descriptions
- Return valid JSON only"""


def build_extraction_prompt(
    chunk_text: str, known_facts: list[dict] | None = None
) -> list[dict]:
    """Build chat messages for extraction."""
    known_str = "None yet." if not known_facts else json.dumps(known_facts, indent=2)

    user_content = f"""Known facts from earlier in the conversation:
{known_str}

New messages to analyze:
{chunk_text}

Extract NEW facts as a JSON array. Each fact needs: "about", "category", "fact", "confidence" (high/medium/low).
Return [] if no new facts. Return ONLY the JSON array, no other text."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---- Step 4: Run extraction through model ----

def extract_json_array(text: str) -> list[dict] | None:
    """Try to parse a JSON array from model output, handling common issues."""
    text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "facts" in result:
            return result["facts"]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    # Look for [...] pattern
    bracket_match = re.search(r"\[[\s\S]*\]", text)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try to find ```json ... ``` block
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_match:
        try:
            result = json.loads(code_match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def run_extraction(
    model_id: str,
    chunks: list[list[Message]],
    max_tokens: int = 1024,
) -> ExtractionResult:
    """Run extraction through a local MLX model."""
    from models.loader import MLXModelLoader, ModelConfig

    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_id}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load model - skip memory check for testing (macOS compresses pages)
    print(f"Loading model...", flush=True)
    # Check if model_id is a path (for unregistered models)
    from models.registry import get_model_spec

    if get_model_spec(model_id) is not None:
        config = ModelConfig(
            model_id=model_id,
            memory_buffer_multiplier=0.0,
            estimated_memory_mb=1,
        )
    else:
        # Treat as direct path
        config = ModelConfig(
            model_path=model_id,
            memory_buffer_multiplier=0.0,
            estimated_memory_mb=1,
        )
    loader = MLXModelLoader(config)
    load_start = time.time()
    loader.load()
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    all_facts: list[dict] = []
    accumulated_facts: list[dict] = []
    time_per_chunk: list[float] = []
    tps_per_chunk: list[float] = []
    chunks_parsed = 0

    try:
        for i, chunk in enumerate(chunks):
            chunk_text = format_chunk(chunk)
            messages = build_extraction_prompt(chunk_text, accumulated_facts or None)

            # Format with chat template
            formatted = loader._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            print(
                f"\nChunk {i+1}/{len(chunks)} ({len(chunk)} messages)...",
                flush=True,
            )

            # Use LFM-recommended params for LFM models, greedy for others
            is_lfm = "lfm" in model_id.lower()
            chunk_start = time.time()
            result = loader.generate_sync(
                formatted,
                max_tokens=max_tokens,
                temperature=0.3 if is_lfm else 0.0,
                top_p=0.1,
                min_p=0.15 if is_lfm else None,
                top_k=50 if is_lfm else None,
                repetition_penalty=1.05 if is_lfm else 1.0,
                pre_formatted=True,
                timeout_seconds=120,
            )
            chunk_time_ms = (time.time() - chunk_start) * 1000

            time_per_chunk.append(chunk_time_ms)
            tps_per_chunk.append(result.tokens_per_second)

            # Parse output
            parsed = extract_json_array(result.text)
            if parsed is not None:
                chunks_parsed += 1
                # Validate each fact has required fields
                valid_facts = []
                for f in parsed:
                    if isinstance(f, dict) and "about" in f and "fact" in f:
                        fact = {
                            "about": str(f.get("about", "")),
                            "category": str(f.get("category", "other")),
                            "fact": str(f.get("fact", "")),
                            "confidence": str(f.get("confidence", "medium")),
                        }
                        valid_facts.append(fact)
                all_facts.extend(valid_facts)
                accumulated_facts.extend(valid_facts)
                print(
                    f"  -> {len(valid_facts)} facts | "
                    f"{result.tokens_generated} tokens | "
                    f"{result.tokens_per_second:.1f} tok/s | "
                    f"{chunk_time_ms:.0f}ms",
                    flush=True,
                )
            else:
                print(
                    f"  -> PARSE FAILED | {result.tokens_generated} tokens | "
                    f"{chunk_time_ms:.0f}ms",
                    flush=True,
                )
                # Show first 200 chars of output for debugging
                preview = result.text[:200].replace("\n", "\\n")
                print(f"     Output: {preview}", flush=True)

    finally:
        # Always unload
        print(f"\nUnloading model...", flush=True)
        loader.unload()

    return ExtractionResult(
        model_id=model_id,
        chunks_processed=len(chunks),
        chunks_parsed=chunks_parsed,
        total_facts=len(all_facts),
        facts=all_facts,
        time_per_chunk_ms=time_per_chunk,
        tokens_per_second=tps_per_chunk,
    )


# ---- Step 5: Evaluate against ground truth ----

def evaluate(
    result: ExtractionResult, ground_truth: list[Fact], threshold: float = 0.35
) -> ExtractionResult:
    """Compute P/R/F1 against ground truth."""
    predicted_facts = [
        Fact(
            about=f["about"],
            category=f.get("category", "other"),
            fact=f["fact"],
            confidence=f.get("confidence", "medium"),
        )
        for f in result.facts
    ]

    # Deduplicate predicted facts (model often repeats across chunks)
    seen_preds: list[Fact] = []
    for pf in predicted_facts:
        is_dup = False
        for sp in seen_preds:
            if pf.about.lower() == sp.about.lower() and word_overlap(pf.fact, sp.fact) > 0.6:
                is_dup = True
                break
        if not is_dup:
            seen_preds.append(pf)
    predicted_facts = seen_preds

    # For each ground truth fact, check if any predicted fact matches
    matched_gt = []
    missed_gt = []
    gt_matched_flags = [False] * len(ground_truth)
    pred_matched_flags = [False] * len(predicted_facts)

    for gi, gt_fact in enumerate(ground_truth):
        for pi, pred_fact in enumerate(predicted_facts):
            if pred_matched_flags[pi]:
                continue
            if facts_match(gt_fact, pred_fact, threshold):
                gt_matched_flags[gi] = True
                pred_matched_flags[pi] = True
                matched_gt.append(gt_fact.fact)
                break

    for gi, gt_fact in enumerate(ground_truth):
        if not gt_matched_flags[gi]:
            missed_gt.append(f"[{gt_fact.about}] {gt_fact.fact}")

    tp = sum(gt_matched_flags)
    fp = sum(1 for m in pred_matched_flags if not m)
    fn = sum(1 for m in gt_matched_flags if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result.precision = precision
    result.recall = recall
    result.f1 = f1
    result.matched_ground_truth = matched_gt
    result.missed_ground_truth = missed_gt

    return result


# ---- Step 6: Report ----

def print_report(results: list[ExtractionResult], ground_truth: list[Fact]) -> None:
    """Print comparison report."""
    print(f"\n{'='*70}", flush=True)
    print(f"EXTRACTION RESULTS COMPARISON", flush=True)
    print(f"Ground truth: {len(ground_truth)} consensus facts (>=2/3 annotators)", flush=True)
    print(f"{'='*70}", flush=True)

    for r in results:
        avg_time = sum(r.time_per_chunk_ms) / len(r.time_per_chunk_ms) if r.time_per_chunk_ms else 0
        avg_tps = sum(r.tokens_per_second) / len(r.tokens_per_second) if r.tokens_per_second else 0

        # Count unique facts (deduplicated)
        unique_count = len(r.matched_ground_truth) + len(r.missed_ground_truth)
        unique_count = r.total_facts  # raw count before dedup shown here

        print(f"\n--- {r.model_id} ---", flush=True)
        print(f"  Chunks parsed:    {r.chunks_parsed}/{r.chunks_processed}", flush=True)
        print(f"  Total facts (raw):{r.total_facts}", flush=True)
        print(f"  Avg time/chunk:   {avg_time:.0f}ms", flush=True)
        print(f"  Avg tokens/sec:   {avg_tps:.1f}", flush=True)
        print(f"  Precision:        {r.precision:.3f}", flush=True)
        print(f"  Recall:           {r.recall:.3f}", flush=True)
        print(f"  F1:               {r.f1:.3f}", flush=True)

        if r.missed_ground_truth:
            print(f"  Missed ({len(r.missed_ground_truth)}):", flush=True)
            for m in r.missed_ground_truth[:10]:
                print(f"    - {m}", flush=True)
            if len(r.missed_ground_truth) > 10:
                print(f"    ... and {len(r.missed_ground_truth) - 10} more", flush=True)


def save_results(
    results: list[ExtractionResult],
    ground_truth: list[Fact],
    output_dir: Path,
) -> None:
    """Save results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save consensus ground truth
    gt_path = output_dir / "consensus_ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump([g.to_dict() for g in ground_truth], f, indent=2)
    print(f"\nSaved ground truth to {gt_path}", flush=True)

    # Save per-model results
    for r in results:
        model_safe = r.model_id.replace("/", "_")
        result_path = output_dir / f"result_{model_safe}.json"
        with open(result_path, "w") as f:
            json.dump(
                {
                    "model_id": r.model_id,
                    "chunks_processed": r.chunks_processed,
                    "chunks_parsed": r.chunks_parsed,
                    "total_facts": r.total_facts,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "avg_time_per_chunk_ms": (
                        sum(r.time_per_chunk_ms) / len(r.time_per_chunk_ms)
                        if r.time_per_chunk_ms
                        else 0
                    ),
                    "avg_tokens_per_second": (
                        sum(r.tokens_per_second) / len(r.tokens_per_second)
                        if r.tokens_per_second
                        else 0
                    ),
                    "facts": r.facts,
                    "matched_ground_truth": r.matched_ground_truth,
                    "missed_ground_truth": r.missed_ground_truth,
                },
                f,
                indent=2,
            )
        print(f"Saved {r.model_id} results to {result_path}", flush=True)

    # Save summary comparison
    summary_path = output_dir / "summary.json"
    summary = []
    for r in results:
        summary.append({
            "model_id": r.model_id,
            "total_facts": r.total_facts,
            "chunks_parsed": f"{r.chunks_parsed}/{r.chunks_processed}",
            "precision": round(r.precision, 3),
            "recall": round(r.recall, 3),
            "f1": round(r.f1, 3),
        })
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}", flush=True)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Test local model fact extraction")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lfm-1.2b"],
        help="Model IDs to test (default: lfm-1.2b)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Messages per chunk (default: 20)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens per generation (default: 1024)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build consensus, don't run models",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.35,
        help="Word overlap threshold for matching (default: 0.35)",
    )
    args = parser.parse_args()

    # Step 1: Load conversation
    print("Loading conversation...", flush=True)
    conv_path = DATA_DIR / "radhika_conversation.json"
    messages = load_conversation(conv_path)
    print(f"  {len(messages)} total messages", flush=True)

    # Step 2: Chunk
    chunks = chunk_conversation(messages, args.chunk_size)
    substantive_count = sum(len(c) for c in chunks)
    print(
        f"  {substantive_count} substantive messages -> {len(chunks)} chunks "
        f"(~{args.chunk_size} msgs each)",
        flush=True,
    )

    # Step 3: Build ground truth consensus
    print("\nBuilding ground truth consensus...", flush=True)
    annotators = {}
    for annotator_name in ["claude", "gemini", "kimi"]:
        path = DATA_DIR / f"annotator_{annotator_name}.json"
        if path.exists():
            facts = load_annotator_facts(path)
            annotators[annotator_name] = facts
            print(f"  {annotator_name}: {len(facts)} facts", flush=True)
        else:
            print(f"  {annotator_name}: NOT FOUND at {path}", flush=True)

    consensus = build_consensus(annotators, min_agreement=2)
    print(f"\n  Consensus: {len(consensus)} facts (>=2/3 annotators agree)", flush=True)

    # Print consensus facts
    print("\n  Consensus facts:", flush=True)
    for i, f in enumerate(consensus):
        print(f"    {i+1}. [{f.about}] ({f.category}) {f.fact}", flush=True)

    if args.dry_run:
        print("\n--dry-run: Skipping model extraction.", flush=True)
        save_results([], consensus, RESULTS_DIR)
        return

    # Step 4: Run models
    results = []
    for model_id in args.models:
        try:
            result = run_extraction(model_id, chunks, args.max_tokens)
            result = evaluate(result, consensus, args.match_threshold)
            results.append(result)
        except Exception as e:
            print(f"\nERROR with {model_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Step 5: Report
    if results:
        print_report(results, consensus)

    # Step 6: Save
    save_results(results, consensus, RESULTS_DIR)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
