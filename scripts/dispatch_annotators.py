"""Dispatch labeling prompts to 3 annotators (gemini, claude, kimi).

Splits 200 messages into batches of 20, builds prompts from the template,
and runs each annotator CLI in parallel. Collects outputs into per-annotator
JSON files in training_data/goldset_v6/.

Usage:
    uv run python scripts/dispatch_annotators.py
    uv run python scripts/dispatch_annotators.py --batch-size 15 --timeout 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "training_data" / "goldset_v6" / "messages_to_label.json"
PROMPT_TEMPLATE = PROJECT_ROOT / "tasks" / "prompts" / "goldset-labeler.md"
OUTPUT_DIR = PROJECT_ROOT / "training_data" / "goldset_v6"

# Agent CLI commands
AGENT_CMDS: dict[str, list[str]] = {
    "gemini": ["gemini"],
    "claude": ["claude", "-p", "{prompt}", "--print"],
    "kimi": ["kimi", "--quiet", "-p", "{prompt}"],
}


def load_messages(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_template(path: Path) -> str:
    with open(path) as f:
        return f.read()


def format_messages_for_prompt(messages: list[dict]) -> str:
    """Format a batch of messages for the labeling prompt."""
    parts = []
    for msg in messages:
        sender = "me" if msg.get("is_from_me") else "them"
        parts.append(f"### {msg['sample_id']} (from: {sender})")
        if msg.get("context_prev"):
            parts.append(f"**Context prev:** {msg['context_prev']}")
        parts.append(f"**Message:** {msg['message_text']}")
        if msg.get("context_next"):
            parts.append(f"**Context next:** {msg['context_next']}")
        parts.append("")
    return "\n".join(parts)


def build_batch_prompt(template: str, batch: list[dict], batch_num: int, total_batches: int) -> str:
    """Build a prompt for a batch of messages."""
    messages_text = format_messages_for_prompt(batch)
    sample_ids = [m["sample_id"] for m in batch]
    prompt = template.replace("{{MESSAGES_BATCH}}", messages_text)
    prompt += f"\n\n---\nBatch {batch_num}/{total_batches}. "
    prompt += f"Label these {len(batch)} messages: {', '.join(sample_ids)}\n"
    prompt += "Output ONLY a JSON array of objects with sample_id and expected_candidates.\n"
    return prompt


def run_agent_on_batch(
    agent: str,
    prompt: str,
    batch_num: int,
    timeout: int,
) -> list[dict] | None:
    """Run a single agent on a single batch prompt. Returns parsed results or None."""
    try:
        if agent == "gemini":
            # Gemini reads from stdin
            result = subprocess.run(
                ["gemini"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
            )
            output = result.stdout
        elif agent == "claude":
            result = subprocess.run(
                ["claude", "-p", prompt, "--print"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
            )
            output = result.stdout
        elif agent == "kimi":
            result = subprocess.run(
                ["kimi", "--quiet", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
            )
            output = result.stdout
        else:
            print(f"  Unknown agent: {agent}", flush=True)
            return None

        if result.returncode != 0:
            print(f"  {agent} batch {batch_num} failed (exit {result.returncode})", flush=True)
            if result.stderr:
                print(f"    stderr: {result.stderr[:200]}", flush=True)
            return None

        return parse_agent_output(output)

    except subprocess.TimeoutExpired:
        print(f"  {agent} batch {batch_num} timed out ({timeout}s)", flush=True)
        return None
    except Exception as e:
        print(f"  {agent} batch {batch_num} error: {e}", flush=True)
        return None


def parse_agent_output(output: str) -> list[dict] | None:
    """Parse agent output into a list of sample annotations."""
    import re

    output = output.strip()

    # Try direct JSON parse
    try:
        result = json.loads(output)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", output, re.DOTALL)
    if code_match:
        try:
            result = json.loads(code_match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding array
    bracket_match = re.search(r"\[.*\]", output, re.DOTALL)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    print(f"    Could not parse output ({len(output)} chars): {output[:100]}...", flush=True)
    return None


def run_agent_all_batches(
    agent: str,
    batches: list[list[dict]],
    template: str,
    timeout: int,
) -> list[dict]:
    """Run one agent across all batches sequentially. Returns merged annotations."""
    all_results: list[dict] = []
    total = len(batches)

    for i, batch in enumerate(batches):
        batch_num = i + 1
        prompt = build_batch_prompt(template, batch, batch_num, total)

        print(f"  {agent}: batch {batch_num}/{total} ({len(batch)} msgs)...", flush=True)
        start = time.time()
        results = run_agent_on_batch(agent, prompt, batch_num, timeout)
        elapsed = time.time() - start

        if results:
            all_results.extend(results)
            print(f"    Got {len(results)} annotations in {elapsed:.1f}s", flush=True)
        else:
            # Create empty annotations for failed batches
            for msg in batch:
                all_results.append({
                    "sample_id": msg["sample_id"],
                    "expected_candidates": [],
                    "_error": "agent_failed",
                })
            print(f"    Failed, using empty annotations for {len(batch)} msgs", flush=True)

    return all_results


def merge_with_metadata(
    annotations: list[dict],
    messages: list[dict],
) -> list[dict]:
    """Merge agent annotations back with original message metadata."""
    msg_by_id = {m["sample_id"]: m for m in messages}
    merged = []
    for ann in annotations:
        sid = ann.get("sample_id", "")
        base = msg_by_id.get(sid, {})
        merged.append({
            "sample_id": sid,
            "message_id": base.get("message_id"),
            "message_text": base.get("message_text", ""),
            "is_from_me": base.get("is_from_me", False),
            "context_prev": base.get("context_prev", ""),
            "context_next": base.get("context_next", ""),
            "slice": base.get("slice", "unknown"),
            "expected_candidates": ann.get("expected_candidates", []),
        })
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch labeling to 3 annotators")
    parser.add_argument("--batch-size", type=int, default=20, help="Messages per batch")
    parser.add_argument("--timeout", type=int, default=300, help="Per-batch timeout (seconds)")
    parser.add_argument("--agents", default="gemini,claude,kimi", help="Comma-separated agents")
    args = parser.parse_args()

    agents = [a.strip() for a in args.agents.split(",")]

    print("Loading data...", flush=True)
    messages = load_messages(DATA_FILE)
    template = load_template(PROMPT_TEMPLATE)
    print(f"  {len(messages)} messages, {len(template)} char template", flush=True)

    # Split into batches
    batches = []
    for i in range(0, len(messages), args.batch_size):
        batches.append(messages[i:i + args.batch_size])
    print(f"  {len(batches)} batches of ~{args.batch_size} messages each", flush=True)

    # Run each agent sequentially (each agent does all batches)
    for agent in agents:
        print(f"\n{'='*60}", flush=True)
        print(f"Running {agent} annotator...", flush=True)
        print(f"{'='*60}", flush=True)

        start = time.time()
        annotations = run_agent_all_batches(agent, batches, template, args.timeout)
        elapsed = time.time() - start

        # Merge with metadata
        merged = merge_with_metadata(annotations, messages)

        # Write output
        output_path = OUTPUT_DIR / f"annotator_{agent}.json"
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        n_with_cands = sum(1 for m in merged if m.get("expected_candidates"))
        print(f"\n  {agent} done in {elapsed:.1f}s", flush=True)
        print(f"  {len(merged)} samples, {n_with_cands} with candidates", flush=True)
        print(f"  Saved to {output_path}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("All annotators complete!", flush=True)
    print(f"Output files:", flush=True)
    for agent in agents:
        print(f"  training_data/goldset_v6/annotator_{agent}.json", flush=True)
    print(f"\nNext: run compute_goldset_iaa.py to merge", flush=True)


if __name__ == "__main__":
    main()
