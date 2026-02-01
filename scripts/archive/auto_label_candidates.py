#!/usr/bin/env python3
"""Auto-label trigger candidates using heuristics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

AUTO_LABEL_PATTERNS = {
    "greeting": [
        (r"^(hey|hi|hello|yo|sup)$", 0.95),
        (r"^how are you", 0.95),
        (r"^good (morning|afternoon|evening)", 0.95),
    ],
    "ack": [
        (r"^(ok|okay|k|kk|sure|bet|got it|sounds good|cool)$", 0.95),
        (r"^(thanks|thank you|thx|ty)$", 0.95),
        (r"^(lol|lmao|haha|hehe)", 0.90),
    ],
    "invitation": [
        (r"(wanna|want to|down to).*(hang|chill|go|grab|play|watch)", 0.90),
        (r"are you free", 0.90),
        (r"^(let\'s|lets) (go|hang|chill|grab)", 0.85),
    ],
    "request": [
        (r"^(can|could|would|will) you", 0.85),
        (r"^(please|pls|plz)", 0.85),
        (r"(pick me up|drop me off|help me|send me)", 0.90),
    ],
    "yn_question": [
        (r"^(do|does|did|is|are|have|has|can|could|will|would) you", 0.85),
        (r"\?$", 0.60),
    ],
    "info_question": [
        (r"^(what|when|where|who|which|how) ", 0.85),
    ],
    "good_news": [
        (r"(got the job|passed|promoted|engaged|married|accepted|won)", 0.85),
        (r"(so happy|so excited|great news|finally|nailed it)", 0.70),
    ],
    "bad_news": [
        (r"(lost|failed|got fired|sick|hurt|injured|hospital)", 0.85),
        (r"(so sad|terrible|awful|unfortunately|broke down)", 0.70),
    ],
    "reaction": [
        (r"(did you see|have you seen|did you hear)", 0.85),
        (r"(can you believe|how crazy|omg|dude.*\?)", 0.80),
    ],
    "statement": [
        (r"^(i[\'m|m| am]) (on my way|here|done|finished)", 0.70),
    ],
}

def auto_label_text(text: str, hint: str = None) -> tuple:
    text_clean = text.strip().lower()
    best_label = None
    best_conf = 0.0
    
    for label, patterns in AUTO_LABEL_PATTERNS.items():
        for pattern, conf in patterns:
            if re.search(pattern, text_clean, re.I):
                if conf > best_conf:
                    best_label = label
                    best_conf = conf
    
    if best_label and best_conf >= 0.70:
        return best_label, best_conf, "auto"
    elif hint:
        return hint, 0.40, "hint"
    else:
        return None, 0.0, "uncertain"

def process_file(input_path: Path, output_path: Path) -> dict:
    results = {"total": 0, "labeled": 0, "uncertain": 0, "by_label": {}}
    
    with open(input_path) as f:
        items = [json.loads(line) for line in f if line.strip()]
    
    labeled_items = []
    for item in items:
        results["total"] += 1
        text = item.get("trigger_text", "")
        hint = item.get("rare_class_hint")
        
        label, conf, method = auto_label_text(text, hint)
        
        item["label"] = label
        item["auto_confidence"] = conf
        item["auto_method"] = method
        item["needs_review"] = conf < 0.85 if label else True
        
        if label:
            results["labeled"] += 1
            results["by_label"][label] = results["by_label"].get(label, 0) + 1
        else:
            results["uncertain"] += 1
        
        labeled_items.append(item)
    
    with open(output_path, 'w') as f:
        for item in labeled_items:
            f.write(json.dumps(item) + '\n')
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", choices=["good_news", "bad_news", "greeting"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    classes = ["good_news", "bad_news", "greeting"] if args.all else [getattr(args, "class")]
    
    for cls in classes:
        input_path = Path(f"results/trigger_candidates_{cls}.jsonl")
        output_path = Path(f"results/trigger_candidates_{cls}_labeled.jsonl")
        
        if not input_path.exists():
            print(f"Skipping {cls}: {input_path} not found")
            continue
        
        print(f"\nProcessing {cls}...")
        results = process_file(input_path, output_path)
        
        print(f"  Total: {results['total']}")
        print(f"  Auto-labeled: {results['labeled']}")
        print(f"  Needs review: {results['uncertain']}")
        print(f"  Output: {output_path}")

if __name__ == "__main__":
    main()
