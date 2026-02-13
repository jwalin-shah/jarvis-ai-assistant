#!/usr/bin/env python3
"""Quick test: free-form fact extraction with LFM2.5-Instruct."""
import json
import sys

sys.path.insert(0, ".")
from models.loader import MLXModelLoader, ModelConfig

loader = MLXModelLoader(ModelConfig())
loader.load()

gold = json.load(open("training_data/gliner_goldset/candidate_gold_merged_r4.json"))
positives = [r for r in gold if r.get("expected_candidates")][:8]

PROMPT_V1 = """What personal facts can you learn about the person who sent this text message?

Message: "My sister works at Google and loves hiking"
Facts:
- has a sister (family)
- sister works at Google (employer)
- sister loves hiking (activity)

Message: "just got home lol"
Facts:
- none

Message: "I've been dealing with my anxiety lately but yoga helps"
Facts:
- has anxiety (health)
- does yoga (activity)

Message: "{msg}"
Facts:"""

# Even simpler - just ask what we learn
PROMPT_V2 = """From this text message, what do we learn about the sender? List each fact on its own line. If nothing, say "nothing".

Message: "{msg}"

We learn:"""

# Super direct
PROMPT_V3 = """Text: "{msg}"
List every personal detail about the sender mentioned above:"""

PROMPTS = {"v1_fewshot": PROMPT_V1, "v2_simple": PROMPT_V2, "v3_direct": PROMPT_V3}

results = []
for pname, ptmpl in PROMPTS.items():
    print(f"\n{'='*60}")
    print(f"PROMPT: {pname}")
    print(f"{'='*60}")
    for rec in positives:
        prompt = ptmpl.format(msg=rec["message_text"])
        result = loader.generate_sync(
            prompt=prompt,
            max_tokens=150,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
            timeout_seconds=30.0,
        )
        entry = {
            "prompt": pname,
            "msg": rec["message_text"][:100],
            "expected": [(c["span_text"], c["span_label"]) for c in rec["expected_candidates"]],
            "llm_output": result.text[:400],
        }
        results.append(entry)
        print(f"MSG: {entry['msg']}")
        print(f"  EXPECTED: {entry['expected']}")
        print(f"  LLM: {result.text[:200]}")
        print()

loader.unload()

with open("results/freeform_test.json", "w") as f:
    json.dump(results, f, indent=2)
