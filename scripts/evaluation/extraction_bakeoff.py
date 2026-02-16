import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.imessage import ChatDBReader
from jarvis.nlp.entailment import verify_entailment_batch
from models.loader import MLXModelLoader, ModelConfig


@dataclass
class BakeoffResult:
    strategy: str
    raw_output: str
    parsed_facts: list[str]
    verified_facts: list[str]
    latency: float


def parse_raw(raw: str):
    facts = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Handle both "- Name: Fact" and "Name: Fact"
        content = line.lstrip("- ").strip()
        if ":" in content:
            facts.append(content)
    return facts


STRATEGIES = {
    "Zero-Shot-Strict": {
        "system": "You extract personal facts from iMessage chats. RULES: 1. ONLY "
        "extract facts EXPLICITLY stated. 2. USE FORMAT: - [Name]: [fact]. 3. NO OTHER TEXT.",
        "user": "Chat:\n{text}\n\nFacts:",
    },
    "Few-Shot-NoHallucinate": {
        "system": "You extract facts in format '- [Name]: [fact]'. "
        "EXAMPLES (FORMAT ONLY): - Alice: lives in NY - Bob: has a cat. "
        "DO NOT COPY EXAMPLES. EXTRACT FROM CHAT ONLY.",
        "user": "Chat:\n{text}\n\nFacts:",
    },
    "Turn-Based-JSON": {
        "system": "You are a chat analyzer. Extract facts as a list of strings "
        "in format 'Name: Fact'.",
        "user": "Analyze turns and list facts learned about each person:\n\n{text}\n\nFacts:",
    },
}


def run_bakeoff(chat_text: str, loader: Any, user_name: str, contact_name: str):
    results = []

    for name, prompts in STRATEGIES.items():
        print(f"\nRunning Strategy: {name}...")

        sys_prompt = prompts["system"].format(user_name=user_name, contact_name=contact_name)
        usr_prompt = prompts["user"].format(text=chat_text)

        # Use ChatML
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]

        formatted = loader._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # NUDGE: Force start with bullet
        formatted += "- "

        t0 = time.time()
        res = loader.generate_sync(
            prompt=formatted, max_tokens=200, temperature=0.0, pre_formatted=True
        )
        latency = time.time() - t0

        raw = "- " + res.text.strip()

        # New robust parse
        parsed = parse_raw(raw)

        # NLI Verify
        verified = []
        if parsed:
            pairs = [(chat_text, p) for p in parsed]
            nli_res = verify_entailment_batch(pairs, threshold=0.15)
            verified = [p for p, (ok, score) in zip(parsed, nli_res) if ok]

        results.append(BakeoffResult(name, raw, parsed, verified, latency))

    return results


def main():
    reader = ChatDBReader()
    config = ModelConfig(model_path="models/lfm-0.7b-4bit", default_temperature=0.1)
    loader = MLXModelLoader(config)
    loader.load()

    # Get a real chat with content
    convs = reader.get_conversations(limit=100)
    targets = [c for c in convs if c.message_count > 10 and c.display_name]
    if not targets:
        print("No suitable chats found!")
        return
    target = targets[0]

    messages = reader.get_messages(target.chat_id, limit=20)
    messages.reverse()

    # Group messages
    turns = []
    curr_sender = "Me" if messages[0].is_from_me else target.display_name
    curr_msgs = []
    for m in messages:
        sender = "Me" if m.is_from_me else target.display_name
        if sender == curr_sender:
            curr_msgs.append(m.text or "")
        else:
            turns.append(f"{curr_sender}: {' '.join(curr_msgs)}")
            curr_sender = sender
            curr_msgs = [m.text or ""]
    turns.append(f"{curr_sender}: {' '.join(curr_msgs)}")
    chat_text = "\n".join(turns)

    print("=" * 80)
    print(f"BAKEOFF: {target.display_name}")
    print("=" * 80)
    print(f"Chat Context:\n{chat_text[:500]}...\n")

    results = run_bakeoff(chat_text, loader, "Jwalin", target.display_name)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    for r in results:
        print(f"\nSTRATEGY: {r.strategy} ({r.latency:.1f}s)")
        print(f"  Raw: {r.raw_output.replace(chr(10), ' | ')[:100]}...")
        print(f"  Parsed: {len(r.parsed_facts)}")
        print(f"  Verified: {len(r.verified_facts)}")
        for f in r.verified_facts:
            print(f"    ✓ {f}")


if __name__ == "__main__":
    main()
