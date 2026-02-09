#!/usr/bin/env python3
"""Label all validation sets (3, 4, 5) with Groq LLM."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not set", flush=True)
    exit(1)

client = Groq(api_key=api_key)

BATCH_SIZE = 10  # Write results every N items

CATEGORY_SYSTEM_PROMPT = """You are a category classifier for text messages. Classify each message into exactly ONE of these 6 categories:

**Categories:**
1. **closing**: Conversation enders (bye, talk later, goodnight, see you, take care)
2. **acknowledge**: Simple acknowledgments with NO new info (ok, got it, thanks, sounds good, cool, yeah, sure, makes sense, np, understood, will do, right, yep, agreed, noted, perfect)
3. **question**: Requests for information (uses ?, asks for details, seeks clarification)
4. **request**: Action requests or commands (imperative verbs, "can you", "please", direct asks for someone to DO something)
5. **emotion**: Emotional reactions or greetings (hi, hey, congrats, haha, lol, love it, sorry, wow, omg, excited)
6. **statement**: Informational statements, updates, or observations (declarative sentences providing info, reporting status)

**Rules:**
- If message has both question mark AND action request → **request** (e.g., "Can you send that?")
- Formulaic greetings like "Hey" or "Hi there" → **emotion**
- Thanks/acknowledgment with no info → **acknowledge**
- Thanks + new info → **statement**
- "Yes/no" answers with context → **statement**
- "Yes/no" alone → **acknowledge**

**Output format:** Return ONLY the category name (lowercase, no extra text)."""


def call_with_retry(func, max_retries=3):
    """Retry API call with exponential backoff."""
    delays = [1, 2, 4]
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = delays[attempt]
            print(
                f"\nRetry {attempt + 1}/{max_retries} after error: {e} (waiting {delay}s)",
                flush=True,
            )
            time.sleep(delay)


def label_set(input_file, output_file):
    """Label a validation set."""
    start_time = time.time()

    with open(input_file) as f:
        data = [json.loads(line) for line in f]

    print(f"\nLabeling {input_file} ({len(data)} messages)...", flush=True)

    # Load existing results if resuming
    existing_count = 0
    if Path(output_file).exists():
        with open(output_file) as f:
            existing_count = sum(1 for line in f if line.strip())
        if existing_count > 0:
            print(f"Resuming from {existing_count} existing results", flush=True)
            data = data[existing_count:]
        else:
            # Truncate empty/stale file from prior interrupted run
            Path(output_file).unlink()

    results = []
    error_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, example in enumerate(tqdm(data, desc=f"Labeling {input_file}")):
        try:
            context_str = (
                "\n".join(f"- {msg}" for msg in example["context"][-3:])
                if example["context"]
                else "[No prior messages]"
            )

            user_prompt = f"""Context (previous messages):
{context_str}

Current message to classify:
"{example["text"]}"

Category:"""

            def api_call():
                return client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": CATEGORY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=10,
                )

            response = call_with_retry(api_call)

            # Track token usage
            if hasattr(response, "usage"):
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens

            label = response.choices[0].message.content.strip().lower()

            valid_categories = {
                "closing",
                "acknowledge",
                "question",
                "request",
                "emotion",
                "statement",
            }
            if label not in valid_categories:
                label = "statement"

            results.append(
                {"text": example["text"], "context": example["context"], "llm_prediction": label}
            )

            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"\nERROR after retries: {e}", flush=True)
            error_count += 1
            results.append(
                {
                    "text": example["text"],
                    "context": example["context"],
                    "llm_prediction": "statement",
                }
            )
            time.sleep(1)

        # Incremental write every BATCH_SIZE items
        if (idx + 1) % BATCH_SIZE == 0:
            mode = "a" if Path(output_file).exists() else "w"
            with open(output_file, mode) as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
            print(f"Wrote batch {idx + 1}/{len(data)}", flush=True)
            results = []

    # Write remaining results
    if results:
        mode = "a" if Path(output_file).exists() else "w"
        with open(output_file, mode) as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

    # Final stats
    elapsed = time.time() - start_time
    print(f"\n✅ Saved to {output_file}", flush=True)
    print(f"\n{'=' * 60}", flush=True)
    print("STATS:", flush=True)
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f}m)", flush=True)
    print(f"  Messages: {len(data) + existing_count}", flush=True)
    print(f"  Errors: {error_count}", flush=True)
    print(f"  Prompt tokens: {total_prompt_tokens:,}", flush=True)
    print(f"  Completion tokens: {total_completion_tokens:,}", flush=True)
    print(f"  Total tokens: {total_prompt_tokens + total_completion_tokens:,}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Show distribution
    with open(output_file) as f:
        all_results = [json.loads(line) for line in f]

    from collections import Counter

    labels = [r["llm_prediction"] for r in all_results]
    dist = Counter(labels)
    print("Distribution:", flush=True)
    for label, count in sorted(dist.items()):
        print(f"  {label:12}: {count:3} ({count / len(all_results) * 100:.1f}%)", flush=True)


# Label Sets 3, 4, 5
for i in range(3, 6):
    input_file = f"validation_set_{i}.jsonl"
    output_file = f"validation_set_{i}_labeled.jsonl"

    if not Path(input_file).exists():
        print(f"Skipping {input_file} (not found)", flush=True)
        continue

    label_set(input_file, output_file)

print("\n✅ All sets labeled!", flush=True)
