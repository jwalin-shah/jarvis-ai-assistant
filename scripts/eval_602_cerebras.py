import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.judge_config import JUDGE_MODEL, get_judge_client

from integrations.imessage.reader import ChatDBReader
from jarvis.reply_service import ReplyService

# Disable excessive logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("eval_602")
logger.setLevel(logging.INFO)

BATCH_SIZE = 5

JUDGE_PROMPT_BATCH = """
You are an expert conversational evaluator grading "Jwalin Shah", a busy, casual, and direct individual.

**Jwalin's Persona:**
- Extremely concise (under 10 words).
- Natural texting style (mostly lowercase, minimal punctuation).
- NO AI fluff, NO social filler ("hope you're well"), NO generic questions back.
- "Need more context" is the ONLY correct reply if the message is ambiguous or missing history.

**Grading Rubric (1-5):**
1 - Fails: Hallucination, AI-assistant behavior, or extremely verbose.
2 - Poor: Sounds like a robot, uses formal capitalization, or misses the point.
3 - Acceptable: Correct info but slightly stiff or generic.
4 - Good: Natural, concise, and correctly grounded.
5 - Excellent: Perfect "Jwalin" vibe—ultra-terse and indistinguishable from a human text.

**Tasks to Evaluate:**
{tasks_formatted}

**Output Format:**
Output ONLY a valid JSON object:
{{
  "evaluations": [
    {{ "score": <int>, "reasoning": "<string>" }},
    ...
  ]
}}
"""


async def evaluate_batch(client, batch_items):
    tasks_formatted = ""
    for i, item in enumerate(batch_items, 1):
        tasks_formatted += (
            f"--- TASK {i} ---\nContext:\n{item['context']}\nReply:\n{item['reply']}\n\n"
        )

    prompt = JUDGE_PROMPT_BATCH.format(tasks_formatted=tasks_formatted)

    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict persona judge. JSON output only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(completion.choices[0].message.content)
        return data.get("evaluations", [])
    except Exception as e:
        return [{"score": 0, "reasoning": f"Error: {e}"}] * len(batch_items)


async def main():
    reader = ChatDBReader()
    service = ReplyService()
    judge_client = get_judge_client()

    print("=" * 80)
    print("🚀 STARTING FINAL 602-CHAT EVALUATION")
    print(f"Persona: Busy/Blunt/Lowercase | Judge: {JUDGE_MODEL}")
    print("=" * 80)

    conversations = reader.get_conversations(limit=1000)
    bot_keywords = [
        "stop",
        "opt out",
        "unsubscribe",
        "verify",
        "code",
        "research",
        ".com",
        "http",
        "63071",
    ]

    valid_chats = []
    for conv in conversations:
        last_text = (conv.last_message_text or "").lower()
        is_bot = any(kw in last_text for kw in bot_keywords) or (
            conv.chat_id.isdigit() and len(conv.chat_id) < 7
        )
        if not is_bot and last_text:
            valid_chats.append(conv)
            if len(valid_chats) >= 602:
                break

    print(f"Found {len(valid_chats)} target conversations.\n")

    items_to_judge = []
    print("STEP 1: Local Generation...")
    start_gen = time.perf_counter()

    for i, conv in enumerate(valid_chats, 1):
        try:
            result = await asyncio.to_thread(
                service.route_legacy, incoming=conv.last_message_text, chat_id=conv.chat_id
            )
            reply = result.get("response", "")
            if not reply:
                continue

            context_msgs, _ = service.context_service.fetch_conversation_context(
                conv.chat_id, limit=5
            )

            items_to_judge.append(
                {
                    "chat_id": conv.chat_id,
                    "contact": conv.display_name or "Unknown",
                    "last_message": conv.last_message_text,
                    "context": "\n".join(context_msgs),
                    "reply": reply,
                }
            )
            if i % 100 == 0:
                avg_time = (time.perf_counter() - start_gen) / i
                print(f"  [{i}/{len(valid_chats)}] Generated. Speed: {avg_time:.2f}s/msg")
        except Exception:
            continue

    print(f"\nSTEP 2: Judging {len(items_to_judge)} replies in batches of {BATCH_SIZE}...")

    final_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_eval_602_{timestamp}.csv"

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["chat_id", "contact", "last_message", "reply", "score", "reasoning"],
        )
        writer.writeheader()

        for i in range(0, len(items_to_judge), BATCH_SIZE):
            batch = items_to_judge[i : i + BATCH_SIZE]
            print(
                f" Batch {i // BATCH_SIZE + 1}/{(len(items_to_judge) - 1) // BATCH_SIZE + 1}...",
                end="",
                flush=True,
            )

            batch_results = await evaluate_batch(judge_client, batch)

            for item, res in zip(batch, batch_results):
                row = {
                    "chat_id": item["chat_id"],
                    "contact": item["contact"],
                    "last_message": item["last_message"].replace("\n", " "),
                    "reply": item["reply"],
                    "score": res.get("score", 0),
                    "reasoning": res.get("reasoning", ""),
                }
                final_results.append(row)
                writer.writerow(row)

            print(" Done.")
            await asyncio.sleep(1.5)

    scores = [r["score"] for r in final_results if r["score"] > 0]
    avg = sum(scores) / len(scores) if scores else 0
    print("\n" + "=" * 40)
    print(f"FINAL SCORE: {avg:.2f}/5.0")
    print(f"Results: {filename}")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
