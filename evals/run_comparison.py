#!/usr/bin/env python3
"""Brute-force model x prompt x sampling comparison. Prints raw outputs for human eval.

Usage:
    uv run python evals/run_comparison.py
    uv run python evals/run_comparison.py --models lfm-1.2b,qwen3-0.6b
    uv run python evals/run_comparison.py --prompts fs_me,inst_short
    uv run python evals/run_comparison.py --tests 0,1,2
    uv run python evals/run_comparison.py --temps 0.1,0.7 --top-ps 0.5,0.95
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psutil
from models.loader import ModelConfig, MLXModelLoader


def mem_mb() -> str:
    rss = psutil.Process().memory_info().rss / 1024 / 1024
    return f"{rss:.0f}MB"


# ── Test cases ────────────────────────────────────────────────────────────────
# Broader coverage: direct questions, logistics, emotional, ambiguous, group, media
TEST_CASES = [
    # Direct questions
    {"ctx": "[15:00] Dad: Did you take out the trash?", "label": "yes_no_q"},
    {"ctx": "[14:00] John: Want to grab lunch tomorrow?", "label": "invite"},
    {"ctx": "[18:30] Sam: Any plans this weekend?", "label": "open_q"},
    {"ctx": "[09:00] Manager: Can you send the Q4 report by EOD?", "label": "work_request"},
    # Logistics / coordination
    {"ctx": "[19:00] Jake: you close?", "label": "eta"},
    {"ctx": "[17:30] Lisa: I'm at the restaurant, where should I park?", "label": "logistics"},
    {"ctx": "[20:15] Mike: what's the wifi password", "label": "info_request"},
    # Emotional / support
    {"ctx": "[20:00] Mike: Work was brutal today\n[20:01] Mike: Boss dumped a project on me last minute", "label": "venting"},
    {"ctx": "[15:00] John: Didn't get the job. Thought the interview went well", "label": "bad_news"},
    {"ctx": "[10:00] Lisa: Just landed in Tokyo!!", "label": "excitement"},
    {"ctx": "[22:00] Sarah: I miss you", "label": "sentimental"},
    # Ambiguous / low-context
    {"ctx": "[11:00] Chris: ?", "label": "question_mark"},
    {"ctx": "[11:00] Unknown: hey", "label": "cold_hey"},
    {"ctx": "[14:00] Tom: lmao remember the thing", "label": "vague_ref"},
    {"ctx": "[12:00] Sarah: [Link]", "label": "link_only"},
    {"ctx": "[16:00] Alex: 👀", "label": "emoji_only"},
    # Social / casual
    {"ctx": "[21:00] Ben: bro that game was insane", "label": "react_sports"},
    {"ctx": "[13:00] Emma: just saw the funniest tiktok", "label": "react_media"},
    {"ctx": "[19:30] Dad: Love you kid, have a good night", "label": "goodnight"},
    {"ctx": "[08:00] Mom: Good morning sweetheart! Have a great day!", "label": "good_morning"},
    # Multi-turn
    {"ctx": "[18:00] Jake: yo\n[18:00] Jake: you free tonight?\n[18:01] Jake: thinking about hitting up that new bar", "label": "multi_invite"},
    {"ctx": "[14:00] Sarah: hey did you hear about the party?\n[14:01] You: no what party\n[14:01] Sarah: at Jake's this Saturday", "label": "multi_followup"},
    # Requests / favors
    {"ctx": "[11:00] Roommate: can you pick up milk on the way home", "label": "favor"},
    {"ctx": "[15:00] Mom: Can you call me when you get a chance?", "label": "call_request"},
]

# ── Prompt strategies ─────────────────────────────────────────────────────────
PROMPTS: dict[str, callable] = {}


def prompt(name: str):
    """Decorator to register a prompt strategy."""
    def decorator(fn):
        PROMPTS[name] = fn
        return fn
    return decorator


# --- Bare completions ---

@prompt("bare_reply")
def _(ctx): return f"{ctx}\nReply:"

@prompt("bare_me")
def _(ctx): return f"{ctx}\nMe:"

@prompt("bare_arrow")
def _(ctx): return f"{ctx}\n>"

@prompt("bare_you")
def _(ctx): return f"{ctx}\nYou:"

@prompt("bare_dash")
def _(ctx): return f"{ctx}\n-"

@prompt("bare_sent")
def _(ctx): return f"{ctx}\nSent:"

@prompt("bare_newline")
def _(ctx): return f"{ctx}\n"

# --- Few-shot ---

_FS = """Alex: wanna get food?
{tag} ya im down, where?

Mom: Call me when you get home
{tag} k

Jordan: you coming tonight?
{tag} yea omw

"""

@prompt("fs_reply")
def _(ctx): return _FS.format(tag="Reply:") + f"{ctx}\nReply:"

@prompt("fs_me")
def _(ctx): return _FS.format(tag="Me:") + f"{ctx}\nMe:"

@prompt("fs_arrow")
def _(ctx): return _FS.format(tag=">") + f"{ctx}\n>"

@prompt("fs_you")
def _(ctx): return _FS.format(tag="You:") + f"{ctx}\nYou:"

@prompt("fs_dash")
def _(ctx): return _FS.format(tag="-") + f"{ctx}\n-"

# --- Few-shot varied ---

@prompt("fs_varied")
def _(ctx): return f"""Friend: you up?
Reply: yeah what's good

Boss: Meeting moved to 3pm
Reply: got it thanks

Ex: hey
Reply: ?

Mom: don't forget to call grandma
Reply: will do

{ctx}
Reply:"""

@prompt("fs_emoji")
def _(ctx): return f"""Friend: happy birthday!!
Reply: tyy 🥳

Roommate: left food in the fridge for u
Reply: ur the best 🙏

Coworker: drinks tonight?
Reply: down 🍻

{ctx}
Reply:"""

# --- Instruction-based ---

@prompt("inst_short")
def _(ctx): return f"Reply to this text in under 10 words:\n\n{ctx}\n\nReply:"

@prompt("inst_no_ai")
def _(ctx): return f"You're a real person texting. NOT an AI. Reply naturally, one line max.\n\n{ctx}\n\nReply:"

@prompt("inst_complete")
def _(ctx): return f"Complete the next message in this text conversation. Output ONLY the reply.\n\n{ctx}\n\nReply:"

@prompt("inst_mimic")
def _(ctx): return f"Read this text thread and write what you'd text back. Keep it real. Max 1 sentence.\n\n{ctx}\n\nYour reply:"

@prompt("inst_terse")
def _(ctx): return f"Respond. 1-5 words only.\n\n{ctx}\n\n>"

# --- Few-shot + instruction combo ---

@prompt("fsi_brief")
def _(ctx): return f"""Reply to texts briefly. Match the vibe.

Alex: wanna get food?
Reply: ya im down, where?

Mom: Call me when you get home
Reply: k

{ctx}
Reply:"""

@prompt("fsi_no_ai")
def _(ctx): return f"""You're texting a friend. Keep it real and short.

them: you free tonight?
you: yeah what time

them: nvm plans changed
you: ah ok no worries

{ctx}
you:"""

@prompt("fsi_persona")
def _(ctx): return f"""You text like a normal 20-something. Short, casual, no AI vibes.

them: wanna hang?
me: ye when

them: running late
me: all good take ur time

{ctx}
me:"""

# --- Script / structured ---

@prompt("script")
def _(ctx): return f"INT. TEXT CONVERSATION\n\n{ctx}\nME: "

@prompt("script_fs")
def _(ctx): return f"""INT. TEXT CONVERSATION

Alex: wanna get food?
ME: ya im down, where?

Mom: Call me when you get home
ME: k

{ctx}
ME: """

@prompt("chat_template")
def _(ctx): return f"User: {ctx}\nReply as a real person texting. One sentence max.\nAssistant:"

@prompt("qa")
def _(ctx): return f"Q: What would you text back to this?\n{ctx}\nA:"

@prompt("qa_short")
def _(ctx): return f"Q: Reply in 1-5 words.\n{ctx}\nA:"

# --- Bracket / tag ---

@prompt("bracket")
def _(ctx): return f"""[them] wanna get food?
[me] ya im down, where?

[them] Call me when you get home
[me] k

{ctx}
[me]"""

@prompt("xml_reply")
def _(ctx): return f"<conversation>\n{ctx}\n</conversation>\n<reply>"

@prompt("xml_short")
def _(ctx): return f"<conversation>\n{ctx}\n</conversation>\n<reply max_words=\"10\">"

# --- Dialogue continuation ---

@prompt("dialogue")
def _(ctx): return f"The following is a text conversation. Write ONLY the next reply.\n\n{ctx}\n\n"

@prompt("roleplay")
def _(ctx): return f"[You are casually texting. Reply to the last message.]\n\n{ctx}\n\nYou:"

@prompt("texting_sim")
def _(ctx): return f"--- iMessage ---\n{ctx}\n📱 Your reply: "

@prompt("social_media")
def _(ctx): return f"💬 DM thread:\n{ctx}\n\nYou replied:"

# ── Sampling configs ──────────────────────────────────────────────────────────
# Each is a dict of kwargs passed to generate_sync
DEFAULT_TEMPS = [0.1, 0.3]
DEFAULT_TOP_PS = [0.1, 0.9]
DEFAULT_MIN_PS = [0.0, 0.15]

# ── Models (LFM + Qwen3 only) ────────────────────────────────────────────────
DEFAULT_MODELS = [
    "lfm-0.3b",
    "lfm-0.7b",
    "lfm-0.7b-4bit",
    "lfm-1.2b",
    "lfm-1.2b-base",
    "lfm-2.6b",
]

DEFAULT_PROMPTS = list(PROMPTS.keys())


def run(models: list[str], prompt_names: list[str], test_indices: list[int],
        temps: list[float], top_ps: list[float], min_ps: list[float],
        max_tokens: int) -> None:
    tests = [TEST_CASES[i] for i in test_indices]
    sampling_configs = list(product(temps, top_ps, min_ps))

    n_calls = len(models) * len(prompt_names) * len(tests) * len(sampling_configs)
    print(f"\n{len(models)} models x {len(prompt_names)} prompts x {len(tests)} tests x {len(sampling_configs)} sampling configs = {n_calls} generations", flush=True)

    # results[model][f"t{temp}_p{top_p}"][prompt_name][test_label] = output_text
    all_results: dict = {}

    for model_id in models:
        print(f"\n{'='*70}", flush=True)
        print(f"[{mem_mb()}] Loading {model_id}...", flush=True)

        try:
            t_load = time.monotonic()
            config = ModelConfig(model_id=model_id)
            loader = MLXModelLoader(config=config)
            loader.load()
            load_s = time.monotonic() - t_load
            print(f"[{mem_mb()}] Loaded in {load_s:.1f}s", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            all_results[model_id] = {}
            continue

        model_results: dict = {}

        for temp, top_p, min_p in sampling_configs:
            sampling_key = f"t{temp}_p{top_p}_mp{min_p}"
            print(f"\n  --- temp={temp} top_p={top_p} min_p={min_p} ---", flush=True)
            sampling_results: dict[str, dict[str, str]] = {}

            for pname in prompt_names:
                pfn = PROMPTS[pname]
                prompt_results = {}

                for tc in tests:
                    prompt_text = pfn(tc["ctx"])
                    t0 = time.monotonic()
                    try:
                        result = loader.generate_sync(
                            prompt=prompt_text,
                            temperature=temp,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            min_p=min_p,
                            top_k=50,
                            repetition_penalty=1.05,
                        )
                        elapsed = time.monotonic() - t0
                        text = result.text.strip().replace("\n", " ↵ ")
                        tps = result.tokens_generated / elapsed if elapsed > 0 else 0
                        display = text[:55] if len(text) > 55 else text
                        print(f"  {pname:>18} | {tc['label']:>14} | {display:<55} | {tps:.0f}t/s", flush=True)
                        prompt_results[tc["label"]] = {
                            "text": text,
                            "tps": round(tps, 1),
                            "tokens": result.tokens_generated,
                            "elapsed_ms": round(elapsed * 1000, 1),
                        }
                    except Exception as e:
                        print(f"  {pname:>18} | {tc['label']:>14} | ERROR: {e}", flush=True)
                        prompt_results[tc["label"]] = {"text": f"[ERROR: {e}]", "tps": 0, "tokens": 0, "elapsed_ms": 0}

                sampling_results[pname] = prompt_results

            model_results[sampling_key] = sampling_results

        all_results[model_id] = model_results
        loader.unload()
        print(f"[{mem_mb()}] Unloaded {model_id}", flush=True)

    # ── Save results ────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "evals" / "results" / "comparison-raw.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "models": models,
            "prompts": prompt_names,
            "tests": [tc["label"] for tc in tests],
            "sampling_configs": [{"temp": t, "top_p": p, "min_p": mp} for t, p, mp in sampling_configs],
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Total generations: {n_calls}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model x prompt x sampling comparison for human eval")
    parser.add_argument("--models", default=None, help="Comma-separated model IDs")
    parser.add_argument("--prompts", default=None, help="Comma-separated prompt names")
    parser.add_argument("--tests", default=None, help="Comma-separated test indices (0-23)")
    parser.add_argument("--temps", default=None, help="Comma-separated temperatures (default: 0.1,0.5,0.7,1.0)")
    parser.add_argument("--top-ps", default=None, help="Comma-separated top_p values (default: 0.1,0.9)")
    parser.add_argument("--min-ps", default=None, help="Comma-separated min_p values (default: 0.0,0.15)")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens (default: 20)")
    parser.add_argument("--list-prompts", action="store_true", help="List all prompt names and exit")
    parser.add_argument("--list-tests", action="store_true", help="List all test cases and exit")
    args = parser.parse_args()

    if args.list_prompts:
        for name in PROMPTS:
            sample = PROMPTS[name](TEST_CASES[0]["ctx"])
            last_line = sample.strip().split("\n")[-1]
            print(f"  {name:>18}: ...{last_line}")
        sys.exit(0)

    if args.list_tests:
        for i, tc in enumerate(TEST_CASES):
            ctx_preview = tc["ctx"].replace("\n", " | ")[:60]
            print(f"  {i:>2}: {tc['label']:>14} | {ctx_preview}")
        sys.exit(0)

    models = args.models.split(",") if args.models else DEFAULT_MODELS
    prompt_names = args.prompts.split(",") if args.prompts else DEFAULT_PROMPTS
    test_indices = [int(i) for i in args.tests.split(",")] if args.tests else list(range(len(TEST_CASES)))
    temps = [float(t) for t in args.temps.split(",")] if args.temps else DEFAULT_TEMPS
    top_ps = [float(p) for p in args.top_ps.split(",")] if args.top_ps else DEFAULT_TOP_PS
    min_ps = [float(p) for p in args.min_ps.split(",")] if args.min_ps else DEFAULT_MIN_PS

    print(f"Models: {', '.join(models)}", flush=True)
    print(f"Prompts: {len(prompt_names)} strategies", flush=True)
    print(f"Tests: {len(test_indices)} cases", flush=True)
    print(f"Temps: {temps}", flush=True)
    print(f"Top-p: {top_ps}", flush=True)
    print(f"Min-p: {min_ps}", flush=True)

    run(models, prompt_names, test_indices, temps, top_ps, min_ps, args.max_tokens)
