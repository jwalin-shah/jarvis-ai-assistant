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
  # noqa: E402
import psutil  # noqa: E402

# noqa: E402
from models.loader import MLXModelLoader, ModelConfig  # noqa: E402


  # noqa: E402
  # noqa: E402
def mem_mb() -> str:  # noqa: E402
    rss = psutil.Process().memory_info().rss / 1024 / 1024  # noqa: E402
    return f"{rss:.0f}MB"  # noqa: E402
  # noqa: E402
  # noqa: E402
# ── Test cases ────────────────────────────────────────────────────────────────  # noqa: E402
# Broader coverage: direct questions, logistics, emotional, ambiguous, group, media  # noqa: E402
TEST_CASES = [  # noqa: E402
    # Direct questions  # noqa: E402
    {"ctx": "[15:00] Dad: Did you take out the trash?", "label": "yes_no_q"},  # noqa: E402
    {"ctx": "[14:00] John: Want to grab lunch tomorrow?", "label": "invite"},  # noqa: E402
    {"ctx": "[18:30] Sam: Any plans this weekend?", "label": "open_q"},  # noqa: E402
    {"ctx": "[09:00] Manager: Can you send the Q4 report by EOD?", "label": "work_request"},  # noqa: E402
    # Logistics / coordination  # noqa: E402
    {"ctx": "[19:00] Jake: you close?", "label": "eta"},  # noqa: E402
    {"ctx": "[17:30] Lisa: I'm at the restaurant, where should I park?", "label": "logistics"},  # noqa: E402
    {"ctx": "[20:15] Mike: what's the wifi password", "label": "info_request"},  # noqa: E402
    # Emotional / support  # noqa: E402
    {  # noqa: E402
        "ctx": "[20:00] Mike: Work was brutal today\n"  # noqa: E402
        "[20:01] Mike: Boss dumped a project on me last minute",  # noqa: E402
        "label": "venting",  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "ctx": "[15:00] John: Didn't get the job. Thought the interview went well",  # noqa: E402
        "label": "bad_news",  # noqa: E402
    },  # noqa: E402
    {"ctx": "[10:00] Lisa: Just landed in Tokyo!!", "label": "excitement"},  # noqa: E402
    {"ctx": "[22:00] Sarah: I miss you", "label": "sentimental"},  # noqa: E402
    # Ambiguous / low-context  # noqa: E402
    {"ctx": "[11:00] Chris: ?", "label": "question_mark"},  # noqa: E402
    {"ctx": "[11:00] Unknown: hey", "label": "cold_hey"},  # noqa: E402
    {"ctx": "[14:00] Tom: lmao remember the thing", "label": "vague_ref"},  # noqa: E402
    {"ctx": "[12:00] Sarah: [Link]", "label": "link_only"},  # noqa: E402
    {"ctx": "[16:00] Alex: 👀", "label": "emoji_only"},  # noqa: E402
    # Social / casual  # noqa: E402
    {"ctx": "[21:00] Ben: bro that game was insane", "label": "react_sports"},  # noqa: E402
    {"ctx": "[13:00] Emma: just saw the funniest tiktok", "label": "react_media"},  # noqa: E402
    {"ctx": "[19:30] Dad: Love you kid, have a good night", "label": "goodnight"},  # noqa: E402
    {"ctx": "[08:00] Mom: Good morning sweetheart! Have a great day!", "label": "good_morning"},  # noqa: E402
    # Multi-turn  # noqa: E402
    {  # noqa: E402
        "ctx": "[18:00] Jake: yo\n[18:00] Jake: you free tonight?\n"  # noqa: E402
        "[18:01] Jake: thinking about hitting up that new bar",  # noqa: E402
        "label": "multi_invite",  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "ctx": "[14:00] Sarah: hey did you hear about the party?\n"  # noqa: E402
        "[14:01] You: no what party\n[14:01] Sarah: at Jake's this Saturday",  # noqa: E402
        "label": "multi_followup",  # noqa: E402
    },  # noqa: E402
    # Requests / favors  # noqa: E402
    {"ctx": "[11:00] Roommate: can you pick up milk on the way home", "label": "favor"},  # noqa: E402
    {"ctx": "[15:00] Mom: Can you call me when you get a chance?", "label": "call_request"},  # noqa: E402
]  # noqa: E402
  # noqa: E402
# ── Prompt strategies ─────────────────────────────────────────────────────────  # noqa: E402
PROMPTS: dict[str, callable] = {}  # noqa: E402
  # noqa: E402
  # noqa: E402
def prompt(name: str):  # noqa: E402
    """Decorator to register a prompt strategy."""  # noqa: E402
  # noqa: E402
    def decorator(fn):  # noqa: E402
        PROMPTS[name] = fn  # noqa: E402
        return fn  # noqa: E402
  # noqa: E402
    return decorator  # noqa: E402
  # noqa: E402
  # noqa: E402
# --- Bare completions ---  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("bare_reply")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"{ctx}\nReply:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("bare_me")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"{ctx}\nMe:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("bare_arrow")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"{ctx}\n>"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("bare_you")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"{ctx}\nYou:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("bare_dash")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"{ctx}\n-"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("bare_sent")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"{ctx}\nSent:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("bare_newline")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"{ctx}\n"  # noqa: E402
  # noqa: E402
  # noqa: E402
# --- Few-shot ---  # noqa: E402
  # noqa: E402
_FS = """Alex: wanna get food?  # noqa: E402
{tag} ya im down, where?  # noqa: E402
  # noqa: E402
Mom: Call me when you get home  # noqa: E402
{tag} k  # noqa: E402
  # noqa: E402
Jordan: you coming tonight?  # noqa: E402
{tag} yea omw  # noqa: E402
  # noqa: E402
"""  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fs_reply")  # noqa: E402
def _(ctx):  # noqa: E402
    return _FS.format(tag="Reply:") + f"{ctx}\nReply:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fs_me")  # noqa: E402
def _(ctx):  # noqa: E402
    return _FS.format(tag="Me:") + f"{ctx}\nMe:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fs_arrow")  # noqa: E402
def _(ctx):  # noqa: E402
    return _FS.format(tag=">") + f"{ctx}\n>"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fs_you")  # noqa: E402
def _(ctx):  # noqa: E402
    return _FS.format(tag="You:") + f"{ctx}\nYou:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fs_dash")  # noqa: E402
def _(ctx):  # noqa: E402
    return _FS.format(tag="-") + f"{ctx}\n-"  # noqa: E402
  # noqa: E402
  # noqa: E402
# --- Few-shot varied ---  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fs_varied")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"""Friend: you up?  # noqa: E402
Reply: yeah what's good  # noqa: E402
  # noqa: E402
Boss: Meeting moved to 3pm  # noqa: E402
Reply: got it thanks  # noqa: E402
  # noqa: E402
Ex: hey  # noqa: E402
Reply: ?  # noqa: E402
  # noqa: E402
Mom: don't forget to call grandma  # noqa: E402
Reply: will do  # noqa: E402
  # noqa: E402
{ctx}  # noqa: E402
Reply:"""  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fs_emoji")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"""Friend: happy birthday!!  # noqa: E402
Reply: tyy 🥳  # noqa: E402
  # noqa: E402
Roommate: left food in the fridge for u  # noqa: E402
Reply: ur the best 🙏  # noqa: E402
  # noqa: E402
Coworker: drinks tonight?  # noqa: E402
Reply: down 🍻  # noqa: E402
  # noqa: E402
{ctx}  # noqa: E402
Reply:"""  # noqa: E402
  # noqa: E402
  # noqa: E402
# --- Instruction-based ---  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("inst_short")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"Reply to this text in under 10 words:\n\n{ctx}\n\nReply:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("inst_no_ai")  # noqa: E402
def _(ctx):  # noqa: E402
    return (  # noqa: E402
        "You're a real person texting. NOT an AI. "  # noqa: E402
        f"Reply naturally, one line max.\n\n{ctx}\n\nReply:"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("inst_complete")  # noqa: E402
def _(ctx):  # noqa: E402
    return (  # noqa: E402
        "Complete the next message in this text conversation. "  # noqa: E402
        f"Output ONLY the reply.\n\n{ctx}\n\nReply:"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("inst_mimic")  # noqa: E402
def _(ctx):  # noqa: E402
    return (  # noqa: E402
        "Read this text thread and write what you'd text back. "  # noqa: E402
        f"Keep it real. Max 1 sentence.\n\n{ctx}\n\nYour reply:"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("inst_terse")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"Respond. 1-5 words only.\n\n{ctx}\n\n>"  # noqa: E402
  # noqa: E402
  # noqa: E402
# --- Few-shot + instruction combo ---  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fsi_brief")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"""Reply to texts briefly. Match the vibe.  # noqa: E402
  # noqa: E402
Alex: wanna get food?  # noqa: E402
Reply: ya im down, where?  # noqa: E402
  # noqa: E402
Mom: Call me when you get home  # noqa: E402
Reply: k  # noqa: E402
  # noqa: E402
{ctx}  # noqa: E402
Reply:"""  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fsi_no_ai")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"""You're texting a friend. Keep it real and short.  # noqa: E402
  # noqa: E402
them: you free tonight?  # noqa: E402
you: yeah what time  # noqa: E402
  # noqa: E402
them: nvm plans changed  # noqa: E402
you: ah ok no worries  # noqa: E402
  # noqa: E402
{ctx}  # noqa: E402
you:"""  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("fsi_persona")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"""You text like a normal 20-something. Short, casual, no AI vibes.  # noqa: E402
  # noqa: E402
them: wanna hang?  # noqa: E402
me: ye when  # noqa: E402
  # noqa: E402
them: running late  # noqa: E402
me: all good take ur time  # noqa: E402
  # noqa: E402
{ctx}  # noqa: E402
me:"""  # noqa: E402
  # noqa: E402
  # noqa: E402
# --- Script / structured ---  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("script")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"INT. TEXT CONVERSATION\n\n{ctx}\nME: "  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("script_fs")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"""INT. TEXT CONVERSATION  # noqa: E402
  # noqa: E402
Alex: wanna get food?  # noqa: E402
ME: ya im down, where?  # noqa: E402
  # noqa: E402
Mom: Call me when you get home  # noqa: E402
ME: k  # noqa: E402
  # noqa: E402
{ctx}  # noqa: E402
ME: """  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("chat_template")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"User: {ctx}\nReply as a real person texting. One sentence max.\nAssistant:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("qa")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"Q: What would you text back to this?\n{ctx}\nA:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("qa_short")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"Q: Reply in 1-5 words.\n{ctx}\nA:"  # noqa: E402
  # noqa: E402
  # noqa: E402
# --- Bracket / tag ---  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("bracket")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"""[them] wanna get food?  # noqa: E402
[me] ya im down, where?  # noqa: E402
  # noqa: E402
[them] Call me when you get home  # noqa: E402
[me] k  # noqa: E402
  # noqa: E402
{ctx}  # noqa: E402
[me]"""  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("xml_reply")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"<conversation>\n{ctx}\n</conversation>\n<reply>"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("xml_short")  # noqa: E402
def _(ctx):  # noqa: E402
    return f'<conversation>\n{ctx}\n</conversation>\n<reply max_words="10">'  # noqa: E402
  # noqa: E402
  # noqa: E402
# --- Dialogue continuation ---  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("dialogue")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"The following is a text conversation. Write ONLY the next reply.\n\n{ctx}\n\n"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("roleplay")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"[You are casually texting. Reply to the last message.]\n\n{ctx}\n\nYou:"  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("texting_sim")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"--- iMessage ---\n{ctx}\n📱 Your reply: "  # noqa: E402
  # noqa: E402
  # noqa: E402
@prompt("social_media")  # noqa: E402
def _(ctx):  # noqa: E402
    return f"💬 DM thread:\n{ctx}\n\nYou replied:"  # noqa: E402
  # noqa: E402
  # noqa: E402
# ── Sampling configs ──────────────────────────────────────────────────────────  # noqa: E402
# Each is a dict of kwargs passed to generate_sync  # noqa: E402
DEFAULT_TEMPS = [0.1, 0.3]  # noqa: E402
DEFAULT_TOP_PS = [0.1, 0.9]  # noqa: E402
DEFAULT_MIN_PS = [0.0, 0.15]  # noqa: E402
  # noqa: E402
# ── Models (LFM + Qwen3 only) ────────────────────────────────────────────────  # noqa: E402
DEFAULT_MODELS = [  # noqa: E402
    "lfm-0.3b",  # noqa: E402
    "lfm-0.7b",  # noqa: E402
    "lfm-0.7b-4bit",  # noqa: E402
    "lfm-1.2b",  # noqa: E402
    "lfm-1.2b-base",  # noqa: E402
    "lfm-2.6b",  # noqa: E402
]  # noqa: E402
  # noqa: E402
DEFAULT_PROMPTS = list(PROMPTS.keys())  # noqa: E402
  # noqa: E402
  # noqa: E402
def run(  # noqa: E402
    models: list[str],  # noqa: E402
    prompt_names: list[str],  # noqa: E402
    test_indices: list[int],  # noqa: E402
    temps: list[float],  # noqa: E402
    top_ps: list[float],  # noqa: E402
    min_ps: list[float],  # noqa: E402
    max_tokens: int,  # noqa: E402
) -> None:  # noqa: E402
    tests = [TEST_CASES[i] for i in test_indices]  # noqa: E402
    sampling_configs = list(product(temps, top_ps, min_ps))  # noqa: E402
  # noqa: E402
    n_calls = len(models) * len(prompt_names) * len(tests) * len(sampling_configs)  # noqa: E402
    print(  # noqa: E402
        f"\n{len(models)} models x {len(prompt_names)} prompts x "  # noqa: E402
        f"{len(tests)} tests x {len(sampling_configs)} sampling configs "  # noqa: E402
        f"= {n_calls} generations",  # noqa: E402
        flush=True,  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    # results[model][f"t{temp}_p{top_p}"][prompt_name][test_label] = output_text  # noqa: E402
    all_results: dict = {}  # noqa: E402
  # noqa: E402
    for model_id in models:  # noqa: E402
        print(f"\n{'=' * 70}", flush=True)  # noqa: E402
        print(f"[{mem_mb()}] Loading {model_id}...", flush=True)  # noqa: E402
  # noqa: E402
        try:  # noqa: E402
            t_load = time.monotonic()  # noqa: E402
            config = ModelConfig(model_id=model_id)  # noqa: E402
            loader = MLXModelLoader(config=config)  # noqa: E402
            loader.load()  # noqa: E402
            load_s = time.monotonic() - t_load  # noqa: E402
            print(f"[{mem_mb()}] Loaded in {load_s:.1f}s", flush=True)  # noqa: E402
        except Exception as e:  # noqa: E402
            print(f"  FAILED: {e}", flush=True)  # noqa: E402
            all_results[model_id] = {}  # noqa: E402
            continue  # noqa: E402
  # noqa: E402
        model_results: dict = {}  # noqa: E402
  # noqa: E402
        for temp, top_p, min_p in sampling_configs:  # noqa: E402
            sampling_key = f"t{temp}_p{top_p}_mp{min_p}"  # noqa: E402
            print(f"\n  --- temp={temp} top_p={top_p} min_p={min_p} ---", flush=True)  # noqa: E402
            sampling_results: dict[str, dict[str, str]] = {}  # noqa: E402
  # noqa: E402
            for pname in prompt_names:  # noqa: E402
                pfn = PROMPTS[pname]  # noqa: E402
                prompt_results = {}  # noqa: E402
  # noqa: E402
                for tc in tests:  # noqa: E402
                    prompt_text = pfn(tc["ctx"])  # noqa: E402
                    t0 = time.monotonic()  # noqa: E402
                    try:  # noqa: E402
                        result = loader.generate_sync(  # noqa: E402
                            prompt=prompt_text,  # noqa: E402
                            temperature=temp,  # noqa: E402
                            max_tokens=max_tokens,  # noqa: E402
                            top_p=top_p,  # noqa: E402
                            min_p=min_p,  # noqa: E402
                            top_k=50,  # noqa: E402
                            repetition_penalty=1.05,  # noqa: E402
                        )  # noqa: E402
                        elapsed = time.monotonic() - t0  # noqa: E402
                        text = result.text.strip().replace("\n", " ↵ ")  # noqa: E402
                        tps = result.tokens_generated / elapsed if elapsed > 0 else 0  # noqa: E402
                        display = text[:55] if len(text) > 55 else text  # noqa: E402
                        print(  # noqa: E402
                            f"  {pname:>18} | {tc['label']:>14} | {display:<55} | {tps:.0f}t/s",  # noqa: E402
                            flush=True,  # noqa: E402
                        )  # noqa: E402
                        prompt_results[tc["label"]] = {  # noqa: E402
                            "text": text,  # noqa: E402
                            "tps": round(tps, 1),  # noqa: E402
                            "tokens": result.tokens_generated,  # noqa: E402
                            "elapsed_ms": round(elapsed * 1000, 1),  # noqa: E402
                        }  # noqa: E402
                    except Exception as e:  # noqa: E402
                        print(f"  {pname:>18} | {tc['label']:>14} | ERROR: {e}", flush=True)  # noqa: E402
                        prompt_results[tc["label"]] = {  # noqa: E402
                            "text": f"[ERROR: {e}]",  # noqa: E402
                            "tps": 0,  # noqa: E402
                            "tokens": 0,  # noqa: E402
                            "elapsed_ms": 0,  # noqa: E402
                        }  # noqa: E402
  # noqa: E402
                sampling_results[pname] = prompt_results  # noqa: E402
  # noqa: E402
            model_results[sampling_key] = sampling_results  # noqa: E402
  # noqa: E402
        all_results[model_id] = model_results  # noqa: E402
        loader.unload()  # noqa: E402
        print(f"[{mem_mb()}] Unloaded {model_id}", flush=True)  # noqa: E402
  # noqa: E402
    # ── Save results ────────────────────────────────────────────────────  # noqa: E402
    out_path = PROJECT_ROOT / "evals" / "results" / "comparison-raw.json"  # noqa: E402
    out_path.parent.mkdir(exist_ok=True)  # noqa: E402
    with open(out_path, "w") as f:  # noqa: E402
        json.dump(  # noqa: E402
            {  # noqa: E402
                "models": models,  # noqa: E402
                "prompts": prompt_names,  # noqa: E402
                "tests": [tc["label"] for tc in tests],  # noqa: E402
                "sampling_configs": [  # noqa: E402
                    {"temp": t, "top_p": p, "min_p": mp} for t, p, mp in sampling_configs  # noqa: E402
                ],  # noqa: E402
                "results": all_results,  # noqa: E402
            },  # noqa: E402
            f,  # noqa: E402
            indent=2,  # noqa: E402
        )  # noqa: E402
    print(f"\nResults saved to {out_path}", flush=True)  # noqa: E402
    print(f"Total generations: {n_calls}", flush=True)  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    parser = argparse.ArgumentParser(  # noqa: E402
        description="Model x prompt x sampling comparison for human eval"  # noqa: E402
    )  # noqa: E402
    parser.add_argument("--models", default=None, help="Comma-separated model IDs")  # noqa: E402
    parser.add_argument("--prompts", default=None, help="Comma-separated prompt names")  # noqa: E402
    parser.add_argument("--tests", default=None, help="Comma-separated test indices (0-23)")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--temps", default=None, help="Comma-separated temperatures (default: 0.1,0.5,0.7,1.0)"  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--top-ps", default=None, help="Comma-separated top_p values (default: 0.1,0.9)"  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--min-ps", default=None, help="Comma-separated min_p values (default: 0.0,0.15)"  # noqa: E402
    )  # noqa: E402
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens (default: 20)")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--list-prompts", action="store_true", help="List all prompt names and exit"  # noqa: E402
    )  # noqa: E402
    parser.add_argument("--list-tests", action="store_true", help="List all test cases and exit")  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    if args.list_prompts:  # noqa: E402
        for name in PROMPTS:  # noqa: E402
            sample = PROMPTS[name](TEST_CASES[0]["ctx"])  # noqa: E402
            last_line = sample.strip().split("\n")[-1]  # noqa: E402
            print(f"  {name:>18}: ...{last_line}")  # noqa: E402
        sys.exit(0)  # noqa: E402
  # noqa: E402
    if args.list_tests:  # noqa: E402
        for i, tc in enumerate(TEST_CASES):  # noqa: E402
            ctx_preview = tc["ctx"].replace("\n", " | ")[:60]  # noqa: E402
            print(f"  {i:>2}: {tc['label']:>14} | {ctx_preview}")  # noqa: E402
        sys.exit(0)  # noqa: E402
  # noqa: E402
    models = args.models.split(",") if args.models else DEFAULT_MODELS  # noqa: E402
    prompt_names = args.prompts.split(",") if args.prompts else DEFAULT_PROMPTS  # noqa: E402
    test_indices = (  # noqa: E402
        [int(i) for i in args.tests.split(",")] if args.tests else list(range(len(TEST_CASES)))  # noqa: E402
    )  # noqa: E402
    temps = [float(t) for t in args.temps.split(",")] if args.temps else DEFAULT_TEMPS  # noqa: E402
    top_ps = [float(p) for p in args.top_ps.split(",")] if args.top_ps else DEFAULT_TOP_PS  # noqa: E402
    min_ps = [float(p) for p in args.min_ps.split(",")] if args.min_ps else DEFAULT_MIN_PS  # noqa: E402
  # noqa: E402
    print(f"Models: {', '.join(models)}", flush=True)  # noqa: E402
    print(f"Prompts: {len(prompt_names)} strategies", flush=True)  # noqa: E402
    print(f"Tests: {len(test_indices)} cases", flush=True)  # noqa: E402
    print(f"Temps: {temps}", flush=True)  # noqa: E402
    print(f"Top-p: {top_ps}", flush=True)  # noqa: E402
    print(f"Min-p: {min_ps}", flush=True)  # noqa: E402
  # noqa: E402
    run(models, prompt_names, test_indices, temps, top_ps, min_ps, args.max_tokens)  # noqa: E402
