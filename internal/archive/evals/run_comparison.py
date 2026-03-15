#!/usr/bin/env python3  # noqa: E501
"""Brute-force model x prompt x sampling comparison. Prints raw outputs for human eval.  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/run_comparison.py  # noqa: E501
    uv run python evals/run_comparison.py --models lfm-1.2b,qwen3-0.6b  # noqa: E501
    uv run python evals/run_comparison.py --prompts fs_me,inst_short  # noqa: E501
    uv run python evals/run_comparison.py --tests 0,1,2  # noqa: E501
    uv run python evals/run_comparison.py --temps 0.1,0.7 --top-ps 0.5,0.95  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from itertools import product  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
import psutil  # noqa: E402  # noqa: E501

# noqa: E501
from models.loader import MLXModelLoader, ModelConfig  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
def mem_mb() -> str:  # noqa: E501
    rss = psutil.Process().memory_info().rss / 1024 / 1024  # noqa: E501
    return f"{rss:.0f}MB"  # noqa: E501
  # noqa: E501
  # noqa: E501
# ── Test cases ────────────────────────────────────────────────────────────────  # noqa: E501
# Broader coverage: direct questions, logistics, emotional, ambiguous, group, media  # noqa: E501
TEST_CASES = [  # noqa: E501
    # Direct questions  # noqa: E501
    {"ctx": "[15:00] Dad: Did you take out the trash?", "label": "yes_no_q"},  # noqa: E501
    {"ctx": "[14:00] John: Want to grab lunch tomorrow?", "label": "invite"},  # noqa: E501
    {"ctx": "[18:30] Sam: Any plans this weekend?", "label": "open_q"},  # noqa: E501
    {"ctx": "[09:00] Manager: Can you send the Q4 report by EOD?", "label": "work_request"},  # noqa: E501
    # Logistics / coordination  # noqa: E501
    {"ctx": "[19:00] Jake: you close?", "label": "eta"},  # noqa: E501
    {"ctx": "[17:30] Lisa: I'm at the restaurant, where should I park?", "label": "logistics"},  # noqa: E501
    {"ctx": "[20:15] Mike: what's the wifi password", "label": "info_request"},  # noqa: E501
    # Emotional / support  # noqa: E501
    {  # noqa: E501
        "ctx": "[20:00] Mike: Work was brutal today\n"  # noqa: E501
        "[20:01] Mike: Boss dumped a project on me last minute",  # noqa: E501
        "label": "venting",  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "ctx": "[15:00] John: Didn't get the job. Thought the interview went well",  # noqa: E501
        "label": "bad_news",  # noqa: E501
    },  # noqa: E501
    {"ctx": "[10:00] Lisa: Just landed in Tokyo!!", "label": "excitement"},  # noqa: E501
    {"ctx": "[22:00] Sarah: I miss you", "label": "sentimental"},  # noqa: E501
    # Ambiguous / low-context  # noqa: E501
    {"ctx": "[11:00] Chris: ?", "label": "question_mark"},  # noqa: E501
    {"ctx": "[11:00] Unknown: hey", "label": "cold_hey"},  # noqa: E501
    {"ctx": "[14:00] Tom: lmao remember the thing", "label": "vague_ref"},  # noqa: E501
    {"ctx": "[12:00] Sarah: [Link]", "label": "link_only"},  # noqa: E501
    {"ctx": "[16:00] Alex: 👀", "label": "emoji_only"},  # noqa: E501
    # Social / casual  # noqa: E501
    {"ctx": "[21:00] Ben: bro that game was insane", "label": "react_sports"},  # noqa: E501
    {"ctx": "[13:00] Emma: just saw the funniest tiktok", "label": "react_media"},  # noqa: E501
    {"ctx": "[19:30] Dad: Love you kid, have a good night", "label": "goodnight"},  # noqa: E501
    {"ctx": "[08:00] Mom: Good morning sweetheart! Have a great day!", "label": "good_morning"},  # noqa: E501
    # Multi-turn  # noqa: E501
    {  # noqa: E501
        "ctx": "[18:00] Jake: yo\n[18:00] Jake: you free tonight?\n"  # noqa: E501
        "[18:01] Jake: thinking about hitting up that new bar",  # noqa: E501
        "label": "multi_invite",  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "ctx": "[14:00] Sarah: hey did you hear about the party?\n"  # noqa: E501
        "[14:01] You: no what party\n[14:01] Sarah: at Jake's this Saturday",  # noqa: E501
        "label": "multi_followup",  # noqa: E501
    },  # noqa: E501
    # Requests / favors  # noqa: E501
    {"ctx": "[11:00] Roommate: can you pick up milk on the way home", "label": "favor"},  # noqa: E501
    {"ctx": "[15:00] Mom: Can you call me when you get a chance?", "label": "call_request"},  # noqa: E501
]  # noqa: E501
  # noqa: E501
# ── Prompt strategies ─────────────────────────────────────────────────────────  # noqa: E501
PROMPTS: dict[str, callable] = {}  # noqa: E501
  # noqa: E501
  # noqa: E501
def prompt(name: str):  # noqa: E501
    """Decorator to register a prompt strategy."""  # noqa: E501
  # noqa: E501
    def decorator(fn):  # noqa: E501
        PROMPTS[name] = fn  # noqa: E501
        return fn  # noqa: E501
  # noqa: E501
    return decorator  # noqa: E501
  # noqa: E501
  # noqa: E501
# --- Bare completions ---  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("bare_reply")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"{ctx}\nReply:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("bare_me")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"{ctx}\nMe:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("bare_arrow")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"{ctx}\n>"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("bare_you")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"{ctx}\nYou:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("bare_dash")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"{ctx}\n-"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("bare_sent")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"{ctx}\nSent:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("bare_newline")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"{ctx}\n"  # noqa: E501
  # noqa: E501
  # noqa: E501
# --- Few-shot ---  # noqa: E501
  # noqa: E501
_FS = """Alex: wanna get food?  # noqa: E501
{tag} ya im down, where?  # noqa: E501
  # noqa: E501
Mom: Call me when you get home  # noqa: E501
{tag} k  # noqa: E501
  # noqa: E501
Jordan: you coming tonight?  # noqa: E501
{tag} yea omw  # noqa: E501
  # noqa: E501
"""  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fs_reply")  # noqa: E501
def _(ctx):  # noqa: E501
    return _FS.format(tag="Reply:") + f"{ctx}\nReply:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fs_me")  # noqa: E501
def _(ctx):  # noqa: E501
    return _FS.format(tag="Me:") + f"{ctx}\nMe:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fs_arrow")  # noqa: E501
def _(ctx):  # noqa: E501
    return _FS.format(tag=">") + f"{ctx}\n>"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fs_you")  # noqa: E501
def _(ctx):  # noqa: E501
    return _FS.format(tag="You:") + f"{ctx}\nYou:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fs_dash")  # noqa: E501
def _(ctx):  # noqa: E501
    return _FS.format(tag="-") + f"{ctx}\n-"  # noqa: E501
  # noqa: E501
  # noqa: E501
# --- Few-shot varied ---  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fs_varied")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"""Friend: you up?  # noqa: E501
Reply: yeah what's good  # noqa: E501
  # noqa: E501
Boss: Meeting moved to 3pm  # noqa: E501
Reply: got it thanks  # noqa: E501
  # noqa: E501
Ex: hey  # noqa: E501
Reply: ?  # noqa: E501
  # noqa: E501
Mom: don't forget to call grandma  # noqa: E501
Reply: will do  # noqa: E501
  # noqa: E501
{ctx}  # noqa: E501
Reply:"""  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fs_emoji")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"""Friend: happy birthday!!  # noqa: E501
Reply: tyy 🥳  # noqa: E501
  # noqa: E501
Roommate: left food in the fridge for u  # noqa: E501
Reply: ur the best 🙏  # noqa: E501
  # noqa: E501
Coworker: drinks tonight?  # noqa: E501
Reply: down 🍻  # noqa: E501
  # noqa: E501
{ctx}  # noqa: E501
Reply:"""  # noqa: E501
  # noqa: E501
  # noqa: E501
# --- Instruction-based ---  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("inst_short")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"Reply to this text in under 10 words:\n\n{ctx}\n\nReply:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("inst_no_ai")  # noqa: E501
def _(ctx):  # noqa: E501
    return (  # noqa: E501
        "You're a real person texting. NOT an AI. "  # noqa: E501
        f"Reply naturally, one line max.\n\n{ctx}\n\nReply:"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("inst_complete")  # noqa: E501
def _(ctx):  # noqa: E501
    return (  # noqa: E501
        "Complete the next message in this text conversation. "  # noqa: E501
        f"Output ONLY the reply.\n\n{ctx}\n\nReply:"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("inst_mimic")  # noqa: E501
def _(ctx):  # noqa: E501
    return (  # noqa: E501
        "Read this text thread and write what you'd text back. "  # noqa: E501
        f"Keep it real. Max 1 sentence.\n\n{ctx}\n\nYour reply:"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("inst_terse")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"Respond. 1-5 words only.\n\n{ctx}\n\n>"  # noqa: E501
  # noqa: E501
  # noqa: E501
# --- Few-shot + instruction combo ---  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fsi_brief")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"""Reply to texts briefly. Match the vibe.  # noqa: E501
  # noqa: E501
Alex: wanna get food?  # noqa: E501
Reply: ya im down, where?  # noqa: E501
  # noqa: E501
Mom: Call me when you get home  # noqa: E501
Reply: k  # noqa: E501
  # noqa: E501
{ctx}  # noqa: E501
Reply:"""  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fsi_no_ai")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"""You're texting a friend. Keep it real and short.  # noqa: E501
  # noqa: E501
them: you free tonight?  # noqa: E501
you: yeah what time  # noqa: E501
  # noqa: E501
them: nvm plans changed  # noqa: E501
you: ah ok no worries  # noqa: E501
  # noqa: E501
{ctx}  # noqa: E501
you:"""  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("fsi_persona")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"""You text like a normal 20-something. Short, casual, no AI vibes.  # noqa: E501
  # noqa: E501
them: wanna hang?  # noqa: E501
me: ye when  # noqa: E501
  # noqa: E501
them: running late  # noqa: E501
me: all good take ur time  # noqa: E501
  # noqa: E501
{ctx}  # noqa: E501
me:"""  # noqa: E501
  # noqa: E501
  # noqa: E501
# --- Script / structured ---  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("script")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"INT. TEXT CONVERSATION\n\n{ctx}\nME: "  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("script_fs")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"""INT. TEXT CONVERSATION  # noqa: E501
  # noqa: E501
Alex: wanna get food?  # noqa: E501
ME: ya im down, where?  # noqa: E501
  # noqa: E501
Mom: Call me when you get home  # noqa: E501
ME: k  # noqa: E501
  # noqa: E501
{ctx}  # noqa: E501
ME: """  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("chat_template")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"User: {ctx}\nReply as a real person texting. One sentence max.\nAssistant:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("qa")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"Q: What would you text back to this?\n{ctx}\nA:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("qa_short")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"Q: Reply in 1-5 words.\n{ctx}\nA:"  # noqa: E501
  # noqa: E501
  # noqa: E501
# --- Bracket / tag ---  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("bracket")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"""[them] wanna get food?  # noqa: E501
[me] ya im down, where?  # noqa: E501
  # noqa: E501
[them] Call me when you get home  # noqa: E501
[me] k  # noqa: E501
  # noqa: E501
{ctx}  # noqa: E501
[me]"""  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("xml_reply")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"<conversation>\n{ctx}\n</conversation>\n<reply>"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("xml_short")  # noqa: E501
def _(ctx):  # noqa: E501
    return f'<conversation>\n{ctx}\n</conversation>\n<reply max_words="10">'  # noqa: E501
  # noqa: E501
  # noqa: E501
# --- Dialogue continuation ---  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("dialogue")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"The following is a text conversation. Write ONLY the next reply.\n\n{ctx}\n\n"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("roleplay")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"[You are casually texting. Reply to the last message.]\n\n{ctx}\n\nYou:"  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("texting_sim")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"--- iMessage ---\n{ctx}\n📱 Your reply: "  # noqa: E501
  # noqa: E501
  # noqa: E501
@prompt("social_media")  # noqa: E501
def _(ctx):  # noqa: E501
    return f"💬 DM thread:\n{ctx}\n\nYou replied:"  # noqa: E501
  # noqa: E501
  # noqa: E501
# ── Sampling configs ──────────────────────────────────────────────────────────  # noqa: E501
# Each is a dict of kwargs passed to generate_sync  # noqa: E501
DEFAULT_TEMPS = [0.1, 0.3]  # noqa: E501
DEFAULT_TOP_PS = [0.1, 0.9]  # noqa: E501
DEFAULT_MIN_PS = [0.0, 0.15]  # noqa: E501
  # noqa: E501
# ── Models (LFM + Qwen3 only) ────────────────────────────────────────────────  # noqa: E501
DEFAULT_MODELS = [  # noqa: E501
    "lfm-0.3b",  # noqa: E501
    "lfm-0.7b",  # noqa: E501
    "lfm-0.7b-4bit",  # noqa: E501
    "lfm-1.2b",  # noqa: E501
    "lfm-1.2b-base",  # noqa: E501
    "lfm-2.6b",  # noqa: E501
]  # noqa: E501
  # noqa: E501
DEFAULT_PROMPTS = list(PROMPTS.keys())  # noqa: E501
  # noqa: E501
  # noqa: E501
def run(  # noqa: E501
    models: list[str],  # noqa: E501
    prompt_names: list[str],  # noqa: E501
    test_indices: list[int],  # noqa: E501
    temps: list[float],  # noqa: E501
    top_ps: list[float],  # noqa: E501
    min_ps: list[float],  # noqa: E501
    max_tokens: int,  # noqa: E501
) -> None:  # noqa: E501
    tests = [TEST_CASES[i] for i in test_indices]  # noqa: E501
    sampling_configs = list(product(temps, top_ps, min_ps))  # noqa: E501
  # noqa: E501
    n_calls = len(models) * len(prompt_names) * len(tests) * len(sampling_configs)  # noqa: E501
    print(  # noqa: E501
        f"\n{len(models)} models x {len(prompt_names)} prompts x "  # noqa: E501
        f"{len(tests)} tests x {len(sampling_configs)} sampling configs "  # noqa: E501
        f"= {n_calls} generations",  # noqa: E501
        flush=True,  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # results[model][f"t{temp}_p{top_p}"][prompt_name][test_label] = output_text  # noqa: E501
    all_results: dict = {}  # noqa: E501
  # noqa: E501
    for model_id in models:  # noqa: E501
        print(f"\n{'=' * 70}", flush=True)  # noqa: E501
        print(f"[{mem_mb()}] Loading {model_id}...", flush=True)  # noqa: E501
  # noqa: E501
        try:  # noqa: E501
            t_load = time.monotonic()  # noqa: E501
            config = ModelConfig(model_id=model_id)  # noqa: E501
            loader = MLXModelLoader(config=config)  # noqa: E501
            loader.load()  # noqa: E501
            load_s = time.monotonic() - t_load  # noqa: E501
            print(f"[{mem_mb()}] Loaded in {load_s:.1f}s", flush=True)  # noqa: E501
        except Exception as e:  # noqa: E501
            print(f"  FAILED: {e}", flush=True)  # noqa: E501
            all_results[model_id] = {}  # noqa: E501
            continue  # noqa: E501
  # noqa: E501
        model_results: dict = {}  # noqa: E501
  # noqa: E501
        for temp, top_p, min_p in sampling_configs:  # noqa: E501
            sampling_key = f"t{temp}_p{top_p}_mp{min_p}"  # noqa: E501
            print(f"\n  --- temp={temp} top_p={top_p} min_p={min_p} ---", flush=True)  # noqa: E501
            sampling_results: dict[str, dict[str, str]] = {}  # noqa: E501
  # noqa: E501
            for pname in prompt_names:  # noqa: E501
                pfn = PROMPTS[pname]  # noqa: E501
                prompt_results = {}  # noqa: E501
  # noqa: E501
                for tc in tests:  # noqa: E501
                    prompt_text = pfn(tc["ctx"])  # noqa: E501
                    t0 = time.monotonic()  # noqa: E501
                    try:  # noqa: E501
                        result = loader.generate_sync(  # noqa: E501
                            prompt=prompt_text,  # noqa: E501
                            temperature=temp,  # noqa: E501
                            max_tokens=max_tokens,  # noqa: E501
                            top_p=top_p,  # noqa: E501
                            min_p=min_p,  # noqa: E501
                            top_k=50,  # noqa: E501
                            repetition_penalty=1.05,  # noqa: E501
                        )  # noqa: E501
                        elapsed = time.monotonic() - t0  # noqa: E501
                        text = result.text.strip().replace("\n", " ↵ ")  # noqa: E501
                        tps = result.tokens_generated / elapsed if elapsed > 0 else 0  # noqa: E501
                        display = text[:55] if len(text) > 55 else text  # noqa: E501
                        print(  # noqa: E501
                            f"  {pname:>18} | {tc['label']:>14} | {display:<55} | {tps:.0f}t/s",  # noqa: E501
                            flush=True,  # noqa: E501
                        )  # noqa: E501
                        prompt_results[tc["label"]] = {  # noqa: E501
                            "text": text,  # noqa: E501
                            "tps": round(tps, 1),  # noqa: E501
                            "tokens": result.tokens_generated,  # noqa: E501
                            "elapsed_ms": round(elapsed * 1000, 1),  # noqa: E501
                        }  # noqa: E501
                    except Exception as e:  # noqa: E501
                        print(f"  {pname:>18} | {tc['label']:>14} | ERROR: {e}", flush=True)  # noqa: E501
                        prompt_results[tc["label"]] = {  # noqa: E501
                            "text": f"[ERROR: {e}]",  # noqa: E501
                            "tps": 0,  # noqa: E501
                            "tokens": 0,  # noqa: E501
                            "elapsed_ms": 0,  # noqa: E501
                        }  # noqa: E501
  # noqa: E501
                sampling_results[pname] = prompt_results  # noqa: E501
  # noqa: E501
            model_results[sampling_key] = sampling_results  # noqa: E501
  # noqa: E501
        all_results[model_id] = model_results  # noqa: E501
        loader.unload()  # noqa: E501
        print(f"[{mem_mb()}] Unloaded {model_id}", flush=True)  # noqa: E501
  # noqa: E501
    # ── Save results ────────────────────────────────────────────────────  # noqa: E501
    out_path = PROJECT_ROOT / "evals" / "results" / "comparison-raw.json"  # noqa: E501
    out_path.parent.mkdir(exist_ok=True)  # noqa: E501
    with open(out_path, "w") as f:  # noqa: E501
        json.dump(  # noqa: E501
            {  # noqa: E501
                "models": models,  # noqa: E501
                "prompts": prompt_names,  # noqa: E501
                "tests": [tc["label"] for tc in tests],  # noqa: E501
                "sampling_configs": [  # noqa: E501
                    {"temp": t, "top_p": p, "min_p": mp} for t, p, mp in sampling_configs  # noqa: E501
                ],  # noqa: E501
                "results": all_results,  # noqa: E501
            },  # noqa: E501
            f,  # noqa: E501
            indent=2,  # noqa: E501
        )  # noqa: E501
    print(f"\nResults saved to {out_path}", flush=True)  # noqa: E501
    print(f"Total generations: {n_calls}", flush=True)  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    parser = argparse.ArgumentParser(  # noqa: E501
        description="Model x prompt x sampling comparison for human eval"  # noqa: E501
    )  # noqa: E501
    parser.add_argument("--models", default=None, help="Comma-separated model IDs")  # noqa: E501
    parser.add_argument("--prompts", default=None, help="Comma-separated prompt names")  # noqa: E501
    parser.add_argument("--tests", default=None, help="Comma-separated test indices (0-23)")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--temps", default=None, help="Comma-separated temperatures (default: 0.1,0.5,0.7,1.0)"  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--top-ps", default=None, help="Comma-separated top_p values (default: 0.1,0.9)"  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--min-ps", default=None, help="Comma-separated min_p values (default: 0.0,0.15)"  # noqa: E501
    )  # noqa: E501
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens (default: 20)")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--list-prompts", action="store_true", help="List all prompt names and exit"  # noqa: E501
    )  # noqa: E501
    parser.add_argument("--list-tests", action="store_true", help="List all test cases and exit")  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    if args.list_prompts:  # noqa: E501
        for name in PROMPTS:  # noqa: E501
            sample = PROMPTS[name](TEST_CASES[0]["ctx"])  # noqa: E501
            last_line = sample.strip().split("\n")[-1]  # noqa: E501
            print(f"  {name:>18}: ...{last_line}")  # noqa: E501
        sys.exit(0)  # noqa: E501
  # noqa: E501
    if args.list_tests:  # noqa: E501
        for i, tc in enumerate(TEST_CASES):  # noqa: E501
            ctx_preview = tc["ctx"].replace("\n", " | ")[:60]  # noqa: E501
            print(f"  {i:>2}: {tc['label']:>14} | {ctx_preview}")  # noqa: E501
        sys.exit(0)  # noqa: E501
  # noqa: E501
    models = args.models.split(",") if args.models else DEFAULT_MODELS  # noqa: E501
    prompt_names = args.prompts.split(",") if args.prompts else DEFAULT_PROMPTS  # noqa: E501
    test_indices = (  # noqa: E501
        [int(i) for i in args.tests.split(",")] if args.tests else list(range(len(TEST_CASES)))  # noqa: E501
    )  # noqa: E501
    temps = [float(t) for t in args.temps.split(",")] if args.temps else DEFAULT_TEMPS  # noqa: E501
    top_ps = [float(p) for p in args.top_ps.split(",")] if args.top_ps else DEFAULT_TOP_PS  # noqa: E501
    min_ps = [float(p) for p in args.min_ps.split(",")] if args.min_ps else DEFAULT_MIN_PS  # noqa: E501
  # noqa: E501
    print(f"Models: {', '.join(models)}", flush=True)  # noqa: E501
    print(f"Prompts: {len(prompt_names)} strategies", flush=True)  # noqa: E501
    print(f"Tests: {len(test_indices)} cases", flush=True)  # noqa: E501
    print(f"Temps: {temps}", flush=True)  # noqa: E501
    print(f"Top-p: {top_ps}", flush=True)  # noqa: E501
    print(f"Min-p: {min_ps}", flush=True)  # noqa: E501
  # noqa: E501
    run(models, prompt_names, test_indices, temps, top_ps, min_ps, args.max_tokens)  # noqa: E501
