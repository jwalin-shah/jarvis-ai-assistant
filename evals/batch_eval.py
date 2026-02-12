#!/usr/bin/env python3
"""Batch evaluation: generate responses and judge quality with LLM.

Runs the local MLX model against test cases from promptfoo.yaml,
checks local assertions (length, anti-AI phrases), then scores each
response with Gemini 2.5 Flash via DeepInfra as an LLM judge.

Usage:
    uv run python evals/batch_eval.py              # local checks only
    uv run python evals/batch_eval.py --judge       # + LLM judge scoring
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from evals.judge_config import JUDGE_MODEL  # noqa: E402
from evals.judge_config import get_judge_client as _get_judge_client  # noqa: E402

ANTI_AI_PHRASES = [
    "i'd be happy to",
    "i hope this helps",
    "let me know if",
    "i understand",
    "as an ai",
    "i'm an ai",
    "certainly!",
    "of course!",
    "great question",
]

# Category definitions for per-category MIPRO v2 optimization
CATEGORIES = [
    "brief",
    "warm",
    "social",
    "clarify",
]

# Test cases with category labels for per-category optimization
TEST_CASES = [
    # =========================================================================
    # brief: Short transactional replies (12 cases: 7 casual + 5 professional)
    # =========================================================================
    {
        "name": "Lunch invitation",
        "category": "brief",
        "context": "[14:00] John: Want to grab lunch tomorrow?",
        "last_message": "Want to grab lunch tomorrow?",
        "tone": "casual",
        "user_style": "brief, friendly",
        "max_words": 15,
        "max_chars": 80,
        "banned": ["sounds great", "absolutely"],
        "rubric": (
            "Is this a natural, casual text reply to a lunch invitation? "
            "Should be brief (<15 words), friendly, and sound human (not AI). "
            "Pass if it sounds like a real person texting."
        ),
    },
    {
        "name": "Running late",
        "category": "brief",
        "context": "[09:15] Alex: Running 10 min late",
        "last_message": "Running 10 min late",
        "tone": "casual",
        "user_style": "supportive, brief",
        "max_words": 12,
        "rubric": (
            "Is this a supportive, brief reply to someone running late? "
            "Good: 'no worries', 'all good', 'take your time'. "
            "Bad: asking why, being passive aggressive, too formal."
        ),
    },
    {
        "name": "Simple yes/no - trash",
        "category": "brief",
        "context": "[15:00] Dad: Did you take out the trash?",
        "last_message": "Did you take out the trash?",
        "tone": "casual",
        "user_style": "direct",
        "max_words": 8,
        "rubric": (
            "Is this a direct answer to 'did you take out trash?' "
            "Should be very brief - ideally just 'yes/yep/yeah' or 'no/not yet'. "
            "Fail if it's more than one sentence."
        ),
    },
    {
        "name": "Group chat confirmation",
        "category": "brief",
        "context": (
            "[Group: Game Night]\n"
            "[14:00] Jake: 7pm Saturday work?\n"
            "[14:05] Lisa: I'm in!\n"
            "[14:10] Tom: Works for me"
        ),
        "last_message": "Works for me",
        "tone": "casual",
        "user_style": "brief group energy",
        "max_words": 6,
        "rubric": (
            "Is this a brief group chat confirmation? Should be 1-5 words max. "
            "Good: 'same', 'count me in', 'down'. "
            "Bad: full sentences, formal responses."
        ),
    },
    {
        "name": "Quick favor - pickup",
        "category": "brief",
        "context": "[17:30] Mom: Can you pick up milk on your way home?",
        "last_message": "Can you pick up milk on your way home?",
        "tone": "casual",
        "user_style": "brief, agreeable",
        "max_words": 10,
        "rubric": (
            "Is this a quick confirmation to a simple favor request? "
            "Good: 'sure thing', 'yep', 'on it'. "
            "Bad: asking for details, long response, overly enthusiastic."
        ),
    },
    {
        "name": "ETA check",
        "category": "brief",
        "context": "[19:00] Jake: you close?",
        "last_message": "you close?",
        "tone": "casual",
        "user_style": "brief",
        "max_words": 8,
        "max_chars": 40,
        "rubric": (
            "Quick reply to 'you close?' asking about arrival. "
            "Good: 'yeah 5 min', 'almost there', 'pulling up'. "
            "Bad: long explanation, formal response."
        ),
    },
    {
        "name": "Confirmation - address",
        "category": "brief",
        "context": "[12:00] Sarah: 123 Main St right?",
        "last_message": "123 Main St right?",
        "tone": "casual",
        "user_style": "direct",
        "max_words": 6,
        "rubric": (
            "Confirm or correct an address. "
            "Good: 'yep that's it', 'yeah', 'no it's 125'. "
            "Bad: long explanation, repeating the full address formally."
        ),
    },
    # =========================================================================
    # warm: Emotional weight - comfort or celebrate (5 cases)
    # =========================================================================
    {
        "name": "Emotional support - venting",
        "category": "warm",
        "context": (
            "[20:00] Mike: Work was brutal today\n"
            "[20:01] Mike: Boss dumped a project on me last minute"
        ),
        "last_message": "Boss dumped a project on me last minute",
        "tone": "casual",
        "user_style": "empathetic friend",
        "banned": ["have you tried", "you should"],
        "rubric": (
            "Is this empathetic without giving unsolicited advice? "
            "Good: 'that sucks', 'ugh sorry'. "
            "Bad: 'have you tried...', 'you should...', therapist-speak."
        ),
    },
    {
        "name": "Emotional support - breakup",
        "category": "warm",
        "context": (
            "[22:00] Sarah: Mark and I broke up\n[22:01] Sarah: I don't even know what happened"
        ),
        "last_message": "I don't even know what happened",
        "tone": "casual",
        "user_style": "warm, supportive",
        "banned": ["you'll find someone", "plenty of fish", "you should"],
        "rubric": (
            "Is this supportive without minimizing or giving cliched advice? "
            "Good: 'I'm so sorry', 'that's rough, I'm here for you'. "
            "Bad: 'you'll find someone better', platitudes, therapist-speak."
        ),
    },
    {
        "name": "Emotional support - bad news",
        "category": "warm",
        "context": "[15:00] John: Didn't get the job. Thought the interview went well",
        "last_message": "Didn't get the job. Thought the interview went well",
        "tone": "casual",
        "user_style": "empathetic, brief",
        "banned": ["everything happens for a reason", "you should"],
        "rubric": (
            "Empathetic response to job rejection. "
            "Good: 'damn that sucks', 'their loss honestly'. "
            "Bad: toxic positivity, unsolicited advice, long pep talk."
        ),
    },
    {
        "name": "Emotional support - health worry",
        "category": "warm",
        "context": (
            "[11:00] Mom: Doctor wants to run more tests\n[11:02] Mom: Trying not to worry"
        ),
        "last_message": "Trying not to worry",
        "tone": "casual",
        "user_style": "caring, reassuring",
        "banned": ["i'm sure it's nothing", "don't worry"],
        "rubric": (
            "Supportive response to a parent's health worry. "
            "Good: 'I'm here for you', 'let me know what they say'. "
            "Bad: dismissing worry ('it's probably nothing'), medical advice."
        ),
    },
    {
        "name": "Emotional support - stressed",
        "category": "warm",
        "context": "[23:00] Alex: Can't sleep. Too much on my mind",
        "last_message": "Can't sleep. Too much on my mind",
        "tone": "casual",
        "user_style": "gentle, supportive",
        "banned": ["have you tried", "you should try"],
        "rubric": (
            "Late night support for a stressed friend. "
            "Good: 'wanna talk about it?', 'I'm up if you need to vent'. "
            "Bad: sleep advice, telling them to relax, dismissive."
        ),
    },
    # =========================================================================
    # brief (professional tone): Formal tone handled by detect_tone() (5 cases)
    # =========================================================================
    {
        "name": "Professional - report request",
        "category": "brief",
        "context": "[09:00] Manager: Can you send the Q4 report by EOD?",
        "last_message": "Can you send the Q4 report by EOD?",
        "tone": "professional",
        "user_style": "professional but not stiff",
        "banned": ["lol", "gonna"],
        "rubric": (
            "Is this professional but not stiff? Should confirm the task briefly. "
            "Good: 'Will do', 'On it', 'I'll have it ready'. "
            "Bad: too casual, too formal/corporate."
        ),
    },
    {
        "name": "Professional - meeting reschedule",
        "category": "brief",
        "context": "[10:00] Client: Need to push our 2pm to Thursday. Does that work?",
        "last_message": "Need to push our 2pm to Thursday. Does that work?",
        "tone": "professional",
        "user_style": "polite, concise",
        "banned": ["lol", "haha", "gonna"],
        "rubric": (
            "Professional response to a meeting reschedule. "
            "Good: 'Thursday works for me', 'Sure, same time?'. "
            "Bad: too casual, overly formal, long response."
        ),
    },
    {
        "name": "Professional - project update",
        "category": "brief",
        "context": (
            "[14:00] Manager: How's the migration project coming along?\n"
            "[14:01] Manager: Board wants an update Friday"
        ),
        "last_message": "Board wants an update Friday",
        "tone": "professional",
        "user_style": "clear, status-oriented",
        "banned": ["lol", "dude"],
        "rubric": (
            "Professional status update response. "
            "Good: 'On track. I'll prep a summary for Friday.', 'Will have slides ready'. "
            "Bad: vague, too casual, overly long."
        ),
    },
    {
        "name": "Professional - thank you",
        "category": "brief",
        "context": "[16:00] Colleague: Thanks for covering the call today, really helped",
        "last_message": "Thanks for covering the call today, really helped",
        "tone": "professional",
        "user_style": "warm professional",
        "banned": ["lol", "np bro"],
        "rubric": (
            "Acknowledge thanks from a colleague professionally but warmly. "
            "Good: 'Happy to help', 'Of course, anytime'. "
            "Bad: too casual, dismissive, overly formal."
        ),
    },
    {
        "name": "Professional - deadline question",
        "category": "brief",
        "context": "[11:30] HR: When can you have the compliance training done?",
        "last_message": "When can you have the compliance training done?",
        "tone": "professional",
        "user_style": "direct, professional",
        "banned": ["lol", "idk"],
        "rubric": (
            "Professional response to a deadline question. "
            "Good: 'I'll have it done by end of week', 'Can finish by Wednesday'. "
            "Bad: vague, too casual, no commitment."
        ),
    },
    # =========================================================================
    # social: Casual conversational (6 cases)
    # =========================================================================
    {
        "name": "Photo reaction",
        "category": "social",
        "context": "[16:00] Emma: [Photo]\n[16:00] Emma: Look at this view!",
        "last_message": "Look at this view!",
        "tone": "casual",
        "user_style": "enthusiastic",
        "banned": ["i can see", "the photo"],
        "rubric": (
            "React to a friend sharing a photo of a nice view. "
            "Should be positive and match enthusiasm. "
            "Bad: describing the photo, generic 'nice', overly formal."
        ),
    },
    {
        "name": "Weekend plans",
        "category": "social",
        "context": "[18:30] Sam: Any plans this weekend?",
        "last_message": "Any plans this weekend?",
        "tone": "casual",
        "user_style": "conversational",
        "max_chars": 120,
        "rubric": (
            "Is this a natural response to 'any plans this weekend?' "
            "Should share plans or ask back. "
            "Good: 'Not yet, you?', 'Might grab brunch, wbu?'. "
            "Bad: formal, overly helpful."
        ),
    },
    {
        "name": "Inside joke / unknown reference",
        "category": "social",
        "context": "[14:00] Tom: lmao remember the thing",
        "last_message": "lmao remember the thing",
        "tone": "casual",
        "user_style": "casual bro",
        "max_chars": 40,
        "rubric": (
            "Reply to an inside joke reference ('the thing') that the model can't "
            "possibly know. Should NOT pretend to know. "
            "Good: 'lol which thing', 'haha yes', 'omg yes'. "
            "Bad: making up a specific memory, detailed response about 'the thing'."
        ),
    },
    {
        "name": "Long time no talk",
        "category": "social",
        "context": "[19:00] College Friend: Dude it's been forever! How are you??",
        "last_message": "Dude it's been forever! How are you??",
        "tone": "casual",
        "user_style": "warm, conversational",
        "max_chars": 120,
        "rubric": (
            "Warm response to a friend reaching out after a long time. "
            "Good: 'I know right! I'm good, how about you?'. "
            "Bad: formal, distant, overly detailed life update."
        ),
    },
    {
        "name": "Travel flex",
        "category": "social",
        "context": "[10:00] Lisa: Just landed in Tokyo!!",
        "last_message": "Just landed in Tokyo!!",
        "tone": "casual",
        "user_style": "excited, enthusiastic",
        "banned": ["i hope you", "have a wonderful"],
        "rubric": (
            "Excited response to a friend's travel announcement. "
            "Good: 'omg so jealous!', 'yesss enjoy!', 'send pics!'. "
            "Bad: formal wishes, travel advice, assistant-like response."
        ),
    },
    {
        "name": "Music share",
        "category": "social",
        "context": "[21:00] Jake: Have you heard the new Kendrick album?",
        "last_message": "Have you heard the new Kendrick album?",
        "tone": "casual",
        "user_style": "casual, opinionated",
        "max_chars": 80,
        "rubric": (
            "Natural response to a music recommendation question. "
            "Good: 'not yet, is it good?', 'yeah it slaps'. "
            "Bad: formal review, overly long, AI-sounding analysis."
        ),
    },
    # =========================================================================
    # clarify: Low-context / ambiguous (7 cases)
    # =========================================================================
    {
        "name": "Ambiguous question mark",
        "category": "clarify",
        "context": "[11:00] Chris: ?",
        "last_message": "?",
        "tone": "casual",
        "user_style": "casual",
        "rubric": (
            "Reply to just a '?' with no context. Should ask for clarification briefly. "
            "Good: 'what's up?', '??', 'hm?'. "
            "Bad: long response, assuming what they mean."
        ),
    },
    {
        "name": "No context - bare hey",
        "category": "clarify",
        "context": "[11:00] Unknown: hey",
        "last_message": "hey",
        "tone": "casual",
        "user_style": "",
        "max_chars": 30,
        "rubric": (
            "Reply to 'hey' from an unknown person with zero context. "
            "Should be very brief - a simple greeting back. "
            "Good: 'hey', 'hey what's up', 'yo'. "
            "Bad: long reply, introducing yourself, asking detailed questions."
        ),
    },
    {
        "name": "Ambiguous forwarded link",
        "category": "clarify",
        "context": "[12:00] Sarah: [Link]",
        "last_message": "[Link]",
        "tone": "casual",
        "user_style": "",
        "max_chars": 40,
        "banned": ["article", "interesting"],
        "rubric": (
            "Someone sent just a link with no text. Model should NOT confabulate "
            "what the link is about. Good: 'what's this?', '?', 'ooh what is it'. "
            "Bad: commenting on the content, assuming what it is."
        ),
    },
    {
        "name": "Stale thread - weeks old",
        "category": "clarify",
        "context": "[3 weeks ago] Dave: hey you free Saturday?",
        "last_message": "hey you free Saturday?",
        "tone": "casual",
        "user_style": "casual",
        "max_chars": 60,
        "rubric": (
            "Replying to a 3-week-old message asking about Saturday. "
            "Should acknowledge the staleness or not pretend it's timely. "
            "Good: 'sorry just saw this', 'lol my bad, super late'. "
            "Bad: answering as if it's current ('yeah I'm free!')."
        ),
    },
    {
        "name": "Emoji only message",
        "category": "clarify",
        "context": "[13:00] Tina: \U0001f602\U0001f602\U0001f602",
        "last_message": "\U0001f602\U0001f602\U0001f602",
        "tone": "casual",
        "user_style": "casual",
        "max_chars": 30,
        "rubric": (
            "Reply to a message that's just laughing emojis with no context. "
            "Good: 'lol', '\U0001f602', 'what', '??'. "
            "Bad: long response, asking detailed questions, pretending to know what's funny."
        ),
    },
    {
        "name": "Voice memo reference",
        "category": "clarify",
        "context": "[16:00] Dan: [Voice Memo]",
        "last_message": "[Voice Memo]",
        "tone": "casual",
        "user_style": "",
        "max_chars": 50,
        "rubric": (
            "Reply to a voice memo that can't be transcribed. "
            "Good: 'can't listen rn, what's up?', 'send a text lol'. "
            "Bad: pretending to have heard it, long response."
        ),
    },
    {
        "name": "Wrong number / random text",
        "category": "clarify",
        "context": "[08:00] Unknown: Tell Maria I'll be there at 6",
        "last_message": "Tell Maria I'll be there at 6",
        "tone": "casual",
        "user_style": "",
        "max_chars": 50,
        "rubric": (
            "Reply to what looks like a wrong-number text. "
            "Good: 'wrong number', 'think you have the wrong person'. "
            "Bad: agreeing to tell Maria, long explanation."
        ),
    },
]


@dataclass
class EvalResult:
    name: str
    category: str
    output: str
    latency_ms: float
    checks_passed: list[str]
    checks_failed: list[str]
    passed: bool
    judge_score: float | None = None
    judge_reasoning: str = ""


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------


def get_judge_client():
    """Create OpenAI-compatible client for the judge model."""
    return _get_judge_client()


def judge_response(client, tc: dict, output: str) -> tuple[float, str]:
    """Score a response using the LLM judge.

    Returns (score 0-10, reasoning).
    """
    rubric = tc.get("rubric", "")
    if not rubric:
        return -1.0, "no rubric"

    prompt = (
        "You are an expert evaluator for a text message reply generator.\n\n"
        f"CONVERSATION:\n{tc['context']}\n\n"
        f"LAST MESSAGE (to reply to):\n{tc['last_message']}\n\n"
        f"GENERATED REPLY:\n{output}\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        "Score the generated reply from 0-10 based on the rubric.\n"
        "Respond in this exact JSON format:\n"
        '{"score": <0-10>, "reasoning": "<1-2 sentences>"}'
    )

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
        # Parse JSON from response (handle markdown fences)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return float(data["score"]), data.get("reasoning", "")
    except Exception as e:
        return -1.0, f"judge error: {e}"


# ---------------------------------------------------------------------------
# Local Checks
# ---------------------------------------------------------------------------


def build_prompt(tc: dict) -> str:
    """Build XML drafter prompt (default strategy)."""
    system = (
        "You draft text message replies matching the sender's exact style.\n"
        "Rules:\n"
        "- Match their texting style exactly "
        "(length, formality, abbreviations, emoji, punctuation)\n"
        "- Sound natural, never like an AI\n"
        '- No phrases like "I hope this helps" or "Let me know"\n'
        "- No formal greetings unless they use them\n"
        "- If the message is unclear or you lack context to reply properly, "
        'respond with just "?"'
    )
    style = (
        f"Tone: {tc['tone']}. Style: {tc['user_style']}"
        if tc["user_style"]
        else f"Tone: {tc['tone']}"
    )
    return (
        f"<system>\n{system}</system>\n\n"
        f"<style>\n{style}\n</style>\n\n"
        f"<conversation>\n{tc['context']}\n</conversation>\n\n"
        f"<last_message>{tc['last_message']}</last_message>\n\n"
        f"<reply>"
    )


def check_result(tc: dict, output: str) -> tuple[list[str], list[str]]:
    """Run local assertions. Returns (passed, failed)."""
    passed = []
    failed = []
    lower = output.lower()

    # Anti-AI phrases (global)
    for phrase in ANTI_AI_PHRASES:
        if phrase in lower:
            failed.append(f"contains anti-AI phrase: '{phrase}'")
        else:
            passed.append(f"no '{phrase}'")

    # Max words
    if "max_words" in tc:
        word_count = len(output.split())
        if word_count <= tc["max_words"]:
            passed.append(f"words={word_count} <= {tc['max_words']}")
        else:
            failed.append(f"words={word_count} > {tc['max_words']}")

    # Max chars
    if "max_chars" in tc:
        if len(output) <= tc["max_chars"]:
            passed.append(f"chars={len(output)} <= {tc['max_chars']}")
        else:
            failed.append(f"chars={len(output)} > {tc['max_chars']}")

    # Banned words
    for word in tc.get("banned", []):
        if word.lower() in lower:
            failed.append(f"contains banned: '{word}'")
        else:
            passed.append(f"no '{word}'")

    # Basic sanity
    if not output.strip():
        failed.append("empty output")
    else:
        passed.append("non-empty output")

    if len(output) > 300:
        failed.append(f"way too long ({len(output)} chars)")
    else:
        passed.append(f"reasonable length ({len(output)} chars)")

    return passed, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Batch Eval")
    parser.add_argument(
        "--judge", action="store_true", help="Enable LLM judge scoring via Cerebras"
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Use DSPy-compiled program instead of raw generation",
    )
    args = parser.parse_args()

    # Setup logging
    log_path = PROJECT_ROOT / "results" / "batch_eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger(__name__)

    strategy = "dspy_optimized" if args.optimized else "xml_drafter"
    print("=" * 70, flush=True)
    print("JARVIS BATCH EVAL - Response Generation", flush=True)
    print("=" * 70, flush=True)
    print(f"Test cases:  {len(TEST_CASES)}", flush=True)
    print(f"Strategy:    {strategy}", flush=True)
    judge_label = f"{JUDGE_MODEL} via DeepInfra" if args.judge else "disabled (use --judge)"
    print(f"LLM judge:   {judge_label}", flush=True)
    print(flush=True)

    # Init judge
    judge_client = None
    if args.judge:
        judge_client = get_judge_client()
        if judge_client is None:
            print("WARNING: CEREBRAS_API_KEY not set in .env - skipping judge", flush=True)
            print("         Put your key in .env and re-run with --judge", flush=True)
        else:
            print(f"Judge ready: {JUDGE_MODEL} via Cerebras", flush=True)
    print(flush=True)

    # Load model / compiled program
    dspy_program = None
    loader = None

    # Set MLX memory limits early to prevent swap thrashing on 8GB systems.
    # loader.load() also sets these, but we set them before any MLX import
    # to guard against accidental early allocation.
    from models.memory_config import apply_embedder_limits

    apply_embedder_limits()

    if args.optimized:
        optimized_dir = PROJECT_ROOT / "evals" / "optimized_reply"
        if not optimized_dir.exists():
            print("ERROR: No compiled program found at evals/optimized_reply/", flush=True)
            print("       Run: uv run python evals/dspy_optimize.py", flush=True)
            return 1
        print("Loading DSPy compiled program...", flush=True)
        load_start = time.perf_counter()
        try:
            import dspy

            from evals.dspy_reply import ReplyModule
            from jarvis.dspy_client import DSPYMLXClient

            student_lm = DSPYMLXClient(max_tokens=50, temperature=0.1)
            dspy.configure(lm=student_lm)
            dspy_program = ReplyModule()
            dspy_program.load(str(optimized_dir))
            load_ms = (time.perf_counter() - load_start) * 1000
            print(f"Compiled program loaded in {load_ms:.0f}ms", flush=True)
        except Exception as e:
            print(f"FATAL: Failed to load compiled program: {e}", flush=True)
            return 1
    else:
        print("Loading MLX model...", flush=True)
        load_start = time.perf_counter()
        try:
            from models.loader import get_model

            loader = get_model()
            if not loader.is_loaded():
                loader.load()
            load_ms = (time.perf_counter() - load_start) * 1000
            print(f"Model loaded in {load_ms:.0f}ms", flush=True)
        except Exception as e:
            print(f"FATAL: Failed to load model: {e}", flush=True)
            return 1

    print(flush=True)
    print("-" * 70, flush=True)

    results: list[EvalResult] = []
    total_start = time.perf_counter()

    # Resume support: load partial results from checkpoint file
    checkpoint_path = PROJECT_ROOT / "results" / "batch_eval_checkpoint.jsonl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    completed_names: set[str] = set()
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                completed_names.add(rec["name"])
                results.append(
                    EvalResult(
                        name=rec["name"],
                        category=rec["category"],
                        output=rec["output"],
                        latency_ms=rec["latency_ms"],
                        checks_passed=rec.get("checks_passed", []),
                        checks_failed=rec.get("checks_failed", []),
                        passed=rec["local_passed"],
                        judge_score=rec.get("judge_score"),
                        judge_reasoning=rec.get("judge_reasoning", ""),
                    )
                )
        if completed_names:
            print(
                f"Resuming: {len(completed_names)}/{len(TEST_CASES)} already completed",
                flush=True,
            )

    checkpoint_f = checkpoint_path.open("a", encoding="utf-8")

    for i, tc in enumerate(tqdm(TEST_CASES, desc="Evaluating"), 1):
        if tc["name"] in completed_names:
            continue
        # Generate via DSPy compiled program or raw model
        gen_start = time.perf_counter()
        try:
            if dspy_program is not None:
                pred = dspy_program(
                    context=tc["context"],
                    last_message=tc["last_message"],
                    tone=tc["tone"],
                    user_style=tc.get("user_style", ""),
                )
                output = pred.reply.strip()
            else:
                prompt = build_prompt(tc)
                result = loader.generate_sync(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=50,
                    top_p=0.1,
                    top_k=50,
                    repetition_penalty=1.05,
                )
                output = result.text.strip()
            latency_ms = (time.perf_counter() - gen_start) * 1000
        except Exception as e:
            output = f"[ERROR: {e}]"
            latency_ms = (time.perf_counter() - gen_start) * 1000

        # Local checks
        passed_checks, failed_checks = check_result(tc, output)
        all_passed = len(failed_checks) == 0

        # Judge scoring
        judge_score = None
        judge_reasoning = ""
        if judge_client and tc.get("rubric"):
            judge_score, judge_reasoning = judge_response(judge_client, tc, output)

        er = EvalResult(
            name=tc["name"],
            category=tc.get("category", "unknown"),
            output=output,
            latency_ms=latency_ms,
            checks_passed=passed_checks,
            checks_failed=failed_checks,
            passed=all_passed,
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
        )
        results.append(er)

        # Write checkpoint incrementally (survives crash)
        checkpoint_f.write(
            json.dumps(
                {
                    "name": er.name,
                    "category": er.category,
                    "output": er.output,
                    "latency_ms": round(er.latency_ms, 1),
                    "local_passed": er.passed,
                    "checks_passed": er.checks_passed,
                    "checks_failed": er.checks_failed,
                    "judge_score": er.judge_score,
                    "judge_reasoning": er.judge_reasoning,
                }
            )
            + "\n"
        )
        checkpoint_f.flush()

        # Print per-case
        status = "PASS" if all_passed else "FAIL"
        cat = tc.get("category", "?")
        print(f"\n[{i:2d}/{len(TEST_CASES)}] [{cat}] {tc['name']}", flush=True)
        print(f'  Output:  "{output}"', flush=True)
        judge_str = ""
        if judge_score is not None and judge_score >= 0:
            judge_str = f" | Judge: {judge_score:.0f}/10"
        print(f"  Latency: {latency_ms:.0f}ms | Local: {status}{judge_str}", flush=True)
        if failed_checks:
            for f in failed_checks:
                print(f"  FAIL: {f}", flush=True)
        if judge_reasoning:
            print(f"  Judge: {judge_reasoning}", flush=True)

    total_ms = (time.perf_counter() - total_start) * 1000

    # Summary
    print(flush=True)
    print("=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    n_passed = sum(1 for r in results if r.passed)
    n_failed = len(results) - n_passed
    latencies = [r.latency_ms for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
    p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
    p95 = sorted_lat[p95_idx] if sorted_lat else 0

    print(
        f"Local pass:   {n_passed}/{len(results)} ({n_passed / len(results) * 100:.0f}%)",
        flush=True,
    )
    print(f"Failed:       {n_failed}", flush=True)
    print(f"Total time:   {total_ms:.0f}ms", flush=True)
    print(f"Avg latency:  {avg_latency:.0f}ms", flush=True)
    print(f"P50 latency:  {p50:.0f}ms", flush=True)
    print(f"P95 latency:  {p95:.0f}ms", flush=True)

    # Judge summary
    scored = [r for r in results if r.judge_score is not None and r.judge_score >= 0]
    if scored:
        scores = [r.judge_score for r in scored]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        pass_7 = sum(1 for s in scores if s >= 7)
        print(flush=True)
        print(
            f"Judge scores: avg={avg_score:.1f}/10  min={min_score:.0f}  max={max_score:.0f}",
            flush=True,
        )
        print(
            f"Judge pass (>=7): {pass_7}/{len(scored)} ({pass_7 / len(scored) * 100:.0f}%)",
            flush=True,
        )

    if n_failed:
        print("\nLocal failures:", flush=True)
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {', '.join(r.checks_failed)}", flush=True)

    # Per-category breakdown
    print(flush=True)
    print("PER-CATEGORY BREAKDOWN", flush=True)
    print("-" * 70, flush=True)
    for cat in CATEGORIES:
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
        cat_passed = sum(1 for r in cat_results if r.passed)
        cat_scored = [r for r in cat_results if r.judge_score is not None and r.judge_score >= 0]
        cat_judge = ""
        if cat_scored:
            cat_avg = sum(r.judge_score for r in cat_scored) / len(cat_scored)
            cat_judge = f"  judge_avg={cat_avg:.1f}/10"
        print(
            f"  {cat:20s}  local={cat_passed}/{len(cat_results)}"
            f" ({cat_passed / len(cat_results) * 100:.0f}%){cat_judge}",
            flush=True,
        )

    if scored:
        low = [r for r in scored if r.judge_score < 7]
        if low:
            print("\nLow judge scores (<7):", flush=True)
            for r in low:
                print(f"  - {r.name}: {r.judge_score:.0f}/10 - {r.judge_reasoning}", flush=True)

    # Save results
    output_path = PROJECT_ROOT / "results" / "batch_eval_latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "strategy": strategy,
        "judge_model": JUDGE_MODEL if scored else None,
        "total": len(results),
        "local_passed": n_passed,
        "local_failed": n_failed,
        "local_pass_rate": round(n_passed / len(results), 4),
        "judge_avg_score": round(avg_score, 2) if scored else None,
        "judge_pass_rate": round(pass_7 / len(scored), 4) if scored else None,
        "latency": {
            "avg_ms": round(avg_latency, 1),
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "total_ms": round(total_ms, 1),
        },
        "results": [
            {
                "name": r.name,
                "category": r.category,
                "output": r.output,
                "latency_ms": round(r.latency_ms, 1),
                "local_passed": r.passed,
                "failed_checks": r.checks_failed,
                "judge_score": r.judge_score,
                "judge_reasoning": r.judge_reasoning,
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    # Clean up checkpoint file after successful completion
    checkpoint_f.close()
    checkpoint_path.unlink(missing_ok=True)
    print(f"\nResults saved to: {output_path}", flush=True)
    print("=" * 70, flush=True)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
