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
  # noqa: E402
# Load .env  # noqa: E402
_env_path = PROJECT_ROOT / ".env"  # noqa: E402
if _env_path.exists():  # noqa: E402
    for line in _env_path.read_text().splitlines():  # noqa: E402
        line = line.strip()  # noqa: E402
        if line and not line.startswith("#") and "=" in line:  # noqa: E402
            key, _, val = line.partition("=")  # noqa: E402
            os.environ.setdefault(key.strip(), val.strip())  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Constants  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
from evals.judge_config import JUDGE_MODEL  # noqa: E402
from evals.judge_config import get_judge_client as _get_judge_client  # noqa: E402

  # noqa: E402
ANTI_AI_PHRASES = [  # noqa: E402
    "i'd be happy to",  # noqa: E402
    "i hope this helps",  # noqa: E402
    "let me know if",  # noqa: E402
    "i understand",  # noqa: E402
    "as an ai",  # noqa: E402
    "i'm an ai",  # noqa: E402
    "certainly!",  # noqa: E402
    "of course!",  # noqa: E402
    "great question",  # noqa: E402
]  # noqa: E402
  # noqa: E402
# Category definitions for per-category MIPRO v2 optimization  # noqa: E402
CATEGORIES = [  # noqa: E402
    "brief",  # noqa: E402
    "warm",  # noqa: E402
    "social",  # noqa: E402
    "clarify",  # noqa: E402
]  # noqa: E402
  # noqa: E402
# Test cases with category labels for per-category optimization  # noqa: E402
TEST_CASES = [  # noqa: E402
    # =========================================================================  # noqa: E402
    # brief: Short transactional replies (12 cases: 7 casual + 5 professional)  # noqa: E402
    # =========================================================================  # noqa: E402
    {  # noqa: E402
        "name": "Lunch invitation",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[14:00] John: Want to grab lunch tomorrow?",  # noqa: E402
        "last_message": "Want to grab lunch tomorrow?",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "brief, friendly",  # noqa: E402
        "max_words": 15,  # noqa: E402
        "max_chars": 80,  # noqa: E402
        "banned": ["sounds great", "absolutely"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Is this a natural, casual text reply to a lunch invitation? "  # noqa: E402
            "Should be brief (<15 words), friendly, and sound human (not AI). "  # noqa: E402
            "Pass if it sounds like a real person texting."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Running late",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[09:15] Alex: Running 10 min late",  # noqa: E402
        "last_message": "Running 10 min late",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "supportive, brief",  # noqa: E402
        "max_words": 12,  # noqa: E402
        "rubric": (  # noqa: E402
            "Is this a supportive, brief reply to someone running late? "  # noqa: E402
            "Good: 'no worries', 'all good', 'take your time'. "  # noqa: E402
            "Bad: asking why, being passive aggressive, too formal."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Simple yes/no - trash",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[15:00] Dad: Did you take out the trash?",  # noqa: E402
        "last_message": "Did you take out the trash?",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "direct",  # noqa: E402
        "max_words": 8,  # noqa: E402
        "rubric": (  # noqa: E402
            "Is this a direct answer to 'did you take out trash?' "  # noqa: E402
            "Should be very brief - ideally just 'yes/yep/yeah' or 'no/not yet'. "  # noqa: E402
            "Fail if it's more than one sentence."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Group chat confirmation",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": (  # noqa: E402
            "[Group: Game Night]\n"  # noqa: E402
            "[14:00] Jake: 7pm Saturday work?\n"  # noqa: E402
            "[14:05] Lisa: I'm in!\n"  # noqa: E402
            "[14:10] Tom: Works for me"  # noqa: E402
        ),  # noqa: E402
        "last_message": "Works for me",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "brief group energy",  # noqa: E402
        "max_words": 6,  # noqa: E402
        "rubric": (  # noqa: E402
            "Is this a brief group chat confirmation? Should be 1-5 words max. "  # noqa: E402
            "Good: 'same', 'count me in', 'down'. "  # noqa: E402
            "Bad: full sentences, formal responses."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Quick favor - pickup",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[17:30] Mom: Can you pick up milk on your way home?",  # noqa: E402
        "last_message": "Can you pick up milk on your way home?",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "brief, agreeable",  # noqa: E402
        "max_words": 10,  # noqa: E402
        "rubric": (  # noqa: E402
            "Is this a quick confirmation to a simple favor request? "  # noqa: E402
            "Good: 'sure thing', 'yep', 'on it'. "  # noqa: E402
            "Bad: asking for details, long response, overly enthusiastic."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "ETA check",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[19:00] Jake: you close?",  # noqa: E402
        "last_message": "you close?",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "brief",  # noqa: E402
        "max_words": 8,  # noqa: E402
        "max_chars": 40,  # noqa: E402
        "rubric": (  # noqa: E402
            "Quick reply to 'you close?' asking about arrival. "  # noqa: E402
            "Good: 'yeah 5 min', 'almost there', 'pulling up'. "  # noqa: E402
            "Bad: long explanation, formal response."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Confirmation - address",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[12:00] Sarah: 123 Main St right?",  # noqa: E402
        "last_message": "123 Main St right?",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "direct",  # noqa: E402
        "max_words": 6,  # noqa: E402
        "rubric": (  # noqa: E402
            "Confirm or correct an address. "  # noqa: E402
            "Good: 'yep that's it', 'yeah', 'no it's 125'. "  # noqa: E402
            "Bad: long explanation, repeating the full address formally."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    # =========================================================================  # noqa: E402
    # warm: Emotional weight - comfort or celebrate (5 cases)  # noqa: E402
    # =========================================================================  # noqa: E402
    {  # noqa: E402
        "name": "Emotional support - venting",  # noqa: E402
        "category": "warm",  # noqa: E402
        "context": (  # noqa: E402
            "[20:00] Mike: Work was brutal today\n"  # noqa: E402
            "[20:01] Mike: Boss dumped a project on me last minute"  # noqa: E402
        ),  # noqa: E402
        "last_message": "Boss dumped a project on me last minute",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "empathetic friend",  # noqa: E402
        "banned": ["have you tried", "you should"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Is this empathetic without giving unsolicited advice? "  # noqa: E402
            "Good: 'that sucks', 'ugh sorry'. "  # noqa: E402
            "Bad: 'have you tried...', 'you should...', therapist-speak."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Emotional support - breakup",  # noqa: E402
        "category": "warm",  # noqa: E402
        "context": (  # noqa: E402
            "[22:00] Sarah: Mark and I broke up\n[22:01] Sarah: I don't even know what happened"  # noqa: E402
        ),  # noqa: E402
        "last_message": "I don't even know what happened",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "warm, supportive",  # noqa: E402
        "banned": ["you'll find someone", "plenty of fish", "you should"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Is this supportive without minimizing or giving cliched advice? "  # noqa: E402
            "Good: 'I'm so sorry', 'that's rough, I'm here for you'. "  # noqa: E402
            "Bad: 'you'll find someone better', platitudes, therapist-speak."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Emotional support - bad news",  # noqa: E402
        "category": "warm",  # noqa: E402
        "context": "[15:00] John: Didn't get the job. Thought the interview went well",  # noqa: E402
        "last_message": "Didn't get the job. Thought the interview went well",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "empathetic, brief",  # noqa: E402
        "banned": ["everything happens for a reason", "you should"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Empathetic response to job rejection. "  # noqa: E402
            "Good: 'damn that sucks', 'their loss honestly'. "  # noqa: E402
            "Bad: toxic positivity, unsolicited advice, long pep talk."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Emotional support - health worry",  # noqa: E402
        "category": "warm",  # noqa: E402
        "context": (  # noqa: E402
            "[11:00] Mom: Doctor wants to run more tests\n[11:02] Mom: Trying not to worry"  # noqa: E402
        ),  # noqa: E402
        "last_message": "Trying not to worry",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "caring, reassuring",  # noqa: E402
        "banned": ["i'm sure it's nothing", "don't worry"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Supportive response to a parent's health worry. "  # noqa: E402
            "Good: 'I'm here for you', 'let me know what they say'. "  # noqa: E402
            "Bad: dismissing worry ('it's probably nothing'), medical advice."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Emotional support - stressed",  # noqa: E402
        "category": "warm",  # noqa: E402
        "context": "[23:00] Alex: Can't sleep. Too much on my mind",  # noqa: E402
        "last_message": "Can't sleep. Too much on my mind",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "gentle, supportive",  # noqa: E402
        "banned": ["have you tried", "you should try"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Late night support for a stressed friend. "  # noqa: E402
            "Good: 'wanna talk about it?', 'I'm up if you need to vent'. "  # noqa: E402
            "Bad: sleep advice, telling them to relax, dismissive."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    # =========================================================================  # noqa: E402
    # brief (professional tone): Formal tone handled by detect_tone() (5 cases)  # noqa: E402
    # =========================================================================  # noqa: E402
    {  # noqa: E402
        "name": "Professional - report request",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[09:00] Manager: Can you send the Q4 report by EOD?",  # noqa: E402
        "last_message": "Can you send the Q4 report by EOD?",  # noqa: E402
        "tone": "professional",  # noqa: E402
        "user_style": "professional but not stiff",  # noqa: E402
        "banned": ["lol", "gonna"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Is this professional but not stiff? Should confirm the task briefly. "  # noqa: E402
            "Good: 'Will do', 'On it', 'I'll have it ready'. "  # noqa: E402
            "Bad: too casual, too formal/corporate."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Professional - meeting reschedule",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[10:00] Client: Need to push our 2pm to Thursday. Does that work?",  # noqa: E402
        "last_message": "Need to push our 2pm to Thursday. Does that work?",  # noqa: E402
        "tone": "professional",  # noqa: E402
        "user_style": "polite, concise",  # noqa: E402
        "banned": ["lol", "haha", "gonna"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Professional response to a meeting reschedule. "  # noqa: E402
            "Good: 'Thursday works for me', 'Sure, same time?'. "  # noqa: E402
            "Bad: too casual, overly formal, long response."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Professional - project update",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": (  # noqa: E402
            "[14:00] Manager: How's the migration project coming along?\n"  # noqa: E402
            "[14:01] Manager: Board wants an update Friday"  # noqa: E402
        ),  # noqa: E402
        "last_message": "Board wants an update Friday",  # noqa: E402
        "tone": "professional",  # noqa: E402
        "user_style": "clear, status-oriented",  # noqa: E402
        "banned": ["lol", "dude"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Professional status update response. "  # noqa: E402
            "Good: 'On track. I'll prep a summary for Friday.', 'Will have slides ready'. "  # noqa: E402
            "Bad: vague, too casual, overly long."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Professional - thank you",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[16:00] Colleague: Thanks for covering the call today, really helped",  # noqa: E402
        "last_message": "Thanks for covering the call today, really helped",  # noqa: E402
        "tone": "professional",  # noqa: E402
        "user_style": "warm professional",  # noqa: E402
        "banned": ["lol", "np bro"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Acknowledge thanks from a colleague professionally but warmly. "  # noqa: E402
            "Good: 'Happy to help', 'Of course, anytime'. "  # noqa: E402
            "Bad: too casual, dismissive, overly formal."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Professional - deadline question",  # noqa: E402
        "category": "brief",  # noqa: E402
        "context": "[11:30] HR: When can you have the compliance training done?",  # noqa: E402
        "last_message": "When can you have the compliance training done?",  # noqa: E402
        "tone": "professional",  # noqa: E402
        "user_style": "direct, professional",  # noqa: E402
        "banned": ["lol", "idk"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Professional response to a deadline question. "  # noqa: E402
            "Good: 'I'll have it done by end of week', 'Can finish by Wednesday'. "  # noqa: E402
            "Bad: vague, too casual, no commitment."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    # =========================================================================  # noqa: E402
    # social: Casual conversational (6 cases)  # noqa: E402
    # =========================================================================  # noqa: E402
    {  # noqa: E402
        "name": "Photo reaction",  # noqa: E402
        "category": "social",  # noqa: E402
        "context": "[16:00] Emma: [Photo]\n[16:00] Emma: Look at this view!",  # noqa: E402
        "last_message": "Look at this view!",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "enthusiastic",  # noqa: E402
        "banned": ["i can see", "the photo"],  # noqa: E402
        "rubric": (  # noqa: E402
            "React to a friend sharing a photo of a nice view. "  # noqa: E402
            "Should be positive and match enthusiasm. "  # noqa: E402
            "Bad: describing the photo, generic 'nice', overly formal."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Weekend plans",  # noqa: E402
        "category": "social",  # noqa: E402
        "context": "[18:30] Sam: Any plans this weekend?",  # noqa: E402
        "last_message": "Any plans this weekend?",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "conversational",  # noqa: E402
        "max_chars": 120,  # noqa: E402
        "rubric": (  # noqa: E402
            "Is this a natural response to 'any plans this weekend?' "  # noqa: E402
            "Should share plans or ask back. "  # noqa: E402
            "Good: 'Not yet, you?', 'Might grab brunch, wbu?'. "  # noqa: E402
            "Bad: formal, overly helpful."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Inside joke / unknown reference",  # noqa: E402
        "category": "social",  # noqa: E402
        "context": "[14:00] Tom: lmao remember the thing",  # noqa: E402
        "last_message": "lmao remember the thing",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "casual bro",  # noqa: E402
        "max_chars": 40,  # noqa: E402
        "rubric": (  # noqa: E402
            "Reply to an inside joke reference ('the thing') that the model can't "  # noqa: E402
            "possibly know. Should NOT pretend to know. "  # noqa: E402
            "Good: 'lol which thing', 'haha yes', 'omg yes'. "  # noqa: E402
            "Bad: making up a specific memory, detailed response about 'the thing'."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Long time no talk",  # noqa: E402
        "category": "social",  # noqa: E402
        "context": "[19:00] College Friend: Dude it's been forever! How are you??",  # noqa: E402
        "last_message": "Dude it's been forever! How are you??",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "warm, conversational",  # noqa: E402
        "max_chars": 120,  # noqa: E402
        "rubric": (  # noqa: E402
            "Warm response to a friend reaching out after a long time. "  # noqa: E402
            "Good: 'I know right! I'm good, how about you?'. "  # noqa: E402
            "Bad: formal, distant, overly detailed life update."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Travel flex",  # noqa: E402
        "category": "social",  # noqa: E402
        "context": "[10:00] Lisa: Just landed in Tokyo!!",  # noqa: E402
        "last_message": "Just landed in Tokyo!!",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "excited, enthusiastic",  # noqa: E402
        "banned": ["i hope you", "have a wonderful"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Excited response to a friend's travel announcement. "  # noqa: E402
            "Good: 'omg so jealous!', 'yesss enjoy!', 'send pics!'. "  # noqa: E402
            "Bad: formal wishes, travel advice, assistant-like response."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Music share",  # noqa: E402
        "category": "social",  # noqa: E402
        "context": "[21:00] Jake: Have you heard the new Kendrick album?",  # noqa: E402
        "last_message": "Have you heard the new Kendrick album?",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "casual, opinionated",  # noqa: E402
        "max_chars": 80,  # noqa: E402
        "rubric": (  # noqa: E402
            "Natural response to a music recommendation question. "  # noqa: E402
            "Good: 'not yet, is it good?', 'yeah it slaps'. "  # noqa: E402
            "Bad: formal review, overly long, AI-sounding analysis."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    # =========================================================================  # noqa: E402
    # clarify: Low-context / ambiguous (7 cases)  # noqa: E402
    # =========================================================================  # noqa: E402
    {  # noqa: E402
        "name": "Ambiguous question mark",  # noqa: E402
        "category": "clarify",  # noqa: E402
        "context": "[11:00] Chris: ?",  # noqa: E402
        "last_message": "?",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "casual",  # noqa: E402
        "rubric": (  # noqa: E402
            "Reply to just a '?' with no context. Should ask for clarification briefly. "  # noqa: E402
            "Good: 'what's up?', '??', 'hm?'. "  # noqa: E402
            "Bad: long response, assuming what they mean."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "No context - bare hey",  # noqa: E402
        "category": "clarify",  # noqa: E402
        "context": "[11:00] Unknown: hey",  # noqa: E402
        "last_message": "hey",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "",  # noqa: E402
        "max_chars": 30,  # noqa: E402
        "rubric": (  # noqa: E402
            "Reply to 'hey' from an unknown person with zero context. "  # noqa: E402
            "Should be very brief - a simple greeting back. "  # noqa: E402
            "Good: 'hey', 'hey what's up', 'yo'. "  # noqa: E402
            "Bad: long reply, introducing yourself, asking detailed questions."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Ambiguous forwarded link",  # noqa: E402
        "category": "clarify",  # noqa: E402
        "context": "[12:00] Sarah: [Link]",  # noqa: E402
        "last_message": "[Link]",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "",  # noqa: E402
        "max_chars": 40,  # noqa: E402
        "banned": ["article", "interesting"],  # noqa: E402
        "rubric": (  # noqa: E402
            "Someone sent just a link with no text. Model should NOT confabulate "  # noqa: E402
            "what the link is about. Good: 'what's this?', '?', 'ooh what is it'. "  # noqa: E402
            "Bad: commenting on the content, assuming what it is."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Stale thread - weeks old",  # noqa: E402
        "category": "clarify",  # noqa: E402
        "context": "[3 weeks ago] Dave: hey you free Saturday?",  # noqa: E402
        "last_message": "hey you free Saturday?",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "casual",  # noqa: E402
        "max_chars": 60,  # noqa: E402
        "rubric": (  # noqa: E402
            "Replying to a 3-week-old message asking about Saturday. "  # noqa: E402
            "Should acknowledge the staleness or not pretend it's timely. "  # noqa: E402
            "Good: 'sorry just saw this', 'lol my bad, super late'. "  # noqa: E402
            "Bad: answering as if it's current ('yeah I'm free!')."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Emoji only message",  # noqa: E402
        "category": "clarify",  # noqa: E402
        "context": "[13:00] Tina: \U0001f602\U0001f602\U0001f602",  # noqa: E402
        "last_message": "\U0001f602\U0001f602\U0001f602",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "casual",  # noqa: E402
        "max_chars": 30,  # noqa: E402
        "rubric": (  # noqa: E402
            "Reply to a message that's just laughing emojis with no context. "  # noqa: E402
            "Good: 'lol', '\U0001f602', 'what', '??'. "  # noqa: E402
            "Bad: long response, asking detailed questions, pretending to know what's funny."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Voice memo reference",  # noqa: E402
        "category": "clarify",  # noqa: E402
        "context": "[16:00] Dan: [Voice Memo]",  # noqa: E402
        "last_message": "[Voice Memo]",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "",  # noqa: E402
        "max_chars": 50,  # noqa: E402
        "rubric": (  # noqa: E402
            "Reply to a voice memo that can't be transcribed. "  # noqa: E402
            "Good: 'can't listen rn, what's up?', 'send a text lol'. "  # noqa: E402
            "Bad: pretending to have heard it, long response."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
    {  # noqa: E402
        "name": "Wrong number / random text",  # noqa: E402
        "category": "clarify",  # noqa: E402
        "context": "[08:00] Unknown: Tell Maria I'll be there at 6",  # noqa: E402
        "last_message": "Tell Maria I'll be there at 6",  # noqa: E402
        "tone": "casual",  # noqa: E402
        "user_style": "",  # noqa: E402
        "max_chars": 50,  # noqa: E402
        "rubric": (  # noqa: E402
            "Reply to what looks like a wrong-number text. "  # noqa: E402
            "Good: 'wrong number', 'think you have the wrong person'. "  # noqa: E402
            "Bad: agreeing to tell Maria, long explanation."  # noqa: E402
        ),  # noqa: E402
    },  # noqa: E402
]  # noqa: E402
  # noqa: E402
  # noqa: E402
@dataclass  # noqa: E402
class EvalResult:  # noqa: E402
    name: str  # noqa: E402
    category: str  # noqa: E402
    output: str  # noqa: E402
    latency_ms: float  # noqa: E402
    checks_passed: list[str]  # noqa: E402
    checks_failed: list[str]  # noqa: E402
    passed: bool  # noqa: E402
    judge_score: float | None = None  # noqa: E402
    judge_reasoning: str = ""  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# LLM Judge  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def get_judge_client():  # noqa: E402
    """Create OpenAI-compatible client for the judge model."""  # noqa: E402
    return _get_judge_client()  # noqa: E402
  # noqa: E402
  # noqa: E402
def judge_response(client, tc: dict, output: str) -> tuple[float, str]:  # noqa: E402
    """Score a response using the LLM judge.  # noqa: E402
  # noqa: E402
    Returns (score 0-10, reasoning).  # noqa: E402
    """  # noqa: E402
    rubric = tc.get("rubric", "")  # noqa: E402
    if not rubric:  # noqa: E402
        return -1.0, "no rubric"  # noqa: E402
  # noqa: E402
    prompt = (  # noqa: E402
        "You are an expert evaluator for a text message reply generator.\n\n"  # noqa: E402
        f"CONVERSATION:\n{tc['context']}\n\n"  # noqa: E402
        f"LAST MESSAGE (to reply to):\n{tc['last_message']}\n\n"  # noqa: E402
        f"GENERATED REPLY:\n{output}\n\n"  # noqa: E402
        f"RUBRIC:\n{rubric}\n\n"  # noqa: E402
        "Score the generated reply from 0-10 based on the rubric.\n"  # noqa: E402
        "Respond in this exact JSON format:\n"  # noqa: E402
        '{"score": <0-10>, "reasoning": "<1-2 sentences>"}'  # noqa: E402
    )  # noqa: E402
  # noqa: E402
    try:  # noqa: E402
        resp = client.chat.completions.create(  # noqa: E402
            model=JUDGE_MODEL,  # noqa: E402
            messages=[{"role": "user", "content": prompt}],  # noqa: E402
            temperature=0.0,  # noqa: E402
            max_tokens=150,  # noqa: E402
        )  # noqa: E402
        text = resp.choices[0].message.content.strip()  # noqa: E402
        # Parse JSON from response (handle markdown fences)  # noqa: E402
        if text.startswith("```"):  # noqa: E402
            text = text.split("```")[1]  # noqa: E402
            if text.startswith("json"):  # noqa: E402
                text = text[4:]  # noqa: E402
        data = json.loads(text)  # noqa: E402
        return float(data["score"]), data.get("reasoning", "")  # noqa: E402
    except Exception as e:  # noqa: E402
        return -1.0, f"judge error: {e}"  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Local Checks  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def build_prompt(tc: dict) -> str:  # noqa: E402
    """Build XML drafter prompt (default strategy)."""  # noqa: E402
    system = (  # noqa: E402
        "You draft text message replies matching the sender's exact style.\n"  # noqa: E402
        "Rules:\n"  # noqa: E402
        "- Match their texting style exactly "  # noqa: E402
        "(length, formality, abbreviations, emoji, punctuation)\n"  # noqa: E402
        "- Sound natural, never like an AI\n"  # noqa: E402
        '- No phrases like "I hope this helps" or "Let me know"\n'  # noqa: E402
        "- No formal greetings unless they use them\n"  # noqa: E402
        "- If the message is unclear or you lack context to reply properly, "  # noqa: E402
        'respond with just "?"'  # noqa: E402
    )  # noqa: E402
    style = (  # noqa: E402
        f"Tone: {tc['tone']}. Style: {tc['user_style']}"  # noqa: E402
        if tc["user_style"]  # noqa: E402
        else f"Tone: {tc['tone']}"  # noqa: E402
    )  # noqa: E402
    return (  # noqa: E402
        f"<system>\n{system}</system>\n\n"  # noqa: E402
        f"<style>\n{style}\n</style>\n\n"  # noqa: E402
        f"<conversation>\n{tc['context']}\n</conversation>\n\n"  # noqa: E402
        f"<last_message>{tc['last_message']}</last_message>\n\n"  # noqa: E402
        f"<reply>"  # noqa: E402
    )  # noqa: E402
  # noqa: E402
  # noqa: E402
def check_result(tc: dict, output: str) -> tuple[list[str], list[str]]:  # noqa: E402
    """Run local assertions. Returns (passed, failed)."""  # noqa: E402
    passed = []  # noqa: E402
    failed = []  # noqa: E402
    lower = output.lower()  # noqa: E402
  # noqa: E402
    # Anti-AI phrases (global)  # noqa: E402
    for phrase in ANTI_AI_PHRASES:  # noqa: E402
        if phrase in lower:  # noqa: E402
            failed.append(f"contains anti-AI phrase: '{phrase}'")  # noqa: E402
        else:  # noqa: E402
            passed.append(f"no '{phrase}'")  # noqa: E402
  # noqa: E402
    # Max words  # noqa: E402
    if "max_words" in tc:  # noqa: E402
        word_count = len(output.split())  # noqa: E402
        if word_count <= tc["max_words"]:  # noqa: E402
            passed.append(f"words={word_count} <= {tc['max_words']}")  # noqa: E402
        else:  # noqa: E402
            failed.append(f"words={word_count} > {tc['max_words']}")  # noqa: E402
  # noqa: E402
    # Max chars  # noqa: E402
    if "max_chars" in tc:  # noqa: E402
        if len(output) <= tc["max_chars"]:  # noqa: E402
            passed.append(f"chars={len(output)} <= {tc['max_chars']}")  # noqa: E402
        else:  # noqa: E402
            failed.append(f"chars={len(output)} > {tc['max_chars']}")  # noqa: E402
  # noqa: E402
    # Banned words  # noqa: E402
    for word in tc.get("banned", []):  # noqa: E402
        if word.lower() in lower:  # noqa: E402
            failed.append(f"contains banned: '{word}'")  # noqa: E402
        else:  # noqa: E402
            passed.append(f"no '{word}'")  # noqa: E402
  # noqa: E402
    # Basic sanity  # noqa: E402
    if not output.strip():  # noqa: E402
        failed.append("empty output")  # noqa: E402
    else:  # noqa: E402
        passed.append("non-empty output")  # noqa: E402
  # noqa: E402
    if len(output) > 300:  # noqa: E402
        failed.append(f"way too long ({len(output)} chars)")  # noqa: E402
    else:  # noqa: E402
        passed.append(f"reasonable length ({len(output)} chars)")  # noqa: E402
  # noqa: E402
    return passed, failed  # noqa: E402
  # noqa: E402
  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
# Main  # noqa: E402
# ---------------------------------------------------------------------------  # noqa: E402
  # noqa: E402
  # noqa: E402
def main() -> int:  # noqa: E402
    import argparse  # noqa: E402
  # noqa: E402
    parser = argparse.ArgumentParser(description="JARVIS Batch Eval")  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--judge", action="store_true", help="Enable LLM judge scoring via Cerebras"  # noqa: E402
    )  # noqa: E402
    parser.add_argument(  # noqa: E402
        "--optimized",  # noqa: E402
        action="store_true",  # noqa: E402
        help="Use DSPy-compiled program instead of raw generation",  # noqa: E402
    )  # noqa: E402
    args = parser.parse_args()  # noqa: E402
  # noqa: E402
    # Setup logging  # noqa: E402
    log_path = PROJECT_ROOT / "results" / "batch_eval.log"  # noqa: E402
    log_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
    logging.basicConfig(  # noqa: E402
        level=logging.INFO,  # noqa: E402
        format="%(asctime)s - %(levelname)s - %(message)s",  # noqa: E402
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],  # noqa: E402
    )  # noqa: E402
    logging.getLogger(__name__)  # noqa: E402
  # noqa: E402
    strategy = "dspy_optimized" if args.optimized else "xml_drafter"  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print("JARVIS BATCH EVAL - Response Generation", flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print(f"Test cases:  {len(TEST_CASES)}", flush=True)  # noqa: E402
    print(f"Strategy:    {strategy}", flush=True)  # noqa: E402
    judge_label = f"{JUDGE_MODEL} via DeepInfra" if args.judge else "disabled (use --judge)"  # noqa: E402
    print(f"LLM judge:   {judge_label}", flush=True)  # noqa: E402
    print(flush=True)  # noqa: E402
  # noqa: E402
    # Init judge  # noqa: E402
    judge_client = None  # noqa: E402
    if args.judge:  # noqa: E402
        judge_client = get_judge_client()  # noqa: E402
        if judge_client is None:  # noqa: E402
            print("WARNING: CEREBRAS_API_KEY not set in .env - skipping judge", flush=True)  # noqa: E402
            print("         Put your key in .env and re-run with --judge", flush=True)  # noqa: E402
        else:  # noqa: E402
            print(f"Judge ready: {JUDGE_MODEL} via Cerebras", flush=True)  # noqa: E402
    print(flush=True)  # noqa: E402
  # noqa: E402
    # Load model / compiled program  # noqa: E402
    dspy_program = None  # noqa: E402
    loader = None  # noqa: E402
  # noqa: E402
    # Set MLX memory limits early to prevent swap thrashing on 8GB systems.  # noqa: E402
    # loader.load() also sets these, but we set them before any MLX import  # noqa: E402
    # to guard against accidental early allocation.  # noqa: E402
    from models.memory_config import apply_embedder_limits  # noqa: E402
  # noqa: E402
    apply_embedder_limits()  # noqa: E402
  # noqa: E402
    if args.optimized:  # noqa: E402
        optimized_dir = PROJECT_ROOT / "evals" / "optimized_reply"  # noqa: E402
        if not optimized_dir.exists():  # noqa: E402
            print("ERROR: No compiled program found at evals/optimized_reply/", flush=True)  # noqa: E402
            print("       Run: uv run python evals/dspy_optimize.py", flush=True)  # noqa: E402
            return 1  # noqa: E402
        print("Loading DSPy compiled program...", flush=True)  # noqa: E402
        load_start = time.perf_counter()  # noqa: E402
        try:  # noqa: E402
            import dspy  # noqa: E402
            from evals.dspy_client import DSPYMLXClient  # noqa: E402
            from evals.dspy_reply import ReplyModule  # noqa: E402
  # noqa: E402
            student_lm = DSPYMLXClient(max_tokens=50, temperature=0.1)  # noqa: E402
            dspy.configure(lm=student_lm)  # noqa: E402
            dspy_program = ReplyModule()  # noqa: E402
            dspy_program.load(str(optimized_dir))  # noqa: E402
            load_ms = (time.perf_counter() - load_start) * 1000  # noqa: E402
            print(f"Compiled program loaded in {load_ms:.0f}ms", flush=True)  # noqa: E402
        except Exception as e:  # noqa: E402
            print(f"FATAL: Failed to load compiled program: {e}", flush=True)  # noqa: E402
            return 1  # noqa: E402
    else:  # noqa: E402
        print("Loading MLX model...", flush=True)  # noqa: E402
        load_start = time.perf_counter()  # noqa: E402
        try:  # noqa: E402
            from models.loader import get_model  # noqa: E402
  # noqa: E402
            loader = get_model()  # noqa: E402
            if not loader.is_loaded():  # noqa: E402
                loader.load()  # noqa: E402
            load_ms = (time.perf_counter() - load_start) * 1000  # noqa: E402
            print(f"Model loaded in {load_ms:.0f}ms", flush=True)  # noqa: E402
        except Exception as e:  # noqa: E402
            print(f"FATAL: Failed to load model: {e}", flush=True)  # noqa: E402
            return 1  # noqa: E402
  # noqa: E402
    print(flush=True)  # noqa: E402
    print("-" * 70, flush=True)  # noqa: E402
  # noqa: E402
    results: list[EvalResult] = []  # noqa: E402
    total_start = time.perf_counter()  # noqa: E402
  # noqa: E402
    # Resume support: load partial results from checkpoint file  # noqa: E402
    checkpoint_path = PROJECT_ROOT / "results" / "batch_eval_checkpoint.jsonl"  # noqa: E402
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
    completed_names: set[str] = set()  # noqa: E402
    if checkpoint_path.exists():  # noqa: E402
        for line in checkpoint_path.read_text().splitlines():  # noqa: E402
            if line.strip():  # noqa: E402
                rec = json.loads(line)  # noqa: E402
                completed_names.add(rec["name"])  # noqa: E402
                results.append(  # noqa: E402
                    EvalResult(  # noqa: E402
                        name=rec["name"],  # noqa: E402
                        category=rec["category"],  # noqa: E402
                        output=rec["output"],  # noqa: E402
                        latency_ms=rec["latency_ms"],  # noqa: E402
                        checks_passed=rec.get("checks_passed", []),  # noqa: E402
                        checks_failed=rec.get("checks_failed", []),  # noqa: E402
                        passed=rec["local_passed"],  # noqa: E402
                        judge_score=rec.get("judge_score"),  # noqa: E402
                        judge_reasoning=rec.get("judge_reasoning", ""),  # noqa: E402
                    )  # noqa: E402
                )  # noqa: E402
        if completed_names:  # noqa: E402
            print(  # noqa: E402
                f"Resuming: {len(completed_names)}/{len(TEST_CASES)} already completed",  # noqa: E402
                flush=True,  # noqa: E402
            )  # noqa: E402
  # noqa: E402
    checkpoint_f = checkpoint_path.open("a", encoding="utf-8")  # noqa: E402
  # noqa: E402
    for i, tc in enumerate(tqdm(TEST_CASES, desc="Evaluating"), 1):  # noqa: E402
        if tc["name"] in completed_names:  # noqa: E402
            continue  # noqa: E402
        # Generate via DSPy compiled program or raw model  # noqa: E402
        gen_start = time.perf_counter()  # noqa: E402
        try:  # noqa: E402
            if dspy_program is not None:  # noqa: E402
                pred = dspy_program(  # noqa: E402
                    context=tc["context"],  # noqa: E402
                    last_message=tc["last_message"],  # noqa: E402
                    tone=tc["tone"],  # noqa: E402
                    user_style=tc.get("user_style", ""),  # noqa: E402
                )  # noqa: E402
                output = pred.reply.strip()  # noqa: E402
            else:  # noqa: E402
                prompt = build_prompt(tc)  # noqa: E402
                result = loader.generate_sync(  # noqa: E402
                    prompt=prompt,  # noqa: E402
                    temperature=0.1,  # noqa: E402
                    max_tokens=50,  # noqa: E402
                    top_p=0.1,  # noqa: E402
                    top_k=50,  # noqa: E402
                    repetition_penalty=1.05,  # noqa: E402
                )  # noqa: E402
                output = result.text.strip()  # noqa: E402
            latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E402
        except Exception as e:  # noqa: E402
            output = f"[ERROR: {e}]"  # noqa: E402
            latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E402
  # noqa: E402
        # Local checks  # noqa: E402
        passed_checks, failed_checks = check_result(tc, output)  # noqa: E402
        all_passed = len(failed_checks) == 0  # noqa: E402
  # noqa: E402
        # Judge scoring  # noqa: E402
        judge_score = None  # noqa: E402
        judge_reasoning = ""  # noqa: E402
        if judge_client and tc.get("rubric"):  # noqa: E402
            judge_score, judge_reasoning = judge_response(judge_client, tc, output)  # noqa: E402
  # noqa: E402
        er = EvalResult(  # noqa: E402
            name=tc["name"],  # noqa: E402
            category=tc.get("category", "unknown"),  # noqa: E402
            output=output,  # noqa: E402
            latency_ms=latency_ms,  # noqa: E402
            checks_passed=passed_checks,  # noqa: E402
            checks_failed=failed_checks,  # noqa: E402
            passed=all_passed,  # noqa: E402
            judge_score=judge_score,  # noqa: E402
            judge_reasoning=judge_reasoning,  # noqa: E402
        )  # noqa: E402
        results.append(er)  # noqa: E402
  # noqa: E402
        # Write checkpoint incrementally (survives crash)  # noqa: E402
        checkpoint_f.write(  # noqa: E402
            json.dumps(  # noqa: E402
                {  # noqa: E402
                    "name": er.name,  # noqa: E402
                    "category": er.category,  # noqa: E402
                    "output": er.output,  # noqa: E402
                    "latency_ms": round(er.latency_ms, 1),  # noqa: E402
                    "local_passed": er.passed,  # noqa: E402
                    "checks_passed": er.checks_passed,  # noqa: E402
                    "checks_failed": er.checks_failed,  # noqa: E402
                    "judge_score": er.judge_score,  # noqa: E402
                    "judge_reasoning": er.judge_reasoning,  # noqa: E402
                }  # noqa: E402
            )  # noqa: E402
            + "\n"  # noqa: E402
        )  # noqa: E402
        checkpoint_f.flush()  # noqa: E402
  # noqa: E402
        # Print per-case  # noqa: E402
        status = "PASS" if all_passed else "FAIL"  # noqa: E402
        cat = tc.get("category", "?")  # noqa: E402
        print(f"\n[{i:2d}/{len(TEST_CASES)}] [{cat}] {tc['name']}", flush=True)  # noqa: E402
        print(f'  Output:  "{output}"', flush=True)  # noqa: E402
        judge_str = ""  # noqa: E402
        if judge_score is not None and judge_score >= 0:  # noqa: E402
            judge_str = f" | Judge: {judge_score:.0f}/10"  # noqa: E402
        print(f"  Latency: {latency_ms:.0f}ms | Local: {status}{judge_str}", flush=True)  # noqa: E402
        if failed_checks:  # noqa: E402
            for f in failed_checks:  # noqa: E402
                print(f"  FAIL: {f}", flush=True)  # noqa: E402
        if judge_reasoning:  # noqa: E402
            print(f"  Judge: {judge_reasoning}", flush=True)  # noqa: E402
  # noqa: E402
    total_ms = (time.perf_counter() - total_start) * 1000  # noqa: E402
  # noqa: E402
    # Summary  # noqa: E402
    print(flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
    print("SUMMARY", flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
  # noqa: E402
    n_passed = sum(1 for r in results if r.passed)  # noqa: E402
    n_failed = len(results) - n_passed  # noqa: E402
    latencies = [r.latency_ms for r in results]  # noqa: E402
    avg_latency = sum(latencies) / len(latencies) if latencies else 0  # noqa: E402
    sorted_lat = sorted(latencies)  # noqa: E402
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0  # noqa: E402
    p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)  # noqa: E402
    p95 = sorted_lat[p95_idx] if sorted_lat else 0  # noqa: E402
  # noqa: E402
    print(  # noqa: E402
        f"Local pass:   {n_passed}/{len(results)} ({n_passed / len(results) * 100:.0f}%)",  # noqa: E402
        flush=True,  # noqa: E402
    )  # noqa: E402
    print(f"Failed:       {n_failed}", flush=True)  # noqa: E402
    print(f"Total time:   {total_ms:.0f}ms", flush=True)  # noqa: E402
    print(f"Avg latency:  {avg_latency:.0f}ms", flush=True)  # noqa: E402
    print(f"P50 latency:  {p50:.0f}ms", flush=True)  # noqa: E402
    print(f"P95 latency:  {p95:.0f}ms", flush=True)  # noqa: E402
  # noqa: E402
    # Judge summary  # noqa: E402
    scored = [r for r in results if r.judge_score is not None and r.judge_score >= 0]  # noqa: E402
    if scored:  # noqa: E402
        scores = [r.judge_score for r in scored]  # noqa: E402
        avg_score = sum(scores) / len(scores)  # noqa: E402
        min_score = min(scores)  # noqa: E402
        max_score = max(scores)  # noqa: E402
        pass_7 = sum(1 for s in scores if s >= 7)  # noqa: E402
        print(flush=True)  # noqa: E402
        print(  # noqa: E402
            f"Judge scores: avg={avg_score:.1f}/10  min={min_score:.0f}  max={max_score:.0f}",  # noqa: E402
            flush=True,  # noqa: E402
        )  # noqa: E402
        print(  # noqa: E402
            f"Judge pass (>=7): {pass_7}/{len(scored)} ({pass_7 / len(scored) * 100:.0f}%)",  # noqa: E402
            flush=True,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    if n_failed:  # noqa: E402
        print("\nLocal failures:", flush=True)  # noqa: E402
        for r in results:  # noqa: E402
            if not r.passed:  # noqa: E402
                print(f"  - {r.name}: {', '.join(r.checks_failed)}", flush=True)  # noqa: E402
  # noqa: E402
    # Per-category breakdown  # noqa: E402
    print(flush=True)  # noqa: E402
    print("PER-CATEGORY BREAKDOWN", flush=True)  # noqa: E402
    print("-" * 70, flush=True)  # noqa: E402
    for cat in CATEGORIES:  # noqa: E402
        cat_results = [r for r in results if r.category == cat]  # noqa: E402
        if not cat_results:  # noqa: E402
            continue  # noqa: E402
        cat_passed = sum(1 for r in cat_results if r.passed)  # noqa: E402
        cat_scored = [r for r in cat_results if r.judge_score is not None and r.judge_score >= 0]  # noqa: E402
        cat_judge = ""  # noqa: E402
        if cat_scored:  # noqa: E402
            cat_avg = sum(r.judge_score for r in cat_scored) / len(cat_scored)  # noqa: E402
            cat_judge = f"  judge_avg={cat_avg:.1f}/10"  # noqa: E402
        print(  # noqa: E402
            f"  {cat:20s}  local={cat_passed}/{len(cat_results)}"  # noqa: E402
            f" ({cat_passed / len(cat_results) * 100:.0f}%){cat_judge}",  # noqa: E402
            flush=True,  # noqa: E402
        )  # noqa: E402
  # noqa: E402
    if scored:  # noqa: E402
        low = [r for r in scored if r.judge_score < 7]  # noqa: E402
        if low:  # noqa: E402
            print("\nLow judge scores (<7):", flush=True)  # noqa: E402
            for r in low:  # noqa: E402
                print(f"  - {r.name}: {r.judge_score:.0f}/10 - {r.judge_reasoning}", flush=True)  # noqa: E402
  # noqa: E402
    # Save results  # noqa: E402
    output_path = PROJECT_ROOT / "results" / "batch_eval_latest.json"  # noqa: E402
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E402
    output_data = {  # noqa: E402
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E402
        "strategy": strategy,  # noqa: E402
        "judge_model": JUDGE_MODEL if scored else None,  # noqa: E402
        "total": len(results),  # noqa: E402
        "local_passed": n_passed,  # noqa: E402
        "local_failed": n_failed,  # noqa: E402
        "local_pass_rate": round(n_passed / len(results), 4),  # noqa: E402
        "judge_avg_score": round(avg_score, 2) if scored else None,  # noqa: E402
        "judge_pass_rate": round(pass_7 / len(scored), 4) if scored else None,  # noqa: E402
        "latency": {  # noqa: E402
            "avg_ms": round(avg_latency, 1),  # noqa: E402
            "p50_ms": round(p50, 1),  # noqa: E402
            "p95_ms": round(p95, 1),  # noqa: E402
            "total_ms": round(total_ms, 1),  # noqa: E402
        },  # noqa: E402
        "results": [  # noqa: E402
            {  # noqa: E402
                "name": r.name,  # noqa: E402
                "category": r.category,  # noqa: E402
                "output": r.output,  # noqa: E402
                "latency_ms": round(r.latency_ms, 1),  # noqa: E402
                "local_passed": r.passed,  # noqa: E402
                "failed_checks": r.checks_failed,  # noqa: E402
                "judge_score": r.judge_score,  # noqa: E402
                "judge_reasoning": r.judge_reasoning,  # noqa: E402
            }  # noqa: E402
            for r in results  # noqa: E402
        ],  # noqa: E402
    }  # noqa: E402
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E402
    # Clean up checkpoint file after successful completion  # noqa: E402
    checkpoint_f.close()  # noqa: E402
    checkpoint_path.unlink(missing_ok=True)  # noqa: E402
    print(f"\nResults saved to: {output_path}", flush=True)  # noqa: E402
    print("=" * 70, flush=True)  # noqa: E402
  # noqa: E402
    return 0 if n_failed == 0 else 1  # noqa: E402
  # noqa: E402
  # noqa: E402
if __name__ == "__main__":  # noqa: E402
    sys.exit(main())  # noqa: E402
