#!/usr/bin/env python3  # noqa: E501
"""Batch evaluation: generate responses and judge quality with LLM.  # noqa: E501
  # noqa: E501
Runs the local MLX model against test cases from promptfoo.yaml,  # noqa: E501
checks local assertions (length, anti-AI phrases), then scores each  # noqa: E501
response with Gemini 2.5 Flash via DeepInfra as an LLM judge.  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    uv run python evals/batch_eval.py              # local checks only  # noqa: E501
    uv run python evals/batch_eval.py --judge       # + LLM judge scoring  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import os  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501

# noqa: E501
from tqdm import tqdm  # noqa: E402  # noqa: E501

  # noqa: E501
PROJECT_ROOT = Path(__file__).parent.parent  # noqa: E501
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E501
  # noqa: E501
# Load .env  # noqa: E501
_env_path = PROJECT_ROOT / ".env"  # noqa: E501
if _env_path.exists():  # noqa: E501
    for line in _env_path.read_text().splitlines():  # noqa: E501
        line = line.strip()  # noqa: E501
        if line and not line.startswith("#") and "=" in line:  # noqa: E501
            key, _, val = line.partition("=")  # noqa: E501
            os.environ.setdefault(key.strip(), val.strip())  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Constants  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
from evals.judge_config import JUDGE_MODEL  # noqa: E402  # noqa: E501
from evals.judge_config import get_judge_client as _get_judge_client  # noqa: E402  # noqa: E501

  # noqa: E501
ANTI_AI_PHRASES = [  # noqa: E501
    "i'd be happy to",  # noqa: E501
    "i hope this helps",  # noqa: E501
    "let me know if",  # noqa: E501
    "i understand",  # noqa: E501
    "as an ai",  # noqa: E501
    "i'm an ai",  # noqa: E501
    "certainly!",  # noqa: E501
    "of course!",  # noqa: E501
    "great question",  # noqa: E501
]  # noqa: E501
  # noqa: E501
# Category definitions for per-category MIPRO v2 optimization  # noqa: E501
CATEGORIES = [  # noqa: E501
    "brief",  # noqa: E501
    "warm",  # noqa: E501
    "social",  # noqa: E501
    "clarify",  # noqa: E501
]  # noqa: E501
  # noqa: E501
# Test cases with category labels for per-category optimization  # noqa: E501
TEST_CASES = [  # noqa: E501
    # =========================================================================  # noqa: E501
    # brief: Short transactional replies (12 cases: 7 casual + 5 professional)  # noqa: E501
    # =========================================================================  # noqa: E501
    {  # noqa: E501
        "name": "Lunch invitation",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[14:00] John: Want to grab lunch tomorrow?",  # noqa: E501
        "last_message": "Want to grab lunch tomorrow?",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "brief, friendly",  # noqa: E501
        "max_words": 15,  # noqa: E501
        "max_chars": 80,  # noqa: E501
        "banned": ["sounds great", "absolutely"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Is this a natural, casual text reply to a lunch invitation? "  # noqa: E501
            "Should be brief (<15 words), friendly, and sound human (not AI). "  # noqa: E501
            "Pass if it sounds like a real person texting."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Running late",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[09:15] Alex: Running 10 min late",  # noqa: E501
        "last_message": "Running 10 min late",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "supportive, brief",  # noqa: E501
        "max_words": 12,  # noqa: E501
        "rubric": (  # noqa: E501
            "Is this a supportive, brief reply to someone running late? "  # noqa: E501
            "Good: 'no worries', 'all good', 'take your time'. "  # noqa: E501
            "Bad: asking why, being passive aggressive, too formal."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Simple yes/no - trash",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[15:00] Dad: Did you take out the trash?",  # noqa: E501
        "last_message": "Did you take out the trash?",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "direct",  # noqa: E501
        "max_words": 8,  # noqa: E501
        "rubric": (  # noqa: E501
            "Is this a direct answer to 'did you take out trash?' "  # noqa: E501
            "Should be very brief - ideally just 'yes/yep/yeah' or 'no/not yet'. "  # noqa: E501
            "Fail if it's more than one sentence."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Group chat confirmation",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": (  # noqa: E501
            "[Group: Game Night]\n"  # noqa: E501
            "[14:00] Jake: 7pm Saturday work?\n"  # noqa: E501
            "[14:05] Lisa: I'm in!\n"  # noqa: E501
            "[14:10] Tom: Works for me"  # noqa: E501
        ),  # noqa: E501
        "last_message": "Works for me",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "brief group energy",  # noqa: E501
        "max_words": 6,  # noqa: E501
        "rubric": (  # noqa: E501
            "Is this a brief group chat confirmation? Should be 1-5 words max. "  # noqa: E501
            "Good: 'same', 'count me in', 'down'. "  # noqa: E501
            "Bad: full sentences, formal responses."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Quick favor - pickup",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[17:30] Mom: Can you pick up milk on your way home?",  # noqa: E501
        "last_message": "Can you pick up milk on your way home?",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "brief, agreeable",  # noqa: E501
        "max_words": 10,  # noqa: E501
        "rubric": (  # noqa: E501
            "Is this a quick confirmation to a simple favor request? "  # noqa: E501
            "Good: 'sure thing', 'yep', 'on it'. "  # noqa: E501
            "Bad: asking for details, long response, overly enthusiastic."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "ETA check",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[19:00] Jake: you close?",  # noqa: E501
        "last_message": "you close?",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "brief",  # noqa: E501
        "max_words": 8,  # noqa: E501
        "max_chars": 40,  # noqa: E501
        "rubric": (  # noqa: E501
            "Quick reply to 'you close?' asking about arrival. "  # noqa: E501
            "Good: 'yeah 5 min', 'almost there', 'pulling up'. "  # noqa: E501
            "Bad: long explanation, formal response."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Confirmation - address",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[12:00] Sarah: 123 Main St right?",  # noqa: E501
        "last_message": "123 Main St right?",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "direct",  # noqa: E501
        "max_words": 6,  # noqa: E501
        "rubric": (  # noqa: E501
            "Confirm or correct an address. "  # noqa: E501
            "Good: 'yep that's it', 'yeah', 'no it's 125'. "  # noqa: E501
            "Bad: long explanation, repeating the full address formally."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    # =========================================================================  # noqa: E501
    # warm: Emotional weight - comfort or celebrate (5 cases)  # noqa: E501
    # =========================================================================  # noqa: E501
    {  # noqa: E501
        "name": "Emotional support - venting",  # noqa: E501
        "category": "warm",  # noqa: E501
        "context": (  # noqa: E501
            "[20:00] Mike: Work was brutal today\n"  # noqa: E501
            "[20:01] Mike: Boss dumped a project on me last minute"  # noqa: E501
        ),  # noqa: E501
        "last_message": "Boss dumped a project on me last minute",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "empathetic friend",  # noqa: E501
        "banned": ["have you tried", "you should"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Is this empathetic without giving unsolicited advice? "  # noqa: E501
            "Good: 'that sucks', 'ugh sorry'. "  # noqa: E501
            "Bad: 'have you tried...', 'you should...', therapist-speak."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Emotional support - breakup",  # noqa: E501
        "category": "warm",  # noqa: E501
        "context": (  # noqa: E501
            "[22:00] Sarah: Mark and I broke up\n[22:01] Sarah: I don't even know what happened"  # noqa: E501
        ),  # noqa: E501
        "last_message": "I don't even know what happened",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "warm, supportive",  # noqa: E501
        "banned": ["you'll find someone", "plenty of fish", "you should"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Is this supportive without minimizing or giving cliched advice? "  # noqa: E501
            "Good: 'I'm so sorry', 'that's rough, I'm here for you'. "  # noqa: E501
            "Bad: 'you'll find someone better', platitudes, therapist-speak."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Emotional support - bad news",  # noqa: E501
        "category": "warm",  # noqa: E501
        "context": "[15:00] John: Didn't get the job. Thought the interview went well",  # noqa: E501
        "last_message": "Didn't get the job. Thought the interview went well",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "empathetic, brief",  # noqa: E501
        "banned": ["everything happens for a reason", "you should"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Empathetic response to job rejection. "  # noqa: E501
            "Good: 'damn that sucks', 'their loss honestly'. "  # noqa: E501
            "Bad: toxic positivity, unsolicited advice, long pep talk."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Emotional support - health worry",  # noqa: E501
        "category": "warm",  # noqa: E501
        "context": (  # noqa: E501
            "[11:00] Mom: Doctor wants to run more tests\n[11:02] Mom: Trying not to worry"  # noqa: E501
        ),  # noqa: E501
        "last_message": "Trying not to worry",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "caring, reassuring",  # noqa: E501
        "banned": ["i'm sure it's nothing", "don't worry"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Supportive response to a parent's health worry. "  # noqa: E501
            "Good: 'I'm here for you', 'let me know what they say'. "  # noqa: E501
            "Bad: dismissing worry ('it's probably nothing'), medical advice."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Emotional support - stressed",  # noqa: E501
        "category": "warm",  # noqa: E501
        "context": "[23:00] Alex: Can't sleep. Too much on my mind",  # noqa: E501
        "last_message": "Can't sleep. Too much on my mind",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "gentle, supportive",  # noqa: E501
        "banned": ["have you tried", "you should try"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Late night support for a stressed friend. "  # noqa: E501
            "Good: 'wanna talk about it?', 'I'm up if you need to vent'. "  # noqa: E501
            "Bad: sleep advice, telling them to relax, dismissive."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    # =========================================================================  # noqa: E501
    # brief (professional tone): Formal tone handled by detect_tone() (5 cases)  # noqa: E501
    # =========================================================================  # noqa: E501
    {  # noqa: E501
        "name": "Professional - report request",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[09:00] Manager: Can you send the Q4 report by EOD?",  # noqa: E501
        "last_message": "Can you send the Q4 report by EOD?",  # noqa: E501
        "tone": "professional",  # noqa: E501
        "user_style": "professional but not stiff",  # noqa: E501
        "banned": ["lol", "gonna"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Is this professional but not stiff? Should confirm the task briefly. "  # noqa: E501
            "Good: 'Will do', 'On it', 'I'll have it ready'. "  # noqa: E501
            "Bad: too casual, too formal/corporate."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Professional - meeting reschedule",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[10:00] Client: Need to push our 2pm to Thursday. Does that work?",  # noqa: E501
        "last_message": "Need to push our 2pm to Thursday. Does that work?",  # noqa: E501
        "tone": "professional",  # noqa: E501
        "user_style": "polite, concise",  # noqa: E501
        "banned": ["lol", "haha", "gonna"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Professional response to a meeting reschedule. "  # noqa: E501
            "Good: 'Thursday works for me', 'Sure, same time?'. "  # noqa: E501
            "Bad: too casual, overly formal, long response."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Professional - project update",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": (  # noqa: E501
            "[14:00] Manager: How's the migration project coming along?\n"  # noqa: E501
            "[14:01] Manager: Board wants an update Friday"  # noqa: E501
        ),  # noqa: E501
        "last_message": "Board wants an update Friday",  # noqa: E501
        "tone": "professional",  # noqa: E501
        "user_style": "clear, status-oriented",  # noqa: E501
        "banned": ["lol", "dude"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Professional status update response. "  # noqa: E501
            "Good: 'On track. I'll prep a summary for Friday.', 'Will have slides ready'. "  # noqa: E501
            "Bad: vague, too casual, overly long."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Professional - thank you",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[16:00] Colleague: Thanks for covering the call today, really helped",  # noqa: E501
        "last_message": "Thanks for covering the call today, really helped",  # noqa: E501
        "tone": "professional",  # noqa: E501
        "user_style": "warm professional",  # noqa: E501
        "banned": ["lol", "np bro"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Acknowledge thanks from a colleague professionally but warmly. "  # noqa: E501
            "Good: 'Happy to help', 'Of course, anytime'. "  # noqa: E501
            "Bad: too casual, dismissive, overly formal."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Professional - deadline question",  # noqa: E501
        "category": "brief",  # noqa: E501
        "context": "[11:30] HR: When can you have the compliance training done?",  # noqa: E501
        "last_message": "When can you have the compliance training done?",  # noqa: E501
        "tone": "professional",  # noqa: E501
        "user_style": "direct, professional",  # noqa: E501
        "banned": ["lol", "idk"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Professional response to a deadline question. "  # noqa: E501
            "Good: 'I'll have it done by end of week', 'Can finish by Wednesday'. "  # noqa: E501
            "Bad: vague, too casual, no commitment."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    # =========================================================================  # noqa: E501
    # social: Casual conversational (6 cases)  # noqa: E501
    # =========================================================================  # noqa: E501
    {  # noqa: E501
        "name": "Photo reaction",  # noqa: E501
        "category": "social",  # noqa: E501
        "context": "[16:00] Emma: [Photo]\n[16:00] Emma: Look at this view!",  # noqa: E501
        "last_message": "Look at this view!",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "enthusiastic",  # noqa: E501
        "banned": ["i can see", "the photo"],  # noqa: E501
        "rubric": (  # noqa: E501
            "React to a friend sharing a photo of a nice view. "  # noqa: E501
            "Should be positive and match enthusiasm. "  # noqa: E501
            "Bad: describing the photo, generic 'nice', overly formal."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Weekend plans",  # noqa: E501
        "category": "social",  # noqa: E501
        "context": "[18:30] Sam: Any plans this weekend?",  # noqa: E501
        "last_message": "Any plans this weekend?",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "conversational",  # noqa: E501
        "max_chars": 120,  # noqa: E501
        "rubric": (  # noqa: E501
            "Is this a natural response to 'any plans this weekend?' "  # noqa: E501
            "Should share plans or ask back. "  # noqa: E501
            "Good: 'Not yet, you?', 'Might grab brunch, wbu?'. "  # noqa: E501
            "Bad: formal, overly helpful."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Inside joke / unknown reference",  # noqa: E501
        "category": "social",  # noqa: E501
        "context": "[14:00] Tom: lmao remember the thing",  # noqa: E501
        "last_message": "lmao remember the thing",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "casual bro",  # noqa: E501
        "max_chars": 40,  # noqa: E501
        "rubric": (  # noqa: E501
            "Reply to an inside joke reference ('the thing') that the model can't "  # noqa: E501
            "possibly know. Should NOT pretend to know. "  # noqa: E501
            "Good: 'lol which thing', 'haha yes', 'omg yes'. "  # noqa: E501
            "Bad: making up a specific memory, detailed response about 'the thing'."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Long time no talk",  # noqa: E501
        "category": "social",  # noqa: E501
        "context": "[19:00] College Friend: Dude it's been forever! How are you??",  # noqa: E501
        "last_message": "Dude it's been forever! How are you??",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "warm, conversational",  # noqa: E501
        "max_chars": 120,  # noqa: E501
        "rubric": (  # noqa: E501
            "Warm response to a friend reaching out after a long time. "  # noqa: E501
            "Good: 'I know right! I'm good, how about you?'. "  # noqa: E501
            "Bad: formal, distant, overly detailed life update."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Travel flex",  # noqa: E501
        "category": "social",  # noqa: E501
        "context": "[10:00] Lisa: Just landed in Tokyo!!",  # noqa: E501
        "last_message": "Just landed in Tokyo!!",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "excited, enthusiastic",  # noqa: E501
        "banned": ["i hope you", "have a wonderful"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Excited response to a friend's travel announcement. "  # noqa: E501
            "Good: 'omg so jealous!', 'yesss enjoy!', 'send pics!'. "  # noqa: E501
            "Bad: formal wishes, travel advice, assistant-like response."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Music share",  # noqa: E501
        "category": "social",  # noqa: E501
        "context": "[21:00] Jake: Have you heard the new Kendrick album?",  # noqa: E501
        "last_message": "Have you heard the new Kendrick album?",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "casual, opinionated",  # noqa: E501
        "max_chars": 80,  # noqa: E501
        "rubric": (  # noqa: E501
            "Natural response to a music recommendation question. "  # noqa: E501
            "Good: 'not yet, is it good?', 'yeah it slaps'. "  # noqa: E501
            "Bad: formal review, overly long, AI-sounding analysis."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    # =========================================================================  # noqa: E501
    # clarify: Low-context / ambiguous (7 cases)  # noqa: E501
    # =========================================================================  # noqa: E501
    {  # noqa: E501
        "name": "Ambiguous question mark",  # noqa: E501
        "category": "clarify",  # noqa: E501
        "context": "[11:00] Chris: ?",  # noqa: E501
        "last_message": "?",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "casual",  # noqa: E501
        "rubric": (  # noqa: E501
            "Reply to just a '?' with no context. Should ask for clarification briefly. "  # noqa: E501
            "Good: 'what's up?', '??', 'hm?'. "  # noqa: E501
            "Bad: long response, assuming what they mean."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "No context - bare hey",  # noqa: E501
        "category": "clarify",  # noqa: E501
        "context": "[11:00] Unknown: hey",  # noqa: E501
        "last_message": "hey",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "",  # noqa: E501
        "max_chars": 30,  # noqa: E501
        "rubric": (  # noqa: E501
            "Reply to 'hey' from an unknown person with zero context. "  # noqa: E501
            "Should be very brief - a simple greeting back. "  # noqa: E501
            "Good: 'hey', 'hey what's up', 'yo'. "  # noqa: E501
            "Bad: long reply, introducing yourself, asking detailed questions."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Ambiguous forwarded link",  # noqa: E501
        "category": "clarify",  # noqa: E501
        "context": "[12:00] Sarah: [Link]",  # noqa: E501
        "last_message": "[Link]",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "",  # noqa: E501
        "max_chars": 40,  # noqa: E501
        "banned": ["article", "interesting"],  # noqa: E501
        "rubric": (  # noqa: E501
            "Someone sent just a link with no text. Model should NOT confabulate "  # noqa: E501
            "what the link is about. Good: 'what's this?', '?', 'ooh what is it'. "  # noqa: E501
            "Bad: commenting on the content, assuming what it is."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Stale thread - weeks old",  # noqa: E501
        "category": "clarify",  # noqa: E501
        "context": "[3 weeks ago] Dave: hey you free Saturday?",  # noqa: E501
        "last_message": "hey you free Saturday?",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "casual",  # noqa: E501
        "max_chars": 60,  # noqa: E501
        "rubric": (  # noqa: E501
            "Replying to a 3-week-old message asking about Saturday. "  # noqa: E501
            "Should acknowledge the staleness or not pretend it's timely. "  # noqa: E501
            "Good: 'sorry just saw this', 'lol my bad, super late'. "  # noqa: E501
            "Bad: answering as if it's current ('yeah I'm free!')."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Emoji only message",  # noqa: E501
        "category": "clarify",  # noqa: E501
        "context": "[13:00] Tina: \U0001f602\U0001f602\U0001f602",  # noqa: E501
        "last_message": "\U0001f602\U0001f602\U0001f602",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "casual",  # noqa: E501
        "max_chars": 30,  # noqa: E501
        "rubric": (  # noqa: E501
            "Reply to a message that's just laughing emojis with no context. "  # noqa: E501
            "Good: 'lol', '\U0001f602', 'what', '??'. "  # noqa: E501
            "Bad: long response, asking detailed questions, pretending to know what's funny."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Voice memo reference",  # noqa: E501
        "category": "clarify",  # noqa: E501
        "context": "[16:00] Dan: [Voice Memo]",  # noqa: E501
        "last_message": "[Voice Memo]",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "",  # noqa: E501
        "max_chars": 50,  # noqa: E501
        "rubric": (  # noqa: E501
            "Reply to a voice memo that can't be transcribed. "  # noqa: E501
            "Good: 'can't listen rn, what's up?', 'send a text lol'. "  # noqa: E501
            "Bad: pretending to have heard it, long response."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
    {  # noqa: E501
        "name": "Wrong number / random text",  # noqa: E501
        "category": "clarify",  # noqa: E501
        "context": "[08:00] Unknown: Tell Maria I'll be there at 6",  # noqa: E501
        "last_message": "Tell Maria I'll be there at 6",  # noqa: E501
        "tone": "casual",  # noqa: E501
        "user_style": "",  # noqa: E501
        "max_chars": 50,  # noqa: E501
        "rubric": (  # noqa: E501
            "Reply to what looks like a wrong-number text. "  # noqa: E501
            "Good: 'wrong number', 'think you have the wrong person'. "  # noqa: E501
            "Bad: agreeing to tell Maria, long explanation."  # noqa: E501
        ),  # noqa: E501
    },  # noqa: E501
]  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class EvalResult:  # noqa: E501
    name: str  # noqa: E501
    category: str  # noqa: E501
    output: str  # noqa: E501
    latency_ms: float  # noqa: E501
    checks_passed: list[str]  # noqa: E501
    checks_failed: list[str]  # noqa: E501
    passed: bool  # noqa: E501
    judge_score: float | None = None  # noqa: E501
    judge_reasoning: str = ""  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# LLM Judge  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_judge_client():  # noqa: E501
    """Create OpenAI-compatible client for the judge model."""  # noqa: E501
    return _get_judge_client()  # noqa: E501
  # noqa: E501
  # noqa: E501
def judge_response(client, tc: dict, output: str) -> tuple[float, str]:  # noqa: E501
    """Score a response using the LLM judge.  # noqa: E501
  # noqa: E501
    Returns (score 0-10, reasoning).  # noqa: E501
    """  # noqa: E501
    rubric = tc.get("rubric", "")  # noqa: E501
    if not rubric:  # noqa: E501
        return -1.0, "no rubric"  # noqa: E501
  # noqa: E501
    prompt = (  # noqa: E501
        "You are an expert evaluator for a text message reply generator.\n\n"  # noqa: E501
        f"CONVERSATION:\n{tc['context']}\n\n"  # noqa: E501
        f"LAST MESSAGE (to reply to):\n{tc['last_message']}\n\n"  # noqa: E501
        f"GENERATED REPLY:\n{output}\n\n"  # noqa: E501
        f"RUBRIC:\n{rubric}\n\n"  # noqa: E501
        "Score the generated reply from 0-10 based on the rubric.\n"  # noqa: E501
        "Respond in this exact JSON format:\n"  # noqa: E501
        '{"score": <0-10>, "reasoning": "<1-2 sentences>"}'  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        resp = client.chat.completions.create(  # noqa: E501
            model=JUDGE_MODEL,  # noqa: E501
            messages=[{"role": "user", "content": prompt}],  # noqa: E501
            temperature=0.0,  # noqa: E501
            max_tokens=150,  # noqa: E501
        )  # noqa: E501
        text = resp.choices[0].message.content.strip()  # noqa: E501
        # Parse JSON from response (handle markdown fences)  # noqa: E501
        if text.startswith("```"):  # noqa: E501
            text = text.split("```")[1]  # noqa: E501
            if text.startswith("json"):  # noqa: E501
                text = text[4:]  # noqa: E501
        data = json.loads(text)  # noqa: E501
        return float(data["score"]), data.get("reasoning", "")  # noqa: E501
    except Exception as e:  # noqa: E501
        return -1.0, f"judge error: {e}"  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Local Checks  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def build_prompt(tc: dict) -> str:  # noqa: E501
    """Build XML drafter prompt (default strategy)."""  # noqa: E501
    system = (  # noqa: E501
        "You draft text message replies matching the sender's exact style.\n"  # noqa: E501
        "Rules:\n"  # noqa: E501
        "- Match their texting style exactly "  # noqa: E501
        "(length, formality, abbreviations, emoji, punctuation)\n"  # noqa: E501
        "- Sound natural, never like an AI\n"  # noqa: E501
        '- No phrases like "I hope this helps" or "Let me know"\n'  # noqa: E501
        "- No formal greetings unless they use them\n"  # noqa: E501
        "- If the message is unclear or you lack context to reply properly, "  # noqa: E501
        'respond with just "?"'  # noqa: E501
    )  # noqa: E501
    style = (  # noqa: E501
        f"Tone: {tc['tone']}. Style: {tc['user_style']}"  # noqa: E501
        if tc["user_style"]  # noqa: E501
        else f"Tone: {tc['tone']}"  # noqa: E501
    )  # noqa: E501
    return (  # noqa: E501
        f"<system>\n{system}</system>\n\n"  # noqa: E501
        f"<style>\n{style}\n</style>\n\n"  # noqa: E501
        f"<conversation>\n{tc['context']}\n</conversation>\n\n"  # noqa: E501
        f"<last_message>{tc['last_message']}</last_message>\n\n"  # noqa: E501
        f"<reply>"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
def check_result(tc: dict, output: str) -> tuple[list[str], list[str]]:  # noqa: E501
    """Run local assertions. Returns (passed, failed)."""  # noqa: E501
    passed = []  # noqa: E501
    failed = []  # noqa: E501
    lower = output.lower()  # noqa: E501
  # noqa: E501
    # Anti-AI phrases (global)  # noqa: E501
    for phrase in ANTI_AI_PHRASES:  # noqa: E501
        if phrase in lower:  # noqa: E501
            failed.append(f"contains anti-AI phrase: '{phrase}'")  # noqa: E501
        else:  # noqa: E501
            passed.append(f"no '{phrase}'")  # noqa: E501
  # noqa: E501
    # Max words  # noqa: E501
    if "max_words" in tc:  # noqa: E501
        word_count = len(output.split())  # noqa: E501
        if word_count <= tc["max_words"]:  # noqa: E501
            passed.append(f"words={word_count} <= {tc['max_words']}")  # noqa: E501
        else:  # noqa: E501
            failed.append(f"words={word_count} > {tc['max_words']}")  # noqa: E501
  # noqa: E501
    # Max chars  # noqa: E501
    if "max_chars" in tc:  # noqa: E501
        if len(output) <= tc["max_chars"]:  # noqa: E501
            passed.append(f"chars={len(output)} <= {tc['max_chars']}")  # noqa: E501
        else:  # noqa: E501
            failed.append(f"chars={len(output)} > {tc['max_chars']}")  # noqa: E501
  # noqa: E501
    # Banned words  # noqa: E501
    for word in tc.get("banned", []):  # noqa: E501
        if word.lower() in lower:  # noqa: E501
            failed.append(f"contains banned: '{word}'")  # noqa: E501
        else:  # noqa: E501
            passed.append(f"no '{word}'")  # noqa: E501
  # noqa: E501
    # Basic sanity  # noqa: E501
    if not output.strip():  # noqa: E501
        failed.append("empty output")  # noqa: E501
    else:  # noqa: E501
        passed.append("non-empty output")  # noqa: E501
  # noqa: E501
    if len(output) > 300:  # noqa: E501
        failed.append(f"way too long ({len(output)} chars)")  # noqa: E501
    else:  # noqa: E501
        passed.append(f"reasonable length ({len(output)} chars)")  # noqa: E501
  # noqa: E501
    return passed, failed  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Main  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    import argparse  # noqa: E501
  # noqa: E501
    parser = argparse.ArgumentParser(description="JARVIS Batch Eval")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--judge", action="store_true", help="Enable LLM judge scoring via Cerebras"  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--optimized",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Use DSPy-compiled program instead of raw generation",  # noqa: E501
    )  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Setup logging  # noqa: E501
    log_path = PROJECT_ROOT / "results" / "batch_eval.log"  # noqa: E501
    log_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    logging.basicConfig(  # noqa: E501
        level=logging.INFO,  # noqa: E501
        format="%(asctime)s - %(levelname)s - %(message)s",  # noqa: E501
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],  # noqa: E501
    )  # noqa: E501
    logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
    strategy = "dspy_optimized" if args.optimized else "xml_drafter"  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print("JARVIS BATCH EVAL - Response Generation", flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print(f"Test cases:  {len(TEST_CASES)}", flush=True)  # noqa: E501
    print(f"Strategy:    {strategy}", flush=True)  # noqa: E501
    judge_label = f"{JUDGE_MODEL} via DeepInfra" if args.judge else "disabled (use --judge)"  # noqa: E501
    print(f"LLM judge:   {judge_label}", flush=True)  # noqa: E501
    print(flush=True)  # noqa: E501
  # noqa: E501
    # Init judge  # noqa: E501
    judge_client = None  # noqa: E501
    if args.judge:  # noqa: E501
        judge_client = get_judge_client()  # noqa: E501
        if judge_client is None:  # noqa: E501
            print("WARNING: CEREBRAS_API_KEY not set in .env - skipping judge", flush=True)  # noqa: E501
            print("         Put your key in .env and re-run with --judge", flush=True)  # noqa: E501
        else:  # noqa: E501
            print(f"Judge ready: {JUDGE_MODEL} via Cerebras", flush=True)  # noqa: E501
    print(flush=True)  # noqa: E501
  # noqa: E501
    # Load model / compiled program  # noqa: E501
    dspy_program = None  # noqa: E501
    loader = None  # noqa: E501
  # noqa: E501
    # Set MLX memory limits early to prevent swap thrashing on 8GB systems.  # noqa: E501
    # loader.load() also sets these, but we set them before any MLX import  # noqa: E501
    # to guard against accidental early allocation.  # noqa: E501
    from models.memory_config import apply_embedder_limits  # noqa: E501
  # noqa: E501
    apply_embedder_limits()  # noqa: E501
  # noqa: E501
    if args.optimized:  # noqa: E501
        optimized_dir = PROJECT_ROOT / "evals" / "optimized_reply"  # noqa: E501
        if not optimized_dir.exists():  # noqa: E501
            print("ERROR: No compiled program found at evals/optimized_reply/", flush=True)  # noqa: E501
            print("       Run: uv run python evals/dspy_optimize.py", flush=True)  # noqa: E501
            return 1  # noqa: E501
        print("Loading DSPy compiled program...", flush=True)  # noqa: E501
        load_start = time.perf_counter()  # noqa: E501
        try:  # noqa: E501
            import dspy  # noqa: E501
            from evals.dspy_client import DSPYMLXClient  # noqa: E501
            from evals.dspy_reply import ReplyModule  # noqa: E501
  # noqa: E501
            student_lm = DSPYMLXClient(max_tokens=50, temperature=0.1)  # noqa: E501
            dspy.configure(lm=student_lm)  # noqa: E501
            dspy_program = ReplyModule()  # noqa: E501
            dspy_program.load(str(optimized_dir))  # noqa: E501
            load_ms = (time.perf_counter() - load_start) * 1000  # noqa: E501
            print(f"Compiled program loaded in {load_ms:.0f}ms", flush=True)  # noqa: E501
        except Exception as e:  # noqa: E501
            print(f"FATAL: Failed to load compiled program: {e}", flush=True)  # noqa: E501
            return 1  # noqa: E501
    else:  # noqa: E501
        print("Loading MLX model...", flush=True)  # noqa: E501
        load_start = time.perf_counter()  # noqa: E501
        try:  # noqa: E501
            from models.loader import get_model  # noqa: E501
  # noqa: E501
            loader = get_model()  # noqa: E501
            if not loader.is_loaded():  # noqa: E501
                loader.load()  # noqa: E501
            load_ms = (time.perf_counter() - load_start) * 1000  # noqa: E501
            print(f"Model loaded in {load_ms:.0f}ms", flush=True)  # noqa: E501
        except Exception as e:  # noqa: E501
            print(f"FATAL: Failed to load model: {e}", flush=True)  # noqa: E501
            return 1  # noqa: E501
  # noqa: E501
    print(flush=True)  # noqa: E501
    print("-" * 70, flush=True)  # noqa: E501
  # noqa: E501
    results: list[EvalResult] = []  # noqa: E501
    total_start = time.perf_counter()  # noqa: E501
  # noqa: E501
    # Resume support: load partial results from checkpoint file  # noqa: E501
    checkpoint_path = PROJECT_ROOT / "results" / "batch_eval_checkpoint.jsonl"  # noqa: E501
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    completed_names: set[str] = set()  # noqa: E501
    if checkpoint_path.exists():  # noqa: E501
        for line in checkpoint_path.read_text().splitlines():  # noqa: E501
            if line.strip():  # noqa: E501
                rec = json.loads(line)  # noqa: E501
                completed_names.add(rec["name"])  # noqa: E501
                results.append(  # noqa: E501
                    EvalResult(  # noqa: E501
                        name=rec["name"],  # noqa: E501
                        category=rec["category"],  # noqa: E501
                        output=rec["output"],  # noqa: E501
                        latency_ms=rec["latency_ms"],  # noqa: E501
                        checks_passed=rec.get("checks_passed", []),  # noqa: E501
                        checks_failed=rec.get("checks_failed", []),  # noqa: E501
                        passed=rec["local_passed"],  # noqa: E501
                        judge_score=rec.get("judge_score"),  # noqa: E501
                        judge_reasoning=rec.get("judge_reasoning", ""),  # noqa: E501
                    )  # noqa: E501
                )  # noqa: E501
        if completed_names:  # noqa: E501
            print(  # noqa: E501
                f"Resuming: {len(completed_names)}/{len(TEST_CASES)} already completed",  # noqa: E501
                flush=True,  # noqa: E501
            )  # noqa: E501
  # noqa: E501
    checkpoint_f = checkpoint_path.open("a", encoding="utf-8")  # noqa: E501
  # noqa: E501
    for i, tc in enumerate(tqdm(TEST_CASES, desc="Evaluating"), 1):  # noqa: E501
        if tc["name"] in completed_names:  # noqa: E501
            continue  # noqa: E501
        # Generate via DSPy compiled program or raw model  # noqa: E501
        gen_start = time.perf_counter()  # noqa: E501
        try:  # noqa: E501
            if dspy_program is not None:  # noqa: E501
                pred = dspy_program(  # noqa: E501
                    context=tc["context"],  # noqa: E501
                    last_message=tc["last_message"],  # noqa: E501
                    tone=tc["tone"],  # noqa: E501
                    user_style=tc.get("user_style", ""),  # noqa: E501
                )  # noqa: E501
                output = pred.reply.strip()  # noqa: E501
            else:  # noqa: E501
                prompt = build_prompt(tc)  # noqa: E501
                result = loader.generate_sync(  # noqa: E501
                    prompt=prompt,  # noqa: E501
                    temperature=0.1,  # noqa: E501
                    max_tokens=50,  # noqa: E501
                    top_p=0.1,  # noqa: E501
                    top_k=50,  # noqa: E501
                    repetition_penalty=1.05,  # noqa: E501
                )  # noqa: E501
                output = result.text.strip()  # noqa: E501
            latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E501
        except Exception as e:  # noqa: E501
            output = f"[ERROR: {e}]"  # noqa: E501
            latency_ms = (time.perf_counter() - gen_start) * 1000  # noqa: E501
  # noqa: E501
        # Local checks  # noqa: E501
        passed_checks, failed_checks = check_result(tc, output)  # noqa: E501
        all_passed = len(failed_checks) == 0  # noqa: E501
  # noqa: E501
        # Judge scoring  # noqa: E501
        judge_score = None  # noqa: E501
        judge_reasoning = ""  # noqa: E501
        if judge_client and tc.get("rubric"):  # noqa: E501
            judge_score, judge_reasoning = judge_response(judge_client, tc, output)  # noqa: E501
  # noqa: E501
        er = EvalResult(  # noqa: E501
            name=tc["name"],  # noqa: E501
            category=tc.get("category", "unknown"),  # noqa: E501
            output=output,  # noqa: E501
            latency_ms=latency_ms,  # noqa: E501
            checks_passed=passed_checks,  # noqa: E501
            checks_failed=failed_checks,  # noqa: E501
            passed=all_passed,  # noqa: E501
            judge_score=judge_score,  # noqa: E501
            judge_reasoning=judge_reasoning,  # noqa: E501
        )  # noqa: E501
        results.append(er)  # noqa: E501
  # noqa: E501
        # Write checkpoint incrementally (survives crash)  # noqa: E501
        checkpoint_f.write(  # noqa: E501
            json.dumps(  # noqa: E501
                {  # noqa: E501
                    "name": er.name,  # noqa: E501
                    "category": er.category,  # noqa: E501
                    "output": er.output,  # noqa: E501
                    "latency_ms": round(er.latency_ms, 1),  # noqa: E501
                    "local_passed": er.passed,  # noqa: E501
                    "checks_passed": er.checks_passed,  # noqa: E501
                    "checks_failed": er.checks_failed,  # noqa: E501
                    "judge_score": er.judge_score,  # noqa: E501
                    "judge_reasoning": er.judge_reasoning,  # noqa: E501
                }  # noqa: E501
            )  # noqa: E501
            + "\n"  # noqa: E501
        )  # noqa: E501
        checkpoint_f.flush()  # noqa: E501
  # noqa: E501
        # Print per-case  # noqa: E501
        status = "PASS" if all_passed else "FAIL"  # noqa: E501
        cat = tc.get("category", "?")  # noqa: E501
        print(f"\n[{i:2d}/{len(TEST_CASES)}] [{cat}] {tc['name']}", flush=True)  # noqa: E501
        print(f'  Output:  "{output}"', flush=True)  # noqa: E501
        judge_str = ""  # noqa: E501
        if judge_score is not None and judge_score >= 0:  # noqa: E501
            judge_str = f" | Judge: {judge_score:.0f}/10"  # noqa: E501
        print(f"  Latency: {latency_ms:.0f}ms | Local: {status}{judge_str}", flush=True)  # noqa: E501
        if failed_checks:  # noqa: E501
            for f in failed_checks:  # noqa: E501
                print(f"  FAIL: {f}", flush=True)  # noqa: E501
        if judge_reasoning:  # noqa: E501
            print(f"  Judge: {judge_reasoning}", flush=True)  # noqa: E501
  # noqa: E501
    total_ms = (time.perf_counter() - total_start) * 1000  # noqa: E501
  # noqa: E501
    # Summary  # noqa: E501
    print(flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
    print("SUMMARY", flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
  # noqa: E501
    n_passed = sum(1 for r in results if r.passed)  # noqa: E501
    n_failed = len(results) - n_passed  # noqa: E501
    latencies = [r.latency_ms for r in results]  # noqa: E501
    avg_latency = sum(latencies) / len(latencies) if latencies else 0  # noqa: E501
    sorted_lat = sorted(latencies)  # noqa: E501
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0  # noqa: E501
    p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)  # noqa: E501
    p95 = sorted_lat[p95_idx] if sorted_lat else 0  # noqa: E501
  # noqa: E501
    print(  # noqa: E501
        f"Local pass:   {n_passed}/{len(results)} ({n_passed / len(results) * 100:.0f}%)",  # noqa: E501
        flush=True,  # noqa: E501
    )  # noqa: E501
    print(f"Failed:       {n_failed}", flush=True)  # noqa: E501
    print(f"Total time:   {total_ms:.0f}ms", flush=True)  # noqa: E501
    print(f"Avg latency:  {avg_latency:.0f}ms", flush=True)  # noqa: E501
    print(f"P50 latency:  {p50:.0f}ms", flush=True)  # noqa: E501
    print(f"P95 latency:  {p95:.0f}ms", flush=True)  # noqa: E501
  # noqa: E501
    # Judge summary  # noqa: E501
    scored = [r for r in results if r.judge_score is not None and r.judge_score >= 0]  # noqa: E501
    if scored:  # noqa: E501
        scores = [r.judge_score for r in scored]  # noqa: E501
        avg_score = sum(scores) / len(scores)  # noqa: E501
        min_score = min(scores)  # noqa: E501
        max_score = max(scores)  # noqa: E501
        pass_7 = sum(1 for s in scores if s >= 7)  # noqa: E501
        print(flush=True)  # noqa: E501
        print(  # noqa: E501
            f"Judge scores: avg={avg_score:.1f}/10  min={min_score:.0f}  max={max_score:.0f}",  # noqa: E501
            flush=True,  # noqa: E501
        )  # noqa: E501
        print(  # noqa: E501
            f"Judge pass (>=7): {pass_7}/{len(scored)} ({pass_7 / len(scored) * 100:.0f}%)",  # noqa: E501
            flush=True,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    if n_failed:  # noqa: E501
        print("\nLocal failures:", flush=True)  # noqa: E501
        for r in results:  # noqa: E501
            if not r.passed:  # noqa: E501
                print(f"  - {r.name}: {', '.join(r.checks_failed)}", flush=True)  # noqa: E501
  # noqa: E501
    # Per-category breakdown  # noqa: E501
    print(flush=True)  # noqa: E501
    print("PER-CATEGORY BREAKDOWN", flush=True)  # noqa: E501
    print("-" * 70, flush=True)  # noqa: E501
    for cat in CATEGORIES:  # noqa: E501
        cat_results = [r for r in results if r.category == cat]  # noqa: E501
        if not cat_results:  # noqa: E501
            continue  # noqa: E501
        cat_passed = sum(1 for r in cat_results if r.passed)  # noqa: E501
        cat_scored = [r for r in cat_results if r.judge_score is not None and r.judge_score >= 0]  # noqa: E501
        cat_judge = ""  # noqa: E501
        if cat_scored:  # noqa: E501
            cat_avg = sum(r.judge_score for r in cat_scored) / len(cat_scored)  # noqa: E501
            cat_judge = f"  judge_avg={cat_avg:.1f}/10"  # noqa: E501
        print(  # noqa: E501
            f"  {cat:20s}  local={cat_passed}/{len(cat_results)}"  # noqa: E501
            f" ({cat_passed / len(cat_results) * 100:.0f}%){cat_judge}",  # noqa: E501
            flush=True,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    if scored:  # noqa: E501
        low = [r for r in scored if r.judge_score < 7]  # noqa: E501
        if low:  # noqa: E501
            print("\nLow judge scores (<7):", flush=True)  # noqa: E501
            for r in low:  # noqa: E501
                print(f"  - {r.name}: {r.judge_score:.0f}/10 - {r.judge_reasoning}", flush=True)  # noqa: E501
  # noqa: E501
    # Save results  # noqa: E501
    output_path = PROJECT_ROOT / "results" / "batch_eval_latest.json"  # noqa: E501
    output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
    output_data = {  # noqa: E501
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),  # noqa: E501
        "strategy": strategy,  # noqa: E501
        "judge_model": JUDGE_MODEL if scored else None,  # noqa: E501
        "total": len(results),  # noqa: E501
        "local_passed": n_passed,  # noqa: E501
        "local_failed": n_failed,  # noqa: E501
        "local_pass_rate": round(n_passed / len(results), 4),  # noqa: E501
        "judge_avg_score": round(avg_score, 2) if scored else None,  # noqa: E501
        "judge_pass_rate": round(pass_7 / len(scored), 4) if scored else None,  # noqa: E501
        "latency": {  # noqa: E501
            "avg_ms": round(avg_latency, 1),  # noqa: E501
            "p50_ms": round(p50, 1),  # noqa: E501
            "p95_ms": round(p95, 1),  # noqa: E501
            "total_ms": round(total_ms, 1),  # noqa: E501
        },  # noqa: E501
        "results": [  # noqa: E501
            {  # noqa: E501
                "name": r.name,  # noqa: E501
                "category": r.category,  # noqa: E501
                "output": r.output,  # noqa: E501
                "latency_ms": round(r.latency_ms, 1),  # noqa: E501
                "local_passed": r.passed,  # noqa: E501
                "failed_checks": r.checks_failed,  # noqa: E501
                "judge_score": r.judge_score,  # noqa: E501
                "judge_reasoning": r.judge_reasoning,  # noqa: E501
            }  # noqa: E501
            for r in results  # noqa: E501
        ],  # noqa: E501
    }  # noqa: E501
    output_path.write_text(json.dumps(output_data, indent=2))  # noqa: E501
    # Clean up checkpoint file after successful completion  # noqa: E501
    checkpoint_f.close()  # noqa: E501
    checkpoint_path.unlink(missing_ok=True)  # noqa: E501
    print(f"\nResults saved to: {output_path}", flush=True)  # noqa: E501
    print("=" * 70, flush=True)  # noqa: E501
  # noqa: E501
    return 0 if n_failed == 0 else 1  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
