#!/usr/bin/env python3
"""Interactive exploration of small MLX models.

Systematically test a model's capabilities: generation quality, instruction
following, tool routing, JSON output, classification, and more.

Usage:
    uv run python scripts/explore_model.py                          # interactive REPL
    uv run python scripts/explore_model.py --suite all              # run all test suites
    uv run python scripts/explore_model.py --suite tools,json       # specific suites
    uv run python scripts/explore_model.py --model lfm-1.2b-extract # different model
    uv run python scripts/explore_model.py --compare                # compare 350m vs 1.2b
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, ".")

# ─── Model Registry ────────────────────────────────────────────────────────

MODELS = {
    "lfm-350m-extract": "models/lfm2-350m-extract-mlx-4bit",
    "lfm-1.2b-extract": "models/lfm2-1.2b-extract-mlx-4bit",
    "lfm-0.3b": "mlx-community/LFM2-350M-4bit",  # base instruct (downloads on first use)
    "lfm-1.2b": "mlx-community/LFM2.5-1.2B-Instruct-MLX-4bit",  # base 1.2B instruct
}


# ─── Data Types ─────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    suite: str
    system: str | None
    user: str
    tools: list[dict] | None = None  # for tool-use tests
    expected: str | None = None  # substring or pattern we expect
    max_tokens: int = 200
    assistant_prefix: str | None = None  # seed the assistant's response
    fim_prefix: str | None = None  # FIM: content before the hole
    fim_suffix: str | None = None  # FIM: content after the hole
    logit_bias: dict[int, float] | None = None  # token_id -> bias (+boost, -suppress)
    inverted: bool = False  # flip expected for inverted-question strategies


@dataclass
class TestResult:
    test: str
    suite: str
    model: str
    prompt: str
    response: str
    time_ms: float
    tokens: int
    expected: str | None = None
    passed: bool | None = None  # None = manual review needed


# ─── Test Suites ────────────────────────────────────────────────────────────

def suite_basics() -> list[TestCase]:
    """Basic generation: can it follow simple instructions?"""
    return [
        TestCase(
            name="complete_sentence",
            suite="basics",
            system=None,
            user="Complete this sentence: The capital of France is",
            expected="Paris",
        ),
        TestCase(
            name="list_3_items",
            suite="basics",
            system=None,
            user="List 3 fruits. Just the names, one per line.",
            max_tokens=50,
        ),
        TestCase(
            name="yes_no_question",
            suite="basics",
            system=None,
            user="Is the sky blue? Answer only yes or no.",
            expected="yes",
            max_tokens=10,
        ),
        TestCase(
            name="follow_format",
            suite="basics",
            system="You are a helpful assistant. Always respond in exactly one sentence.",
            user="What is Python?",
            max_tokens=80,
        ),
        TestCase(
            name="system_prompt_adherence",
            suite="basics",
            system="You only speak in uppercase. Never use lowercase letters.",
            user="Hello, how are you?",
            max_tokens=50,
        ),
        TestCase(
            name="refuse_hallucination",
            suite="basics",
            system=None,
            user='What is "flibberdygook quantum"? If you don\'t know, say "I don\'t know".',
            expected="don't know",
            max_tokens=50,
        ),
    ]


def suite_json() -> list[TestCase]:
    """JSON output: can it produce valid, structured JSON?"""
    return [
        TestCase(
            name="simple_json",
            suite="json",
            system="You are a JSON extraction system. Output ONLY valid JSON, no other text.",
            user='Extract the name and age: "John is 25 years old"',
            expected='"name"',
            max_tokens=100,
        ),
        TestCase(
            name="simple_json_seeded",
            suite="json",
            system="You are a JSON extraction system. Output ONLY valid JSON, no other text.",
            user='Extract the name and age: "John is 25 years old"',
            assistant_prefix='{"',
            expected="John",
            max_tokens=100,
        ),
        TestCase(
            name="json_array",
            suite="json",
            system="Output valid JSON only.",
            user='List the colors mentioned: "I like red, blue, and green". Output as {"colors": [...]}',
            expected='"colors"',
            max_tokens=100,
        ),
        TestCase(
            name="json_empty",
            suite="json",
            system="Extract facts as JSON. If no facts found, output {\"facts\": []}",
            user='Message: "haha ok"',
            expected="[]",
            max_tokens=50,
        ),
        TestCase(
            name="json_empty_seeded",
            suite="json",
            system="Extract facts as JSON. If no facts found, output {\"facts\": []}",
            user='Message: "haha ok"',
            assistant_prefix='{"facts": [',
            expected="]",
            max_tokens=50,
        ),
        TestCase(
            name="json_nested",
            suite="json",
            system="Output valid JSON only.",
            user='Extract: "Alice works at Google in NYC". Format: {"person": {"name": "...", "employer": "...", "city": "..."}}',
            expected='"person"',
            max_tokens=150,
        ),
        TestCase(
            name="json_nested_seeded",
            suite="json",
            system="Output valid JSON only.",
            user='Extract: "Alice works at Google in NYC". Format: {"person": {"name": "...", "employer": "...", "city": "..."}}',
            assistant_prefix='{"person": {"name": "',
            expected="Alice",
            max_tokens=150,
        ),
        TestCase(
            name="json_no_markdown",
            suite="json",
            system="Output raw JSON only. Do NOT wrap in ```json blocks.",
            user='Extract: "Bob likes pizza". Output: {"name": "...", "preference": "..."}',
            max_tokens=80,
        ),
    ]


def suite_classification() -> list[TestCase]:
    """Classification: can it categorize inputs?"""
    return [
        TestCase(
            name="sentiment_positive",
            suite="classification",
            system="Classify the sentiment as: positive, negative, or neutral. Output ONE word only.",
            user="I love this new restaurant!",
            expected="positive",
            max_tokens=5,
        ),
        TestCase(
            name="sentiment_negative",
            suite="classification",
            system="Classify the sentiment as: positive, negative, or neutral. Output ONE word only.",
            user="This is terrible and I hate it.",
            expected="negative",
            max_tokens=5,
        ),
        TestCase(
            name="intent_3way",
            suite="classification",
            system="Classify the user's intent as: question, request, or statement. Output ONE word.",
            user="Can you tell me the weather?",
            expected="question",
            max_tokens=5,
        ),
        TestCase(
            name="intent_request",
            suite="classification",
            system="Classify the user's intent as: question, request, or statement. Output ONE word.",
            user="Please set a reminder for 3pm.",
            expected="request",
            max_tokens=5,
        ),
        TestCase(
            name="topic_classification",
            suite="classification",
            system=(
                "Classify the message into exactly one category: "
                "food, travel, work, health, social. Output ONE word."
            ),
            user="I'm going to the gym after work today",
            expected="health",
            max_tokens=5,
        ),
        TestCase(
            name="binary_gate",
            suite="classification",
            system=None,
            user=(
                'Does this message contain personal facts about someone? '
                'Answer ONLY "yes" or "no".\n\nMessage: "I just moved to Austin"'
            ),
            expected="yes",
            max_tokens=5,
        ),
        TestCase(
            name="binary_gate_no",
            suite="classification",
            system=None,
            user=(
                'Does this message contain personal facts about someone? '
                'Answer ONLY "yes" or "no".\n\nMessage: "lol ok"'
            ),
            expected="no",
            max_tokens=5,
        ),
    ]


def suite_extraction() -> list[TestCase]:
    """Information extraction from text."""
    return [
        TestCase(
            name="extract_name_location",
            suite="extraction",
            system="Extract personal facts. Format: type: value (one per line). If none, say 'none'.",
            user='Message: "I just moved to Austin from Denver"',
            expected="Austin",
            max_tokens=100,
        ),
        TestCase(
            name="extract_relationship",
            suite="extraction",
            system="Extract personal facts. Format: type: value (one per line). If none, say 'none'.",
            user='Message: "My sister Sarah works at Apple"',
            expected="sister",
            max_tokens=100,
        ),
        TestCase(
            name="extract_none",
            suite="extraction",
            system="Extract personal facts. Format: type: value (one per line). If none, say 'none'.",
            user='Message: "haha yeah that\'s funny"',
            expected="none",
            max_tokens=50,
        ),
        TestCase(
            name="extract_preference",
            suite="extraction",
            system="Extract personal facts. Format: type: value (one per line). If none, say 'none'.",
            user='Message: "I\'m allergic to peanuts so I can\'t eat that"',
            expected="peanut",
            max_tokens=100,
        ),
        TestCase(
            name="extract_multiple",
            suite="extraction",
            system="Extract ALL personal facts. Format: type: value (one per line).",
            user='Message: "My friend Jake is a doctor in Boston and he loves running"',
            max_tokens=150,
        ),
    ]


def suite_tools() -> list[TestCase]:
    """Tool use: can it select and call tools?"""
    search_tool = {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    }
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a math calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"],
            },
        },
    }
    reminder_tool = {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a specific time",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Reminder text"},
                    "time": {"type": "string", "description": "Time for the reminder"},
                },
                "required": ["message", "time"],
            },
        },
    }
    all_tools = [search_tool, calculator_tool, reminder_tool]

    return [
        # LFM2.5 tool call format: <|tool_call_start|>[func_name(param="val")]<|tool_call_end|>
        TestCase(
            name="tool_search",
            suite="tools",
            system="You are a helpful assistant with access to tools. Use tools when needed.",
            user="What's the weather in San Francisco?",
            tools=all_tools,
            expected="search_web",
            max_tokens=150,
        ),
        TestCase(
            name="tool_calculate",
            suite="tools",
            system="You are a helpful assistant with access to tools. Use tools when needed.",
            user="What is 1547 * 823?",
            tools=all_tools,
            expected="calculate",
            max_tokens=150,
        ),
        TestCase(
            name="tool_reminder",
            suite="tools",
            system="You are a helpful assistant with access to tools. Use tools when needed.",
            user="Remind me to call mom at 3pm tomorrow",
            tools=all_tools,
            expected="set_reminder",
            max_tokens=150,
        ),
        TestCase(
            name="tool_no_tool_needed",
            suite="tools",
            system="You are a helpful assistant with access to tools. Only use tools when necessary.",
            user="Hello, how are you?",
            tools=all_tools,
            max_tokens=100,
        ),
        TestCase(
            name="tool_2_tools",
            suite="tools",
            system="You are a helpful assistant with access to tools.",
            user="Search for the population of Tokyo and calculate 15% of it.",
            tools=all_tools,
            expected="search_web",
            max_tokens=200,
        ),
        # Test with single tool (easier routing decision)
        TestCase(
            name="tool_single_option",
            suite="tools",
            system="You are a helpful assistant. Use the search tool to answer questions you don't know.",
            user="What is the current price of Bitcoin?",
            tools=[search_tool],
            expected="search_web",
            max_tokens=150,
        ),
    ]


def suite_reasoning() -> list[TestCase]:
    """Basic reasoning: how well does it think?"""
    return [
        TestCase(
            name="simple_logic",
            suite="reasoning",
            system=None,
            user="If all dogs are animals, and Rex is a dog, is Rex an animal? Answer yes or no.",
            expected="yes",
            max_tokens=20,
        ),
        TestCase(
            name="comparison",
            suite="reasoning",
            system=None,
            user="Which is larger: 0.8 or 0.75? Just give the number.",
            expected="0.8",
            max_tokens=10,
        ),
        TestCase(
            name="counting",
            suite="reasoning",
            system=None,
            user="How many vowels are in the word 'beautiful'? Just the number.",
            expected="5",
            max_tokens=10,
        ),
        TestCase(
            name="pattern",
            suite="reasoning",
            system=None,
            user="What comes next: 2, 4, 8, 16, ?",
            expected="32",
            max_tokens=10,
        ),
        TestCase(
            name="negation",
            suite="reasoning",
            system=None,
            user='Is this statement negative or positive: "I don\'t dislike chocolate"? Answer one word.',
            max_tokens=10,
        ),
    ]


def suite_constrained() -> list[TestCase]:
    """Constrained output: can it follow strict formatting rules?"""
    return [
        TestCase(
            name="single_word",
            suite="constrained",
            system="Respond with exactly ONE word. No punctuation, no explanation.",
            user="What color is grass?",
            expected="green",
            max_tokens=5,
        ),
        TestCase(
            name="number_only",
            suite="constrained",
            system="Respond with ONLY a number. No words, no units, no explanation.",
            user="How many legs does a spider have?",
            expected="8",
            max_tokens=5,
        ),
        TestCase(
            name="comma_separated",
            suite="constrained",
            system="List items as comma-separated values. No numbers, no bullets, no newlines.",
            user="Name 3 colors of the rainbow.",
            max_tokens=30,
        ),
        TestCase(
            name="fixed_template",
            suite="constrained",
            system=None,
            user=(
                "Fill in the template EXACTLY as shown:\n"
                "NAME: [person's name]\nCITY: [city]\nJOB: [job]\n\n"
                '"Alice is a teacher in Boston"'
            ),
            expected="Alice",
            max_tokens=50,
        ),
        TestCase(
            name="xml_tags",
            suite="constrained",
            system="Output results in XML tags: <result>your answer</result>",
            user="What is 2+2?",
            expected="<result>",
            max_tokens=30,
        ),
    ]


def suite_gate() -> list[TestCase]:
    """Binary gating: can the model decide yes/no on whether a message has extractable facts?"""

    # ─── Strategy 1: System prompt classification ────────────────────
    s1 = (
        "You are a text message classifier trained by Liquid AI. "
        "Classify whether a message contains personal facts (name, location, job, health, preference, relationship). "
        "Respond with exactly one word: yes or no."
    )

    # ─── Strategy 2: Ultra-short, no system prompt ───────────────────
    s2 = "yes or no only:"

    # ─── Strategy 3: Few-shot in system prompt ───────────────────────
    s3 = (
        "You are a fact detector. Classify messages as containing personal facts or not.\n\n"
        "Message: \"I moved to Austin\"\nAnswer: yes\n\n"
        "Message: \"lol ok\"\nAnswer: no\n\n"
        "Message: \"my sister works at Google\"\nAnswer: yes\n\n"
        "Message: \"haha\"\nAnswer: no\n\n"
    )

    # ─── Strategy 4: Assistant prefix seeding ────────────────────────
    # Force the model to start with a structured response
    s4_sys = (
        "You classify text messages. A message has facts if it mentions a name, "
        "place, job, health condition, preference, or relationship. "
        "Reply with ONLY: has_facts: yes or has_facts: no"
    )

    # ─── Strategy 5: Contrastive (show what IS and ISN'T a fact) ─────
    s5 = (
        "Personal facts: names, locations, jobs, health, relationships, preferences.\n"
        "NOT personal facts: greetings, reactions, acknowledgments, emotions, opinions about nothing specific.\n\n"
        "Classify the message. Output ONLY 'fact' or 'no_fact'."
    )

    # ─── Strategy 6: Task decomposition ──────────────────────────────
    s6 = (
        "Step 1: Read the message.\n"
        "Step 2: Does it mention a specific person, place, job, health issue, or preference?\n"
        "Step 3: If yes, output 'fact'. If no, output 'no_fact'.\n"
        "Output ONLY the final answer from Step 3."
    )

    # ─── Strategy 7: Completion-style (no system, pure pattern) ──────
    # No system prompt - put everything in user message as a pattern

    messages_yes = [
        ("I went to California last week", "yes_location"),
        ("just started my new job at Apple", "yes_job"),
        ("I cant eat gluten", "yes_health"),
        ("my sister works at Google", "yes_relationship"),
        ("we just signed a lease in Denver", "yes_moving"),
        ("just landed in NYC!", "yes_travel"),
        ("I have a doctors appointment tomorrow", "yes_health2"),
        ("gonna go hiking this weekend", "yes_activity"),
    ]
    messages_no = [
        ("lol", "no_lol"),
        ("haha thats funny", "no_laugh"),
        ("ok", "no_ok"),
        ("hey whats up", "no_greeting"),
        ("sounds good", "no_ack"),
        ("yeah", "no_yeah"),
        ("omg", "no_omg"),
        ("nice!", "no_nice"),
    ]

    tests = []

    # Strategy 1: simple system prompt
    for msg, label in messages_yes:
        tests.append(TestCase(
            name=f"s1_{label}",
            suite="gate",
            system=s1,
            user=f'Message: "{msg}"',
            expected="yes",
            max_tokens=5,
        ))
    for msg, label in messages_no:
        tests.append(TestCase(
            name=f"s1_{label}",
            suite="gate",
            system=s1,
            user=f'Message: "{msg}"',
            expected="no",
            max_tokens=5,
        ))

    # Strategy 2: ultra short
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s2_{label}",
            suite="gate",
            system=None,
            user=f'{s2} does "{msg}" contain personal facts?',
            expected="yes",
            max_tokens=5,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s2_{label}",
            suite="gate",
            system=None,
            user=f'{s2} does "{msg}" contain personal facts?',
            expected="no",
            max_tokens=5,
        ))

    # Strategy 3: few-shot (matching the example format in system prompt)
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s3_{label}",
            suite="gate",
            system=s3,
            user=f'Message: "{msg}"\nAnswer:',
            expected="yes",
            max_tokens=5,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s3_{label}",
            suite="gate",
            system=s3,
            user=f'Message: "{msg}"\nAnswer:',
            expected="no",
            max_tokens=5,
        ))

    # Strategy 4: assistant prefix seeding - force "has_facts: "
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s4_{label}",
            suite="gate",
            system=s4_sys,
            user=f'Message: "{msg}"',
            assistant_prefix="has_facts: ",
            expected="yes",
            max_tokens=5,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s4_{label}",
            suite="gate",
            system=s4_sys,
            user=f'Message: "{msg}"',
            assistant_prefix="has_facts: ",
            expected="no",
            max_tokens=5,
        ))

    # Strategy 5: contrastive (define what IS and ISN'T a fact)
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s5_{label}",
            suite="gate",
            system=s5,
            user=f'Message: "{msg}"',
            expected="fact",
            max_tokens=5,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s5_{label}",
            suite="gate",
            system=s5,
            user=f'Message: "{msg}"',
            expected="no_fact",
            max_tokens=5,
        ))

    # Strategy 6: task decomposition (step-by-step)
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s6_{label}",
            suite="gate",
            system=s6,
            user=f'Message: "{msg}"',
            expected="fact",
            max_tokens=5,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s6_{label}",
            suite="gate",
            system=s6,
            user=f'Message: "{msg}"',
            expected="no_fact",
            max_tokens=5,
        ))

    # Strategy 7: pure completion pattern (no system prompt)
    # Present it as a pattern to complete, not a question to answer
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s7_{label}",
            suite="gate",
            system=None,
            user=(
                'Classify each message as "fact" or "chat".\n\n'
                '"I moved to Austin" = fact\n'
                '"lol" = chat\n'
                '"my sister is a nurse" = fact\n'
                '"ok cool" = chat\n'
                '"gonna grab lunch" = chat\n'
                '"I work at Google" = fact\n\n'
                f'"{msg}" ='
            ),
            expected="fact",
            max_tokens=5,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s7_{label}",
            suite="gate",
            system=None,
            user=(
                'Classify each message as "fact" or "chat".\n\n'
                '"I moved to Austin" = fact\n'
                '"lol" = chat\n'
                '"my sister is a nurse" = fact\n'
                '"ok cool" = chat\n'
                '"gonna grab lunch" = chat\n'
                '"I work at Google" = fact\n\n'
                f'"{msg}" ='
            ),
            expected="chat",
            max_tokens=5,
        ))

    # ─── Strategy 8: INVERTED question ─────────────────────────────
    # The model always says "yes". So flip the question:
    # "Is this small talk?" → yes for chat, no for facts
    # Works WITH the model's yes-bias instead of against it
    s8 = (
        "You classify text messages. "
        "Is this message small talk, a reaction, or a greeting with NO personal information? "
        "Respond with exactly one word: yes or no."
    )
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s8_{label}",
            suite="gate",
            system=s8,
            user=f'Message: "{msg}"',
            expected="no",  # inverted: facts should get "no" (not small talk)
            max_tokens=5,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s8_{label}",
            suite="gate",
            system=s8,
            user=f'Message: "{msg}"',
            expected="yes",  # inverted: chat should get "yes" (is small talk)
            max_tokens=5,
        ))

    # ─── Strategy 9: Scoring instead of binary ─────────────────────
    # Don't ask yes/no. Ask for a number 0-5.
    # Threshold at 2+ = has facts
    s9 = (
        "Rate how much personal information this message contains on a scale of 0-5.\n"
        "0 = no info (greetings, reactions like lol/ok/haha)\n"
        "1 = vague reference\n"
        "3 = clear fact (location, job, health)\n"
        "5 = multiple specific facts\n"
        "Output ONLY the number."
    )
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s9_{label}",
            suite="gate",
            system=s9,
            user=f'Message: "{msg}"',
            expected=None,  # manual review - check if scores are reasonable
            max_tokens=5,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s9_{label}",
            suite="gate",
            system=s9,
            user=f'Message: "{msg}"',
            expected="0",  # chat should score 0
            max_tokens=5,
        ))

    # ─── Strategy 11: Logit bias - equalize yes/no probabilities ───
    # Don't push toward "no". Instead, suppress "yes" slightly so both
    # options are viable. The model has a ~5-point logit gap favoring "yes".
    # We reduce that gap so the actual content determines the answer.
    # Token IDs: yes=12184, Yes=12948, ' yes'=18672, no=2744, No=5048, ' no'=1295
    equalize_bias = {
        12184: -3.0,   # "yes" - suppress
        12948: -3.0,   # "Yes" - suppress
        18672: -3.0,   # " yes" - suppress
        2744: 0.0,     # "no" - leave as is
        5048: 0.0,     # "No" - leave as is
        1295: 0.0,     # " no" - leave as is
    }
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s11_{label}",
            suite="gate",
            system=s1,  # reuse strategy 1 prompt (was all-yes)
            user=f'Message: "{msg}"',
            expected="yes",
            max_tokens=5,
            logit_bias=equalize_bias,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s11_{label}",
            suite="gate",
            system=s1,
            user=f'Message: "{msg}"',
            expected="no",
            max_tokens=5,
            logit_bias=equalize_bias,
        ))

    # ─── Strategy 10: Extract-or-empty ─────────────────────────────
    # Don't ask binary. Ask it to extract. If output is empty/short = no facts.
    # This uses the model's strength (extraction) to solve the gating problem.
    s10 = (
        "Extract personal facts from this message. "
        "Output ONLY the facts as 'type: value', one per line. "
        "If there are no personal facts, output exactly: EMPTY"
    )
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"s10_{label}",
            suite="gate",
            system=s10,
            user=f'"{msg}"',
            expected=None,  # review: should have facts
            max_tokens=30,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"s10_{label}",
            suite="gate",
            system=s10,
            user=f'"{msg}"',
            expected="EMPTY",
            max_tokens=30,
        ))

    return tests


def suite_gate2() -> list[TestCase]:
    """Gate v2: more aggressive strategies to fix yes-bias."""
    messages_yes = [
        ("I went to California last week", "yes_location"),
        ("just started my new job at Apple", "yes_job"),
        ("I cant eat gluten", "yes_health"),
        ("my sister works at Google", "yes_relationship"),
        ("we just signed a lease in Denver", "yes_moving"),
        ("I have a doctors appointment tomorrow", "yes_health2"),
    ]
    messages_no = [
        ("lol", "no_lol"),
        ("haha thats funny", "no_laugh"),
        ("ok", "no_ok"),
        ("hey whats up", "no_greeting"),
        ("sounds good", "no_ack"),
        ("yeah", "no_yeah"),
    ]

    tests = []

    # ─── A: Heavy negative few-shot (6 no, 2 yes) ───────────────────
    # Counter the yes-bias by showing MORE negative examples
    sa = (
        "Classify if a message has personal facts. Most messages are just chat.\n\n"
        "Message: \"lol\" -> 0\n"
        "Message: \"ok cool\" -> 0\n"
        "Message: \"haha\" -> 0\n"
        "Message: \"I moved to Austin\" -> 1\n"
        "Message: \"yeah\" -> 0\n"
        "Message: \"sounds good\" -> 0\n"
        "Message: \"nice\" -> 0\n"
        "Message: \"my sister is a nurse\" -> 1\n"
    )
    for msg, label in messages_yes:
        tests.append(TestCase(
            name=f"a_{label}",
            suite="gate2",
            system=sa,
            user=f'Message: "{msg}" ->',
            expected="1",
            max_tokens=3,
        ))
    for msg, label in messages_no:
        tests.append(TestCase(
            name=f"a_{label}",
            suite="gate2",
            system=sa,
            user=f'Message: "{msg}" ->',
            expected="0",
            max_tokens=3,
        ))

    # ─── B: Constrained token generation ─────────────────────────────
    # Use logit bias to ONLY allow 0 and 1 tokens, suppress everything else
    # This is the nuclear option: the model can ONLY output 0 or 1
    # Token IDs for 0 and 1:
    #   "0" = need to find, "1" = need to find
    # We suppress "yes"/"no" and boost "0"/"1"
    constrain_bias = {
        12184: -100.0,  # "yes" - kill
        12948: -100.0,  # "Yes" - kill
        18672: -100.0,  # " yes" - kill
        2744: -100.0,   # "no" - kill
        5048: -100.0,   # "No" - kill
        1295: -100.0,   # " no" - kill
    }
    sb_sys = (
        "Classify if this message contains personal facts.\n"
        "Output 1 for facts, 0 for no facts.\n\n"
        "Examples:\n"
        "\"lol\" = 0\n"
        "\"I live in Austin\" = 1\n"
        "\"ok\" = 0\n"
        "\"my sister is a doctor\" = 1\n"
        "\"haha yeah\" = 0\n"
        "\"just got a job at Tesla\" = 1\n"
    )
    for msg, label in messages_yes:
        tests.append(TestCase(
            name=f"b_{label}",
            suite="gate2",
            system=sb_sys,
            user=f'"{msg}" =',
            expected="1",
            max_tokens=3,
            logit_bias=constrain_bias,
        ))
    for msg, label in messages_no:
        tests.append(TestCase(
            name=f"b_{label}",
            suite="gate2",
            system=sb_sys,
            user=f'"{msg}" =',
            expected="0",
            max_tokens=3,
            logit_bias=constrain_bias,
        ))

    # ─── C: Gentle logit bias (-1.5) + few-shot ─────────────────────
    # s11 used -3.0 which overcorrected. Try -1.5 with few-shot
    gentle_bias = {
        12184: -1.5,   # "yes" - gentle suppress
        12948: -1.5,   # "Yes"
        18672: -1.5,   # " yes"
    }
    for msg, label in messages_yes:
        tests.append(TestCase(
            name=f"c_{label}",
            suite="gate2",
            system=sa,  # reuse heavy negative few-shot
            user=f'Message: "{msg}" ->',
            expected="1",
            max_tokens=3,
            logit_bias=gentle_bias,
        ))
    for msg, label in messages_no:
        tests.append(TestCase(
            name=f"c_{label}",
            suite="gate2",
            system=sa,
            user=f'Message: "{msg}" ->',
            expected="0",
            max_tokens=3,
            logit_bias=gentle_bias,
        ))

    # ─── D: Two-step classification ──────────────────────────────────
    # Step 1: "What type of message is this?" (the model CAN label)
    # Then we map the label to has_facts/no_facts in code
    sd = (
        "Classify this message into ONE category:\n"
        "- greeting (hi, hey, whats up)\n"
        "- reaction (lol, haha, nice, omg, wow)\n"
        "- acknowledgment (ok, sure, sounds good, yeah, yep)\n"
        "- information (contains a name, place, job, health fact, plan, or preference)\n\n"
        "Output ONLY the category name."
    )
    for msg, label in messages_yes:
        tests.append(TestCase(
            name=f"d_{label}",
            suite="gate2",
            system=sd,
            user=f'Message: "{msg}"',
            expected="information",
            max_tokens=5,
        ))
    for msg, label in messages_no:
        tests.append(TestCase(
            name=f"d_{label}",
            suite="gate2",
            system=sd,
            user=f'Message: "{msg}"',
            # any of these = no facts
            expected=None,  # manual review - check category
            max_tokens=5,
        ))

    # ─── E: Prefix seed + few-shot completion ────────────────────────
    # Combine s7 (completion pattern, best at 75%) with assistant prefix
    for msg, label in messages_yes:
        tests.append(TestCase(
            name=f"e_{label}",
            suite="gate2",
            system=None,
            user=(
                'Label each message. Most are chat.\n\n'
                '"lol" = 0\n'
                '"ok cool" = 0\n'
                '"I moved to Austin" = 1\n'
                '"haha" = 0\n'
                '"sounds good" = 0\n'
                '"my sister is a nurse" = 1\n'
                '"yeah" = 0\n'
                '"nice!" = 0\n'
                '"gonna grab lunch" = 0\n'
                '"I work at Google" = 1\n\n'
                f'"{msg}" ='
            ),
            assistant_prefix=" ",
            expected="1",
            max_tokens=3,
        ))
    for msg, label in messages_no:
        tests.append(TestCase(
            name=f"e_{label}",
            suite="gate2",
            system=None,
            user=(
                'Label each message. Most are chat.\n\n'
                '"lol" = 0\n'
                '"ok cool" = 0\n'
                '"I moved to Austin" = 1\n'
                '"haha" = 0\n'
                '"sounds good" = 0\n'
                '"my sister is a nurse" = 1\n'
                '"yeah" = 0\n'
                '"nice!" = 0\n'
                '"gonna grab lunch" = 0\n'
                '"I work at Google" = 1\n\n'
                f'"{msg}" ='
            ),
            assistant_prefix=" ",
            expected="0",
            max_tokens=3,
        ))

    # ─── F: Message type with assistant prefix seed ──────────────────
    # Force start with the category, model just completes
    for msg, label in messages_yes[:4]:
        tests.append(TestCase(
            name=f"f_{label}",
            suite="gate2",
            system=(
                "Classify messages into: chat, reaction, greeting, or info.\n"
                "chat = casual conversation with no facts\n"
                "reaction = lol, haha, nice, omg\n"
                "greeting = hi, hey, whats up\n"
                "info = contains personal facts (name, place, job, health)"
            ),
            user=f'Message: "{msg}"\nCategory:',
            assistant_prefix=" ",
            expected="info",
            max_tokens=5,
        ))
    for msg, label in messages_no[:4]:
        tests.append(TestCase(
            name=f"f_{label}",
            suite="gate2",
            system=(
                "Classify messages into: chat, reaction, greeting, or info.\n"
                "chat = casual conversation with no facts\n"
                "reaction = lol, haha, nice, omg\n"
                "greeting = hi, hey, whats up\n"
                "info = contains personal facts (name, place, job, health)"
            ),
            user=f'Message: "{msg}"\nCategory:',
            assistant_prefix=" ",
            expected=None,  # review - should be chat/reaction/greeting
            max_tokens=5,
        ))

    return tests


def suite_postprocess() -> list[TestCase]:
    """Test extraction + post-processing: get clean structured output from verbose model."""
    # The key insight: ask the model to extract in a PARSEABLE format,
    # then we post-process in code. Don't fight the hallucination - filter it.
    sys = (
        "Extract personal facts from this conversation segment.\n"
        "Output ONLY in this exact format, one per line:\n"
        "FACT: [person] | [type] | [value]\n"
        "If no facts, output: FACT: none\n"
        "Do not explain. Do not add commentary."
    )
    return [
        TestCase(
            name="pp_simple",
            suite="postprocess",
            system=sys,
            user=(
                'Alex: "I just moved to Austin"\n'
                'Jake: "nice! for work?"'
            ),
            expected="FACT:",
            max_tokens=60,
        ),
        TestCase(
            name="pp_multi_fact",
            suite="postprocess",
            system=sys,
            user=(
                'Sarah: "my sister works at Google in NYC"\n'
                'Mike: "thats cool"'
            ),
            expected="FACT:",
            max_tokens=80,
        ),
        TestCase(
            name="pp_coref",
            suite="postprocess",
            system=sys,
            user=(
                'Alex: "Jake just got a job at Tesla"\n'
                'Sarah: "where is he moving?"\n'
                'Alex: "Portland"'
            ),
            expected="FACT:",
            max_tokens=80,
        ),
        TestCase(
            name="pp_none",
            suite="postprocess",
            system=sys,
            user=(
                'Alex: "lol"\n'
                'Jake: "haha yeah"\n'
                'Alex: "ikr"'
            ),
            expected="none",
            max_tokens=30,
        ),
        TestCase(
            name="pp_health",
            suite="postprocess",
            system=sys,
            user='Jake: "I found out I cant eat gluten, doctor says its celiac"',
            expected="FACT:",
            max_tokens=60,
        ),
        TestCase(
            name="pp_plans",
            suite="postprocess",
            system=sys,
            user=(
                'Alex: "lets go hiking at Mt Tam Saturday"\n'
                'Jake: "sure, 8am?"\n'
                'Alex: "yeah I will drive"'
            ),
            expected="FACT:",
            max_tokens=80,
        ),
    ]


def suite_depronoun() -> list[TestCase]:
    """Test if replacing pronouns with names/placeholders improves extraction."""
    sys = (
        "You read text messages and extract facts about the people mentioned. "
        "Output one fact per line as: subject - type: value. "
        "If no facts, say 'none'."
    )
    return [
        # ─── Pair 1: "I" vs named ─────────────────────────────────────
        TestCase(
            name="pronoun_I_location",
            suite="depronoun",
            system=sys,
            user='Message from unknown sender: "I went to California last week"',
            expected="California",
            max_tokens=80,
        ),
        TestCase(
            name="named_location",
            suite="depronoun",
            system=sys,
            user='Message from Alex: "Alex went to California last week"',
            expected="California",
            max_tokens=80,
        ),
        TestCase(
            name="placeholder_location",
            suite="depronoun",
            system=sys,
            user='[Person A] went to California last week',
            expected="California",
            max_tokens=80,
        ),

        # ─── Pair 2: "I" vs named - job ──────────────────────────────
        TestCase(
            name="pronoun_I_job",
            suite="depronoun",
            system=sys,
            user='Message from unknown sender: "I just started working at Tesla"',
            expected="Tesla",
            max_tokens=80,
        ),
        TestCase(
            name="named_job",
            suite="depronoun",
            system=sys,
            user='Message from Sarah: "Sarah just started working at Tesla"',
            expected="Tesla",
            max_tokens=80,
        ),
        TestCase(
            name="placeholder_job",
            suite="depronoun",
            system=sys,
            user='[Person A] just started working at Tesla',
            expected="Tesla",
            max_tokens=80,
        ),

        # ─── Pair 3: "my" vs named relationship ─────────────────────
        TestCase(
            name="pronoun_my_sister",
            suite="depronoun",
            system=sys,
            user='Message from unknown sender: "my sister works at Google"',
            expected="Google",
            max_tokens=80,
        ),
        TestCase(
            name="named_sister",
            suite="depronoun",
            system=sys,
            user="Message from Alex: \"Alex's sister works at Google\"",
            expected="Google",
            max_tokens=80,
        ),

        # ─── Pair 4: "I" vs named - preference ──────────────────────
        TestCase(
            name="pronoun_I_pref",
            suite="depronoun",
            system=sys,
            user='Message from unknown sender: "I cant eat gluten"',
            expected="gluten",
            max_tokens=80,
        ),
        TestCase(
            name="named_pref",
            suite="depronoun",
            system=sys,
            user='Message from Jake: "Jake cant eat gluten"',
            expected="gluten",
            max_tokens=80,
        ),

        # ─── Pair 5: multi-turn with pronouns vs names ──────────────
        TestCase(
            name="pronoun_multiturn",
            suite="depronoun",
            system=sys,
            user=(
                'Conversation:\n'
                'Person A: "I\'m heading to Austin this weekend"\n'
                'Person B: "nice! what are you doing there?"\n\n'
                'Extract facts about Person A from this conversation.'
            ),
            expected="Austin",
            max_tokens=100,
        ),
        TestCase(
            name="named_multiturn",
            suite="depronoun",
            system=sys,
            user=(
                'Conversation:\n'
                'Alex: "Alex is heading to Austin this weekend"\n'
                'Sarah: "nice! what is Alex doing there?"\n\n'
                'Extract facts about Alex from this conversation.'
            ),
            expected="Austin",
            max_tokens=100,
        ),

        # ─── Pair 6: "we" vs named ──────────────────────────────────
        TestCase(
            name="pronoun_we",
            suite="depronoun",
            system=sys,
            user='Message from unknown sender: "we just signed a lease in Denver"',
            expected="Denver",
            max_tokens=80,
        ),
        TestCase(
            name="named_we",
            suite="depronoun",
            system=sys,
            user='Message from Alex: "Alex and roommate just signed a lease in Denver"',
            expected="Denver",
            max_tokens=80,
        ),

        # ─── Gating: should still say none ───────────────────────────
        TestCase(
            name="pronoun_none",
            suite="depronoun",
            system=sys,
            user='Message from unknown sender: "lol ok"',
            expected="none",
            max_tokens=30,
        ),
        TestCase(
            name="named_none",
            suite="depronoun",
            system=sys,
            user='Message from Alex: "lol ok"',
            expected="none",
            max_tokens=30,
        ),
    ]


def suite_understanding() -> list[TestCase]:
    """Basic message understanding: can it grasp what someone is saying?"""
    sys = (
        "You read text messages and extract what the person is doing, planning, or sharing. "
        "Output one fact per line as: type: value. Types: plan, location, time, person, "
        "preference, activity, health, work. If nothing, say 'none'."
    )
    return [
        # Simple, explicit statements - should be easy
        TestCase(
            name="going_somewhere",
            suite="understanding",
            system=sys,
            user='Message: "I went to California last week"',
            expected="California",
            max_tokens=80,
        ),
        TestCase(
            name="meeting_time",
            suite="understanding",
            system=sys,
            user='Message: "lets meet at 3"',
            expected="3",
            max_tokens=80,
        ),
        TestCase(
            name="food_preference",
            suite="understanding",
            system=sys,
            user='Message: "I love sushi"',
            expected="sushi",
            max_tokens=80,
        ),
        TestCase(
            name="job_mention",
            suite="understanding",
            system=sys,
            user='Message: "just started my new job at Apple"',
            expected="Apple",
            max_tokens=80,
        ),
        TestCase(
            name="health_info",
            suite="understanding",
            system=sys,
            user='Message: "I have a doctors appointment tomorrow"',
            expected="doctor",
            max_tokens=80,
        ),
        TestCase(
            name="activity_plan",
            suite="understanding",
            system=sys,
            user='Message: "gonna go hiking this weekend"',
            expected="hiking",
            max_tokens=80,
        ),
        # Slightly harder - informal/abbreviated
        TestCase(
            name="informal_plan",
            suite="understanding",
            system=sys,
            user='Message: "omw to the gym rn"',
            expected="gym",
            max_tokens=80,
        ),
        TestCase(
            name="implicit_location",
            suite="understanding",
            system=sys,
            user='Message: "just landed in NYC!"',
            expected="NYC",
            max_tokens=80,
        ),
        TestCase(
            name="dinner_plan",
            suite="understanding",
            system=sys,
            user='Message: "dinner at 7 tonight?"',
            expected="7",
            max_tokens=80,
        ),
        # Should return none - no real facts
        TestCase(
            name="empty_message",
            suite="understanding",
            system=sys,
            user='Message: "lol"',
            expected="none",
            max_tokens=30,
        ),
        TestCase(
            name="empty_greeting",
            suite="understanding",
            system=sys,
            user='Message: "hey whats up"',
            expected="none",
            max_tokens=30,
        ),
        TestCase(
            name="empty_reaction",
            suite="understanding",
            system=sys,
            user='Message: "haha thats funny"',
            expected="none",
            max_tokens=30,
        ),
        # Harder - requires some comprehension
        TestCase(
            name="negative_preference",
            suite="understanding",
            system=sys,
            user='Message: "I cant eat gluten"',
            expected="gluten",
            max_tokens=80,
        ),
        TestCase(
            name="relationship",
            suite="understanding",
            system=sys,
            user='Message: "my brother just graduated from Stanford"',
            expected="brother",
            max_tokens=80,
        ),
        TestCase(
            name="moving",
            suite="understanding",
            system=sys,
            user='Message: "we just signed a lease in Denver"',
            expected="Denver",
            max_tokens=80,
        ),
    ]


def suite_multiturn() -> list[TestCase]:
    """Multi-turn understanding: can it use conversation context?"""
    sys = (
        "You read a conversation between two people. Extract facts about the people "
        "from the LAST message, using the earlier messages as context. "
        "Output one fact per line as: type: value. If nothing, say 'none'."
    )

    def convo(messages: list[str]) -> str:
        """Format a multi-message conversation."""
        lines = []
        for i, msg in enumerate(messages):
            speaker = "Person A" if i % 2 == 0 else "Person B"
            lines.append(f"{speaker}: {msg}")
        return "Conversation:\n" + "\n".join(lines) + "\n\nFacts from the last message:"

    return [
        # Context needed to understand "there"
        TestCase(
            name="there_reference",
            suite="multiturn",
            system=sys,
            user=convo([
                "I'm heading to Austin this weekend",
                "nice! what are you doing there?",
            ]),
            expected="Austin",
            max_tokens=100,
        ),
        # Context needed to understand "she"
        TestCase(
            name="she_reference",
            suite="multiturn",
            system=sys,
            user=convo([
                "My sister Sarah just got promoted",
                "she works at Google right?",
            ]),
            expected="Google",
            max_tokens=100,
        ),
        # Context needed to understand "it"
        TestCase(
            name="it_reference",
            suite="multiturn",
            system=sys,
            user=convo([
                "have you tried the new Thai place on Main St?",
                "yeah it was amazing, best pad thai ever",
            ]),
            expected="Thai",
            max_tokens=100,
        ),
        # Time reference from context
        TestCase(
            name="time_context",
            suite="multiturn",
            system=sys,
            user=convo([
                "want to grab coffee tomorrow?",
                "sure what time",
                "how about 2pm at Starbucks",
            ]),
            expected="2pm",
            max_tokens=100,
        ),
        # No new facts in last message
        TestCase(
            name="no_new_facts",
            suite="multiturn",
            system=sys,
            user=convo([
                "I just moved to Seattle",
                "oh cool!",
            ]),
            expected="none",
            max_tokens=50,
        ),
        # Last message adds to earlier context
        TestCase(
            name="additive_context",
            suite="multiturn",
            system=sys,
            user=convo([
                "I got a new job",
                "congrats! where?",
                "Tesla, starting next Monday",
            ]),
            expected="Tesla",
            max_tokens=100,
        ),
        # Pronoun resolution across 3 turns
        TestCase(
            name="pronoun_chain",
            suite="multiturn",
            system=sys,
            user=convo([
                "Do you know Jake?",
                "yeah he's my roommate",
                "he just told me he's moving to Portland",
            ]),
            expected="Portland",
            max_tokens=100,
        ),
        # Plans that span messages
        TestCase(
            name="plan_across_turns",
            suite="multiturn",
            system=sys,
            user=convo([
                "lets do something Saturday",
                "hiking?",
                "yeah! Mt Tam, lets leave at 8am",
            ]),
            expected="Mt Tam",
            max_tokens=100,
        ),
    ]


def suite_coreference() -> list[TestCase]:
    """Coreference resolution: can it figure out who/what pronouns refer to?"""
    sys = (
        "Read the text and answer the question. Give a short, direct answer."
    )
    return [
        TestCase(
            name="he_simple",
            suite="coreference",
            system=sys,
            user='"Jake is a doctor. He works in Boston." Who works in Boston?',
            expected="Jake",
            max_tokens=20,
        ),
        TestCase(
            name="she_simple",
            suite="coreference",
            system=sys,
            user='"My sister lives in Denver. She loves it there." Who lives in Denver?',
            expected="sister",
            max_tokens=20,
        ),
        TestCase(
            name="it_place",
            suite="coreference",
            system=sys,
            user='"I tried the new Thai place. It was amazing." What was amazing?',
            expected="Thai",
            max_tokens=20,
        ),
        TestCase(
            name="there_location",
            suite="coreference",
            system=sys,
            user='"I went to California. I loved it there." Where is "there"?',
            expected="California",
            max_tokens=20,
        ),
        TestCase(
            name="they_people",
            suite="coreference",
            system=sys,
            user='"Mom and Dad are visiting. They arrive Friday." Who arrives Friday?',
            expected="Mom",
            max_tokens=20,
        ),
        TestCase(
            name="two_people",
            suite="coreference",
            system=sys,
            user='"Sarah and Mike went to dinner. She ordered pasta." Who ordered pasta?',
            expected="Sarah",
            max_tokens=20,
        ),
        TestCase(
            name="possessive",
            suite="coreference",
            system=sys,
            user='"My friend Jake got a dog. His dog is named Rex." Whose dog is Rex?',
            expected="Jake",
            max_tokens=20,
        ),
        TestCase(
            name="that_thing",
            suite="coreference",
            system=sys,
            user='"I applied to Google. I really hope I get that job." What job?',
            expected="Google",
            max_tokens=20,
        ),
    ]


def suite_fim() -> list[TestCase]:
    """FIM (Fill-in-the-Middle): constrained generation with prefix + suffix anchoring."""
    return [
        # ─── Basic FIM: does it even work? ───────────────────────────────
        TestCase(
            name="fim_simple_fill",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix="The capital of France is ",
            fim_suffix=". It is known for the Eiffel Tower.",
            expected="Paris",
            max_tokens=20,
        ),
        TestCase(
            name="fim_number_fill",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix="A spider has ",
            fim_suffix=" legs and belongs to the class Arachnida.",
            expected="8",
            max_tokens=10,
        ),

        # ─── Extraction with FIM: constrained fact extraction ────────────
        TestCase(
            name="fim_extract_location",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Message: "I just moved to Austin last week"\n'
                'Facts:\n- location: '
            ),
            fim_suffix="\n- source: text message",
            expected="Austin",
            max_tokens=20,
        ),
        TestCase(
            name="fim_extract_person_job",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Message: "My sister works at Google"\n'
                'Facts:\n- relationship: sister\n- employer: '
            ),
            fim_suffix="\n- source: text message",
            expected="Google",
            max_tokens=20,
        ),
        TestCase(
            name="fim_extract_health",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Message: "I\'m allergic to peanuts so I can\'t eat that"\n'
                'Facts:\n- health: '
            ),
            fim_suffix="\n- source: text message",
            expected="peanut",
            max_tokens=30,
        ),

        # ─── Binary gating with FIM ─────────────────────────────────────
        TestCase(
            name="fim_gate_yes",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Does this message contain personal facts?\n'
                'Message: "I just started working at Tesla"\n'
                'Answer: '
            ),
            fim_suffix="\nIf yes, extract them below.",
            expected="yes",
            max_tokens=10,
        ),
        TestCase(
            name="fim_gate_no",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Does this message contain personal facts?\n'
                'Message: "lol ok"\n'
                'Answer: '
            ),
            fim_suffix="\nEnd of analysis.",
            expected="no",
            max_tokens=10,
        ),

        # ─── Form filling with FIM ──────────────────────────────────────
        TestCase(
            name="fim_form_fill",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Message: "My friend Jake is a doctor in Boston"\n\n'
                'Person: '
            ),
            fim_suffix=(
                '\nOccupation: doctor\n'
                'Location: Boston\n'
                'Relationship: friend'
            ),
            expected="Jake",
            max_tokens=20,
        ),
        TestCase(
            name="fim_form_occupation",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Message: "just started my new job at Apple"\n\n'
                'Has job info: yes\n'
                'Employer: '
            ),
            fim_suffix="\nStart date: recent\nSource: text message",
            expected="Apple",
            max_tokens=20,
        ),

        # ─── Coreference with FIM ───────────────────────────────────────
        TestCase(
            name="fim_coref_she",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Context: "My sister Sarah lives in Denver. She loves it there."\n'
                'Question: Who lives in Denver?\n'
                'Answer: '
            ),
            fim_suffix="\nConfidence: high",
            expected="Sarah",
            max_tokens=20,
        ),
        TestCase(
            name="fim_coref_there",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Context: "I went to California. I loved it there."\n'
                'Question: Where is "there"?\n'
                'Answer: '
            ),
            fim_suffix="\nConfidence: high",
            expected="California",
            max_tokens=20,
        ),

        # ─── Multi-turn context with FIM ─────────────────────────────────
        TestCase(
            name="fim_multiturn_where",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Person A: "I\'m heading to Austin this weekend"\n'
                'Person B: "nice! what are you doing there?"\n\n'
                'Where is "there"? Answer: '
            ),
            fim_suffix="\nFact: Person A is going to Austin.",
            expected="Austin",
            max_tokens=20,
        ),
        TestCase(
            name="fim_multiturn_who",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Person A: "My sister Sarah just got promoted"\n'
                'Person B: "she works at Google right?"\n\n'
                'Who works at Google? Answer: '
            ),
            fim_suffix="\nFact: Sarah works at Google.",
            expected="Sarah",
            max_tokens=20,
        ),

        # ─── Reply drafting with FIM ─────────────────────────────────────
        TestCase(
            name="fim_reply_accept",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Friend: "Want to grab dinner tonight at 7?"\n'
                'You: "'
            ),
            fim_suffix=(
                '"\nFriend: "Great, see you at 7!"'
            ),
            max_tokens=30,
        ),
        TestCase(
            name="fim_reply_decline",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Friend: "Want to go hiking Saturday?"\n'
                'You: "'
            ),
            fim_suffix=(
                '"\nFriend: "No worries, maybe next time!"'
            ),
            max_tokens=30,
        ),

        # ─── Structured JSON with FIM ────────────────────────────────────
        TestCase(
            name="fim_json_constrained",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Message: "My brother Jake lives in Seattle"\n'
                '{"facts": [{"type": "relationship", "value": "brother named Jake"}, '
                '{"type": "location", "value": "'
            ),
            fim_suffix='"}]}',
            expected="Seattle",
            max_tokens=20,
        ),
        TestCase(
            name="fim_json_empty",
            suite="fim",
            system=None, user="(FIM mode)",
            fim_prefix=(
                'Message: "lol ok"\n'
                '{"facts": ['
            ),
            fim_suffix="]}",
            expected="]",
            max_tokens=10,
        ),
    ]


def suite_topics() -> list[TestCase]:
    """Topic segmentation: can it detect topic boundaries and label segments?"""

    # ─── Task 1: Detect topic boundary (same or different?) ──────────
    boundary_sys = (
        "You analyze conversations. Given the last few messages and a new message, "
        "decide if the new message is about the SAME topic or a NEW topic. "
        "Output ONLY: same or new"
    )

    # ─── Task 2: Label the topic of a segment ───────────────────────
    label_sys = (
        "You label conversation topics. Given a group of messages, output a short "
        "topic label (2-4 words). Examples: making plans, work news, health update, "
        "small talk, food preferences."
    )

    # ─── Task 3: Segment + extract in one shot ──────────────────────
    extract_sys = (
        "You read conversation segments and extract personal facts. "
        "First, state the topic in 2-3 words. "
        "Then list facts as: person - type: value. "
        "If no facts, say 'none'."
    )

    return [
        # ── Boundary detection: same topic ────────────────────────────
        TestCase(
            name="boundary_same_plans",
            suite="topics",
            system=boundary_sys,
            user=(
                'Recent messages:\n'
                'Alex: "want to grab dinner tonight?"\n'
                'Jake: "sure where"\n\n'
                'New message:\n'
                'Alex: "how about that Thai place on Main"\n\n'
                'Same topic or new topic?'
            ),
            expected="same",
            max_tokens=5,
        ),
        TestCase(
            name="boundary_same_work",
            suite="topics",
            system=boundary_sys,
            user=(
                'Recent messages:\n'
                'Sarah: "I got a promotion at work"\n'
                'Mike: "congrats! what role?"\n\n'
                'New message:\n'
                'Sarah: "senior engineer, starting next month"\n\n'
                'Same topic or new topic?'
            ),
            expected="same",
            max_tokens=5,
        ),
        TestCase(
            name="boundary_same_travel",
            suite="topics",
            system=boundary_sys,
            user=(
                'Recent messages:\n'
                'Alex: "I am heading to Austin this weekend"\n'
                'Jake: "nice what are you doing there"\n\n'
                'New message:\n'
                'Alex: "visiting my sister, she just moved there"\n\n'
                'Same topic or new topic?'
            ),
            expected="same",
            max_tokens=5,
        ),

        # ── Boundary detection: new topic ─────────────────────────────
        TestCase(
            name="boundary_new_topic_shift",
            suite="topics",
            system=boundary_sys,
            user=(
                'Recent messages:\n'
                'Alex: "that movie was so good"\n'
                'Jake: "yeah the ending was crazy"\n\n'
                'New message:\n'
                'Alex: "btw did you hear Jake got a new job"\n\n'
                'Same topic or new topic?'
            ),
            expected="new",
            max_tokens=5,
        ),
        TestCase(
            name="boundary_new_btw",
            suite="topics",
            system=boundary_sys,
            user=(
                'Recent messages:\n'
                'Sarah: "lets meet at 3pm"\n'
                'Mike: "sounds good"\n\n'
                'New message:\n'
                'Sarah: "oh also I cant eat gluten anymore"\n\n'
                'Same topic or new topic?'
            ),
            expected="new",
            max_tokens=5,
        ),
        TestCase(
            name="boundary_new_unrelated",
            suite="topics",
            system=boundary_sys,
            user=(
                'Recent messages:\n'
                'Alex: "I love this new restaurant"\n'
                'Jake: "the pasta was amazing"\n\n'
                'New message:\n'
                'Alex: "my sister just got into Stanford"\n\n'
                'Same topic or new topic?'
            ),
            expected="new",
            max_tokens=5,
        ),

        # ── Topic labeling ────────────────────────────────────────────
        TestCase(
            name="label_dinner_plans",
            suite="topics",
            system=label_sys,
            user=(
                'Messages:\n'
                'Alex: "want to grab dinner tonight?"\n'
                'Jake: "sure where"\n'
                'Alex: "how about that Thai place on Main"\n'
                'Jake: "perfect see you at 7"'
            ),
            expected="dinner",
            max_tokens=15,
        ),
        TestCase(
            name="label_job_news",
            suite="topics",
            system=label_sys,
            user=(
                'Messages:\n'
                'Sarah: "guess what I got the job!"\n'
                'Mike: "at Tesla?"\n'
                'Sarah: "yeah starting next Monday"\n'
                'Mike: "thats amazing congrats"'
            ),
            expected="job",
            max_tokens=15,
        ),
        TestCase(
            name="label_small_talk",
            suite="topics",
            system=label_sys,
            user=(
                'Messages:\n'
                'Alex: "hey"\n'
                'Jake: "whats up"\n'
                'Alex: "not much you?"\n'
                'Jake: "same lol"'
            ),
            expected="small talk",
            max_tokens=15,
        ),
        TestCase(
            name="label_health",
            suite="topics",
            system=label_sys,
            user=(
                'Messages:\n'
                'Sarah: "I have a doctors appointment tomorrow"\n'
                'Mike: "everything ok?"\n'
                'Sarah: "yeah just a checkup, but I found out I cant eat gluten"\n'
                'Mike: "oh no thats rough"'
            ),
            expected="health",
            max_tokens=15,
        ),

        # ── Segment extraction: the real test ─────────────────────────
        # Give a topic segment, extract facts with full context
        TestCase(
            name="segment_extract_trip",
            suite="topics",
            system=extract_sys,
            user=(
                'Conversation segment between Alex and Jake:\n'
                'Alex: "I am heading to Austin this weekend"\n'
                'Jake: "nice what are you doing there"\n'
                'Alex: "visiting my sister Sarah, she just moved there"\n'
                'Jake: "tell her I said hi"'
            ),
            expected="Austin",
            max_tokens=100,
        ),
        TestCase(
            name="segment_extract_job",
            suite="topics",
            system=extract_sys,
            user=(
                'Conversation segment between Sarah and Mike:\n'
                'Sarah: "guess what I got the job!"\n'
                'Mike: "at Tesla?"\n'
                'Sarah: "yeah starting next Monday"\n'
                'Mike: "thats amazing congrats"'
            ),
            expected="Tesla",
            max_tokens=100,
        ),
        TestCase(
            name="segment_extract_health",
            suite="topics",
            system=extract_sys,
            user=(
                'Conversation segment between Jake and Alex:\n'
                'Jake: "I found out I cant eat gluten"\n'
                'Alex: "oh no since when"\n'
                'Jake: "doctor told me last week, celiac disease"\n'
                'Alex: "that sucks, my cousin has that too"'
            ),
            expected="gluten",
            max_tokens=100,
        ),
        TestCase(
            name="segment_extract_none",
            suite="topics",
            system=extract_sys,
            user=(
                'Conversation segment between Alex and Jake:\n'
                'Alex: "lol"\n'
                'Jake: "haha yeah"\n'
                'Alex: "so funny"\n'
                'Jake: "ikr"'
            ),
            expected="none",
            max_tokens=50,
        ),

        # ── Segment extraction with pronoun resolution ────────────────
        TestCase(
            name="segment_coref_she",
            suite="topics",
            system=extract_sys,
            user=(
                'Conversation segment between Alex and Jake:\n'
                'Alex: "my sister Sarah just got promoted"\n'
                'Jake: "where does she work again"\n'
                'Alex: "Google, she is a senior engineer now"\n'
                'Jake: "thats awesome"'
            ),
            expected="Google",
            max_tokens=100,
        ),
        TestCase(
            name="segment_coref_there",
            suite="topics",
            system=extract_sys,
            user=(
                'Conversation segment between Sarah and Mike:\n'
                'Sarah: "Mike did you know Jake moved to Portland"\n'
                'Mike: "no way, whats he doing there"\n'
                'Sarah: "he got a job at Nike"\n'
                'Mike: "good for him"'
            ),
            expected="Portland",
            max_tokens=100,
        ),
    ]


def suite_stress() -> list[TestCase]:
    """Stress test using REAL conversation data from training_data/conversation_facts/.

    These are actual iMessage segments from the Radhika conversation with
    ground-truth facts annotated by Claude/Gemini/Kimi. Tests what the 350M
    model actually needs to do: extract facts from real messy text with
    slang, typos, reactions, and multi-message context.
    """
    sys_extract = (
        "Extract personal facts from this conversation segment.\n"
        "Output one fact per line as: FACT: [person] | [type] | [value]\n"
        "Resolve pronouns to actual names. If no facts, output: FACT: none"
    )

    return [
        # ── Segment 1: Cousin connection (relationship fact spread across msgs)
        # Ground truth: Shilpan Shah is Radhika's cousin
        TestCase(
            name="real_cousin_connection",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "also how do u know Shilpan shah? He\u2019s my cousin "\n'
                'Jwalin: "Ohhh what"\n'
                'Jwalin: "I barely know him I just met him once at my cousins place"\n'
                'Jwalin: "For his wife\u2019s bday"\n'
                'Jwalin: "I have not seen him in basically a year"\n'
                'Radhika: "That\u2019s crazy, that\u2019s so funny"'
            ),
            expected="Shilpan",
            max_tokens=150,
        ),

        # ── Segment 2: Concussion history (health facts across messages)
        # Ground truth: Jwalin got concussion from ultimate frisbee, took 5 years,
        #               got rear-ended for second concussion
        TestCase(
            name="real_concussion_history",
            suite="stress",
            system=sys_extract,
            user=(
                'Jwalin: "LOL not rlly I just got a rlly bad concussion"\n'
                'Jwalin: "And then I got another one in February "\n'
                'Radhika: "HOW DID THIS EVEN HAPPEN"\n'
                'Jwalin: "OK so the first one was ultimate frisbee took like 5 years to get better"\n'
                'Jwalin: "The second one was I got rear ended"\n'
                'Jwalin: "After I was finally better"'
            ),
            expected="concussion",
            max_tokens=200,
        ),

        # ── Segment 3: Location + lifestyle (Fresno, social life)
        # Ground truth: Radhika lives in Fresno, far from family, social butterfly
        TestCase(
            name="real_fresno_life",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "LOL I\u2019m playing. Def not an alcoholic, but I do drink from time to time. '
                'I\u2019m in Fresno, so there\u2019s not much to do except drink "\n'
                'Radhika: "YOOO IM TRYNA LEAVE"\n'
                'Radhika: "long term here is a nightmare to me. Far away from bay/socal '
                'so no proximity to fam or friends"\n'
                'Radhika: "I have a social life, but not as poppin as I\u2019d like"\n'
                'Radhika: "I am a huge social butterfly so there are days I get super lonely"'
            ),
            expected="Fresno",
            max_tokens=200,
        ),

        # ── Segment 4: Job hunting + career (multiple facts)
        # Ground truth: Jwalin unemployed, CS major, looking at Zoox/Skild,
        #               not ready for software roles
        TestCase(
            name="real_job_hunt",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "are you on the job hunt rn?"\n'
                'Jwalin: "Yep very unemployed"\n'
                'Jwalin: "Are u hiring "\n'
                'Radhika: "I was on the job hunt too a while ago, it\u2019s rough in this economy"\n'
                'Radhika: "I got a lot of buddies in software I can ask around for you"\n'
                'Jwalin: "Im not ready for software roles"\n'
                'Jwalin: "Like I wanna do it but I cant just jump into the deep end"\n'
                'Jwalin: "So just gotta start somewhere and slowly work my way up"'
            ),
            expected="unemployed",
            max_tokens=200,
        ),

        # ── Segment 5: Education + height (quick facts)
        # Ground truth: Jwalin went to UTD, CS major, 5'7"
        TestCase(
            name="real_education_height",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "UTD and UTA are notorious for being party colleges LOL. '
                'AND WOW you\u2019re a sanskaari boy. Let\u2019s get you married off"\n'
                'Radhika: "Wait you\u2019re a compsci major right?"\n'
                'Jwalin: "Also I would be helllllla impressed if u cud marry me off '
                'just graduated w bachelors unemployed konkuss "\n'
                'Radhika: "Im assuming you are \u2026. lol idk how tall you are? TELL MEEE"\n'
                'Jwalin: "5\u20197 lol"'
            ),
            expected="5'7",
            max_tokens=200,
        ),

        # ── Segment 6: Outdoor activities + limitations
        # Ground truth: Jwalin dabbles in hiking, wants skydiving/scuba,
        #               can't ski anymore due to concussion
        TestCase(
            name="real_activities",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "ARE YOU A HIKER? Thrill seeker?"\n'
                'Jwalin: "I dabble in hiking I didn\u2019t doo that much tho cuz I was driving '
                'and then park then driving to the next place"\n'
                'Jwalin: "I was exhausted"\n'
                'Jwalin: "I want to be a thrill seeker but I think the universe is telling me to stop"\n'
                'Jwalin: "I\u2019ll def go skydiving, wanna learn scuba diving but I think skiiing '
                'and snow are donezo forever now cuz konkuss"'
            ),
            expected="skydiving",
            max_tokens=200,
        ),

        # ── Segment 7: Healthcare career (Radhika's work)
        # Ground truth: Radhika works in healthcare admin, passionate about it
        TestCase(
            name="real_healthcare_career",
            suite="stress",
            system=sys_extract,
            user=(
                'Jwalin: "Ok but like what kinda jobs in healthcare just admin stuff? '
                'U wanna be working w patients d2d"\n'
                'Radhika: "It has to be healthcare. Can\u2019t do my life without it"\n'
                'Radhika: "Yep admin meeting for 3 days"'
            ),
            expected="healthcare",
            max_tokens=150,
        ),

        # ── Segment 8: Book recommendation
        # Ground truth: Radhika reading "Good to Great", Jwalin loves reading
        TestCase(
            name="real_book_rec",
            suite="stress",
            system=sys_extract,
            user=(
                'Jwalin: "Cud always read a book"\n'
                'Radhika: "I have a great book I\u2019ve been reading called \u201cgood to great\u201d"\n'
                'Radhika: "Highly recommend the read"\n'
                'Jwalin: "Omg she can read"\n'
                'Jwalin: "I love reading"\n'
                'Jwalin: "I read so much"'
            ),
            expected="good to great",
            max_tokens=150,
        ),

        # ── Segment 9: Disney job reference (indirect fact)
        # Ground truth: Rishi works at Disney, can be used as reference
        TestCase(
            name="real_disney_ref",
            suite="stress",
            system=sys_extract,
            user=(
                'Jwalin: "Are u hiring "\n'
                'Radhika: "Lmfao no, but check disney careers and use Rishi as a reference"'
            ),
            expected="Disney",
            max_tokens=100,
        ),

        # ── Segment 10: Drinking habits (preference facts)
        # Ground truth: Jwalin barely drinks, didn't start till 21
        TestCase(
            name="real_drinking_habits",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "I bet you\u2019re equally an alcoholic. You went to UTD"\n'
                'Jwalin: "I actually didn\u2019t drink till I was 21"\n'
                'Jwalin: "I barely drink tho"'
            ),
            expected="drink",
            max_tokens=100,
        ),

        # ── Segment 11: Pure banter (NO facts - should output none)
        # Ground truth: No extractable facts
        TestCase(
            name="real_banter_nofacts",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "LMAOOOOOOOOOO"\n'
                'Jwalin: "At some random balbat party"\n'
                'Radhika: "THE DANCE THE SONG"\n'
                'Jwalin: "How/why do u rmr that"\n'
                'Radhika: "omg idek"'
            ),
            expected="none",
            max_tokens=60,
        ),

        # ── Segment 12: Instagram exchange (contact info)
        # Ground truth: Jwalin's instagram is jwalin.shah
        TestCase(
            name="real_instagram",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "What\u2019s your Instagram"\n'
                'Jwalin: "Uhhh good question"\n'
                'Jwalin: "Jwalin.shah"\n'
                'Jwalin: "So original ik"'
            ),
            expected="jwalin.shah",
            max_tokens=100,
        ),

        # ── Segment 13: Reactions only (NO facts)
        # Ground truth: Tapback reactions contain no personal facts
        TestCase(
            name="real_reactions_only",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "Loved \u201cThanks ig  \u201d"\n'
                'Radhika: "Emphasized \u201cLOL not rlly I just got a rlly bad concussion \u201d"\n'
                'Radhika: "Emphasized \u201cAnd then I got another one in February \u201d"\n'
                'Radhika: "Laughed at \u201cOmg she can read \u201d"'
            ),
            expected="none",
            max_tokens=60,
        ),

        # ── Segment 14: Yosemite trip + family
        # Ground truth: Jwalin's masi coming from London, Yosemite trip Aug 16-17
        TestCase(
            name="real_yosemite_trip",
            suite="stress",
            system=sys_extract,
            user=(
                'Jwalin: "But I think we are going to Yosemite on 16th17th weekend '
                'cuz my masi is coming from London"\n'
                'Radhika: "ARE YOU A HIKER? Thrill seeker?"'
            ),
            expected="Yosemite",
            max_tokens=150,
        ),

        # ── Segment 15: Health insurance advice
        # Ground truth: Radhika knowledgeable about health insurance
        TestCase(
            name="real_health_insurance",
            suite="stress",
            system=sys_extract,
            user=(
                'Jwalin: "Greetings I have been told u are the person to talk to for healthy insurance"\n'
                'Jwalin: "Health* why wud that even autocorrect"\n'
                'Radhika: "I\u2019m so sorry it\u2019s been a whirlwind this week"\n'
                'Radhika: "Yes you should write down everything including disabilities '
                'because you will probs get more covered under a plan if you share it now"'
            ),
            expected="insurance",
            max_tokens=150,
        ),

        # ── Segment 16: Company perks (indirect work fact)
        # Ground truth: Radhika's company pays for flights
        TestCase(
            name="real_company_perks",
            suite="stress",
            system=sys_extract,
            user=(
                'Radhika: "WHY WOULD I. Company paid flight"\n'
                'Radhika: "Lemme chillaxxxx"\n'
                'Radhika: "I know, but I had a staff member drop me off LMAO"\n'
                'Radhika: "Flying from Fresno to Santa Ana"'
            ),
            expected="Fresno",
            max_tokens=150,
        ),

        # ── Segment 17: Unreplied messages (behavior fact)
        # Ground truth: Jwalin has ~50 unreplied texts since March
        TestCase(
            name="real_unreplied_msgs",
            suite="stress",
            system=sys_extract,
            user=(
                'Jwalin: "I have ppl I haven\u2019t replied to since like beginning of march "\n'
                'Jwalin: "Prolly like 50 "'
            ),
            expected="50",
            max_tokens=100,
        ),

        # ── Segment 18: Zoox/Skild job targets
        # Ground truth: Jwalin looking at Zoox for data collection, Skild for robot operator
        TestCase(
            name="real_job_targets",
            suite="stress",
            system=sys_extract,
            user=(
                'Jwalin: "Unless I get zoox for data collection or robot operator at skild ill start that"\n'
                'Radhika: "Manifesting for you "'
            ),
            expected="zoox",
            max_tokens=100,
        ),
    ]


def suite_formats() -> list[TestCase]:
    """Test different text formatting strategies on the SAME content.

    Does the model perform better with quotes, timestamps, brackets,
    plain text, or other formatting? Uses real Radhika conversation segments.
    """
    sys = (
        "Extract personal facts from this conversation segment.\n"
        "Output one fact per line as: FACT: [person] | [type] | [value]\n"
        "Resolve pronouns to actual names. If no facts, output: FACT: none"
    )

    # Same content: Radhika lives in Fresno, wants to leave
    # Format A: Speaker: "message" (current approach)
    # Format B: [Speaker] message (bracket, no quotes)
    # Format C: Speaker says: message (natural language)
    # Format D: Speaker (timestamp): message
    # Format E: Plain text, no speaker labels
    # Format F: Chat log style with >

    return [
        # ── Format A: Speaker: "message" (baseline) ─────────────────
        TestCase(
            name="fmt_a_quotes",
            suite="formats",
            system=sys,
            user=(
                'Radhika: "I\'m in Fresno, so there\'s not much to do except drink"\n'
                'Radhika: "YOOO IM TRYNA LEAVE"\n'
                'Radhika: "long term here is a nightmare to me. '
                'Far away from bay/socal so no proximity to fam or friends"'
            ),
            expected="Fresno",
            max_tokens=150,
        ),
        # ── Format B: [Speaker] message ──────────────────────────────
        TestCase(
            name="fmt_b_brackets",
            suite="formats",
            system=sys,
            user=(
                "[Radhika] I'm in Fresno, so there's not much to do except drink\n"
                "[Radhika] YOOO IM TRYNA LEAVE\n"
                "[Radhika] long term here is a nightmare to me. "
                "Far away from bay/socal so no proximity to fam or friends"
            ),
            expected="Fresno",
            max_tokens=150,
        ),
        # ── Format C: Speaker says: message ──────────────────────────
        TestCase(
            name="fmt_c_says",
            suite="formats",
            system=sys,
            user=(
                "Radhika says: I'm in Fresno, so there's not much to do except drink\n"
                "Radhika says: YOOO IM TRYNA LEAVE\n"
                "Radhika says: long term here is a nightmare to me. "
                "Far away from bay/socal so no proximity to fam or friends"
            ),
            expected="Fresno",
            max_tokens=150,
        ),
        # ── Format D: Speaker (timestamp): message ───────────────────
        TestCase(
            name="fmt_d_timestamp",
            suite="formats",
            system=sys,
            user=(
                "Radhika (Jul 20, 8:19pm): I'm in Fresno, so there's not much to do except drink\n"
                "Radhika (Jul 23, 11:33am): YOOO IM TRYNA LEAVE\n"
                "Radhika (Jul 23, 11:33am): long term here is a nightmare to me. "
                "Far away from bay/socal so no proximity to fam or friends"
            ),
            expected="Fresno",
            max_tokens=150,
        ),
        # ── Format E: Plain text, no labels ──────────────────────────
        TestCase(
            name="fmt_e_plain",
            suite="formats",
            system=sys,
            user=(
                "I'm in Fresno, so there's not much to do except drink\n"
                "YOOO IM TRYNA LEAVE\n"
                "long term here is a nightmare to me. "
                "Far away from bay/socal so no proximity to fam or friends"
            ),
            expected="Fresno",
            max_tokens=150,
        ),
        # ── Format F: Chat log with > ────────────────────────────────
        TestCase(
            name="fmt_f_chatlog",
            suite="formats",
            system=sys,
            user=(
                "> Radhika: I'm in Fresno, so there's not much to do except drink\n"
                "> Radhika: YOOO IM TRYNA LEAVE\n"
                "> Radhika: long term here is a nightmare to me. "
                "Far away from bay/socal so no proximity to fam or friends"
            ),
            expected="Fresno",
            max_tokens=150,
        ),

        # ── Same formats but with multi-speaker coreference ──────────
        # Content: Shilpan is Radhika's cousin (needs to resolve "He's")
        TestCase(
            name="fmt_a_coref",
            suite="formats",
            system=sys,
            user=(
                'Radhika: "also how do u know Shilpan shah? He\'s my cousin"\n'
                'Jwalin: "I barely know him I just met him once at my cousins place"'
            ),
            expected="cousin",
            max_tokens=100,
        ),
        TestCase(
            name="fmt_b_coref",
            suite="formats",
            system=sys,
            user=(
                "[Radhika] also how do u know Shilpan shah? He's my cousin\n"
                "[Jwalin] I barely know him I just met him once at my cousins place"
            ),
            expected="cousin",
            max_tokens=100,
        ),
        TestCase(
            name="fmt_d_coref",
            suite="formats",
            system=sys,
            user=(
                "Radhika (6:53pm): also how do u know Shilpan shah? He's my cousin\n"
                "Jwalin (6:54pm): I barely know him I just met him once at my cousins place"
            ),
            expected="cousin",
            max_tokens=100,
        ),
        TestCase(
            name="fmt_e_coref",
            suite="formats",
            system=sys,
            user=(
                "also how do u know Shilpan shah? He's my cousin\n"
                "I barely know him I just met him once at my cousins place"
            ),
            expected="cousin",
            max_tokens=100,
        ),

        # ── System prompt variations (same content, different instructions)
        # Content: concussion history
        TestCase(
            name="sys_minimal",
            suite="formats",
            system="Extract facts. One per line: FACT: person | type | value",
            user=(
                '[Jwalin] LOL not rlly I just got a rlly bad concussion\n'
                '[Jwalin] the first one was ultimate frisbee took like 5 years to get better\n'
                '[Jwalin] The second one was I got rear ended'
            ),
            expected="concussion",
            max_tokens=150,
        ),
        TestCase(
            name="sys_detailed",
            suite="formats",
            system=(
                "You are analyzing iMessage conversations to extract personal facts.\n"
                "Personal facts include: location, job, health, relationships, preferences.\n"
                "For each fact found, output: FACT: [person] | [category] | [detail]\n"
                "Resolve all pronouns (he/she/they/I) to the actual person's name.\n"
                "If no personal facts exist, output: FACT: none"
            ),
            user=(
                '[Jwalin] LOL not rlly I just got a rlly bad concussion\n'
                '[Jwalin] the first one was ultimate frisbee took like 5 years to get better\n'
                '[Jwalin] The second one was I got rear ended'
            ),
            expected="concussion",
            max_tokens=150,
        ),
        TestCase(
            name="sys_fewshot",
            suite="formats",
            system=(
                "Extract personal facts from conversations.\n\n"
                "Example:\n"
                "[Sarah] I just moved to Austin for my new job at Apple\n"
                "FACT: Sarah | location | moved to Austin\n"
                "FACT: Sarah | work | job at Apple\n\n"
                "Example:\n"
                "[Mike] lol yeah that was funny\n"
                "FACT: none\n\n"
                "Now extract facts from the following:"
            ),
            user=(
                '[Jwalin] LOL not rlly I just got a rlly bad concussion\n'
                '[Jwalin] the first one was ultimate frisbee took like 5 years to get better\n'
                '[Jwalin] The second one was I got rear ended'
            ),
            expected="concussion",
            max_tokens=150,
        ),
    ]


def suite_goldset() -> list[TestCase]:
    """Test on REAL iMessage segments pulled from chat.db with contact names resolved.

    These are actual conversation segments from multiple different chats,
    using the "Speaker says:" format that tested best. Each has real names
    from AddressBook contact resolution.
    """
    sys = (
        "Extract personal facts from this conversation segment.\n"
        "Output one fact per line as: FACT: [person] | [type] | [value]\n"
        "Resolve pronouns to actual names. If no facts, output: FACT: none"
    )

    return [
        # ── Mihir: AI company interview + parachute ──────────────────
        # Facts: Mihir has interview with AI company and Parachute
        TestCase(
            name="gs_mihir_interview",
            suite="goldset",
            system=sys,
            user=(
                "Jwalin says: Big monies\n"
                "Jwalin says: Little\n"
                "Jwalin says: Dont get lost\n"
                "Mihir says: leave ai\n"
                "Mihir says: i have interview with the ai company as well as parachute\n"
                "Mihir says: no you figure it out\n"
                "Mihir says: flight is at 7 little\n"
                "Mihir says: when do we wish to reach"
            ),
            expected="interview",
            max_tokens=150,
        ),

        # ── Mihir: Recreating esophagus (research) ───────────────────
        # Facts: Mihir is researching/recreating esophagus
        TestCase(
            name="gs_mihir_research",
            suite="goldset",
            system=sys,
            user=(
                "Mihir says: Its just for tmrw\n"
                "Jwalin says: grind\n"
                "Jwalin says: what u researching\n"
                "Mihir says: I'm trying to recreate esophagus"
            ),
            expected="esophagus",
            max_tokens=100,
        ),

        # ── Mateo: Jwalin got fired ──────────────────────────────────
        # Facts: Jwalin got fired, started a mutiny, did software stuff
        TestCase(
            name="gs_mateo_fired",
            suite="goldset",
            system=sys,
            user=(
                "Jwalin says: And other issues with my behavior\n"
                "Mateo says: Thats insane\n"
                "Mateo says: They told you to no do the extra responsibilities\n"
                "Jwalin says: I did but then I started a mutiny\n"
                "Mateo says: How is that not your job\n"
                "Jwalin says: So they fired me real quick\n"
                "Mateo says: LOL\n"
                "Jwalin says: I even clarified Tuesday so software stuff"
            ),
            expected="fired",
            max_tokens=150,
        ),

        # ── Rachit: Sunroom + job ────────────────────────────────────
        # Facts: Rachit built sunroom in backyard themselves
        TestCase(
            name="gs_rachit_sunroom",
            suite="goldset",
            system=sys,
            user=(
                "Rachit says: we built the sunroom in my backyard right\n"
                "Rachit says: like ourselves\n"
                "Rachit says: yeah for all sorts of other stuff\n"
                "Rachit says: I remember a long time ago\n"
                "Rachit says: When did you get the job\n"
                "Rachit says: Let's meet up there come over to the office"
            ),
            expected="sunroom",
            max_tokens=150,
        ),

        # ── Rachit: Termination/severance legal advice ───────────────
        # Facts: Rachit gives legal advice about termination/severance
        TestCase(
            name="gs_rachit_severance",
            suite="goldset",
            system=sys,
            user=(
                "Rachit says: any updates from today?\n"
                'Rachit says: Loved "Did acupuncture"\n'
                "Rachit says: wait, so is this confirmation? but why are they "
                "delaying sending you termination confirmation?\n"
                "Rachit says: interesting. if you're being terminated with the cause, "
                "then technically they don't have to pay you any severance"
            ),
            expected="termination",
            max_tokens=150,
        ),

        # ── Danalila: Big sister ─────────────────────────────────────
        # Facts: Someone (Danalila's daughter/relative?) is about to be a big sister
        TestCase(
            name="gs_danalila_sister",
            suite="goldset",
            system=sys,
            user=(
                "Danalila says: Yeahh\n"
                "Danalila says: She is about to be a big sister\n"
                "Jwalin says: fun times\n"
                "Jwalin says: is she excited\n"
                "Jwalin says: or no"
            ),
            expected="sister",
            max_tokens=100,
        ),

        # ── Shannon: Happy hour + Tyler ──────────────────────────────
        # Facts: Shannon invites to happy hour, Tyler is confused about rules
        TestCase(
            name="gs_shannon_work",
            suite="goldset",
            system=sys,
            user=(
                "Shannon says: I think you can join this happy hour right?\n"
                "Shannon says: Since it's not the all hands\n"
                "Shannon says: Ohhhhhhh!!\n"
                "Shannon says: But then I guess Tyler can't? He confuses me"
            ),
            expected="happy hour",
            max_tokens=100,
        ),

        # ── Shannon: Jwalin leading mutiny at work ───────────────────
        # Facts: Jwalin emailed for raises, leading "mutiny", removed from shack
        TestCase(
            name="gs_shannon_mutiny",
            suite="goldset",
            system=sys,
            user=(
                "Shannon says: What happened\n"
                "Jwalin says: I had operators email for raises\n"
                'Jwalin says: I\'m leading a "mutiny"\n'
                "Jwalin says: Take the rest of the week off\n"
                "Jwalin says: But I got removed from shack"
            ),
            expected="mutiny",
            max_tokens=150,
        ),

        # ── Faith: OSHA complaint + fired ────────────────────────────
        # Facts: Jwalin submitting OSHA complaint, got fired
        TestCase(
            name="gs_faith_osha",
            suite="goldset",
            system=sys,
            user=(
                "Faith says: Oh who is you\n"
                "Faith says: Well you're submitting an Osha complaint right? "
                "So you'll kinda be doing it anyway\n"
                "Faith says: Lol what\n"
                "Faith says: File a complaint? Yeah"
            ),
            expected="Osha",
            max_tokens=100,
        ),

        # ── Dad: Honda dealership + car shopping ─────────────────────
        # Facts: Shopping at Envision Honda of Milpitas, Dad asking about offer
        TestCase(
            name="gs_dad_car",
            suite="goldset",
            system=sys,
            user=(
                "Jwalin says: Get peanut butter\n"
                "Dad says: When you get a chance, please share full offer details "
                "including things like benefits, overtime, etc. Looking forward to start ASAP.\n"
                "Dad says: Have you created EVgo account?\n"
                "Jwalin says: Yes"
            ),
            expected="offer",
            max_tokens=120,
        ),

        # ── Soham: Zip code + car research ───────────────────────────
        # Facts: Jwalin's zip code is 75287, Dad talked to Soham
        TestCase(
            name="gs_soham_zip",
            suite="goldset",
            system=sys,
            user=(
                "Soham says: whats your zip code?\n"
                "Soham says: can you give me edit access to the sheet?\n"
                "Jwalin says: 75287\n"
                "Jwalin says: and yes one sec\n"
                "Jwalin says: did my dad talk to u?\n"
                "Soham says: yeah"
            ),
            expected="75287",
            max_tokens=100,
        ),

        # ── Sangati: Medicine pickup + pain ──────────────────────────
        # Facts: Jwalin in pain, can barely open medicine, Sangati picking up drugs
        TestCase(
            name="gs_sangati_meds",
            suite="goldset",
            system=sys,
            user=(
                "Sangati says: Do you wanna come over for lunch?\n"
                "Jwalin says: Prolly not I not tryna drive at all it actually hurts "
                "rlly bad I cud barely open my medicine\n"
                "Jwalin says: Oh speaking of when ur done w mission peak can u get "
                "some of my drugs from pacific commons target cvs\n"
                "Sangati says: Fair enuf but ya I gotcha\n"
                "Sangati says: Send me the addy"
            ),
            expected="medicine",
            max_tokens=150,
        ),

        # ── Ishani: Chicago trip to see a game ───────────────────────
        # Facts: Ishani going to see team in Chicago on Sunday
        TestCase(
            name="gs_ishani_chicago",
            suite="goldset",
            system=sys,
            user=(
                "Ishani says: Yes unfortunately they're so mid\n"
                "Jwalin says: Also our special teams has rlly locked tf in\n"
                "Ishani says: I'm gonna see them in Chicago this Sunday tho"
            ),
            expected="Chicago",
            max_tokens=100,
        ),

        # ── Rutu: New iPhone ─────────────────────────────────────────
        # Facts: Rutu got an iPhone, asks about Texas
        TestCase(
            name="gs_rutu_iphone",
            suite="goldset",
            system=sys,
            user=(
                "Jwalin says: Dang u have an iphone now\n"
                "Rutu says: yuhhh\n"
                "Rutu says: thx\n"
                "Rutu says: hows texas (?)\n"
                "Jwalin says: Look at you moving up in the world\n"
                "Jwalin says: So proud\n"
                "Jwalin says: Texas is goin"
            ),
            expected="iphone",
            max_tokens=100,
        ),

        # ── Banter only (no facts) ───────────────────────────────────
        TestCase(
            name="gs_banter_nofacts",
            suite="goldset",
            system=sys,
            user=(
                "Jwalin says: Loved an image\n"
                "Jwalin says: yes\n"
                "Jwalin says: monke\n"
                "Jwalin says: spider monke\n"
                "Jwalin says: no\n"
                "Jwalin says: im gonna spider monke u\n"
                "Jwalin says: surprise mf"
            ),
            expected="none",
            max_tokens=60,
        ),

        # ── Mateo: Pittsburgh work drama ─────────────────────────────
        # Facts: Aditya and Cooper blamed Jwalin, Mateo wants to leave
        TestCase(
            name="gs_mateo_blame",
            suite="goldset",
            system=sys,
            user=(
                "Mateo says: Or maybe they can just go back to pittsburgh\n"
                "Jwalin says: lol u think it isn't a shit show over there\n"
                "Mateo says: Dude someones got to tell him\n"
                "Jwalin says: You've seen their data\n"
                "Mateo says: Yeah they just dont care anymore. "
                "If all the leadership moved from here i wouldnt either\n"
                "Jwalin says: Aditya and cooper 100% just blamed everything on me\n"
                "Mateo says: I need to claw my way out of here gdi\n"
                "Mateo says: Oh for sure"
            ),
            expected="Aditya",
            max_tokens=150,
        ),
    ]


SUITES = {
    "basics": suite_basics,
    "json": suite_json,
    "classification": suite_classification,
    "extraction": suite_extraction,
    "tools": suite_tools,
    "reasoning": suite_reasoning,
    "constrained": suite_constrained,
    "gate": suite_gate,
    "gate2": suite_gate2,
    "postprocess": suite_postprocess,
    "depronoun": suite_depronoun,
    "understanding": suite_understanding,
    "multiturn": suite_multiturn,
    "coreference": suite_coreference,
    "fim": suite_fim,
    "topics": suite_topics,
    "stress": suite_stress,
    "formats": suite_formats,
    "goldset": suite_goldset,
}


# ─── Model Loading ──────────────────────────────────────────────────────────

_loaded_model = None
_loaded_tokenizer = None
_loaded_name = None


def get_model(model_name: str):
    """Load model (cached)."""
    global _loaded_model, _loaded_tokenizer, _loaded_name

    if _loaded_name == model_name:
        return _loaded_model, _loaded_tokenizer

    # Unload previous
    if _loaded_model is not None:
        import gc
        import mlx.core as mx
        del _loaded_model, _loaded_tokenizer
        gc.collect()
        mx.clear_cache()

    import mlx.core as mx
    from mlx_lm import load

    mx.set_memory_limit(1 * 1024 * 1024 * 1024)
    mx.set_cache_limit(512 * 1024 * 1024)

    path = MODELS[model_name]
    print(f"\nLoading {model_name} ({path})...", flush=True)
    t0 = time.time()
    _loaded_model, _loaded_tokenizer = load(path)
    print(f"Loaded in {time.time() - t0:.1f}s\n", flush=True)
    _loaded_name = model_name
    return _loaded_model, _loaded_tokenizer


def generate(
    model, tokenizer, test: TestCase,
    temperature: float = 0.1,
    debug: bool = False,
) -> tuple[str, float, int]:
    """Generate response for a test case.

    Uses Liquid AI's recommended settings:
      temperature=0.1, top_k=50, top_p=0.1, repetition_penalty=1.05
    """
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

    # FIM mode: bypass chat template entirely
    if test.fim_prefix is not None and test.fim_suffix is not None:
        prompt = (
            f"<|fim_pre|>{test.fim_prefix}"
            f"<|fim_suf|>{test.fim_suffix}"
            f"<|fim_mid|>"
        )
    else:
        messages = []
        if test.system:
            messages.append({"role": "system", "content": test.system})
        messages.append({"role": "user", "content": test.user})

        # Apply chat template with tools if specified
        try:
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if test.tools:
                kwargs["tools"] = test.tools
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as e:
            if debug:
                print(f"    [DEBUG] chat template failed: {e}", flush=True)
            prompt = test.user

        # Seed the assistant response if prefix given
        if test.assistant_prefix:
            prompt = prompt + test.assistant_prefix

    if debug:
        print(f"    [DEBUG] Raw prompt:\n{'─'*40}", flush=True)
        print(prompt, flush=True)
        print(f"{'─'*40}", flush=True)

    # Liquid AI recommended: temp=0.1, top_k=50, top_p=0.1, repetition_penalty=1.05
    sampler = make_sampler(temp=temperature, top_p=0.1, top_k=50)
    repetition_penalty = make_repetition_penalty(1.05)

    logits_processors = [repetition_penalty]

    # Add logit bias if specified
    if test.logit_bias:
        import mlx.core as mx

        bias_map = test.logit_bias

        def apply_logit_bias(tokens: mx.array, logits: mx.array) -> mx.array:
            for token_id, bias in bias_map.items():
                if bias != 0.0:
                    logits = logits.at[..., token_id].add(bias)
            return logits

        logits_processors.append(apply_logit_bias)

    t0 = time.time()
    response = mlx_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=test.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
    )
    elapsed_ms = (time.time() - t0) * 1000
    token_count = len(tokenizer.encode(response)) if response else 0

    # If we seeded a prefix, prepend it to the response for clarity
    if test.assistant_prefix:
        response = test.assistant_prefix + response
    # For FIM, show just the infilled content (don't prepend prefix)
    # The response IS the middle content

    return response, elapsed_ms, token_count


# ─── Evaluation ─────────────────────────────────────────────────────────────

def check_result(test: TestCase, response: str) -> bool | None:
    """Auto-check if response matches expected. None = needs manual review."""
    if test.expected is None:
        return None

    resp_lower = response.strip().lower()
    expected_lower = test.expected.lower()

    # Check if expected substring appears in response
    if expected_lower in resp_lower:
        return True

    # For very short expected values, check first word
    if len(test.expected.split()) == 1:
        first_word = resp_lower.split()[0] if resp_lower.split() else ""
        if expected_lower in first_word:
            return True

    return False


# ─── Runners ────────────────────────────────────────────────────────────────

def run_suite(
    model_name: str, suite_name: str, temperature: float = 0.1,
    debug: bool = False,
) -> list[TestResult]:
    """Run a single test suite."""
    model, tokenizer = get_model(model_name)
    tests = SUITES[suite_name]()

    print(f"{'─'*60}", flush=True)
    print(f"  Suite: {suite_name} ({len(tests)} tests) | Model: {model_name}", flush=True)
    print(f"{'─'*60}", flush=True)

    results = []
    for i, test in enumerate(tests):
        print(f"\n  [{i+1}/{len(tests)}] {test.name}", flush=True)
        if test.system:
            print(f"    System: {test.system[:80]}...", flush=True)
        user_preview = test.user[:80].encode("utf-8", errors="replace").decode("utf-8")
        print(f"    User: {user_preview}{'...' if len(test.user) > 80 else ''}", flush=True)
        if test.tools:
            print(f"    Tools: {[t['function']['name'] for t in test.tools]}", flush=True)
        if test.assistant_prefix:
            print(f"    Assistant prefix: {test.assistant_prefix!r}", flush=True)
        if test.fim_prefix is not None:
            pre_preview = test.fim_prefix[-60:] if len(test.fim_prefix) > 60 else test.fim_prefix
            suf_preview = test.fim_suffix[:60] if test.fim_suffix and len(test.fim_suffix) > 60 else test.fim_suffix
            print(f"    FIM prefix: ...{pre_preview!r}", flush=True)
            print(f"    FIM suffix: {suf_preview!r}...", flush=True)

        response, time_ms, tokens = generate(
            model, tokenizer, test, temperature, debug=debug,
        )
        passed = check_result(test, response)

        status = "PASS" if passed is True else ("FAIL" if passed is False else "REVIEW")
        print(f"    Response ({time_ms:.0f}ms, {tokens}tok) [{status}]:", flush=True)
        for line in response.strip().split("\n")[:5]:
            print(f"      {line}", flush=True)
        if test.expected:
            print(f"    Expected: {test.expected}", flush=True)

        results.append(TestResult(
            test=test.name, suite=suite_name, model=model_name,
            prompt=test.user, response=response,
            time_ms=time_ms, tokens=tokens,
            expected=test.expected, passed=passed,
        ))

    return results


def run_all_suites(
    model_name: str, suite_names: list[str], temperature: float = 0.1,
    debug: bool = False,
) -> list[TestResult]:
    """Run multiple suites and print summary."""
    all_results = []
    for suite_name in suite_names:
        results = run_suite(model_name, suite_name, temperature, debug=debug)
        all_results.extend(results)

    print_summary(all_results)
    return all_results


def run_compare(
    suite_names: list[str], temperature: float = 0.1,
    model_names: list[str] | None = None,
) -> None:
    """Run the same suites on multiple models and compare."""
    models = model_names or ["lfm-350m-extract", "lfm-1.2b-extract"]
    all_results: dict[str, list[TestResult]] = {}

    for model_name in models:
        results = []
        for suite_name in suite_names:
            results.extend(run_suite(model_name, suite_name, temperature))
        all_results[model_name] = results

    # Print comparison
    print(f"\n{'='*70}", flush=True)
    print(f"COMPARISON: {' vs '.join(models)}", flush=True)
    print(f"{'='*70}", flush=True)

    # Build column headers from model names (abbreviated)
    short_names = [m.replace("lfm-", "").replace("-extract", "X") for m in models]

    for suite_name in suite_names:
        print(f"\n  Suite: {suite_name}", flush=True)
        header = f"  {'Test':<25}"
        for sn in short_names:
            header += f" {sn:>10} {sn+'ms':>8}"
        print(header, flush=True)
        print(f"  {'─'*(25 + len(models) * 20)}", flush=True)

        suite_results = {
            m: [r for r in all_results[m] if r.suite == suite_name]
            for m in models
        }

        # Iterate over tests (zip across all models)
        test_lists = [suite_results[m] for m in models]
        for row in zip(*test_lists):
            line = f"  {row[0].test:<25}"
            for r in row:
                s = "PASS" if r.passed is True else ("FAIL" if r.passed is False else " -- ")
                line += f" {s:>10} {r.time_ms:>7.0f}ms"
            print(line, flush=True)

    # Overall stats
    for model_name in models:
        results = all_results[model_name]
        passed = sum(1 for r in results if r.passed is True)
        failed = sum(1 for r in results if r.passed is False)
        review = sum(1 for r in results if r.passed is None)
        avg_ms = sum(r.time_ms for r in results) / len(results) if results else 0
        print(f"\n  {model_name}: {passed} pass, {failed} fail, {review} review, avg {avg_ms:.0f}ms", flush=True)


def run_interactive(model_name: str) -> None:
    """Interactive REPL for poking at the model."""
    model, tokenizer = get_model(model_name)

    print(f"\n{'='*60}", flush=True)
    print(f"Interactive mode - {model_name}", flush=True)
    print(f"Commands:", flush=True)
    print(f"  /system <text>     Set system prompt", flush=True)
    print(f"  /clear             Clear system prompt", flush=True)
    print(f"  /temp <float>      Set temperature (default 0.0)", flush=True)
    print(f"  /tokens <int>      Set max tokens (default 200)", flush=True)
    print(f"  /tools             Enable sample tools", flush=True)
    print(f"  /notools           Disable tools", flush=True)
    print(f"  /raw               Show raw prompt sent to model", flush=True)
    print(f"  /suite <name>      Run a test suite", flush=True)
    print(f"  /quit              Exit", flush=True)
    print(f"{'='*60}\n", flush=True)

    system_prompt = None
    temperature = 0.1
    max_tokens = 200
    tools = None
    show_raw = False

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!", flush=True)
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split(maxsplit=1)
            command = cmd[0].lower()
            arg = cmd[1] if len(cmd) > 1 else ""

            if command == "/quit":
                break
            elif command == "/system":
                system_prompt = arg if arg else None
                print(f"  System prompt: {system_prompt}", flush=True)
            elif command == "/clear":
                system_prompt = None
                print("  System prompt cleared.", flush=True)
            elif command == "/temp":
                temperature = float(arg) if arg else 0.0
                print(f"  Temperature: {temperature}", flush=True)
            elif command == "/tokens":
                max_tokens = int(arg) if arg else 200
                print(f"  Max tokens: {max_tokens}", flush=True)
            elif command == "/tools":
                tools = suite_tools()[0].tools  # grab sample tools
                print(f"  Tools enabled: {[t['function']['name'] for t in tools]}", flush=True)
            elif command == "/notools":
                tools = None
                print("  Tools disabled.", flush=True)
            elif command == "/raw":
                show_raw = not show_raw
                print(f"  Show raw prompt: {show_raw}", flush=True)
            elif command == "/suite":
                if arg in SUITES:
                    run_suite(model_name, arg, temperature)
                else:
                    print(f"  Unknown suite. Available: {', '.join(SUITES.keys())}", flush=True)
            else:
                print(f"  Unknown command: {command}", flush=True)
            continue

        test = TestCase(
            name="interactive",
            suite="interactive",
            system=system_prompt,
            user=user_input,
            tools=tools,
            max_tokens=max_tokens,
        )

        if show_raw:
            messages = []
            if test.system:
                messages.append({"role": "system", "content": test.system})
            messages.append({"role": "user", "content": test.user})
            try:
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                if test.tools:
                    kwargs["tools"] = test.tools
                raw = tokenizer.apply_chat_template(messages, **kwargs)
                print(f"\n--- RAW PROMPT ---\n{raw}\n--- END ---\n", flush=True)
            except Exception as e:
                print(f"  (raw prompt error: {e})", flush=True)

        response, time_ms, tokens = generate(model, tokenizer, test, temperature)
        print(f"\nmodel ({time_ms:.0f}ms, {tokens}tok)> {response}\n", flush=True)


# ─── Output ─────────────────────────────────────────────────────────────────

def print_summary(results: list[TestResult]) -> None:
    """Print a summary table."""
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    by_suite: dict[str, list[TestResult]] = {}
    for r in results:
        by_suite.setdefault(r.suite, []).append(r)

    total_pass, total_fail, total_review = 0, 0, 0

    for suite_name, suite_results in by_suite.items():
        passed = sum(1 for r in suite_results if r.passed is True)
        failed = sum(1 for r in suite_results if r.passed is False)
        review = sum(1 for r in suite_results if r.passed is None)
        avg_ms = sum(r.time_ms for r in suite_results) / len(suite_results)

        total_pass += passed
        total_fail += failed
        total_review += review

        print(f"\n  {suite_name}:", flush=True)
        print(f"    {passed} pass, {failed} fail, {review} review | avg {avg_ms:.0f}ms", flush=True)

        for r in suite_results:
            status = "PASS" if r.passed is True else ("FAIL" if r.passed is False else " -- ")
            print(f"    [{status}] {r.test:<30} {r.time_ms:>6.0f}ms", flush=True)

    total = len(results)
    print(f"\n  TOTAL: {total_pass}/{total_pass + total_fail} auto-checks passed, "
          f"{total_review} need review, avg {sum(r.time_ms for r in results)/total:.0f}ms", flush=True)


def save_results(results: list[TestResult], output_path: str) -> None:
    """Save results to JSON for later analysis."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}", flush=True)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Explore small MLX model capabilities")
    parser.add_argument(
        "--model", type=str, default="lfm-350m-extract",
        help=f"Model name. Available: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--suite", type=str, default=None,
        help=f"Comma-separated suites to run. Available: {', '.join(SUITES.keys())}, all",
    )
    parser.add_argument("--compare", action="store_true", help="Compare 350m vs 1.2b")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature (default 0.0)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--debug", action="store_true", help="Show raw prompts sent to model")
    args = parser.parse_args()

    if args.model not in MODELS:
        print(f"Unknown model: {args.model}. Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    if args.suite:
        suite_names = list(SUITES.keys()) if args.suite == "all" else args.suite.split(",")
        for s in suite_names:
            if s not in SUITES:
                print(f"Unknown suite: {s}. Available: {', '.join(SUITES.keys())}")
                sys.exit(1)

        if args.compare:
            # Use --model to specify which models to compare
            # Default: extract vs base at same size
            compare_models = None
            if args.model != "lfm-350m-extract":
                # User specified a model, compare it with its counterpart
                compare_models = args.model.split(",") if "," in args.model else None
            run_compare(suite_names, args.temp, model_names=compare_models)
        else:
            results = run_all_suites(args.model, suite_names, args.temp, debug=args.debug)
            if args.output:
                save_results(results, args.output)
    else:
        # Interactive mode
        run_interactive(args.model)


if __name__ == "__main__":
    main()
