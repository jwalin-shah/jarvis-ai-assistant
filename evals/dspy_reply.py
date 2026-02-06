"""DSPy signature, module, metric, and training data for reply generation.

Defines the optimization target that DSPy compiles against:
- ReplySignature: typed I/O contract
- ReplyModule: thin wrapper around dspy.Predict
- cerebras_judge_metric: LLM-as-judge scorer (same rubric as batch_eval)
- TRAIN_EXAMPLES: dspy.Example dataset derived from batch_eval TEST_CASES
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import dspy

# ---------------------------------------------------------------------------
# Load .env for CEREBRAS_API_KEY (same as batch_eval)
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
JUDGE_MODEL = "qwen-3-235b-a22b-instruct-2507"


# ---------------------------------------------------------------------------
# Signature + Module
# ---------------------------------------------------------------------------
class ReplySignature(dspy.Signature):
    """Generate a natural text message reply."""

    context: str = dspy.InputField(desc="Conversation history with timestamps")
    last_message: str = dspy.InputField(desc="The message to reply to")
    tone: str = dspy.InputField(desc="Tone: casual or professional")
    user_style: str = dspy.InputField(desc="User's texting style description")
    reply: str = dspy.OutputField(desc="Brief, natural reply matching the user's style")


class ReplyModule(dspy.Module):
    """Thin module wrapping Predict so DSPy can compile it."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(ReplySignature)

    def forward(self, **kwargs: str) -> dspy.Prediction:
        return self.generate(**kwargs)


# ---------------------------------------------------------------------------
# Training examples (derived from batch_eval.TEST_CASES)
# ---------------------------------------------------------------------------
def _build_examples() -> list[dspy.Example]:
    """Convert batch_eval test cases into dspy.Examples for the trainset."""
    from evals.batch_eval import TEST_CASES

    examples = []
    for tc in TEST_CASES:
        ex = dspy.Example(
            context=tc["context"],
            last_message=tc["last_message"],
            tone=tc["tone"],
            user_style=tc.get("user_style", ""),
        ).with_inputs("context", "last_message", "tone", "user_style")
        # Attach rubric as metadata for the metric function
        ex._rubric = tc.get("rubric", "")
        ex._max_words = tc.get("max_words")
        ex._max_chars = tc.get("max_chars")
        ex._banned = tc.get("banned", [])
        examples.append(ex)
    return examples


TRAIN_EXAMPLES: list[dspy.Example] = _build_examples()


# ---------------------------------------------------------------------------
# Metric: Cerebras LLM judge (mirrors batch_eval.judge_response)
# ---------------------------------------------------------------------------
def _get_judge_client():
    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    if not api_key or api_key == "your-key-here":
        return None
    from openai import OpenAI

    return OpenAI(base_url=CEREBRAS_BASE_URL, api_key=api_key)


_judge_client = None


def cerebras_judge_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Return True if judge scores the reply >= 7/10.

    Used by DSPy optimizers (BootstrapFewShot, MIPROv2) as the metric function.
    """
    global _judge_client
    if _judge_client is None:
        _judge_client = _get_judge_client()
    if _judge_client is None:
        # No API key: fall back to length heuristic
        reply = prediction.reply.strip()
        return 0 < len(reply) <= 120

    reply_text = prediction.reply.strip()
    rubric = getattr(example, "_rubric", "")
    if not rubric:
        return len(reply_text) > 0

    prompt = (
        "You are an expert evaluator for a text message reply generator.\n\n"
        f"CONVERSATION:\n{example.context}\n\n"
        f"LAST MESSAGE (to reply to):\n{example.last_message}\n\n"
        f"GENERATED REPLY:\n{reply_text}\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        "Score the generated reply from 0-10 based on the rubric.\n"
        "Respond in this exact JSON format:\n"
        '{"score": <0-10>, "reasoning": "<1-2 sentences>"}'
    )

    try:
        resp = _judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        score = float(data["score"])
        return score >= 7.0
    except Exception:
        # On judge failure, fall back to basic check
        return 0 < len(reply_text) <= 120
