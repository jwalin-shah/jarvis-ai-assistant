"""DSPy signature, module, metric, and training data for reply generation.

Defines the optimization target that DSPy compiles against:
- ReplySignature: typed I/O contract (generic)
- CategoryReplySignature: per-category signatures with tailored output desc
- ReplyModule / CategoryReplyModule: thin wrappers around dspy.Predict
- judge_metric: continuous LLM-as-judge scorer via Cerebras
- TRAIN_EXAMPLES: dspy.Example dataset derived from batch_eval TEST_CASES
- get_category_examples(): filter examples by category
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import dspy

from evals.judge_config import JUDGE_MODEL, get_judge_client as _get_judge_client


# ---------------------------------------------------------------------------
# Signature + Module (generic, backward-compatible)
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
# Per-Category Signatures (tailored output field descriptions)
# ---------------------------------------------------------------------------
CATEGORY_OUTPUT_DESC: dict[str, str] = {
    "brief": (
        "Ultra-brief confirmation or answer (1-5 words). "
        "No greetings, no filler, just the response."
    ),
    "warm": (
        "Brief, empathetic reply showing you care. No advice, no toxic positivity, just support."
    ),
    "social": (
        "Warm, conversational reply matching their energy. Natural and friendly, not formal."
    ),
    "clarify": (
        "Handle ambiguity gracefully. Don't assume or confabulate. "
        "Brief, honest response when context is unclear."
    ),
}


def make_category_signature(category: str) -> type[dspy.Signature]:
    """Dynamically create a category-specific ReplySignature.

    The output field `reply` gets a tailored description based on category.
    """
    desc = CATEGORY_OUTPUT_DESC.get(
        category,
        "Brief, natural reply matching the user's style",
    )

    # Create a new Signature class dynamically
    attrs = {
        "__doc__": f"Generate a natural text message reply ({category}).",
        "__annotations__": {
            "context": str,
            "last_message": str,
            "tone": str,
            "user_style": str,
            "reply": str,
        },
        "context": dspy.InputField(desc="Conversation history with timestamps"),
        "last_message": dspy.InputField(desc="The message to reply to"),
        "tone": dspy.InputField(desc="Tone: casual or professional"),
        "user_style": dspy.InputField(desc="User's texting style description"),
        "reply": dspy.OutputField(desc=desc),
    }
    sig_cls = type(f"ReplySignature_{category}", (dspy.Signature,), attrs)
    return sig_cls


class CategoryReplyModule(dspy.Module):
    """Module with category-specific signature for per-category optimization."""

    def __init__(self, category: str) -> None:
        super().__init__()
        sig = make_category_signature(category)
        self.generate = dspy.Predict(sig)
        self.category = category

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
        # Attach metadata for the metric function
        ex._rubric = tc.get("rubric", "")
        ex._max_words = tc.get("max_words")
        ex._max_chars = tc.get("max_chars")
        ex._banned = tc.get("banned", [])
        ex._category = tc.get("category", "unknown")
        examples.append(ex)
    return examples


TRAIN_EXAMPLES: list[dspy.Example] = _build_examples()


def get_category_examples(category: str) -> list[dspy.Example]:
    """Filter TRAIN_EXAMPLES to those matching the given category."""
    return [ex for ex in TRAIN_EXAMPLES if getattr(ex, "_category", "") == category]


def get_all_categories() -> list[str]:
    """Return sorted unique categories present in TRAIN_EXAMPLES."""
    cats = {getattr(ex, "_category", "unknown") for ex in TRAIN_EXAMPLES}
    return sorted(cats)


# ---------------------------------------------------------------------------
# Output cleaning (strip DSPy artifacts before judging)
# ---------------------------------------------------------------------------

# DSPy delimiters that leak into small model outputs
_DSPY_ARTIFACTS = re.compile(
    r"\[\[.*?##.*?##.*?\]\]"  # [[ ## completed ## ]], [[ ## reply ## ]], etc.
    r"|---"  # Markdown separators DSPy inserts
    r"|^\s*\[completed\]\s*$"  # [completed] on its own line
    r"|<\|im_end\|>"  # Chat template leaks
    r"|<\|endoftext\|>",  # EOS token leaks
    re.IGNORECASE | re.MULTILINE,
)


def clean_reply(text: str) -> str:
    """Strip DSPy formatting artifacts and normalize reply text.

    Removes leaked delimiters, trims whitespace, and takes only the first
    line if the model generated multiple (common with small LMs).
    """
    text = _DSPY_ARTIFACTS.sub("", text).strip()
    # Take first non-empty line only (model sometimes rambles after the reply)
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line
    return text


# ---------------------------------------------------------------------------
# Metric: LLM judge via Gemini 2.5 Flash (mirrors batch_eval.judge_response)
# ---------------------------------------------------------------------------

# _get_judge_client imported from evals.judge_config


_judge_client = None


def judge_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Return continuous 0.0-1.0 score from judge (score/10).

    Used by DSPy optimizers (BootstrapFewShot, MIPROv2) as the metric function.
    Returns a continuous float for better optimization signal instead of binary pass/fail.
    """
    global _judge_client
    if _judge_client is None:
        _judge_client = _get_judge_client()
    if _judge_client is None:
        # No API key: fall back to length heuristic
        reply = prediction.reply.strip()
        return 0.5 if 0 < len(reply) <= 120 else 0.0

    reply_text = prediction.reply.strip()
    rubric = getattr(example, "_rubric", "")
    if not rubric:
        return 0.5 if len(reply_text) > 0 else 0.0

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
        return score / 10.0  # Normalize to 0.0-1.0 for DSPy
    except Exception:
        # On judge failure, fall back to basic check
        return 0.5 if 0 < len(reply_text) <= 120 else 0.0


# Backward-compatible alias
cerebras_judge_metric = judge_metric
