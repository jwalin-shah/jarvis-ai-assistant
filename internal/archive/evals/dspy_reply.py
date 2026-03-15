"""DSPy signature, module, metric, and training data for reply generation.  # noqa: E501
  # noqa: E501
Defines the optimization target that DSPy compiles against:  # noqa: E501
- ReplySignature: typed I/O contract (generic)  # noqa: E501
- CategoryReplySignature: per-category signatures with tailored output desc  # noqa: E501
- ReplyModule / CategoryReplyModule: thin wrappers around dspy.Predict  # noqa: E501
- judge_metric: continuous LLM-as-judge scorer via Cerebras  # noqa: E501
- TRAIN_EXAMPLES: dspy.Example dataset derived from batch_eval TEST_CASES  # noqa: E501
- get_category_examples(): filter examples by category  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import json  # noqa: E501
import re  # noqa: E501

# noqa: E501
import dspy  # noqa: E501
from evals.judge_config import JUDGE_MODEL  # noqa: E402  # noqa: E501
from evals.judge_config import get_judge_client as _get_judge_client  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Signature + Module (generic, backward-compatible)  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
class ReplySignature(dspy.Signature):  # noqa: E501
    """Generate a natural text message reply."""  # noqa: E501
  # noqa: E501
    context: str = dspy.InputField(desc="Conversation history with timestamps")  # noqa: E501
    last_message: str = dspy.InputField(desc="The message to reply to")  # noqa: E501
    tone: str = dspy.InputField(desc="Tone: casual or professional")  # noqa: E501
    user_style: str = dspy.InputField(desc="User's texting style description")  # noqa: E501
    reply: str = dspy.OutputField(desc="Brief, natural reply matching the user's style")  # noqa: E501
  # noqa: E501
  # noqa: E501
class ReplyModule(dspy.Module):  # noqa: E501
    """Thin module wrapping Predict so DSPy can compile it."""  # noqa: E501
  # noqa: E501
    def __init__(self) -> None:  # noqa: E501
        super().__init__()  # noqa: E501
        self.generate = dspy.Predict(ReplySignature)  # noqa: E501
  # noqa: E501
    def forward(self, **kwargs: str) -> dspy.Prediction:  # noqa: E501
        return self.generate(**kwargs)  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Per-Category Signatures (tailored output field descriptions)  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
CATEGORY_OUTPUT_DESC: dict[str, str] = {  # noqa: E501
    "brief": (  # noqa: E501
        "Ultra-brief confirmation or answer (1-5 words). "  # noqa: E501
        "No greetings, no filler, just the response."  # noqa: E501
    ),  # noqa: E501
    "warm": (  # noqa: E501
        "Brief, empathetic reply showing you care. No advice, no toxic positivity, just support."  # noqa: E501
    ),  # noqa: E501
    "social": (  # noqa: E501
        "Warm, conversational reply matching their energy. Natural and friendly, not formal."  # noqa: E501
    ),  # noqa: E501
    "clarify": (  # noqa: E501
        "Handle ambiguity gracefully. Don't assume or confabulate. "  # noqa: E501
        "Brief, honest response when context is unclear."  # noqa: E501
    ),  # noqa: E501
}  # noqa: E501
  # noqa: E501
  # noqa: E501
def make_category_signature(category: str) -> type[dspy.Signature]:  # noqa: E501
    """Dynamically create a category-specific ReplySignature.  # noqa: E501
  # noqa: E501
    The output field `reply` gets a tailored description based on category.  # noqa: E501
    """  # noqa: E501
    desc = CATEGORY_OUTPUT_DESC.get(  # noqa: E501
        category,  # noqa: E501
        "Brief, natural reply matching the user's style",  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    # Create a new Signature class dynamically  # noqa: E501
    attrs = {  # noqa: E501
        "__doc__": f"Generate a natural text message reply ({category}).",  # noqa: E501
        "__annotations__": {  # noqa: E501
            "context": str,  # noqa: E501
            "last_message": str,  # noqa: E501
            "tone": str,  # noqa: E501
            "user_style": str,  # noqa: E501
            "reply": str,  # noqa: E501
        },  # noqa: E501
        "context": dspy.InputField(desc="Conversation history with timestamps"),  # noqa: E501
        "last_message": dspy.InputField(desc="The message to reply to"),  # noqa: E501
        "tone": dspy.InputField(desc="Tone: casual or professional"),  # noqa: E501
        "user_style": dspy.InputField(desc="User's texting style description"),  # noqa: E501
        "reply": dspy.OutputField(desc=desc),  # noqa: E501
    }  # noqa: E501
    sig_cls = type(f"ReplySignature_{category}", (dspy.Signature,), attrs)  # noqa: E501
    return sig_cls  # noqa: E501
  # noqa: E501
  # noqa: E501
class CategoryReplyModule(dspy.Module):  # noqa: E501
    """Module with category-specific signature for per-category optimization."""  # noqa: E501
  # noqa: E501
    def __init__(self, category: str) -> None:  # noqa: E501
        super().__init__()  # noqa: E501
        sig = make_category_signature(category)  # noqa: E501
        self.generate = dspy.Predict(sig)  # noqa: E501
        self.category = category  # noqa: E501
  # noqa: E501
    def forward(self, **kwargs: str) -> dspy.Prediction:  # noqa: E501
        return self.generate(**kwargs)  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Training examples (derived from batch_eval.TEST_CASES)  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
def _build_examples() -> list[dspy.Example]:  # noqa: E501
    """Convert batch_eval test cases into dspy.Examples for the trainset."""  # noqa: E501
    from evals.batch_eval import TEST_CASES  # noqa: E501
  # noqa: E501
    examples = []  # noqa: E501
    for tc in TEST_CASES:  # noqa: E501
        ex = dspy.Example(  # noqa: E501
            context=tc["context"],  # noqa: E501
            last_message=tc["last_message"],  # noqa: E501
            tone=tc["tone"],  # noqa: E501
            user_style=tc.get("user_style", ""),  # noqa: E501
        ).with_inputs("context", "last_message", "tone", "user_style")  # noqa: E501
        # Attach metadata for the metric function  # noqa: E501
        ex._rubric = tc.get("rubric", "")  # noqa: E501
        ex._max_words = tc.get("max_words")  # noqa: E501
        ex._max_chars = tc.get("max_chars")  # noqa: E501
        ex._banned = tc.get("banned", [])  # noqa: E501
        ex._category = tc.get("category", "unknown")  # noqa: E501
        examples.append(ex)  # noqa: E501
    return examples  # noqa: E501
  # noqa: E501
  # noqa: E501
TRAIN_EXAMPLES: list[dspy.Example] = _build_examples()  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_category_examples(category: str) -> list[dspy.Example]:  # noqa: E501
    """Filter TRAIN_EXAMPLES to those matching the given category."""  # noqa: E501
    return [ex for ex in TRAIN_EXAMPLES if getattr(ex, "_category", "") == category]  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_all_categories() -> list[str]:  # noqa: E501
    """Return sorted unique categories present in TRAIN_EXAMPLES."""  # noqa: E501
    cats = {getattr(ex, "_category", "unknown") for ex in TRAIN_EXAMPLES}  # noqa: E501
    return sorted(cats)  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Output cleaning (strip DSPy artifacts before judging)  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
# DSPy delimiters that leak into small model outputs  # noqa: E501
_DSPY_ARTIFACTS = re.compile(  # noqa: E501
    r"\[\[.*?##.*?##.*?\]\]"  # [[ ## completed ## ]], [[ ## reply ## ]], etc.  # noqa: E501
    r"|---"  # Markdown separators DSPy inserts  # noqa: E501
    r"|^\s*\[completed\]\s*$"  # [completed] on its own line  # noqa: E501
    r"|<\|im_end\|>"  # Chat template leaks  # noqa: E501
    r"|<\|endoftext\|>",  # EOS token leaks  # noqa: E501
    re.IGNORECASE | re.MULTILINE,  # noqa: E501
)  # noqa: E501
  # noqa: E501
  # noqa: E501
def clean_reply(text: str) -> str:  # noqa: E501
    """Strip DSPy formatting artifacts and normalize reply text.  # noqa: E501
  # noqa: E501
    Removes leaked delimiters, trims whitespace, and takes only the first  # noqa: E501
    line if the model generated multiple (common with small LMs).  # noqa: E501
    """  # noqa: E501
    text = _DSPY_ARTIFACTS.sub("", text).strip()  # noqa: E501
    # Take first non-empty line only (model sometimes rambles after the reply)  # noqa: E501
    for line in text.split("\n"):  # noqa: E501
        line = line.strip()  # noqa: E501
        if line:  # noqa: E501
            return line  # noqa: E501
    return text  # noqa: E501
  # noqa: E501
  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
# Metric: LLM judge via Gemini 2.5 Flash (mirrors batch_eval.judge_response)  # noqa: E501
# ---------------------------------------------------------------------------  # noqa: E501
  # noqa: E501
# _get_judge_client imported from evals.judge_config  # noqa: E501
  # noqa: E501
  # noqa: E501
_judge_client = None  # noqa: E501
  # noqa: E501
  # noqa: E501
def judge_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:  # noqa: E501
    """Return continuous 0.0-1.0 score from judge (score/10).  # noqa: E501
  # noqa: E501
    Used by DSPy optimizers (BootstrapFewShot, MIPROv2) as the metric function.  # noqa: E501
    Returns a continuous float for better optimization signal instead of binary pass/fail.  # noqa: E501
    """  # noqa: E501
    global _judge_client  # noqa: E501
    if _judge_client is None:  # noqa: E501
        _judge_client = _get_judge_client()  # noqa: E501
    if _judge_client is None:  # noqa: E501
        # No API key: fall back to length heuristic  # noqa: E501
        reply = prediction.reply.strip()  # noqa: E501
        return 0.5 if 0 < len(reply) <= 120 else 0.0  # noqa: E501
  # noqa: E501
    reply_text = prediction.reply.strip()  # noqa: E501
    rubric = getattr(example, "_rubric", "")  # noqa: E501
    if not rubric:  # noqa: E501
        return 0.5 if len(reply_text) > 0 else 0.0  # noqa: E501
  # noqa: E501
    prompt = (  # noqa: E501
        "You are an expert evaluator for a text message reply generator.\n\n"  # noqa: E501
        f"CONVERSATION:\n{example.context}\n\n"  # noqa: E501
        f"LAST MESSAGE (to reply to):\n{example.last_message}\n\n"  # noqa: E501
        f"GENERATED REPLY:\n{reply_text}\n\n"  # noqa: E501
        f"RUBRIC:\n{rubric}\n\n"  # noqa: E501
        "Score the generated reply from 0-10 based on the rubric.\n"  # noqa: E501
        "Respond in this exact JSON format:\n"  # noqa: E501
        '{"score": <0-10>, "reasoning": "<1-2 sentences>"}'  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        resp = _judge_client.chat.completions.create(  # noqa: E501
            model=JUDGE_MODEL,  # noqa: E501
            messages=[{"role": "user", "content": prompt}],  # noqa: E501
            temperature=0.0,  # noqa: E501
            max_tokens=150,  # noqa: E501
        )  # noqa: E501
        text = resp.choices[0].message.content.strip()  # noqa: E501
        if text.startswith("```"):  # noqa: E501
            text = text.split("```")[1]  # noqa: E501
            if text.startswith("json"):  # noqa: E501
                text = text[4:]  # noqa: E501
        data = json.loads(text)  # noqa: E501
        score = float(data["score"])  # noqa: E501
        return score / 10.0  # Normalize to 0.0-1.0 for DSPy  # noqa: E501
    except Exception:  # noqa: E501
        # On judge failure, fall back to basic check  # noqa: E501
        return 0.5 if 0 < len(reply_text) <= 120 else 0.0  # noqa: E501
  # noqa: E501
  # noqa: E501
# Backward-compatible alias  # noqa: E501
cerebras_judge_metric = judge_metric  # noqa: E501
