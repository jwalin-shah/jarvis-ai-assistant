"""Enhanced DSPy signatures and logic for reply generation.

Focuses on:
- Style-constrained generation (length, tone, punctuation)
- CoT with style reasoning
- Rubric-based judging
"""

import dspy

# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class GenerateReply(dspy.Signature):
    """Generate a natural text message reply matching the user's style."""

    context = dspy.InputField(desc="Recent conversation history")
    last_message = dspy.InputField(desc="The message to reply to")
    tone = dspy.InputField(desc="Desired tone (casual, professional, etc)")
    user_style = dspy.InputField(desc="Description of user's texting style")

    reasoning = dspy.OutputField(desc="Brief thought process on style and content")
    reply = dspy.OutputField(desc="The text message reply")


class JudgeReply(dspy.Signature):
    """Judge a text message reply based on specific criteria."""

    context = dspy.InputField()
    last_message = dspy.InputField()
    reply = dspy.InputField()
    rubric = dspy.InputField(desc="Criteria to judge against")

    score = dspy.OutputField(desc="Float score between 0.0 and 10.0")
    reasoning = dspy.OutputField(desc="Why this score was given")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATEGORY_SIGNATURES = {
    "question": """They asked a question. Your job: answer it directly.

Rules:
- Give the answer first, then brief context if needed
- If you don't know, say 'idk' or similar
- Keep it under 15 words unless complex""",
    "statement": """They made a statement. Your job: acknowledge and/or pivot.

Rules:
- React naturally (not too enthusiastic, not dismissive)
- Add something to move conversation forward OR just acknowledge
- Match their length (long message → longer reply, short → short)""",
    "scheduling": """They want to meet/schedule. Your job: confirm or propose.

Rules:
- Be specific about time/place
- Be decisive (don't say 'maybe' unless you mean it)
- Use 'bet', 'sounds good', 'sure' for agreement""",
}

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class ReplyGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateReply)

    def forward(self, context, last_message, tone, user_style):
        return self.generate(
            context=context, last_message=last_message, tone=tone, user_style=user_style
        )
