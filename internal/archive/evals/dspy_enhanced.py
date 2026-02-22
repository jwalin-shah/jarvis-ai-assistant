#!/usr/bin/env python3
"""Enhanced DSPy modules with better signatures for text message generation.

This extends the basic dspy_reply.py with:
- Style-aware signatures
- Two-stage generation (intent → reply)
- Constraint-aware generation
"""

from __future__ import annotations

import dspy

# =============================================================================
# Enhanced Signatures
# =============================================================================


class StyleAwareReplySignature(dspy.Signature):
    """Generate a text reply that matches the user's exact texting style."""

    context: str = dspy.InputField(desc="Recent conversation messages with timestamps")
    last_message: str = dspy.InputField(desc="The message you need to reply to")
    message_category: str = dspy.InputField(
        desc="Type of message: question, request, emotion, statement, acknowledge, closing"
    )
    user_style: str = dspy.InputField(
        desc="User's style: brief/verbose, formal/casual, emoji usage, abbreviations"
    )
    conversation_tone: str = dspy.InputField(
        desc="Tone detected: casual, professional, excited, sad, angry, neutral"
    )

    reply: str = dspy.OutputField(
        desc=(
            "A brief, natural text message reply (1-2 sentences max). "
            "Match their style exactly. No AI phrases. Sound human."
        )
    )


class IntentThenReplySignature(dspy.Signature):
    """Analyze intent, then generate appropriate reply."""

    context: str = dspy.InputField(desc="Conversation history")
    last_message: str = dspy.InputField(desc="Message to analyze")

    intent_type: str = dspy.OutputField(
        desc="What they want: info, action, support, chat, confirm, end_conversation"
    )
    urgency: str = dspy.OutputField(desc="How urgent: immediate, today, sometime, not_urgent")
    suggested_tone: str = dspy.OutputField(
        desc="Best tone: empathetic, enthusiastic, neutral, brief, professional"
    )
    reply: str = dspy.OutputField(desc="Reply matching the intent and tone")


class ConstraintAwareReplySignature(dspy.Signature):
    """Generate reply respecting hard constraints."""

    context: str = dspy.InputField(desc="Conversation")
    last_message: str = dspy.InputField(desc="Message to reply to")
    max_length_chars: int = dspy.InputField(desc="Maximum characters allowed")
    max_sentences: int = dspy.InputField(desc="Maximum sentences allowed")
    banned_phrases: str = dspy.InputField(desc="Comma-separated list of phrases to NEVER use")
    required_elements: str = dspy.InputField(
        desc="Comma-separated elements that SHOULD be included (optional)"
    )

    reply: str = dspy.OutputField(desc="Reply respecting all constraints")


# =============================================================================
# Category-Specific Signatures (More Detailed)
# =============================================================================

CATEGORY_SIGNATURES = {
    "question": """They asked a question. Your job: answer it directly.
    
Rules:
- Give the answer first, then brief context if needed
- Don't ask counter-questions unless critical
- Match their specificity level (vague question → brief answer)
- If you don't know, say so honestly""",
    "request": """They're asking you to do something.

Rules:
- Say yes, no, or maybe clearly
- If yes: confirm what you'll do
- If no: brief reason optional
- If maybe: say what you need to decide""",
    "emotion": """They're sharing feelings (good or bad).

Rules:
- Match their energy (celebrate wins, commiserate losses)
- NEVER give advice unless explicitly asked
- NEVER minimize their feelings ("at least...", "everything happens...")
- Show you're listening, not solving""",
    "statement": """They're sharing information or chatting.

Rules:
- React naturally (not too enthusiastic, not dismissive)  
- Add something to move conversation forward OR just acknowledge
- Match their length (long message → longer reply, short → short)""",
    "acknowledge": """Brief acknowledgment needed.

Rules:
- 1-5 words maximum
- Just confirm you heard them
- Examples: "ok", "sounds good", "got it", "np", "👍""",
    "closing": """Conversation is ending.

Rules:
- Mirror their farewell energy
- Don't drag it out
- Examples: "later!", "ttyl", "night!", "👋""",
}


def make_enhanced_category_signature(category: str) -> type[dspy.Signature]:
    """Create a highly tailored signature for a category."""

    category_desc = CATEGORY_SIGNATURES.get(
        category, "Generate a natural text reply matching the user's style"
    )

    attrs = {
        "__doc__": f"Generate {category} reply",
        "__annotations__": {
            "context": str,
            "last_message": str,
            "user_style": str,
            "conversation_tone": str,
            "reply": str,
        },
        "context": dspy.InputField(desc="Recent conversation with timestamps and sender labels"),
        "last_message": dspy.InputField(desc="The message to reply to"),
        "user_style": dspy.InputField(
            desc="Detected style: brief/verbose, lowercase/proper, emoji level"
        ),
        "conversation_tone": dspy.InputField(desc="Current emotional tone"),
        "reply": dspy.OutputField(desc=category_desc),
    }

    return type(f"Enhanced{category.title()}Signature", (dspy.Signature,), attrs)


# =============================================================================
# Enhanced Modules
# =============================================================================


class StyleAwareReplyModule(dspy.Module):
    """Reply module with style awareness and constraints."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(StyleAwareReplySignature)

    def forward(
        self,
        context: str,
        last_message: str,
        message_category: str = "statement",
        user_style: str = "casual",
        conversation_tone: str = "neutral",
    ) -> dspy.Prediction:
        return self.generate(
            context=context,
            last_message=last_message,
            message_category=message_category,
            user_style=user_style,
            conversation_tone=conversation_tone,
        )


class TwoStageReplyModule(dspy.Module):
    """Two-stage: analyze intent, then generate reply."""

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.Predict(IntentThenReplySignature)

    def forward(
        self,
        context: str,
        last_message: str,
        **kwargs,
    ) -> dspy.Prediction:
        # Single call does both (signature has multiple outputs)
        result = self.analyze(context=context, last_message=last_message)

        return dspy.Prediction(
            reply=result.reply,
            intent=result.intent_type,
            urgency=result.urgency,
            tone=result.suggested_tone,
        )


class EnhancedCategoryModule(dspy.Module):
    """Per-category module with enhanced signatures."""

    def __init__(self, category: str) -> None:
        super().__init__()
        self.category = category
        sig = make_enhanced_category_signature(category)
        self.generate = dspy.Predict(sig)

    def forward(
        self,
        context: str,
        last_message: str,
        user_style: str = "",
        conversation_tone: str = "casual",
        **kwargs,
    ) -> dspy.Prediction:
        return self.generate(
            context=context,
            last_message=last_message,
            user_style=user_style,
            conversation_tone=conversation_tone,
        )


# =============================================================================
# Optimization Configurations
# =============================================================================

OPTIMIZATION_PRESETS = {
    "conservative": {
        "optimizer": "BootstrapFewShot",
        "max_bootstrapped_demos": 2,
        "max_labeled_demos": 3,
        "num_trials": 5,
        "description": "Fast, safe, minimal overfitting risk",
    },
    "balanced": {
        "optimizer": "MIPROv2",
        "num_candidates": 5,
        "num_trials": 15,
        "max_bootstrapped_demos": 3,
        "description": "Good quality/compute balance",
    },
    "aggressive": {
        "optimizer": "MIPROv2",
        "num_candidates": 10,
        "num_trials": 30,
        "max_bootstrapped_demos": 5,
        "description": "Maximum quality, expensive",
    },
}


def get_optimization_config(preset: str = "balanced") -> dict:
    """Get optimization configuration preset."""
    return OPTIMIZATION_PRESETS.get(preset, OPTIMIZATION_PRESETS["balanced"])


# =============================================================================
# Testing / Evaluation Helpers
# =============================================================================


def compare_modules(
    test_cases: list[dict],
    modules: dict[str, dspy.Module],
    judge_client=None,
) -> dict:
    """Compare multiple modules on same test cases.

    Args:
        test_cases: List of {context, last_message, category, ideal_reply}
        modules: Dict of {name: module}
        judge_client: Optional judge client for scoring

    Returns:
        Comparison results by module
    """
    import time

    results = {name: {"replies": [], "latencies": [], "scores": []} for name in modules}

    for tc in test_cases:
        for name, module in modules.items():
            start = time.perf_counter()
            try:
                pred = module(**tc)
                reply = pred.reply.strip()
            except Exception as e:
                reply = f"[ERROR: {e}]"
            latency = time.perf_counter() - start

            results[name]["replies"].append(reply)
            results[name]["latencies"].append(latency * 1000)

            # Score if judge available
            if judge_client:
                # TODO: Implement judge scoring
                pass

    # Compute stats
    for name in results:
        latencies = results[name]["latencies"]
        results[name]["avg_latency"] = sum(latencies) / len(latencies) if latencies else 0
        results[name]["p95_latency"] = (
            sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        )

    return results


if __name__ == "__main__":
    # Quick test
    print("Enhanced DSPy modules loaded.")
    print("\nAvailable modules:")
    print("  - StyleAwareReplyModule")
    print("  - TwoStageReplyModule")
    print("  - EnhancedCategoryModule")
    print("\nAvailable presets:")
    for name, config in OPTIMIZATION_PRESETS.items():
        print(f"  - {name}: {config['description']}")
