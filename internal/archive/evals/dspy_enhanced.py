#!/usr/bin/env python3  # noqa: E501
"""Enhanced DSPy modules with better signatures for text message generation.  # noqa: E501
  # noqa: E501
This extends the basic dspy_reply.py with:  # noqa: E501
- Style-aware signatures  # noqa: E501
- Two-stage generation (intent → reply)  # noqa: E501
- Constraint-aware generation  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import dspy  # noqa: E501


  # noqa: E501
# =============================================================================  # noqa: E501
# Enhanced Signatures  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
class StyleAwareReplySignature(dspy.Signature):  # noqa: E501
    """Generate a text reply that matches the user's exact texting style."""  # noqa: E501
  # noqa: E501
    context: str = dspy.InputField(desc="Recent conversation messages with timestamps")  # noqa: E501
    last_message: str = dspy.InputField(desc="The message you need to reply to")  # noqa: E501
    message_category: str = dspy.InputField(  # noqa: E501
        desc="Type of message: question, request, emotion, statement, acknowledge, closing"  # noqa: E501
    )  # noqa: E501
    user_style: str = dspy.InputField(  # noqa: E501
        desc="User's style: brief/verbose, formal/casual, emoji usage, abbreviations"  # noqa: E501
    )  # noqa: E501
    conversation_tone: str = dspy.InputField(  # noqa: E501
        desc="Tone detected: casual, professional, excited, sad, angry, neutral"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    reply: str = dspy.OutputField(  # noqa: E501
        desc=(  # noqa: E501
            "A brief, natural text message reply (1-2 sentences max). "  # noqa: E501
            "Match their style exactly. No AI phrases. Sound human."  # noqa: E501
        )  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
class IntentThenReplySignature(dspy.Signature):  # noqa: E501
    """Analyze intent, then generate appropriate reply."""  # noqa: E501
  # noqa: E501
    context: str = dspy.InputField(desc="Conversation history")  # noqa: E501
    last_message: str = dspy.InputField(desc="Message to analyze")  # noqa: E501
  # noqa: E501
    intent_type: str = dspy.OutputField(  # noqa: E501
        desc="What they want: info, action, support, chat, confirm, end_conversation"  # noqa: E501
    )  # noqa: E501
    urgency: str = dspy.OutputField(desc="How urgent: immediate, today, sometime, not_urgent")  # noqa: E501
    suggested_tone: str = dspy.OutputField(  # noqa: E501
        desc="Best tone: empathetic, enthusiastic, neutral, brief, professional"  # noqa: E501
    )  # noqa: E501
    reply: str = dspy.OutputField(desc="Reply matching the intent and tone")  # noqa: E501
  # noqa: E501
  # noqa: E501
class ConstraintAwareReplySignature(dspy.Signature):  # noqa: E501
    """Generate reply respecting hard constraints."""  # noqa: E501
  # noqa: E501
    context: str = dspy.InputField(desc="Conversation")  # noqa: E501
    last_message: str = dspy.InputField(desc="Message to reply to")  # noqa: E501
    max_length_chars: int = dspy.InputField(desc="Maximum characters allowed")  # noqa: E501
    max_sentences: int = dspy.InputField(desc="Maximum sentences allowed")  # noqa: E501
    banned_phrases: str = dspy.InputField(desc="Comma-separated list of phrases to NEVER use")  # noqa: E501
    required_elements: str = dspy.InputField(  # noqa: E501
        desc="Comma-separated elements that SHOULD be included (optional)"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    reply: str = dspy.OutputField(desc="Reply respecting all constraints")  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Category-Specific Signatures (More Detailed)  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
CATEGORY_SIGNATURES = {  # noqa: E501
    "question": """They asked a question. Your job: answer it directly.  # noqa: E501
  # noqa: E501
Rules:  # noqa: E501
- Give the answer first, then brief context if needed  # noqa: E501
- Don't ask counter-questions unless critical  # noqa: E501
- Match their specificity level (vague question → brief answer)  # noqa: E501
- If you don't know, say so honestly""",  # noqa: E501
    "request": """They're asking you to do something.  # noqa: E501
  # noqa: E501
Rules:  # noqa: E501
- Say yes, no, or maybe clearly  # noqa: E501
- If yes: confirm what you'll do  # noqa: E501
- If no: brief reason optional  # noqa: E501
- If maybe: say what you need to decide""",  # noqa: E501
    "emotion": """They're sharing feelings (good or bad).  # noqa: E501
  # noqa: E501
Rules:  # noqa: E501
- Match their energy (celebrate wins, commiserate losses)  # noqa: E501
- NEVER give advice unless explicitly asked  # noqa: E501
- NEVER minimize their feelings ("at least...", "everything happens...")  # noqa: E501
- Show you're listening, not solving""",  # noqa: E501
    "statement": """They're sharing information or chatting.  # noqa: E501
  # noqa: E501
Rules:  # noqa: E501
- React naturally (not too enthusiastic, not dismissive)  # noqa: E501
- Add something to move conversation forward OR just acknowledge  # noqa: E501
- Match their length (long message → longer reply, short → short)""",  # noqa: E501
    "acknowledge": """Brief acknowledgment needed.  # noqa: E501
  # noqa: E501
Rules:  # noqa: E501
- 1-5 words maximum  # noqa: E501
- Just confirm you heard them  # noqa: E501
- Examples: "ok", "sounds good", "got it", "np", "👍""",  # noqa: E501
    "closing": """Conversation is ending.  # noqa: E501
  # noqa: E501
Rules:  # noqa: E501
- Mirror their farewell energy  # noqa: E501
- Don't drag it out  # noqa: E501
- Examples: "later!", "ttyl", "night!", "👋""",  # noqa: E501
}  # noqa: E501
  # noqa: E501
  # noqa: E501
def make_enhanced_category_signature(category: str) -> type[dspy.Signature]:  # noqa: E501
    """Create a highly tailored signature for a category."""  # noqa: E501
  # noqa: E501
    category_desc = CATEGORY_SIGNATURES.get(  # noqa: E501
        category, "Generate a natural text reply matching the user's style"  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    attrs = {  # noqa: E501
        "__doc__": f"Generate {category} reply",  # noqa: E501
        "__annotations__": {  # noqa: E501
            "context": str,  # noqa: E501
            "last_message": str,  # noqa: E501
            "user_style": str,  # noqa: E501
            "conversation_tone": str,  # noqa: E501
            "reply": str,  # noqa: E501
        },  # noqa: E501
        "context": dspy.InputField(desc="Recent conversation with timestamps and sender labels"),  # noqa: E501
        "last_message": dspy.InputField(desc="The message to reply to"),  # noqa: E501
        "user_style": dspy.InputField(  # noqa: E501
            desc="Detected style: brief/verbose, lowercase/proper, emoji level"  # noqa: E501
        ),  # noqa: E501
        "conversation_tone": dspy.InputField(desc="Current emotional tone"),  # noqa: E501
        "reply": dspy.OutputField(desc=category_desc),  # noqa: E501
    }  # noqa: E501
  # noqa: E501
    return type(f"Enhanced{category.title()}Signature", (dspy.Signature,), attrs)  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Enhanced Modules  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
class StyleAwareReplyModule(dspy.Module):  # noqa: E501
    """Reply module with style awareness and constraints."""  # noqa: E501
  # noqa: E501
    def __init__(self) -> None:  # noqa: E501
        super().__init__()  # noqa: E501
        self.generate = dspy.Predict(StyleAwareReplySignature)  # noqa: E501
  # noqa: E501
    def forward(  # noqa: E501
        self,  # noqa: E501
        context: str,  # noqa: E501
        last_message: str,  # noqa: E501
        message_category: str = "statement",  # noqa: E501
        user_style: str = "casual",  # noqa: E501
        conversation_tone: str = "neutral",  # noqa: E501
    ) -> dspy.Prediction:  # noqa: E501
        return self.generate(  # noqa: E501
            context=context,  # noqa: E501
            last_message=last_message,  # noqa: E501
            message_category=message_category,  # noqa: E501
            user_style=user_style,  # noqa: E501
            conversation_tone=conversation_tone,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
  # noqa: E501
class TwoStageReplyModule(dspy.Module):  # noqa: E501
    """Two-stage: analyze intent, then generate reply."""  # noqa: E501
  # noqa: E501
    def __init__(self) -> None:  # noqa: E501
        super().__init__()  # noqa: E501
        self.analyze = dspy.Predict(IntentThenReplySignature)  # noqa: E501
  # noqa: E501
    def forward(  # noqa: E501
        self,  # noqa: E501
        context: str,  # noqa: E501
        last_message: str,  # noqa: E501
        **kwargs,  # noqa: E501
    ) -> dspy.Prediction:  # noqa: E501
        # Single call does both (signature has multiple outputs)  # noqa: E501
        result = self.analyze(context=context, last_message=last_message)  # noqa: E501
  # noqa: E501
        return dspy.Prediction(  # noqa: E501
            reply=result.reply,  # noqa: E501
            intent=result.intent_type,  # noqa: E501
            urgency=result.urgency,  # noqa: E501
            tone=result.suggested_tone,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
  # noqa: E501
class EnhancedCategoryModule(dspy.Module):  # noqa: E501
    """Per-category module with enhanced signatures."""  # noqa: E501
  # noqa: E501
    def __init__(self, category: str) -> None:  # noqa: E501
        super().__init__()  # noqa: E501
        self.category = category  # noqa: E501
        sig = make_enhanced_category_signature(category)  # noqa: E501
        self.generate = dspy.Predict(sig)  # noqa: E501
  # noqa: E501
    def forward(  # noqa: E501
        self,  # noqa: E501
        context: str,  # noqa: E501
        last_message: str,  # noqa: E501
        user_style: str = "",  # noqa: E501
        conversation_tone: str = "casual",  # noqa: E501
        **kwargs,  # noqa: E501
    ) -> dspy.Prediction:  # noqa: E501
        return self.generate(  # noqa: E501
            context=context,  # noqa: E501
            last_message=last_message,  # noqa: E501
            user_style=user_style,  # noqa: E501
            conversation_tone=conversation_tone,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Optimization Configurations  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
OPTIMIZATION_PRESETS = {  # noqa: E501
    "conservative": {  # noqa: E501
        "optimizer": "BootstrapFewShot",  # noqa: E501
        "max_bootstrapped_demos": 2,  # noqa: E501
        "max_labeled_demos": 3,  # noqa: E501
        "num_trials": 5,  # noqa: E501
        "description": "Fast, safe, minimal overfitting risk",  # noqa: E501
    },  # noqa: E501
    "balanced": {  # noqa: E501
        "optimizer": "MIPROv2",  # noqa: E501
        "num_candidates": 5,  # noqa: E501
        "num_trials": 15,  # noqa: E501
        "max_bootstrapped_demos": 3,  # noqa: E501
        "description": "Good quality/compute balance",  # noqa: E501
    },  # noqa: E501
    "aggressive": {  # noqa: E501
        "optimizer": "MIPROv2",  # noqa: E501
        "num_candidates": 10,  # noqa: E501
        "num_trials": 30,  # noqa: E501
        "max_bootstrapped_demos": 5,  # noqa: E501
        "description": "Maximum quality, expensive",  # noqa: E501
    },  # noqa: E501
}  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_optimization_config(preset: str = "balanced") -> dict:  # noqa: E501
    """Get optimization configuration preset."""  # noqa: E501
    return OPTIMIZATION_PRESETS.get(preset, OPTIMIZATION_PRESETS["balanced"])  # noqa: E501
  # noqa: E501
  # noqa: E501
# =============================================================================  # noqa: E501
# Testing / Evaluation Helpers  # noqa: E501
# =============================================================================  # noqa: E501
  # noqa: E501
  # noqa: E501
def compare_modules(  # noqa: E501
    test_cases: list[dict],  # noqa: E501
    modules: dict[str, dspy.Module],  # noqa: E501
    judge_client=None,  # noqa: E501
) -> dict:  # noqa: E501
    """Compare multiple modules on same test cases.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        test_cases: List of {context, last_message, category, ideal_reply}  # noqa: E501
        modules: Dict of {name: module}  # noqa: E501
        judge_client: Optional judge client for scoring  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Comparison results by module  # noqa: E501
    """  # noqa: E501
    import time  # noqa: E501
  # noqa: E501
    results = {name: {"replies": [], "latencies": [], "scores": []} for name in modules}  # noqa: E501
  # noqa: E501
    for tc in test_cases:  # noqa: E501
        for name, module in modules.items():  # noqa: E501
            start = time.perf_counter()  # noqa: E501
            try:  # noqa: E501
                pred = module(**tc)  # noqa: E501
                reply = pred.reply.strip()  # noqa: E501
            except Exception as e:  # noqa: E501
                reply = f"[ERROR: {e}]"  # noqa: E501
            latency = time.perf_counter() - start  # noqa: E501
  # noqa: E501
            results[name]["replies"].append(reply)  # noqa: E501
            results[name]["latencies"].append(latency * 1000)  # noqa: E501
  # noqa: E501
            # Score if judge available  # noqa: E501
            if judge_client:  # noqa: E501
                # TODO: Implement judge scoring  # noqa: E501
                pass  # noqa: E501
  # noqa: E501
    # Compute stats  # noqa: E501
    for name in results:  # noqa: E501
        latencies = results[name]["latencies"]  # noqa: E501
        results[name]["avg_latency"] = sum(latencies) / len(latencies) if latencies else 0  # noqa: E501
        results[name]["p95_latency"] = (  # noqa: E501
            sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    return results  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    # Quick test  # noqa: E501
    print("Enhanced DSPy modules loaded.")  # noqa: E501
    print("\nAvailable modules:")  # noqa: E501
    print("  - StyleAwareReplyModule")  # noqa: E501
    print("  - TwoStageReplyModule")  # noqa: E501
    print("  - EnhancedCategoryModule")  # noqa: E501
    print("\nAvailable presets:")  # noqa: E501
    for name, config in OPTIMIZATION_PRESETS.items():  # noqa: E501
        print(f"  - {name}: {config['description']}")  # noqa: E501
