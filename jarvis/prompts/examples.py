"""Few-shot examples and metadata for prompt engineering.

Contains all static example data used across the prompt system:
reply generation, summarization, search, and thread-specific examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# =============================================================================
# Prompt Metadata & Versioning
# =============================================================================

PROMPT_VERSION = "1.0.0"
PROMPT_LAST_UPDATED = "2026-01-26"


@dataclass
class PromptMetadata:
    """Metadata for a prompt or example set.

    Attributes:
        name: Human-readable name for the prompt/example set
        version: Semantic version string (e.g., "1.0.0")
        last_updated: ISO date string of last update
        description: Brief description of the prompt's purpose
    """

    name: str
    version: str = PROMPT_VERSION
    last_updated: str = PROMPT_LAST_UPDATED
    description: str = ""


@dataclass
class FewShotExample:
    """A few-shot example for prompt engineering.

    Attributes:
        context: The conversation context
        output: The expected output/reply
        tone: The tone of the example (casual/professional)
    """

    context: str
    output: str
    tone: Literal["casual", "professional"] = "casual"


# Reply generation examples - casual tone
CASUAL_REPLY_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[10:30] John: Want to grab lunch?",
        output="Sure! What time works for you?",
        tone="casual",
    ),
    FewShotExample(
        context="[14:00] Mom: Don't forget dinner Sunday",
        output="I'll be there! Should I bring anything?",
        tone="casual",
    ),
    FewShotExample(
        context="[09:15] Alex: Running 10 min late",
        output="No worries, see you soon!",
        tone="casual",
    ),
    FewShotExample(
        context="[18:30] Sam: Did you see the game last night?",
        output="Yes! That ending was incredible",
        tone="casual",
    ),
    FewShotExample(
        context="[11:00] Lisa: Can you pick up milk on your way home?",
        output="Sure thing, anything else?",
        tone="casual",
    ),
]

# Reply generation examples - professional tone
PROFESSIONAL_REPLY_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[09:00] Manager: Can you send the Q4 report by EOD?",
        output="Absolutely, I'll have it ready by 5 PM.",
        tone="professional",
    ),
    FewShotExample(
        context="[14:30] Client: When can we schedule a follow-up meeting?",
        output="I'm available Thursday or Friday afternoon. Would either work for you?",
        tone="professional",
    ),
    FewShotExample(
        context="[10:15] HR: Please complete your annual review form",
        output="Thank you for the reminder. I'll complete it today.",
        tone="professional",
    ),
    FewShotExample(
        context="[16:00] Colleague: Could you review my proposal draft?",
        output="Happy to help. I'll review it and send feedback by tomorrow morning.",
        tone="professional",
    ),
    FewShotExample(
        context="[11:30] Vendor: The shipment will be delayed by 2 days",
        output="Thanks for letting me know. Please send the updated tracking info when available.",
        tone="professional",
    ),
]

# Summarization examples
SUMMARIZATION_EXAMPLES: list[tuple[str, str]] = [
    (
        """[Mon 9:00] John: Hey, want to meet for coffee Tuesday?
[Mon 9:05] You: Sure, what time?
[Mon 9:10] John: How about 3pm at Blue Bottle?
[Mon 9:12] You: Perfect, see you then""",
        """Summary:
- Coffee meeting planned for Tuesday at 3pm
- Location: Blue Bottle
- Confirmed with John""",
    ),
    (
        """[Tue 14:00] Boss: Can you handle the client presentation Friday?
[Tue 14:05] You: Yes, I can do that
[Tue 14:10] Boss: Great, they want to see the Q3 numbers
[Tue 14:15] Boss: Also include the growth projections
[Tue 14:20] You: Got it, I'll prepare both""",
        """Summary:
- Action item: Prepare client presentation for Friday
- Include: Q3 numbers and growth projections
- Commitment made to prepare materials""",
    ),
    (
        """[Wed 10:00] Mom: Are you coming for Thanksgiving?
[Wed 10:15] You: Yes, I'll be there
[Wed 10:20] Mom: Can you bring your famous pie?
[Wed 10:25] You: Of course!
[Wed 10:30] Mom: Dinner is at 4pm""",
        """Summary:
- Thanksgiving plans confirmed
- Bringing pie (action item)
- Dinner time: 4pm""",
    ),
]

# Search/question answering examples
SEARCH_ANSWER_EXAMPLES: list[tuple[str, str, str]] = [
    (
        """[Mon] John: Let's meet at 123 Main St
[Tue] John: Actually, let's do 456 Oak Ave instead""",
        "Where are we meeting?",
        "You're meeting at 456 Oak Ave (John changed it from the original 123 Main St).",
    ),
    (
        """[10:00] Sarah: The project deadline is March 15th
[10:30] Sarah: We need the draft by March 10th for review""",
        "When is the deadline?",
        "The project deadline is March 15th, with a draft due March 10th for review.",
    ),
    (
        """[Mon] Alex: I recommend trying Sushi Palace
[Mon] Alex: They have great lunch specials
[Tue] Mom: That new Italian place is good too""",
        "What restaurants were recommended?",
        "Two restaurants were recommended: Sushi Palace (by Alex, noted for lunch specials) "
        "and a new Italian place (by Mom).",
    ),
]


# =============================================================================
# Thread-Specific Few-Shot Examples
# =============================================================================

# Examples for logistics/coordination threads - concise responses
LOGISTICS_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[14:00] Sarah: What time works for pickup?\n[14:05] Me: How about 5pm?",
        output="5pm works! See you then.",
        tone="casual",
    ),
    FewShotExample(
        context="[10:00] John: Running late, stuck in traffic\n[10:05] John: Be there in 15",
        output="No worries, take your time!",
        tone="casual",
    ),
    FewShotExample(
        context="[18:30] Mom: Which entrance should I use?",
        output="Use the main entrance on Oak Street. I'll meet you there.",
        tone="casual",
    ),
]

# Examples for emotional support threads - empathetic responses
EMOTIONAL_SUPPORT_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context=(
            "[20:00] Sarah: Had the worst day at work today\n"
            "[20:05] Sarah: My project got cancelled"
        ),
        output=(
            "I'm so sorry to hear that. That's really frustrating, "
            "especially after all your hard work. Want to talk about it?"
        ),
        tone="casual",
    ),
    FewShotExample(
        context=(
            "[15:00] John: Just found out I didn't get the job\n[15:02] John: Feeling pretty down"
        ),
        output=(
            "That's really disappointing, I know you were excited about it. "
            "Their loss though - something better will come along. I'm here if you need to vent."
        ),
        tone="casual",
    ),
    FewShotExample(
        context="[09:00] Mom: Missing dad a lot today",
        output=(
            "I know, I miss him too. It's okay to have those days. "
            "Want to look through some old photos together later?"
        ),
        tone="casual",
    ),
]

# Examples for planning threads - includes action items
PLANNING_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context=(
            "[14:00] Alex: Want to do dinner Saturday?\n"
            "[14:05] Me: Sure!\n"
            "[14:10] Alex: Any restaurant preferences?"
        ),
        output=(
            "How about Italian? There's a new place downtown I've been wanting to try. "
            "I can make a reservation for 7pm if that works?"
        ),
        tone="casual",
    ),
    FewShotExample(
        context=(
            "[10:00] Sarah: We should plan a birthday surprise for Mike\n"
            "[10:05] Sarah: Any ideas?"
        ),
        output=(
            "Definitely! We could do a small surprise at his favorite bar. "
            "I can coordinate with Lisa on decorations. Should we aim for next Friday?"
        ),
        tone="casual",
    ),
    FewShotExample(
        context=(
            "[16:00] John: Thinking about a camping trip next month\n"
            "[16:05] John: You interested?"
        ),
        output=(
            "Count me in! I can bring the tent and cooler. "
            "What dates are you thinking? Should I check with the others too?"
        ),
        tone="casual",
    ),
]

# Examples for catching up threads - warm, conversational
CATCHING_UP_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[19:00] Lisa: Haven't heard from you in ages! How are things?",
        output=(
            "I know, it's been too long! Things are good - busy with work but can't complain. "
            "How about you? Still enjoying the new job?"
        ),
        tone="casual",
    ),
    FewShotExample(
        context=(
            "[11:00] College Friend: Just saw your post about the promotion!\n"
            "[11:02] College Friend: So proud of you!"
        ),
        output=(
            "Thanks so much! It's been a wild ride. "
            "We should catch up properly soon - coffee this weekend?"
        ),
        tone="casual",
    ),
]

# Examples for quick exchange threads - brief responses
QUICK_EXCHANGE_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[12:00] Tom: Got the tickets!",
        output="Awesome, thanks!",
        tone="casual",
    ),
    FewShotExample(
        context="[09:00] Boss: Can you join the 2pm call?",
        output="Yes, I'll be there.",
        tone="professional",
    ),
]

# Thread examples organized by topic for the registry
THREAD_EXAMPLES: dict[str, list[FewShotExample]] = {
    "logistics": LOGISTICS_THREAD_EXAMPLES,
    "warm": EMOTIONAL_SUPPORT_THREAD_EXAMPLES,
    "planning": PLANNING_THREAD_EXAMPLES,
    "social": CATCHING_UP_THREAD_EXAMPLES,
    "brief": QUICK_EXCHANGE_THREAD_EXAMPLES,
    # Legacy aliases for thread topic enum values
    "emotional_support": EMOTIONAL_SUPPORT_THREAD_EXAMPLES,
    "catching_up": CATCHING_UP_THREAD_EXAMPLES,
    "quick_exchange": QUICK_EXCHANGE_THREAD_EXAMPLES,
}
