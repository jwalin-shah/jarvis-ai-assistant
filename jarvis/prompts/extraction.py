"""Extraction prompts and model config for instruction-based fact extraction.

Used by InstructionFactExtractor (sentence-to-triple pipeline) and shared
model paths with BatchedInstructionFactExtractor.
"""

from __future__ import annotations

# Model path by tier (single source for both extractors)
EXTRACTION_MODELS: dict[str, str] = {
    "1.2b": "models/lfm2-1.2b-extract-mlx-4bit",
    "0.7b": "models/lfm-0.7b-4bit",
    "350m": "models/lfm2-350m-extract-mlx-4bit",
}

# --- Single-Pass Triple Extraction ---
EXTRACTION_SYSTEM_PROMPT = """Extract durable facts from chat turns.
FORMAT: - [Person Name] | [lives in/works at/likes/etc] | [Fact Detail]
RULES:
- ONLY stable facts (identity, job, location, habits, preferences).
- NO one-off logistics, news, or junk.
- Use 3rd-person wording. Subject must be a specific person or "{user_name}".
- If no claims, output exactly: NONE
- No commentary or markdown."""

EXTRACTION_USER_PROMPT = """Conversation:
{text}

Triples:
- """

# Shims for compatibility while we migrate away from two-pass
VERIFY_SYSTEM_PROMPT = "Convert to triples: - [Subject] | [Predicate] | [Object]"
VERIFY_USER_PROMPT = "{facts}"

# --- Batched Extraction ---
BATCH_EXTRACTION_USER_PROMPT = """Conversation Segments:
{segments_text}

Return JSONL durable claims now (or NONE):
"""

BATCH_VERIFY_SYSTEM_PROMPT = """You are a precise fact-checker and structurer.
Review these extracted claims against the conversation segments and convert them into triples.

RULES:
1. Output format: [Segment N] - [Subject] | [Predicate] | [Object]
2. Subject must be a SPECIFIC PERSON'S NAME or "{user_name}".
3. Predicate: Short verb phrase (1-3 words).
4. Object: Capture the essential detail (1-4 words).
5. Ensure the [Segment N] prefix matches the source segment exactly.
6. Remove conversational filler, duplicates, or ungrounded claims.
7. DO NOT output any other text or commentary."""

BATCH_VERIFY_USER_PROMPT = """Conversation Segments for Context:
{segments_text}

Claims to Verify and Structure:
{facts}

Verified Structured Facts ([Segment N] - Subject | Predicate | Object):
- """

# Steer the model away from "talking" or hedging
NEGATIVE_CONSTRAINTS: list[str] = [
    "Note:",
    "Commentary:",
    "Unverified:",
    "The fact",
    "supported by",
    "appears to",
    "suggests",
    "likely",
    "possibly",
    "maybe",
    "No mention",
    "Not specified",
    "Unknown",
    "Doesn't say",
    "The user",
    "The contact",
    "According to",
    "Group",
    "Chat",
]
