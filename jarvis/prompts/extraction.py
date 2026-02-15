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

# --- Pass 1: Claim extraction (JSONL) ---
EXTRACTION_SYSTEM_PROMPT = """You extract durable personal facts from chat turns.

Task:
- Return ONLY stable personal claims that are useful for a long-lived profile.

Allowed claim types:
- identity/relationship (family, partner, close friend)
- work/school/role/company
- home/current location (not temporary travel)
- durable preferences (likes/dislikes/habits)
- stable schedule preference (e.g., "usually free after 6")

Do NOT extract:
- one-off logistics (meetups, ETAs, "on my way", "can we call", orders, deliveries)
- sports/news chatter, jokes, reactions, tapbacks
- meta speech ("X said", "X asked", "X mentioned")
- facts about group chats/platforms/companies unless clearly about a person
- speculative/uncertain claims

Output format:
- Output JSONL only: one JSON object per line, no markdown.
- Schema:
  {{"subject": "<person>", "speaker": "<speaker>", "claim": "<durable fact>",
   "evidence_quote": "<exact quote from conversation>"}}
- Max 3 claims per segment.
- If no claims at all, output exactly: NONE

Hard rules:
- Only explicit claims from text (no inference).
- Use 3rd-person wording.
- Subject must be a person, never a group/chat.
- `evidence_quote` must be verbatim text copied from the conversation.
- Do not use placeholders, brackets, or variables (no "[City]", "[Job Title]", "<unknown>").
- No headings, markdown, commentary, or extra labels."""

EXTRACTION_USER_PROMPT = """Chat Turns:
{text}

Return JSONL durable claims now (or NONE):
"""

# --- Pass 2: Structure claims as triples ---
VERIFY_SYSTEM_PROMPT = """You are a precise fact-structurer.
Convert natural language claims into structured [Subject] | [Predicate] | [Object] triples.

RULES:
1. Output ONLY the triples in format: - [Subject] | [Predicate] | [Object]
2. Subject must be a SPECIFIC PERSON'S NAME or "{user_name}".
   - NEVER use group names (e.g., "{contact_name}", "The Chat") as the Subject.
3. Predicate: Short verb phrase (1-3 words). e.g., "lives in", "likes", "works at".
   - NO long sentences. NO snake_case.
4. Object: Capture the essential detail (1-4 words). NO MARKDOWN.
5. DO NOT output any other text or commentary."""

VERIFY_USER_PROMPT = """Chat for Context:
{text}

Claims to Structure:
{facts}

Structured Personal Facts (Subject | Predicate | Object):
- """

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
