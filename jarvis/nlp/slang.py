"""Slang Expansion - Expand common slang/abbreviations for better embedding alignment.

iMessage conversations contain a lot of slang and abbreviations that embedding models
(trained on formal text) may not understand well. This module provides a simple
dictionary-based expansion to improve embedding quality.

Usage:
    from jarvis.nlp.slang import expand_slang

    text = "u coming rn?"
    expanded = expand_slang(text)  # "you coming right now?"
"""

import re

# Slang/abbreviation expansion map
# Keys are lowercase, values are the expanded form
SLANG_MAP: dict[str, str] = {
    # Pronouns and basics
    "u": "you",
    "ur": "your",
    "r": "are",
    "y": "why",
    "n": "and",
    "b": "be",
    "c": "see",
    "k": "okay",
    # Common abbreviations
    "rn": "right now",
    "nvm": "never mind",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "lmk": "let me know",
    "idk": "I don't know",
    "idc": "I don't care",
    "omw": "on my way",
    "otw": "on the way",
    "btw": "by the way",
    "wya": "where you at",
    "wbu": "what about you",
    "hbu": "how about you",
    "ily": "I love you",
    "ilysm": "I love you so much",
    "brb": "be right back",
    "ttyl": "talk to you later",
    "jk": "just kidding",
    "np": "no problem",
    "ty": "thank you",
    "yw": "you're welcome",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "tho": "though",
    "bc": "because",
    "cuz": "because",
    "b4": "before",
    "2day": "today",
    "2morrow": "tomorrow",
    "2nite": "tonight",
    "tmrw": "tomorrow",
    "tmw": "tomorrow",
    # With/without
    "w/": "with",
    "w/o": "without",
    "wo": "without",
    # Affirmations and reactions
    "ya": "yeah",
    "yea": "yeah",
    "yep": "yes",
    "yup": "yes",
    "nah": "no",
    "nope": "no",
    "aight": "alright",
    "ight": "alright",
    "aite": "alright",
    "oki": "okay",
    "kk": "okay",
    "ofc": "of course",
    "def": "definitely",
    "obvi": "obviously",
    "prolly": "probably",
    "probs": "probably",
    "sry": "sorry",
    "srry": "sorry",
    "mb": "my bad",
    # Questions
    "wdym": "what do you mean",
    "wym": "what you mean",
    "hmu": "hit me up",
    "lmao": "laughing my ass off",
    "lol": "laughing out loud",
    "rofl": "rolling on the floor laughing",
    # Slang expressions
    "rly": "really",
    "rlly": "really",
    "v": "very",
    "sm": "so much",
    "fyi": "for your information",
    "iirc": "if I recall correctly",
    "afaik": "as far as I know",
    "asap": "as soon as possible",
    "omg": "oh my god",
    "omfg": "oh my god",
    "wtf": "what the fuck",
    "wth": "what the hell",
    "smh": "shaking my head",
    "ngl": "not gonna lie",
    "fr": "for real",
    "frfr": "for real for real",
    "lowkey": "kind of",
    "highkey": "definitely",
    "bet": "definitely",  # Slang for agreement, not just acknowledgment
    "fam": "family",
    "bro": "brother",
    "sis": "sister",
    "finna": "going to",
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "kinda": "kind of",
    "sorta": "sort of",
    "lemme": "let me",
    "gimme": "give me",
    "dunno": "don't know",
    "gotcha": "got you",
    "gn": "good night",
    "gm": "good morning",
    "cya": "see you",
    "ttys": "talk to you soon",
    "bbl": "be back later",
    "afk": "away from keyboard",
    "dm": "direct message",
    "irl": "in real life",
    "tmi": "too much information",
    # Gen-Z slang additions
    "slay": "amazing",
    "bussin": "really good",
    "mid": "mediocre",
    "sus": "suspicious",
    "cap": "lie",
    "no cap": "no lie",
    "periodt": "period",
    "snatched": "looking good",
    "stan": "big fan",
    "simp": "overly devoted",
    "hits different": "feels unique",
    "main character": "protagonist energy",
    "understood the assignment": "did it perfectly",
    "rent free": "constantly thinking about",
    "its giving": "it resembles",
    "ate": "did exceptionally well",
    "bestie": "best friend",
    "girlie": "girl",
    "bffr": "be for real",
    "ong": "on god",
    "npc": "uninteresting person",
    "gyat": "exclamation of attraction",
    "rizz": "charisma",
    "no rizz": "no charisma",
    "valid": "acceptable",
    "based": "admirable",
    "cringe": "embarrassing",
    "vibe": "atmosphere",
    "vibes": "atmosphere",
    "vibing": "relaxing",
    "yeet": "throw",
    "ghosted": "ignored",
    "ghosting": "ignoring",
    "deadass": "seriously",
    "slaps": "is great",
    "fire": "excellent",
    "lit": "exciting",
    "goat": "greatest of all time",
    "goated": "the best",
    "tea": "gossip",
    "spill the tea": "share the gossip",
    "salty": "bitter",
    "extra": "over the top",
    "flex": "show off",
    "flexing": "showing off",
    "thicc": "curvy",
    "clout": "influence",
    "fomo": "fear of missing out",
    "jomo": "joy of missing out",
    "oof": "expression of discomfort",
    "yikes": "expression of concern",
}

# Pre-compile regex for word boundary matching
# We need to match whole words only to avoid expanding "burn" -> "beurn"
_WORD_BOUNDARY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in SLANG_MAP) + r")\b", re.I
)


def expand_slang(text: str) -> str:
    """Expand common slang/abbreviations for better embedding alignment.

    Uses word-boundary matching to avoid partial word replacements.
    Preserves original capitalization where possible.

    Args:
        text: Text with potential slang/abbreviations.

    Returns:
        Text with slang expanded to full forms.

    Examples:
        >>> expand_slang("u coming rn?")
        'you coming right now?'
        >>> expand_slang("wya? hmu when ur free")
        'where you at? hit me up when your free'
    """
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in SLANG_MAP:
            replacement = SLANG_MAP[lower]
            # Try to preserve capitalization
            # Check for all-caps only on multi-char words
            if len(word) > 1 and word.isupper():
                return replacement.upper()
            elif word[0].isupper():
                return replacement.capitalize()
            return replacement
        return word

    return _WORD_BOUNDARY_PATTERN.sub(_replace, text)


def get_slang_map() -> dict[str, str]:
    """Get a copy of the slang expansion map.

    Returns:
        Dictionary mapping slang terms to their expansions.
    """
    return SLANG_MAP.copy()


__all__ = ["expand_slang", "get_slang_map", "SLANG_MAP"]
