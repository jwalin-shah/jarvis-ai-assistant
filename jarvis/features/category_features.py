"""Shared feature extraction for category classification.

Single source of truth for all category classification features.
Eliminates train/serve skew by using identical feature extraction.

Feature layout (531 total):
- 384 BERT embeddings (via embedder.encode, normalized)
- 26 hand-crafted features (structure, mobilization, context, reactions)
- 94 spaCy features (14 original + 80 new: NER, deps, tokens, morphology)
- 19 new hand-crafted features (8 error-analysis + 11 high-value additions)
- 8 hard-class features (closing/request specific)
Total: 147 non-BERT features, 531 total with BERT

Usage:
    from jarvis.features.category_features import CategoryFeatureExtractor

    extractor = CategoryFeatureExtractor()
    features = extractor.extract_all(
        text="Want to grab lunch?",
        context=["Hey", "What's up"],
        mob_pressure="high",
        mob_type="answer"
    )
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import spacy

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Shared regex patterns (reused from text_normalizer)
EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
)

ABBREVIATION_RE = re.compile(
    r"\b(lol|lmao|omg|wtf|brb|btw|smh|tbh|imo|idk|ngl|fr|rn|ong|nvm|wya|hmu|"
    r"fyi|asap|dm|irl|fomo|goat|sus|bet|cap|no cap)\b",
    re.IGNORECASE,
)

PROFESSIONAL_KEYWORDS_RE = re.compile(
    r"\b(meeting|deadline|project|report|schedule|conference|presentation|"
    r"budget|client|invoice|proposal)\b",
    re.IGNORECASE,
)

# Feature extraction patterns
REACTION_PATTERNS = ["Laughed at", "Loved", "Liked", "Disliked", "Emphasized", "Questioned"]
EMOTIONAL_MARKERS = ["lmao", "lol", "xd", "haha", "omg", "bruh", "rip", "lmfao", "rofl"]
QUESTION_STARTERS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "did",
    "do",
    "does",
    "can",
    "could",
    "would",
    "will",
    "should",
}
IMPERATIVE_VERBS = {
    "make",
    "send",
    "get",
    "tell",
    "show",
    "give",
    "come",
    "take",
    "call",
    "help",
    "let",
}
REQUEST_VERBS = {"send", "give", "help", "tell", "show", "let", "call", "get", "make", "take"}
PROMISE_VERBS = {"promise", "guarantee", "commit", "swear"}
AGREEMENT_WORDS = {
    "sure",
    "okay",
    "ok",
    "yes",
    "yeah",
    "yep",
    "yup",
    "sounds good",
    "bet",
    "fs",
}
FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "myself"}
SECOND_PERSON_PRONOUNS = {"you", "your", "yours", "yourself"}
THIRD_PERSON_PRONOUNS = {
    "he",
    "she",
    "it",
    "they",
    "him",
    "her",
    "them",
    "his",
    "hers",
    "their",
}
POS_TAGS = ["VERB", "NOUN", "ADJ", "ADV", "PRON", "DET", "ADP", "INTJ"]
FINE_GRAINED_TAGS = [
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "MD",
    "WDT",
    "WP",
    "WRB",
    "UH",
    "JJR",
]
DEP_LABELS = [
    "nsubj",
    "dobj",
    "ROOT",
    "aux",
    "neg",
    "advmod",
    "amod",
    "prep",
    "pobj",
    "conj",
    "ccomp",
    "xcomp",
    "acl",
    "relcl",
    "mark",
]
ENTITY_TYPES_ORIGINAL = ["PERSON", "DATE", "TIME", "GPE"]
ENTITY_TYPES_NEW = [
    "MONEY",
    "CARDINAL",
    "ORDINAL",
    "PERCENT",
    "QUANTITY",
    "FAC",
    "PRODUCT",
    "EVENT",
    "LANGUAGE",
    "LAW",
    "WORK_OF_ART",
    "NORP",
    "LOC",
    "ORG",
]
PAST_TENSE_TAGS = {"VBD", "VBN"}
PRESENT_TENSE_TAGS = {"VBP", "VBZ", "VBG"}
BRIEF_AGREEMENTS = {"ok", "okay", "k", "yeah", "yep", "yup", "sure", "cool", "bet", "fs", "aight"}

# New hand-crafted feature patterns (from error analysis)
CLAUSE_AFTER_AGREEMENT_RE = re.compile(
    r"\b(ok|okay|sure|yeah|yep|yup|cool|bet|sounds good)\b.*\b(but|though|however|if|when|"
    r"because|since|as long as|unless|after|before)\b",
    re.IGNORECASE,
)

GREETING_PATTERN_RE = re.compile(
    r"^(hey|hi|hello|yo|sup|what's up|wassup|heyy|hiya|heya)\b", re.IGNORECASE
)

ELONGATED_WORD_RE = re.compile(r"(\w)\1{2,}")  # e.g., "yaasss", "nooo", "omgggg"

# Proposal patterns from text_normalizer
PROPOSAL_PATTERNS_RE = re.compile(
    r"\b(how about|what about|should we|shall we|let's|lets|we could)\b", re.IGNORECASE
)

# Implicit request patterns
IMPLICIT_REQUEST_RE = re.compile(
    r"\b(i need|i want|i wanna|i gotta|i have to)\b.*\b(you|your|help|know)\b", re.IGNORECASE
)

# === HARD-CLASS FEATURES (for closing and request) ===

# Closing-specific patterns
GOODBYE_PHRASES = {
    "bye",
    "goodbye",
    "later",
    "ttyl",
    "talk to you later",
    "gotta go",
    "gtg",
    "see you",
    "see ya",
    "catch you",
    "peace",
    "cya",
    "l8r",
    "take care",
    "talk soon",
    "talk later",
    "goodnight",
    "gnight",
    "nite",
    "sleep well",
}

CLOSING_EMOJIS = {"👋", "✌️", "✌", "💯", "✨", "🙏", "😴", "😊", "💤"}

TIME_CONSTRAINT_PHRASES = {
    "gotta run",
    "gtg",
    "g2g",
    "have to go",
    "running late",
    "in a hurry",
    "gotta jet",
    "about to leave",
}

# Request-specific patterns
POLITE_REQUEST_MODAL_RE = re.compile(r"^(can|could|would|will)\s+(you|u)\s+", re.IGNORECASE)

IMPERATIVE_WITH_PLEASE_RE = re.compile(r"^(please\s+\w+|(\w+\s+)*please)", re.IGNORECASE)

CONDITIONAL_REQUEST_RE = re.compile(
    r"\b(if you (could|can|would)|when you (can|get a chance|have time)|whenever you)\b",
    re.IGNORECASE,
)

NEED_WANT_YOU_RE = re.compile(
    r"\b(i need you to|i want you to|i'd like you to|need you to|want you to)\b", re.IGNORECASE
)

# Additional feature patterns (context overlap, question types, thanks/apology, urgency, emoji sentiment)
THANKS_MARKERS = {"thanks", "thank you", "thx", "ty", "tysm", "appreciate", "appreciated"}
APOLOGY_MARKERS = {"sorry", "my bad", "apologize", "apologies", "oops", "whoops", "my fault"}
URGENCY_MARKERS = {
    "asap",
    "urgent",
    "quick",
    "quickly",
    "hurry",
    "rush",
    "rushing",
    "now",
    "right now",
    "immediately",
}
FUTURE_TIME_MARKERS = {
    "later",
    "tomorrow",
    "tonight",
    "next week",
    "next time",
    "soon",
    "eventually",
}

# Emoji sentiment sets (common emotional emojis)
POSITIVE_EMOJIS = {
    "😊",
    "😀",
    "😁",
    "😂",
    "🤣",
    "😃",
    "😄",
    "😆",
    "😉",
    "😍",
    "🥰",
    "😘",
    "❤️",
    "💕",
    "💖",
    "👍",
    "👏",
    "🙌",
    "🎉",
    "🎊",
    "✨",
    "💯",
    "🔥",
    "😎",
    "🤩",
    "😇",
    "🥳",
    "💪",
    "🙏",
    "💗",
}
NEGATIVE_EMOJIS = {
    "😢",
    "😭",
    "😞",
    "😔",
    "😟",
    "😕",
    "🙁",
    "☹️",
    "😣",
    "😖",
    "😫",
    "😩",
    "😤",
    "😠",
    "😡",
    "🤬",
    "💔",
    "😰",
    "😨",
    "😱",
    "😥",
    "😪",
    "🤦",
    "🤷",
    "💀",
    "😒",
    "🙄",
}
NEUTRAL_EMOJIS = {"🤔", "🙃", "😐", "😑", "🤨", "🧐", "😶", "👀", "💬", "🗣️", "👋", "✌️"}


class CategoryFeatureExtractor:
    """Single source of truth for category classification features.

    Parse spaCy doc once, reuse for all feature extraction methods.
    """

    def __init__(self, nlp: spacy.Language | None = None) -> None:
        """Initialize feature extractor.

        Args:
            nlp: Optional spaCy model. If None, lazy-loads en_core_web_sm.
        """
        self._nlp = nlp
        self._spacy_available: bool | None = None  # None = not checked yet

    @property
    def nlp(self) -> spacy.Language:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            if self._spacy_available is False:
                raise RuntimeError("spaCy model en_core_web_sm not available")
            try:
                self._nlp = spacy.load("en_core_web_sm")
                self._spacy_available = True
            except OSError as e:
                self._spacy_available = False
                raise RuntimeError(
                    "spaCy model en_core_web_sm not installed. "
                    "Run: python -m spacy download en_core_web_sm"
                ) from e
        return self._nlp

    def extract_hand_crafted(
        self,
        text: str,
        context: list[str] | None = None,
        mob_pressure: str = "none",
        mob_type: str = "answer",
    ) -> NDArray[np.float32]:
        """Extract 26 original hand-crafted features.

        Args:
            text: Message text
            context: Previous messages in conversation
            mob_pressure: Mobilization pressure level (high/medium/low/none)
            mob_type: Mobilization response type (commitment/answer/emotional)

        Returns:
            26-dim feature array
        """
        context = context or []
        features: list[float] = []
        text_lower = text.lower()
        words = text.split()
        total_words = len(words)

        # Message structure (5)
        features.append(float(len(text)))
        features.append(float(total_words))
        features.append(float(text.count("?")))
        features.append(float(text.count("!")))
        features.append(float(len(EMOJI_RE.findall(text))))

        # Mobilization one-hots (7)
        for level in ("high", "medium", "low", "none"):
            features.append(1.0 if mob_pressure == level else 0.0)
        for rtype in ("commitment", "answer", "emotional"):
            features.append(1.0 if mob_type == rtype else 0.0)

        # Tone flags (2)
        features.append(1.0 if PROFESSIONAL_KEYWORDS_RE.search(text) else 0.0)
        features.append(1.0 if ABBREVIATION_RE.search(text) else 0.0)

        # Context features (3)
        features.append(float(len(context)))
        avg_ctx_len = float(np.mean([len(m) for m in context])) if context else 0.0
        features.append(avg_ctx_len)
        features.append(1.0 if len(context) == 0 else 0.0)

        # Style features (2)
        abbr_count = len(ABBREVIATION_RE.findall(text))
        features.append(abbr_count / max(total_words, 1))
        capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
        features.append(capitalized / max(len(words) - 1, 1))

        # Reaction/emotion features (7)
        is_reaction_msg = 1.0 if any(text.startswith(p) for p in REACTION_PATTERNS) else 0.0
        features.append(is_reaction_msg)

        emotional_count = sum(text_lower.count(marker) for marker in EMOTIONAL_MARKERS)
        features.append(float(emotional_count))

        last_word = words[-1].lower() if words else ""
        ends_with_emotion = 1.0 if last_word in EMOTIONAL_MARKERS else 0.0
        features.append(ends_with_emotion)

        first_word = words[0].lower() if words else ""
        question_first = 1.0 if first_word in QUESTION_STARTERS else 0.0
        features.append(question_first)

        imperative_first = 1.0 if first_word in IMPERATIVE_VERBS else 0.0
        features.append(imperative_first)

        is_brief_agreement = (
            1.0 if total_words <= 3 and any(w.lower() in BRIEF_AGREEMENTS for w in words) else 0.0
        )
        features.append(is_brief_agreement)

        exclamatory = 1.0 if ((text.endswith("!") or text.isupper()) and total_words <= 5) else 0.0
        features.append(exclamatory)

        return np.array(features, dtype=np.float32)

    def _extract_original_14(
        self,
        text: str,
        text_lower: str,
        doc: spacy.tokens.Doc,
    ) -> tuple[list[float], float, float, int, int]:
        """Extract original 14 features. Returns (features, you_modal, modal_count, first_person, second_person)."""
        features: list[float] = []
        total_tokens = len(doc)

        has_imperative = (
            1.0 if len(doc) > 0 and doc[0].pos_ == "VERB" and doc[0].tag_ == "VB" else 0.0
        )
        features.append(has_imperative)

        you_modal = (
            1.0
            if any(
                p in text_lower
                for p in ["can you", "could you", "would you", "will you", "should you"]
            )
            else 0.0
        )
        features.append(you_modal)

        has_request = 1.0 if any(token.lemma_ in REQUEST_VERBS for token in doc) else 0.0
        features.append(has_request)

        starts_modal = 1.0 if len(doc) > 0 and doc[0].tag_ in ("MD", "VB") else 0.0
        features.append(starts_modal)

        features.append(1.0 if you_modal and "?" in text else 0.0)

        i_will = (
            1.0
            if any(p in text_lower for p in ["i'll", "i will", "i'm gonna", "ima", "imma"])
            else 0.0
        )
        features.append(i_will)

        has_promise = 1.0 if any(token.lemma_ in PROMISE_VERBS for token in doc) else 0.0
        features.append(has_promise)

        first_person = sum(1 for token in doc if token.text.lower() in FIRST_PERSON_PRONOUNS)
        features.append(float(first_person))

        has_agreement = 1.0 if any(word in text_lower for word in AGREEMENT_WORDS) else 0.0
        features.append(has_agreement)

        modal_count = sum(1 for token in doc if token.tag_ == "MD")
        features.append(float(modal_count))

        features.append(float(sum(1 for token in doc if token.pos_ == "VERB")))

        second_person = sum(1 for token in doc if token.text.lower() in SECOND_PERSON_PRONOUNS)
        features.append(float(second_person))

        has_neg = 1.0 if any(token.dep_ == "neg" for token in doc) else 0.0
        features.append(has_neg)

        is_question = (
            1.0
            if "?" in text or any(token.tag_ in ("WDT", "WP", "WP$", "WRB") for token in doc)
            else 0.0
        )
        features.append(is_question)

        return features, you_modal, float(modal_count), first_person, second_person

    @staticmethod
    def _extract_pos_ratios(doc: spacy.tokens.Doc, total_tokens: int) -> list[float]:
        """Extract 8 POS ratio features."""
        pos_counts = {p: 0 for p in POS_TAGS}
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
        return [pos_counts[pos] / max(total_tokens, 1) for pos in POS_TAGS]

    @staticmethod
    def _extract_fine_grained_tags(doc: spacy.tokens.Doc) -> list[float]:
        """Extract 12 fine-grained tag presence features."""
        tag_set = {token.tag_ for token in doc}
        return [1.0 if tag in tag_set else 0.0 for tag in FINE_GRAINED_TAGS]

    @staticmethod
    def _extract_dependency_counts(doc: spacy.tokens.Doc, total_tokens: int) -> list[float]:
        """Extract 15 dependency relation ratio features."""
        dep_counts = {d: 0 for d in DEP_LABELS}
        for token in doc:
            if token.dep_ in dep_counts:
                dep_counts[token.dep_] += 1
        return [dep_counts[dep] / max(total_tokens, 1) for dep in DEP_LABELS]

    @staticmethod
    def _extract_entity_features(doc: spacy.tokens.Doc, total_tokens: int) -> list[float]:
        """Extract 21 named entity features."""
        ent_labels = {ent.label_ for ent in doc.ents}
        features: list[float] = []
        for etype in ENTITY_TYPES_ORIGINAL:
            features.append(1.0 if etype in ent_labels else 0.0)
        features.append(float(len(doc.ents)))
        features.append(len(doc.ents) / max(total_tokens, 1))
        for etype in ENTITY_TYPES_NEW:
            features.append(1.0 if etype in ent_labels else 0.0)
        features.append(1.0 if ("CARDINAL" in ent_labels or "MONEY" in ent_labels) else 0.0)
        return features

    @staticmethod
    def _extract_sentence_structure(doc: spacy.tokens.Doc, total_tokens: int) -> list[float]:
        """Extract 5 sentence structure features."""
        sents = list(doc.sents)
        sent_count = len(sents)
        avg_sent_len = float(np.mean([len(s) for s in sents])) if sents else 0.0
        max_sent_len = float(max(len(s) for s in sents)) if sents else 0.0
        noun_chunks = list(doc.noun_chunks)
        return [
            float(sent_count),
            avg_sent_len,
            max_sent_len,
            float(len(noun_chunks)),
            len(noun_chunks) / max(total_tokens, 1),
        ]

    @staticmethod
    def _extract_token_properties(doc: spacy.tokens.Doc, total_tokens: int) -> list[float]:
        """Extract 11 token property features."""
        t = max(total_tokens, 1)
        stop_count = sum(1 for token in doc if token.is_stop)
        alpha_count = sum(1 for token in doc if token.is_alpha)
        digit_count = sum(1 for token in doc if token.is_digit)
        avg_word_len = (
            float(np.mean([len(token.text) for token in doc])) if total_tokens > 0 else 0.0
        )
        punct_count = sum(1 for token in doc if token.is_punct)
        url_count = sum(1 for token in doc if token.like_url)
        like_num_count = sum(1 for token in doc if token.like_num)
        like_email_count = sum(1 for token in doc if token.like_email)
        is_currency_count = sum(1 for token in doc if token.is_currency)
        is_quote_count = sum(1 for token in doc if token.is_quote)
        return [
            stop_count / t,
            alpha_count / t,
            digit_count / t,
            avg_word_len,
            punct_count / t,
            float(url_count),
            like_num_count / t,
            float(like_email_count),
            float(is_currency_count),
            float(is_quote_count),
            url_count / t,
        ]

    @staticmethod
    def _extract_morphology(
        doc: spacy.tokens.Doc,
        text_lower: str,
        total_tokens: int,
        modal_count: float,
        first_person: int,
        second_person: int,
    ) -> list[float]:
        """Extract 8 morphology features."""
        t = max(total_tokens, 1)
        past_tense_count = sum(1 for token in doc if token.tag_ in PAST_TENSE_TAGS)
        present_tense_count = sum(1 for token in doc if token.tag_ in PRESENT_TENSE_TAGS)

        has_imperative_mood = 0.0
        if len(doc) > 0:
            if doc[0].tag_ == "VB" or any(
                token.tag_ == "VB" and token.dep_ == "ROOT" for token in doc
            ):
                has_imperative_mood = 1.0

        has_conditional = (
            1.0 if modal_count > 0 and any(w in text_lower for w in ["if", "would"]) else 0.0
        )

        third_person = sum(1 for token in doc if token.text.lower() in THIRD_PERSON_PRONOUNS)

        has_passive = 0.0
        if len(doc) >= 2:
            for i in range(len(doc) - 1):
                if doc[i].lemma_ == "be" and doc[i].dep_ == "auxpass" and doc[i + 1].tag_ == "VBN":
                    has_passive = 1.0
                    break

        return [
            past_tense_count / t,
            present_tense_count / t,
            has_imperative_mood,
            has_conditional,
            first_person / t,
            second_person / t,
            third_person / t,
            has_passive,
        ]

    def extract_spacy_features(
        self,
        text: str,
        doc: spacy.tokens.Doc | None = None,
    ) -> NDArray[np.float32]:
        """Extract 94 spaCy linguistic features (14 original + 80 new).

        Args:
            text: Message text
            doc: Optional pre-parsed spaCy doc (reuse to avoid re-parsing)

        Returns:
            94-dim feature array
        """
        if doc is None:
            doc = self.nlp(text)

        text_lower = text.lower()
        total_tokens = len(doc)

        original, you_modal, modal_count, first_person, second_person = self._extract_original_14(
            text, text_lower, doc
        )

        features: list[float] = original
        features.extend(self._extract_pos_ratios(doc, total_tokens))
        features.extend(self._extract_fine_grained_tags(doc))
        features.extend(self._extract_dependency_counts(doc, total_tokens))
        features.extend(self._extract_entity_features(doc, total_tokens))
        features.extend(self._extract_sentence_structure(doc, total_tokens))
        features.extend(self._extract_token_properties(doc, total_tokens))
        features.extend(
            self._extract_morphology(
                doc,
                text_lower,
                total_tokens,
                modal_count,
                first_person,
                second_person,
            )
        )

        return np.array(features, dtype=np.float32)

    def extract_new_hand_crafted(
        self,
        text: str,
        doc: spacy.tokens.Doc | None = None,
        context: list[str] | None = None,
    ) -> NDArray[np.float32]:
        """Extract 19 new hand-crafted features (8 error analysis + 11 additional).

        Error analysis features (8):
        - Clause after agreement, aux inversion, emotional marker position, etc.

        Additional features (11):
        - Context lexical overlap (1)
        - Question types (3): yes/no, wh, tag
        - Thanks/apology markers (2)
        - Urgency/temporal markers (2)
        - Emoji sentiment (3): positive, negative, neutral

        Args:
            text: Message text
            doc: Optional pre-parsed spaCy doc
            context: Previous messages in conversation

        Returns:
            19-dim feature array
        """
        if doc is None:
            doc = self.nlp(text)

        features: list[float] = []
        text_lower = text.lower()
        words = text.split()
        total_words = len(words)

        # 1. has_clause_after_agreement (~20 errors)
        # "ok but...", "sure if...", "yeah though..."
        has_clause_after_agreement = 1.0 if CLAUSE_AFTER_AGREEMENT_RE.search(text_lower) else 0.0
        features.append(has_clause_after_agreement)

        # 2. aux_subject_inversion (~10 errors)
        # "Can I...", "Will you..." without "?" = implicit question
        aux_subject_inversion = 0.0
        if len(doc) > 1 and doc[0].tag_ == "MD" and not text.endswith("?"):
            aux_subject_inversion = 1.0
        features.append(aux_subject_inversion)

        # 3. emotional_marker_position_ratio (~10 errors)
        # Position of first emotional marker / total words (earlier = more emotional)
        emotional_marker_pos = -1
        for i, word in enumerate(words):
            if word.lower() in EMOTIONAL_MARKERS:
                emotional_marker_pos = i
                break
        emotional_marker_position_ratio = (
            emotional_marker_pos / max(total_words, 1) if emotional_marker_pos >= 0 else 1.0
        )
        features.append(emotional_marker_position_ratio)

        # 4. addressee_imperative (~8 errors)
        # NER PERSON at start + imperative verb following
        addressee_imperative = 0.0
        if len(doc) > 1:
            if doc[0].ent_type_ == "PERSON" and doc[1].tag_ == "VB":
                addressee_imperative = 1.0
        features.append(addressee_imperative)

        # 5. implicit_request (~8 errors)
        # "I need/want/wanna" + VP pattern
        implicit_request = 1.0 if IMPLICIT_REQUEST_RE.search(text_lower) else 0.0
        features.append(implicit_request)

        # 6. greeting_pattern (~6 errors)
        # "hey/hi/hello/yo/sup" + optional name
        greeting_pattern = 1.0 if GREETING_PATTERN_RE.match(text_lower) else 0.0
        features.append(greeting_pattern)

        # 7. elongated_word_ratio (~4 errors)
        # "yaasss", "nooo", "omgggg" count / total words
        elongated_words = len(ELONGATED_WORD_RE.findall(text))
        elongated_word_ratio = elongated_words / max(total_words, 1)
        features.append(elongated_word_ratio)

        # 8. proposal_pattern (~4 errors)
        # "how about", "what about", "should we", "let's"
        proposal_pattern = 1.0 if PROPOSAL_PATTERNS_RE.search(text_lower) else 0.0
        features.append(proposal_pattern)

        # === ADDITIONAL 11 FEATURES ===

        # 9. context_lexical_overlap - Strong signal for acknowledgment
        context_overlap = 0.0
        if context and len(context) >= 1:
            text_words = set(text_lower.split())
            context_words = set(" ".join(context[-2:]).lower().split())
            if text_words:
                overlap = len(text_words & context_words)
                context_overlap = overlap / len(text_words)
        features.append(context_overlap)

        # 10-12. Question type classification (3 features)
        # Yes/no question: starts with modal/aux
        is_yes_no_q = 0.0
        if len(doc) > 0 and doc[0].tag_ in ("MD", "VBP", "VBZ", "VBD") and "?" in text:
            is_yes_no_q = 1.0
        features.append(is_yes_no_q)

        # Wh-question: starts with wh-word
        is_wh_q = 0.0
        if len(doc) > 0 and doc[0].tag_ in ("WDT", "WP", "WP$", "WRB"):
            is_wh_q = 1.0
        features.append(is_wh_q)

        # Tag question: ends with pattern like "right?", "yeah?", "isn't it?"
        is_tag_q = 0.0
        if text.endswith("?"):
            last_words = " ".join(words[-3:]).lower()
            tag_patterns = ["right?", "yeah?", "ok?", "no?", "yes?", "huh?", "eh?"]
            if any(pattern in last_words for pattern in tag_patterns):
                is_tag_q = 1.0
        features.append(is_tag_q)

        # 13-14. Thanks and apology markers
        has_thanks = 1.0 if any(marker in text_lower for marker in THANKS_MARKERS) else 0.0
        features.append(has_thanks)

        has_apology = 1.0 if any(marker in text_lower for marker in APOLOGY_MARKERS) else 0.0
        features.append(has_apology)

        # 15-16. Urgency and future time markers
        has_urgency = 1.0 if any(marker in text_lower for marker in URGENCY_MARKERS) else 0.0
        features.append(has_urgency)

        has_future_time = (
            1.0 if any(marker in text_lower for marker in FUTURE_TIME_MARKERS) else 0.0
        )
        features.append(has_future_time)

        # 17-19. Emoji sentiment (positive, negative, neutral counts)
        positive_emoji_count = sum(1 for char in text if char in POSITIVE_EMOJIS)
        features.append(float(positive_emoji_count))

        negative_emoji_count = sum(1 for char in text if char in NEGATIVE_EMOJIS)
        features.append(float(negative_emoji_count))

        neutral_emoji_count = sum(1 for char in text if char in NEUTRAL_EMOJIS)
        features.append(float(neutral_emoji_count))

        return np.array(features, dtype=np.float32)

    def extract_hard_class_features(
        self,
        text: str,
        doc: spacy.tokens.Doc | None = None,
    ) -> NDArray[np.float32]:
        """Extract 8 features targeting hard classes (closing, request).

        Closing features (4):
        - has_goodbye_phrase: explicit goodbye/farewell words
        - has_closing_emoji: emoji commonly used in farewells
        - has_time_constraint: mentions of urgency/needing to leave
        - ends_with_exclamation: "later!" vs "later" (more definitive)

        Request features (4):
        - has_polite_modal_start: "can/could/would you" at message start
        - has_imperative_with_please: politeness marker with command
        - has_conditional_request: "if you could", "when you can"
        - has_need_want_you: "I need/want you to" direct request

        Args:
            text: Message text
            doc: Optional pre-parsed spaCy doc

        Returns:
            8-dim feature array
        """
        if doc is None:
            doc = self.nlp(text)

        features: list[float] = []
        text_lower = text.lower()

        # === CLOSING FEATURES ===

        # 1. has_goodbye_phrase
        has_goodbye = 1.0 if any(phrase in text_lower for phrase in GOODBYE_PHRASES) else 0.0
        features.append(has_goodbye)

        # 2. has_closing_emoji
        has_closing_emoji = 1.0 if any(emoji in text for emoji in CLOSING_EMOJIS) else 0.0
        features.append(has_closing_emoji)

        # 3. has_time_constraint
        has_time_constraint = (
            1.0 if any(phrase in text_lower for phrase in TIME_CONSTRAINT_PHRASES) else 0.0
        )
        features.append(has_time_constraint)

        # 4. ends_with_exclamation
        ends_with_exclamation = 1.0 if text.rstrip().endswith("!") else 0.0
        features.append(ends_with_exclamation)

        # === REQUEST FEATURES ===

        # 5. has_polite_modal_start
        has_polite_modal = 1.0 if POLITE_REQUEST_MODAL_RE.search(text) else 0.0
        features.append(has_polite_modal)

        # 6. has_imperative_with_please
        has_please = 1.0 if IMPERATIVE_WITH_PLEASE_RE.search(text) else 0.0
        features.append(has_please)

        # 7. has_conditional_request
        has_conditional = 1.0 if CONDITIONAL_REQUEST_RE.search(text) else 0.0
        features.append(has_conditional)

        # 8. has_need_want_you
        has_need_want = 1.0 if NEED_WANT_YOU_RE.search(text) else 0.0
        features.append(has_need_want)

        return np.array(features, dtype=np.float32)

    def extract_multilabel_indicators(
        self,
        text: str,
        doc: spacy.tokens.Doc | None = None,
    ) -> NDArray[np.float32]:
        """Extract 10 features indicating multi-label likelihood.

        These help the model learn when a message has multiple intents
        vs a single clear intent.

        Returns:
            10-dim feature array
        """
        if doc is None:
            doc = self.nlp(text)

        features: list[float] = []
        text_lower = text.lower()
        words = text.split()

        # 1. num_sentences (normalized)
        num_sentences = max(1, text.count(".") + text.count("?") + text.count("!"))
        features.append(min(num_sentences / 3.0, 1.0))  # Cap at 3

        # 2. has_conjunction
        conjunctions = ["but", "and", "also", "though", "however", "plus"]
        has_conj = 1.0 if any(f" {c} " in text_lower for c in conjunctions) else 0.0
        features.append(has_conj)

        # 3. mixed_punctuation
        has_question = "?" in text
        has_exclamation = "!" in text
        has_period = "." in text and not text.strip().endswith("...")
        mixed_punct = 1.0 if sum([has_question, has_exclamation, has_period]) >= 2 else 0.0
        features.append(mixed_punct)

        # 4. is_very_short (<4 words → likely single-label)
        is_very_short = 1.0 if len(words) < 4 else 0.0
        features.append(is_very_short)

        # 5. is_long (>25 words → might be multi-label)
        is_long = 1.0 if len(words) > 25 else 0.0
        features.append(is_long)

        # 6. punctuation_diversity
        punct_types = sum(
            [
                "?" in text,
                "!" in text,
                "," in text,
                "." in text,
                ";" in text,
            ]
        )
        features.append(min(punct_types / 3.0, 1.0))

        # 7. has_thanks_plus_more
        has_thanks = any(word in text_lower for word in ["thanks", "thank you", "thx"])
        thanks_plus_more = 1.0 if (has_thanks and len(words) > 3) else 0.0
        features.append(thanks_plus_more)

        # 8. question_with_context (long question might include statement)
        question_with_context = 1.0 if ("?" in text and len(words) > 12) else 0.0
        features.append(question_with_context)

        # 9. words_per_sentence
        words_per_sentence = len(words) / max(num_sentences, 1)
        features.append(min(words_per_sentence / 15.0, 1.0))  # Normalize

        # 10. has_discourse_marker (transitions between intents)
        discourse_markers = ["so", "anyway", "well", "btw", "also", "oh"]
        has_marker = (
            1.0
            if any(f"{m} " in text_lower or f" {m}" in text_lower for m in discourse_markers)
            else 0.0
        )
        features.append(has_marker)

        return np.array(features, dtype=np.float32)

    def extract_all(
        self,
        text: str,
        context: list[str] | None = None,
        mob_pressure: str = "none",
        mob_type: str = "answer",
    ) -> NDArray[np.float32]:
        """Extract all 147 non-BERT features.

        Parse spaCy doc ONCE, reuse for all extraction methods.

        Feature breakdown:
        - 26 hand-crafted (structure, mobilization, context, reactions)
        - 94 spaCy (POS, tags, deps, NER, tokens, morphology)
        - 19 new hand-crafted (error analysis + high-value additions)
        - 8 hard-class (closing/request specific)

        Args:
            text: Message text
            context: Previous messages in conversation
            mob_pressure: Mobilization pressure level
            mob_type: Mobilization response type

        Returns:
            147-dim feature array (26 + 94 + 19 + 8)
        """
        # Parse once, reuse
        doc = self.nlp(text)

        # Extract all feature groups
        hand_crafted = self.extract_hand_crafted(text, context, mob_pressure, mob_type)
        spacy_feats = self.extract_spacy_features(text, doc)
        new_hand_crafted = self.extract_new_hand_crafted(text, doc, context)
        hard_class_feats = self.extract_hard_class_features(text, doc)

        # Concatenate
        return np.concatenate([hand_crafted, spacy_feats, new_hand_crafted, hard_class_feats])

    def extract_all_batch(
        self,
        texts: list[str],
        contexts: list[list[str] | None] | None = None,
        mob_pressures: list[str] | None = None,
        mob_types: list[str] | None = None,
    ) -> list[NDArray[np.float32]]:
        """Extract all 147 non-BERT features for a batch of texts.

        Uses nlp.pipe() for 5-10x faster spaCy processing vs individual nlp() calls.

        Args:
            texts: List of message texts.
            contexts: Optional list of context lists (one per text).
            mob_pressures: Optional list of mobilization pressures.
            mob_types: Optional list of mobilization response types.

        Returns:
            List of 147-dim feature arrays.
        """
        if not texts:
            return []

        n = len(texts)
        if contexts is None:
            contexts = [None] * n
        if mob_pressures is None:
            mob_pressures = ["none"] * n
        if mob_types is None:
            mob_types = ["answer"] * n

        docs = list(self.nlp.pipe(texts, batch_size=50))

        results: list[NDArray[np.float32]] = []
        for i, (text, doc) in enumerate(zip(texts, docs)):
            hand_crafted = self.extract_hand_crafted(
                text,
                contexts[i],
                mob_pressures[i],
                mob_types[i],
            )
            spacy_feats = self.extract_spacy_features(text, doc)
            new_hand_crafted = self.extract_new_hand_crafted(text, doc, contexts[i])
            hard_class_feats = self.extract_hard_class_features(text, doc)
            results.append(
                np.concatenate([hand_crafted, spacy_feats, new_hand_crafted, hard_class_feats])
            )

        return results


class FeatureConfig:
    """Feature dimensions and metadata.

    Feature layout (531 total):
    - [0:384]   = BERT embedding (L2-normalized)
    - [384:410] = 26 hand-crafted (structure, mobilization, context, reactions)
    - [410:504] = 94 spaCy (POS, tags, deps, NER, tokens, morphology)
    - [504:523] = 19 new hand-crafted (error-analysis + high-value additions)
    - [523:531] = 8 hard-class (closing/request specific)

    Context BERT embeddings were removed to eliminate train-serve skew.
    Previously, context BERT (384 dims at indices 384:768) was zeroed at
    inference but present during training, causing a distribution mismatch.
    """

    # Feature group sizes
    BERT_DIM = 384
    HAND_CRAFTED_DIM = 26
    SPACY_DIM = 94  # 14 original + 80 new (15 NER + 5 deps + 5 tokens + 55 from before)
    NEW_HAND_CRAFTED_DIM = 19  # 8 error-analysis + 11 high-value additions
    HARD_CLASS_DIM = 8
    MULTILABEL_INDICATOR_DIM = 10  # Closing + request specific features (unused in hardclass)
    TOTAL_NON_BERT = HAND_CRAFTED_DIM + SPACY_DIM + NEW_HAND_CRAFTED_DIM + HARD_CLASS_DIM  # 147
    TOTAL_DIM = BERT_DIM + TOTAL_NON_BERT  # 531

    # Feature index ranges (for ColumnTransformer)
    BERT_START = 0
    BERT_END = BERT_DIM
    HAND_CRAFTED_START = BERT_END
    HAND_CRAFTED_END = HAND_CRAFTED_START + HAND_CRAFTED_DIM
    SPACY_START = HAND_CRAFTED_END
    SPACY_END = SPACY_START + SPACY_DIM
    NEW_HAND_CRAFTED_START = SPACY_END
    NEW_HAND_CRAFTED_END = NEW_HAND_CRAFTED_START + NEW_HAND_CRAFTED_DIM
    HARD_CLASS_START = NEW_HAND_CRAFTED_END
    HARD_CLASS_END = HARD_CLASS_START + HARD_CLASS_DIM

    # Binary feature indices (no scaling needed)
    # [0:384] = BERT (passthrough, already normalized)
    # [389:396] = mobilization one-hots within hand-crafted (passthrough, binary)
    # Everything else gets scaled
    BINARY_INDICES = list(
        range(HAND_CRAFTED_START + 5, HAND_CRAFTED_START + 12)
    )  # Mobilization one-hots

    @classmethod
    def get_scaling_indices(cls) -> tuple[list[int], list[int], list[int]]:
        """Get feature indices for ColumnTransformer scaling.

        Feature layout:
        - [0:384]   = BERT → passthrough (already L2-normalized)
        - [389:396] = mobilization one-hots → passthrough (binary)
        - everything else → StandardScaler

        Returns:
            (bert_indices, binary_indices, scale_indices)
        """
        bert_indices = list(range(cls.BERT_START, cls.BERT_END))
        binary_indices = cls.BINARY_INDICES
        scale_indices = [
            i for i in range(cls.TOTAL_DIM) if i not in bert_indices and i not in binary_indices
        ]
        return bert_indices, binary_indices, scale_indices
