"""Shared feature extraction for category classification.

Single source of truth for all category classification features.
Eliminates train/serve skew by using identical feature extraction.

Features:
- 384 BERT embeddings (via embedder.encode, normalized)
- 26 hand-crafted features (structure, mobilization, context, reactions)
- 94 spaCy features (14 original + 80 new: NER, deps, tokens, morphology)
- 19 new hand-crafted features (8 error-analysis + 11 high-value additions)
- 8 hard-class features (closing/request specific)
Total: 531 non-BERT features (147 without BERT, 531 with BERT)

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
    "what", "why", "how", "when", "where", "who",
    "did", "do", "does", "can", "could", "would", "will", "should"
}
IMPERATIVE_VERBS = {
    "make", "send", "get", "tell", "show", "give",
    "come", "take", "call", "help", "let"
}
BRIEF_AGREEMENTS = {
    "ok", "okay", "k", "yeah", "yep", "yup",
    "sure", "cool", "bet", "fs", "aight"
}

# New hand-crafted feature patterns (from error analysis)
CLAUSE_AFTER_AGREEMENT_RE = re.compile(
    r"\b(ok|okay|sure|yeah|yep|yup|cool|bet|sounds good)\b.*\b(but|though|however|if|when|"
    r"because|since|as long as|unless|after|before)\b",
    re.IGNORECASE
)

GREETING_PATTERN_RE = re.compile(
    r"^(hey|hi|hello|yo|sup|what's up|wassup|heyy|hiya|heya)\b",
    re.IGNORECASE
)

ELONGATED_WORD_RE = re.compile(r"(\w)\1{2,}")  # e.g., "yaasss", "nooo", "omgggg"

# Proposal patterns from text_normalizer
PROPOSAL_PATTERNS_RE = re.compile(
    r"\b(how about|what about|should we|shall we|let's|lets|we could)\b",
    re.IGNORECASE
)

# Implicit request patterns
IMPLICIT_REQUEST_RE = re.compile(
    r"\b(i need|i want|i wanna|i gotta|i have to)\b.*\b(you|your|help|know)\b",
    re.IGNORECASE
)

# === HARD-CLASS FEATURES (for closing and request) ===

# Closing-specific patterns
GOODBYE_PHRASES = {
    "bye", "goodbye", "later", "ttyl", "talk to you later", "gotta go", "gtg",
    "see you", "see ya", "catch you", "peace", "cya", "l8r", "take care",
    "talk soon", "talk later", "goodnight", "gnight", "nite", "sleep well"
}

CLOSING_EMOJIS = {"👋", "✌️", "✌", "💯", "✨", "🙏", "😴", "😊", "💤"}

TIME_CONSTRAINT_PHRASES = {
    "gotta run", "gtg", "g2g", "have to go", "running late", "in a hurry",
    "gotta jet", "about to leave"
}

# Request-specific patterns
POLITE_REQUEST_MODAL_RE = re.compile(
    r"^(can|could|would|will)\s+(you|u)\s+",
    re.IGNORECASE
)

IMPERATIVE_WITH_PLEASE_RE = re.compile(
    r"^(please\s+\w+|(\w+\s+)*please)",
    re.IGNORECASE
)

CONDITIONAL_REQUEST_RE = re.compile(
    r"\b(if you (could|can|would)|when you (can|get a chance|have time)|whenever you)\b",
    re.IGNORECASE
)

NEED_WANT_YOU_RE = re.compile(
    r"\b(i need you to|i want you to|i'd like you to|need you to|want you to)\b",
    re.IGNORECASE
)

# Additional feature patterns (context overlap, question types, thanks/apology, urgency, emoji sentiment)
THANKS_MARKERS = {"thanks", "thank you", "thx", "ty", "tysm", "appreciate", "appreciated"}
APOLOGY_MARKERS = {"sorry", "my bad", "apologize", "apologies", "oops", "whoops", "my fault"}
URGENCY_MARKERS = {"asap", "urgent", "quick", "quickly", "hurry", "rush", "rushing", "now", "right now", "immediately"}
FUTURE_TIME_MARKERS = {"later", "tomorrow", "tonight", "next week", "next time", "soon", "eventually"}

# Emoji sentiment sets (common emotional emojis)
POSITIVE_EMOJIS = {
    "😊", "😀", "😁", "😂", "🤣", "😃", "😄", "😆", "😉", "😍", "🥰", "😘", "❤️", "💕", "💖",
    "👍", "👏", "🙌", "🎉", "🎊", "✨", "💯", "🔥", "😎", "🤩", "😇", "🥳", "💪", "🙏", "💗"
}
NEGATIVE_EMOJIS = {
    "😢", "😭", "😞", "😔", "😟", "😕", "🙁", "☹️", "😣", "😖", "😫", "😩", "😤", "😠", "😡",
    "🤬", "💔", "😰", "😨", "😱", "😥", "😪", "🤦", "🤷", "💀", "😒", "🙄"
}
NEUTRAL_EMOJIS = {
    "🤔", "🙃", "😐", "😑", "🤨", "🧐", "😶", "👀", "💬", "🗣️", "👋", "✌️"
}


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

    @property
    def nlp(self) -> spacy.Language:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
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
            1.0 if total_words <= 3 and any(w.lower() in BRIEF_AGREEMENTS for w in words)
            else 0.0
        )
        features.append(is_brief_agreement)

        exclamatory = 1.0 if (text.endswith("!") or text.isupper() and total_words <= 5) else 0.0
        features.append(exclamatory)

        return np.array(features, dtype=np.float32)

    def extract_spacy_features(
        self,
        text: str,
        doc: spacy.tokens.Doc | None = None,
    ) -> NDArray[np.float32]:
        """Extract 94 spaCy linguistic features (14 original + 80 new).

        New features:
        - 8 POS ratios
        - 12 fine-grained tags
        - 15 dependency relations (10 original + 5 new)
        - 21 named entity types (6 original + 15 new)
        - 5 sentence structure
        - 11 token properties (6 original + 5 new)
        - 8 morphology features

        Args:
            text: Message text
            doc: Optional pre-parsed spaCy doc (reuse to avoid re-parsing)

        Returns:
            94-dim feature array
        """
        if doc is None:
            doc = self.nlp(text)

        features: list[float] = []
        text_lower = text.lower()
        total_tokens = len(doc)

        # === ORIGINAL 14 FEATURES ===

        # 1. has_imperative
        has_imperative = 0.0
        if len(doc) > 0 and doc[0].pos_ == "VERB" and doc[0].tag_ == "VB":
            has_imperative = 1.0
        features.append(has_imperative)

        # 2. you_modal
        you_modal = 1.0 if any(
            p in text_lower for p in ["can you", "could you", "would you", "will you", "should you"]
        ) else 0.0
        features.append(you_modal)

        # 3. request_verb
        request_verbs = {"send", "give", "help", "tell", "show", "let", "call", "get", "make", "take"}
        has_request = 1.0 if any(token.lemma_ in request_verbs for token in doc) else 0.0
        features.append(has_request)

        # 4. starts_modal
        starts_modal = 0.0
        if len(doc) > 0 and doc[0].tag_ in ("MD", "VB"):
            starts_modal = 1.0
        features.append(starts_modal)

        # 5. directive_question
        directive_q = 1.0 if you_modal and "?" in text else 0.0
        features.append(directive_q)

        # 6. i_will
        i_will = 1.0 if any(
            p in text_lower for p in ["i'll", "i will", "i'm gonna", "ima", "imma"]
        ) else 0.0
        features.append(i_will)

        # 7. promise_verb
        promise_verbs = {"promise", "guarantee", "commit", "swear"}
        has_promise = 1.0 if any(token.lemma_ in promise_verbs for token in doc) else 0.0
        features.append(has_promise)

        # 8. first_person_count
        first_person = sum(
            1 for token in doc if token.text.lower() in ("i", "me", "my", "mine", "myself")
        )
        features.append(float(first_person))

        # 9. agreement
        agreement_words = {"sure", "okay", "ok", "yes", "yeah", "yep", "yup", "sounds good", "bet", "fs"}
        has_agreement = 1.0 if any(word in text_lower for word in agreement_words) else 0.0
        features.append(has_agreement)

        # 10. modal_count
        modal_count = sum(1 for token in doc if token.tag_ == "MD")
        features.append(float(modal_count))

        # 11. verb_count
        verb_count = sum(1 for token in doc if token.pos_ == "VERB")
        features.append(float(verb_count))

        # 12. second_person_count
        second_person = sum(
            1 for token in doc if token.text.lower() in ("you", "your", "yours", "yourself")
        )
        features.append(float(second_person))

        # 13. has_negation
        has_neg = 1.0 if any(token.dep_ == "neg" for token in doc) else 0.0
        features.append(has_neg)

        # 14. is_interrogative
        is_question = 1.0 if "?" in text or any(
            token.tag_ in ("WDT", "WP", "WP$", "WRB") for token in doc
        ) else 0.0
        features.append(is_question)

        # === NEW 55 FEATURES ===

        # POS ratios (8): VERB, NOUN, ADJ, ADV, PRON, DET, ADP, INTJ
        pos_counts = {"VERB": 0, "NOUN": 0, "ADJ": 0, "ADV": 0, "PRON": 0, "DET": 0, "ADP": 0, "INTJ": 0}
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1

        for pos in ["VERB", "NOUN", "ADJ", "ADV", "PRON", "DET", "ADP", "INTJ"]:
            features.append(pos_counts[pos] / max(total_tokens, 1))

        # Fine-grained tags (12): Binary presence of specific tags
        target_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD", "WDT", "WP", "WRB", "UH", "JJR"]
        tag_set = {token.tag_ for token in doc}
        for tag in target_tags:
            features.append(1.0 if tag in tag_set else 0.0)

        # Dependency counts (15): 10 original + 5 new (ccomp, xcomp, acl, relcl, mark)
        dep_counts = {
            "nsubj": 0, "dobj": 0, "ROOT": 0, "aux": 0, "neg": 0,
            "advmod": 0, "amod": 0, "prep": 0, "pobj": 0, "conj": 0,
            "ccomp": 0, "xcomp": 0, "acl": 0, "relcl": 0, "mark": 0
        }
        for token in doc:
            if token.dep_ in dep_counts:
                dep_counts[token.dep_] += 1

        for dep in ["nsubj", "dobj", "ROOT", "aux", "neg", "advmod", "amod", "prep", "pobj", "conj",
                    "ccomp", "xcomp", "acl", "relcl", "mark"]:
            features.append(dep_counts[dep] / max(total_tokens, 1))

        # Named entities (21): 6 original + 15 new entity types
        ent_labels = {ent.label_ for ent in doc.ents}
        # Original 6
        features.append(1.0 if "PERSON" in ent_labels else 0.0)
        features.append(1.0 if "DATE" in ent_labels else 0.0)
        features.append(1.0 if "TIME" in ent_labels else 0.0)
        features.append(1.0 if "GPE" in ent_labels else 0.0)
        features.append(float(len(doc.ents)))
        features.append(len(doc.ents) / max(total_tokens, 1))
        # New 15 entity types
        features.append(1.0 if "MONEY" in ent_labels else 0.0)
        features.append(1.0 if "CARDINAL" in ent_labels else 0.0)
        features.append(1.0 if "ORDINAL" in ent_labels else 0.0)
        features.append(1.0 if "PERCENT" in ent_labels else 0.0)
        features.append(1.0 if "QUANTITY" in ent_labels else 0.0)
        features.append(1.0 if "FAC" in ent_labels else 0.0)
        features.append(1.0 if "PRODUCT" in ent_labels else 0.0)
        features.append(1.0 if "EVENT" in ent_labels else 0.0)
        features.append(1.0 if "LANGUAGE" in ent_labels else 0.0)
        features.append(1.0 if "LAW" in ent_labels else 0.0)
        features.append(1.0 if "WORK_OF_ART" in ent_labels else 0.0)
        features.append(1.0 if "NORP" in ent_labels else 0.0)
        features.append(1.0 if "LOC" in ent_labels else 0.0)
        features.append(1.0 if "ORG" in ent_labels else 0.0)
        features.append(1.0 if ("CARDINAL" in ent_labels or "MONEY" in ent_labels) else 0.0)

        # Sentence structure (5): sent_count, avg_sent_len, max_sent_len, noun_chunk_count, noun_chunk_ratio
        sents = list(doc.sents)
        sent_count = len(sents)
        features.append(float(sent_count))

        avg_sent_len = float(np.mean([len(sent) for sent in sents])) if sents else 0.0
        features.append(avg_sent_len)

        max_sent_len = float(max([len(sent) for sent in sents])) if sents else 0.0
        features.append(max_sent_len)

        noun_chunks = list(doc.noun_chunks)
        features.append(float(len(noun_chunks)))
        features.append(len(noun_chunks) / max(total_tokens, 1))

        # Token properties (11): 6 original + 5 new
        stop_count = sum(1 for token in doc if token.is_stop)
        features.append(stop_count / max(total_tokens, 1))

        alpha_count = sum(1 for token in doc if token.is_alpha)
        features.append(alpha_count / max(total_tokens, 1))

        digit_count = sum(1 for token in doc if token.is_digit)
        features.append(digit_count / max(total_tokens, 1))

        avg_word_len = float(np.mean([len(token.text) for token in doc])) if total_tokens > 0 else 0.0
        features.append(avg_word_len)

        punct_count = sum(1 for token in doc if token.is_punct)
        features.append(punct_count / max(total_tokens, 1))

        url_count = sum(1 for token in doc if token.like_url)
        features.append(float(url_count))

        # New 5 token attributes
        like_num_count = sum(1 for token in doc if token.like_num)
        features.append(like_num_count / max(total_tokens, 1))

        like_email_count = sum(1 for token in doc if token.like_email)
        features.append(float(like_email_count))

        is_currency_count = sum(1 for token in doc if token.is_currency)
        features.append(float(is_currency_count))

        is_quote_count = sum(1 for token in doc if token.is_quote)
        features.append(float(is_quote_count))

        like_url_ratio = url_count / max(total_tokens, 1)
        features.append(like_url_ratio)

        # Morphology (8): past_tense_ratio, present_tense_ratio, has_imperative_mood, has_conditional,
        # person_1/2/3, has_passive
        past_tense_tags = {"VBD", "VBN"}
        past_tense_count = sum(1 for token in doc if token.tag_ in past_tense_tags)
        features.append(past_tense_count / max(total_tokens, 1))

        present_tense_tags = {"VBP", "VBZ", "VBG"}
        present_tense_count = sum(1 for token in doc if token.tag_ in present_tense_tags)
        features.append(present_tense_count / max(total_tokens, 1))

        # Imperative mood (VB at start or standalone VB ROOT)
        has_imperative_mood = 0.0
        if len(doc) > 0:
            if doc[0].tag_ == "VB" or any(token.tag_ == "VB" and token.dep_ == "ROOT" for token in doc):
                has_imperative_mood = 1.0
        features.append(has_imperative_mood)

        # Conditional (modal + if/would)
        has_conditional = 1.0 if modal_count > 0 and any(
            w in text_lower for w in ["if", "would"]
        ) else 0.0
        features.append(has_conditional)

        # Person (1st/2nd/3rd person pronoun ratio)
        first_person_ratio = first_person / max(total_tokens, 1)
        features.append(first_person_ratio)

        second_person_ratio = second_person / max(total_tokens, 1)
        features.append(second_person_ratio)

        third_person_pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "their"}
        third_person = sum(1 for token in doc if token.text.lower() in third_person_pronouns)
        features.append(third_person / max(total_tokens, 1))

        # Passive voice (heuristic: aux be + VBN)
        has_passive = 0.0
        for i in range(len(doc) - 1):
            if doc[i].lemma_ == "be" and doc[i].dep_ == "auxpass" and doc[i+1].tag_ == "VBN":
                has_passive = 1.0
                break
        features.append(has_passive)

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

        has_future_time = 1.0 if any(marker in text_lower for marker in FUTURE_TIME_MARKERS) else 0.0
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
        has_time_constraint = 1.0 if any(
            phrase in text_lower for phrase in TIME_CONSTRAINT_PHRASES
        ) else 0.0
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
        num_sentences = max(1, text.count('.') + text.count('?') + text.count('!'))
        features.append(min(num_sentences / 3.0, 1.0))  # Cap at 3

        # 2. has_conjunction
        conjunctions = ['but', 'and', 'also', 'though', 'however', 'plus']
        has_conj = 1.0 if any(f' {c} ' in text_lower for c in conjunctions) else 0.0
        features.append(has_conj)

        # 3. mixed_punctuation
        has_question = '?' in text
        has_exclamation = '!' in text
        has_period = '.' in text and not text.strip().endswith('...')
        mixed_punct = 1.0 if sum([has_question, has_exclamation, has_period]) >= 2 else 0.0
        features.append(mixed_punct)

        # 4. is_very_short (<4 words → likely single-label)
        is_very_short = 1.0 if len(words) < 4 else 0.0
        features.append(is_very_short)

        # 5. is_long (>25 words → might be multi-label)
        is_long = 1.0 if len(words) > 25 else 0.0
        features.append(is_long)

        # 6. punctuation_diversity
        punct_types = sum([
            '?' in text,
            '!' in text,
            ',' in text,
            '.' in text,
            ';' in text,
        ])
        features.append(min(punct_types / 3.0, 1.0))

        # 7. has_thanks_plus_more
        has_thanks = any(word in text_lower for word in ['thanks', 'thank you', 'thx'])
        thanks_plus_more = 1.0 if (has_thanks and len(words) > 3) else 0.0
        features.append(thanks_plus_more)

        # 8. question_with_context (long question might include statement)
        question_with_context = 1.0 if ('?' in text and len(words) > 12) else 0.0
        features.append(question_with_context)

        # 9. words_per_sentence
        words_per_sentence = len(words) / max(num_sentences, 1)
        features.append(min(words_per_sentence / 15.0, 1.0))  # Normalize

        # 10. has_discourse_marker (transitions between intents)
        discourse_markers = ['so', 'anyway', 'well', 'btw', 'also', 'oh']
        has_marker = 1.0 if any(f'{m} ' in text_lower or f' {m}' in text_lower for m in discourse_markers) else 0.0
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


class FeatureConfig:
    """Feature dimensions and metadata."""

    # Feature group sizes
    BERT_DIM = 384
    CONTEXT_BERT_DIM = 384  # NEW: Context embeddings
    HAND_CRAFTED_DIM = 26
    SPACY_DIM = 94  # 14 original + 80 new (15 NER + 5 deps + 5 tokens + 55 from before)
    NEW_HAND_CRAFTED_DIM = 19  # 8 error-analysis + 11 high-value additions
    HARD_CLASS_DIM = 8
    MULTILABEL_INDICATOR_DIM = 10  # Closing + request specific features (unused in hardclass)
    TOTAL_NON_BERT = HAND_CRAFTED_DIM + SPACY_DIM + NEW_HAND_CRAFTED_DIM + HARD_CLASS_DIM  # 147
    TOTAL_DIM = BERT_DIM + CONTEXT_BERT_DIM + TOTAL_NON_BERT  # 915

    # Feature index ranges (for ColumnTransformer)
    BERT_START = 0
    BERT_END = BERT_DIM
    CONTEXT_BERT_START = BERT_END
    CONTEXT_BERT_END = CONTEXT_BERT_START + CONTEXT_BERT_DIM
    HAND_CRAFTED_START = CONTEXT_BERT_END
    HAND_CRAFTED_END = HAND_CRAFTED_START + HAND_CRAFTED_DIM
    SPACY_START = HAND_CRAFTED_END
    SPACY_END = SPACY_START + SPACY_DIM
    NEW_HAND_CRAFTED_START = SPACY_END
    NEW_HAND_CRAFTED_END = NEW_HAND_CRAFTED_START + NEW_HAND_CRAFTED_DIM
    HARD_CLASS_START = NEW_HAND_CRAFTED_END
    HARD_CLASS_END = HARD_CLASS_START + HARD_CLASS_DIM

    # Binary feature indices (no scaling needed)
    # [0:384] = current BERT (passthrough, already normalized)
    # [384:768] = context BERT (passthrough, already normalized)
    # [768:775] = mobilization one-hots within hand-crafted (passthrough, binary)
    # Everything else gets scaled
    BINARY_INDICES = list(range(HAND_CRAFTED_START + 5, HAND_CRAFTED_START + 12))  # Mobilization one-hots

    @classmethod
    def get_scaling_indices(cls) -> tuple[list[int], list[int], list[int]]:
        """Get feature indices for ColumnTransformer scaling.

        Feature layout:
        - [0:384] = current BERT → passthrough (already L2-normalized)
        - [384:768] = context BERT → passthrough (already L2-normalized)
        - [768:775] = mobilization one-hots → passthrough (binary)
        - [775:915] = everything else → StandardScaler

        Returns:
            (bert_indices, binary_indices, scale_indices)
        """
        # Both BERT embeddings (current + context) are already normalized
        bert_indices = list(range(cls.BERT_START, cls.CONTEXT_BERT_END))
        binary_indices = cls.BINARY_INDICES
        scale_indices = [
            i for i in range(cls.TOTAL_DIM)
            if i not in bert_indices and i not in binary_indices
        ]
        return bert_indices, binary_indices, scale_indices
