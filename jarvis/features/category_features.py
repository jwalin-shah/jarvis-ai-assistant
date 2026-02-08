"""Shared feature extraction for category classification.

Single source of truth for all category classification features.
Eliminates train/serve skew by using identical feature extraction.

Features:
- 384 BERT embeddings (via embedder.encode, normalized)
- 26 hand-crafted features (structure, mobilization, context, reactions)
- ~69 spaCy features (14 original + 55 new targeted features)
- 8 new hand-crafted features (from error analysis)
Total: ~487 features

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
        """Extract ~69 spaCy linguistic features (14 original + 55 new).

        Args:
            text: Message text
            doc: Optional pre-parsed spaCy doc (reuse to avoid re-parsing)

        Returns:
            ~69-dim feature array
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

        # Dependency counts (10): nsubj, dobj, ROOT, aux, neg, advmod, amod, prep, pobj, conj
        dep_counts = {
            "nsubj": 0, "dobj": 0, "ROOT": 0, "aux": 0, "neg": 0,
            "advmod": 0, "amod": 0, "prep": 0, "pobj": 0, "conj": 0
        }
        for token in doc:
            if token.dep_ in dep_counts:
                dep_counts[token.dep_] += 1

        for dep in ["nsubj", "dobj", "ROOT", "aux", "neg", "advmod", "amod", "prep", "pobj", "conj"]:
            features.append(dep_counts[dep] / max(total_tokens, 1))

        # Named entities (6): has_PERSON, has_DATE, has_TIME, has_GPE, entity_count, entity_ratio
        ent_labels = {ent.label_ for ent in doc.ents}
        features.append(1.0 if "PERSON" in ent_labels else 0.0)
        features.append(1.0 if "DATE" in ent_labels else 0.0)
        features.append(1.0 if "TIME" in ent_labels else 0.0)
        features.append(1.0 if "GPE" in ent_labels else 0.0)
        features.append(float(len(doc.ents)))
        features.append(len(doc.ents) / max(total_tokens, 1))

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

        # Token properties (6): stop_word_ratio, alpha_ratio, digit_ratio, avg_word_len, punct_ratio, like_url_count
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
        """Extract 8 new hand-crafted features from error analysis.

        Args:
            text: Message text
            doc: Optional pre-parsed spaCy doc
            context: Previous messages in conversation

        Returns:
            8-dim feature array
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

        return np.array(features, dtype=np.float32)

    def extract_all(
        self,
        text: str,
        context: list[str] | None = None,
        mob_pressure: str = "none",
        mob_type: str = "answer",
    ) -> NDArray[np.float32]:
        """Extract all ~103 non-BERT features.

        Parse spaCy doc ONCE, reuse for all extraction methods.

        Args:
            text: Message text
            context: Previous messages in conversation
            mob_pressure: Mobilization pressure level
            mob_type: Mobilization response type

        Returns:
            ~103-dim feature array (26 + ~69 + 8)
        """
        # Parse once, reuse
        doc = self.nlp(text)

        # Extract all feature groups
        hand_crafted = self.extract_hand_crafted(text, context, mob_pressure, mob_type)
        spacy_feats = self.extract_spacy_features(text, doc)
        new_hand_crafted = self.extract_new_hand_crafted(text, doc, context)

        # Concatenate
        return np.concatenate([hand_crafted, spacy_feats, new_hand_crafted])


class FeatureConfig:
    """Feature dimensions and metadata."""

    # Feature group sizes
    BERT_DIM = 384
    HAND_CRAFTED_DIM = 26
    SPACY_DIM = 69  # 14 original + 55 new
    NEW_HAND_CRAFTED_DIM = 8
    TOTAL_NON_BERT = HAND_CRAFTED_DIM + SPACY_DIM + NEW_HAND_CRAFTED_DIM  # 103
    TOTAL_DIM = BERT_DIM + TOTAL_NON_BERT  # 487

    # Feature index ranges (for ColumnTransformer)
    BERT_START = 0
    BERT_END = BERT_DIM
    HAND_CRAFTED_START = BERT_END
    HAND_CRAFTED_END = HAND_CRAFTED_START + HAND_CRAFTED_DIM
    SPACY_START = HAND_CRAFTED_END
    SPACY_END = SPACY_START + SPACY_DIM
    NEW_HAND_CRAFTED_START = SPACY_END
    NEW_HAND_CRAFTED_END = NEW_HAND_CRAFTED_START + NEW_HAND_CRAFTED_DIM

    # Binary feature indices (no scaling needed)
    # Hand-crafted: mobilization one-hots (indices 5-11), tone flags (12-13),
    # is_no_context (15), reaction flags (16, 18, 20, 21, 22, 23)
    # SpaCy: many binary (has_imperative, you_modal, etc.)
    # For simplicity: scale all non-BERT features except one-hot encoded mobilization
    BINARY_INDICES = list(range(HAND_CRAFTED_START + 5, HAND_CRAFTED_START + 12))  # Mobilization one-hots

    @classmethod
    def get_scaling_indices(cls) -> tuple[list[int], list[int], list[int]]:
        """Get feature indices for ColumnTransformer scaling.

        Returns:
            (bert_indices, binary_indices, scale_indices)
        """
        bert_indices = list(range(cls.BERT_START, cls.BERT_END))
        binary_indices = cls.BINARY_INDICES
        scale_indices = [
            i for i in range(cls.TOTAL_DIM)
            if i not in bert_indices and i not in binary_indices
        ]
        return bert_indices, binary_indices, scale_indices
