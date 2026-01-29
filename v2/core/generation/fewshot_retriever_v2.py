"""Improved few-shot example retrieval for reply generation.

V2 improvements:
1. Multi-signal matching: content + style + message type
2. Better filtering: length similarity, response quality
3. Conversation context embedding (not just last message)
4. Message type detection (question, statement, reaction)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

TEST_SET_FILE = Path("results/test_set/test_data.jsonl")
EMBEDDINGS_CACHE_V2 = Path("results/fewshot/example_embeddings_v2.npz")


@dataclass
class FewShotExample:
    """A few-shot example from your actual conversations."""
    conversation: str
    your_reply: str
    contact: str
    cluster: int
    # Computed features
    similarity: float = 0.0
    message_type: str = ""  # question, statement, reaction, greeting
    reply_length: int = 0
    has_lol: bool = False
    has_abbreviations: bool = False
    ends_with_punctuation: bool = False


@dataclass
class MatchCriteria:
    """Criteria for matching examples."""
    prefer_cluster: int | None = None
    target_length: int | None = None  # Approximate reply length
    message_type: str | None = None  # question, statement, etc.
    use_lol: bool | None = None
    min_similarity: float = 0.5


def detect_message_type(text: str) -> str:
    """Detect the type of message."""
    text = text.lower().strip()

    # Reactions (very short, single word/phrase)
    if len(text) < 15:
        reactions = ["haha", "lol", "lmao", "nice", "cool", "wow", "damn", "yea", "yeah", "ok", "k", "bet", "fr", "true"]
        if any(text.startswith(r) or text == r for r in reactions):
            return "reaction"

    # Questions
    if text.endswith("?") or text.startswith(("what", "when", "where", "who", "why", "how", "can", "do", "does", "did", "is", "are", "will", "would", "should", "could", "wanna", "gonna")):
        return "question"

    # Greetings
    greetings = ["hey", "hi", "hello", "yo", "sup", "what's up", "whats up"]
    if any(text.startswith(g) for g in greetings):
        return "greeting"

    # Default to statement
    return "statement"


def extract_style_features(text: str) -> dict:
    """Extract style features from a reply."""
    abbrevs = [r"\bu\b", r"\bur\b", r"\brn\b", r"\btmrw\b", r"\bidk\b", r"\byea\b", r"\bya\b"]

    return {
        "length": len(text),
        "has_lol": "lol" in text.lower() or "haha" in text.lower(),
        "has_abbreviations": any(re.search(a, text.lower()) for a in abbrevs),
        "ends_with_punctuation": text.rstrip()[-1:] in ".!?" if text else False,
        "word_count": len(text.split()),
    }


class ImprovedFewShotRetriever:
    """Improved few-shot retrieval with multi-signal matching."""

    def __init__(self):
        self._examples: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._context_embeddings: np.ndarray | None = None  # Full conversation context
        self._response_embeddings: np.ndarray | None = None  # NEW: Embed the responses too
        self._model = None
        self._style_features: list[dict] = []

    def load_examples(self) -> bool:
        """Load examples from test set."""
        if not TEST_SET_FILE.exists():
            print(f"No test set found at {TEST_SET_FILE}")
            return False

        self._examples = []
        with open(TEST_SET_FILE) as f:
            for line in f:
                ex = json.loads(line)
                # Pre-compute style features
                reply = ex.get("gold_response", "")
                ex["_style"] = extract_style_features(reply)
                ex["_message_type"] = detect_message_type(ex.get("last_message", ""))
                self._examples.append(ex)

        print(f"Loaded {len(self._examples)} examples")
        return True

    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return self._model

    def build_embeddings(self, force: bool = False):
        """Build embeddings for all examples."""
        if not self._examples:
            self.load_examples()

        if not force and EMBEDDINGS_CACHE_V2.exists():
            data = np.load(EMBEDDINGS_CACHE_V2)
            self._embeddings = data["last_message_embeddings"]
            self._context_embeddings = data["context_embeddings"]
            if "response_embeddings" in data:
                self._response_embeddings = data["response_embeddings"]
            print(f"Loaded cached embeddings: {self._embeddings.shape}")
            return

        print("Building embeddings for examples...")
        model = self._get_model()

        # 1. Embed last message (what they said)
        last_messages = [ex.get("last_message", ex.get("prompt", "")[-200:]) for ex in self._examples]
        self._embeddings = model.encode(
            last_messages,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

        # 2. Embed conversation context (last 3 messages combined)
        contexts = []
        for ex in self._examples:
            prompt = ex.get("prompt", "")
            # Get last 3 lines of conversation
            lines = [l for l in prompt.split("\n") if l.startswith(("me:", "them:"))][-3:]
            contexts.append(" ".join(lines))

        self._context_embeddings = model.encode(
            contexts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

        # 3. NEW: Embed the responses (gold_response)
        print("Building response embeddings...")
        responses = [ex.get("gold_response", "") for ex in self._examples]
        self._response_embeddings = model.encode(
            responses,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

        # Cache
        EMBEDDINGS_CACHE_V2.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            EMBEDDINGS_CACHE_V2,
            last_message_embeddings=self._embeddings,
            context_embeddings=self._context_embeddings,
            response_embeddings=self._response_embeddings,
        )
        print(f"Cached embeddings: {self._embeddings.shape}")

    def find_similar(
        self,
        query: str,
        conversation_context: str = "",
        criteria: MatchCriteria | None = None,
        n: int = 3,
    ) -> list[FewShotExample]:
        """Find similar examples with multi-signal matching.

        Args:
            query: The last message to reply to
            conversation_context: Last few messages for context matching
            criteria: Matching criteria (cluster, length, type, etc.)
            n: Number of examples to return

        Returns:
            List of matched examples, sorted by relevance
        """
        if self._embeddings is None:
            self.build_embeddings()

        if self._embeddings is None or len(self._examples) == 0:
            return []

        criteria = criteria or MatchCriteria()
        model = self._get_model()

        # Embed query
        query_emb = model.encode([query], normalize_embeddings=True)[0]

        # Embed context if provided
        context_emb = None
        if conversation_context:
            context_emb = model.encode([conversation_context], normalize_embeddings=True)[0]

        # Compute base similarity scores
        message_sims = np.dot(self._embeddings, query_emb)

        # Add context similarity if available
        if context_emb is not None and self._context_embeddings is not None:
            context_sims = np.dot(self._context_embeddings, context_emb)
            # Weighted combination: 70% message, 30% context
            similarities = message_sims * 0.7 + context_sims * 0.3
        else:
            similarities = message_sims

        # Apply boosting based on criteria
        query_type = detect_message_type(query)

        for i, ex in enumerate(self._examples):
            boost = 0.0

            # Cluster match boost
            if criteria.prefer_cluster is not None and ex.get("cluster") == criteria.prefer_cluster:
                boost += 0.15

            # Message type match boost
            ex_type = ex.get("_message_type", "")
            if ex_type == query_type:
                boost += 0.1

            # Length similarity boost (for similar response lengths)
            if criteria.target_length is not None:
                ex_len = ex.get("_style", {}).get("length", 30)
                len_ratio = min(ex_len, criteria.target_length) / max(ex_len, criteria.target_length, 1)
                if len_ratio > 0.5:
                    boost += 0.05 * len_ratio

            # LOL usage match
            if criteria.use_lol is not None:
                ex_has_lol = ex.get("_style", {}).get("has_lol", False)
                if ex_has_lol == criteria.use_lol:
                    boost += 0.05

            similarities[i] += boost

        # Get top candidates
        top_indices = np.argsort(similarities)[::-1][:n * 3]

        results = []
        seen_replies = set()

        for idx in top_indices:
            if len(results) >= n:
                break

            ex = self._examples[idx]
            sim = float(similarities[idx])

            # Apply minimum similarity threshold
            if sim < criteria.min_similarity:
                continue

            reply = ex.get("gold_response", "")

            # Skip duplicates
            reply_lower = reply.lower().strip()
            if reply_lower in seen_replies:
                continue
            seen_replies.add(reply_lower)

            # Skip very short or very long
            if len(reply) < 2 or len(reply) > 120:
                continue

            # Skip low-quality replies
            if reply.startswith(("I ", "I'm ", "I'll ", "I've ")) and len(reply) > 60:
                # Likely too formal
                continue

            style = ex.get("_style", {})

            results.append(FewShotExample(
                conversation=ex.get("prompt", ex.get("conversation", "")),
                your_reply=reply,
                contact=ex.get("contact", ""),
                cluster=ex.get("cluster", -1),
                similarity=sim,
                message_type=ex.get("_message_type", ""),
                reply_length=style.get("length", 0),
                has_lol=style.get("has_lol", False),
                has_abbreviations=style.get("has_abbreviations", False),
                ends_with_punctuation=style.get("ends_with_punctuation", False),
            ))

        return results

    def find_by_response_style(
        self,
        target_response: str,
        criteria: MatchCriteria | None = None,
        n: int = 3,
    ) -> list[FewShotExample]:
        """Find examples with similar RESPONSE styles.

        Instead of matching by input, find examples where the user's response
        is similar to what we expect. This is useful for finding examples
        with the right "vibe" - short reactions, casual acknowledgments, etc.

        Args:
            target_response: Example of the type of response we want
            criteria: Additional matching criteria
            n: Number of examples to return
        """
        if self._response_embeddings is None:
            self.build_embeddings()

        if self._response_embeddings is None or len(self._examples) == 0:
            return []

        criteria = criteria or MatchCriteria()
        model = self._get_model()

        # Embed target response
        target_emb = model.encode([target_response], normalize_embeddings=True)[0]

        # Find similar responses
        similarities = np.dot(self._response_embeddings, target_emb)

        # Apply cluster boost
        if criteria.prefer_cluster is not None:
            for i, ex in enumerate(self._examples):
                if ex.get("cluster") == criteria.prefer_cluster:
                    similarities[i] += 0.1

        # Get top candidates
        top_indices = np.argsort(similarities)[::-1][:n * 3]

        results = []
        seen_replies = set()

        for idx in top_indices:
            if len(results) >= n:
                break

            ex = self._examples[idx]
            sim = float(similarities[idx])

            if sim < criteria.min_similarity:
                continue

            reply = ex.get("gold_response", "")

            reply_lower = reply.lower().strip()
            if reply_lower in seen_replies:
                continue
            seen_replies.add(reply_lower)

            if len(reply) < 2 or len(reply) > 120:
                continue

            style = ex.get("_style", {})

            results.append(FewShotExample(
                conversation=ex.get("prompt", ex.get("conversation", "")),
                your_reply=reply,
                contact=ex.get("contact", ""),
                cluster=ex.get("cluster", -1),
                similarity=sim,
                message_type=ex.get("_message_type", ""),
                reply_length=style.get("length", 0),
                has_lol=style.get("has_lol", False),
                has_abbreviations=style.get("has_abbreviations", False),
                ends_with_punctuation=style.get("ends_with_punctuation", False),
            ))

        return results

    def find_hybrid(
        self,
        query: str,
        expected_style: str = "",
        criteria: MatchCriteria | None = None,
        n: int = 3,
    ) -> list[FewShotExample]:
        """Hybrid retrieval: combine input similarity + response style.

        This is the best approach: find examples where both the input
        is similar AND the response style matches what we want.

        Args:
            query: The message to reply to
            expected_style: Example of expected response style (e.g., "lol ok")
            criteria: Matching criteria
            n: Number of examples

        Returns:
            Examples sorted by combined score
        """
        if self._embeddings is None or self._response_embeddings is None:
            self.build_embeddings()

        if not self._examples:
            return []

        criteria = criteria or MatchCriteria()
        model = self._get_model()

        # Embed query
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        input_sims = np.dot(self._embeddings, query_emb)

        # If expected style provided, also consider response similarity
        if expected_style:
            style_emb = model.encode([expected_style], normalize_embeddings=True)[0]
            response_sims = np.dot(self._response_embeddings, style_emb)
            # Combine: 60% input match, 40% response style match
            combined_sims = input_sims * 0.6 + response_sims * 0.4
        else:
            combined_sims = input_sims

        # Apply cluster boost
        if criteria.prefer_cluster is not None:
            for i, ex in enumerate(self._examples):
                if ex.get("cluster") == criteria.prefer_cluster:
                    combined_sims[i] += 0.1

        # Get top candidates
        top_indices = np.argsort(combined_sims)[::-1][:n * 3]

        results = []
        seen_replies = set()

        for idx in top_indices:
            if len(results) >= n:
                break

            ex = self._examples[idx]
            sim = float(combined_sims[idx])

            if sim < criteria.min_similarity:
                continue

            reply = ex.get("gold_response", "")

            reply_lower = reply.lower().strip()
            if reply_lower in seen_replies:
                continue
            seen_replies.add(reply_lower)

            if len(reply) < 2 or len(reply) > 120:
                continue

            style = ex.get("_style", {})

            results.append(FewShotExample(
                conversation=ex.get("prompt", ex.get("conversation", "")),
                your_reply=reply,
                contact=ex.get("contact", ""),
                cluster=ex.get("cluster", -1),
                similarity=sim,
                message_type=ex.get("_message_type", ""),
                reply_length=style.get("length", 0),
                has_lol=style.get("has_lol", False),
                has_abbreviations=style.get("has_abbreviations", False),
                ends_with_punctuation=style.get("ends_with_punctuation", False),
            ))

        return results


# Singleton
_retriever_v2: ImprovedFewShotRetriever | None = None


def get_retriever_v2() -> ImprovedFewShotRetriever:
    """Get or create the singleton retriever."""
    global _retriever_v2
    if _retriever_v2 is None:
        _retriever_v2 = ImprovedFewShotRetriever()
        _retriever_v2.load_examples()
    return _retriever_v2


def get_improved_examples(
    query: str,
    conversation_context: str = "",
    cluster: int | None = None,
    target_length: int | None = None,
    use_lol: bool | None = None,
    n: int = 3,
    min_similarity: float = 0.5,
) -> list[FewShotExample]:
    """Convenience function to get improved few-shot examples."""
    criteria = MatchCriteria(
        prefer_cluster=cluster,
        target_length=target_length,
        use_lol=use_lol,
        min_similarity=min_similarity,
    )
    return get_retriever_v2().find_similar(
        query=query,
        conversation_context=conversation_context,
        criteria=criteria,
        n=n,
    )


# Test
if __name__ == "__main__":
    retriever = ImprovedFewShotRetriever()
    retriever.load_examples()
    retriever.build_embeddings()

    # Test queries with different types
    test_cases = [
        ("wanna grab dinner later?", "question"),
        ("haha that's hilarious", "reaction"),
        ("hey what's up", "greeting"),
        ("I'm heading to the store", "statement"),
    ]

    for query, expected_type in test_cases:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Detected type: {detect_message_type(query)} (expected: {expected_type})")
        print("=" * 60)

        # Test with criteria
        criteria = MatchCriteria(
            min_similarity=0.4,
            target_length=25,
        )
        examples = retriever.find_similar(query, criteria=criteria, n=3)

        for ex in examples:
            print(f"  [{ex.similarity:.2f}] [{ex.message_type}] \"{ex.your_reply}\"")
            print(f"         len={ex.reply_length}, lol={ex.has_lol}, cluster={ex.cluster}")
