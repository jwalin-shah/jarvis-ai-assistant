"""Few-shot example retrieval for reply generation.

Finds similar past conversations and uses your actual replies as examples.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEST_SET_FILE = Path("results/test_set/test_data.jsonl")
EMBEDDINGS_CACHE = Path("results/fewshot/example_embeddings.npz")


@dataclass
class FewShotExample:
    """A few-shot example from your actual conversations."""
    conversation: str  # The conversation context
    your_reply: str  # What you actually said
    contact: str
    relationship: str
    similarity: float = 0.0


class FewShotRetriever:
    """Retrieves similar conversations as few-shot examples."""

    def __init__(self):
        self._examples: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._model = None

    def load_examples(self) -> bool:
        """Load examples from test set."""
        if not TEST_SET_FILE.exists():
            print(f"No test set found at {TEST_SET_FILE}")
            return False

        self._examples = []
        with open(TEST_SET_FILE) as f:
            for line in f:
                self._examples.append(json.loads(line))

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

        if not force and EMBEDDINGS_CACHE.exists():
            data = np.load(EMBEDDINGS_CACHE)
            self._embeddings = data["embeddings"]
            print(f"Loaded cached embeddings: {self._embeddings.shape}")
            return

        print("Building embeddings for examples...")
        model = self._get_model()

        # Embed the last message (what they said that you're replying to)
        texts = [ex.get("last_message", ex.get("conversation", "")[-200:]) for ex in self._examples]

        self._embeddings = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

        # Cache
        EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(EMBEDDINGS_CACHE, embeddings=self._embeddings)
        print(f"Cached embeddings: {self._embeddings.shape}")

    def find_similar(self, query: str, n: int = 3, same_cluster: int | None = None) -> list[FewShotExample]:
        """Find similar examples to use as few-shot.

        Args:
            query: The message to reply to (or conversation snippet)
            n: Number of examples to return
            same_cluster: If provided, prefer examples from same cluster

        Returns:
            List of similar examples with your actual replies
        """
        if self._embeddings is None:
            self.build_embeddings()

        if self._embeddings is None or len(self._examples) == 0:
            return []

        # Embed query
        model = self._get_model()
        query_emb = model.encode([query], normalize_embeddings=True)[0]

        # Compute similarities
        similarities = np.dot(self._embeddings, query_emb)

        # Get top indices
        if same_cluster is not None:
            # Boost scores for same cluster
            for i, ex in enumerate(self._examples):
                if ex.get("cluster") == same_cluster:
                    similarities[i] += 0.1

        top_indices = np.argsort(similarities)[::-1][:n * 2]  # Get extra to filter

        results = []
        seen_replies = set()

        for idx in top_indices:
            if len(results) >= n:
                break

            ex = self._examples[idx]
            reply = ex.get("gold_response", "")

            # Skip duplicates
            if reply.lower() in seen_replies:
                continue
            seen_replies.add(reply.lower())

            # Skip very short or very long
            if len(reply) < 2 or len(reply) > 100:
                continue

            results.append(FewShotExample(
                conversation=ex.get("conversation", ex.get("last_message", "")),
                your_reply=reply,
                contact=ex.get("contact", ""),
                relationship=ex.get("relationship", ""),
                similarity=float(similarities[idx]),
            ))

        return results

    def format_examples(self, examples: list[FewShotExample], style: str = "minimal") -> str:
        """Format examples for inclusion in prompt.

        Args:
            examples: List of FewShotExample
            style: "minimal" (just reply), "context" (with conversation), "full"
        """
        if not examples:
            return ""

        if style == "minimal":
            # Just show example replies
            lines = ["Example replies in your style:"]
            for ex in examples:
                lines.append(f'- "{ex.your_reply}"')
            return "\n".join(lines)

        elif style == "context":
            # Show last message + your reply
            lines = ["Examples of how you reply:"]
            for ex in examples:
                # Get last "them:" line
                conv_lines = ex.conversation.strip().split("\n")
                last_them = next((l for l in reversed(conv_lines) if l.startswith("them:")), "them: ...")
                lines.append(f'{last_them}')
                lines.append(f'me: {ex.your_reply}')
                lines.append("")
            return "\n".join(lines)

        else:  # full
            lines = ["Examples from your past conversations:"]
            for i, ex in enumerate(examples, 1):
                lines.append(f"\n--- Example {i} ---")
                # Last few lines of conversation
                conv_lines = ex.conversation.strip().split("\n")[-4:]
                lines.extend(conv_lines)
                lines.append(f"me: {ex.your_reply}")
            return "\n".join(lines)


# Singleton
_retriever: FewShotRetriever | None = None


def get_retriever() -> FewShotRetriever:
    """Get or create the singleton retriever."""
    global _retriever
    if _retriever is None:
        _retriever = FewShotRetriever()
        _retriever.load_examples()
    return _retriever


def get_fewshot_examples(query: str, n: int = 3) -> list[FewShotExample]:
    """Convenience function to get few-shot examples."""
    return get_retriever().find_similar(query, n=n)


# Test
if __name__ == "__main__":
    retriever = FewShotRetriever()
    retriever.load_examples()
    retriever.build_embeddings()

    # Test queries
    queries = [
        "wanna grab dinner later?",
        "how are you doing?",
        "can you pick me up?",
        "that's hilarious lol",
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("=" * 50)

        examples = retriever.find_similar(query, n=3)
        for ex in examples:
            print(f"  [{ex.similarity:.2f}] them: ...{ex.conversation[-50:]}...")
            print(f"         you: \"{ex.your_reply}\"")
