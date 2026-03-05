import re
import time

from jarvis.classifiers.relationship_classifier import _is_emoji

# Use normal string for unicode escape handling
_EMOJI_RE = re.compile("[\U0001f300-\U0001f9ff\u2600-\u27bf\U0001f000-\U0001f02f]")


def count_emoji_regex(text: str) -> int:
    return len(_EMOJI_RE.findall(text))


def count_emoji_original(text: str) -> int:
    return sum(1 for c in text if _is_emoji(c))


def main():
    text_with_emoji = (
        "Hello world! This is a test string to see how fast emoji detection is. ⭐😭😢" * 10
    )
    text_without_emoji = (
        "Hello world! This is a test string to see how fast emoji detection is." * 10
    )

    start_time = time.time()
    for _ in range(100000):
        count_emoji_original(text_with_emoji)
    end_time = time.time()
    print(f"Original (with emoji): 100000 strings in {end_time - start_time:.4f}s.")

    start_time = time.time()
    for _ in range(100000):
        count_emoji_regex(text_with_emoji)
    end_time = time.time()
    print(f"Regex (with emoji): 100000 strings in {end_time - start_time:.4f}s.")

    start_time = time.time()
    for _ in range(100000):
        count_emoji_original(text_without_emoji)
    end_time = time.time()
    print(f"Original (no emoji): 100000 strings in {end_time - start_time:.4f}s.")

    start_time = time.time()
    for _ in range(100000):
        count_emoji_regex(text_without_emoji)
    end_time = time.time()
    print(f"Regex (no emoji): 100000 strings in {end_time - start_time:.4f}s.")


if __name__ == "__main__":
    main()
