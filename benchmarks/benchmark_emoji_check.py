import time
import re

# Use normal string for unicode escape handling
_EMOJI_RE = re.compile("[\U0001F300-\U0001F9FF\u2600-\u27BF\U0001F000-\U0001F02F]")

def count_emoji_regex(text: str) -> int:
    return len(_EMOJI_RE.findall(text))

from jarvis.classifiers.relationship_classifier import _is_emoji

def count_emoji_original(text: str) -> int:
    return sum(1 for c in text if _is_emoji(c))

def main():
    text_with_emoji = "Hello world! This is a test string to see how fast emoji detection is. ⭐😭😢" * 10
    text_without_emoji = "Hello world! This is a test string to see how fast emoji detection is." * 10

    start_time = time.time()
    for _ in range(100000):
        count_emoji_original(text_with_emoji)
    end_time = time.time()
    print(f"Original Count (with emoji): Processed 100000 strings in {end_time - start_time:.4f} seconds.")

    start_time = time.time()
    for _ in range(100000):
        count_emoji_regex(text_with_emoji)
    end_time = time.time()
    print(f"Regex Count (with emoji): Processed 100000 strings in {end_time - start_time:.4f} seconds.")

    start_time = time.time()
    for _ in range(100000):
        count_emoji_original(text_without_emoji)
    end_time = time.time()
    print(f"Original Count (no emoji): Processed 100000 strings in {end_time - start_time:.4f} seconds.")

    start_time = time.time()
    for _ in range(100000):
        count_emoji_regex(text_without_emoji)
    end_time = time.time()
    print(f"Regex Count (no emoji): Processed 100000 strings in {end_time - start_time:.4f} seconds.")

if __name__ == '__main__':
    main()
