import timeit
import re

def _is_emoji_old(text: str) -> bool:
    for char in text:
        code = ord(char)
        if 0x1F300 <= code <= 0x1F9FF or 0x2600 <= code <= 0x27BF or 0x1F000 <= code <= 0x1F02F:
            return True
    return False

def count_emoji_old(text):
    return sum(1 for c in text if _is_emoji_old(c))

_EMOJI_RE = re.compile("[\U0001F300-\U0001F9FF\u2600-\u27BF\U0001F000-\U0001F02F\u2B50]")

def _is_emoji_new(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))

def count_emoji_new(text):
    return len(_EMOJI_RE.findall(text))

text_short = "Hello ⭐ world 🌍 !"
text_long = "Hello ⭐ world 🌍 ! 🎲 🀄" * 100
text_no_emoji = "This is a regular text without any emojis at all." * 100

print("Short text (old):", timeit.timeit(lambda: count_emoji_old(text_short), number=10000))
print("Short text (new):", timeit.timeit(lambda: count_emoji_new(text_short), number=10000))

print("Long text (old):", timeit.timeit(lambda: count_emoji_old(text_long), number=10000))
print("Long text (new):", timeit.timeit(lambda: count_emoji_new(text_long), number=10000))

print("No emoji (old):", timeit.timeit(lambda: count_emoji_old(text_no_emoji), number=10000))
print("No emoji (new):", timeit.timeit(lambda: count_emoji_new(text_no_emoji), number=10000))
