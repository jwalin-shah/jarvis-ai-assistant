import re
import timeit

_EMOJI_RE = re.compile("[\U0001F300-\U0001F9FF\u2600-\u27BF\U0001F000-\U0001F02F\u2B50]")

def _is_emoji_old(text: str) -> bool:
    for char in text:
        code = ord(char)
        if 0x1F300 <= code <= 0x1F9FF or 0x2600 <= code <= 0x27BF or 0x1F000 <= code <= 0x1F02F:
            return True
    return False

def count_old(text):
    return sum(1 for c in text if _is_emoji_old(c))

def count_new(text):
    return len(_EMOJI_RE.findall(text))

text = "Hello world! This is a test string. Let's add some emojis: 😀😎🌟⭐. And some more text just to make it longer and more realistic for a message."

def main():
    print("Old count time:", timeit.timeit("count_old(text)", globals=globals(), number=100000))
    print("New count time:", timeit.timeit("count_new(text)", globals=globals(), number=100000))

if __name__ == "__main__":
    main()
