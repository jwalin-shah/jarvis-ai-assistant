
import timeit
import re
from jarvis.text_normalizer import normalize_text

# Define locally for comparison since we are removing it from the codebase
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")

# Prepare data
text = "This   is  a    test   string   with   multiple    spaces.\n" * 100
text_unicode = "This\u00A0is\u00A0a\u00A0test\u00A0string\u00A0with\u00A0unicode\u00A0spaces.\n" * 100

def current_implementation(text):
    lines = text.split("\n")
    normalized_lines = []
    for line in lines:
        line = _WHITESPACE_PATTERN.sub(" ", line.strip())
        if line:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)

def optimized_implementation(text):
    lines = text.split("\n")
    normalized_lines = []
    for line in lines:
        line = " ".join(line.split())
        if line:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)

def benchmark():
    print("Running benchmarks...")

    # ASCII spaces
    t_current = timeit.timeit(lambda: current_implementation(text), number=1000)
    t_optimized = timeit.timeit(lambda: optimized_implementation(text), number=1000)

    print(f"ASCII Text - Current: {t_current:.4f}s")
    print(f"ASCII Text - Optimized: {t_optimized:.4f}s")
    print(f"Speedup: {t_current / t_optimized:.2f}x")

    # Unicode spaces
    t_current_uni = timeit.timeit(lambda: current_implementation(text_unicode), number=1000)
    t_optimized_uni = timeit.timeit(lambda: optimized_implementation(text_unicode), number=1000)

    print(f"Unicode Text - Current: {t_current_uni:.4f}s")
    print(f"Unicode Text - Optimized: {t_optimized_uni:.4f}s")
    print(f"Speedup: {t_current_uni / t_optimized_uni:.2f}x")

    # Correctness check
    print("\nCorrectness Check:")
    res_current = current_implementation("Hello\u00A0world")
    res_optimized = optimized_implementation("Hello\u00A0world")
    print(f"Input: 'Hello\\u00A0world'")
    print(f"Current result: '{res_current}' (len={len(res_current)})")
    print(f"Optimized result: '{res_optimized}' (len={len(res_optimized)})")

    if res_current != res_optimized:
        print("Note: Results differ (expected for unicode correction)")
        if len(res_optimized) < len(res_current):
             print("Optimization correctly normalized unicode spaces.")

    # Verify the actual function uses the optimized logic (implicitly)
    # We can't easily check implementation details from outside,
    # but we can check behavior on unicode if we haven't applied changes yet.
    # After changes, normalize_text should match optimized_implementation behavior.
    print("\nVerifying jarvis.text_normalizer.normalize_text behavior:")
    real_res = normalize_text("Hello\u00A0world")
    print(f"Real result: '{real_res}'")

if __name__ == "__main__":
    benchmark()
