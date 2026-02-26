
import random
import string
import timeit

from jarvis.text_normalizer import extract_text_features

random.seed(42)

def generate_random_text(word_count):
    words = []
    for _ in range(word_count):
        word_len = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
    return ' '.join(words)

def run_benchmark():
    # Test cases with different lengths
    short_texts = [generate_random_text(3) for _ in range(1000)]
    medium_texts = [generate_random_text(20) for _ in range(1000)]
    long_texts = [generate_random_text(100) for _ in range(1000)]

    all_texts = short_texts + medium_texts + long_texts

    print(f"Benchmarking extract_text_features with {len(all_texts)} texts...")

    def task():
        for text in all_texts:
            extract_text_features(text)

    iterations = 500
    total_time = timeit.timeit(task, number=iterations)
    avg_time_per_call = (total_time / (iterations * len(all_texts))) * 1_000_000  # microseconds

    print(f"Total time for {iterations} iterations: {total_time:.4f} seconds")
    print(f"Average time per call: {avg_time_per_call:.4f} microseconds")

if __name__ == "__main__":
    run_benchmark()
