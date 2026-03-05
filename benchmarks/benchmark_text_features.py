import time
import timeit
from jarvis.text_normalizer import extract_text_features

def main():
    texts = [
        "Sounds good! I will be there at 5pm. See you!",
        "btw, can you send me that document?",
        "ok",
        "😭 😭 😭",
        "This is a longer message that has more words in it to test the word count feature. " * 5
    ] * 1000

    start_time = time.time()
    for text in texts:
        extract_text_features(text)
    end_time = time.time()

    print(f"Processed {len(texts)} texts in {end_time - start_time:.4f} seconds.")

if __name__ == '__main__':
    main()
