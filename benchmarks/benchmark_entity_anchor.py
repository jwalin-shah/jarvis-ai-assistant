import time
from jarvis.topics.entity_anchor import get_tracker

def main():
    tracker = get_tracker()
    texts = [
        "Hey, are you going to the party tonight?",
        "I talked to John about the new project.",
        "The server is down again, we need to fix it.",
        "Let's meet at 5pm to discuss the marketing strategy.",
        "Can you send me the report by tomorrow?",
        "I'm feeling a bit sick, might not make it.",
        "Did you see the latest movie?",
        "We need to buy groceries: milk, eggs, bread.",
        "The weather is so nice today!",
        "Don't forget to pay the internet bill."
    ] * 100 # 1000 messages

    print(f"Testing {len(texts)} messages...")

    start = time.time()
    for text in texts:
        tracker.get_anchors(text)
    seq_time = time.time() - start
    print(f"Sequential time: {seq_time:.3f}s")

    start = time.time()
    tracker.get_anchors_batch(texts)
    batch_time = time.time() - start
    print(f"Batch time: {batch_time:.3f}s")

    print(f"Speedup: {seq_time/batch_time:.2f}x")

if __name__ == "__main__":
    main()
