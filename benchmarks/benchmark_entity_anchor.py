import time
from jarvis.topics.entity_anchor import EntityAnchorTracker

def run_benchmark():
    tracker = EntityAnchorTracker()
    texts = ["Hey how are you?", "I am going to the store.", "Did you see John today?", "Let's meet at the park."] * 100

    print("Running sequential...")
    start = time.time()
    res1 = [tracker.get_anchors(t) for t in texts]
    seq_time = time.time() - start

    print("Running batch...")
    start = time.time()
    res2 = tracker.get_anchors_batch(texts)
    batch_time = time.time() - start

    print(f"Sequential: {seq_time:.4f}s")
    print(f"Batch: {batch_time:.4f}s")
    print(f"Speedup: {seq_time / batch_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()
