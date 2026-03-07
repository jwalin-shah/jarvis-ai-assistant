import time

from jarvis.topics.entity_anchor import EntityAnchorTracker


def benchmark():
    tracker = EntityAnchorTracker()
    texts = [f"This is sentence number {i} with a name like John and a noun like Apple."
             for i in range(1000)]

    start = time.time()
    for text in texts:
        tracker.get_anchors(text)
    end = time.time()
    print(f"Sequential processing time: {end - start:.4f} seconds")

    start2 = time.time()
    tracker.get_anchors_batch(texts)
    end2 = time.time()
    print(f"Batch processing time: {end2 - start2:.4f} seconds")

if __name__ == "__main__":
    benchmark()
