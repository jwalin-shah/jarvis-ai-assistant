import time
from jarvis.topics.entity_anchor import EntityAnchorTracker

tracker = EntityAnchorTracker()

texts = ["Hello there, this is a test about Apple and Microsoft."] * 1000

t0 = time.time()
for t in texts:
    tracker.get_anchors(t)
t1 = time.time()
print(f"Sequential: {t1-t0:.2f}s")

def get_anchors_batch(texts):
    docs = tracker.nlp.pipe(texts, batch_size=50)
    results = []
    for doc in docs:
        anchors = set()
        for ent in doc.ents:
            anchors.add(ent.text.lower())
        for chunk in doc.noun_chunks:
            chunk_text = chunk.root.text.lower()
            if len(chunk_text) > 2 and chunk.root.pos_ != "PRON":
                anchors.add(chunk_text)
        results.append(anchors)
    return results

t2 = time.time()
get_anchors_batch(texts)
t3 = time.time()
print(f"Batch: {t3-t2:.2f}s")
