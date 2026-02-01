"""Final Trigger Classifier: Patterns + 5-class SVM (best performing)."""
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
import pickle
import os

# Merge map: 10 classes -> 5 classes
MERGE_MAP = {
    'info_question': 'question',
    'yn_question': 'question',
    'invitation': 'action',
    'request': 'action',
    'good_news': 'emotional',
    'bad_news': 'emotional',
    'reaction': 'emotional',
    'ack': 'acknowledgment',
    'greeting': 'acknowledgment',
    'statement': 'statement',
}

class FinalTriggerClassifier:
    """Production-ready trigger classifier."""
    
    TAPBACK_PATTERN = re.compile(r'^(Liked|Loved|Laughed at|Emphasized|Disliked|Questioned)\s+["\'""]', re.IGNORECASE)
    EXACT_ACKS = {'ok', 'okay', 'k', 'kk', 'bet', 'yea', 'yeah', 'yes', 'ya', 'ye', 
                  'nah', 'no', 'nope', 'aight', 'ight', 'alright', 'cool', 'nice', 
                  'word', 'true', 'facts', 'same', 'mood', 'lol', 'lmao', 'fs', 
                  'gotcha', 'np', 'ty', 'thx', 'thanks', 'yup', 'yep', 'mhm'}
    
    def __init__(self):
        self.model = None
        self.clf = None
        self.labels = ['acknowledgment', 'action', 'emotional', 'question', 'statement']
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}
    
    def pattern_match(self, text):
        """High-precision patterns."""
        text_lower = text.lower().strip()
        text_clean = re.sub(r'[^\w\s]', '', text_lower).strip()
        
        if self.TAPBACK_PATTERN.match(text):
            return 'acknowledgment', 0.99
        if text_clean in self.EXACT_ACKS:
            return 'acknowledgment', 0.95
        if text.strip().endswith('?'):
            return 'question', 0.85
        return None, 0.0
    
    def train(self, data_path):
        """Train on labeled data."""
        with open(data_path) as f:
            pairs = [json.loads(line) for line in f if line.strip()]
        
        # Apply merge
        for p in pairs:
            p['merged_label'] = MERGE_MAP[p['label']]
        
        texts = [p['trigger_text'] for p in pairs]
        label_ids = [self.label2id[p['merged_label']] for p in pairs]
        
        print("Loading sentence transformer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Encoding texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        print("Training SVM...")
        self.clf = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', 
                       probability=True, random_state=42)
        self.clf.fit(embeddings, label_ids)
        print(f"Trained on {len(pairs)} examples")
    
    def classify(self, text):
        """Classify a trigger. Returns (label, confidence, method)."""
        # Try patterns
        pattern_label, pattern_conf = self.pattern_match(text)
        if pattern_label and pattern_conf >= 0.85:
            return pattern_label, pattern_conf, 'pattern'
        
        # SVM
        embedding = self.model.encode([text])
        probs = self.clf.predict_proba(embedding)[0]
        pred_idx = np.argmax(probs)
        return self.id2label[pred_idx], float(probs[pred_idx]), 'svm'
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/svm.pkl', 'wb') as f:
            pickle.dump(self.clf, f)
        with open(f'{path}/config.json', 'w') as f:
            json.dump({'labels': self.labels, 'merge_map': MERGE_MAP}, f)
        print(f"Saved to {path}/")
    
    def load(self, path):
        with open(f'{path}/svm.pkl', 'rb') as f:
            self.clf = pickle.load(f)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


def main():
    import random
    from sklearn.metrics import classification_report, accuracy_score
    
    print("="*60)
    print("FINAL TRIGGER CLASSIFIER (5 merged classes)")
    print("="*60)
    
    # Load and split
    with open('/tmp/pairs_500_labeled.jsonl') as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    
    for p in pairs:
        p['merged_label'] = MERGE_MAP[p['label']]
    
    random.seed(42)
    by_label = {}
    for i, p in enumerate(pairs):
        if p['merged_label'] not in by_label:
            by_label[p['merged_label']] = []
        by_label[p['merged_label']].append(i)
    
    train_idx, test_idx = [], []
    for idxs in by_label.values():
        random.shuffle(idxs)
        split = int(0.8 * len(idxs))
        train_idx.extend(idxs[:split])
        test_idx.extend(idxs[split:])
    
    train_pairs = [pairs[i] for i in train_idx]
    test_pairs = [pairs[i] for i in test_idx]
    
    # Save train set
    with open('/tmp/train_merged.jsonl', 'w') as f:
        for p in train_pairs:
            f.write(json.dumps(p) + '\n')
    
    print(f"\nTrain: {len(train_pairs)}, Test: {len(test_pairs)}")
    
    # Train
    clf = FinalTriggerClassifier()
    clf.train('/tmp/train_merged.jsonl')
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    true_labels = []
    pred_labels = []
    methods = {'pattern': 0, 'svm': 0}
    
    for pair in test_pairs:
        true = clf.label2id[pair['merged_label']]
        pred_label, conf, method = clf.classify(pair['trigger_text'])
        pred = clf.label2id[pred_label]
        true_labels.append(true)
        pred_labels.append(pred)
        methods[method] += 1
    
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\nOVERALL ACCURACY: {acc:.1%}")
    print(f"Methods: {methods}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, 
                                target_names=clf.labels, digits=2, zero_division=0))
    
    # Per-class
    print("Per-class accuracy:")
    for label in clf.labels:
        lid = clf.label2id[label]
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == lid and t == p)
        total = sum(1 for t in true_labels if t == lid)
        if total > 0:
            print(f"  {label:15} {correct:3}/{total:3} = {correct/total:.1%}")
    
    # Save
    clf.save('/tmp/final_trigger_classifier')
    
    # Demo
    print("\n" + "="*60)
    print("DEMO PREDICTIONS")
    print("="*60)
    demos = [
        "Liked \"that's hilarious\"",
        "what time are you coming?",
        "yeah sounds good",
        "I just got back from the gym",
        "YESSSS WE WON",
        "can you pick me up at 5?",
        "damn that sucks bro",
    ]
    for text in demos:
        label, conf, method = clf.classify(text)
        print(f"  \"{text}\" -> {label} ({method}, {conf:.2f})")

if __name__ == '__main__':
    main()
