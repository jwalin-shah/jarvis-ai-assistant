# Learning Over Time

## 3A. Feedback Loop

**Goal**: Learn from user acceptance/rejection of suggestions.

### Feedback Signals

| Signal | Interpretation |
|--------|----------------|
| Accepted | Sent as-is → positive |
| Modified | Edited before sending → weak positive |
| Rejected | Different response typed → negative |
| Ignored | No action → neutral |

### Implementation

```python
def record_feedback(response_id, action):
    feedback_store.record(response_id, action, timestamp=now())
```

**Training updates:**
- Periodically retrain classifiers with feedback
- Adjust similarity thresholds based on acceptance
- Update quality scores for pairs

**Expected Impact**: Acceptance rate 60% → 75%+
**Effort**: Medium (2-3 weeks)
**Dependency**: Desktop app integration

---

## 3B. Online Learning (Adaptive Thresholds)

**Goal**: Dynamically adjust routing thresholds.

### Implementation

```python
class AdaptiveThresholds:
    def __init__(self):
        self.quick_reply = 0.95
        self.context = 0.65
        self.generate = 0.45
        self.window = deque(maxlen=1000)

    def update(self, similarity, was_accepted):
        self.window.append((similarity, was_accepted))
        if len(self.window) >= 100:
            self._recalibrate()
```

**Recalibration logic:**
- If quick_reply acceptance < 80%, raise threshold
- If too few quick_replies, lower threshold
- Bounded: never below 0.80 or above 0.98

**Expected Impact**: Quick reply rate 5% → 10-15%, accuracy maintained
**Effort**: Low (1 week)
**Dependency**: Feedback loop (3A)

---

## 3C. Conversation Outcome Tracking

**Goal**: Track whether conversations ended positively.

### Outcome Detection

```python
def detect_outcome(conversation):
    last_messages = conversation.messages[-3:]

    positive_signals = ["thanks", "sounds good", "see you"]
    negative_signals = ["whatever", "fine", "forget it"]

    return compute_outcome_score(last_messages, ...)
```

**Quality adjustment:**
- Positive-outcome responses get quality boost
- Negative-outcome responses get penalty

**Effort**: Medium (2 weeks)
**Dependency**: Conversation threading

---

## 3D. Cluster Refinement

**Goal**: Improve response clusters based on usage.

### Implementation

```python
def refine_clusters(min_samples=50):
    for cluster in clusters:
        if cluster.acceptance_rate < 0.5:
            sub_clusters = recluster(cluster.members)
            update_cluster_assignments(sub_clusters)
```

**Expected Impact**: Cluster purity 0.70 → 0.85+, retrieval precision +15%
**Effort**: Medium (2-3 weeks)
