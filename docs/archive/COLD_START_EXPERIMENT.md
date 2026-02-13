# Cold Start Classification Experiment

## Problem

New user downloads JARVIS. Zero labeled data. Need to bootstrap a classifier that works.

## Constraints

- No pre-labeled data from user
- Must work on Apple Silicon (MLX)
- Memory budget: 8GB
- Should be fast enough for good UX during onboarding

## Key Finding: Encoders Beat LLMs for Classification

**The pivot:** After extensive experiments, we found that encoder-based classification is:
- **70x faster** (sub-2ms vs ~140ms for LLM)
- **Lower memory** (~30MB vs ~800MB for LLM)
- **Comparable accuracy** with proper setup

This changes the entire approach from "LLM labels → train classifier" to "encoder zero-shot / clustering".

---

## Experimental Results

### Encoder Comparison (Zero-Shot)

Tested 4 encoder models on classification tasks:

| Model | Trigger Acc | Trigger Time | Response Acc | Response Time |
|-------|-------------|--------------|--------------|---------------|
| **bge-small** | **84%** | 1.49ms | 70% | 1.25ms |
| gte-tiny | 77% | 1.12ms | **74.7%** | 0.84ms |
| minilm-l6 | 74% | 1.11ms | 54.7% | 0.83ms |
| bge-micro | 80% | **0.85ms** | 67.3% | **0.77ms** |

**Note:** These are **accuracy** metrics, not F1 scores. Zero-shot experiments haven't calculated F1 yet.

**Per-class breakdown (bge-small, from `embedding_zeroshot_experiment.json`):**

Trigger (binary: needs_action vs casual):
- needs_action: 82.5% accuracy
- casual: 85% accuracy

Response (old 3-class: positive/negative/neutral):
- positive: 68.8% accuracy
- negative: 37.5% accuracy (weak - often confused with neutral)
- neutral: 77.1% accuracy

**Key insights:**
- bge-small: Best trigger accuracy (84%)
- gte-tiny: Best response accuracy (74.7%), good speed
- bge-micro: Fastest (0.8ms), still decent (80% trigger)
- All models are sub-2ms - suitable for real-time classification
- Response "negative" class is the weakest - needs better category definition

### Trained Classifiers (SVM on Embeddings)

When trained on labeled data, SVM classifiers significantly outperform zero-shot:

| Classifier | CV F1 | Test F1 | Test Acc |
|------------|-------|---------|----------|
| Response SVM (bge-small) | 0.78 | 0.80 | **83.3%** |
| Trigger SVM (bge-small) | - | 0.82 | ~85% |

This is ~10% better than zero-shot (70%), but requires labeled data.

### Comparison: Zero-Shot vs Trained (Summary)

| Approach | Trigger | Response | Labeled Data? | F1 Available? |
|----------|---------|----------|---------------|---------------|
| Zero-shot (bge-small) | 84% acc | 70% acc | No | **No** |
| Trained SVM | 82% F1 | 80% F1 | Yes (~5k samples) | Yes |

**Gap:** We need to calculate F1 for zero-shot to do a fair comparison.

### LLM Labeling Results

LLM-based labeling (LFM 1.2B) was tested but found to be:
- Slower (~140ms per message)
- Higher memory (~800MB)
- Similar accuracy to encoders for simple classification

**Conclusion:** For classification tasks, encoders are preferred.

---

## Current Label Taxonomy

### Trigger Classification (Incoming Messages)

**Fine-grained (5 classes):**
- `commitment` - Requests/invitations requiring yes/no ("Wanna hang?", "Can you send...")
- `question` - Direct questions ("What are you doing?", "U want food?")
- `statement` - Just informing, no action needed ("Rishi will drop me off")
- `reaction` - Emotional responses ("Ooooof", "Damn big boi")
- `social` - Social niceties ("Yeeep", "Hahah good luck")

**Collapsed for zero-shot (2 classes):**
- `needs_action` = commitment + question
- `casual` = statement + reaction + social

### Response Classification (Outgoing Messages)

**Fine-grained (6 classes):**
- `AGREE` - "I'm down", "bet", "Yee"
- `DECLINE` - "I'm out", "im not coming"
- `DEFER` - "Maybe later", "could be down"
- `QUESTION` - Follow-up questions
- `REACTION` - "bruhhh", "Hahahaah"
- `OTHER` - Everything else (statements, info)

**Collapsed for zero-shot (3 classes):**
- `answered_yes` = AGREE
- `answered_no` = DECLINE
- `no_answer` = DEFER + QUESTION + REACTION + OTHER

---

## New Approach: Unsupervised Clustering

### Why Clustering?

Instead of forcing pre-defined labels, let the data reveal natural categories:
1. **Generalizes better** - categories emerge from actual usage patterns
2. **No labeling needed** - true cold start
3. **Discovers unknown categories** - might find patterns we didn't anticipate

### Clustering Experiment Setup

**Script (archived):** `archive/scripts/experiment_clustering.py`

**Approach:**
1. Pull ALL messages from iMessage DB (~400k)
2. Separate incoming vs outgoing
3. Embed with all 4 encoder models
4. Cluster with K-Means (various K) and HDBSCAN
5. Inspect clusters to discover natural categories

### Clustering Algorithms

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **K-Means** | Fast, forces all points | Must specify K | Exploring structure |
| **K-Means (MLX)** | GPU-accelerated | Must specify K | Large datasets |
| **HDBSCAN** | Auto-detects K, handles noise | Slower | Finding natural clusters |
| **UMAP+HDBSCAN** | Better for high-dim | Two-step | Dense embeddings |

### Usage

```bash
# Ensure embed server is running
uv run python scripts/minimal_mlx_embed_server.py &

# Basic K-Means exploration
uv run python archive/scripts/experiment_clustering.py --k 5 10 15

# GPU-accelerated K-Means with MLX
uv run python archive/scripts/experiment_clustering.py --mlx --k 5 10 15

# HDBSCAN (auto-detects number of clusters)
uv run python archive/scripts/experiment_clustering.py --hdbscan --min-cluster-size 50

# Test all encoder models
uv run python archive/scripts/experiment_clustering.py --models bge-small gte-tiny minilm-l6 bge-micro

# Full run with cluster samples printed
uv run python archive/scripts/experiment_clustering.py --mlx --k 5 10 15 20 --print-samples
```

---

## Infrastructure

### Minimal MLX Embed Server

**Script:** `scripts/minimal_mlx_embed_server.py`

Custom BERT implementation in pure MLX that reduces memory overhead:
- **Before:** ~350MB (mlx_embedding_models library)
- **After:** ~30MB (minimal implementation)
- **Savings:** ~320MB RAM

**Features:**
- JSON-RPC over Unix socket (same protocol as original)
- Supports: bge-small, gte-tiny, minilm-l6, bge-micro
- CLS pooling for bge-small, mean pooling for others
- Automatic model loading on first request

**Usage:**
```bash
uv run python scripts/minimal_mlx_embed_server.py
# Socket: /tmp/jarvis-embed-minimal.sock
```

### Embedding Adapter

**File:** `jarvis/embedding_adapter.py`

Unified interface for embeddings:
- Connects to MLX embed server via Unix socket
- Supports model switching via config
- Includes caching layer (LRU with TTL)
- Thread-safe singleton pattern

---

## Gold Standard Test Data

### SetFit Training Samples

Carefully selected samples closest to class centroids (most representative):

**Location:** `results/setfit_training/`
- `selected_trigger_8.jsonl` - 40 trigger samples (8 per class)
- `selected_response_8.jsonl` - 48 response samples (8 per class)

These are "textbook" examples - clear, unambiguous, suitable for evaluation.

**Examples:**

Trigger:
- commitment: "Lets go", "Come through", "Let's hang bro"
- question: "Are u still up", "WHEN ARE U MOVING"
- reaction: "Ooooof", "Damn big boi"

Response:
- AGREE: "I'm down", "bet im in then"
- DECLINE: "im not coming", "Im out"
- DEFER: "Maybe later", "could be down"

---

## Experiment Scripts

| Script | Purpose |
|--------|---------|
| `archive/scripts/experiment_clustering.py` | Unsupervised clustering on all messages (archived) |
| `archive/scripts/experiment_all_encoders.py` | Compare 4 encoder models on zero-shot (archived) |
| `archive/scripts/experiment_embedding_zeroshot_v2.py` | Test different category schemes (archived) |
| `archive/scripts/experiment_llm_labeling.py` | LLM-based labeling (deprecated approach, archived) |
| `scripts/train_all_classifiers.py` | Train SVM/LR/RF/XGBoost on embeddings |
| `scripts/train_setfit.py` | SetFit contrastive learning |
| `scripts/minimal_mlx_embed_server.py` | Lightweight embed server |

---

## Results Directory Structure

```
results/
├── clustering/                    # Clustering experiment outputs
├── encoder_comparison.json        # 4-model zero-shot comparison
├── classifier_training/           # Trained classifier results
│   ├── training_results.json
│   └── results_*.json
├── setfit_training/               # SetFit training data & results
│   ├── selected_trigger_8.jsonl   # Gold standard trigger samples
│   ├── selected_response_8.jsonl  # Gold standard response samples
│   └── setfit_results.json
└── response_category_comparison.json  # Category scheme experiments
```

---

## Next Steps

### Phase 1: Run Clustering Experiment
1. Start minimal MLX embed server
2. Run clustering on all ~400k messages
3. Inspect clusters to discover natural categories
4. Compare with our hand-crafted taxonomy

### Phase 2: Validate Categories
1. Use gold standard samples to evaluate
2. Test zero-shot with discovered categories
3. Compare accuracy to current approach

### Phase 3: Implement Cold Start Pipeline
1. New user → embed recent messages
2. Cluster → discover their message patterns
3. Use zero-shot with appropriate categories
4. No training data needed

---

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| Cold start time | < 30 sec | TBD |
| Classification latency | < 5ms | **< 2ms** |
| Memory overhead | < 100MB | **~30MB** |
| Trigger accuracy (zero-shot) | > 80% | **84%** |
| Response accuracy (zero-shot) | > 70% | **70-75%** |

---

## Metrics Gap: What We're Missing

### Current State
- **Trained classifiers:** Have F1 scores (82% macro F1)
- **Zero-shot:** Only have accuracy, no F1 calculated yet

### Needed Experiments
1. **Calculate F1 for zero-shot** - Use same test set as trained classifiers
2. **Compare apples-to-apples** - Same evaluation set, same metrics
3. **Per-class F1 for zero-shot** - Identify weak categories

### Why F1 Matters
- Accuracy can be misleading with imbalanced classes
- F1 balances precision and recall
- Macro F1 treats all classes equally (important for minority classes)

---

## Open Questions

1. **What natural clusters exist?** - Pending clustering experiment
2. **Do clusters align with our taxonomy?** - Will compare after clustering
3. **Can zero-shot match trained classifiers?** - Currently ~10% gap (accuracy), need F1 comparison
4. **Best encoder for each task?** - bge-small for trigger, gte-tiny for response
5. **What's the F1 for zero-shot?** - Need to calculate on same test set as trained classifiers

---

## Historical Context

### Original Approach (Deprecated)
The original plan was:
1. Embed messages
2. Cluster for diversity sampling
3. LLM labels representative samples
4. Train SetFit on LLM labels

**Why deprecated:** LLM labeling is slower and not significantly more accurate than encoder zero-shot. Direct encoder classification is simpler and faster.

### Current Approach
1. Embed messages with lightweight MLX server
2. Cluster to discover natural categories (unsupervised)
3. Zero-shot classify using category descriptions
4. No training data required

This is a true "cold start" solution - works with zero labeled data.
