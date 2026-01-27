# Overnight Experiment Guide

## What This Tests

Runs the **3 best embedding models** (based on MTEB scores) on ALL your iMessage messages:

| Model | MTEB Score | Strength | Size | Expected Runtime |
|-------|------------|----------|------|------------------|
| **all-mpnet-base-v2** | 87-88 STS | Semantic similarity | 110M | ~2 hrs |
| **NV-Embed-v2** | 62.65 Retrieval | Finding similar texts | 7.9B | ~3 hrs |
| **Qwen3-Embedding-8B** | ~76 Classification | Categorizing texts | 8B | ~3 hrs |

**Total Runtime:** ~6-8 hours

---

## How to Run

### Start the Experiment (Tonight):

```bash
# Navigate to project
cd ~/coding/jarvis-ai-assistant

# Start experiment (runs in background)
nohup ./scripts/overnight_best_models.sh > overnight.log 2>&1 &

# Get process ID (save this!)
echo $!

# Check it's running
tail -f overnight.log  # Ctrl+C to exit
```

### Monitor Progress (Optional):

```bash
# Check current status
tail -f results/latest/experiment.log

# See which task is running
ps aux | grep "benchmarks.templates.mine"
```

### Stop Experiment (If Needed):

```bash
# Kill by process ID (from earlier)
kill <PID>

# Or kill by name
pkill -f "benchmarks.templates.mine"
```

---

## What Happens Overnight

```
Hour 0-2:   Task 1 - all-mpnet-base-v2 (Best STS)
            ├─ Load 22k+ messages
            ├─ Encode with 110M param model
            ├─ Cluster similar patterns (DBSCAN)
            └─ Extract templates (coverage: ?%)

Hour 2-5:   Task 2 - NV-Embed-v2 (Best Retrieval)
            ├─ Load 22k+ messages
            ├─ Encode with 7.9B param model (SLOW!)
            ├─ Cluster similar patterns
            └─ Extract templates (coverage: ?%)

Hour 5-8:   Task 3 - Qwen3-Embedding-8B (Best Classification)
            ├─ Load 22k+ messages
            ├─ Encode with 8B param model
            ├─ Cluster similar patterns
            └─ Extract templates (coverage: ?%)

Hour 8:     Generate Report
            ├─ Compare all 3 models
            ├─ Show top templates from each
            └─ Create REPORT.md
```

---

## Results Tomorrow Morning

### Quick Check:

```bash
# View the report
cat results/latest/REPORT.md

# Or open in editor
code results/latest/REPORT.md
```

### What to Look For:

1. **Coverage** - Which model found the most patterns?
   - Target: 30-50% (vs current 6.2%)

2. **Template Quality** - Which templates are useful?
   - Look at top 10 from each model
   - Are they better than manual templates?

3. **Unique Insights** - Did different models find different patterns?
   - Patterns only NV-Embed found (retrieval strength)
   - Patterns only mpnet found (semantic similarity strength)

4. **Winner** - Which model should JARVIS use?
   - Best coverage?
   - Best templates?
   - Worth the extra compute?

### Compare Against Baseline:

Current (all-MiniLM-L6-v2 with manual templates):
- **Coverage: 6.2%** (33/528 messages)
- **Templates: 91 manual** (79% unused!)
- **Top template:** "quick_affirmative" (5 uses)

Expected improvements:
- **Coverage: 30-50%** (10× better!)
- **Templates: 50-150 data-driven**
- **Top template:** ???

---

## Troubleshooting

### If it crashes:

```bash
# Check the log
tail -100 results/latest/experiment.log

# Common issues:
# 1. Out of memory (8B models are BIG)
#    → Close other apps
#    → Restart and run one model at a time

# 2. Model download failed
#    → Check internet connection
#    → Models are large (NV-Embed: ~16GB)

# 3. iMessage permission denied
#    → Grant Full Disk Access in System Settings
```

### Run One Model at a Time:

If 8B models are too much:

```bash
# Just test all-mpnet-base-v2 (110M - smaller)
uv run python -m benchmarks.templates.mine \
    --sample-size 100000 \
    --model "sentence-transformers/all-mpnet-base-v2" \
    --output results/mpnet_only.json
```

---

## Analysis Tomorrow

Questions to answer:

### 1. Which model won?
- [ ] Highest coverage?
- [ ] Best template quality?
- [ ] Most useful patterns?

### 2. Does MTEB predict real-world performance?
- [ ] Did "best STS" (mpnet) actually produce best templates?
- [ ] Did "best Retrieval" (NV-Embed) find more patterns?
- [ ] Did "best Classification" (Qwen3) categorize better?

### 3. Is bigger better?
- [ ] Did 7.9B/8B models beat 110M model?
- [ ] Worth the extra compute?
- [ ] Or is mpnet "good enough"?

### 4. What's next?
- [ ] Which model for production?
- [ ] Hybrid approach (different models for different tasks)?
- [ ] Need to tune clustering parameters?

---

## Expected Files Tomorrow

```
results/
└── overnight_20260126_220000/
    ├── REPORT.md                    ← START HERE
    ├── experiment.log              ← Full execution log
    ├── templates_mpnet.json        ← all-mpnet results
    ├── templates_nvembed.json      ← NV-Embed results
    └── templates_qwen3.json        ← Qwen3 results

results/latest → overnight_20260126_220000/  ← Symlink for easy access
```

---

## Timeline

```
Tonight 10pm:   Start experiment
Tonight 10:01:  Go to bed! 😴

Tomorrow 6am:   Wake up
Tomorrow 6:01:  cat results/latest/REPORT.md
Tomorrow 6:15:  Choose winning model
Tomorrow 6:30:  Deploy to JARVIS
```

🎯 **Goal:** Find the model that produces the best templates for real iMessage conversations!
