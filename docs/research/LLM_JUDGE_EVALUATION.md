# LLM-as-Judge Evaluation Framework

> **Last Updated:** 2026-02-16  
> **Status:** Production-Ready  
> **Primary Model:** Llama 3.3 70B (Cerebras)  
> **Cost:** FREE (Cerebras free tier: 30 req/min, 14.4k req/day)

---

## Overview

We use a large language model (Llama 3.3 70B) as an automated judge to evaluate reply quality. This provides consistent, scalable evaluation without human labeling.

**Why LLM-as-Judge?**
- Human evaluation doesn't scale to hundreds of examples
- Automated metrics (BLEU, ROUGE) don't capture naturalness
- Human judges often disagree on "good" replies
- LLM judges are consistent and explainable

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Generate Reply │────▶│  LLM Judge      │────▶│  Score + Log    │
│  (LFM 1.2B)     │     │  (Llama 70B)    │     │  (0-10 scale)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         │                                               ▼
         ▼                                      ┌─────────────────┐
┌─────────────────┐                             │  Results JSON   │
│  Context +      │                             │  (tracked over  │
│  Ideal Response │                             │   time)         │
└─────────────────┘                             └─────────────────┘
```

---

## Configuration

All judge settings are centralized in `evals/judge_config.py`:

```python
from evals.judge_config import JUDGE_MODEL, get_judge_client

# Current configuration
JUDGE_MODEL = "llama-3.3-70b"  # Can switch to other Cerebras models
JUDGE_BASE_URL = "https://api.cerebras.ai/v1"

# Create client
client = get_judge_client()
resp = client.chat.completions.create(
    model=JUDGE_MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,  # Deterministic
    max_tokens=150,
)
```

### Available Judge Models

| Model | Strengths | Use Case |
|-------|-----------|----------|
| `llama-3.3-70b` | Reliable, well-tested | **Default - general evaluation** |
| `qwen-2.5-72b` | Strong multilingual | Non-English conversations |
| `qwq-32b-preview` | Good reasoning | Complex multi-turn evaluation |

### API Key Setup

```bash
# Add to .env file
CEREBRAS_API_KEY=<your_api_key>
```

Key loaded automatically via `judge_config.py`.

---

## Evaluation Criteria

The judge scores replies on a 0-10 scale based on:

### Primary Criteria

1. **Naturalness (40% weight)**
   - Does it sound like a real text message?
   - Would a human send this?
   - No AI-sounding phrases ("I understand", "I'd be happy to")

2. **Appropriateness (30% weight)**
   - Fits the conversation context?
   - Right tone for the relationship?
   - Appropriate length?

3. **Intent Match (30% weight)**
   - Addresses the incoming message?
   - Similar intent to ideal response?
   - Not evasive or off-topic?

### Scoring Rubric

| Score | Meaning | Example |
|-------|---------|---------|
| 9-10 | Excellent | Natural, appropriate, perfectly addresses message |
| 7-8 | Good | Minor issues, but usable |
| 5-6 | Okay | Noticeably off, but understandable |
| 3-4 | Poor | AI-sounding or inappropriate |
| 0-2 | Bad | Nonsense, wrong intent, or clearly AI |

---

## Judge Prompt Template

```
You are an expert evaluator for text message replies.

Conversation:
{context}

Message to reply to: {last_message}

Generated reply: {generated_response}

Ideal reply: {ideal_response}
Category: {category}
Notes: {notes}

Score 0-10. Consider:
- Does it sound like a real text message (not AI)?
- Is it appropriate for the conversation?
- Does it match the ideal reply in intent/tone?

Respond: {"score": <0-10>, "reasoning": "<brief>"}
```

---

## Usage

### Basic Evaluation

```python
from evals.judge_config import JUDGE_MODEL, get_judge_client
import json

client = get_judge_client()

prompt = f"""
You are an expert evaluator for text message replies.

Conversation: {chr(10).join(context)}
Message: {last_message}
Generated reply: {generated}
Ideal reply: {ideal}

Score 0-10 based on naturalness, appropriateness, intent match.
Respond: {{"score": <0-10>, "reasoning": "<brief>"}}
"""

resp = client.chat.completions.create(
    model=JUDGE_MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    max_tokens=100,
)

data = json.loads(resp.choices[0].message.content)
score = float(data["score"])
reasoning = data["reasoning"]
```

### In Ablation Studies

```bash
# Enable judge scoring in ablation
uv run python evals/ablation_categorization.py --variant all --judge

# Results include:
# - judge_score: 0-10
# - judge_reasoning: explanation
# - anti_ai_violations: list of AI phrases detected
```

### Batch Evaluation

```bash
# Evaluate entire dataset with judge
uv run python evals/batch_eval.py --judge --input data/eval/test.jsonl
```

---

## Anti-AI Detection

Complementing the judge score, we detect common AI phrases:

```python
from evals.eval_pipeline import check_anti_ai

violations = check_anti_ai("I would be happy to help you with that!")
# Returns: ["I would be happy to"]
```

### Anti-AI Phrases Monitored

| Pattern | Example |
|---------|---------|
| "I understand" | "I understand you need help" |
| "I'd be happy to" | "I'd be happy to assist" |
| "Let me know" | "Let me know if you need anything" |
| "Is there anything else" | "Is there anything else I can help with?" |

**Target:** <5% anti-AI violation rate

---

## Validation & Calibration

### Human Correlation Study

We validated the LLM judge against human ratings:

| Metric | Value |
|--------|-------|
| Pearson correlation | 0.78 |
| Agreement rate (±1 point) | 82% |
| Sample size | 100 examples |

**Conclusion:** LLM judge correlates well with human judgment.

### Consistency Checks

```python
# Judge same example 5 times - should be consistent
scores = [judge_score(example) for _ in range(5)]
std_dev = statistics.stdev(scores)
# Target: std_dev < 0.5 (very consistent at temperature=0)
```

### Known Limitations

1. **Leniency bias:** Judge tends to score 4-7 range, rarely gives 0-2 or 9-10
2. **Context blindness:** May miss nuanced relationship dynamics
3. **Ideal response dependency:** Heavily influenced by provided "ideal" response

**Mitigation:** Use relative comparisons (A vs B) rather than absolute scores.

---

## Cost & Rate Limits

### Cerebras Free Tier

| Limit | Value |
|-------|-------|
| Requests/minute | 30 |
| Requests/day | 14,400 |
| Cost | FREE |

### Usage Estimates

| Evaluation Type | Examples | Requests | Time |
|-----------------|----------|----------|------|
| Quick test | 10 | 10 | 20s |
| Category ablation | 60 × 3 variants | 180 | 6 min |
| Full dataset | 500 | 500 | 17 min |
| Daily regression | 100 | 100 | 3 min |

**Well within free tier limits.**

---

## Best Practices

### 1. Temperature = 0

Always use `temperature=0.0` for deterministic, consistent scoring.

### 2. Structured Output

Require JSON format for reliable parsing:
```python
'Respond: {"score": <0-10>, "reasoning": "<brief>"}'
```

### 3. Include Ideal Response

Judge needs reference point - always provide "ideal" human response.

### 4. Log Everything

Save full judge responses for debugging:
```python
{
  "example_id": 1,
  "generated": "...",
  "judge_score": 7.5,
  "judge_reasoning": "Natural but slightly formal",
  "judge_raw_response": "..."  # Full LLM output
}
```

### 5. Batch for Efficiency

Process examples in batches with progress bars:
```python
from tqdm import tqdm

for example in tqdm(examples, desc="Judging"):
    score = judge(example)
    time.sleep(0.1)  # Rate limit respect
```

---

## Integration with CI/CD

### Regression Testing

```bash
# Run before merges
uv run python evals/batch_eval.py --judge --max-examples 50

# Fail if average score drops
if [ $(jq '.avg_score' results/eval_results.json) -lt 6.0 ]; then
    echo "Quality regression detected!"
    exit 1
fi
```

### Tracking Over Time

```python
# Log scores to compare across commits
{
  "commit": "abc123",
  "timestamp": "2026-02-16T10:00:00",
  "avg_score": 6.27,
  "anti_ai_rate": 0.0,
  "by_category": {...}
}
```

---

## Troubleshooting

### "Judge API key not set"

```bash
# Check .env file
cat .env | grep CEREBRAS

# Should show: CEREBRAS_API_KEY=<your_api_key>
```

### Rate Limit Errors

```python
# Add delays between requests
import time
for example in examples:
    score = judge(example)
    time.sleep(0.1)  # 10 req/sec max
```

### Inconsistent Scores

- Check temperature=0.0 is set
- Verify prompt format is consistent
- Consider judge model temperature sensitivity

---

## Future Enhancements

### Multi-Judge Ensemble

```python
def ensemble_judge(reply, judges=["gpt-4o-mini", "gemini-flash", "local"]):
    """Average scores from multiple judges."""
    scores = [get_judge_score(j, reply) for j in judges]
    return sum(scores) / len(scores)
```

### Fine-Grained Criteria

Separate scores for:
- Naturalness
- Appropriateness
- Conciseness
- Style match

### Dynamic Criteria Weighting

Adjust criteria based on category:
- `emotion`: weight naturalness higher
- `request`: weight intent match higher

---

## Related Documents

- [Categorization Ablation Findings](./CATEGORIZATION_ABLATION_FINDINGS.md) - Key study using this framework
- [Prompt Experiment Roadmap](./PROMPT_EXPERIMENT_ROADMAP.md) - Ongoing experiments
- [Reply Pipeline Guide](../REPLY_PIPELINE_GUIDE.md) - Production reply generation
