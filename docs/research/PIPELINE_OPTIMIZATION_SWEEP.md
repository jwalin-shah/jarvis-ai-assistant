# Pipeline Optimization Sweep Findings

> **Date:** February 16, 2026
> **Model:** Liquid AI LFM-0.7B / LFM-1.2B
> **Teacher/Judge:** GPT-OSS-120B (via Cerebras)

## Objective
To find the optimal "architectural" settings (context depth, sampling parameters) and "textual" instructions for the global reply pipeline, moving away from category-specific logic.

## Summary of Results

| Configuration | Metric (0-10 Scale) | Note |
|---------------|-------------------|------|
| D=10, RP=1.05 | 2.03 | High confusion, rambling |
| D=5, RP=1.05  | 3.00 | Moderate performance |
| D=3, RP=1.0   | 3.73 | Baseline winner |
| D=3, RP=1.05  | 4.00 | Improved brevity |
| **D=3, RP=1.1 (Winner)** | **4.50** | **Best tone match, no echoing** |

## Key Findings

### 1. The "Context Trap" for Small Models
Contrary to intuition, **more context made the model significantly worse.**
- At **Depth 10**, the 0.7B model often got confused by older parts of the conversation and began echoing previous messages or hallucinating.
- At **Depth 3**, the model stayed "on rails," focusing only on the immediate exchange. This resulted in a ~25% score improvement.

### 2. ChatML vs. Plain Text
The Liquid AI LFM model showed marked improvement in instruction following when using the official ChatML format:
```text
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{context}
{last_message}<|im_end|>
<|im_start|>assistant
```
Plain text prompts often caused the model to include "Reply:" or other labels in the actual response.

### 3. Repetition Penalty Sensitivity
- **1.0 (None):** Model occasionally looped or echoed the user's message.
- **1.05:** Better, but still some "AI-isms" leaked through.
- **1.1:** The "sweet spot." Reduced looping while maintaining coherence. Higher values (1.2+) caused the model to use unusual words to avoid common ones.

### 4. Zero-Shot Winning Instruction
MIPROv2 (via GPT-OSS-120B) identified this winning instruction for Depth 3:
> "Reply in the same tone as the last user message: brief, casual, and human‑like. Match their style, keep it short, and avoid any AI references or filler words."

## Implementation Changes
1.  Updated `jarvis/config.py` to set `context_depth=3` and `repetition_penalty=1.1`.
2.  Refactored `jarvis/prompts/constants.py` to use ChatML templates globally.
3.  Enabled `logit_bias` support in `models/loader.py` to allow negative constraints on AI-related tokens.

## Next Steps
- Implement active logit bias for tokens like `AI`, `language`, `model`, and `help`.
- Test if `top_p` or `top_k` shifts provide any further gains (current baseline: 0.9 / 40).
