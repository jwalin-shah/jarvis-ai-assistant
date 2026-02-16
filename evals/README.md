# Prompt Evaluation

This directory contains the promptfoo setup for systematically evaluating and improving JARVIS text message reply prompts.

## Quick Start

```bash
# Install promptfoo (one-time)
make eval-setup

# Run evaluation
make eval

# View results in browser
make eval-view
```

## Structure

```
evals/
├── promptfoo.yaml          # Main config (test cases, assertions)
├── jarvis_provider.py      # Provider script (calls JARVIS MLX model)
├── prompts/                # Prompt variations to test
│   ├── reply_v1_baseline.txt
│   ├── reply_v2_concise.txt
│   └── reply_v3_style_match.txt
├── results/                # Evaluation results (gitignored)
└── README.md
```

## How It Works

1. **Prompts**: Each file in `prompts/` is a prompt template with variables like `{{context}}`, `{{last_message}}`, etc.

2. **Provider**: `jarvis_provider.py` receives prompts from promptfoo and generates replies using the JARVIS MLX model.

3. **Assertions**: Test cases define expected behavior:
   - `llm-rubric`: LLM judges if response meets criteria
   - `javascript`: Custom checks (e.g., length limits)
   - `not-contains`: Ensures certain phrases are avoided

4. **Results**: JSON output in `results/` with pass/fail for each test case.

## Adding Test Cases

Edit `promptfoo.yaml` to add new test cases:

```yaml
tests:
  - description: 'My new test'
    vars:
      context: "[10:00] Friend: Hey what's up?"
      last_message: "Hey what's up?"
      tone: 'casual'
      user_style: 'brief, uses abbreviations'
    assert:
      - type: llm-rubric
        value: 'Response should be casual and brief'
```

## Adding Prompt Variations

1. Create a new file in `prompts/` (e.g., `reply_v4_minimal.txt`)
2. Use template variables: `{{context}}`, `{{last_message}}`, `{{tone}}`, `{{user_style}}`
3. Add it to `promptfoo.yaml` under `prompts:`

## LLM Judge Configuration

By default, promptfoo uses OpenAI for `llm-rubric` evaluations. To use a different provider:

```yaml
# In promptfoo.yaml
defaultTest:
  options:
    provider:
      id: openai:gpt-4o-mini # or anthropic:claude-3-haiku
```

Set your API key:

```bash
export OPENAI_API_KEY=<your_api_key>
# or
export ANTHROPIC_API_KEY=<your_api_key>
```

## Tips

- **Start small**: Begin with 5-10 test cases covering common scenarios
- **Diverse inputs**: Include casual, professional, edge cases
- **Iterate**: After each eval, identify patterns in failures and adjust prompts
- **Version prompts**: Keep old versions to compare improvements

## Metrics to Track

- **Pass rate**: % of test cases passing all assertions
- **Brevity**: Average response length (shorter is usually better for texts)
- **Style match**: How well responses match user_style
- **AI-ness**: Frequency of AI-like phrases ("I'd be happy to", "That sounds")
