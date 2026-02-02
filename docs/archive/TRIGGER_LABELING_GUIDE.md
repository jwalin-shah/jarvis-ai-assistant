# Trigger Labeling Guide

Label incoming message triggers for the JARVIS classifier.

## Labels (5 types)

### COMMITMENT - Invitations & Requests
Asking someone to do something or join an activity.
**Examples:** "Want to grab lunch?", "Can you pick me up?", "Are you free tonight?"
**Signals:** "wanna", "want to", "can you", "let's", "down to"

### QUESTION - Information seeking
Yes/no questions or info questions that need an answer.
**Examples:** "What time?", "Did you finish?", "Where are we going?", "Is it raining?"
**Signals:** Usually ends with "?", WH-words (what/when/where/who/why/how)
**Not:** "How are you?" → SOCIAL (greeting, doesn't need real answer)

### REACTION - Emotional content
Sharing news or prompting an emotional response.
**Examples:** "I got the job!", "That's crazy!", "Damn bro", "I'm so stressed"
**Signals:** Emotional words (damn, crazy, omg, fuck, awesome, sad, happy)
**Not:** Tapbacks → SOCIAL

### SOCIAL - Greetings, Acks & Tapbacks
Social lubricant - greetings, acknowledgments, tapbacks (Liked/Loved/Laughed at).
**Examples:** "hey", "ok", "thanks", "lol", "Loved 'message'", "sure", "bet"
**Signals:** Tapbacks, greeting words, short acks
**Length:** Usually 1-5 words

### STATEMENT - Neutral information (default)
Sharing information without specific response expectation.
**Examples:** "I'm on my way", "Meeting moved to 3pm", "Just finished work"
**Use when:** No clear signal for other categories

## Quick Reference

| Label | Ends with ? | Key signals |
|-------|-------------|-------------|
| COMMITMENT | ~9% | "wanna", "can you", "let's", invitations |
| QUESTION | ~37% | WH-words, yes/no starters (do you, is it) |
| REACTION | ~0% | Emotional words, short exclamations |
| SOCIAL | ~1% | Tapbacks (32%), greetings, acks |
| STATEMENT | ~0% | "I" statements, neutral info |

## Priority Rules

1. **Tapbacks:** Always SOCIAL
   - "Loved 'message'" → SOCIAL
   - "Laughed at 'joke'" → SOCIAL

2. **Greeting + Question:** Label by question type
   - "hey are you free?" → COMMITMENT
   - "hi what's up?" → SOCIAL (not seeking real info)

3. **Multi-intent:** Pick PRIMARY intent
   - "Want to hang out and can you pick me up?" → COMMITMENT

4. **When in doubt:** Use STATEMENT

## Data Location

- Labeled data: `data/trigger_labeling.jsonl`
- 3,000 labeled examples
- Train with: `uv run python -m scripts.train_trigger_classifier --save-best`
- Model: `~/.jarvis/trigger_classifier_model/`

## Current Accuracy

| Label | F1 Score | Notes |
|-------|----------|-------|
| SOCIAL | 77% | Tapbacks are reliable |
| QUESTION | 74% | "?" and WH-words help |
| REACTION | 69% | Emotional words help |
| STATEMENT | 69% | Fallback category |
| COMMITMENT | 68% | Hardest - needs more examples |
| **Overall** | **71%** | Macro F1 |
