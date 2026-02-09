#!/bin/bash
# Label eval dataset chunks using Gemini CLI (stdin→stdout piping).
#
# Usage:
#   1. uv run python scripts/label_eval_gemini.py prepare
#   2. bash scripts/label_eval_gemini.sh
#   3. uv run python scripts/label_eval_gemini.py merge
set -euo pipefail

CHUNKS_DIR="evals/data/chunks"
PROMPT='You are a text message classifier. For each input message, assign category and mobilization.

**category** (exactly one):
- "acknowledge" — reactions, tapbacks, short confirmations (ok, sure, yes, got it, thumbs up, Liked/Loved/Emphasized messages)
- "closing" — conversation endings (bye, goodnight, talk later, see you)
- "emotion" — expressing feelings without asking/requesting (I miss you, that sucks, so happy, lol, haha)
- "question" — asking for information or confirmation (where are you?, did you eat?, what time?)
- "request" — asking someone to DO something (call me, pick up milk, send the file, can you help?)
- "statement" — sharing information, updates, opinions (I am at home, the movie was good, just landed)

**mobilization** (exactly one):
- "NONE" — no response needed (reactions, acknowledgments, closings)
- "LOW" — optional/casual response (statements, emotions, general chat)
- "MEDIUM" — should respond (direct questions, soft requests)
- "HIGH" — must respond urgently (urgent requests, time-sensitive questions, someone waiting on you)

Rules:
- "Liked/Loved/Emphasized/Laughed at/Disliked" messages are ALWAYS category=acknowledge, mobilization=NONE
- Short reactions (lol, haha, ok, sure, yep, nice) are acknowledge + NONE
- If thread context is provided, use it to understand the message better
- When ambiguous, prefer the simpler category

Input: One JSON per line with "id", "text", optionally "thread"
Output: One JSON per line with "id", "category", "mobilization"

Output ONLY valid JSON lines. No explanations, no markdown, no extra text.'

if [ ! -d "$CHUNKS_DIR" ]; then
    echo "ERROR: No chunks found. Run: uv run python scripts/label_eval_gemini.py prepare"
    exit 1
fi

INPUT_FILES=()
for f in "$CHUNKS_DIR"/chunk_*.txt; do
    [[ "$f" == *_output.txt ]] && continue
    INPUT_FILES+=("$f")
done

echo "Found ${#INPUT_FILES[@]} chunk files to process"
echo ""

completed=0
skipped=0
failed=0

for chunk_file in "${INPUT_FILES[@]}"; do
    base=$(basename "$chunk_file" .txt)
    output_file="$CHUNKS_DIR/${base}_output.txt"

    expected=$(wc -l < "$chunk_file" | tr -d ' ')

    if [ -f "$output_file" ]; then
        existing=$(wc -l < "$output_file" | tr -d ' ')
        if [ "$existing" -ge "$expected" ]; then
            echo "SKIP $base: $existing/$expected labels"
            skipped=$((skipped + 1))
            continue
        fi
        echo "REDO $base: only $existing/$expected labels"
    fi

    echo "Processing $base ($expected examples)..."

    if gemini -p "$PROMPT" < "$chunk_file" > "$output_file" 2>/dev/null; then
        result_count=$(wc -l < "$output_file" | tr -d ' ')
        if [ "$result_count" -ge "$expected" ]; then
            echo "  OK: $result_count/$expected labels"
            completed=$((completed + 1))
        else
            echo "  PARTIAL: $result_count/$expected labels (may need re-run)"
            completed=$((completed + 1))
        fi
    else
        echo "  ERROR: gemini failed on $base"
        rm -f "$output_file"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Summary: $completed completed, $skipped skipped, $failed failed"
echo "Run: uv run python scripts/label_eval_gemini.py merge"
