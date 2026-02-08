#!/usr/bin/env python3
"""
Validate LinearSVC category classifier against Groq LLM on real iMessage data.

Compares LinearSVC (fast, local) vs Groq Llama 3.3 70B (API) on 100 unseen messages.
Reports agreement rate and disagreement patterns.
"""
import json
import os
import random
import re
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import spacy
from groq import Groq
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.embedding_adapter import get_embedder

# Regex patterns for hand-crafted features (synced with production)
EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
)

PROFESSIONAL_KEYWORDS_RE = re.compile(
    r"\b(meeting|deadline|project|report|schedule|conference|presentation|"
    r"budget|client|invoice|proposal)\b",
    re.IGNORECASE,
)

ABBREVIATION_RE = re.compile(
    r"\b(lol|lmao|omg|wtf|brb|btw|smh|tbh|imo|idk|ngl|fr|rn|ong|nvm|wya|hmu|"
    r"fyi|asap|dm|irl|fomo|goat|sus|bet|cap|no cap)\b",
    re.IGNORECASE,
)


def extract_spacy_features(text: str, nlp) -> np.ndarray:
    """Extract 14 SpaCy linguistic features."""
    doc = nlp(text)
    features = []

    # 1. has_imperative
    has_imperative = 0.0
    if len(doc) > 0 and doc[0].pos_ == "VERB" and doc[0].tag_ == "VB":
        has_imperative = 1.0
    features.append(has_imperative)

    # 2. you_modal
    text_lower = text.lower()
    you_modal = 1.0 if any(p in text_lower for p in ["can you", "could you", "would you", "will you", "should you"]) else 0.0
    features.append(you_modal)

    # 3. request_verb
    request_verbs = {"send", "give", "help", "tell", "show", "let", "call", "get", "make", "take"}
    has_request = 1.0 if any(token.lemma_ in request_verbs for token in doc) else 0.0
    features.append(has_request)

    # 4. starts_modal
    starts_modal = 0.0
    if len(doc) > 0 and doc[0].tag_ in ("MD", "VB"):
        starts_modal = 1.0
    features.append(starts_modal)

    # 5. directive_question
    directive_q = 1.0 if you_modal and "?" in text else 0.0
    features.append(directive_q)

    # 6. i_will
    i_will = 1.0 if any(p in text_lower for p in ["i'll", "i will", "i'm gonna", "ima", "imma"]) else 0.0
    features.append(i_will)

    # 7. promise_verb
    promise_verbs = {"promise", "guarantee", "commit", "swear"}
    has_promise = 1.0 if any(token.lemma_ in promise_verbs for token in doc) else 0.0
    features.append(has_promise)

    # 8. first_person_count
    first_person = sum(1 for token in doc if token.text.lower() in ("i", "me", "my", "mine", "myself"))
    features.append(float(first_person))

    # 9. agreement
    agreement_words = {"sure", "okay", "ok", "yes", "yeah", "yep", "yup", "sounds good", "bet", "fs"}
    has_agreement = 1.0 if any(word in text_lower for word in agreement_words) else 0.0
    features.append(has_agreement)

    # 10. modal_count
    modal_count = sum(1 for token in doc if token.tag_ == "MD")
    features.append(float(modal_count))

    # 11. verb_count
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    features.append(float(verb_count))

    # 12. second_person_count
    second_person = sum(1 for token in doc if token.text.lower() in ("you", "your", "yours", "yourself"))
    features.append(float(second_person))

    # 13. has_negation
    has_neg = 1.0 if any(token.dep_ == "neg" for token in doc) else 0.0
    features.append(has_neg)

    # 14. is_interrogative
    is_question = 1.0 if "?" in text or any(token.tag_ in ("WDT", "WP", "WP$", "WRB") for token in doc) else 0.0
    features.append(is_question)

    return np.array(features, dtype=np.float32)


def extract_hand_crafted_features(text: str, context: list[str]) -> np.ndarray:
    """Extract 26 hand-crafted features (enhanced with reaction/emotion detection)."""
    features: list[float] = []
    text_lower = text.lower()
    words = text.split()
    total_words = len(words)

    # Message structure (5)
    features.append(float(len(text)))
    features.append(float(total_words))
    features.append(float(text.count("?")))
    features.append(float(text.count("!")))
    features.append(float(len(EMOJI_RE.findall(text))))

    # Mobilization one-hots (7) - default to "none" and "answer" for validation
    for level in ("high", "medium", "low", "none"):
        features.append(1.0 if level == "none" else 0.0)
    for rtype in ("commitment", "answer", "emotional"):
        features.append(1.0 if rtype == "answer" else 0.0)

    # Tone flags (2)
    features.append(1.0 if PROFESSIONAL_KEYWORDS_RE.search(text) else 0.0)
    features.append(1.0 if ABBREVIATION_RE.search(text) else 0.0)

    # Context features (3)
    features.append(float(len(context)))
    avg_ctx_len = float(np.mean([len(m) for m in context])) if context else 0.0
    features.append(avg_ctx_len)
    features.append(1.0 if len(context) == 0 else 0.0)

    # Style features (2)
    abbr_count = len(ABBREVIATION_RE.findall(text))
    features.append(abbr_count / max(total_words, 1))
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    features.append(capitalized / max(len(words) - 1, 1))

    # NEW: Reaction/emotion features (7)
    # 1. Is this an iMessage reaction/tapback?
    reaction_patterns = ["Laughed at", "Loved", "Liked", "Disliked", "Emphasized", "Questioned"]
    is_reaction = 1.0 if any(text.startswith(p) for p in reaction_patterns) else 0.0
    features.append(is_reaction)

    # 2. Emotional marker count (lmao, lol, xd, haha, bruh, rip, omg)
    emotional_markers = ["lmao", "lol", "xd", "haha", "omg", "bruh", "rip", "lmfao", "rofl"]
    emotional_count = sum(text_lower.count(marker) for marker in emotional_markers)
    features.append(float(emotional_count))

    # 3. Does message END with emotional marker?
    last_word = words[-1].lower() if words else ""
    ends_with_emotion = 1.0 if last_word in emotional_markers else 0.0
    features.append(ends_with_emotion)

    # 4. Question word at start
    question_starters = {"what", "why", "how", "when", "where", "who", "did", "do", "does", "can", "could", "would", "will", "should"}
    first_word = words[0].lower() if words else ""
    question_first = 1.0 if first_word in question_starters else 0.0
    features.append(question_first)

    # 5. Imperative verb at start
    imperative_verbs = {"make", "send", "get", "tell", "show", "give", "come", "take", "call", "help", "let"}
    imperative_first = 1.0 if first_word in imperative_verbs else 0.0
    features.append(imperative_first)

    # 6. Brief agreement phrase
    brief_agreements = {"ok", "okay", "k", "yeah", "yep", "yup", "sure", "cool", "bet", "fs", "aight"}
    is_brief_agreement = 1.0 if total_words <= 3 and any(w in brief_agreements for w in words) else 0.0
    features.append(is_brief_agreement)

    # 7. Exclamatory ending
    exclamatory = 1.0 if (text.endswith("!") or text.isupper() and total_words <= 5) else 0.0
    features.append(exclamatory)

    return np.array(features, dtype=np.float32)


# Load category labeling prompt (same as training)
CATEGORY_SYSTEM_PROMPT = """You are a category classifier for text messages. Classify each message into exactly ONE of these 6 categories:

**Categories:**
1. **closing**: Conversation enders (bye, talk later, goodnight, see you, take care)
2. **acknowledge**: Simple acknowledgments with no new info (ok, got it, thanks, sounds good, cool, yeah, sure, makes sense, np, understood, will do, right, yep, agreed, noted, perfect)
3. **question**: Requests for information (uses ?, asks for details, seeks clarification)
4. **request**: Action requests or commands (imperative verbs, "can you", "please", direct asks for someone to DO something)
5. **emotion**: Emotional reactions or greetings (hi, hey, congrats, haha, lol, love it, sorry, wow, omg, excited)
6. **statement**: Informational statements, updates, or observations (declarative sentences providing info, reporting status)

**Rules:**
- If message has both question mark AND action request → **request** (e.g., "Can you send that?")
- Formulaic greetings like "Hey" or "Hi there" → **emotion**
- Thanks/acknowledgment with no info → **acknowledge**
- Thanks + new info → **statement**
- "Yes/no" answers with context → **statement**
- "Yes/no" alone → **acknowledge**

**Output format:** Return ONLY the category name (lowercase, no extra text)."""


def load_real_messages(db_path: str, sample_size: int = 100, exclude_texts: set = None) -> list[dict]:
    """Load random real messages from chat.db that are NOT in training set."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Get messages with context (last 3 messages in thread)
    query = """
    WITH ranked_messages AS (
        SELECT
            m.ROWID,
            m.text,
            m.is_from_me,
            m.cache_roomnames as thread_id,
            m.date,
            ROW_NUMBER() OVER (
                PARTITION BY m.cache_roomnames
                ORDER BY m.date DESC
            ) as msg_rank
        FROM message m
        WHERE m.text IS NOT NULL
        AND LENGTH(m.text) > 0
        AND m.text NOT LIKE 'Loved%'
        AND m.text NOT LIKE 'Liked%'
        AND m.text NOT LIKE 'Emphasized%'
    )
    SELECT DISTINCT
        curr.text,
        curr.thread_id,
        curr.date
    FROM ranked_messages curr
    WHERE curr.msg_rank <= 1000  -- Recent messages only
    ORDER BY RANDOM()
    LIMIT ?
    """

    cursor.execute(query, (sample_size * 3,))  # Get 3x to account for filtering
    messages = []

    for row in cursor.fetchall():
        text, thread_id, date = row
        if exclude_texts and text in exclude_texts:
            continue  # Skip messages in training set

        # Get context (previous 3 messages in thread)
        context_query = """
        SELECT text, is_from_me
        FROM message
        WHERE cache_roomnames = ?
        AND date < ?
        AND text IS NOT NULL
        ORDER BY date DESC
        LIMIT 3
        """
        cursor.execute(context_query, (thread_id, date))
        context = [row[0] for row in cursor.fetchall()]
        context.reverse()  # Chronological order

        messages.append({
            "text": text,
            "context": context,
        })

        if len(messages) >= sample_size:
            break

    conn.close()
    return messages


def get_svm_predictions(messages: list[dict], model_path: str, embedder, nlp) -> list[str]:
    """Get LinearSVC predictions for messages."""
    model = joblib.load(model_path)

    # Extract features (same as training)
    texts = [m["text"] for m in messages]
    contexts = [m["context"] for m in messages]

    # BERT embeddings
    print("Extracting BERT embeddings...", flush=True)
    bert_embeds = []
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT"):
        batch = texts[i:i+batch_size]
        embeds = embedder.encode(batch)
        bert_embeds.append(embeds)
    bert_embeds = np.vstack(bert_embeds)

    # Hand-crafted features
    print("Extracting hand-crafted features...", flush=True)
    hand_features = []
    for text, ctx in tqdm(zip(texts, contexts), total=len(texts), desc="Hand-crafted"):
        feats = extract_hand_crafted_features(text, ctx)
        hand_features.append(feats)
    hand_features = np.array(hand_features)

    # SpaCy features
    print("Extracting spaCy features...", flush=True)
    spacy_features = []
    for text in tqdm(texts, desc="SpaCy"):
        feats = extract_spacy_features(text, nlp)
        spacy_features.append(feats)
    spacy_features = np.array(spacy_features)

    # Combine features
    X = np.hstack([bert_embeds, hand_features, spacy_features])

    # Predict
    predictions = model.predict(X)
    return predictions.tolist()


def get_llm_predictions(messages: list[dict], api_key: str) -> list[str]:
    """Get Groq LLM predictions for messages."""
    client = Groq(api_key=api_key)
    predictions = []

    for msg in tqdm(messages, desc="Groq LLM"):
        # Build prompt with context
        context_str = ""
        if msg["context"]:
            context_str = "**Context (previous messages):**\n"
            for i, ctx_msg in enumerate(msg["context"], 1):
                context_str += f"{i}. {ctx_msg}\n"
            context_str += "\n"

        prompt = f"{context_str}**Message to classify:**\n{msg['text']}"

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": CATEGORY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=10,
            )

            prediction = response.choices[0].message.content.strip().lower()
            predictions.append(prediction)
            time.sleep(0.06)  # Rate limit: 1000 RPM = 16.6/sec

        except Exception as e:
            print(f"Error getting LLM prediction: {e}", flush=True)
            predictions.append("error")

    return predictions


def main():
    print("="*70, flush=True)
    print("VALIDATION: LinearSVC vs Groq LLM on Real Messages", flush=True)
    print("="*70, flush=True)

    # Paths
    db_path = Path.home() / "Library/Messages/chat.db"
    model_path = Path("models/category_svm_v2.joblib")
    training_data_path = Path("llm_category_labels.jsonl")

    # Load training texts to exclude
    print("\nLoading training data to exclude...", flush=True)
    exclude_texts = set()
    with open(training_data_path) as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                exclude_texts.add(example["text"])
    print(f"Excluding {len(exclude_texts)} training examples", flush=True)

    # Load real messages
    print("\nLoading 100 real messages from chat.db...", flush=True)
    messages = load_real_messages(str(db_path), sample_size=100, exclude_texts=exclude_texts)
    print(f"Loaded {len(messages)} messages", flush=True)

    # Get LinearSVC predictions
    print("\n" + "="*70, flush=True)
    print("LINEAR SVC PREDICTIONS", flush=True)
    print("="*70, flush=True)
    embedder = get_embedder()
    nlp = spacy.load("en_core_web_sm")
    svm_predictions = get_svm_predictions(messages, str(model_path), embedder, nlp)

    # Get Groq LLM predictions
    print("\n" + "="*70, flush=True)
    print("GROQ LLM PREDICTIONS", flush=True)
    print("="*70, flush=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set", flush=True)
        return
    llm_predictions = get_llm_predictions(messages, api_key)

    # Compare results
    print("\n" + "="*70, flush=True)
    print("COMPARISON RESULTS", flush=True)
    print("="*70, flush=True)

    agreements = sum(1 for s, l in zip(svm_predictions, llm_predictions) if s == l)
    agreement_rate = agreements / len(messages) * 100

    print(f"\nAgreement rate: {agreements}/{len(messages)} ({agreement_rate:.1f}%)", flush=True)

    # Disagreements by category
    from collections import Counter, defaultdict

    disagreements = defaultdict(list)
    for i, (svm, llm) in enumerate(zip(svm_predictions, llm_predictions)):
        if svm != llm:
            disagreements[f"{llm} → {svm}"].append(messages[i]["text"][:80])

    if disagreements:
        print("\nDisagreement patterns (LLM → SVM):", flush=True)
        for pattern, examples in sorted(disagreements.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {pattern}: {len(examples)} cases", flush=True)
            for ex in examples[:2]:  # Show 2 examples
                print(f"    - \"{ex}...\"", flush=True)

    # Save results
    output_path = Path("validation_results.jsonl")
    with open(output_path, "w") as f:
        for msg, svm, llm in zip(messages, svm_predictions, llm_predictions):
            result = {
                "text": msg["text"],
                "context": msg["context"],
                "svm_prediction": svm,
                "llm_prediction": llm,
                "agree": svm == llm,
            }
            f.write(json.dumps(result) + "\n")

    print(f"\nFull results saved to {output_path}", flush=True)
    print(f"\n✓ Validation complete! Agreement: {agreement_rate:.1f}%", flush=True)


if __name__ == "__main__":
    main()
