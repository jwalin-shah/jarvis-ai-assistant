#!/usr/bin/env python3
"""Annotate round-4 GLiNER candidate goldset with a strict durable-fact policy.

This script labels `expected_candidates` for sampled iMessage rows by combining:
- message-level keep/discard scoring from the trained message gate model
- conservative linguistic rules for durable personal facts
- candidate span filtering and coarse fact-type mapping

Outputs:
- round-4 labeled JSON + CSV
- merged candidate-gold JSON (base + round4, deduped by message_id)
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.sparse import csr_matrix, hstack

from scripts.train_message_gate import MessageGateFeatures

ROUND4_INPUT = Path("training_data/gliner_goldset_round4/sampled_messages.json")
ROUND4_OUTPUT_JSON = Path("training_data/gliner_goldset_round4/candidate_gold_labeled.json")
ROUND4_OUTPUT_CSV = Path("training_data/gliner_goldset_round4/candidate_gold_labeled.csv")
BASE_MERGED = Path("training_data/gliner_goldset/candidate_gold_merged_r3.json")
OUTPUT_MERGED = Path("training_data/gliner_goldset/candidate_gold_merged_r4.json")
MESSAGE_GATE_PATH = Path("models/message_gate.pkl")

VAGUE_SPANS = {
    "it",
    "this",
    "that",
    "thing",
    "stuff",
    "them",
    "there",
    "here",
    "me",
    "you",
    "he",
    "she",
    "they",
    "we",
    "rn",
    "yk",
    "nah",
    "ya",
    "bro",
}

KINSHIP_TERMS = {
    "mom",
    "mother",
    "dad",
    "father",
    "brother",
    "sister",
    "cousin",
    "uncle",
    "aunt",
    "grandma",
    "grandpa",
    "parents",
    "mama",
    "papa",
    "wife",
    "husband",
    "girlfriend",
    "boyfriend",
    "partner",
    "friend",
}

ENTITY_MIN_SCORE = {
    "family_member": 0.50,
    "place": 0.55,
    "org": 0.55,
    "health_condition": 0.45,
    "activity": 0.45,
    "food_item": 0.45,
    "job_role": 0.50,
}

RE_SYSTEM = re.compile(
    r"(cvs pharmacy|prescription is ready|unsubscribe|check out this job|"
    r"apply now|fyi - up to \d+)",
    re.IGNORECASE,
)
RE_RECRUITING = re.compile(
    r"(recruiter|we['’]?re hiring|your resume|offer you|part[- ]?time|full[- ]?time|"
    r"interested in|event staff|pay:\s*\d+|hours:\s*\d+|online recruitment agencies)",
    re.IGNORECASE,
)
RE_AUTOMATED_NOTICE = re.compile(
    r"(call \d{3}-\d{3}-\d{4}|mon[- ]fri|download deputy|attached to this email|"
    r"this document needs to be completed|ready to schedule|text stop to stop)",
    re.IGNORECASE,
)
RE_ADVICE = re.compile(
    r"\b(say you['’]?ve|tell (him|her|them)|you should|u should)\b",
    re.IGNORECASE,
)
RE_TAPBACK = re.compile(
    r'^(liked|loved|laughed at|emphasized|questioned|disliked)\s+["“]',
    re.IGNORECASE,
)
RE_LOGISTICS = re.compile(
    r"\b(on my way|leave at|tonight|tomorrow|pickup|pick up|drop off|coming over|be there|"
    r"can you|where are you|at \d{1,2}(:\d{2})?\s*(am|pm)?)\b",
    re.IGNORECASE,
)
RE_SPORTS = re.compile(
    r"\b(nba|nfl|mlb|nhl|rams|chiefs|warriors|cowboys|lakers|mavs|playoffs|fantasy|"
    r"touchdown)\b",
    re.IGNORECASE,
)

RE_HEALTH = re.compile(
    r"\b(pain|injur\w*|dizz\w*|vertigo|hospital\w*|therapy|symptom\w*|headache|"
    r"depress\w*|anx\w*|allerg\w*|sick|doctor|docs)\b",
    re.IGNORECASE,
)
RE_WORK = re.compile(
    r"\b(work(ing)?|job|intern\w*|employer|company|startup|manager|engineer|analyst|"
    r"offer|full[- ]?time|part[- ]?time)\b",
    re.IGNORECASE,
)
RE_WORK_PERSONAL = re.compile(
    r"\b(i|i'm|im|he|she|they|my dad|my mom|my brother|my sister)\b.*\b("
    r"work|working|intern\w*|job|role|position|offer|manager|engineer|analyst"
    r")\b",
    re.IGNORECASE,
)
RE_EDU = re.compile(
    r"\b(undergrad|college|university|school|class|major|graduate|graduat\w*|course|"
    r"externship|acceptance letter)\b",
    re.IGNORECASE,
)
RE_LOCATION = re.compile(
    r"\b(live in|lived in|living in|from|based in|moving to|moved to|back to|in [A-Z][a-z]+)\b"
)
RE_PREF_PERSONAL = re.compile(
    r"\b(i|i'm|im|my dad|my mom|he|she|they)\s+"
    r"(really\s+)?(like|love|liked|hate|prefer|favorite|enjoy|into)\b",
    re.IGNORECASE,
)
RE_HOBBY = re.compile(
    r"\b(reading|read|ski\w*|yoga|climb\w*|guitar|cricket|gaming|run\w*|workout)\b",
    re.IGNORECASE,
)
RE_REL = re.compile(
    r"\b(my\s+)?(mom|mother|dad|father|brother|sister|cousin|uncle|aunt|parents|wife|"
    r"husband|partner|friend)\b",
    re.IGNORECASE,
)

RE_HEALTH_FALLBACK = re.compile(
    r"\b(vertigo|dizziness|injury|pain|depressed|anxiety|hospitalized|therapy|symptoms?)\b",
    re.IGNORECASE,
)
RE_LOC_FALLBACK = re.compile(r"\b(?:in|from|to|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
RE_PREF_FALLBACK = re.compile(
    r"\b(idli|dosa|falafel|paneer|chocolate|reading|yoga|guitar|skiing|cricket)\b",
    re.IGNORECASE,
)
RE_WORK_FALLBACK = re.compile(
    r"\b(data analyst|software engineer|product manager|intern|internship|teacher|"
    r"professor|job)\b",
    re.IGNORECASE,
)
RE_LOCATION_DECLARATIVE = re.compile(
    r"\b(i|i'm|im|he|she|they|my\s+(dad|mom|brother|sister|friend|partner))\s+"
    r"(live|lived|living|moved|moving|move|from|based)\b",
    re.IGNORECASE,
)

GENERIC_PLACE = {
    "home",
    "house",
    "apartment",
    "airport",
    "campus",
    "class",
    "work",
    "school",
    "office",
    "store",
    "temple",
    "place",
    "founders",
    "bart",
    "freezer isle",
}
GENERIC_ORG = {
    "company",
    "team",
    "startup",
    "district",
    "club",
    "event",
    "job",
    "authorities",
    "email",
    "document",
    "participants",
    "manager",
}
GENERIC_HEALTH = {"pandemic", "home tests", "medications"}
GENERIC_ACTIVITY = {"stuff", "thing", "things", "class"}

LOWERCASE_LOCATIONS = {
    "dallas",
    "austin",
    "houston",
    "sf",
    "nyc",
    "california",
    "texas",
    "fremont",
    "san jose",
    "new york",
    "menlo park",
    "delaware",
    "uk",
    "hawaii",
}
LOCATION_ABBREVIATIONS = {"sf", "nyc", "la", "sfo", "lax", "usa", "uk"}

COARSE_LABEL_PREF: dict[str, list[str]] = {
    "health": ["health_condition", "activity", "org"],
    "work": ["org", "job_role", "activity"],
    "education": ["org", "activity", "job_role", "place"],
    "location": ["place", "org"],
    "preference": ["food_item", "activity", "org", "place"],
    "hobby": ["activity", "food_item", "org", "place"],
    "relationship": ["family_member"],
    "other": ["job_role", "org", "health_condition", "place", "activity", "food_item"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROUND4_INPUT)
    parser.add_argument("--output-json", type=Path, default=ROUND4_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=ROUND4_OUTPUT_CSV)
    parser.add_argument("--message-gate", type=Path, default=MESSAGE_GATE_PATH)
    parser.add_argument("--base-merged", type=Path, default=BASE_MERGED)
    parser.add_argument("--merged-output", type=Path, default=OUTPUT_MERGED)
    parser.add_argument("--gate-threshold", type=float, default=0.72)
    parser.add_argument("--gate-threshold-strong", type=float, default=0.58)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_short_question(text: str) -> bool:
    t = text.strip()
    return t.endswith("?") and len(t.split()) <= 10


def is_question(text: str) -> bool:
    t = text.strip()
    return "?" in t and t.endswith("?")


def is_first_or_third_person_statement(text: str) -> bool:
    return bool(
        re.search(
            r"\b(i|i'm|im|my|he|she|they|my dad|my mom|my brother|my sister)\b",
            text,
            re.IGNORECASE,
        )
    )


def has_weak_object_only(text: str) -> bool:
    return bool(
        re.search(
            r"\b(i|im|i'm)\s+(like|love|hate)\s+(it|this|that)\b",
            text,
            re.IGNORECASE,
        )
    )


def coarse_type(text: str) -> str:
    if RE_HEALTH.search(text) and is_first_or_third_person_statement(text):
        return "health"
    if RE_WORK_PERSONAL.search(text):
        return "work"
    if RE_EDU.search(text) and is_first_or_third_person_statement(text):
        return "education"
    if RE_LOCATION_DECLARATIVE.search(text):
        return "location"
    if re.search(r"^\s*back in [A-Z][a-z]+", text):
        return "location"
    if RE_LOCATION.search(text) and re.search(
        r"\b(back in|from|moving to|moved to)\b", text, re.IGNORECASE
    ):
        return "location"
    if RE_PREF_PERSONAL.search(text) and not has_weak_object_only(text):
        if RE_HOBBY.search(text):
            return "hobby"
        return "preference"
    if RE_REL.search(text):
        return "relationship"
    return "other"


def is_strong_positive(text: str, coarse: str) -> bool:
    if coarse in {"health", "work", "education", "location"}:
        return True
    if (
        coarse in {"preference", "hobby"}
        and RE_PREF_FALLBACK.search(text)
        and not has_weak_object_only(text)
    ):
        return True
    if coarse == "relationship" and RE_REL.search(text) and not RE_LOGISTICS.search(text):
        return True
    return False


def is_strong_negative(text: str) -> bool:
    if RE_SYSTEM.search(text):
        return True
    if RE_RECRUITING.search(text):
        return True
    if RE_AUTOMATED_NOTICE.search(text):
        return True
    if RE_ADVICE.search(text):
        return True
    if RE_TAPBACK.search(text):
        return True
    if is_short_question(text):
        return True
    if RE_LOGISTICS.search(text) and not RE_HEALTH.search(text):
        return True
    if RE_SPORTS.search(text) and not RE_PREF_PERSONAL.search(text):
        return True
    if "http://" in text.lower() or "https://" in text.lower():
        return True
    return False


def bucket_for_gate(slice_name: str) -> str:
    if slice_name == "hard_negative":
        return "negative"
    return "likely"


def load_message_gate(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def gate_scores(rows: list[dict[str, Any]], gate_payload: dict[str, Any]) -> list[float]:
    model = gate_payload["model"]
    vectorizer = gate_payload["vectorizer"]
    scaler = gate_payload["scaler"]
    feature_extractor = MessageGateFeatures()

    texts = [str(r["message_text"]) for r in rows]
    is_from_me = [bool(r.get("is_from_me", False)) for r in rows]
    buckets = [bucket_for_gate(str(r.get("slice", ""))) for r in rows]

    x_text = vectorizer.transform(texts)
    x_num_arr = feature_extractor.transform(texts, is_from_me, buckets)
    x_num = csr_matrix(scaler.transform(x_num_arr))
    x = hstack([x_text, x_num], format="csr")

    if hasattr(model, "predict_proba"):
        return list(model.predict_proba(x)[:, 1])

    decision = model.decision_function(x)
    return list(1.0 / (1.0 + pow(2.718281828, -decision)))


def candidate_ok(candidate: dict[str, Any], text: str, coarse: str) -> bool:
    span = normalize_text(str(candidate.get("span_text", "")))
    label = str(candidate.get("span_label", ""))
    score = float(candidate.get("gliner_score", 0.0) or 0.0)
    span_l = span.lower()
    text_l = text.lower().replace("’", "'")

    if len(span) < 2:
        return False
    if span_l in VAGUE_SPANS:
        return False
    if score < ENTITY_MIN_SCORE.get(label, 0.5):
        return False
    if span_l not in text_l:
        return False

    allowed_by_coarse = {
        "health": {"health_condition"},
        "work": {"org", "job_role"},
        "education": {"org", "job_role"},
        "location": {"place"},
        "preference": {"food_item", "activity"},
        "hobby": {"activity", "food_item"},
        "relationship": {"family_member"},
        "other": {"org", "job_role", "health_condition"},
    }

    if label == "family_member":
        if coarse != "relationship":
            return False
        return any(term in span_l for term in KINSHIP_TERMS)

    if label not in allowed_by_coarse.get(coarse, set()):
        return False

    if label == "place":
        if span_l in GENERIC_PLACE:
            return False
        if span_l in LOWERCASE_LOCATIONS or span_l in LOCATION_ABBREVIATIONS:
            return True
        if span.isupper() and 2 <= len(span) <= 4:
            return True
        words = span.split()
        if len(words) >= 2 and all(w[:1].isupper() for w in words):
            return True
        if len(words) == 1 and words[0][:1].isupper():
            return False
        return False

    if label == "org":
        if span_l in {"nah", "yo", "bro", "bb"} or span_l in GENERIC_ORG:
            return False
        if " system" in span_l or span_l.endswith(" format"):
            return False
        if span.isupper() and len(span) > 4 and span_l not in {"utd", "ucd", "ucla"}:
            return False
        if len(span_l) <= 2:
            return False

    if label == "health_condition" and span_l in GENERIC_HEALTH:
        return False
    if label == "job_role" and span_l in {"job", "work"}:
        return False
    if label == "activity" and span_l in GENERIC_ACTIVITY:
        return False

    return True


def infer_fact_type(coarse: str, label: str, text: str, suggested: str) -> str:
    if suggested and suggested != "other_personal_fact":
        return suggested

    t = text.lower()
    if coarse == "health":
        if "allerg" in t and label in {"food_item", "health_condition"}:
            return "health.allergy"
        return "health.condition"
    if coarse == "work":
        if label == "job_role":
            return "work.job_title"
        if any(k in t for k in ["used to work", "quit", "left "]):
            return "work.former_employer"
        return "work.employer"
    if coarse == "education":
        return "personal.school"
    if coarse == "location":
        if any(k in t for k in ["moving", "moved to", "back to"]):
            return "location.future"
        if any(k in t for k in ["used to live", "lived in", "from "]):
            return "location.past"
        return "location.current"
    if coarse == "relationship":
        return "relationship.family"
    if coarse in {"preference", "hobby"}:
        if label == "food_item":
            if any(k in t for k in ["hate", "can't stand", "dislike"]):
                return "preference.food_dislike"
            return "preference.food_like"
        return "preference.activity"
    return "other_personal_fact"


def choose_candidates(row: dict[str, Any], keep: bool, coarse: str) -> list[dict[str, str]]:
    if not keep:
        return []

    text = str(row["message_text"])
    suggested = list(row.get("suggested_candidates") or [])

    valid = [c for c in suggested if candidate_ok(c, text, coarse)]
    valid.sort(key=lambda c: float(c.get("gliner_score", 0.0) or 0.0), reverse=True)

    preferred = COARSE_LABEL_PREF.get(coarse, COARSE_LABEL_PREF["other"])
    picked: list[dict[str, Any]] = []

    for label in preferred:
        label_items = [c for c in valid if str(c.get("span_label", "")) == label]
        if not label_items:
            continue
        picked = [label_items[0]]
        break

    if not picked and valid:
        picked = [valid[0]]

    out: list[dict[str, str]] = []
    for cand in picked:
        span = normalize_text(str(cand.get("span_text", "")))
        label = str(cand.get("span_label", ""))
        if not span:
            continue
        out.append(
            {
                "span_text": span,
                "span_label": label,
                "fact_type": infer_fact_type(
                    coarse=coarse,
                    label=label,
                    text=text,
                    suggested=str(cand.get("fact_type", "")),
                ),
            }
        )

    if out:
        return out

    if coarse == "health":
        match = RE_HEALTH_FALLBACK.search(text)
        if match:
            return [
                {
                    "span_text": normalize_text(match.group(0)),
                    "span_label": "health_condition",
                    "fact_type": "health.condition",
                }
            ]

    if coarse == "location":
        match = RE_LOC_FALLBACK.search(text)
        if match:
            candidate = normalize_text(match.group(1))
            cand_l = candidate.lower()
            words = candidate.split()
            is_known = cand_l in LOWERCASE_LOCATIONS or cand_l in LOCATION_ABBREVIATIONS
            is_multiword = len(words) >= 2 and all(w[:1].isupper() for w in words)
            is_short_upper = candidate.isupper() and 2 <= len(candidate) <= 4
            if is_known or is_multiword or is_short_upper:
                return [
                    {
                        "span_text": candidate,
                        "span_label": "place",
                        "fact_type": infer_fact_type("location", "place", text, ""),
                    }
                ]

    if coarse in {"preference", "hobby"}:
        match = RE_PREF_FALLBACK.search(text)
        if match:
            span = normalize_text(match.group(0))
            is_food = span.lower() in {"idli", "dosa", "falafel", "paneer", "chocolate"}
            label = "food_item" if is_food else "activity"
            return [
                {
                    "span_text": span,
                    "span_label": label,
                    "fact_type": infer_fact_type(coarse, label, text, ""),
                }
            ]

    if coarse in {"work", "education"}:
        match = RE_WORK_FALLBACK.search(text)
        if match:
            span = normalize_text(match.group(0))
            label = "job_role" if coarse == "work" else "activity"
            return [
                {
                    "span_text": span,
                    "span_label": label,
                    "fact_type": infer_fact_type(coarse, label, text, ""),
                }
            ]

    if coarse == "relationship":
        match = RE_REL.search(text)
        if match:
            return [
                {
                    "span_text": normalize_text(match.group(0)),
                    "span_label": "family_member",
                    "fact_type": "relationship.family",
                }
            ]

    return []


def classify_slice(source_slice: str, expected: list[dict[str, str]], keep: bool) -> str:
    if expected:
        return "positive"
    if source_slice == "hard_negative":
        return "hard_negative"
    if keep:
        return "near_miss"
    return "random_negative"


def merge_expected(
    base_expected: list[dict[str, Any]],
    new_expected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    merged = list(base_expected)
    seen = {
        (normalize_text(str(c.get("span_text", ""))).lower(), str(c.get("span_label", "")))
        for c in base_expected
    }
    added = 0
    for cand in new_expected:
        span = normalize_text(str(cand.get("span_text", "")))
        label = str(cand.get("span_label", ""))
        key = (span.lower(), label)
        if not span or key in seen:
            continue
        merged.append(
            {
                "span_text": span,
                "span_label": label,
                "fact_type": str(cand.get("fact_type", "other_personal_fact")),
            }
        )
        seen.add(key)
        added += 1
    return merged, added


def merge_with_base(
    base_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int, int]:
    merged = [dict(r) for r in base_rows]
    idx_by_mid = {int(r["message_id"]): i for i, r in enumerate(merged)}

    added_rows = 0
    overlap_rows = 0
    added_cands = 0

    for row in new_rows:
        mid = int(row["message_id"])
        if mid not in idx_by_mid:
            merged.append(dict(row))
            idx_by_mid[mid] = len(merged) - 1
            added_rows += 1
            continue

        overlap_rows += 1
        existing = merged[idx_by_mid[mid]]
        existing_expected = list(existing.get("expected_candidates") or [])
        new_expected = list(row.get("expected_candidates") or [])
        combined, added = merge_expected(existing_expected, new_expected)
        existing["expected_candidates"] = combined
        if combined:
            existing["slice"] = "positive"
        added_cands += added

    return merged, added_rows, overlap_rows, added_cands


def annotate_rows(
    rows: list[dict[str, Any]],
    gate_payload: dict[str, Any],
    threshold: float,
    threshold_strong: float,
) -> list[dict[str, Any]]:
    scores = gate_scores(rows, gate_payload)
    labeled: list[dict[str, Any]] = []

    for row, score in zip(rows, scores):
        text = str(row["message_text"])
        source_slice = str(row.get("slice", ""))
        coarse = coarse_type(text)
        strong_pos = is_strong_positive(text, coarse)
        strong_neg = is_strong_negative(text)
        question = is_question(text)

        if source_slice == "hard_negative":
            keep = False
        elif RE_SYSTEM.search(text) or RE_TAPBACK.search(text):
            keep = False
        elif RE_RECRUITING.search(text) or RE_AUTOMATED_NOTICE.search(text):
            keep = False
        elif RE_ADVICE.search(text):
            keep = False
        elif strong_neg and not strong_pos:
            keep = False
        elif question:
            keep = False
        elif coarse == "other":
            keep = False
        elif score >= threshold and not strong_neg:
            keep = True
        elif score >= threshold_strong and strong_pos:
            keep = True
        elif strong_pos and coarse in {"health", "work", "education"} and score >= 0.50:
            keep = True
        elif coarse == "relationship" and score >= 0.80 and not RE_LOGISTICS.search(text):
            keep = True
        else:
            keep = False

        expected = choose_candidates(row=row, keep=keep, coarse=coarse)

        out = dict(row)
        out["source_slice"] = source_slice
        out["gate_score"] = round(float(score), 4)
        out["auto_coarse_type"] = coarse
        out["gold_keep"] = "1" if keep else "0"
        out["slice"] = classify_slice(source_slice, expected, keep)
        out["expected_candidates"] = expected
        labeled.append(out)

    return labeled


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "slice",
        "source_slice",
        "message_id",
        "chat_rowid",
        "chat_id",
        "chat_display_name",
        "is_from_me",
        "sender_handle",
        "message_date",
        "message_text",
        "context_prev",
        "context_next",
        "gate_score",
        "auto_coarse_type",
        "gold_keep",
        "suggested_candidates_json",
        "expected_candidates_json",
        "gold_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.get("sample_id", ""),
                    "slice": row.get("slice", ""),
                    "source_slice": row.get("source_slice", ""),
                    "message_id": row.get("message_id", ""),
                    "chat_rowid": row.get("chat_rowid", ""),
                    "chat_id": row.get("chat_id", ""),
                    "chat_display_name": row.get("chat_display_name", ""),
                    "is_from_me": row.get("is_from_me", ""),
                    "sender_handle": row.get("sender_handle", ""),
                    "message_date": row.get("message_date", ""),
                    "message_text": row.get("message_text", ""),
                    "context_prev": row.get("context_prev", ""),
                    "context_next": row.get("context_next", ""),
                    "gate_score": row.get("gate_score", ""),
                    "auto_coarse_type": row.get("auto_coarse_type", ""),
                    "gold_keep": row.get("gold_keep", ""),
                    "suggested_candidates_json": json.dumps(
                        row.get("suggested_candidates") or [],
                        ensure_ascii=True,
                    ),
                    "expected_candidates_json": json.dumps(
                        row.get("expected_candidates") or [],
                        ensure_ascii=True,
                    ),
                    "gold_notes": row.get("gold_notes", ""),
                }
            )


def summarize(rows: list[dict[str, Any]], title: str) -> None:
    slice_counts = Counter(str(r.get("slice", "")) for r in rows)
    keeps = sum(1 for r in rows if str(r.get("gold_keep", "")) == "1")
    pos = sum(1 for r in rows if r.get("expected_candidates"))
    total_cands = sum(len(r.get("expected_candidates") or []) for r in rows)
    by_label = Counter()
    by_type = Counter()

    for r in rows:
        for c in r.get("expected_candidates") or []:
            by_label[str(c.get("span_label", ""))] += 1
            by_type[str(c.get("fact_type", ""))] += 1

    print(f"{title}:")
    print(f"  records: {len(rows)}")
    print(f"  keep=1: {keeps}")
    print(f"  positives (with candidates): {pos}")
    print(f"  total expected candidates: {total_cands}")
    print(f"  slices: {dict(slice_counts)}")
    print(f"  top labels: {dict(by_label.most_common(8))}")
    print(f"  top fact types: {dict(by_type.most_common(8))}")


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    gate_payload = load_message_gate(args.message_gate)
    labeled = annotate_rows(
        rows=rows,
        gate_payload=gate_payload,
        threshold=args.gate_threshold,
        threshold_strong=args.gate_threshold_strong,
    )

    write_json(args.output_json, labeled)
    write_csv(args.output_csv, labeled)
    summarize(labeled, "Round-4 labeled")
    print(f"  wrote json: {args.output_json}")
    print(f"  wrote csv:  {args.output_csv}")

    with args.base_merged.open("r", encoding="utf-8") as f:
        base_rows = json.load(f)
    merged, added_rows, overlap_rows, added_cands = merge_with_base(base_rows, labeled)
    write_json(args.merged_output, merged)
    summarize(merged, "Merged candidate gold (r4)")
    print(f"  wrote merged json: {args.merged_output}")
    print(f"  merged added rows: {added_rows}")
    print(f"  merged overlap rows: {overlap_rows}")
    print(f"  merged added candidate spans on overlaps: {added_cands}")


if __name__ == "__main__":
    main()
