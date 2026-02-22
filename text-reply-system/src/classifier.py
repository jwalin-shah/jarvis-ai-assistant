from __future__ import annotations

import os
import random
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from src.config import AppConfig


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    used_majority_vote: bool


class ResponseClassifier:
    """Zero-shot response classifier with constrained category scoring."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.categories = config.categories
        self.threshold = config.runtime.classifier_confidence_threshold
        self._model = None
        self._tokenizer = None
        self._using_fallback = True
        self._try_load_model()

    def _try_load_model(self) -> None:
        if os.getenv("TEXT_REPLY_SYSTEM_SKIP_MODEL_LOAD") == "1":
            self._model = None
            self._tokenizer = None
            self._using_fallback = True
            return
        from mlx_lm import load  # type: ignore

        self._model = None
        self._tokenizer = None
        self._using_fallback = True
        for model_id in self.config.model_candidates("classifier_base"):
            try:
                self._model, self._tokenizer = load(model_id)
                self._using_fallback = False
                return
            except Exception:
                continue

    def classify(self, incoming_message: str) -> ClassificationResult:
        if self._using_fallback:
            return self._heuristic_classify(incoming_message)

        probs = self._category_probabilities(incoming_message)
        if not probs:
            return self._heuristic_classify(incoming_message)

        best_cat, best_prob = max(probs.items(), key=lambda kv: kv[1])
        if best_prob >= self.threshold:
            return ClassificationResult(category=best_cat, confidence=float(best_prob), used_majority_vote=False)

        # Ambiguity handling: sample classifications and majority vote.
        n = self.config.runtime.classifier_vote_samples
        temp = max(1e-6, self.config.runtime.classifier_vote_temperature)
        cats = list(probs.keys())
        p = np.array([probs[c] for c in cats], dtype=np.float64)
        p = np.power(np.clip(p, 1e-9, 1.0), 1.0 / temp)
        p = p / np.sum(p)

        votes: list[str] = [random.choices(cats, weights=p.tolist(), k=1)[0] for _ in range(n)]
        winner = max(set(votes), key=votes.count)
        count = votes.count(winner)
        if count > n // 2:
            return ClassificationResult(
                category=winner,
                confidence=float(probs.get(winner, best_prob)),
                used_majority_vote=True,
            )
        return ClassificationResult(category="tricky", confidence=float(best_prob), used_majority_vote=True)

    def _category_probabilities(self, incoming_message: str) -> dict[str, float]:
        if self._model is None or self._tokenizer is None:
            return {}

        prompt = (
            "<|im_start|>system\n"
            "Classify this text message into exactly one category.\n"
            "Categories: casual, question, logistics, emotional, invitation, decline,\n"
            "flirty, confrontational, hype, informational, tricky<|im_end|>\n"
            "<|im_start|>user\n"
            f"{incoming_message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        try:
            prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            if not prompt_ids:
                return {}

            # Preferred path: constrain next token to category tokens only.
            token_to_category: dict[int, str] = {}
            unresolved: list[str] = []
            for category in self.categories:
                cids = self._tokenizer.encode(f" {category}", add_special_tokens=False)
                if len(cids) == 1:
                    token_to_category[cids[0]] = category
                else:
                    unresolved.append(category)

            x = mx.array([prompt_ids], dtype=mx.int32)
            logits = np.array(self._model(x)[0, -1], dtype=np.float64)
            if token_to_category:
                allowed = list(token_to_category.keys())
                vals = np.array([logits[tid] for tid in allowed], dtype=np.float64)
                vals = vals - np.max(vals)
                p = np.exp(vals)
                p = p / np.sum(p)
                probs = {token_to_category[tid]: float(pi) for tid, pi in zip(allowed, p)}
            else:
                probs = {}

            # Fallback for categories without single-token forms.
            if unresolved:
                seq_scores = {c: self._sequence_logprob(prompt_ids, f" {c}") for c in unresolved}
                arr = np.array([seq_scores[c] for c in unresolved], dtype=np.float64)
                arr = arr - np.max(arr)
                p2 = np.exp(arr)
                p2 = p2 / np.sum(p2)
                for c, pi in zip(unresolved, p2):
                    probs[c] = float(pi)

            # Normalize across all categories and ensure all keys exist.
            for c in self.categories:
                probs.setdefault(c, 1e-9)
            arr = np.array([probs[c] for c in self.categories], dtype=np.float64)
            arr = np.clip(arr, 1e-12, None)
            arr = arr / np.sum(arr)
            return {c: float(pi) for c, pi in zip(self.categories, arr)}
        except Exception:
            return {}

    def _sequence_logprob(self, prompt_ids: list[int], continuation: str) -> float:
        if self._tokenizer is None or self._model is None:
            return -1e9
        cont_ids = self._tokenizer.encode(continuation, add_special_tokens=False)
        if not cont_ids:
            return -1e9

        full_ids = prompt_ids + cont_ids
        x = mx.array([full_ids], dtype=mx.int32)
        logits = self._model(x)

        score = 0.0
        start = len(prompt_ids)
        for i, tid in enumerate(cont_ids):
            pos = start + i - 1
            if pos < 0:
                continue
            vec = np.array(logits[0, pos])
            vmax = float(np.max(vec))
            logsumexp = vmax + float(np.log(np.sum(np.exp(vec - vmax))))
            score += float(vec[tid] - logsumexp)
        return score

    def _heuristic_classify(self, text: str) -> ClassificationResult:
        lowered = text.lower().strip()
        if not lowered:
            return ClassificationResult(category="casual", confidence=0.65, used_majority_vote=False)

        rule_hits = {
            "question": 1 if "?" in lowered else 0,
            "logistics": int(any(w in lowered for w in ["when", "where", "time", "address", "eta", "omw", "on my way"])),
            "emotional": int(any(w in lowered for w in ["sad", "upset", "stressed", "anxious", "hurt", "sorry"])),
            "invitation": int(any(w in lowered for w in ["come", "join", "hang", "party", "dinner", "tonight"])),
            "decline": int(any(w in lowered for w in ["can't", "cannot", "won't", "not able", "maybe another time"])),
            "flirty": int(any(w in lowered for w in ["cute", "miss you", "babe", "hot", "date"])),
            "confrontational": int(any(w in lowered for w in ["why did you", "mad", "angry", "annoyed", "seriously"])),
            "hype": int(any(w in lowered for w in ["let's go", "lfg", "hyped", "fire", "insane", "!!!"])),
            "informational": int(any(w in lowered for w in ["article", "link", "news", "read", "info", "details"])),
        }

        best = max(rule_hits, key=rule_hits.get)
        if rule_hits[best] > 0:
            return ClassificationResult(category=best, confidence=0.78, used_majority_vote=False)

        if len(lowered.split()) <= 4:
            return ClassificationResult(category="casual", confidence=0.7, used_majority_vote=False)
        return ClassificationResult(category="tricky", confidence=0.55, used_majority_vote=False)
