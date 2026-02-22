from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from src.config import AppConfig


@dataclass
class RewardModelOutput:
    score: float


class StyleRewardModel:
    """Style reward model with trainable linear reward head over LM hidden features."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._using_fallback = True
        self._model = None
        self._tokenizer = None
        self._adapter_loaded = False
        self._head_w: np.ndarray | None = None
        self._head_b: float = 0.0
        self._try_load_base_model()

    def _try_load_base_model(self) -> None:
        if os.getenv("TEXT_REPLY_SYSTEM_SKIP_MODEL_LOAD") == "1":
            self._using_fallback = True
            return
        from mlx_lm import load  # type: ignore

        self._using_fallback = True
        candidates = self.config.model_candidates("reward_base")
        if not candidates:
            candidates = self.config.model_candidates("classifier_base")
        for model_id in candidates:
            try:
                self._model, self._tokenizer = load(model_id)
                self._using_fallback = False
                return
            except Exception:
                continue

    def load_style_adapter(self) -> None:
        adapter_path = Path(self.config.models["style_rm_adapter"])
        head_file = adapter_path / "reward_head.npz"
        if head_file.exists():
            data = np.load(head_file)
            self._head_w = data["w"].astype(np.float32)
            self._head_b = float(data["b"])
            self._adapter_loaded = True
        else:
            self._adapter_loaded = False

    def unload_style_adapter(self) -> None:
        self._adapter_loaded = False

    def score_candidates(
        self,
        relationship: str,
        incoming_message: str,
        candidate_replies: list[str],
    ) -> list[float]:
        if self._using_fallback or self._head_w is None:
            return [
                self._fallback_score(relationship, incoming_message, candidate)
                for candidate in candidate_replies
            ]
        scores: list[float] = []
        for candidate in candidate_replies:
            feat = self.extract_feature(relationship, incoming_message, candidate)
            if feat is None:
                scores.append(self._fallback_score(relationship, incoming_message, candidate))
                continue
            logit = float(np.dot(self._head_w, feat) + self._head_b)
            score = 1.0 / (1.0 + np.exp(-logit))
            scores.append(float(np.clip(score, 0.0, 1.0)))
        return scores

    def extract_feature(self, relationship: str, incoming_message: str, reply: str) -> np.ndarray | None:
        if self._model is None or self._tokenizer is None:
            return None
        text = f"[contact: {relationship}] [message: {incoming_message}] [reply: {reply}]"
        try:
            ids = self._tokenizer.encode(text, add_special_tokens=True)
            if not ids:
                return None
            x = mx.array([ids], dtype=mx.int32)
            hidden = self._model.model(x)
            # Mean-pool token features for short-text reward scoring.
            feat = np.array(hidden.mean(axis=1)[0], dtype=np.float32)
            return feat
        except Exception:
            return None

    def set_head(self, w: np.ndarray, b: float) -> None:
        self._head_w = w.astype(np.float32)
        self._head_b = float(b)

    @staticmethod
    def _fallback_score(relationship: str, incoming_message: str, reply: str) -> float:
        rel_bonus = 0.05 if relationship in {"friend", "partner", "family"} else 0.0
        in_len = max(1, len(incoming_message.split()))
        out_len = len(reply.split())
        length_ratio = min(out_len / in_len, 2.0)
        brevity = np.exp(-abs(length_ratio - 1.0))
        lower = reply.lower()
        style_bonus = 0.08 if any(tok in lower for tok in ["yeah", "yep", "lol", "haha", "omw", "got you"]) else 0.0
        punctuation_penalty = 0.12 if reply.count("!") > 3 else 0.0

        raw = 0.55 + rel_bonus + 0.25 * float(brevity) + style_bonus - punctuation_penalty
        return float(np.clip(raw, 0.0, 1.0))
