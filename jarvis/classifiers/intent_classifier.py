"""Intent classification backends for two-step reply routing.

Supports:
- Keyword fallback (always available)
- Transformers backends (FalconsAI/DeBERTa/Flan-T5/etc.)
- MLX prompt-based intent classification for MLX-compatible instruction models
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tokenizers import Tokenizer

ZERO_SHOT_LABEL_DESCRIPTIONS: dict[str, str] = {
    "no_reply_ack": "acknowledgment or reaction that does not need a reply",
    "no_reply_closing": "closing message ending the conversation",
    "reply_casual_chat": "casual chat that expects a light reply",
    "reply_question_info": "question asking for information or an answer",
    "reply_request_action": "request asking someone to do something",
    "reply_urgent_action": "urgent coordination requiring a quick response",
    "reply_emotional_support": "emotional message seeking support or empathy",
}


@dataclass
class IntentResult:
    """Result from intent classification."""

    intent: str
    confidence: float
    method: str
    all_scores: dict[str, float] = field(default_factory=dict)


class IntentClassifier(Protocol):
    """Interface for fallback intent classifiers."""

    def classify(self, text: str, intent_options: list[str]) -> IntentResult:
        """Classify a text into one of the provided intent options."""


@dataclass(frozen=True)
class IntentModelAlias:
    """Alias mapping for MLX intent backends."""

    name: str
    env_var: str
    default_path: str
    preferred_backend: str = "hf"
    task: str = "sequence"


INTENT_MODEL_ALIASES: dict[str, IntentModelAlias] = {
    "falconsai": IntentModelAlias(
        name="falconsai",
        env_var="JARVIS_INTENT_MODEL_FALCONSAI",
        default_path="Falconsai/intent_classification",
        preferred_backend="mlx_distilbert",
        task="sequence",
    ),
    "deberta": IntentModelAlias(
        name="deberta",
        env_var="JARVIS_INTENT_MODEL_DEBERTA",
        default_path="microsoft/deberta-base-mnli",
        preferred_backend="hf",
        task="zero_shot",
    ),
    "bart": IntentModelAlias(
        name="bart",
        env_var="JARVIS_INTENT_MODEL_BART",
        default_path="facebook/bart-large-mnli",
        preferred_backend="hf",
        task="zero_shot",
    ),
    "flant5": IntentModelAlias(
        name="flant5",
        env_var="JARVIS_INTENT_MODEL_FLANT5",
        default_path="google/flan-t5-base",
        preferred_backend="hf",
        task="seq2seq",
    ),
    "mindpadi": IntentModelAlias(
        name="mindpadi",
        env_var="JARVIS_INTENT_MODEL_MINDPADI",
        default_path="mindpadi/mental-health-chatbot-intent-classifier",
        preferred_backend="hf",
        task="sequence",
    ),
    # Optional MLX-native aliases (instruction models)
    "mlx-qwen-0.5b": IntentModelAlias(
        name="mlx-qwen-0.5b",
        env_var="JARVIS_INTENT_MODEL_MLX_QWEN_05B",
        default_path="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        preferred_backend="mlx",
        task="prompt",
    ),
    "mlx-qwen-1.5b": IntentModelAlias(
        name="mlx-qwen-1.5b",
        env_var="JARVIS_INTENT_MODEL_MLX_QWEN_15B",
        default_path="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        preferred_backend="mlx",
        task="prompt",
    ),
}


def _closest_intent_option(text: str, intent_options: list[str]) -> tuple[str, float]:
    normalized = text.strip().lower()
    for option in intent_options:
        if option.lower() == normalized:
            return option, 0.88
    for option in intent_options:
        if option.lower() in normalized or normalized in option.lower():
            return option, 0.72
    return (intent_options[0] if intent_options else "reply_casual_chat", 0.50)


class KeywordIntentClassifier:
    """Fast deterministic fallback classifier.

    This is used as the default fallback implementation and is designed to be
    easy to replace with an ML-backed classifier later.
    """

    _NO_REPLY_PATTERN = re.compile(
        r"^(ok(ay)?|k|kk|sure|bet|got it|gotcha|thanks|thx|ty|bye|cya|ttyl|gn|gm|lol|lmao)[!?.]*$"
    )
    _QUESTION_PATTERN = re.compile(
        r"^(wya|wyd|hbu|what|where|when|who|why|how|which|r u|are you|do you|did you|"
        r"can you|could you|u free|you free)\b"
    )
    _REQUEST_PATTERN = re.compile(
        r"^(can you|could you|would you|please|lmk\b|let me know\b|wanna\b|down\??$|"
        r"thoughts\??$|text me\b|call me\b|pick me up\b|send me\b)"
    )
    _EMOTIONAL_PATTERN = re.compile(
        r"(omg|wow|yay|ugh|damn|congrats|congratulations|i love you|i miss you|i'm so|im so|!!+)"
    )

    def classify(self, text: str, intent_options: list[str]) -> IntentResult:
        normalized = " ".join(text.lower().split())

        scored: list[tuple[str, float]] = []

        for intent in intent_options:
            score = 0.0
            if intent.startswith("no_reply"):
                if self._NO_REPLY_PATTERN.match(normalized):
                    score = 0.95
                elif len(normalized) <= 3:
                    score = 0.85
                else:
                    score = 0.10
            elif intent.startswith("reply_question"):
                score = 0.90 if self._QUESTION_PATTERN.search(normalized) else 0.15
            elif intent.startswith("reply_request") or intent.startswith("reply_urgent"):
                score = 0.90 if self._REQUEST_PATTERN.search(normalized) else 0.12
            elif intent.startswith("reply_emotional"):
                score = 0.85 if self._EMOTIONAL_PATTERN.search(normalized) else 0.18
            elif intent.startswith("reply_casual"):
                score = 0.55
            scored.append((intent, score))

        if not scored:
            return IntentResult(
                intent="reply_casual_chat",
                confidence=0.0,
                method="keyword_fallback",
            )

        scored.sort(key=lambda x: x[1], reverse=True)
        best_intent, best_score = scored[0]
        return IntentResult(
            intent=best_intent,
            confidence=float(best_score),
            method="keyword_fallback",
            all_scores={k: float(v) for k, v in scored},
        )


class MLXPromptIntentClassifier:
    """Prompt-based intent classifier running on an MLX model.

    This backend works with any instruction/chat model converted for MLX.
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        model_alias: str | None = None,
        max_tokens: int = 8,
    ) -> None:
        from models.loader import MLXModelLoader, ModelConfig

        resolved_path = model_path
        if model_alias:
            alias = INTENT_MODEL_ALIASES.get(model_alias)
            if alias is None:
                raise ValueError(f"Unknown model_alias '{model_alias}'")
            resolved_path = os.getenv(alias.env_var, alias.default_path)

        if not resolved_path:
            raise ValueError("model_path or model_alias is required for MLXPromptIntentClassifier")

        self._loader = MLXModelLoader(ModelConfig(model_path=resolved_path))
        self._max_tokens = max_tokens
        self._method = f"mlx_prompt:{resolved_path}"

    @staticmethod
    def _extract_label(raw: str, intent_options: list[str]) -> tuple[str, float]:
        lines = raw.strip().splitlines()
        normalized = lines[0].strip().strip('"').strip("'") if lines else ""
        normalized_l = normalized.lower()

        exact = {opt.lower(): opt for opt in intent_options}
        if normalized_l in exact:
            return exact[normalized_l], 0.88

        for option in intent_options:
            if option.lower() in normalized_l:
                return option, 0.72

        if intent_options:
            return intent_options[0], 0.50
        return "reply_casual_chat", 0.50

    def classify(self, text: str, intent_options: list[str]) -> IntentResult:
        if not self._loader.is_loaded():
            self._loader.load()

        options_text = ", ".join(intent_options)
        prompt = (
            "Classify this text message into exactly one label.\n"
            f"Labels: {options_text}\n"
            f"Message: {text}\n"
            "Return only the label."
        )
        result = self._loader.generate_sync(
            prompt,
            max_tokens=self._max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            pre_formatted=False,
        )
        best_label, confidence = self._extract_label(result.text, intent_options)
        return IntentResult(
            intent=best_label,
            confidence=confidence,
            method=self._method,
            all_scores={best_label: confidence},
        )


class _MLXDistilEmbeddings(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg["vocab_size"], cfg["dim"])
        self.position_embeddings = nn.Embedding(cfg["max_position_embeddings"], cfg["dim"])
        self.LayerNorm = nn.LayerNorm(cfg["dim"], eps=1e-12)

    def __call__(self, input_ids: mx.array) -> mx.array:
        seq_len = input_ids.shape[1]
        pos = mx.arange(seq_len)
        out = self.word_embeddings(input_ids) + self.position_embeddings(pos)
        return self.LayerNorm(out)


class _MLXDistilAttention(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        dim = cfg["dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = dim // self.n_heads
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, attention_mask: mx.array | None) -> mx.array:
        bsz, seq, dim = x.shape
        q = self.q_lin(x).reshape(bsz, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_lin(x).reshape(bsz, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_lin(x).reshape(bsz, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :].astype(mx.float32)) * -1e9
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(bsz, seq, dim)
        return self.out_lin(out)


class _MLXDistilLayer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        dim = cfg["dim"]
        hidden_dim = cfg["hidden_dim"]
        self.attention = _MLXDistilAttention(cfg)
        self.sa_layer_norm = nn.LayerNorm(dim, eps=1e-12)
        self.ffn = nn.Module()
        self.ffn.lin1 = nn.Linear(dim, hidden_dim)
        self.ffn.lin2 = nn.Linear(hidden_dim, dim)
        self.output_layer_norm = nn.LayerNorm(dim, eps=1e-12)

    def __call__(self, x: mx.array, attention_mask: mx.array | None) -> mx.array:
        x = self.sa_layer_norm(x + self.attention(x, attention_mask))
        ff = self.ffn.lin2(nn.gelu(self.ffn.lin1(x)))
        return self.output_layer_norm(x + ff)


class _MLXDistilTransformer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.layer = [_MLXDistilLayer(cfg) for _ in range(cfg["n_layers"])]

    def __call__(self, x: mx.array, attention_mask: mx.array | None) -> mx.array:
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


class _MLXDistilBertModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.embeddings = _MLXDistilEmbeddings(cfg)
        self.transformer = _MLXDistilTransformer(cfg)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        return self.transformer(self.embeddings(input_ids), attention_mask)


class _MLXDistilBertForSequenceClassification(nn.Module):
    def __init__(self, cfg: dict, num_labels: int) -> None:
        super().__init__()
        self.distilbert = _MLXDistilBertModel(cfg)
        self.pre_classifier = nn.Linear(cfg["dim"], cfg["dim"])
        self.classifier = nn.Linear(cfg["dim"], num_labels)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        hidden = self.distilbert(input_ids, attention_mask)
        pooled = hidden[:, 0]
        return self.classifier(nn.relu(self.pre_classifier(pooled)))


class MLXDistilBertIntentClassifier:
    """Native MLX DistilBERT sequence-classification backend."""

    def __init__(
        self,
        model_path: str,
        *,
        max_length: int = 128,
    ) -> None:
        self._model_path = Path(
            snapshot_download(
                repo_id=model_path,
                allow_patterns=[
                    "*.safetensors",
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.txt",
                    "special_tokens_map.json",
                ],
            )
        )
        cfg = json.loads((self._model_path / "config.json").read_text())
        self._id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
        num_labels = int(cfg.get("num_labels", len(self._id2label) or 2))
        self._model = _MLXDistilBertForSequenceClassification(cfg, num_labels=num_labels)
        self._max_length = max_length
        self._method = f"mlx_distilbert:{model_path}"

        tok_json = self._model_path / "tokenizer.json"
        if tok_json.exists():
            self._tokenizer = Tokenizer.from_file(str(tok_json))
            self._use_hf_tokenizer = False
        else:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
            self._use_hf_tokenizer = True

        self._pad_id = 0
        if not self._use_hf_tokenizer:
            pad_id = self._tokenizer.token_to_id("[PAD]")
            if pad_id is not None:
                self._pad_id = pad_id

        self._load_weights()

    def _load_weights(self) -> None:
        safetensor_files = sorted(self._model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors found in {self._model_path}")

        loaded: dict[str, mx.array] = {}
        for file in safetensor_files:
            with safe_open(str(file), framework="np") as f:
                for key in f.keys():
                    loaded[key] = mx.array(f.get_tensor(key))
        self._model.load_weights(list(loaded.items()))
        mx.eval(self._model.parameters())

    def _encode(self, text: str) -> tuple[mx.array, mx.array]:
        if self._use_hf_tokenizer:
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=self._max_length,
                padding="max_length",
            )
            ids = encoded["input_ids"]
            attn = encoded["attention_mask"]
        else:
            e = self._tokenizer.encode(text)
            ids = e.ids[: self._max_length]
            attn = [1] * len(ids)
            if len(ids) < self._max_length:
                pad_len = self._max_length - len(ids)
                ids.extend([self._pad_id] * pad_len)
                attn.extend([0] * pad_len)

        return (
            mx.array([ids], dtype=mx.int32),
            mx.array([attn], dtype=mx.float32),
        )

    def classify(self, text: str, intent_options: list[str]) -> IntentResult:
        input_ids, attention_mask = self._encode(text)
        logits = self._model(input_ids, attention_mask)
        probs = mx.softmax(logits, axis=-1)
        probs_np = np.array(probs)[0].tolist()

        scores: dict[str, float] = {opt: 0.0 for opt in intent_options}
        for idx, prob in enumerate(probs_np):
            label = str(self._id2label.get(idx, f"label_{idx}"))
            mapped, mapped_score = _closest_intent_option(label, intent_options)
            scores[mapped] = max(scores[mapped], float(prob) * mapped_score)

        best_intent, best_score = max(scores.items(), key=lambda kv: kv[1])
        return IntentResult(
            intent=best_intent,
            confidence=float(best_score),
            method=self._method,
            all_scores={k: float(v) for k, v in scores.items()},
        )


class HFTransformersIntentClassifier:
    """HuggingFace Transformers-backed intent classifier.

    This is the real backend for FalconsAI/DeBERTa/Flan-T5 style checkpoints.
    """

    def __init__(self, model_id: str, task: str = "sequence") -> None:
        self._model_id = model_id
        self._task = task
        self._method = f"hf:{model_id}"

        if task == "zero_shot":
            from transformers import pipeline

            self._pipe = pipeline(
                "zero-shot-classification",
                model=model_id,
            )
            return

        if task == "seq2seq":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            return

        # Default: sequence classification model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._id2label = getattr(self._model.config, "id2label", {})

    @staticmethod
    def _closest_option(text: str, intent_options: list[str]) -> tuple[str, float]:
        return _closest_intent_option(text, intent_options)

    @staticmethod
    def _build_zero_shot_candidates(intent_options: list[str]) -> tuple[list[str], dict[str, str]]:
        candidates: list[str] = []
        label_to_intent: dict[str, str] = {}
        for intent in intent_options:
            candidate = ZERO_SHOT_LABEL_DESCRIPTIONS.get(intent, intent.replace("_", " "))
            candidates.append(candidate)
            label_to_intent[candidate] = intent
        return candidates, label_to_intent

    def classify(self, text: str, intent_options: list[str]) -> IntentResult:
        if self._task == "zero_shot":
            candidates, label_to_intent = self._build_zero_shot_candidates(intent_options)
            out = self._pipe(text, candidates, multi_label=False)
            best_label = out["labels"][0]
            best = label_to_intent.get(best_label, best_label)
            score = float(out["scores"][0])
            all_scores: dict[str, float] = {intent: 0.0 for intent in intent_options}
            for label, raw_score in zip(out["labels"], out["scores"]):
                mapped_intent = label_to_intent.get(label)
                if mapped_intent is not None:
                    all_scores[mapped_intent] = max(all_scores[mapped_intent], float(raw_score))
            return IntentResult(
                intent=best,
                confidence=score,
                method=self._method,
                all_scores=all_scores,
            )

        if self._task == "seq2seq":
            prompt = (
                f"Classify message into one label: {', '.join(intent_options)}.\n"
                f"Message: {text}\nLabel:"
            )
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self._model.generate(**inputs, max_length=12)
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            best, score = self._closest_option(decoded, intent_options)
            return IntentResult(
                intent=best,
                confidence=score,
                method=self._method,
                all_scores={best: score},
            )

        # Sequence classification
        import torch

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        ranked = sorted(
            ((int(i), float(p)) for i, p in enumerate(probs.tolist())),
            key=lambda x: x[1],
            reverse=True,
        )

        scores: dict[str, float] = {opt: 0.0 for opt in intent_options}
        for idx, score in ranked:
            label = str(self._id2label.get(idx, f"LABEL_{idx}")).lower()
            mapped, mapped_score = self._closest_option(label, intent_options)
            scores[mapped] = max(scores[mapped], mapped_score * score)

        best_intent, best_score = max(scores.items(), key=lambda kv: kv[1])
        return IntentResult(
            intent=best_intent,
            confidence=float(best_score),
            method=self._method,
            all_scores={k: float(v) for k, v in scores.items()},
        )


def create_intent_classifier(
    backend: str = "keyword",
    *,
    model_path: str | None = None,
    model_alias: str | None = None,
) -> IntentClassifier:
    """Create an intent classifier backend."""
    backend = backend.strip().lower()
    if backend == "keyword":
        return KeywordIntentClassifier()
    if backend == "hf":
        if model_alias:
            alias = INTENT_MODEL_ALIASES.get(model_alias)
            if alias is None:
                raise ValueError(f"Unknown model_alias '{model_alias}'")
            resolved = os.getenv(alias.env_var, alias.default_path)
            return HFTransformersIntentClassifier(model_id=resolved, task=alias.task)
        if not model_path:
            raise ValueError("model_path or model_alias is required for hf backend")
        return HFTransformersIntentClassifier(model_id=model_path, task="sequence")
    if backend == "mlx_distilbert":
        if model_alias:
            alias = INTENT_MODEL_ALIASES.get(model_alias)
            if alias is None:
                raise ValueError(f"Unknown model_alias '{model_alias}'")
            resolved = os.getenv(alias.env_var, alias.default_path)
            return MLXDistilBertIntentClassifier(model_path=resolved)
        if not model_path:
            raise ValueError("model_path or model_alias is required for mlx_distilbert backend")
        return MLXDistilBertIntentClassifier(model_path=model_path)
    if backend == "mlx":
        return MLXPromptIntentClassifier(model_path=model_path, model_alias=model_alias)
    if backend == "alias":
        if not model_alias:
            raise ValueError("model_alias is required for alias backend")
        alias = INTENT_MODEL_ALIASES.get(model_alias)
        if alias is None:
            raise ValueError(f"Unknown model_alias '{model_alias}'")
        resolved = os.getenv(alias.env_var, alias.default_path)
        if alias.preferred_backend == "mlx_distilbert":
            return MLXDistilBertIntentClassifier(model_path=resolved)
        if alias.preferred_backend == "mlx":
            return MLXPromptIntentClassifier(model_path=resolved)
        return HFTransformersIntentClassifier(model_id=resolved, task=alias.task)
    raise ValueError(f"Unknown intent backend '{backend}'")


__all__ = [
    "INTENT_MODEL_ALIASES",
    "IntentModelAlias",
    "IntentClassifier",
    "IntentResult",
    "KeywordIntentClassifier",
    "MLXDistilBertIntentClassifier",
    "HFTransformersIntentClassifier",
    "MLXPromptIntentClassifier",
    "create_intent_classifier",
]
