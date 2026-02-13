"""Pure MLX GLiNER for GPU-accelerated Named Entity Recognition.

Ports urchade/gliner_medium-v2.1 (DeBERTa v3-base + span scoring layers) to MLX
for Metal GPU inference on Apple Silicon. Reuses models/deberta.py for the backbone.

Architecture:
    DeBERTa v3-base (768h, 12L) → Linear(768→512) → BiLSTM(512→256×2)
    → SpanMarkerV0 → PromptProjection → bilinear scoring → sigmoid → NMS

Thread Safety:
    Uses MLXModelLoader._mlx_load_lock for all GPU operations.

Usage:
    from models.gliner_mlx import get_mlx_gliner

    gliner = get_mlx_gliner()
    entities = gliner.predict_entities("I work at Google in Austin",
                                       ["person name", "place", "organization"])
"""

from __future__ import annotations

import gc
import logging
import re
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

from models.deberta import DebertaModel, convert_hf_weights
from models.memory_config import gpu_context

logger = logging.getLogger(__name__)

HF_CACHE = Path.home() / ".cache/huggingface/hub"

# GLiNER model
DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"

# DeBERTa v3-base config for gliner_medium-v2.1
DEBERTA_CONFIG = {
    "vocab_size": 128004,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "position_buckets": 256,
    "norm_rel_ebd": "layer_norm",
    "layer_norm_eps": 1e-7,
    "pos_att_type": "c2p|p2c",
    "share_att_key": True,
}

# GLiNER-specific config
GLINER_CONFIG = {
    "hidden_size": 512,  # projection dimension (post-DeBERTa)
    "max_width": 12,  # max span length in words
    "encoder_hidden": 768,  # DeBERTa output dim
}

# Special token IDs (added to DeBERTa v3-base vocab)
ENT_TOKEN_ID = 128002  # <<ENT>>
SEP_TOKEN_ID = 128003  # <<SEP>>
CLS_TOKEN_ID = 1  # [CLS]
SEP2_TOKEN_ID = 2  # [SEP] (end-of-sequence)
PAD_TOKEN_ID = 0  # [PAD]


# ---------------------------------------------------------------------------
# BiLSTM
# ---------------------------------------------------------------------------


class BiLSTM(nn.Module):
    """Bidirectional LSTM using two MLX LSTM instances."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.forward_lstm = nn.LSTM(input_size, hidden_size, bias=True)
        self.backward_lstm = nn.LSTM(input_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. x: (B, L, D) -> (B, L, 2*hidden_size)."""
        # Forward direction
        fwd_hidden, _ = self.forward_lstm(x)  # (B, L, H)

        # Backward direction: reverse input, run LSTM, reverse output
        x_rev = x[:, ::-1, :]
        bwd_hidden, _ = self.backward_lstm(x_rev)  # (B, L, H)
        bwd_hidden = bwd_hidden[:, ::-1, :]

        return mx.concatenate([fwd_hidden, bwd_hidden], axis=-1)  # (B, L, 2*H)


# ---------------------------------------------------------------------------
# SpanMarkerV0
# ---------------------------------------------------------------------------


class ProjectionLayer(nn.Module):
    """Linear(D, 4D) -> ReLU -> Linear(4D, D)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim * 4)
        self.linear2 = nn.Linear(output_dim * 4, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.relu(self.linear1(x)))


class SpanMarkerV0(nn.Module):
    """Span representation via start/end projection + concat + output MLP."""

    def __init__(self, hidden_size: int, max_width: int) -> None:
        super().__init__()
        self.max_width = max_width
        self.project_start = ProjectionLayer(hidden_size, hidden_size)
        self.project_end = ProjectionLayer(hidden_size, hidden_size)
        self.out_project = ProjectionLayer(hidden_size * 2, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Build span representations.

        Args:
            x: (B, L, D) word-level embeddings

        Returns:
            (B, num_words, max_width, D) span representations
        """
        B, L, D = x.shape  # noqa: N806

        start_rep = self.project_start(x)  # (B, L, D)
        end_rep = self.project_end(x)  # (B, L, D)
        mx.eval(start_rep, end_rep)

        # Build all (start, start+width) spans using index gather
        # For each start position i and width w, the span is [i, i+w] inclusive
        span_reps = []
        for w in range(self.max_width):
            # Gather end representations: for position i, get end_rep[i+w]
            # Positions where i+w >= L get zero-padded
            if w == 0:
                end_w = end_rep
            else:
                # Build index array for gather: [w, w+1, ..., L-1, L-1, L-1, ...]
                indices = mx.clip(mx.arange(L) + w, 0, L - 1)
                end_w = end_rep[:, indices, :]
                # Zero out positions where i+w >= L
                valid = mx.arange(L) + w < L  # (L,) bool
                end_w = end_w * valid[None, :, None]

            cat = mx.concatenate([start_rep, end_w], axis=-1)  # (B, L, 2D)
            projected = self.out_project(nn.relu(cat))  # (B, L, D)
            span_reps.append(projected)

        # Stack: (B, L, max_width, D)
        return mx.stack(span_reps, axis=2)


# ---------------------------------------------------------------------------
# PromptProjection
# ---------------------------------------------------------------------------


class PromptProjection(nn.Module):
    """2-layer MLP for entity type embeddings."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.relu(self.linear1(x)))


# ---------------------------------------------------------------------------
# Full GLiNER Model
# ---------------------------------------------------------------------------


class GLiNERModel(nn.Module):
    """GLiNER = DeBERTa + projection + BiLSTM + SpanMarker + PromptProjection."""

    def __init__(
        self,
        deberta_config: dict,
        hidden_size: int = 512,
        max_width: int = 12,
    ) -> None:
        super().__init__()
        encoder_hidden = deberta_config["hidden_size"]

        self.deberta = DebertaModel(deberta_config)
        self.projection = nn.Linear(encoder_hidden, hidden_size)
        self.rnn = BiLSTM(hidden_size, hidden_size // 2)  # 256 each dir -> 512
        self.span_rep = SpanMarkerV0(hidden_size, max_width)
        self.prompt_rep = PromptProjection(hidden_size)
        self.max_width = max_width

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        word_mask: mx.array,
        ent_positions: mx.array,
        num_words: int,
        num_labels: int,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
            word_mask: (B, seq_len) - word index for each subtoken (0=skip)
            ent_positions: (B, num_labels) - subtoken positions of <<ENT>> tokens
            num_words: max number of text words in batch
            num_labels: number of entity labels

        Returns:
            (B, num_words, max_width, num_labels) logits
        """
        B = input_ids.shape[0]  # noqa: N806
        D = DEBERTA_CONFIG["hidden_size"]  # noqa: N806

        # 1. DeBERTa encode
        token_embeds = self.deberta(input_ids, attention_mask)  # (B, seq_len, 768)

        # 2. Extract label embeddings at <<ENT>> positions (vectorized gather)
        # ent_positions: (B, num_labels) with indices into seq_len
        # Clamp negative positions to 0 (will be masked out)
        safe_pos = mx.maximum(ent_positions, 0)  # (B, num_labels)
        # Gather: for each batch b and label c, get token_embeds[b, safe_pos[b,c]]
        label_embeds = mx.take_along_axis(
            token_embeds,
            mx.broadcast_to(safe_pos[:, :, None], (B, num_labels, D)),
            axis=1,
        )  # (B, num_labels, 768)
        # Zero out invalid positions
        valid_mask = (ent_positions >= 0).astype(mx.float32)[:, :, None]
        label_embeds = label_embeds * valid_mask

        # Project labels: 768 -> 512 -> PromptProjection
        label_embeds = self.projection(label_embeds)
        label_embeds = self.prompt_rep(label_embeds)  # (B, num_labels, 512)

        # 3. Extract word embeddings (scatter-add subtokens by word ID)
        # word_mask: (B, seq_len) with 1-indexed word positions (0=skip)
        # Vectorized via identity-matrix indexing: eye[word_mask] produces one-hot,
        # then matmul accumulates token embeddings per word.
        wm_clamped = mx.clip(word_mask, 0, num_words)  # (B, seq_len)
        eye = mx.eye(num_words + 1)  # (num_words+1, num_words+1) identity
        one_hot = eye[wm_clamped]  # (B, seq_len, num_words+1) - fancy indexing
        assignment = one_hot[:, :, 1:]  # (B, seq_len, num_words) - drop 0/skip column
        # (B, num_words, seq_len) @ (B, seq_len, D) = (B, num_words, D)
        word_embeds = mx.transpose(assignment, axes=(0, 2, 1)) @ token_embeds
        # Project words: 768 -> 512
        word_embeds = self.projection(word_embeds)

        # 4. BiLSTM
        word_embeds = self.rnn(word_embeds)  # (B, num_words, 512)

        # Force eval to materialize shapes before span rep
        mx.eval(word_embeds, label_embeds)

        # 5. Span representations
        span_reps = self.span_rep(word_embeds)  # (B, num_words, max_width, 512)

        # 6. Bilinear scoring
        scores = mx.einsum("blwd,bcd->blwc", span_reps, label_embeds)

        return scores


# ---------------------------------------------------------------------------
# Weight Converter
# ---------------------------------------------------------------------------


def convert_gliner_weights(
    pt_weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Convert PyTorch GLiNER weight names to MLX model paths.

    Mapping:
        token_rep_layer.bert_layer.model.* -> deberta.*
            (then apply standard DeBERTa HF->MLX renames)
        token_rep_layer.projection.* -> projection.*
        rnn.lstm.weight_ih_l0 -> rnn.forward_lstm.Wx
        rnn.lstm.weight_hh_l0 -> rnn.forward_lstm.Wh
        rnn.lstm.bias_ih_l0 + bias_hh_l0 -> rnn.forward_lstm.bias
        rnn.lstm.*_reverse -> rnn.backward_lstm.*
        span_rep_layer.span_rep_layer.project_start.0.* -> span_rep.project_start.linear1.*
        span_rep_layer.span_rep_layer.project_start.3.* -> span_rep.project_start.linear2.*
        span_rep_layer.span_rep_layer.project_end.* -> span_rep.project_end.*
        span_rep_layer.span_rep_layer.out_project.* -> span_rep.out_project.*
        prompt_rep_layer.0.* -> prompt_rep.linear1.*
        prompt_rep_layer.3.* -> prompt_rep.linear2.*
    """
    converted: dict[str, mx.array] = {}
    lstm_biases: dict[str, mx.array] = {}

    for name, weight in pt_weights.items():
        if "position_ids" in name:
            continue

        # DeBERTa backbone weights
        if name.startswith("token_rep_layer.bert_layer.model."):
            stripped = name[len("token_rep_layer.bert_layer.model.") :]
            converted[f"__deberta__.{stripped}"] = weight
            continue

        # Projection layer
        if name.startswith("token_rep_layer.projection."):
            suffix = name[len("token_rep_layer.projection.") :]
            converted[f"projection.{suffix}"] = weight
            continue

        # LSTM weights
        if name.startswith("rnn.lstm."):
            lstm_name = name[len("rnn.lstm.") :]

            if "weight_ih_l0_reverse" in lstm_name:
                converted["rnn.backward_lstm.Wx"] = weight
            elif "weight_hh_l0_reverse" in lstm_name:
                converted["rnn.backward_lstm.Wh"] = weight
            elif "weight_ih_l0" in lstm_name:
                converted["rnn.forward_lstm.Wx"] = weight
            elif "weight_hh_l0" in lstm_name:
                converted["rnn.forward_lstm.Wh"] = weight
            elif "bias" in lstm_name:
                # Accumulate biases - MLX combines ih+hh
                lstm_biases[lstm_name] = weight
            continue

        # Span representation layer
        if name.startswith("span_rep_layer.span_rep_layer."):
            rest = name[len("span_rep_layer.span_rep_layer.") :]
            # Map Sequential index -> named sublayer
            rest = re.sub(r"\.0\.", ".linear1.", rest)
            rest = re.sub(r"\.3\.", ".linear2.", rest)
            converted[f"span_rep.{rest}"] = weight
            continue

        # Prompt representation layer
        if name.startswith("prompt_rep_layer."):
            rest = name[len("prompt_rep_layer.") :]
            rest = re.sub(r"^0\.", "linear1.", rest)
            rest = re.sub(r"^3\.", "linear2.", rest)
            converted[f"prompt_rep.{rest}"] = weight
            continue

        logger.warning("Unmapped GLiNER weight: %s", name)

    # Combine LSTM biases (ih + hh)
    if "bias_ih_l0" in lstm_biases and "bias_hh_l0" in lstm_biases:
        converted["rnn.forward_lstm.bias"] = lstm_biases["bias_ih_l0"] + lstm_biases["bias_hh_l0"]
    if "bias_ih_l0_reverse" in lstm_biases and "bias_hh_l0_reverse" in lstm_biases:
        converted["rnn.backward_lstm.bias"] = (
            lstm_biases["bias_ih_l0_reverse"] + lstm_biases["bias_hh_l0_reverse"]
        )

    # Apply DeBERTa HF->MLX renames
    deberta_weights = {
        k[len("__deberta__.") :]: v for k, v in converted.items() if k.startswith("__deberta__.")
    }
    mlx_deberta = convert_hf_weights(deberta_weights)

    # Replace __deberta__ placeholders with converted names
    final: dict[str, mx.array] = {
        k: v for k, v in converted.items() if not k.startswith("__deberta__.")
    }
    for k, v in mlx_deberta.items():
        final[f"deberta.{k}"] = v

    return final


# ---------------------------------------------------------------------------
# Tokenizer Wrapper
# ---------------------------------------------------------------------------


class GLiNERTokenizer:
    """Tokenizer for GLiNER input format.

    Handles the special input format:
        [CLS] <<ENT>> label1_tokens <<ENT>> label2_tokens ... <<SEP>> text_tokens [SEP]

    Uses the DeBERTa v3 fast tokenizer with GLiNER's special tokens.
    """

    def __init__(self, tokenizer: Tokenizer, max_length: int = 384) -> None:
        self._tok = tokenizer
        self._tok.enable_truncation(max_length=max_length)
        self.max_length = max_length

    def tokenize_input(
        self,
        words: list[str],
        labels: list[str],
    ) -> dict:
        """Build GLiNER input from text words and entity labels.

        Args:
            words: Pre-split text words.
            labels: Entity type label strings.

        Returns:
            Dict with input_ids, attention_mask, word_mask, ent_positions,
            word_to_char (word idx -> (char_start, char_end)).
        """
        # Build word list: [<<ENT>>, label1, <<ENT>>, label2, ..., <<SEP>>, word1, ...]
        prompt_words: list[str] = []
        for label in labels:
            prompt_words.append("<<ENT>>")
            prompt_words.append(label)
        prompt_words.append("<<SEP>>")

        all_words = prompt_words + words
        prompt_len = len(prompt_words)  # number of prompt "words"

        # Tokenize each word to subtokens
        input_ids: list[int] = [CLS_TOKEN_ID]
        word_ids: list[int] = [-1]  # -1 for special tokens

        for word_idx, word in enumerate(all_words):
            if word == "<<ENT>>":
                input_ids.append(ENT_TOKEN_ID)
                word_ids.append(word_idx)
            elif word == "<<SEP>>":
                input_ids.append(SEP_TOKEN_ID)
                word_ids.append(word_idx)
            else:
                self._tok.no_padding()
                encoding = self._tok.encode(word, add_special_tokens=False)
                for tid in encoding.ids:
                    input_ids.append(tid)
                    word_ids.append(word_idx)

        input_ids.append(SEP2_TOKEN_ID)
        word_ids.append(-1)

        # Truncate to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            word_ids = word_ids[: self.max_length]
            # Ensure ends with [SEP]
            input_ids[-1] = SEP2_TOKEN_ID
            word_ids[-1] = -1

        attention_mask = [1] * len(input_ids)

        # Build word_mask: 0 for prompt/special tokens, 1-indexed for text words
        word_mask: list[int] = []
        prev_word_id = -2  # sentinel
        text_word_counter = 0
        for wid in word_ids:
            if wid < 0:
                # Special token (CLS, SEP)
                word_mask.append(0)
            elif wid < prompt_len:
                # Prompt token
                word_mask.append(0)
            else:
                # Text word
                if wid != prev_word_id:
                    # First subtoken of a new word
                    text_word_counter += 1
                    word_mask.append(text_word_counter)
                else:
                    # Continuation subtoken
                    word_mask.append(0)
            prev_word_id = wid

        # Find <<ENT>> positions (for label embedding extraction)
        ent_positions: list[int] = []
        for i, tid in enumerate(input_ids):
            if tid == ENT_TOKEN_ID:
                ent_positions.append(i)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_mask": word_mask,
            "ent_positions": ent_positions,
            "num_words": text_word_counter,
        }

    def tokenize_batch(
        self,
        word_lists: list[list[str]],
        labels: list[str],
    ) -> dict:
        """Tokenize and pad a batch of texts.

        Args:
            word_lists: List of pre-split word lists.
            labels: Entity type labels (same for all texts).

        Returns:
            Dict with batched, padded arrays.
        """
        batch = [self.tokenize_input(words, labels) for words in word_lists]

        max_seq = max(len(b["input_ids"]) for b in batch)
        max_words = max(b["num_words"] for b in batch)
        num_labels = len(labels)

        input_ids = np.zeros((len(batch), max_seq), dtype=np.int32)
        attention_mask = np.zeros((len(batch), max_seq), dtype=np.int32)
        word_mask = np.zeros((len(batch), max_seq), dtype=np.int32)
        ent_positions = np.full((len(batch), num_labels), -1, dtype=np.int32)

        for i, b in enumerate(batch):
            seq_len = len(b["input_ids"])
            input_ids[i, :seq_len] = b["input_ids"]
            attention_mask[i, :seq_len] = b["attention_mask"]
            word_mask[i, :seq_len] = b["word_mask"]
            for j, pos in enumerate(b["ent_positions"]):
                if j < num_labels:
                    ent_positions[i, j] = pos

        return {
            "input_ids": mx.array(input_ids),
            "attention_mask": mx.array(attention_mask),
            "word_mask": mx.array(word_mask),
            "ent_positions": mx.array(ent_positions),
            "num_words": max_words,
            "num_labels": num_labels,
        }


# ---------------------------------------------------------------------------
# Span Decoder
# ---------------------------------------------------------------------------


def decode_spans(
    logits: np.ndarray,
    num_words_list: list[int],
    labels: list[str],
    threshold: float = 0.5,
    max_width: int = 12,
    flat_ner: bool = True,
) -> list[list[tuple[int, int, str, float]]]:
    """Decode model logits to entity spans.

    Args:
        logits: (B, L, W, C) raw logits
        num_words_list: actual number of words per batch item
        labels: entity label names
        threshold: sigmoid probability threshold
        max_width: maximum span width
        flat_ner: if True, remove overlapping spans (keep highest score)

    Returns:
        List of lists of (start_word, end_word, label, score) tuples.
        start_word is inclusive, end_word is inclusive.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    batch_results: list[list[tuple[int, int, str, float]]] = []

    for b in range(logits.shape[0]):
        num_words = num_words_list[b]
        spans: list[tuple[int, int, str, float]] = []

        # Find all (start, width, class) above threshold
        indices = np.argwhere(probs[b] > threshold)  # (N, 3) - [pos, width, class]

        for idx in indices:
            start = int(idx[0])
            width = int(idx[1])
            class_idx = int(idx[2])

            end = start + width  # inclusive end
            if end >= num_words:
                continue
            if start >= num_words:
                continue

            score = float(probs[b, start, width, class_idx])
            label = labels[class_idx]
            spans.append((start, end, label, score))

        # Greedy non-overlapping selection
        if flat_ner and spans:
            spans.sort(key=lambda x: -x[3])  # sort by score descending
            kept: list[tuple[int, int, str, float]] = []
            for span in spans:
                s, e, _, _ = span
                overlaps = any(not (e < ks or s > ke) for ks, ke, _, _ in kept)
                if not overlaps:
                    kept.append(span)
            spans = sorted(kept, key=lambda x: x[0])  # re-sort by position

        batch_results.append(spans)

    return batch_results


# ---------------------------------------------------------------------------
# MLXGLiNER Inference Class
# ---------------------------------------------------------------------------


class MLXGLiNER:
    """MLX GLiNER inference engine.

    Thread-safe:
    - _encode_lock serializes tokenizer calls
    - MLXModelLoader._mlx_load_lock serializes GPU operations
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GLINER_MODEL,
        dtype: str = "float32",
    ) -> None:
        self._model_name = model_name
        self._dtype = mx.float16 if dtype == "float16" else mx.float32
        self._encode_lock = threading.Lock()
        self.model: GLiNERModel | None = None
        self.tokenizer: GLiNERTokenizer | None = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the GLiNER model. Thread-safe via GPU lock."""
        if self._loaded:
            return

        with gpu_context():
            if self._loaded:
                return

            start = time.time()

            # Find model files
            gliner_dir = self._find_gliner_snapshot()
            tokenizer_path = self._find_tokenizer()

            # Load tokenizer
            base_tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.tokenizer = GLiNERTokenizer(base_tokenizer, max_length=384)

            # Build model
            logger.info(
                "Loading MLX GLiNER model %s (hidden=768→512, 12 layers, BiLSTM)",
                self._model_name,
            )

            self.model = GLiNERModel(
                DEBERTA_CONFIG,
                hidden_size=GLINER_CONFIG["hidden_size"],
                max_width=GLINER_CONFIG["max_width"],
            )

            # Load and convert weights
            weights_path = gliner_dir / "model.safetensors"
            if not weights_path.exists():
                raise FileNotFoundError(f"model.safetensors not found in {gliner_dir}")

            pt_weights = mx.load(str(weights_path))
            converted = convert_gliner_weights(pt_weights)
            del pt_weights  # Free ~744MB before loading converted weights
            gc.collect()

            # Optional fp16 conversion (saves ~372MB but ~2x slower on Metal)
            if self._dtype == mx.float16:
                converted = {k: v.astype(mx.float16) for k, v in converted.items()}
                logger.info("Using fp16 weights (372MB vs 744MB fp32)")

            self.model.load_weights(list(converted.items()))
            mx.eval(self.model.parameters())

            self._loaded = True
            dtype_str = "fp16" if self._dtype == mx.float16 else "fp32"
            logger.info("MLX GLiNER loaded in %.2fs (%s)", time.time() - start, dtype_str)

    def _find_gliner_snapshot(self) -> Path:
        """Find the GLiNER model snapshot in HF cache."""
        hf_repo = self._model_name.replace("/", "--")
        model_dir = HF_CACHE / f"models--{hf_repo}"

        if not model_dir.exists():
            self._download_model()
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"GLiNER model not found: {model_dir}. "
                    f"Download with: huggingface-cli download {self._model_name}"
                )

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            raise FileNotFoundError(f"No snapshots in {model_dir}")
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            raise FileNotFoundError(f"Empty snapshots dir: {snapshots_dir}")
        return snapshots[0]

    def _find_tokenizer(self) -> Path:
        """Find a DeBERTa v3 tokenizer.json in HF cache.

        Checks NLI cross-encoder first (already has tokenizer.json),
        falls back to DeBERTa v3-base.
        """
        # Try NLI model tokenizer (same DeBERTa v3 family, already has tokenizer.json)
        nli_dir = HF_CACHE / "models--cross-encoder--nli-deberta-v3-xsmall"
        if nli_dir.exists():
            for snap in (nli_dir / "snapshots").iterdir():
                tok_path = snap / "tokenizer.json"
                if tok_path.exists():
                    return tok_path

        # Try DeBERTa v3-base
        base_dir = HF_CACHE / "models--microsoft--deberta-v3-base"
        if base_dir.exists():
            for snap in (base_dir / "snapshots").iterdir():
                tok_path = snap / "tokenizer.json"
                if tok_path.exists():
                    return tok_path

        raise FileNotFoundError(
            "No DeBERTa v3 tokenizer.json found in HF cache. "
            "Download with: huggingface-cli download cross-encoder/nli-deberta-v3-xsmall"
        )

    def _download_model(self) -> None:
        """Download GLiNER model from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download

            logger.info("Downloading GLiNER model: %s", self._model_name)
            snapshot_download(
                self._model_name,
                allow_patterns=[
                    "gliner_config.json",
                    "model.safetensors",
                ],
            )
        except ImportError:
            logger.warning("huggingface_hub not installed, cannot auto-download")
        except Exception as e:
            logger.warning("Failed to download %s: %s", self._model_name, e)

    def unload(self) -> None:
        """Unload model to free memory."""
        with gpu_context():
            self.model = None
            self.tokenizer = None
            self._loaded = False
            gc.collect()
            mx.clear_cache()
            logger.info("Unloaded MLX GLiNER model")

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[dict]:
        """Predict entities in a single text.

        Args:
            text: Input text.
            labels: Entity type labels.
            threshold: Confidence threshold.
            flat_ner: If True, remove overlapping spans.

        Returns:
            List of dicts with keys: start, end, text, label, score.
        """
        results = self.predict_batch([text], labels, threshold=threshold, flat_ner=flat_ner)
        return results[0]

    def predict_batch(
        self,
        texts: list[str],
        labels: list[str],
        batch_size: int = 8,
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[list[dict]]:
        """Predict entities in a batch of texts.

        Args:
            texts: Input texts.
            labels: Entity type labels (same for all texts).
            batch_size: Processing batch size.
            threshold: Confidence threshold.
            flat_ner: If True, remove overlapping spans.

        Returns:
            List of lists of entity dicts.
        """
        if not self._loaded:
            self.load_model()

        if not texts or not labels:
            return [[] for _ in texts]

        all_results: list[list[dict]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]

            # Split texts into words and track char offsets
            word_lists: list[list[str]] = []
            char_maps: list[list[tuple[int, int]]] = []  # (start, end) per word

            for text in batch_texts:
                words, offsets = _split_words(text)
                word_lists.append(words)
                char_maps.append(offsets)

            num_words_list = [len(wl) for wl in word_lists]

            # Tokenize + forward pass under single GPU lock to prevent race
            # conditions on shared tokenizer state and Metal GPU.
            with gpu_context():
                batch_data = self.tokenizer.tokenize_batch(word_lists, labels)

                logits = self.model(
                    batch_data["input_ids"],
                    batch_data["attention_mask"],
                    batch_data["word_mask"],
                    batch_data["ent_positions"],
                    batch_data["num_words"],
                    batch_data["num_labels"],
                )
                mx.eval(logits)

            logits_np = np.array(logits)

            # Decode spans
            batch_spans = decode_spans(
                logits_np,
                num_words_list,
                labels,
                threshold=threshold,
                max_width=GLINER_CONFIG["max_width"],
                flat_ner=flat_ner,
            )

            # Map word indices to character offsets
            for text, spans, char_map in zip(batch_texts, batch_spans, char_maps):
                entities: list[dict] = []
                for start_word, end_word, label, score in spans:
                    if start_word >= len(char_map) or end_word >= len(char_map):
                        continue
                    char_start = char_map[start_word][0]
                    char_end = char_map[end_word][1]
                    entity_text = text[char_start:char_end]
                    entities.append(
                        {
                            "start": char_start,
                            "end": char_end,
                            "text": entity_text,
                            "label": label,
                            "score": score,
                        }
                    )
                all_results.append(entities)

        return all_results

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._loaded


# ---------------------------------------------------------------------------
# Word Splitter
# ---------------------------------------------------------------------------


def _split_words(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Split text into words with character offsets.

    Returns:
        (words, offsets) where offsets[i] = (char_start, char_end) for words[i].
    """
    words: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        words.append(match.group())
        offsets.append((match.start(), match.end()))
    return words, offsets


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_gliner: MLXGLiNER | None = None
_gliner_lock = threading.Lock()


def get_mlx_gliner(
    model_name: str = DEFAULT_GLINER_MODEL,
    dtype: str = "float32",
) -> MLXGLiNER:
    """Get or create the singleton MLX GLiNER instance.

    Args:
        model_name: HuggingFace model name.
        dtype: "float32" (default, faster) or "float16" (half memory, slower).
    """
    global _gliner

    if _gliner is not None:
        return _gliner

    with _gliner_lock:
        if _gliner is None:
            _gliner = MLXGLiNER(model_name=model_name, dtype=dtype)
        return _gliner


def reset_mlx_gliner() -> None:
    """Reset the singleton. Unloads model and clears instance."""
    global _gliner

    with _gliner_lock:
        if _gliner is not None:
            _gliner.unload()
        _gliner = None
