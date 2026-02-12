"""Pure MLX DeBERTa-v3 for sequence classification (NLI).

Implements disentangled attention from DeBERTa:
- Content-to-content (standard attention)
- Content-to-position (query attends to relative position)
- Position-to-content (key attends to relative position)

Target: cross-encoder/nli-deberta-v3-xsmall (22M params, 12 layers, 384 hidden)
Memory: ~90MB loaded (fits well within 8GB budget alongside BERT + LFM)

Thread Safety:
    Uses MLXModelLoader._mlx_load_lock for all GPU operations.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class DebertaEmbeddings(nn.Module):
    """Word embeddings + LayerNorm. No position embeddings (uses relative)."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config.get("layer_norm_eps", 1e-7))

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.LayerNorm(self.word_embeddings(input_ids))


# ---------------------------------------------------------------------------
# Disentangled Self-Attention
# ---------------------------------------------------------------------------


class DisentangledAttention(nn.Module):
    """DeBERTa disentangled self-attention with log-bucketed relative positions."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads

        self.query_proj = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.key_proj = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.value_proj = nn.Linear(config["hidden_size"], config["hidden_size"])

        self.position_buckets = config.get("position_buckets", 256)
        self.max_relative = config.get("max_position_embeddings", 512)

        pos_att_raw = config.get("pos_att_type", "c2p|p2c")
        self.pos_att_type: list[str] = (
            pos_att_raw.split("|") if isinstance(pos_att_raw, str) else list(pos_att_raw)
        )

        self.share_att_key: bool = config.get("share_att_key", True)
        if not self.share_att_key:
            self.pos_key_proj = nn.Linear(config["hidden_size"], config["hidden_size"])
            self.pos_query_proj = nn.Linear(config["hidden_size"], config["hidden_size"])

        n_components = 1 + ("c2p" in self.pos_att_type) + ("p2c" in self.pos_att_type)
        self._scale = 1.0 / math.sqrt(self.head_dim * n_components)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None,
        rel_embeddings: mx.array,
    ) -> mx.array:
        B, S, _ = hidden_states.shape  # noqa: N806

        q = self._heads(self.query_proj(hidden_states), B, S)
        k = self._heads(self.key_proj(hidden_states), B, S)
        v = self._heads(self.value_proj(hidden_states), B, S)

        # Content-to-content
        scores = q @ k.transpose(0, 1, 3, 2)

        # Relative position components
        rel_idx = self._rel_pos_indices(S)

        if "c2p" in self.pos_att_type:
            pk_proj = self.key_proj if self.share_att_key else self.pos_key_proj
            pk = self._heads(pk_proj(rel_embeddings), 1, -1)
            c2p_all = q @ pk.transpose(0, 1, 3, 2)  # (B, H, S, 2*buckets)
            idx = mx.broadcast_to(rel_idx[None, None], (B, self.num_heads, S, S))
            scores = scores + mx.take_along_axis(c2p_all, idx, axis=-1)

        if "p2c" in self.pos_att_type:
            pq_proj = self.query_proj if self.share_att_key else self.pos_query_proj
            pq = self._heads(pq_proj(rel_embeddings), 1, -1)
            p2c_all = k @ pq.transpose(0, 1, 3, 2)  # (B, H, S, 2*buckets)
            idx_t = mx.broadcast_to(rel_idx.transpose(1, 0)[None, None], (B, self.num_heads, S, S))
            scores = scores + mx.take_along_axis(p2c_all, idx_t, axis=-1).transpose(0, 1, 3, 2)

        scores = scores * self._scale
        if attention_mask is not None:
            scores = scores + attention_mask

        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, -1)
        return out

    def _heads(self, x: mx.array, B: int, S: int) -> mx.array:  # noqa: N803
        """Reshape (B, S, H) -> (B, num_heads, S, head_dim)."""
        return x.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _rel_pos_indices(self, seq_len: int) -> mx.array:
        """Compute log-bucketed relative position indices. Shape: (S, S)."""
        pos = mx.arange(seq_len)
        rel = pos[:, None] - pos[None, :]
        return self._log_bucket(rel)

    def _log_bucket(self, rel_pos: mx.array) -> mx.array:
        """Map relative positions to log-bucket indices in [0, 2*buckets-1]."""
        sign = mx.sign(rel_pos)
        mid = self.position_buckets // 2
        abs_pos = mx.minimum(mx.abs(rel_pos), self.max_relative - 1).astype(mx.float32)

        is_small = abs_pos <= mid
        denom = max(math.log((self.max_relative - 1) / mid), 1e-6)
        log_pos = mx.ceil(mx.log(abs_pos / mid + 1e-6) / denom * (mid - 1)) + mid
        log_pos = mx.minimum(log_pos, self.position_buckets - 1)

        bucket = mx.where(is_small, abs_pos, log_pos)
        indices = (sign * bucket).astype(mx.int32) + self.position_buckets
        return mx.clip(indices, 0, 2 * self.position_buckets - 1)


# ---------------------------------------------------------------------------
# Transformer Layers
# ---------------------------------------------------------------------------


class DebertaLayer(nn.Module):
    """Single DeBERTa transformer layer."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        eps = config.get("layer_norm_eps", 1e-7)
        hidden = config["hidden_size"]
        intermediate = config["intermediate_size"]

        self.attention = DisentangledAttention(config)
        self.output_proj = nn.Linear(hidden, hidden)
        self.attn_norm = nn.LayerNorm(hidden, eps=eps)
        self.intermediate = nn.Linear(hidden, intermediate)
        self.output = nn.Linear(intermediate, hidden)
        self.ff_norm = nn.LayerNorm(hidden, eps=eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None,
        rel_emb: mx.array,
    ) -> mx.array:
        attn = self.output_proj(self.attention(x, mask, rel_emb))
        x = self.attn_norm(x + attn)
        ff = self.output(nn.gelu(self.intermediate(x)))
        return self.ff_norm(x + ff)


class DebertaEncoder(nn.Module):
    """DeBERTa encoder with shared relative position embeddings."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        buckets = config.get("position_buckets", 256)
        self.rel_embeddings = nn.Embedding(2 * buckets, config["hidden_size"])
        if config.get("norm_rel_ebd", "") == "layer_norm":
            self.norm = nn.LayerNorm(config["hidden_size"], eps=config.get("layer_norm_eps", 1e-7))
        self.layers = [DebertaLayer(config) for _ in range(config["num_hidden_layers"])]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array | None) -> mx.array:
        rel_emb = self.rel_embeddings.weight
        if hasattr(self, "norm"):
            rel_emb = self.norm(rel_emb)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rel_emb)
        return hidden_states


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class DebertaModel(nn.Module):
    """DeBERTa-v3 base model (embeddings + encoder)."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        x = self.embeddings(input_ids)
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :].astype(mx.float32)) * -10000.0
        else:
            mask = None
        return self.encoder(x, mask)


class DebertaForSequenceClassification(nn.Module):
    """DeBERTa-v3 with [CLS] pooler + classification head.

    For NLI: num_labels=3 (contradiction, entailment, neutral).
    """

    def __init__(self, config: dict, num_labels: int = 3) -> None:
        super().__init__()
        self.deberta = DebertaModel(config)
        hidden = config.get("pooler_hidden_size", config["hidden_size"])
        self.pooler = nn.Linear(config["hidden_size"], hidden)
        self.classifier = nn.Linear(hidden, num_labels)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass returning raw logits of shape (batch, num_labels)."""
        hidden = self.deberta(input_ids, attention_mask)
        pooled = nn.gelu(self.pooler(hidden[:, 0]))
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Weight Converter
# ---------------------------------------------------------------------------


def convert_hf_weights(hf_weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Convert HuggingFace DeBERTa-v3 weight names to MLX model paths.

    Mapping (HF -> MLX):
        encoder.layer.N          -> encoder.layers.N
        .attention.self.X        -> .attention.X
        .attention.output.dense  -> .output_proj
        .attention.output.LayerNorm -> .attn_norm
        .intermediate.dense      -> .intermediate
        .output.dense            -> .output
        .output.LayerNorm        -> .ff_norm
        encoder.LayerNorm        -> encoder.norm
        pooler.dense             -> pooler
    """
    converted: dict[str, mx.array] = {}

    for hf_name, weight in hf_weights.items():
        if "position_ids" in hf_name:
            continue

        name = hf_name
        name = name.replace("encoder.layer.", "encoder.layers.")
        name = name.replace(".attention.self.", ".attention.")
        name = name.replace(".attention.output.dense", ".output_proj")
        name = name.replace(".attention.output.LayerNorm", ".attn_norm")
        name = name.replace(".intermediate.dense", ".intermediate")
        # .output.dense and .output.LayerNorm (FF layer only, attention already renamed)
        name = name.replace(".output.dense", ".output")
        name = name.replace(".output.LayerNorm", ".ff_norm")
        name = name.replace("encoder.LayerNorm", "encoder.norm")
        name = name.replace("pooler.dense", "pooler")

        converted[name] = weight

    return converted
