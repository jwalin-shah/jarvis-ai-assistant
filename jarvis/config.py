"""JARVIS Configuration System.

Loads and validates configuration from ~/.jarvis/config.json.
Uses Pydantic for schema validation with sensible defaults.

Supports migration from older config versions while preserving existing values.

Usage:
    from jarvis.config import get_config, save_config

    config = get_config()
    print(config.model_path)
    print(config.ui.theme)

    # Modify and save
    config.ui.theme = "dark"
    save_config(config)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".jarvis" / "config.json"


def validate_path(path: str | Path, description: str = "path") -> Path:
    """Validate a filesystem path, rejecting path traversal attempts.

    Args:
        path: The path to validate.
        description: Human-readable description for error messages.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If the path contains traversal sequences or is invalid.
    """
    path_str = str(path)
    # Reject explicit traversal sequences
    if ".." in path_str.split(os.sep) or ".." in path_str.split("/"):
        raise ValueError(f"Path traversal detected in {description}: {path_str}")
    # Reject null bytes (common injection technique)
    if "\x00" in path_str:
        raise ValueError(f"Null byte detected in {description}")
    resolved = Path(path_str).resolve()
    return resolved


# Current config schema version for migration tracking
CONFIG_VERSION = 13


class MemoryThresholds(BaseModel):
    """Memory thresholds for mode selection."""

    full_mode_mb: int = 8000
    lite_mode_mb: int = 4000


class UIConfig(BaseModel):
    """UI preferences for the Tauri frontend.

    Attributes:
        theme: Color theme preference ("light", "dark", or "system").
        font_size: Font size in pixels (12-24).
        show_timestamps: Whether to show message timestamps.
        compact_mode: Use compact layout with less padding.
    """

    theme: Literal["light", "dark", "system"] = "system"
    font_size: int = Field(default=14, ge=12, le=24)
    show_timestamps: bool = True
    compact_mode: bool = False


class SearchConfig(BaseModel):
    """Search preferences.

    Attributes:
        default_limit: Default number of search results to return.
        default_date_range_days: Default date range for searches (None = no limit).
    """

    default_limit: int = Field(default=50, ge=1, le=1000)
    default_date_range_days: int | None = Field(default=None, ge=1)


class ChatConfig(BaseModel):
    """Chat preferences.

    Attributes:
        stream_responses: Stream responses as they're generated.
        show_typing_indicator: Show typing indicator while generating.
    """

    stream_responses: bool = True
    show_typing_indicator: bool = True


class AdaptiveThresholdConfig(BaseModel):
    """Configuration for adaptive threshold adjustment based on feedback.

    Adaptive thresholds learn from user feedback (acceptance/rejection patterns)
    to optimize routing decisions. When enabled, the system analyzes feedback
    at different similarity score ranges to find optimal threshold boundaries.

    Attributes:
        enabled: Whether adaptive threshold adjustment is enabled.
        min_feedback_samples: Minimum feedback samples required before adaptation starts.
            Prevents premature adaptation with insufficient data.
        adaptation_window_hours: Hours of feedback history to consider for adaptation.
            Older feedback is weighted less. Use 0 for all-time.
        learning_rate: How quickly thresholds adapt to new feedback (0.0-1.0).
            Higher values = faster adaptation but more volatility.
        update_interval_minutes: How often to recompute adaptive thresholds.
            Lower values = more responsive but more CPU overhead.
        min_threshold_bounds: Minimum allowed values for each threshold.
            Prevents thresholds from dropping too low.
        max_threshold_bounds: Maximum allowed values for each threshold.
            Prevents thresholds from going too high.
        similarity_bucket_size: Size of similarity score buckets for analysis (e.g., 0.05 = 5%).
            Smaller buckets = finer granularity but need more data.
        acceptance_target: Target acceptance rate for adaptive optimization.
            System will try to find thresholds achieving this rate.
    """

    enabled: bool = False
    min_feedback_samples: int = Field(default=50, ge=10, le=1000)
    adaptation_window_hours: int = Field(default=168, ge=0, le=8760)  # 0 = all-time, max 1 year
    learning_rate: float = Field(default=0.2, ge=0.01, le=1.0)
    update_interval_minutes: int = Field(default=60, ge=5, le=1440)
    min_threshold_bounds: dict[str, float] = Field(
        default_factory=lambda: {
            "quick_reply": 0.80,
            "context": 0.50,
            "generate": 0.30,
        }
    )
    max_threshold_bounds: dict[str, float] = Field(
        default_factory=lambda: {
            "quick_reply": 0.99,
            "context": 0.85,
            "generate": 0.65,
        }
    )
    similarity_bucket_size: float = Field(default=0.05, ge=0.01, le=0.2)
    acceptance_target: float = Field(default=0.70, ge=0.3, le=0.95)


class RoutingConfig(BaseModel):
    """Routing thresholds and A/B testing configuration."""

    quick_reply_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    context_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    generate_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    coherence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    max_quick_replies: int = Field(default=5, ge=1, le=20)
    min_response_similarity: float = Field(default=0.60, ge=0.0, le=1.0)
    ab_test_group: str = Field(default="control")
    ab_test_thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)
    adaptive: AdaptiveThresholdConfig = Field(default_factory=AdaptiveThresholdConfig)
    # DEPRECATED: Use quick_reply_threshold instead (renamed for clarity)
    template_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for the API.

    Attributes:
        enabled: Whether rate limiting is enabled.
        requests_per_minute: Maximum requests per minute for read endpoints.
            Generation endpoints get 1/6 of this limit.
        generation_timeout_seconds: Timeout for generation requests.
        read_timeout_seconds: Timeout for read requests.
    """

    enabled: bool = True
    requests_per_minute: int = Field(default=60, ge=1, le=1000)
    generation_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    read_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)


class ModelSettings(BaseModel):
    """Model configuration for text generation.

    Attributes:
        model_id: Model identifier from the registry (e.g., "qwen-1.5b").
        auto_select: Automatically select the best model based on available RAM.
        max_tokens_reply: Maximum tokens for reply generation.
        max_tokens_summary: Maximum tokens for summarization.
        temperature: Sampling temperature for generation (0.0-2.0).
        generation_timeout_seconds: Timeout for model generation in seconds.
        idle_timeout_seconds: Unload model after this many seconds of inactivity.
            Set to 0 to disable automatic unloading.
        warm_on_startup: Pre-load model when warmer starts (increases startup time).
    """

    model_id: str = "lfm-0.7b"  # Using 0.7B model for both extraction and generation
    auto_select: bool = True
    max_tokens_reply: int = Field(default=150, ge=1, le=2048)
    max_tokens_summary: int = Field(default=500, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    generation_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    idle_timeout_seconds: float = Field(default=300.0, ge=0.0, le=3600.0)
    warm_on_startup: bool = False
    # Turbo Mode: Speculative decoding + KV cache optimization
    speculative_enabled: bool = True
    speculative_draft_model_id: str = "lfm-350m"
    speculative_num_draft_tokens: int = Field(default=4, ge=1, le=10)

    kv_cache_bits: int = Field(default=8, ge=2, le=16)


class TaskQueueConfig(BaseModel):
    """Task queue configuration for background operations.

    Attributes:
        max_completed_tasks: Maximum completed tasks to keep in memory.
        worker_poll_interval: Seconds between queue polls when idle.
        max_retries: Default maximum retry attempts for failed tasks.
        auto_start_worker: Automatically start the worker when queue is used.
    """

    max_completed_tasks: int = Field(default=100, ge=10, le=1000)
    worker_poll_interval: float = Field(default=1.0, ge=0.1, le=10.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    auto_start_worker: bool = True


class ClassifierThresholds(BaseModel):
    """Centralized classifier threshold configuration.

    These thresholds control the confidence levels at which various classifiers
    make decisions. All values are between 0.0 and 1.0.

    Attributes:
        message_rule_confidence: Confidence for rule-based message classification matches.
        message_embedding_threshold: Minimum embedding similarity to use for message type.
        message_rule_fallback: Below this confidence, try embedding classification.

        intent_confidence: Minimum confidence to return a specific intent.
        intent_quick_reply: Higher threshold for quick reply intent detection.

        # Note: response_* and trigger_svm_* thresholds removed - classifiers deprecated
    """

    # Intent classifier thresholds (from intent.py)
    intent_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    intent_quick_reply: float = Field(default=0.80, ge=0.0, le=1.0)

    # Note: Response and trigger classifier thresholds removed - classifiers deprecated
    # Current classifiers: intent, category, relationship, response_mobilization


class MetricsConfig(BaseModel):
    """Metrics collection configuration.

    Attributes:
        enabled: Whether metrics collection is enabled. Set to False for
            high-throughput scenarios where metrics overhead is unacceptable.
        buffer_size: Number of metrics to buffer before flushing to database.
        flush_interval_seconds: Maximum time between flushes in seconds.
    """

    enabled: bool = True
    buffer_size: int = Field(default=100, ge=1, le=10000)
    flush_interval_seconds: float = Field(default=5.0, ge=0.1, le=300.0)


class DigestConfig(BaseModel):
    """Digest generation preferences.

    Attributes:
        enabled: Whether digest generation is enabled.
        schedule: Digest schedule ("daily" or "weekly").
        preferred_time: Preferred time for digest generation (HH:MM format).
        include_action_items: Include detected action items in digest.
        include_stats: Include message statistics in digest.
        max_conversations: Maximum conversations to analyze for digest.
        export_format: Default export format ("markdown" or "html").
    """

    enabled: bool = True
    schedule: Literal["daily", "weekly"] = "daily"
    preferred_time: str = Field(default="08:00", pattern=r"^\d{2}:\d{2}$")
    include_action_items: bool = True
    include_stats: bool = True
    max_conversations: int = Field(default=50, ge=10, le=200)
    export_format: Literal["markdown", "html"] = "markdown"


class EmbeddingConfig(BaseModel):
    """Embedding model configuration.

    Supports multiple embedding models with versioned artifact storage.
    Changing the model requires retraining classifiers and rebuilding indexes.

    Available models (via mlx-embedding-models):
        - "bge-small": BAAI/bge-small-en-v1.5 (12 layers, ~100-150ms, MTEB ~62)
        - "gte-tiny": TaylorAI/gte-tiny (6 layers, ~50-70ms, MTEB ~57)
        - "minilm-l6": all-MiniLM-L6-v2 (6 layers, ~50-70ms, MTEB ~56)
        - "bge-micro": TaylorAI/bge-micro-v2 (3 layers, ~30-40ms, MTEB ~54)

    Attributes:
        model_name: Registry name for the embedding model.
        mlx_service_socket: Unix socket path for the MLX embedding microservice.
            Using Unix sockets provides ~10-50x lower latency than HTTP for local IPC.
    """

    model_name: str = "bge-small"
    mlx_service_socket: str = str(Path.home() / ".jarvis" / "jarvis-embed.sock")


class VecSearchConfig(BaseModel):
    """sqlite-vec search configuration.

    Attributes:
        embedding_dim: Dimension of embedding vectors.
        binary_prefilter_k: Number of candidates from binary hamming scan
            before int8 rerank in cross-contact queries.
        contact_boost: Score multiplier for same-contact matches.
    """

    embedding_dim: int = 384
    binary_prefilter_k: int = Field(default=100, ge=10, le=1000)
    contact_boost: float = Field(default=1.2, ge=1.0, le=3.0)


class RetrievalConfig(BaseModel):
    """Retrieval configuration.

    Attributes:
        reranker_enabled: Enable cross-encoder reranking of vec_search results.
        reranker_model: Cross-encoder model name from the registry.
        reranker_top_k: Number of results to return after reranking.
        reranker_candidates: Number of candidates to retrieve before reranking.
    """

    reranker_enabled: bool = False
    reranker_model: str = "ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = Field(default=3, ge=1, le=20)
    reranker_candidates: int = Field(default=10, ge=1, le=100)


class NormalizationProfile(BaseModel):
    """Text normalization settings for a specific task pipeline."""

    filter_garbage: bool = False
    filter_attributed_artifacts: bool = False
    drop_url_only: bool = False
    mask_entities: bool = False
    normalize_emojis: bool = False
    preserve_url_domain: bool = False
    replace_codes: bool = False
    ner_enabled: bool = True
    ner_model: str = "en_core_web_trf"
    expand_slang: bool = False
    spell_check: bool = False
    filter_non_english: bool = False
    min_length: int = Field(default=3, ge=0, le=1000)
    max_length: int = Field(default=500, ge=1, le=5000)


def _normalization_profile(
    *,
    normalize_emojis: bool = False,
    ner_enabled: bool = False,
    spell_check: bool = False,
    min_length: int = 1,
    max_length: int = 1000,
    **kwargs: Any,
) -> NormalizationProfile:
    """Build a NormalizationProfile with shared defaults for pipeline use."""
    return NormalizationProfile(
        filter_garbage=True,
        filter_attributed_artifacts=True,
        drop_url_only=True,
        mask_entities=False,
        preserve_url_domain=True,
        replace_codes=True,
        expand_slang=True,
        filter_non_english=False,
        normalize_emojis=normalize_emojis,
        ner_enabled=ner_enabled,
        spell_check=spell_check,
        min_length=min_length,
        max_length=max_length,
        **kwargs,
    )


class SegmentationConfig(BaseModel):
    """Configuration for semantic topic segmentation.

    Controls how conversations are segmented into topic-coherent chunks
    using embedding similarity, entity continuity, and text features.

    Attributes:
        enabled: Whether semantic segmentation is enabled. If False, falls back
            to time-based bundling.
        window_size: Size of sliding window for computing embedding centroids.
        similarity_threshold: Cosine similarity below this indicates topic drift.
        entity_weight: Weight for entity continuity in boundary score (0-1).
        entity_jaccard_threshold: Jaccard similarity below this indicates
            entity discontinuity between adjacent messages.
        time_gap_minutes: Hard boundary if time gap exceeds this (minutes).
        soft_gap_minutes: Contributes to boundary score if gap exceeds this.
        coreference_enabled: Whether to resolve pronouns before embedding.
            Requires fastcoref package.
        coreference_model: FastCoref model name (default: biu-nlp/f-coref).
        use_topic_shift_markers: Whether to use text markers like "btw", "anyway".
        topic_shift_weight: Weight for topic shift markers in boundary score.
        min_segment_messages: Minimum messages per segment (smaller merged).
        max_segment_messages: Maximum messages per segment (larger split).
        boundary_threshold: Score threshold for creating a boundary (0-1).
        forward_context_window: Messages to look ahead for continuity check.
        forward_continuity_threshold: Similarity above this means topic continues
            (reduces false boundary scores).
    """

    enabled: bool = True
    window_size: int = Field(default=3, ge=1, le=10)
    similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    entity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    entity_jaccard_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    time_gap_minutes: float = Field(default=30.0, ge=1.0, le=1440.0)
    soft_gap_minutes: float = Field(default=10.0, ge=0.0, le=1440.0)
    coreference_enabled: bool = False
    coreference_model: str = "biu-nlp/f-coref"
    use_topic_shift_markers: bool = True
    topic_shift_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    min_segment_messages: int = Field(default=1, ge=1, le=100)
    max_segment_messages: int = Field(default=50, ge=5, le=500)
    boundary_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    forward_context_window: int = Field(default=2, ge=1, le=10)
    forward_continuity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class NormalizationConfig(BaseModel):
    """Normalization profiles for different tasks."""

    extraction: NormalizationProfile = Field(
        default_factory=lambda: _normalization_profile(
            normalize_emojis=False,  # Keep original emojis (LLM understands them)
            ner_enabled=False,
            spell_check=False,  # Disabled: mangles entity names (mom→mon, Xbox→box)
            min_length=1,
            max_length=2000,
        )
    )
    classification: NormalizationProfile = Field(
        default_factory=lambda: _normalization_profile(
            normalize_emojis=True,
            ner_enabled=True,
            spell_check=True,
            min_length=1,
            max_length=1000,
        )
    )
    chunk_embedding: NormalizationProfile = Field(
        default_factory=lambda: _normalization_profile(
            normalize_emojis=True,
            ner_enabled=False,
            spell_check=True,
            min_length=1,
            max_length=2000,
        )
    )
    topic_modeling: NormalizationProfile = Field(
        default_factory=lambda: _normalization_profile(
            normalize_emojis=True,
            ner_enabled=True,
            spell_check=False,
            min_length=3,
            max_length=1000,
        )
    )


class ServerConfig(BaseModel):
    """Socket server configuration.

    Attributes:
        websocket_host: Host address for the WebSocket server.
        cors_origins: Allowed CORS origins for WebSocket connections.
    """

    websocket_host: str = "127.0.0.1"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["tauri://localhost", "http://localhost", "http://127.0.0.1"]
    )


class NERConfig(BaseModel):
    """NER client configuration.

    Attributes:
        connect_timeout: Timeout in seconds for connecting to the NER service.
        read_timeout: Timeout in seconds for reading NER service responses.
    """

    connect_timeout: float = Field(default=2.0, ge=0.1, le=30.0)
    read_timeout: float = Field(default=10.0, ge=0.1, le=120.0)


class JarvisConfig(BaseModel):
    """JARVIS configuration schema.

    Attributes:
        config_version: Schema version for migration tracking.
        model_path: HuggingFace model path for MLX inference (deprecated, use model.model_id).
        memory_thresholds: Memory thresholds for mode selection.
        imessage_default_limit: Default limit for iMessage search (deprecated).
        ui: UI preferences for the Tauri frontend.
        search: Search preferences.
        chat: Chat preferences.
        routing: Routing thresholds and A/B configuration.
        model: Model configuration for text generation.
        rate_limit: Rate limiting configuration for the API.
        task_queue: Task queue configuration for background operations.
        metrics: Metrics collection configuration (enable/disable, batching).
        digest: Digest generation preferences.
        classifier_thresholds: Centralized classifier threshold configuration.
        server: Socket server configuration (WebSocket host, CORS origins).
        ner: NER client configuration (timeouts).
    """

    config_version: int = CONFIG_VERSION
    model_path: str = "models/lfm-0.7b-4bit"
    memory_thresholds: MemoryThresholds = Field(default_factory=MemoryThresholds)
    imessage_default_limit: int = 50
    ui: UIConfig = Field(default_factory=UIConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    model: ModelSettings = Field(default_factory=ModelSettings)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    task_queue: TaskQueueConfig = Field(default_factory=TaskQueueConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    digest: DigestConfig = Field(default_factory=DigestConfig)
    classifier_thresholds: ClassifierThresholds = Field(default_factory=ClassifierThresholds)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vec_search: VecSearchConfig = Field(default_factory=VecSearchConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    ner: NERConfig = Field(default_factory=NERConfig)


# Module-level singleton with thread safety
_config: JarvisConfig | None = None
_config_lock = threading.Lock()


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v1 to v2: Add ui/search/chat sections."""
    # Migrate imessage_default_limit to search.default_limit if not already set
    if "search" not in data:
        data["search"] = {}
    if "default_limit" not in data["search"] and "imessage_default_limit" in data:
        data["search"]["default_limit"] = data["imessage_default_limit"]

    # Add default sections if missing
    if "ui" not in data:
        data["ui"] = {}
    if "chat" not in data:
        data["chat"] = {}

    return data


def _migrate_v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v2 to v3: Add model section."""
    # Add model section if missing
    if "model" not in data:
        data["model"] = {}

    # Migrate model_path to model.model_id if possible
    if "model_path" in data and "model_id" not in data["model"]:
        # Map known paths to model IDs
        path_to_id = {
            "mlx-community/Qwen2.5-0.5B-Instruct-4bit": "qwen-0.5b",
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit": "qwen-1.5b",
            "mlx-community/Qwen2.5-3B-Instruct-4bit": "qwen-3b",
        }
        model_path = data["model_path"]
        if model_path in path_to_id:
            data["model"]["model_id"] = path_to_id[model_path]

    return data


def _migrate_v3_to_v6(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v3 to v6: Add rate_limit, task_queue, digest sections."""
    # Add rate_limit section if missing
    if "rate_limit" not in data:
        data["rate_limit"] = {}

    # Add task_queue section if missing
    if "task_queue" not in data:
        data["task_queue"] = {}

    # Add digest section if missing
    if "digest" not in data:
        data["digest"] = {}

    return data


def _migrate_v6_to_v7(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v6 to v7: Add routing section."""
    if "routing" not in data:
        data["routing"] = {}

    return data


def _migrate_v7_to_v8(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v7 to v8: Migrate template_similarity_threshold."""
    # Migrate template_similarity_threshold to routing.quick_reply_threshold
    # Only migrate if legacy field has a non-default value (0.7) and routing
    # section doesn't have an explicit quick_reply_threshold set
    legacy_threshold = data.get("template_similarity_threshold")
    if "routing" not in data:
        data["routing"] = {}

    routing = data["routing"]
    if (
        legacy_threshold is not None
        and legacy_threshold != 0.7
        and "quick_reply_threshold" not in routing
    ):
        logger.info(
            f"Migrating template_similarity_threshold={legacy_threshold} "
            f"to routing.quick_reply_threshold"
        )
        routing["quick_reply_threshold"] = legacy_threshold

    # Remove deprecated template_similarity_threshold field (no longer in schema)
    if "template_similarity_threshold" in data:
        del data["template_similarity_threshold"]

    # Also migrate template_threshold to quick_reply_threshold
    if "template_threshold" in routing and "quick_reply_threshold" not in routing:
        logger.info("Migrating routing.template_threshold to routing.quick_reply_threshold")
        routing["quick_reply_threshold"] = routing["template_threshold"]
        del routing["template_threshold"]

    return data


def _migrate_v8_to_v9(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v8 to v9: Migrate mlx_service_url to mlx_service_socket."""
    # Migrate mlx_service_url to mlx_service_socket
    if "embedding" not in data:
        data["embedding"] = {}

    embedding = data["embedding"]
    if "mlx_service_url" in embedding:
        # Remove the old HTTP URL field - new socket path will use default
        logger.info("Removing deprecated mlx_service_url, using Unix socket instead")
        del embedding["mlx_service_url"]

    # Update socket path from /tmp to ~/.jarvis for security
    legacy_tmp_socket = str(Path(tempfile.gettempdir()) / "jarvis-embed.sock")
    if "mlx_service_socket" in embedding and embedding["mlx_service_socket"] == legacy_tmp_socket:
        new_socket_path = str(Path.home() / ".jarvis" / "jarvis-embed.sock")
        logger.info(f"Migrating socket path from /tmp to {new_socket_path}")
        embedding["mlx_service_socket"] = new_socket_path

    return data


def _migrate_v9_to_v10(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v9 to v10: Add retrieval section."""
    # Add retrieval section if missing
    if "retrieval" not in data:
        data["retrieval"] = {}

    return data


def _migrate_v10_to_v11(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v10 to v11: Add segmentation section."""
    # Add segmentation section if missing
    if "segmentation" not in data:
        data["segmentation"] = {}

    return data


def _migrate_v11_to_v12(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v11 to v12: Migrate faiss_index to vec_search."""
    # Migrate faiss_index -> vec_search
    if "faiss_index" in data:
        logger.info("Removing deprecated faiss_index config, using vec_search defaults")
        del data["faiss_index"]
    if "vec_search" not in data:
        data["vec_search"] = {}

    return data


def _migrate_v12_to_v13(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from v12 to v13: No-op (socket path migration done in v8->v9)."""
    return data


# Migration registry mapping target versions to migration functions
_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    2: _migrate_v1_to_v2,
    3: _migrate_v2_to_v3,
    6: _migrate_v3_to_v6,
    7: _migrate_v6_to_v7,
    8: _migrate_v7_to_v8,
    9: _migrate_v8_to_v9,
    10: _migrate_v9_to_v10,
    11: _migrate_v10_to_v11,
    12: _migrate_v11_to_v12,
    13: _migrate_v12_to_v13,
}


def _migrate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate config data from older versions to current schema.

    Preserves existing values while adding new defaults for missing fields.
    Uses a migration registry pattern to apply version-specific migrations.

    Args:
        data: Raw config data loaded from file.

    Returns:
        Migrated config data compatible with current schema.
    """
    version = data.get("config_version", 1)

    # Apply migrations sequentially
    for target_version in sorted(_MIGRATIONS.keys()):
        if version < target_version:
            logger.info(f"Migrating config from version {version} to {target_version}")
            data = _MIGRATIONS[target_version](data)
            version = target_version

    # Update to current version
    data["config_version"] = CONFIG_VERSION

    return data


def load_config(config_path: Path | None = None) -> JarvisConfig:
    """Load configuration from file, return defaults if missing/invalid.

    Automatically migrates older config versions while preserving existing values.
    If migration occurs, the updated config is saved back to disk.

    Args:
        config_path: Optional path to config file. Defaults to ~/.jarvis/config.json.

    Returns:
        JarvisConfig instance with loaded or default values.
    """
    path = config_path or CONFIG_PATH

    if not path.exists():
        logger.debug(f"Config file not found at {path}, using defaults")
        return JarvisConfig()

    try:
        with path.open() as f:
            data: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in config file {path}: {e}, using defaults")
        return JarvisConfig()
    except OSError as e:
        logger.warning(f"Cannot read config file {path}: {e}, using defaults")
        return JarvisConfig()

    # Track original version for migration detection
    original_version = data.get("config_version", 1)

    # Migrate from older versions
    data = _migrate_config(data)

    try:
        config = JarvisConfig.model_validate(data)

        # Persist migrated config so migration doesn't run on every startup
        if original_version < CONFIG_VERSION:
            logger.info(f"Persisting migrated config (v{original_version} -> v{CONFIG_VERSION})")
            save_config(config, path)

        return config
    except ValidationError as e:
        logger.warning(f"Config validation failed: {e}, using defaults")
        return JarvisConfig()


def save_config(config: JarvisConfig, config_path: Path | None = None) -> bool:
    """Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to config file. Defaults to ~/.jarvis/config.json.

    Returns:
        True if saved successfully, False otherwise.
    """
    path = config_path or CONFIG_PATH

    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write config with pretty formatting
        with path.open("w") as f:
            json.dump(config.model_dump(), f, indent=2)

        # Restrict permissions to owner-only (config may contain sensitive settings)
        os.chmod(path, 0o600)

        logger.debug(f"Configuration saved to {path}")
        return True

    except OSError as e:
        logger.error(f"Failed to save config to {path}: {e}")
        return False


def get_config() -> JarvisConfig:
    """Get singleton configuration instance.

    Uses double-check locking for thread safety.

    Returns:
        Shared JarvisConfig instance.
    """
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
    return _config


def reset_config() -> None:
    """Reset singleton configuration for testing."""
    global _config
    with _config_lock:
        _config = None


# =============================================================================
# Embedding Model Artifact Paths
# =============================================================================


def get_embedding_artifacts_dir() -> Path:
    """Get the directory for embedding model artifacts.

    Returns versioned path based on the configured embedding model:
        ~/.jarvis/embeddings/{model_name}/

    This allows multiple embedding models to coexist without conflicts.
    Each model has its own classifiers, centroids, and embeddings database.

    Returns:
        Path to the embedding artifacts directory for the current model.
    """
    config = get_config()
    base_dir = Path.home() / ".jarvis" / "embeddings" / config.embedding.model_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_category_classifier_path() -> Path:
    """Get path to the category classifier model directory."""
    return get_embedding_artifacts_dir() / "category_classifier_model"


def get_embeddings_db_path() -> Path:
    """Get path to the embeddings database.

    Returns:
        Path to embeddings.db for the current embedding model.
    """
    return get_embedding_artifacts_dir() / "embeddings.db"
