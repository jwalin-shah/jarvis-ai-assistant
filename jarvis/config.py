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
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".jarvis" / "config.json"

# Current config schema version for migration tracking
CONFIG_VERSION = 9


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


class RoutingConfig(BaseModel):
    """Routing thresholds and A/B testing configuration."""

    quick_reply_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    context_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    generate_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    ab_test_group: str = Field(default="control")
    ab_test_thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)
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

    model_id: str = "lfm-1.2b"
    auto_select: bool = True
    max_tokens_reply: int = Field(default=150, ge=1, le=2048)
    max_tokens_summary: int = Field(default=500, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    generation_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    idle_timeout_seconds: float = Field(default=300.0, ge=0.0, le=3600.0)
    warm_on_startup: bool = False


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

        response_centroid_verify: Minimum similarity to verify structural hints.
        response_centroid_margin: Margin for centroid override decisions.
        response_low_confidence: Below this, default to ANSWER for safety.
        response_decline_confidence: Higher threshold for DECLINE (frequently misclassified).
        response_defer_confidence: Higher threshold for DEFER predictions.
        response_agree_confidence: Higher threshold for AGREE predictions.

        trigger_svm_default: Default SVM threshold for trigger classification.
        trigger_svm_commitment: SVM threshold for COMMITMENT triggers.
        trigger_svm_question: SVM threshold for QUESTION triggers.
        trigger_svm_reaction: SVM threshold for REACTION triggers.
        trigger_svm_social: SVM threshold for SOCIAL triggers.
        trigger_svm_statement: SVM threshold for STATEMENT triggers.
        trigger_centroid_verify: Centroid verification threshold for triggers.
        trigger_centroid_margin: Centroid margin for trigger classification.
    """

    # Message classifier thresholds (from message_classifier.py)
    message_rule_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    message_embedding_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    message_rule_fallback: float = Field(default=0.70, ge=0.0, le=1.0)

    # Intent classifier thresholds (from intent.py)
    intent_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    intent_quick_reply: float = Field(default=0.80, ge=0.0, le=1.0)

    # Response classifier thresholds (from response_classifier.py)
    response_centroid_verify: float = Field(default=0.50, ge=0.0, le=1.0)
    response_centroid_margin: float = Field(default=0.10, ge=0.0, le=1.0)
    response_low_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    response_decline_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    response_defer_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    response_agree_confidence: float = Field(default=0.80, ge=0.0, le=1.0)

    # Trigger classifier thresholds (from trigger_classifier.py)
    trigger_svm_default: float = Field(default=0.35, ge=0.0, le=1.0)
    trigger_svm_commitment: float = Field(default=0.50, ge=0.0, le=1.0)
    trigger_svm_question: float = Field(default=0.35, ge=0.0, le=1.0)
    trigger_svm_reaction: float = Field(default=0.40, ge=0.0, le=1.0)
    trigger_svm_social: float = Field(default=0.25, ge=0.0, le=1.0)
    trigger_svm_statement: float = Field(default=0.40, ge=0.0, le=1.0)
    trigger_centroid_verify: float = Field(default=0.40, ge=0.0, le=1.0)
    trigger_centroid_margin: float = Field(default=0.15, ge=0.0, le=1.0)


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
    mlx_service_socket: str = "/tmp/jarvis-embed.sock"


class FAISSIndexConfig(BaseModel):
    """FAISS index configuration for vector search.

    Controls index type and compression level based on benchmarks from
    docs/improvements.md Appendix A (148K real messages, bge-small 384-dim).

    Index Types:
        - "flat": IndexFlatIP brute force. 100% recall, highest memory (~586MB/400K).
        - "ivfpq_4x": IVFPQ 384x8. 92% recall, 4x compression (~155MB/400K). DEFAULT.
        - "ivfpq_8x": IVFPQ 192x8. 88% recall, 8x compression (~81MB/400K).
        - "ivf": IndexIVFFlat. 93% recall, no compression (same size as flat).

    Attributes:
        index_type: Type of FAISS index to build.
        ivf_nprobe: Number of clusters to search (higher = more accurate, slower).
        pq_training_ratio: Fraction of vectors to use for PQ training (0.1-1.0).
        min_vectors_for_compression: Minimum vectors before using compression.
            Below this threshold, flat index is used regardless of index_type.
    """

    index_type: Literal["flat", "ivf", "ivfpq_4x", "ivfpq_8x"] = "ivfpq_4x"
    ivf_nprobe: int = Field(default=128, ge=1, le=512)
    pq_training_ratio: float = Field(default=1.0, ge=0.1, le=1.0)
    min_vectors_for_compression: int = Field(default=1000, ge=100, le=50000)


class JarvisConfig(BaseModel):
    """JARVIS configuration schema.

    Attributes:
        config_version: Schema version for migration tracking.
        model_path: HuggingFace model path for MLX inference (deprecated, use model.model_id).
        template_similarity_threshold: DEPRECATED - use routing.quick_reply_threshold instead.
            Kept for backwards compatibility. Non-default values are migrated to
            routing.quick_reply_threshold during config load.
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
    """

    config_version: int = CONFIG_VERSION
    model_path: str = "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit"
    template_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
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
    faiss_index: FAISSIndexConfig = Field(default_factory=FAISSIndexConfig)


# Module-level singleton with thread safety
_config: JarvisConfig | None = None
_config_lock = threading.Lock()


def _migrate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate config data from older versions to current schema.

    Preserves existing values while adding new defaults for missing fields.
    Handles migration from:
    - v1 (no version field) to v2 (with ui/search/chat sections)
    - v2 to v3 (with model section)
    - v3 to v4 (with task_queue section)

    Args:
        data: Raw config data loaded from file.

    Returns:
        Migrated config data compatible with current schema.
    """
    version = data.get("config_version", 1)

    if version < 2:
        logger.info(f"Migrating config from version {version} to {CONFIG_VERSION}")

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

        version = 2

    if version < 3:
        logger.info(f"Migrating config from version {version} to {CONFIG_VERSION}")

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

        version = 3

    if version < 6:
        logger.info(f"Migrating config from version {version} to {CONFIG_VERSION}")

        # Add rate_limit section if missing
        if "rate_limit" not in data:
            data["rate_limit"] = {}

        # Add task_queue section if missing
        if "task_queue" not in data:
            data["task_queue"] = {}

        # Add digest section if missing
        if "digest" not in data:
            data["digest"] = {}

        version = 6

    if version < 7:
        logger.info(f"Migrating config from version {version} to {CONFIG_VERSION}")

        if "routing" not in data:
            data["routing"] = {}

        version = 7

    if version < 8:
        logger.info(f"Migrating config from version {version} to {CONFIG_VERSION}")

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

        # Also migrate template_threshold to quick_reply_threshold
        if "template_threshold" in routing and "quick_reply_threshold" not in routing:
            logger.info("Migrating routing.template_threshold to routing.quick_reply_threshold")
            routing["quick_reply_threshold"] = routing["template_threshold"]
            del routing["template_threshold"]

        version = 8

    if version < 9:
        logger.info(f"Migrating config from version {version} to {CONFIG_VERSION}")

        # Migrate mlx_service_url to mlx_service_socket
        if "embedding" not in data:
            data["embedding"] = {}

        embedding = data["embedding"]
        if "mlx_service_url" in embedding:
            # Remove the old HTTP URL field - new socket path will use default
            logger.info("Removing deprecated mlx_service_url, using Unix socket instead")
            del embedding["mlx_service_url"]

        version = 9

    # Update version
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


def get_trigger_classifier_path() -> Path:
    """Get path to the trigger classifier model directory.

    Returns:
        Path to trigger_classifier_model/ for the current embedding model.
    """
    return get_embedding_artifacts_dir() / "trigger_classifier_model"


def get_response_classifier_path() -> Path:
    """Get path to the response classifier model directory.

    Returns:
        Path to response_classifier_model/ for the current embedding model.
    """
    return get_embedding_artifacts_dir() / "response_classifier_model"


def get_embeddings_db_path() -> Path:
    """Get path to the embeddings database.

    Returns:
        Path to embeddings.db for the current embedding model.
    """
    return get_embedding_artifacts_dir() / "embeddings.db"
