"""Quality assurance system for JARVIS.

Provides comprehensive quality metrics, hallucination detection, and real-time
quality gates for response generation.

Modules:
    hallucination - Multi-model ensemble hallucination detection
    factuality - Fact verification against message history
    consistency - Self-consistency checking across responses
    grounding - Source attribution tracking
    dimensions - Quality dimension scoring (factual, coherence, relevance, etc.)
    gates - Real-time quality gates with configurable thresholds
    dashboard - Quality metrics tracking and regression detection
    feedback - Feedback integration and learning from user edits
"""

from jarvis.quality.consistency import (
    ConsistencyChecker,
    ConsistencyResult,
    get_consistency_checker,
)
from jarvis.quality.dashboard import (
    QualityAlert,
    QualityDashboard,
    QualityTrend,
    get_quality_dashboard,
)
from jarvis.quality.dimensions import (
    CoherenceScorer,
    FactualScorer,
    LengthScorer,
    PersonalizationScorer,
    QualityDimension,
    QualityDimensionResult,
    QualityDimensionScorer,
    RelevanceScorer,
    ToneScorer,
)
from jarvis.quality.factuality import (
    Claim,
    FactChecker,
    FactualityResult,
    get_fact_checker,
)
from jarvis.quality.feedback import (
    FeedbackCollector,
    FeedbackEntry,
    FeedbackStats,
    get_feedback_collector,
)
from jarvis.quality.gates import (
    GateDecision,
    QualityGate,
    QualityGateConfig,
    QualityGateResult,
    get_quality_gate,
)
from jarvis.quality.grounding import (
    Attribution,
    GroundingChecker,
    GroundingResult,
    get_grounding_checker,
)
from jarvis.quality.hallucination import (
    EnsembleHallucinationDetector,
    HallucinationResult,
    get_hallucination_detector,
)

__all__ = [
    # Hallucination detection
    "EnsembleHallucinationDetector",
    "HallucinationResult",
    "get_hallucination_detector",
    # Factuality checking
    "FactChecker",
    "FactualityResult",
    "Claim",
    "get_fact_checker",
    # Consistency checking
    "ConsistencyChecker",
    "ConsistencyResult",
    "get_consistency_checker",
    # Grounding/Attribution
    "GroundingChecker",
    "GroundingResult",
    "Attribution",
    "get_grounding_checker",
    # Quality dimensions
    "QualityDimension",
    "QualityDimensionResult",
    "QualityDimensionScorer",
    "FactualScorer",
    "CoherenceScorer",
    "RelevanceScorer",
    "ToneScorer",
    "LengthScorer",
    "PersonalizationScorer",
    # Quality gates
    "QualityGate",
    "QualityGateConfig",
    "QualityGateResult",
    "GateDecision",
    "get_quality_gate",
    # Dashboard
    "QualityDashboard",
    "QualityTrend",
    "QualityAlert",
    "get_quality_dashboard",
    # Feedback
    "FeedbackCollector",
    "FeedbackEntry",
    "FeedbackStats",
    "get_feedback_collector",
]
