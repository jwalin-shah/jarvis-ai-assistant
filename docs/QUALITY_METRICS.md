# Quality Metrics Guide

Comprehensive quality assurance system for JARVIS response generation.

## Overview

The quality module provides multi-dimensional assessment of generated responses:

- **Hallucination Detection**: Multi-model ensemble for detecting fabricated content
- **Factuality Checking**: Verification against conversation context
- **Consistency Checking**: Self-consistency and history consistency
- **Source Attribution**: Grounding responses in source material
- **Quality Dimensions**: Factual, coherence, relevance, tone, length, personalization
- **Real-time Gates**: Pre-send quality validation with configurable thresholds
- **Dashboard**: Metrics tracking, trend analysis, regression detection
- **Feedback Integration**: Learning from user edits and ratings

## Architecture

```
jarvis/quality/
├── __init__.py          # Package exports
├── hallucination.py     # Ensemble hallucination detection
├── factuality.py        # Fact verification
├── consistency.py       # Self/history consistency
├── grounding.py         # Source attribution
├── dimensions.py        # Quality dimension scorers
├── gates.py             # Real-time quality gates
├── dashboard.py         # Metrics tracking
└── feedback.py          # Feedback integration
```

## Quick Start

```python
from jarvis.quality import get_quality_gate, get_quality_dashboard

# Check response quality
gate = get_quality_gate()
result = gate.check(
    response="Sure, I can help with that!",
    source="Can you help me with this project?",
)

if result.should_send:
    print(f"Quality score: {result.quality_score:.2f}")
else:
    print(f"Quality issues: {result.all_issues}")

# Fast check for real-time use (<50ms)
fast_result = gate.check_fast(response, source)
```

## Quality Dimensions

### 1. Hallucination Detection

Detects fabricated or ungrounded content using ensemble methods:

```python
from jarvis.quality import get_hallucination_detector

detector = get_hallucination_detector()
result = detector.detect(
    source="Meeting at noon tomorrow",
    response="The meeting with John is at noon in the 5th floor conference room",
)

print(f"Hallucination score: {result.hallucination_score:.2f}")
print(f"Severity: {result.severity.value}")
print(f"Issues: {result.issues}")
```

**Ensemble Components:**
- **HHEM**: Vectara Hallucination Evaluation Model (CrossEncoder)
- **NLI**: Natural Language Inference entailment checking
- **Semantic Similarity**: Embedding-based similarity
- **Token Overlap**: Fast keyword-based check

**Configuration:**
```python
from jarvis.quality.hallucination import EnsembleHallucinationDetector

detector = EnsembleHallucinationDetector(
    gate_threshold=0.5,      # Max hallucination score to pass
    enable_hhem=True,        # Enable HHEM (slow but accurate)
    enable_nli=True,         # Enable NLI check
    enable_similarity=True,  # Enable embedding similarity
    enable_overlap=True,     # Enable fast token overlap
)
```

### 2. Factuality Checking

Verifies claims against conversation context:

```python
from jarvis.quality import get_fact_checker

checker = get_fact_checker()
result = checker.check_factuality(
    response="The deadline is Friday at 5pm.",
    context=["Project deadline: end of week", "Review meeting Friday"],
)

print(f"Factuality score: {result.factuality_score:.2f}")
print(f"Verified claims: {result.verified_count}")
print(f"Refuted claims: {result.refuted_count}")
```

### 3. Consistency Checking

Ensures internal and historical consistency:

```python
from jarvis.quality import get_consistency_checker

checker = get_consistency_checker()
result = checker.check_consistency(
    response="I'll be there at noon. Actually, I can't make noon.",
    history=["Let's meet at noon", "See you tomorrow!"],
)

print(f"Consistency score: {result.consistency_score:.2f}")
print(f"Self-consistent: {result.is_self_consistent}")
print(f"History-consistent: {result.is_history_consistent}")
```

### 4. Source Attribution

Tracks which parts of responses are grounded:

```python
from jarvis.quality import get_grounding_checker

checker = get_grounding_checker()
result = checker.check_grounding(
    response="The meeting is at noon. I'll bring the report.",
    sources=["Meeting scheduled for noon today"],
)

print(f"Grounding score: {result.grounding_score:.2f}")
print(f"Direct quotes: {result.direct_quote_count}")
print(f"Paraphrases: {result.paraphrase_count}")
print(f"Ungrounded: {result.ungrounded_count}")
```

### 5. Quality Dimensions

Score responses across multiple dimensions:

```python
from jarvis.quality.dimensions import MultiDimensionScorer, QualityDimension

scorer = MultiDimensionScorer()
result = scorer.score_all(
    response="Thank you for reaching out. I'd be happy to help!",
    context="Can you assist me with this?",
)

print(f"Overall score: {result.overall_score:.2f}")
for dim, dim_result in result.results.items():
    print(f"  {dim.value}: {dim_result.score:.2f}")
```

**Dimensions:**
- **Factual**: Grounded in conversation context
- **Coherence**: Logical flow and clarity
- **Relevance**: Addresses the query
- **Tone**: Appropriate formality/empathy
- **Length**: Appropriate for context
- **Personalization**: Tailored to recipient

## Quality Gates

### Basic Usage

```python
from jarvis.quality import get_quality_gate, QualityGateConfig

# Default configuration
gate = get_quality_gate()
result = gate.check(response, source)

# Strict configuration
config = QualityGateConfig.strict()
strict_gate = QualityGate(config)

# Lenient configuration
config = QualityGateConfig.lenient()
lenient_gate = QualityGate(config)
```

### Gate Decisions

- **PASS**: Response passes all checks
- **SOFT_FAIL**: Marginal quality, warn but allow
- **HARD_FAIL**: Quality too low, block/rewrite

### Gate Actions

- **NONE**: No action needed
- **WARN**: Show warning to user
- **SUGGEST_EDIT**: Suggest specific edits
- **AUTO_REWRITE**: Automatically rewrite (if enabled)
- **BLOCK**: Block response entirely

### Custom Configuration

```python
config = QualityGateConfig(
    # Hallucination gate
    hallucination_enabled=True,
    hallucination_threshold=0.5,
    hallucination_soft_threshold=0.7,

    # Factuality gate
    factuality_enabled=True,
    factuality_threshold=0.6,

    # Auto-rewrite settings
    auto_rewrite_enabled=False,
    auto_rewrite_max_attempts=2,

    # Weights for overall score
    weights={
        "hallucination": 0.25,
        "factuality": 0.20,
        "consistency": 0.15,
        "grounding": 0.15,
        "coherence": 0.15,
        "relevance": 0.10,
    },
)
```

## Quality Dashboard

### Tracking Metrics

```python
from jarvis.quality import get_quality_dashboard

dashboard = get_quality_dashboard()

# Record a quality check
dashboard.record_quality_check(
    dimension_scores={"coherence": 0.8, "relevance": 0.9},
    overall_score=0.85,
    model_name="llama-3.2",
    latency_ms=50.0,
)

# Get summary
summary = dashboard.get_summary()
print(f"Total checks: {summary['total_checks']}")
print(f"Dimensions: {summary['dimensions']}")
```

### Trend Analysis

```python
# Get quality trends over time
trends = dashboard.get_trends(days=7)
for trend in trends:
    print(f"{trend.dimension}: {trend.change_percent:+.1f}% ({trend.trend_direction})")
```

### Model Comparison

```python
# Compare quality across models
comparisons = dashboard.get_model_comparison()
for comp in comparisons:
    print(f"#{comp.ranking}: {comp.model_name} - {comp.overall_score:.2f}")
```

### Regression Detection

```python
# Detect quality regression
regression, change = dashboard.detect_regression(dimension="overall")
if regression:
    print(f"Quality regression detected: {change:.1f}%")
```

### Alerts

```python
# Get quality alerts
alerts = dashboard.get_alerts(severity=AlertSeverity.CRITICAL)
for alert in alerts:
    print(f"[{alert.severity.value}] {alert.message}")
```

## Feedback Integration

### Recording Feedback

```python
from jarvis.quality import get_feedback_collector

collector = get_feedback_collector()

# User accepted suggestion as-is
collector.record_acceptance(
    original_text="Sure, I can help!",
    quality_scores={"overall": 0.9},
    contact_id="contact_123",
)

# User edited before sending
entry = collector.record_edit(
    original_text="This is a long detailed response.",
    edited_text="Short reply.",
    contact_id="contact_123",
)
print(f"Edit type: {entry.edit_type.value}")
print(f"Edit distance: {entry.edit_distance}")

# User rejected suggestion
collector.record_rejection(
    original_text="Rejected response.",
    quality_scores={"overall": 0.3},
)

# User rated suggestion
collector.record_rating(
    original_text="Great response!",
    rating=5,  # 1-5 scale
)
```

### Analyzing Feedback

```python
# Get feedback statistics
stats = collector.get_stats()
print(f"Acceptance rate: {stats.acceptance_rate:.1%}")
print(f"Average edit distance: {stats.avg_edit_distance:.1f}")
print(f"Average rating: {stats.avg_rating}")

# Get learned contact preferences
preferences = collector.get_contact_preferences("contact_123")
print(f"Length preference: {preferences.get('length_preference')}")
print(f"Tone preference: {preferences.get('tone_preference')}")

# Get quality score calibration
calibration = collector.get_calibration_factor("coherence")
print(f"Coherence calibration factor: {calibration:.2f}")
```

## Benchmarking

### Running Benchmarks

```bash
# Run standard benchmark
uv run python -m benchmarks.quality_benchmark run --dataset standard

# Compare two models
uv run python -m benchmarks.quality_benchmark compare --model-a default --model-b new

# Generate human evaluation batch
uv run python -m benchmarks.quality_benchmark generate-eval --name batch_1
```

### Programmatic Benchmarking

```python
from benchmarks.quality_benchmark import (
    QualityBenchmark,
    StandardDataset,
    ABTestFramework,
)

# Run benchmark
benchmark = QualityBenchmark(model_name="my_model")
samples = StandardDataset.get_samples()
report = benchmark.run_benchmark(samples)

print(f"Pass rate: {report.pass_rate:.1%}")
print(f"Mean quality: {report.mean_scores['overall']:.2f}")
print(f"P95 latency: {report.p95_latency_ms:.1f}ms")

# A/B test
ab_framework = ABTestFramework()
result = ab_framework.compare_models("model_a", "model_b")
print(f"Winner: {result.winner} (confidence: {result.confidence:.1%})")
```

### Regression Testing

```python
# Load baseline report
baseline = BenchmarkReport(...)  # From saved file

# Run regression test
passed, changes = benchmark.run_regression_test(
    baseline_report=baseline,
    regression_threshold=0.05,  # 5% decline threshold
)

if not passed:
    print("Regression detected!")
    for dim, change in changes.items():
        if change < -0.05:
            print(f"  {dim}: {change*100:.1f}%")
```

## Tuning Thresholds

### General Guidelines

1. **Start lenient, tighten gradually** - Begin with QualityGateConfig.lenient() and adjust based on user feedback

2. **Monitor acceptance rates** - If acceptance rate drops below 60%, thresholds may be too strict

3. **Balance false positives vs negatives**:
   - High threshold = more false negatives (bad responses slip through)
   - Low threshold = more false positives (good responses blocked)

4. **Use feedback calibration** - The feedback collector learns actual quality from user behavior

### Recommended Thresholds by Use Case

| Use Case | Hallucination | Factuality | Coherence | Relevance |
|----------|--------------|------------|-----------|-----------|
| Customer Service | 0.4 | 0.7 | 0.6 | 0.6 |
| Casual Chat | 0.6 | 0.5 | 0.5 | 0.5 |
| Professional Email | 0.4 | 0.7 | 0.7 | 0.7 |
| Quick Replies | 0.7 | 0.4 | 0.4 | 0.5 |

### Monitoring and Adjustment

1. Track gate pass rates over time
2. Monitor user edit patterns
3. Analyze rejection reasons
4. Use A/B testing for threshold changes
5. Calibrate scores using feedback data

## Adding New Quality Dimensions

### 1. Create Scorer Class

```python
from jarvis.quality.dimensions import QualityDimensionScorer, QualityDimension

class CustomScorer(QualityDimensionScorer):
    dimension = QualityDimension.CUSTOM  # Add new enum value
    default_threshold = 0.5

    def score(
        self,
        response: str,
        context: str | None = None,
        **kwargs,
    ) -> QualityDimensionResult:
        # Implement scoring logic
        score = self._compute_score(response, context)

        return QualityDimensionResult(
            dimension=self.dimension,
            score=score,
            issues=self._identify_issues(response),
            suggestions=self._generate_suggestions(response),
        )
```

### 2. Register in MultiDimensionScorer

```python
class MultiDimensionScorer:
    def __init__(self):
        self._scorers = {
            # ... existing scorers
            QualityDimension.CUSTOM: CustomScorer(),
        }
```

### 3. Add Gate Check (Optional)

```python
class QualityGate:
    def _check_custom(self, response, context):
        # Implement gate check
        pass
```

## Performance Targets

- **Quality check latency**: <100ms (fast check <50ms)
- **Hallucination detection rate**: 95%+
- **False positive rate**: <10%
- **Gate pass rate**: 70-80% (adjustable)

## API Reference

See `jarvis/quality/__init__.py` for complete exports:

```python
from jarvis.quality import (
    # Hallucination
    EnsembleHallucinationDetector,
    HallucinationResult,
    get_hallucination_detector,

    # Factuality
    FactChecker,
    FactualityResult,
    get_fact_checker,

    # Consistency
    ConsistencyChecker,
    ConsistencyResult,
    get_consistency_checker,

    # Grounding
    GroundingChecker,
    GroundingResult,
    get_grounding_checker,

    # Dimensions
    QualityDimension,
    QualityDimensionScorer,
    MultiDimensionScorer,

    # Gates
    QualityGate,
    QualityGateConfig,
    QualityGateResult,
    GateDecision,
    get_quality_gate,

    # Dashboard
    QualityDashboard,
    get_quality_dashboard,

    # Feedback
    FeedbackCollector,
    get_feedback_collector,
)
```
