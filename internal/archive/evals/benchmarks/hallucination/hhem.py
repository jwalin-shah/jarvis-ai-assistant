"""HHEM hallucination evaluator implementation.  # noqa: E501
  # noqa: E501
Workstream 2: HHEM Hallucination Benchmark  # noqa: E501
  # noqa: E501
Implements the HallucinationEvaluator protocol from contracts/hallucination.py  # noqa: E501
using the vectara/hallucination_evaluation_model from HuggingFace.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
import logging  # noqa: E501
from datetime import UTC, datetime  # noqa: E402  # noqa: E501
from statistics import mean, median, stdev  # noqa: E402  # noqa: E501
from typing import Any, Protocol, runtime_checkable  # noqa: E402  # noqa: E501

# noqa: E501
from jarvis.contracts.hallucination import (  # noqa: E402  # noqa: E501
    HHEMBenchmarkResult,
    HHEMResult,
)

  # noqa: E501
logger = logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
# Batch size for efficient HHEM evaluation  # noqa: E501
_BATCH_SIZE = 16  # noqa: E501
  # noqa: E501
  # noqa: E501
@runtime_checkable  # noqa: E501
class HHEMModelProtocol(Protocol):  # noqa: E501
    """Protocol for HHEM model to enable mocking in tests."""  # noqa: E501
  # noqa: E501
    def predict(self, pairs: list[list[str]]) -> list[float]:  # noqa: E501
        """Predict hallucination scores for source/summary pairs.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            pairs: List of [source, summary] pairs  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            List of scores from 0 (hallucinated) to 1 (grounded)  # noqa: E501
        """  # noqa: E501
        ...  # noqa: E501
  # noqa: E501
  # noqa: E501
class VectaraHHEMModel:  # noqa: E501
    """Wrapper for the Vectara HHEM model from HuggingFace.  # noqa: E501
  # noqa: E501
    Uses vectara/hallucination_evaluation_model via sentence-transformers.  # noqa: E501
    """  # noqa: E501
  # noqa: E501
    def __init__(self) -> None:  # noqa: E501
        """Initialize the HHEM model wrapper."""  # noqa: E501
        self._model: Any = None  # noqa: E501
  # noqa: E501
    def _ensure_model(self) -> Any:  # noqa: E501
        """Lazy load the HHEM model.  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            The loaded CrossEncoder model  # noqa: E501
        """  # noqa: E501
        if self._model is None:  # noqa: E501
            logger.info("Loading HHEM model: vectara/hallucination_evaluation_model")  # noqa: E501
            try:  # noqa: E501
                from sentence_transformers import CrossEncoder  # noqa: E501
  # noqa: E501
                self._model = CrossEncoder("vectara/hallucination_evaluation_model")  # noqa: E501
            except Exception as e:  # noqa: E501
                logger.error("Failed to load HHEM model: %s", e)  # noqa: E501
                raise  # noqa: E501
        return self._model  # noqa: E501
  # noqa: E501
    def predict(self, pairs: list[list[str]]) -> list[float]:  # noqa: E501
        """Predict hallucination scores for source/summary pairs.  # noqa: E501
  # noqa: E501
        The model scores text from 0 (hallucinated) to 1 (grounded).  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            pairs: List of [source, summary] pairs where source is the  # noqa: E501
                   original text and summary is the generated summary.  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            List of scores from 0 (hallucinated) to 1 (grounded)  # noqa: E501
        """  # noqa: E501
        if not pairs:  # noqa: E501
            return []  # noqa: E501
  # noqa: E501
        model = self._ensure_model()  # noqa: E501
        # CrossEncoder.predict returns numpy array of scores  # noqa: E501
        scores = model.predict(pairs)  # noqa: E501
  # noqa: E501
        # Ensure we return a list of floats  # noqa: E501
        return [float(s) for s in scores]  # noqa: E501
  # noqa: E501
  # noqa: E501
class HHEMEvaluator:  # noqa: E501
    """HHEM-based hallucination evaluator.  # noqa: E501
  # noqa: E501
    Implements HallucinationEvaluator protocol from contracts/hallucination.py.  # noqa: E501
  # noqa: E501
    Uses the vectara/hallucination_evaluation_model to score text on a scale  # noqa: E501
    from 0 (hallucinated) to 1 (grounded).  # noqa: E501
    """  # noqa: E501
  # noqa: E501
    def __init__(self, model: HHEMModelProtocol | None = None) -> None:  # noqa: E501
        """Initialize the evaluator.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model: Optional HHEM model instance for dependency injection.  # noqa: E501
                   If not provided, uses the real Vectara model.  # noqa: E501
        """  # noqa: E501
        self._model: HHEMModelProtocol = model if model is not None else VectaraHHEMModel()  # noqa: E501
  # noqa: E501
    def evaluate_single(self, source: str, summary: str) -> float:  # noqa: E501
        """Return HHEM score for a source/summary pair.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            source: Original source text  # noqa: E501
            summary: Generated summary to evaluate  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            Score from 0 (hallucinated) to 1 (grounded)  # noqa: E501
        """  # noqa: E501
        scores = self._model.predict([[source, summary]])  # noqa: E501
        return scores[0] if scores else 0.0  # noqa: E501
  # noqa: E501
    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[float]:  # noqa: E501
        """Batch evaluate multiple pairs. More efficient than single calls.  # noqa: E501
  # noqa: E501
        Processes pairs in batches of 16-32 for optimal performance.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            pairs: List of (source, summary) tuples  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            List of scores from 0 (hallucinated) to 1 (grounded)  # noqa: E501
        """  # noqa: E501
        if not pairs:  # noqa: E501
            return []  # noqa: E501
  # noqa: E501
        # Convert tuples to list format expected by model  # noqa: E501
        all_pairs = [[source, summary] for source, summary in pairs]  # noqa: E501
        all_scores: list[float] = []  # noqa: E501
  # noqa: E501
        # Process in batches for efficiency  # noqa: E501
        for i in range(0, len(all_pairs), _BATCH_SIZE):  # noqa: E501
            batch = all_pairs[i : i + _BATCH_SIZE]  # noqa: E501
            batch_scores = self._model.predict(batch)  # noqa: E501
            all_scores.extend(batch_scores)  # noqa: E501
  # noqa: E501
        return all_scores  # noqa: E501
  # noqa: E501
    def run_benchmark(  # noqa: E501
        self,  # noqa: E501
        model_name: str,  # noqa: E501
        dataset: list[tuple[str, str, str]],  # noqa: E501
        prompt_templates: list[str],  # noqa: E501
    ) -> HHEMBenchmarkResult:  # noqa: E501
        """Run full benchmark and return aggregate results.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model_name: Name of the model being evaluated  # noqa: E501
            dataset: List of (source_text, generated_summary, template_name) tuples  # noqa: E501
            prompt_templates: List of prompt template names to filter by (or all if empty)  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            HHEMBenchmarkResult with aggregate statistics  # noqa: E501
        """  # noqa: E501
        if not dataset:  # noqa: E501
            return HHEMBenchmarkResult(  # noqa: E501
                model_name=model_name,  # noqa: E501
                num_samples=0,  # noqa: E501
                mean_score=0.0,  # noqa: E501
                median_score=0.0,  # noqa: E501
                std_score=0.0,  # noqa: E501
                pass_rate_at_05=0.0,  # noqa: E501
                pass_rate_at_07=0.0,  # noqa: E501
                results=[],  # noqa: E501
                timestamp=datetime.now(UTC).isoformat(),  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        # Filter by templates if specified  # noqa: E501
        if prompt_templates:  # noqa: E501
            filtered_dataset = [  # noqa: E501
                (src, summ, tpl) for src, summ, tpl in dataset if tpl in prompt_templates  # noqa: E501
            ]  # noqa: E501
        else:  # noqa: E501
            filtered_dataset = list(dataset)  # noqa: E501
  # noqa: E501
        if not filtered_dataset:  # noqa: E501
            return HHEMBenchmarkResult(  # noqa: E501
                model_name=model_name,  # noqa: E501
                num_samples=0,  # noqa: E501
                mean_score=0.0,  # noqa: E501
                median_score=0.0,  # noqa: E501
                std_score=0.0,  # noqa: E501
                pass_rate_at_05=0.0,  # noqa: E501
                pass_rate_at_07=0.0,  # noqa: E501
                results=[],  # noqa: E501
                timestamp=datetime.now(UTC).isoformat(),  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        # Prepare pairs for batch evaluation  # noqa: E501
        pairs = [(src, summ) for src, summ, _ in filtered_dataset]  # noqa: E501
        logger.info("Evaluating %d source/summary pairs", len(pairs))  # noqa: E501
  # noqa: E501
        # Evaluate all pairs  # noqa: E501
        scores = self.evaluate_batch(pairs)  # noqa: E501
  # noqa: E501
        # Create individual results  # noqa: E501
        timestamp = datetime.now(UTC).isoformat()  # noqa: E501
        results: list[HHEMResult] = []  # noqa: E501
        for (source, summary, template), score in zip(filtered_dataset, scores, strict=True):  # noqa: E501
            results.append(  # noqa: E501
                HHEMResult(  # noqa: E501
                    model_name=model_name,  # noqa: E501
                    prompt_template=template,  # noqa: E501
                    source_text=source,  # noqa: E501
                    generated_summary=summary,  # noqa: E501
                    hhem_score=score,  # noqa: E501
                    timestamp=timestamp,  # noqa: E501
                )  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        # Compute aggregate statistics  # noqa: E501
        mean_score = mean(scores)  # noqa: E501
        median_score = median(scores)  # noqa: E501
        std_score = stdev(scores) if len(scores) > 1 else 0.0  # noqa: E501
        pass_05 = sum(1 for s in scores if s >= 0.5) / len(scores)  # noqa: E501
        pass_07 = sum(1 for s in scores if s >= 0.7) / len(scores)  # noqa: E501
  # noqa: E501
        return HHEMBenchmarkResult(  # noqa: E501
            model_name=model_name,  # noqa: E501
            num_samples=len(results),  # noqa: E501
            mean_score=mean_score,  # noqa: E501
            median_score=median_score,  # noqa: E501
            std_score=std_score,  # noqa: E501
            pass_rate_at_05=pass_05,  # noqa: E501
            pass_rate_at_07=pass_07,  # noqa: E501
            results=results,  # noqa: E501
            timestamp=timestamp,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_evaluator(model: HHEMModelProtocol | None = None) -> HHEMEvaluator:  # noqa: E501
    """Get an HHEM evaluator instance.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        model: Optional model for dependency injection (testing)  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Configured HHEMEvaluator instance  # noqa: E501
    """  # noqa: E501
    return HHEMEvaluator(model=model)  # noqa: E501
