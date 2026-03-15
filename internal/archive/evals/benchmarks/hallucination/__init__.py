"""HHEM hallucination evaluation benchmark (Workstream 2).  # noqa: E501
  # noqa: E501
This module implements the HallucinationEvaluator protocol using the  # noqa: E501
vectara/hallucination_evaluation_model from HuggingFace.  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    python -m benchmarks.hallucination.run --output results/hhem.json  # noqa: E501
  # noqa: E501
Components:  # noqa: E501
    - hhem: HHEMEvaluator class implementing the protocol  # noqa: E501
    - datasets: Test case generation for email summarization  # noqa: E501
    - run: CLI entrypoint for running benchmarks  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from evals.benchmarks.hallucination.datasets import (  # noqa: E501
    EmailTestCase,  # noqa: E501
    generate_grounded_pairs,  # noqa: E501
    generate_hallucinated_pairs,  # noqa: E501
    generate_mixed_dataset,  # noqa: E501
    generate_test_cases,  # noqa: E501
    get_dataset_metadata,  # noqa: E501
)  # noqa: E501
from evals.benchmarks.hallucination.hhem import (  # noqa: E501
    HHEMEvaluator,  # noqa: E501
    HHEMModelProtocol,  # noqa: E501
    VectaraHHEMModel,  # noqa: E501
    get_evaluator,  # noqa: E501
)  # noqa: E501

  # noqa: E501
__all__ = [  # noqa: E501
    # Evaluator  # noqa: E501
    "HHEMEvaluator",  # noqa: E501
    "HHEMModelProtocol",  # noqa: E501
    "VectaraHHEMModel",  # noqa: E501
    "get_evaluator",  # noqa: E501
    # Datasets  # noqa: E501
    "EmailTestCase",  # noqa: E501
    "generate_grounded_pairs",  # noqa: E501
    "generate_hallucinated_pairs",  # noqa: E501
    "generate_mixed_dataset",  # noqa: E501
    "generate_test_cases",  # noqa: E501
    "get_dataset_metadata",  # noqa: E501
]  # noqa: E501
