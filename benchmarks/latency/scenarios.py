"""Test scenario definitions for latency benchmarking.

Workstream 4: Latency Benchmark

Defines cold, warm, and hot start scenarios with representative prompts.
"""

from dataclasses import dataclass
from typing import Literal

# Re-export Scenario type from contracts
Scenario = Literal["cold", "warm", "hot"]


@dataclass
class LatencyScenario:
    """Definition of a latency test scenario."""

    name: str
    scenario_type: Scenario
    prompt: str
    max_tokens: int
    description: str


# Default test prompts for each scenario type
_COLD_START_PROMPT = "What is the capital of France?"
_WARM_START_PROMPT = "Explain the concept of machine learning in simple terms."
_HOT_START_PROMPT = "What is the capital of France?"  # Same as cold for caching test

# Token generation targets for benchmarking
DEFAULT_MAX_TOKENS = 50


def get_default_scenarios() -> list[LatencyScenario]:
    """Return default test scenarios for all scenario types.

    Returns:
        List of LatencyScenario definitions for cold, warm, and hot starts.
    """
    return [
        LatencyScenario(
            name="cold_start",
            scenario_type="cold",
            prompt=_COLD_START_PROMPT,
            max_tokens=DEFAULT_MAX_TOKENS,
            description="Fresh app launch - model loads from disk",
        ),
        LatencyScenario(
            name="warm_start",
            scenario_type="warm",
            prompt=_WARM_START_PROMPT,
            max_tokens=DEFAULT_MAX_TOKENS,
            description="Model already in memory, new user query",
        ),
        LatencyScenario(
            name="hot_start",
            scenario_type="hot",
            prompt=_HOT_START_PROMPT,
            max_tokens=DEFAULT_MAX_TOKENS,
            description="Repeated similar queries (tests prompt caching)",
        ),
    ]


def get_scenario_by_type(scenario_type: Scenario) -> LatencyScenario:
    """Get the default test scenario for a given type.

    Args:
        scenario_type: One of 'cold', 'warm', or 'hot'.

    Returns:
        LatencyScenario for the specified type.

    Raises:
        ValueError: If scenario_type is not valid.
    """
    scenarios = {s.scenario_type: s for s in get_default_scenarios()}
    if scenario_type not in scenarios:
        valid = list(scenarios.keys())
        msg = f"Invalid scenario type: {scenario_type}. Valid types: {valid}"
        raise ValueError(msg)
    return scenarios[scenario_type]


# Extended prompts for comprehensive benchmarking
BENCHMARK_PROMPTS = {
    "short": [
        "Hello",
        "What time is it?",
        "Thanks!",
    ],
    "medium": [
        "Explain quantum computing",
        "What is the difference between AI and ML?",
        "How does photosynthesis work?",
    ],
    "long": [
        "Write a detailed explanation of how neural networks learn from data, "
        "including the concepts of forward propagation, backpropagation, and gradient descent.",
        "Describe the process of software development from requirements gathering to deployment, "
        "including all intermediate steps and best practices.",
        "Explain the history and evolution of programming languages, starting from machine code "
        "to modern high-level languages, and discuss their key features.",
    ],
}


def get_benchmark_prompts(category: str = "medium") -> list[str]:
    """Get benchmark prompts for a specific category.

    Args:
        category: One of 'short', 'medium', or 'long'.

    Returns:
        List of prompts for the category.

    Raises:
        ValueError: If category is not valid.
    """
    if category not in BENCHMARK_PROMPTS:
        valid = list(BENCHMARK_PROMPTS.keys())
        msg = f"Invalid prompt category: {category}. Valid categories: {valid}"
        raise ValueError(msg)
    return BENCHMARK_PROMPTS[category]
