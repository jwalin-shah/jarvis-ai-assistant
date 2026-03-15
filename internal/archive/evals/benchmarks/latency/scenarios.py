"""Test scenario definitions for latency benchmarking.  # noqa: E501
  # noqa: E501
Workstream 4: Latency Benchmark  # noqa: E501
  # noqa: E501
Defines cold, warm, and hot start scenarios with representative prompts.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
from typing import Literal  # noqa: E402  # noqa: E501

  # noqa: E501
# Re-export Scenario type from contracts  # noqa: E501
Scenario = Literal["cold", "warm", "hot"]  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class LatencyScenario:  # noqa: E501
    """Definition of a latency test scenario."""  # noqa: E501
  # noqa: E501
    name: str  # noqa: E501
    scenario_type: Scenario  # noqa: E501
    prompt: str  # noqa: E501
    max_tokens: int  # noqa: E501
    description: str  # noqa: E501
  # noqa: E501
  # noqa: E501
# Default test prompts for each scenario type  # noqa: E501
_COLD_START_PROMPT = "What is the capital of France?"  # noqa: E501
_WARM_START_PROMPT = "Explain the concept of machine learning in simple terms."  # noqa: E501
_HOT_START_PROMPT = "What is the capital of France?"  # Same as cold for caching test  # noqa: E501
  # noqa: E501
# Token generation targets for benchmarking  # noqa: E501
DEFAULT_MAX_TOKENS = 50  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_default_scenarios() -> list[LatencyScenario]:  # noqa: E501
    """Return default test scenarios for all scenario types.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        List of LatencyScenario definitions for cold, warm, and hot starts.  # noqa: E501
    """  # noqa: E501
    return [  # noqa: E501
        LatencyScenario(  # noqa: E501
            name="cold_start",  # noqa: E501
            scenario_type="cold",  # noqa: E501
            prompt=_COLD_START_PROMPT,  # noqa: E501
            max_tokens=DEFAULT_MAX_TOKENS,  # noqa: E501
            description="Fresh app launch - model loads from disk",  # noqa: E501
        ),  # noqa: E501
        LatencyScenario(  # noqa: E501
            name="warm_start",  # noqa: E501
            scenario_type="warm",  # noqa: E501
            prompt=_WARM_START_PROMPT,  # noqa: E501
            max_tokens=DEFAULT_MAX_TOKENS,  # noqa: E501
            description="Model already in memory, new user query",  # noqa: E501
        ),  # noqa: E501
        LatencyScenario(  # noqa: E501
            name="hot_start",  # noqa: E501
            scenario_type="hot",  # noqa: E501
            prompt=_HOT_START_PROMPT,  # noqa: E501
            max_tokens=DEFAULT_MAX_TOKENS,  # noqa: E501
            description="Repeated similar queries (tests prompt caching)",  # noqa: E501
        ),  # noqa: E501
    ]  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_scenario_by_type(scenario_type: Scenario) -> LatencyScenario:  # noqa: E501
    """Get the default test scenario for a given type.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        scenario_type: One of 'cold', 'warm', or 'hot'.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        LatencyScenario for the specified type.  # noqa: E501
  # noqa: E501
    Raises:  # noqa: E501
        ValueError: If scenario_type is not valid.  # noqa: E501
    """  # noqa: E501
    scenarios = {s.scenario_type: s for s in get_default_scenarios()}  # noqa: E501
    if scenario_type not in scenarios:  # noqa: E501
        valid = list(scenarios.keys())  # noqa: E501
        msg = f"Invalid scenario type: {scenario_type}. Valid types: {valid}"  # noqa: E501
        raise ValueError(msg)  # noqa: E501
    return scenarios[scenario_type]  # noqa: E501
  # noqa: E501
  # noqa: E501
# Extended prompts for comprehensive benchmarking  # noqa: E501
BENCHMARK_PROMPTS = {  # noqa: E501
    "short": [  # noqa: E501
        "Hello",  # noqa: E501
        "What time is it?",  # noqa: E501
        "Thanks!",  # noqa: E501
    ],  # noqa: E501
    "medium": [  # noqa: E501
        "Explain quantum computing",  # noqa: E501
        "What is the difference between AI and ML?",  # noqa: E501
        "How does photosynthesis work?",  # noqa: E501
    ],  # noqa: E501
    "long": [  # noqa: E501
        "Write a detailed explanation of how neural networks learn from data, "  # noqa: E501
        "including the concepts of forward propagation, backpropagation, and gradient descent.",  # noqa: E501
        "Describe the process of software development from requirements gathering to deployment, "  # noqa: E501
        "including all intermediate steps and best practices.",  # noqa: E501
        "Explain the history and evolution of programming languages, starting from machine code "  # noqa: E501
        "to modern high-level languages, and discuss their key features.",  # noqa: E501
    ],  # noqa: E501
}  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_benchmark_prompts(category: str = "medium") -> list[str]:  # noqa: E501
    """Get benchmark prompts for a specific category.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        category: One of 'short', 'medium', or 'long'.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        List of prompts for the category.  # noqa: E501
  # noqa: E501
    Raises:  # noqa: E501
        ValueError: If category is not valid.  # noqa: E501
    """  # noqa: E501
    if category not in BENCHMARK_PROMPTS:  # noqa: E501
        valid = list(BENCHMARK_PROMPTS.keys())  # noqa: E501
        msg = f"Invalid prompt category: {category}. Valid categories: {valid}"  # noqa: E501
        raise ValueError(msg)  # noqa: E501
    return BENCHMARK_PROMPTS[category]  # noqa: E501
