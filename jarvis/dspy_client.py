"""DSPy Client Adapter for MLX Generator.

This module provides a wrapper around the local MLXGenerator to make it compatible
with the DSPy framework. This allows using DSPy's powerful prompt optimization
and compilation features with the local-first MLX model.

Usage:
    import dspy
    from jarvis.dspy_client import DSPYMLXClient
    
    # Configure DSPy to use the local model
    lm = DSPYMLXClient()
    dspy.settings.configure(lm=lm)
    
    # Now use DSPy normally
    class BasicQA(dspy.Signature):
        question = dspy.InputField()
        answer = dspy.OutputField()
        
    predictor = dspy.Predict(BasicQA)
    response = predictor(question="How does this help latency?")
"""

import dspy
from typing import Any, Optional, List, Union
from contracts.models import GenerationRequest
from models.generator import MLXGenerator
from models.loader import get_model

class DSPYMLXClient(dspy.LM):
    """Adapter to make MLXGenerator compatible with DSPy."""
    
    def __init__(
        self,
        model_name: str = "mlx-local",
        max_tokens: int = 200,
        temperature: float = 0.1,
        **kwargs
    ):
        """Initialize the DSPy client wrapper.
        
        Args:
            model_name: Identifier for the model (mostly for logging)
            max_tokens: Default max tokens for generation
            temperature: Default temperature
            **kwargs: Additional args passed to base dspy.LM
        """
        super().__init__(model=model_name, **kwargs)
        self.generator = MLXGenerator(loader=get_model(), skip_templates=True)
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> List[Union[str, dict[str, Any]]]:
        """Execute the generation request from DSPy.
        
        DSPy passes the raw prompt (including its own few-shot examples) here.
        We forward it to the MLXGenerator.
        """
        # Extract generation parameters with fallbacks
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        temperature = kwargs.get("temperature", self.default_temperature)
        
        # Build the standard GenerationRequest
        # Note: DSPy handles the prompt construction (examples, history),
        # so we treat the whole input as the 'prompt'.
        req = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            # Pass other params if needed
        )
        
        # Execute using the local model
        response = self.generator.generate(req)
        
        # Format for DSPy
        # DSPy expects a list of completion strings
        completions = [response.text]
        
        if only_completed:
            return completions
            
        # If full response requested, might need dictionary format
        # For now, simplistic implementation
        return completions

    def basic_request(self, prompt: str, **kwargs):
        """Simplified request interface required by some DSPy internals."""
        return self.__call__(prompt, **kwargs)

    def request(self, prompt: str, **kwargs):
        """Standard request interface."""
        return self.__call__(prompt, **kwargs)
