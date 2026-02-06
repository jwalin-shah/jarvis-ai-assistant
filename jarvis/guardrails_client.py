"""Guardrails AI Integration for MLX.

Ensures model outputs meet safety and structure requirements before reaching the user.
"""

from guardrails import Guard
from guardrails.validators import FailResult, Validator, register_validator
from contracts.models import GenerationRequest
from models.generator import MLXGenerator
from models.loader import get_model

# 1. Define a Custom Local Validator (No external API calls)
@register_validator(name="jarvis/max_length", data_type="string")
class MaxLength(Validator):
    """Validates that the generated text is not too long."""
    
    def __init__(self, max_words: int = 50, on_fail="fix"):
        super().__init__(on_fail=on_fail, max_words=max_words)
        self.max_words = max_words

    def validate(self, value: str, metadata: dict) -> str | FailResult:
        if len(value.split()) > self.max_words:
            return FailResult(
                error_message=f"Response is too long ({len(value.split())} words > {self.max_words}).",
                fix_value=" ".join(value.split()[:self.max_words]) + "..."
            )
        return value

# 2. The Guarded Generator Wrapper
class GuardedMLXGenerator:
    """Wraps MLXGenerator with Guardrails validation."""
    
    def __init__(self):
        self.generator = MLXGenerator(loader=get_model(), skip_templates=True)
        
        # Define the RAIL (The Rules)
        # We enforce that the output must be a string and adhere to max length
        self.guard = Guard.from_string(
            validators=[MaxLength(max_words=30, on_fail="fix")],
            description="A concise helpful assistant"
        )

    def generate(self, prompt: str) -> str:
        """Generate a response that is guaranteed to be safe/valid."""
        
        # Define the callable that Guardrails will invoke
        def prompt_callable(msg):
            req = GenerationRequest(prompt=msg, max_tokens=50)
            return self.generator.generate(req).text

        # Run with validation
        # Guardrails will call the model, check the output, and fix it if needed
        validation_outcome = self.guard(
            llm_api=prompt_callable,
            prompt=prompt,
            num_reasks=1
        )
        
        return validation_outcome.validated_output
