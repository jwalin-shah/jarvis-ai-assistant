"""DSPy Client Adapter for MLX Generator.

Wraps the local MLXModelLoader to be compatible with DSPy 3.x (BaseLM.forward API).

Usage:
    import dspy
    from jarvis.dspy_client import DSPYMLXClient

    lm = DSPYMLXClient()
    dspy.configure(lm=lm)

    class BasicQA(dspy.Signature):
        question = dspy.InputField()
        answer = dspy.OutputField()

    predictor = dspy.Predict(BasicQA)
    response = predictor(question="How does this help latency?")
"""

from __future__ import annotations

import time
from typing import Any

import dspy

from models.loader import get_model


class _Msg:
    """Minimal message object with attribute access for DSPy response parsing."""

    def __init__(self, content: str):
        self.role = "assistant"
        self.content = content


class _Choice:
    """Minimal choice object matching OpenAI completion format."""

    def __init__(self, text: str, index: int = 0):
        self.finish_reason = "stop"
        self.index = index
        self.message = _Msg(text)


class _Usage:
    """Token usage stub that supports dict() conversion."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

    def __iter__(self):
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


class _CompletionResponse:
    """Minimal OpenAI-compatible completion response for DSPy."""

    def __init__(self, text: str, model: str):
        self.choices = [_Choice(text)]
        self.model = model
        self.created = int(time.time())
        self.usage = _Usage(completion_tokens=len(text.split()))

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class DSPYMLXClient(dspy.BaseLM):
    """Adapter that exposes the local MLX model as a DSPy BaseLM."""

    def __init__(
        self,
        model_name: str = "mlx-local",
        max_tokens: int = 200,
        temperature: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(
            model=model_name,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._loader = get_model()

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _CompletionResponse:
        """Generate a completion via the local MLX model.

        Returns a single OpenAI-compatible response object. The base class
        __call__ passes this to _process_completion which iterates .choices.
        """
        # Build a single prompt string from either input form
        if messages:
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"<system>\n{content}\n</system>")
                elif role == "assistant":
                    parts.append(f"<assistant>\n{content}\n</assistant>")
                else:
                    parts.append(content)
            text_prompt = "\n\n".join(parts)
        elif prompt:
            text_prompt = prompt
        else:
            text_prompt = ""

        max_tokens = kwargs.get("max_tokens", self.kwargs.get("max_tokens", 200))
        temperature = kwargs.get("temperature", self.kwargs.get("temperature", 0.1))

        if not self._loader.is_loaded():
            self._loader.load()

        result = self._loader.generate_sync(
            prompt=text_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return _CompletionResponse(text=result.text, model=self.model)
