"""DSPy Client Adapter for MLX Generator.  # noqa: E501
  # noqa: E501
Wraps the local MLXModelLoader to be compatible with DSPy 3.x (BaseLM.forward API).  # noqa: E501
  # noqa: E501
Usage:  # noqa: E501
    import dspy  # noqa: E501
    from evals.dspy_client import DSPYMLXClient  # noqa: E501
  # noqa: E501
    lm = DSPYMLXClient()  # noqa: E501
    dspy.configure(lm=lm)  # noqa: E501
  # noqa: E501
    class BasicQA(dspy.Signature):  # noqa: E501
        question = dspy.InputField()  # noqa: E501
        answer = dspy.OutputField()  # noqa: E501
  # noqa: E501
    predictor = dspy.Predict(BasicQA)  # noqa: E501
    response = predictor(question="How does this help latency?")  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import time  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501

  # noqa: E501
try:  # noqa: E501
    import dspy  # noqa: E501
except ImportError:  # noqa: E501
    dspy = None  # type: ignore[assignment]  # noqa: E501
  # noqa: E501
from models.loader import get_model  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
class _Msg:  # noqa: E501
    """Minimal message object with attribute access for DSPy response parsing."""  # noqa: E501
  # noqa: E501
    def __init__(self, content: str):  # noqa: E501
        self.role = "assistant"  # noqa: E501
        self.content = content  # noqa: E501
  # noqa: E501
  # noqa: E501
class _Choice:  # noqa: E501
    """Minimal choice object matching OpenAI completion format."""  # noqa: E501
  # noqa: E501
    def __init__(self, text: str, index: int = 0):  # noqa: E501
        self.finish_reason = "stop"  # noqa: E501
        self.index = index  # noqa: E501
        self.message = _Msg(text)  # noqa: E501
  # noqa: E501
  # noqa: E501
class _Usage:  # noqa: E501
    """Token usage stub that supports dict() conversion."""  # noqa: E501
  # noqa: E501
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):  # noqa: E501
        self.prompt_tokens = prompt_tokens  # noqa: E501
        self.completion_tokens = completion_tokens  # noqa: E501
        self.total_tokens = prompt_tokens + completion_tokens  # noqa: E501
  # noqa: E501
    def __iter__(self):  # noqa: E501
        yield "prompt_tokens", self.prompt_tokens  # noqa: E501
        yield "completion_tokens", self.completion_tokens  # noqa: E501
        yield "total_tokens", self.total_tokens  # noqa: E501
  # noqa: E501
  # noqa: E501
class _CompletionResponse:  # noqa: E501
    """Minimal OpenAI-compatible completion response for DSPy."""  # noqa: E501
  # noqa: E501
    def __init__(self, text: str, model: str):  # noqa: E501
        self.choices = [_Choice(text)]  # noqa: E501
        self.model = model  # noqa: E501
        self.created = int(time.time())  # noqa: E501
        # Approximate token count: ~1.3 tokens per word for English text  # noqa: E501
        word_count = len(text.split())  # noqa: E501
        self.usage = _Usage(completion_tokens=int(word_count * 1.3))  # noqa: E501
  # noqa: E501
    def __getitem__(self, key: str) -> Any:  # noqa: E501
        return getattr(self, key)  # noqa: E501
  # noqa: E501
  # noqa: E501
_BaseLM = dspy.BaseLM if dspy is not None else object  # noqa: E501
  # noqa: E501
  # noqa: E501
class DSPYMLXClient(_BaseLM):  # type: ignore[misc]  # noqa: E501
    """Adapter that exposes the local MLX model as a DSPy BaseLM."""  # noqa: E501
  # noqa: E501
    def __init__(  # noqa: E501
        self,  # noqa: E501
        model_name: str = "mlx-local",  # noqa: E501
        max_tokens: int = 200,  # noqa: E501
        temperature: float = 0.1,  # noqa: E501
        **kwargs: Any,  # noqa: E501
    ):  # noqa: E501
        if dspy is None:  # noqa: E501
            raise ImportError("DSPy is not installed. Install it with: pip install dspy-ai")  # noqa: E501
        super().__init__(  # noqa: E501
            model=model_name,  # noqa: E501
            model_type="chat",  # noqa: E501
            temperature=temperature,  # noqa: E501
            max_tokens=max_tokens,  # noqa: E501
            **kwargs,  # noqa: E501
        )  # noqa: E501
        self._loader = get_model()  # noqa: E501
  # noqa: E501
    def forward(  # noqa: E501
        self,  # noqa: E501
        prompt: str | None = None,  # noqa: E501
        messages: list[dict[str, Any]] | None = None,  # noqa: E501
        **kwargs: Any,  # noqa: E501
    ) -> _CompletionResponse:  # noqa: E501
        """Generate a completion via the local MLX model.  # noqa: E501
  # noqa: E501
        Returns a single OpenAI-compatible response object. The base class  # noqa: E501
        __call__ passes this to _process_completion which iterates .choices.  # noqa: E501
        """  # noqa: E501
        # Build a single prompt string from either input form  # noqa: E501
        if messages:  # noqa: E501
            parts = []  # noqa: E501
            for msg in messages:  # noqa: E501
                role = msg.get("role", "user")  # noqa: E501
                content = msg.get("content", "")  # noqa: E501
                if role == "system":  # noqa: E501
                    parts.append(f"<system>\n{content}\n</system>")  # noqa: E501
                elif role == "assistant":  # noqa: E501
                    parts.append(f"<assistant>\n{content}\n</assistant>")  # noqa: E501
                else:  # noqa: E501
                    parts.append(content)  # noqa: E501
            text_prompt = "\n\n".join(parts)  # noqa: E501
        elif prompt:  # noqa: E501
            text_prompt = prompt  # noqa: E501
        else:  # noqa: E501
            text_prompt = ""  # noqa: E501
  # noqa: E501
        max_tokens = kwargs.get("max_tokens", self.kwargs.get("max_tokens", 200))  # noqa: E501
        temperature = kwargs.get("temperature", self.kwargs.get("temperature", 0.1))  # noqa: E501
  # noqa: E501
        if not self._loader.is_loaded():  # noqa: E501
            self._loader.load()  # noqa: E501
  # noqa: E501
        result = self._loader.generate_sync(  # noqa: E501
            prompt=text_prompt,  # noqa: E501
            max_tokens=max_tokens,  # noqa: E501
            temperature=temperature,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        return _CompletionResponse(text=result.text, model=self.model)  # noqa: E501
