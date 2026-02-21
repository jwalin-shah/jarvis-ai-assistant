from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from jarvis.config import JarvisConfig
from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    GenerationRequest,
    IntentType,
    MessageContext,
    UrgencyLevel,
)
from jarvis.reply_service_generation import (
    build_generation_request,
    generate_llm_reply,
    to_model_generation_request,
)


class _StubContextService:
    def fetch_conversation_context(self, chat_id: str, limit: int = 3) -> tuple[list[str], int]:
        return [], limit

    def get_relationship_profile(self, contact, chat_id: str | None) -> tuple[dict, dict]:
        return {}, {}

    def is_bot_chat(self, chat_id: str | None, cname: str | None) -> bool:
        return False


class _StubGenerator:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, request) -> SimpleNamespace:  # pragma: no cover - simple stub
        return SimpleNamespace(text=self._text)


class _StubService:
    def __init__(self, generated_text: str = "") -> None:
        self.context_service = _StubContextService()
        self.generator = _StubGenerator(generated_text)

    def _resolve_instruction(self, instruction, category_name, category_config, classification):
        return instruction or "reply naturally"


def _classification() -> ClassificationResult:
    return ClassificationResult(
        intent=IntentType.STATEMENT,
        category=CategoryType.FULL_RESPONSE,
        urgency=UrgencyLevel.LOW,
        confidence=0.5,
        requires_knowledge=False,
        metadata={"category_name": "statement"},
    )


def _context() -> MessageContext:
    return MessageContext(
        chat_id="chat-1",
        message_text="this is a sufficiently long incoming message",
        is_from_me=False,
        timestamp=datetime.now(UTC),
        metadata={"contact_name": "Alex"},
    )


def _config(
    *,
    rag: bool = False,
    few_shot: bool = False,
    newline_stop: bool = True,
    word_cap_mode: str = "hard_10",
    confidence_mode: str = "legacy",
):
    return SimpleNamespace(
        model=SimpleNamespace(
            max_tokens_reply=64,
            negative_constraints=[],
            temperature=0.1,
        ),
        reply_pipeline=SimpleNamespace(
            reply_enable_rag=rag,
            reply_enable_few_shot=few_shot,
            reply_newline_stop_enabled=newline_stop,
            reply_short_msg_gate_enabled=True,
            reply_word_cap_mode=word_cap_mode,
            reply_confidence_mode=confidence_mode,
            reply_category_instruction_mode="universal",
        ),
        similarity_thresholds=SimpleNamespace(
            high_confidence=0.7,
            medium_confidence=0.45,
        ),
    )


def test_reply_pipeline_config_defaults() -> None:
    cfg = JarvisConfig()
    assert cfg.reply_pipeline.reply_enable_rag is False
    assert cfg.reply_pipeline.reply_enable_few_shot is False
    assert cfg.reply_pipeline.reply_word_cap_mode == "soft_25"
    assert cfg.reply_pipeline.reply_newline_stop_enabled is False
    assert cfg.reply_pipeline.reply_short_msg_gate_enabled is True
    assert cfg.reply_pipeline.reply_category_instruction_mode == "universal"
    assert cfg.reply_pipeline.reply_confidence_mode == "legacy"


def test_to_model_generation_request_honors_newline_stop_flag(monkeypatch) -> None:
    from jarvis import reply_service_generation as rsg

    monkeypatch.setattr(rsg, "get_config", lambda: _config(newline_stop=False))

    req = GenerationRequest(
        context=_context(),
        classification=_classification(),
        extraction=None,
    )
    model_req = to_model_generation_request(_StubService(), req)
    assert "\n" not in model_req.stop_sequences


def test_to_model_generation_request_honors_soft_token_caps(monkeypatch) -> None:
    from jarvis import reply_service_generation as rsg

    monkeypatch.setattr(rsg, "get_config", lambda: _config(word_cap_mode="soft_25"))
    req = GenerationRequest(
        context=_context(),
        classification=_classification(),
        extraction=None,
    )
    model_req = to_model_generation_request(_StubService(), req)
    assert model_req.max_tokens == 25


def test_build_generation_request_honors_rag_and_few_shot_flags(monkeypatch) -> None:
    from jarvis import reply_service_generation as rsg

    monkeypatch.setattr(rsg, "get_config", lambda: _config(rag=True, few_shot=True))

    context = _context()
    search_results = [{"text": "doc", "similarity": 0.9}]
    request = build_generation_request(
        _StubService(),
        context=context,
        classification=_classification(),
        search_results=search_results,
        contact=None,
        thread=["Them: hi", "Me: yo"],
    )

    assert request.retrieved_docs == search_results
    assert isinstance(request.few_shot_examples, list)


def test_generate_llm_reply_word_cap_modes(monkeypatch) -> None:
    from jarvis import reply_service_generation as rsg

    long_text = "one two three four five six seven eight nine ten eleven twelve"
    request = GenerationRequest(
        context=_context(),
        classification=_classification(),
        extraction=None,
    )

    monkeypatch.setattr(rsg, "get_config", lambda: _config(word_cap_mode="hard_10"))
    hard_result = generate_llm_reply(
        _StubService(generated_text=long_text),
        request=request,
        search_results=[],
        thread=["a", "b"],
    )
    assert len(hard_result.response.split()) == 10

    monkeypatch.setattr(rsg, "get_config", lambda: _config(word_cap_mode="none"))
    soft_result = generate_llm_reply(
        _StubService(generated_text=long_text),
        request=request,
        search_results=[],
        thread=["a", "b"],
    )
    assert len(soft_result.response.split()) > 10


def test_generate_llm_reply_short_message_gate_can_be_disabled(monkeypatch) -> None:
    from jarvis import reply_service_generation as rsg

    cfg = _config(word_cap_mode="none")
    cfg.reply_pipeline.reply_short_msg_gate_enabled = False
    monkeypatch.setattr(rsg, "get_config", lambda: cfg)

    context = MessageContext(
        chat_id="chat-1",
        message_text="where?",
        is_from_me=False,
        timestamp=datetime.now(UTC),
        metadata={"contact_name": "Alex"},
    )
    request = GenerationRequest(
        context=context,
        classification=_classification(),
        extraction=None,
    )
    result = generate_llm_reply(
        _StubService(generated_text="sounds good see you soon"),
        request=request,
        search_results=[],
        thread=["Them: where?"],
    )
    assert result.response != ""
