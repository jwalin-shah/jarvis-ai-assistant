"""Comprehensive tests for prompt assembly logic.

Tests the full prompt building pipeline: PromptBuilder (models/prompt_builder.py),
prompt template builders (jarvis/prompts/builders.py), and the RAG-enhanced
prompt assembly path (build_prompt_from_request).

CRITICAL: These tests verify the actual prompt STRING content, not just that
methods were called. A malformed prompt silently degrades generation quality.
"""

from __future__ import annotations

import re
from datetime import datetime

from contracts.models import GenerationRequest as ModelGenerationRequest
from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    IntentType,
    MessageContext,
    RAGDocument,
    UrgencyLevel,
)
from jarvis.contracts.pipeline import (
    GenerationRequest as PipelineGenerationRequest,
)
from jarvis.prompts.builders import (
    _format_examples,
    _format_relationship_context,
    _format_search_examples,
    _format_similar_exchanges,
    _format_summary_examples,
    _truncate_context,
    build_prompt_from_request,
    build_rag_reply_prompt,
    build_reply_prompt,
    build_search_answer_prompt,
    build_summary_prompt,
    estimate_tokens,
    format_facts_for_prompt,
    is_within_token_limit,
)
from jarvis.prompts.constants import (
    MAX_CONTEXT_CHARS,
    RAG_REPLY_PROMPT,
    SYSTEM_PREFIX,
    FewShotExample,
)
from models.prompt_builder import PromptBuilder

# =============================================================================
# PromptBuilder (models/prompt_builder.py) Tests
# =============================================================================


class TestPromptBuilderBasicAssembly:
    """Test PromptBuilder.build() assembles sections correctly."""

    def setup_method(self):
        self.builder = PromptBuilder()

    def test_prompt_only_produces_task_section(self):
        req = ModelGenerationRequest(prompt="Reply to this message")
        result = self.builder.build(req)
        assert result == "### Your Task:\nReply to this message"

    def test_context_documents_appear_before_task(self):
        req = ModelGenerationRequest(
            prompt="Reply to this",
            context_documents=["Doc 1 content", "Doc 2 content"],
        )
        result = self.builder.build(req)
        context_pos = result.index("### Relevant Context:")
        task_pos = result.index("### Your Task:")
        assert context_pos < task_pos
        assert "Doc 1 content" in result
        assert "Doc 2 content" in result

    def test_few_shot_examples_appear_between_context_and_task(self):
        req = ModelGenerationRequest(
            prompt="Reply to this",
            context_documents=["Some context"],
            few_shot_examples=[("What's up?", "Not much"), ("Hey", "Hi there")],
        )
        result = self.builder.build(req)
        context_pos = result.index("### Relevant Context:")
        examples_pos = result.index("### Examples:")
        task_pos = result.index("### Your Task:")
        assert context_pos < examples_pos < task_pos

    def test_examples_without_context_documents(self):
        req = ModelGenerationRequest(
            prompt="Reply to this",
            few_shot_examples=[("Input A", "Output A")],
        )
        result = self.builder.build(req)
        assert "### Relevant Context:" not in result
        assert "### Examples:" in result
        assert "### Your Task:" in result
        assert "Input: Input A" in result
        assert "Output: Output A" in result

    def test_section_separator_is_double_newline(self):
        req = ModelGenerationRequest(
            prompt="Task text",
            context_documents=["Doc text"],
            few_shot_examples=[("In", "Out")],
        )
        result = self.builder.build(req)
        assert "Doc text\n\n### Examples:" in result
        assert "Output: Out\n\n### Your Task:" in result

    def test_document_separator_is_dash_line(self):
        req = ModelGenerationRequest(
            prompt="Task",
            context_documents=["First doc", "Second doc", "Third doc"],
        )
        result = self.builder.build(req)
        assert "First doc\n---\nSecond doc\n---\nThird doc" in result

    def test_multiple_examples_formatted_correctly(self):
        req = ModelGenerationRequest(
            prompt="Task",
            few_shot_examples=[
                ("Question 1", "Answer 1"),
                ("Question 2", "Answer 2"),
            ],
        )
        result = self.builder.build(req)
        assert "Input: Question 1\nOutput: Answer 1" in result
        assert "Input: Question 2\nOutput: Answer 2" in result
        assert "Answer 1\n\nInput: Question 2" in result

    def test_whitespace_in_documents_is_stripped(self):
        req = ModelGenerationRequest(
            prompt="Task",
            context_documents=["  padded doc  ", "\n\nmore padding\n\n"],
        )
        result = self.builder.build(req)
        assert "padded doc\n---\nmore padding" in result

    def test_whitespace_in_examples_is_stripped(self):
        req = ModelGenerationRequest(
            prompt="Task",
            few_shot_examples=[("  input  ", "  output  ")],
        )
        result = self.builder.build(req)
        assert "Input: input\nOutput: output" in result

    def test_prompt_text_is_stripped(self):
        req = ModelGenerationRequest(prompt="  actual prompt  ")
        result = self.builder.build(req)
        assert result == "### Your Task:\nactual prompt"


class TestPromptBuilderPassthrough:
    """Test XML passthrough mode in PromptBuilder.build()."""

    def setup_method(self):
        self.builder = PromptBuilder()

    def test_xml_prompt_without_extras_passes_through(self):
        xml_prompt = "<system>You are a bot</system>\n<reply>hello</reply>"
        req = ModelGenerationRequest(prompt=xml_prompt)
        result = self.builder.build(req)
        assert result == xml_prompt

    def test_xml_prompt_with_leading_whitespace_passes_through(self):
        xml_prompt = "  <system>content</system>"
        req = ModelGenerationRequest(prompt=xml_prompt)
        result = self.builder.build(req)
        assert result == xml_prompt

    def test_xml_prompt_with_context_docs_does_not_passthrough(self):
        xml_prompt = "<system>content</system>"
        req = ModelGenerationRequest(
            prompt=xml_prompt,
            context_documents=["extra doc"],
        )
        result = self.builder.build(req)
        assert "### Relevant Context:" in result
        assert "### Your Task:" in result
        assert "extra doc" in result

    def test_xml_prompt_with_examples_does_not_passthrough(self):
        xml_prompt = "<system>content</system>"
        req = ModelGenerationRequest(
            prompt=xml_prompt,
            few_shot_examples=[("in", "out")],
        )
        result = self.builder.build(req)
        assert "### Examples:" in result
        assert "### Your Task:" in result

    def test_non_xml_prompt_always_wraps(self):
        req = ModelGenerationRequest(prompt="Just reply to this")
        result = self.builder.build(req)
        assert "### Your Task:" in result
        assert "Just reply to this" in result


# =============================================================================
# Reply Prompt Assembly Tests
# =============================================================================


class TestBuildReplyPromptAssembly:
    """Test that build_reply_prompt assembles all sections correctly."""

    def test_basic_assembly_has_all_sections(self):
        result = build_reply_prompt(
            context="[10:00] Alice: Want to grab coffee?",
            last_message="Want to grab coffee?",
        )
        required_sections = [
            "### Conversation Context:",
            "### Instructions:",
            "### Examples:",
            "### Last message to reply to:",
            "### Your reply:",
        ]
        for section in required_sections:
            assert section in result, f"Missing section: {section}"

    def test_context_is_inserted_verbatim(self):
        context = "[10:00] Alice: Want to grab coffee?\n[10:01] Bob: Sure, when?"
        result = build_reply_prompt(context=context, last_message="Sure, when?")
        assert context in result

    def test_last_message_is_inserted_verbatim(self):
        last_msg = "Can you pick me up at 5?"
        result = build_reply_prompt(context="Test context", last_message=last_msg)
        assert f"### Last message to reply to:\n{last_msg}" in result

    def test_ends_with_reply_marker(self):
        result = build_reply_prompt(context="Test", last_message="Test")
        assert result.rstrip().endswith("### Your reply:")

    def test_casual_tone_string_in_instructions(self):
        result = build_reply_prompt(context="Test", last_message="Test", tone="casual")
        assert "casual/friendly" in result

    def test_professional_tone_string_in_instructions(self):
        result = build_reply_prompt(context="Test", last_message="Test", tone="professional")
        assert "professional/formal" in result

    def test_custom_instruction_included(self):
        result = build_reply_prompt(
            context="Test",
            last_message="Test",
            instruction="Be very brief, max 3 words",
        )
        assert "Be very brief, max 3 words" in result

    def test_examples_section_has_context_reply_format(self):
        result = build_reply_prompt(context="Test", last_message="Test")
        assert "Context:" in result
        assert "Reply:" in result

    def test_no_raw_template_placeholders_in_output(self):
        result = build_reply_prompt(
            context="Some context",
            last_message="A message",
            instruction="Be helpful",
            tone="casual",
        )
        unresolved = re.findall(r"\{[a-z_]+\}", result)
        assert unresolved == [], f"Unresolved placeholders found: {unresolved}"


# =============================================================================
# RAG Reply Prompt Assembly Tests
# =============================================================================


class TestBuildRagReplyPromptAssembly:
    """Test that build_rag_reply_prompt produces correct XML-structured prompts."""

    def test_basic_rag_prompt_has_xml_structure(self):
        result = build_rag_reply_prompt(
            context="[10:00] Alice: Hey!",
            last_message="Hey!",
            contact_name="Alice",
        )
        expected_tags = [
            "<system>",
            "</system>",
            "<style",
            "</style>",
            "<relationships>",
            "</relationships>",
            "<facts>",
            "</facts>",
            "<examples>",
            "</examples>",
            "<conversation>",
            "</conversation>",
            "<instruction>",
            "</instruction>",
            "<last_message>",
            "</last_message>",
            "<reply>",
        ]
        for tag in expected_tags:
            assert tag in result, f"Missing XML tag: {tag}"

    def test_contact_name_in_style_tag(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="John",
        )
        assert '<style contact="John">' in result

    def test_similar_exchanges_formatted(self):
        exchanges = [
            ("Hey, what's up?", "Not much, you?"),
            ("Want to grab lunch?", "Sure, where?"),
        ]
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
            similar_exchanges=exchanges,
        )
        assert "Example 1:" in result
        assert "Hey, what's up?" in result
        assert "Your reply: Not much, you?" in result
        assert "Example 2:" in result

    def test_no_similar_exchanges_shows_fallback(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
            similar_exchanges=[],
        )
        assert "(No similar past exchanges found)" in result

    def test_instruction_in_instruction_tag(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
            instruction="Confirm dinner at 7pm",
        )
        assert "<instruction>Confirm dinner at 7pm</instruction>" in result

    def test_empty_instruction_produces_empty_tag(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
        )
        assert "<instruction></instruction>" in result

    def test_contact_facts_included(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
            contact_facts="lives_in: Austin, works_at: Google",
        )
        assert "<facts>\nlives_in: Austin, works_at: Google\n</facts>" in result

    def test_no_contact_facts_shows_none(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
        )
        assert "<facts>\n(none)\n</facts>" in result

    def test_relationship_graph_included(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
            relationship_graph="Alice -- friend --> Bob",
        )
        assert "Alice -- friend --> Bob" in result

    def test_no_relationship_graph_shows_none(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
        )
        assert "<relationships>\n(none)\n</relationships>" in result

    def test_last_message_in_last_message_tag(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Can you help me move Saturday?",
            contact_name="Alice",
        )
        assert "<last_message>Can you help me move Saturday?</last_message>" in result

    def test_prompt_ends_with_reply_tag(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
        )
        assert result.rstrip().endswith("<reply>")

    def test_rag_prompt_starts_with_system_prefix(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
        )
        assert result.startswith(SYSTEM_PREFIX)

    def test_no_raw_template_placeholders_in_rag_output(self):
        result = build_rag_reply_prompt(
            context="Some context",
            last_message="A message",
            contact_name="Alice",
            instruction="Be brief",
            contact_facts="likes: coffee",
            relationship_graph="Alice -> friend -> Bob",
            similar_exchanges=[("hey", "hi")],
        )
        unresolved = re.findall(r"\{[a-z_]+\}", result)
        assert unresolved == [], f"Unresolved placeholders found: {unresolved}"


# =============================================================================
# build_prompt_from_request Tests
# =============================================================================


class TestBuildPromptFromRequest:
    """Test the pipeline GenerationRequest -> prompt string conversion."""

    @staticmethod
    def _make_pipeline_request(
        message_text: str = "What's for dinner?",
        context_messages: list[str] | None = None,
        retrieved_docs: list[RAGDocument] | None = None,
        few_shot_examples: list[dict[str, str]] | None = None,
        instruction: str | None = None,
        contact_name: str | None = None,
        contact_facts: str = "",
        relationship_graph: str = "",
    ) -> PipelineGenerationRequest:
        metadata: dict = {}
        if context_messages is not None:
            metadata["context_messages"] = context_messages
        if instruction is not None:
            metadata["instruction"] = instruction
        if contact_name is not None:
            metadata["contact_name"] = contact_name
        if contact_facts:
            metadata["contact_facts"] = contact_facts
        if relationship_graph:
            metadata["relationship_graph"] = relationship_graph

        context = MessageContext(
            chat_id="chat123",
            message_text=message_text,
            is_from_me=False,
            timestamp=datetime(2026, 2, 11, 12, 0, 0),
            metadata=metadata,
        )
        classification = ClassificationResult(
            intent=IntentType.QUESTION,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.85,
            requires_knowledge=True,
        )
        return PipelineGenerationRequest(
            context=context,
            classification=classification,
            extraction=None,
            retrieved_docs=retrieved_docs or [],
            few_shot_examples=few_shot_examples or [],
        )

    def test_basic_request_produces_valid_prompt(self):
        req = self._make_pipeline_request()
        result = build_prompt_from_request(req)
        assert "<system>" in result
        assert "<reply>" in result
        assert "What's for dinner?" in result

    def test_context_messages_in_conversation_section(self):
        req = self._make_pipeline_request(
            context_messages=["Hey, what are we doing tonight?", "I was thinking dinner"],
        )
        result = build_prompt_from_request(req)
        assert "Hey, what are we doing tonight?" in result
        assert "I was thinking dinner" in result

    def test_retrieved_docs_become_similar_exchanges(self):
        docs = [
            RAGDocument(
                content="What's for lunch?",
                source="rag",
                score=0.9,
                metadata={"response_text": "Let's get tacos"},
            ),
        ]
        req = self._make_pipeline_request(retrieved_docs=docs)
        result = build_prompt_from_request(req)
        assert "What's for lunch?" in result
        assert "Let's get tacos" in result

    def test_few_shot_examples_become_exchanges(self):
        examples = [
            {"input": "How's it going?", "output": "Pretty good, you?"},
        ]
        req = self._make_pipeline_request(few_shot_examples=examples)
        result = build_prompt_from_request(req)
        assert "How's it going?" in result
        assert "Pretty good, you?" in result

    def test_instruction_passed_through(self):
        req = self._make_pipeline_request(instruction="Be very concise")
        result = build_prompt_from_request(req)
        assert "Be very concise" in result

    def test_contact_name_passed_through(self):
        req = self._make_pipeline_request(contact_name="Sarah")
        result = build_prompt_from_request(req)
        assert '<style contact="Sarah">' in result

    def test_contact_facts_passed_through(self):
        req = self._make_pipeline_request(contact_facts="lives_in: NYC")
        result = build_prompt_from_request(req)
        assert "lives_in: NYC" in result

    def test_relationship_graph_passed_through(self):
        req = self._make_pipeline_request(relationship_graph="Sarah -- roommate --> Alex")
        result = build_prompt_from_request(req)
        assert "Sarah -- roommate --> Alex" in result

    def test_fallback_to_message_text_when_no_context(self):
        req = self._make_pipeline_request(
            message_text="Hey what's up?",
            context_messages=None,
        )
        req.context.metadata.pop("thread", None)
        result = build_prompt_from_request(req)
        assert "Hey what's up?" in result

    def test_default_contact_name_is_them(self):
        req = self._make_pipeline_request(contact_name=None)
        result = build_prompt_from_request(req)
        assert '<style contact="them">' in result


# =============================================================================
# Context Truncation Tests
# =============================================================================


class TestContextTruncation:
    """Test context truncation behavior."""

    def test_short_context_not_truncated(self):
        short = "A short context"
        assert _truncate_context(short) == short

    def test_long_context_is_truncated(self):
        long_context = "x" * (MAX_CONTEXT_CHARS + 1000)
        result = _truncate_context(long_context)
        assert len(result) <= MAX_CONTEXT_CHARS + 100
        assert "[Earlier messages truncated]" in result

    def test_truncation_keeps_recent_messages(self):
        messages = [f"[{i:03d}] Message number {i}" for i in range(200)]
        long_context = "\n".join(messages)
        result = _truncate_context(long_context)
        assert "Message number 199" in result
        assert "Message number 198" in result

    def test_truncation_tries_to_break_at_newline(self):
        messages = [f"Line {i}: {'x' * 50}" for i in range(200)]
        long_context = "\n".join(messages)
        result = _truncate_context(long_context)
        after_prefix = result.split("[Earlier messages truncated]\n", 1)
        if len(after_prefix) == 2:
            first_line = after_prefix[1].split("\n")[0]
            assert first_line.startswith("Line ")

    def test_truncation_in_reply_prompt(self):
        long_context = "Message line\n" * 500
        result = build_reply_prompt(long_context, "last message")
        assert "[Earlier messages truncated]" in result
        assert "### Your reply:" in result

    def test_truncation_in_rag_prompt(self):
        long_context = "Message line\n" * 500
        result = build_rag_reply_prompt(
            context=long_context,
            last_message="last message",
            contact_name="Alice",
        )
        assert "[Earlier messages truncated]" in result
        assert "<reply>" in result


# =============================================================================
# Empty/Missing Input Tests
# =============================================================================


class TestEmptyInputs:
    """Test prompt assembly with empty or missing inputs."""

    def test_empty_context_still_valid(self):
        result = build_reply_prompt(context="", last_message="Hello")
        assert "### Your reply:" in result
        assert "### Conversation Context:" in result

    def test_empty_last_message_still_valid(self):
        result = build_reply_prompt(context="Some context", last_message="")
        assert "### Your reply:" in result

    def test_no_facts_no_graph_no_exchanges(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
            similar_exchanges=[],
            contact_facts="",
            relationship_graph="",
        )
        assert "(No similar past exchanges found)" in result
        assert "(none)" in result
        assert "<reply>" in result

    def test_prompt_builder_with_empty_context_docs_list(self):
        builder = PromptBuilder()
        req = ModelGenerationRequest(
            prompt="Task",
            context_documents=[],
            few_shot_examples=[],
        )
        result = builder.build(req)
        assert "### Relevant Context:" not in result
        assert "### Examples:" not in result
        assert "### Your Task:" in result

    def test_pipeline_request_with_empty_docs_and_examples(self):
        context = MessageContext(
            chat_id="chat1",
            message_text="Hey",
            is_from_me=False,
            timestamp=datetime(2026, 1, 1),
            metadata={},
        )
        classification = ClassificationResult(
            intent=IntentType.GREETING,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.LOW,
            confidence=0.9,
            requires_knowledge=False,
        )
        req = PipelineGenerationRequest(
            context=context,
            classification=classification,
            extraction=None,
            retrieved_docs=[],
            few_shot_examples=[],
        )
        result = build_prompt_from_request(req)
        assert "<system>" in result
        assert "<reply>" in result
        assert "Hey" in result


# =============================================================================
# Special Characters and Unicode Tests
# =============================================================================


class TestSpecialCharacters:
    """Test prompt assembly with special characters."""

    def test_unicode_emoji_in_message(self):
        result = build_reply_prompt(
            context="[10:00] Alice: Hey there!",
            last_message="See you soon! \U0001f44b\U0001f60a",
        )
        assert "\U0001f44b" in result
        assert "\U0001f60a" in result

    def test_chinese_characters(self):
        result = build_rag_reply_prompt(
            context="[10:00] Alice: \u4f60\u597d",
            last_message="\u4f60\u597d",
            contact_name="Alice",
        )
        assert "\u4f60\u597d" in result

    def test_curly_braces_in_message(self):
        result = build_rag_reply_prompt(
            context="Test with {braces}",
            last_message="Check this {json: value}",
            contact_name="Alice",
        )
        assert "{braces}" in result or "braces" in result
        assert "<reply>" in result

    def test_newlines_in_message(self):
        multiline = "Line 1\nLine 2\nLine 3"
        result = build_reply_prompt(context=multiline, last_message="Last line")
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_quotes_in_message(self):
        result = build_rag_reply_prompt(
            context='She said "hello"',
            last_message="He replied 'sure'",
            contact_name="Alice",
        )
        assert '"hello"' in result
        assert "'sure'" in result

    def test_backslashes_in_message(self):
        result = build_rag_reply_prompt(
            context="path\\to\\file",
            last_message="C:\\Users\\test",
            contact_name="Alice",
        )
        assert "path\\to\\file" in result


# =============================================================================
# Token Budget Tests
# =============================================================================


class TestTokenBudget:
    """Test that prompts respect token limits."""

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_proportional(self):
        text = "a" * 400
        tokens = estimate_tokens(text)
        assert tokens == 100

    def test_short_prompt_within_limit(self):
        result = build_reply_prompt(context="Short", last_message="Short")
        assert is_within_token_limit(result)

    def test_rag_prompt_within_limit(self):
        result = build_rag_reply_prompt(
            context="A few messages here",
            last_message="Short message",
            contact_name="Alice",
            similar_exchanges=[("Hey", "Hi"), ("What's up?", "Not much")],
        )
        assert is_within_token_limit(result)

    def test_truncated_context_keeps_prompt_manageable(self):
        long_context = "Message line content here\n" * 1000
        result = build_reply_prompt(context=long_context, last_message="Short")
        assert len(result) < MAX_CONTEXT_CHARS * 3

    def test_is_within_token_limit_custom(self):
        text = "A" * 100
        assert is_within_token_limit(text, limit=50) is True
        assert is_within_token_limit(text, limit=10) is False


# =============================================================================
# Template Variable Completeness Tests
# =============================================================================


class TestTemplateVariables:
    """Test that all template placeholders are filled in every prompt type."""

    def test_reply_template_all_vars_filled(self):
        result = build_reply_prompt(
            context="Test context",
            last_message="Test message",
            instruction="Be brief",
            tone="casual",
        )
        unresolved = re.findall(r"\{[a-z_]+\}", result)
        assert unresolved == [], f"Unresolved: {unresolved}"

    def test_summary_template_all_vars_filled(self):
        result = build_summary_prompt(context="Test conversation", focus="action items")
        unresolved = re.findall(r"\{[a-z_]+\}", result)
        assert unresolved == [], f"Unresolved: {unresolved}"

    def test_search_template_all_vars_filled(self):
        result = build_search_answer_prompt(
            context="Test messages", question="Where is the meeting?"
        )
        unresolved = re.findall(r"\{[a-z_]+\}", result)
        assert unresolved == [], f"Unresolved: {unresolved}"

    def test_rag_template_all_vars_filled(self):
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
            similar_exchanges=[("hi", "hey")],
            instruction="Be brief",
            contact_facts="likes: coffee",
            relationship_graph="Alice -> friend -> Bob",
        )
        unresolved = re.findall(r"\{[a-z_]+\}", result)
        assert unresolved == [], f"Unresolved: {unresolved}"

    def test_summary_template_without_focus_no_leftover(self):
        result = build_summary_prompt(context="Test conversation")
        assert "{focus_instruction}" not in result


# =============================================================================
# Context Document / RAG Document Formatting Tests
# =============================================================================


class TestContextDocumentFormatting:
    """Test that RAG documents are properly formatted in prompts."""

    def test_single_rag_document(self):
        docs = [
            RAGDocument(
                content="Want to grab dinner?",
                source="chat1",
                score=0.95,
                metadata={"response_text": "Sure, 7pm?"},
            ),
        ]
        context = MessageContext(
            chat_id="chat1",
            message_text="Want to eat?",
            is_from_me=False,
            timestamp=datetime(2026, 1, 1),
            metadata={"instruction": ""},
        )
        classification = ClassificationResult(
            intent=IntentType.QUESTION,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.8,
            requires_knowledge=True,
        )
        req = PipelineGenerationRequest(
            context=context,
            classification=classification,
            extraction=None,
            retrieved_docs=docs,
            few_shot_examples=[],
        )
        result = build_prompt_from_request(req)
        assert "Want to grab dinner?" in result
        assert "Sure, 7pm?" in result

    def test_multiple_rag_documents_ordered(self):
        docs = [
            RAGDocument(
                content=f"Trigger {i}",
                source="chat1",
                score=0.9 - i * 0.1,
                metadata={"response_text": f"Response {i}"},
            )
            for i in range(5)
        ]
        context = MessageContext(
            chat_id="chat1",
            message_text="Test",
            is_from_me=False,
            timestamp=datetime(2026, 1, 1),
            metadata={"instruction": ""},
        )
        classification = ClassificationResult(
            intent=IntentType.QUESTION,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.8,
            requires_knowledge=True,
        )
        req = PipelineGenerationRequest(
            context=context,
            classification=classification,
            extraction=None,
            retrieved_docs=docs,
            few_shot_examples=[],
        )
        result = build_prompt_from_request(req)
        assert "Trigger 0" in result
        assert "Trigger 1" in result
        assert "Trigger 2" in result


# =============================================================================
# Example Formatting Tests
# =============================================================================


class TestExampleFormatting:
    """Test low-level example formatting functions."""

    def test_format_few_shot_examples(self):
        examples = [
            FewShotExample(context="Hey there", output="Hi!"),
            FewShotExample(context="What's up?", output="Not much"),
        ]
        result = _format_examples(examples)
        assert "Context: Hey there\nReply: Hi!" in result
        assert "Context: What's up?\nReply: Not much" in result
        assert "Hi!\n\nContext: What's up?" in result

    def test_format_summary_examples(self):
        examples = [
            ("Alice: Let's meet at 3\nBob: Sure", "Summary: Meeting at 3"),
        ]
        result = _format_summary_examples(examples)
        assert "Conversation:\nAlice: Let's meet at 3\nBob: Sure" in result
        assert "Summary: Meeting at 3" in result

    def test_format_search_examples(self):
        examples = [
            ("Alice: Coffee at 3", "When is coffee?", "At 3"),
        ]
        result = _format_search_examples(examples)
        assert "Messages:\nAlice: Coffee at 3" in result
        assert "Question: When is coffee?" in result
        assert "Answer: At 3" in result

    def test_format_similar_exchanges_empty(self):
        result = _format_similar_exchanges([])
        assert result == "(No similar past exchanges found)"

    def test_format_similar_exchanges_truncates_long_context(self):
        long_ctx = "A" * 300
        exchanges = [(long_ctx, "short reply")]
        result = _format_similar_exchanges(exchanges)
        assert "..." in result
        assert len(result.split("Context: ")[1].split("\n")[0]) < 210

    def test_format_similar_exchanges_max_three(self):
        exchanges = [(f"ctx {i}", f"resp {i}") for i in range(10)]
        result = _format_similar_exchanges(exchanges)
        assert "Example 3:" in result
        assert "Example 4:" not in result


# =============================================================================
# Format Facts for Prompt Tests
# =============================================================================


class TestFormatFactsForPrompt:
    """Test format_facts_for_prompt output."""

    def test_empty_facts_returns_empty(self):
        assert format_facts_for_prompt([]) == ""

    def test_facts_below_confidence_excluded(self):
        from jarvis.contacts.contact_profile import Fact

        facts = [
            Fact(
                category="personal",
                subject="coffee",
                predicate="likes",
                value="",
                confidence=0.3,
                source_text="test",
            ),
        ]
        result = format_facts_for_prompt(facts)
        assert result == ""

    def test_facts_formatted_compactly(self):
        from jarvis.contacts.contact_profile import Fact

        facts = [
            Fact(
                category="personal",
                subject="Austin",
                predicate="lives_in",
                value="",
                confidence=0.9,
                source_text="test",
            ),
            Fact(
                category="work",
                subject="Google",
                predicate="works_at",
                value="",
                confidence=0.8,
                source_text="test",
            ),
        ]
        result = format_facts_for_prompt(facts)
        assert "lives_in: Austin" in result
        assert "works_at: Google" in result
        assert ", " in result

    def test_max_facts_limit(self):
        from jarvis.contacts.contact_profile import Fact

        facts = [
            Fact(
                category="personal",
                subject=f"fact_{i}",
                predicate=f"pred_{i}",
                value="",
                confidence=0.9,
                source_text="test",
            )
            for i in range(20)
        ]
        result = format_facts_for_prompt(facts, max_facts=3)
        assert result.count("pred_") == 3


# =============================================================================
# Relationship Context Formatting Tests
# =============================================================================


class TestRelationshipContextFormatting:
    """Test _format_relationship_context output."""

    def test_basic_relationship_context(self):
        result = _format_relationship_context(
            contact_context=None,
            tone="casual",
            avg_length=25.0,
        )
        assert "Tone: casual" in result
        assert "brief messages" in result

    def test_very_short_messages_description(self):
        result = _format_relationship_context(
            contact_context=None,
            tone="casual",
            avg_length=10.0,
        )
        assert "very short messages" in result

    def test_longer_messages_description(self):
        result = _format_relationship_context(
            contact_context=None,
            tone="professional",
            avg_length=100.0,
        )
        assert "longer messages" in result

    def test_user_messages_trigger_style_analysis(self):
        result = _format_relationship_context(
            contact_context=None,
            tone="casual",
            avg_length=50.0,
            user_messages=["yeah", "k sounds good", "lol ok", "u coming?"],
        )
        assert "Tone:" in result
        assert "Avg length:" in result


# =============================================================================
# Summary and Search Prompt Tests
# =============================================================================


class TestSummaryPromptAssembly:
    """Test summary prompt assembly."""

    def test_summary_prompt_has_all_sections(self):
        result = build_summary_prompt(context="Test conversation")
        sections = ["### Conversation:", "### Instructions:", "### Examples:", "### Summary:"]
        for section in sections:
            assert section in result, f"Missing section: {section}"

    def test_summary_with_focus(self):
        result = build_summary_prompt(context="Test", focus="action items")
        assert "Focus especially on: action items" in result

    def test_summary_without_focus(self):
        result = build_summary_prompt(context="Test", focus=None)
        assert "Focus especially on" not in result

    def test_summary_ends_with_marker(self):
        result = build_summary_prompt(context="Test")
        assert result.rstrip().endswith("### Summary:")


class TestSearchPromptAssembly:
    """Test search answer prompt assembly."""

    def test_search_prompt_has_all_sections(self):
        result = build_search_answer_prompt(context="Messages", question="Where?")
        sections = [
            "### Messages:",
            "### Question:",
            "### Instructions:",
            "### Examples:",
            "### Answer:",
        ]
        for section in sections:
            assert section in result, f"Missing section: {section}"

    def test_question_in_prompt(self):
        result = build_search_answer_prompt(
            context="Messages", question="What time is the meeting?"
        )
        assert "What time is the meeting?" in result

    def test_search_ends_with_marker(self):
        result = build_search_answer_prompt(context="Test", question="Test?")
        assert result.rstrip().endswith("### Answer:")


# =============================================================================
# PromptBuilder + RAG Prompt Integration Test
# =============================================================================


class TestPromptBuilderWithRagPrompt:
    """Test PromptBuilder passthrough with RAG-generated prompts."""

    def test_rag_prompt_passes_through_prompt_builder(self):
        rag_prompt = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
        )
        builder = PromptBuilder()
        model_req = ModelGenerationRequest(
            prompt=rag_prompt,
            context_documents=[],
            few_shot_examples=[],
        )
        result = builder.build(model_req)
        assert result == rag_prompt
        assert "### Your Task:" not in result

    def test_rag_prompt_gets_wrapped_if_extras_added(self):
        rag_prompt = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
        )
        builder = PromptBuilder()
        model_req = ModelGenerationRequest(
            prompt=rag_prompt,
            context_documents=["Additional context"],
        )
        result = builder.build(model_req)
        assert "### Relevant Context:" in result
        assert "### Your Task:" in result
        assert "Additional context" in result


# =============================================================================
# System Prefix Tests
# =============================================================================


class TestSystemPrefix:
    """Test SYSTEM_PREFIX used for KV cache reuse."""

    def test_system_prefix_is_nonempty(self):
        assert SYSTEM_PREFIX
        assert len(SYSTEM_PREFIX) > 50

    def test_system_prefix_starts_with_system_tag(self):
        assert SYSTEM_PREFIX.startswith("<system>")

    def test_system_prefix_contains_rules(self):
        assert "Rules:" in SYSTEM_PREFIX
        assert "Match their texting style" in SYSTEM_PREFIX

    def test_rag_prompt_template_starts_with_system_prefix(self):
        assert RAG_REPLY_PROMPT.template.startswith(SYSTEM_PREFIX.rstrip())


# =============================================================================
# Edge Case: Very Long Inputs
# =============================================================================


class TestVeryLongInputs:
    """Test behavior with extremely long inputs."""

    def test_very_long_message_text(self):
        long_msg = "A" * 10000
        result = build_rag_reply_prompt(
            context=long_msg,
            last_message=long_msg,
            contact_name="Alice",
        )
        assert "[Earlier messages truncated]" in result
        assert "<reply>" in result

    def test_many_similar_exchanges(self):
        exchanges = [(f"ctx {i}", f"resp {i}") for i in range(50)]
        result = build_rag_reply_prompt(
            context="Test",
            last_message="Test",
            contact_name="Alice",
            similar_exchanges=exchanges,
        )
        assert "Example 3:" in result
        assert "Example 4:" not in result

    def test_many_context_documents_in_prompt_builder(self):
        builder = PromptBuilder()
        req = ModelGenerationRequest(
            prompt="Task",
            context_documents=[f"Document {i}" for i in range(20)],
        )
        result = builder.build(req)
        assert "Document 0" in result
        assert "Document 19" in result
        assert result.count("---") == 19
