from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.classifier import ClassificationResult, ResponseClassifier
from src.config import AppConfig
from src.generator import GeneratedCandidate, ReplyGenerator
from src.reward_model import StyleRewardModel
from src.soft_bon import soft_best_of_n


@dataclass
class PipelineResult:
    final_reply: str
    category: str
    category_confidence: float
    used_majority_vote: bool
    candidates: list[GeneratedCandidate]
    scores: list[float]
    selected_score: float
    warning: str | None


class TextReplyPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.classifier = ResponseClassifier(config)
        self.generator = ReplyGenerator(config)
        self.reward_model = StyleRewardModel(config)

    def run(
        self,
        incoming_message: str,
        recent_messages: list[str] | None = None,
        contact_name: str | None = None,
        relationship: str | None = None,
    ) -> PipelineResult:
        recent_messages = recent_messages or []
        contact_name = contact_name or self.config.runtime.default_contact_name
        relationship = relationship or self.config.runtime.default_relationship

        cls: ClassificationResult = self.classifier.classify(incoming_message)
        if cls.category not in self.config.categories:
            cls = ClassificationResult(category="casual", confidence=0.5, used_majority_vote=cls.used_majority_vote)

        candidates = self.generator.generate_candidates(
            incoming_message=incoming_message,
            category=cls.category,
            recent_messages=recent_messages,
            contact_name=contact_name,
            relationship=relationship,
        )
        if not candidates:
            raise RuntimeError("No valid candidates generated")

        replies = [c.reply for c in candidates]
        self.reward_model.load_style_adapter()
        scores = self.reward_model.score_candidates(
            relationship=relationship,
            incoming_message=incoming_message,
            candidate_replies=replies,
        )

        selected, selected_score, fallback_used = soft_best_of_n(
            candidates=replies,
            scores=scores,
            temperature=self.config.soft_bon.temperature,
            min_threshold=self.config.soft_bon.min_score_threshold,
        )

        warning = None
        if fallback_used:
            warning = "all_candidates_below_threshold"

        return PipelineResult(
            final_reply=selected,
            category=cls.category,
            category_confidence=cls.confidence,
            used_majority_vote=cls.used_majority_vote,
            candidates=candidates,
            scores=scores,
            selected_score=selected_score,
            warning=warning,
        )

    def debug_dict(self, result: PipelineResult) -> dict[str, Any]:
        return {
            "final_reply": result.final_reply,
            "category": result.category,
            "category_confidence": result.category_confidence,
            "used_majority_vote": result.used_majority_vote,
            "selected_score": result.selected_score,
            "warning": result.warning,
            "candidates": [c.reply for c in result.candidates],
            "strategies": [c.strategy for c in result.candidates],
            "scores": result.scores,
        }
