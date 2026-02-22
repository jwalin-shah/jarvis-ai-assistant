from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime

from src.config import AppConfig, SamplingParams


@dataclass
class GeneratedCandidate:
    strategy: str
    reply: str
    raw: str


class ReplyGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._using_fallback = True
        self._try_load_model()

    def _try_load_model(self) -> None:
        if os.getenv("TEXT_REPLY_SYSTEM_SKIP_MODEL_LOAD") == "1":
            self._model = None
            self._tokenizer = None
            self._using_fallback = True
            return
        from mlx_lm import load  # type: ignore

        self._model = None
        self._tokenizer = None
        self._using_fallback = True
        for model_id in self.config.model_candidates("generator"):
            try:
                self._model, self._tokenizer = load(model_id)
                self._using_fallback = False
                return
            except Exception:
                continue

    def build_prompt(
        self,
        incoming_message: str,
        recent_messages: list[str],
        category: str,
        contact_name: str,
        relationship: str,
    ) -> str:
        cfg = self.config.runtime
        ts = datetime.now().strftime(cfg.timestamp_format)
        strategy_template = self.config.strategy_templates.get(category, self.config.strategy_templates["casual"])
        context = "\n".join(recent_messages[-5:]) if recent_messages else "(none)"
        return (
            f"<|im_start|>system\n"
            f"You are texting as {cfg.user_name}. You're messaging {contact_name} ({relationship}).\n"
            f"Current time: {ts}. Response type: {category}.\n\n"
            f"Strategy guidance: {strategy_template}\n\n"
            f"Write a <strategy> tag (5-15 words about tone and intent) followed by your text reply.\n"
            f"Keep the reply natural and concise - typical text message length.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Recent conversation:\n{context}\n\n"
            f"Latest message from {contact_name}: \"{incoming_message}\"\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<strategy>"
        )

    def generate_candidates(
        self,
        incoming_message: str,
        category: str,
        recent_messages: list[str],
        contact_name: str,
        relationship: str,
    ) -> list[GeneratedCandidate]:
        params = self.config.sampling.get(category, self.config.sampling["casual"])
        prompt = self.build_prompt(
            incoming_message=incoming_message,
            recent_messages=recent_messages,
            category=category,
            contact_name=contact_name,
            relationship=relationship,
        )

        raw_outputs: list[str]
        if self._using_fallback:
            raw_outputs = self._fallback_generate(incoming_message, category, params)
        else:
            raw_outputs = self._model_generate(prompt, params, n_samples=params.n_samples, max_tokens=120)
            if not raw_outputs:
                raw_outputs = self._fallback_generate(incoming_message, category, params)

        parsed = [self._parse_candidate(o) for o in raw_outputs]
        cleaned = self._filter_candidates(parsed)

        retry_limit = self.config.runtime.candidate_retry_limit
        min_valid = self.config.runtime.min_valid_candidates
        retries = 0
        while len(cleaned) < min_valid and retries < retry_limit:
            retries += 1
            if self._using_fallback:
                extra = self._fallback_generate(incoming_message, category, params, n_override=max(2, params.n_samples // 2))
            else:
                extra = self._model_generate(prompt, params, n_samples=max(2, params.n_samples // 2), max_tokens=120)
                if not extra:
                    extra = self._fallback_generate(incoming_message, category, params, n_override=max(2, params.n_samples // 2))
            cleaned = self._filter_candidates(cleaned + [self._parse_candidate(o) for o in extra])

        _ = prompt
        return cleaned

    def generate_generic_replies(
        self,
        incoming_message: str,
        recent_messages: list[str],
        n_samples: int = 4,
    ) -> list[str]:
        """Generate generic alternatives (used for rejected preference samples)."""
        params = SamplingParams(
            temperature=0.7,
            min_p=0.08,
            rep_penalty=1.0,
            n_samples=n_samples,
        )
        context = "\n".join(recent_messages[-5:]) if recent_messages else "(none)"
        prompt = (
            "<|im_start|>system\n"
            "Write a natural, concise text reply. Keep it generic and neutral.\n"
            "Do not include strategy tags or explanations.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Recent conversation:\n{context}\n\n"
            f"Latest incoming message: \"{incoming_message}\"\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        if self._using_fallback:
            raws = self._fallback_generate(incoming_message, "casual", params, n_override=n_samples)
            return [self._parse_candidate(x).reply for x in raws]

        raws = self._model_generate(prompt, params, n_samples=n_samples, max_tokens=80)
        if not raws:
            raws = self._fallback_generate(incoming_message, "casual", params, n_override=n_samples)
        replies = [x.strip() for x in raws if x and x.strip()]
        dedup: dict[str, str] = {}
        for r in replies:
            dedup.setdefault(r.lower(), r)
        return list(dedup.values())

    def _model_generate(
        self,
        prompt: str,
        params: SamplingParams,
        n_samples: int,
        max_tokens: int,
    ) -> list[str]:
        if self._model is None or self._tokenizer is None:
            return []
        try:
            from mlx_lm import generate  # type: ignore
            from mlx_lm.sample_utils import make_sampler  # type: ignore

            sampler = make_sampler(temp=params.temperature, min_p=params.min_p)
            outputs: list[str] = []
            for _ in range(max(1, n_samples)):
                text = generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                text = (text or "").strip()
                if text:
                    outputs.append(text)
            return outputs
        except Exception:
            return []

    def _fallback_generate(
        self,
        incoming_message: str,
        category: str,
        params: SamplingParams,
        n_override: int | None = None,
    ) -> list[str]:
        n = n_override or params.n_samples
        base_strategy = self.config.strategy_templates.get(category, "be concise and natural")
        msg = incoming_message.strip().rstrip("?")
        outputs = []
        for i in range(n):
            strategy = f"{base_strategy}; variant {i + 1}"
            reply = self._simple_reply(msg, category, i)
            outputs.append(f"<strategy>{strategy}</strategy> {reply}")
        return outputs

    @staticmethod
    def _simple_reply(msg: str, category: str, seed: int) -> str:
        variants = {
            "question": [
                "Yeah, short answer: yes. Want me to send details?",
                "I think so - I'd do it this way. Want a quick breakdown?",
                "Good question. I'd say yes, with a small tweak.",
            ],
            "logistics": [
                "Yep - I'm on it. I can be there in about 20 mins.",
                "Works for me. Let's lock 7:30 at the same spot.",
                "Confirmed. I'll text when I'm leaving.",
            ],
            "emotional": [
                "I'm really sorry you're dealing with that. Want to talk it out?",
                "That sounds heavy - I'm here for you.",
                "I hear you. If you want, we can figure this out together.",
            ],
        }
        generic = [
            f"Haha fair - {msg.lower()} works for me.",
            "I'm down. Keep me posted.",
            "Totally. Let's do it.",
        ]
        bank = variants.get(category, generic)
        return bank[seed % len(bank)]

    @staticmethod
    def _parse_candidate(raw: str) -> GeneratedCandidate:
        m = re.search(r"<strategy>(.*?)</strategy>", raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            strategy = m.group(1).strip()
            reply = (raw[: m.start()] + raw[m.end() :]).strip()
        elif "</strategy>" in raw:
            before, after = raw.split("</strategy>", 1)
            strategy = before.replace("<strategy>", "").strip()
            reply = after.strip()
        else:
            strategy = ""
            reply = raw.strip()
        return GeneratedCandidate(strategy=strategy, reply=reply, raw=raw)

    @staticmethod
    def _filter_candidates(candidates: list[GeneratedCandidate]) -> list[GeneratedCandidate]:
        dedup: dict[str, GeneratedCandidate] = {}
        for c in candidates:
            reply = c.reply.strip()
            if not reply:
                continue
            if len(reply.split()) < 2:
                continue
            if any(ch * 8 in reply for ch in ["?", "!", "."]):
                continue
            dedup.setdefault(reply.lower(), c)
        return list(dedup.values())
