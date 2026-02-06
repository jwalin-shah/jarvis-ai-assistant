"""Pair artifacts, style targets, and validated pair operations mixin."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from jarvis.db.models import ContactStyleTargets, Pair, PairArtifact

if TYPE_CHECKING:
    from jarvis.db.core import JarvisDBBase


class ArtifactMixin:
    """Mixin providing pair artifact, style target, and validated pair operations."""

    def add_artifact(
        self: JarvisDBBase,
        pair_id: int,
        context_json: str | None = None,
        gate_a_reason: str | None = None,
        gate_c_scores_json: str | None = None,
        raw_trigger_text: str | None = None,
        raw_response_text: str | None = None,
    ) -> PairArtifact:
        """Add or update artifacts for a pair.

        Args:
            pair_id: ID of the pair these artifacts belong to.
            context_json: Structured context window (JSON list).
            gate_a_reason: Why Gate A rejected (if rejected).
            gate_c_scores_json: Raw NLI scores (JSON dict).
            raw_trigger_text: Original text before normalization.
            raw_response_text: Original text before normalization.

        Returns:
            The created or updated PairArtifact.
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pair_artifacts
                (pair_id, context_json, gate_a_reason, gate_c_scores_json,
                 raw_trigger_text, raw_response_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pair_id,
                    context_json,
                    gate_a_reason,
                    gate_c_scores_json,
                    raw_trigger_text,
                    raw_response_text,
                ),
            )
            return PairArtifact(
                pair_id=pair_id,
                context_json=context_json,
                gate_a_reason=gate_a_reason,
                gate_c_scores_json=gate_c_scores_json,
                raw_trigger_text=raw_trigger_text,
                raw_response_text=raw_response_text,
            )

    def get_artifact(self: JarvisDBBase, pair_id: int) -> PairArtifact | None:
        """Get artifacts for a pair."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM pair_artifacts WHERE pair_id = ?", (pair_id,))
            row = cursor.fetchone()
            if row:
                return PairArtifact(
                    pair_id=row["pair_id"],
                    context_json=row["context_json"],
                    gate_a_reason=row["gate_a_reason"],
                    gate_c_scores_json=row["gate_c_scores_json"],
                    raw_trigger_text=row["raw_trigger_text"],
                    raw_response_text=row["raw_response_text"],
                )
            return None

    def clear_artifacts(self: JarvisDBBase) -> int:
        """Delete all artifacts."""
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM pair_artifacts")
            return cursor.rowcount

    # -------------------------------------------------------------------------
    # Contact Style Targets Operations (v6+)
    # -------------------------------------------------------------------------

    def set_style_targets(
        self: JarvisDBBase,
        contact_id: int,
        median_reply_length: int = 10,
        punctuation_rate: float = 0.5,
        emoji_rate: float = 0.1,
        greeting_rate: float = 0.2,
    ) -> ContactStyleTargets:
        """Set style targets for a contact.

        Args:
            contact_id: Contact ID.
            median_reply_length: Median word count.
            punctuation_rate: Fraction with ending punctuation.
            emoji_rate: Fraction containing emojis.
            greeting_rate: Fraction starting with greeting.

        Returns:
            The created or updated ContactStyleTargets.
        """
        with self.connection() as conn:
            now = datetime.now()
            conn.execute(
                """
                INSERT OR REPLACE INTO contact_style_targets
                (contact_id, median_reply_length, punctuation_rate,
                 emoji_rate, greeting_rate, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (contact_id, median_reply_length, punctuation_rate, emoji_rate, greeting_rate, now),
            )
            return ContactStyleTargets(
                contact_id=contact_id,
                median_reply_length=median_reply_length,
                punctuation_rate=punctuation_rate,
                emoji_rate=emoji_rate,
                greeting_rate=greeting_rate,
                updated_at=now,
            )

    def get_style_targets(self: JarvisDBBase, contact_id: int) -> ContactStyleTargets | None:
        """Get style targets for a contact."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM contact_style_targets WHERE contact_id = ?",
                (contact_id,),
            )
            row = cursor.fetchone()
            if row:
                return ContactStyleTargets(
                    contact_id=row["contact_id"],
                    median_reply_length=row["median_reply_length"],
                    punctuation_rate=row["punctuation_rate"],
                    emoji_rate=row["emoji_rate"],
                    greeting_rate=row["greeting_rate"],
                    updated_at=row["updated_at"],
                )
            return None

    # -------------------------------------------------------------------------
    # Validated Pair Operations (v6+ extraction pipeline)
    # -------------------------------------------------------------------------

    def add_validated_pair(
        self: JarvisDBBase,
        trigger_text: str,
        response_text: str,
        trigger_timestamp: datetime,
        response_timestamp: datetime,
        chat_id: str,
        contact_id: int | None = None,
        trigger_msg_id: int | None = None,
        response_msg_id: int | None = None,
        trigger_msg_ids: list[int] | None = None,
        response_msg_ids: list[int] | None = None,
        quality_score: float = 1.0,
        flags: dict[str, Any] | None = None,
        is_group: bool = False,
        # Gate results
        gate_a_passed: bool = True,
        gate_b_score: float | None = None,
        gate_c_verdict: str | None = None,
        validity_status: str = "valid",
        # Artifacts (stored separately)
        context_json: str | None = None,
        gate_a_reason: str | None = None,
        gate_c_scores_json: str | None = None,
        raw_trigger_text: str | None = None,
        raw_response_text: str | None = None,
    ) -> Pair | None:
        """Add a validated pair with gate results and artifacts.

        This is the v6+ version of add_pair that includes validity gate
        results and stores heavy artifacts in a separate table.

        Returns:
            The created Pair, or None if duplicate.
        """
        with self.connection() as conn:
            trigger_msg_ids_json = json.dumps(trigger_msg_ids) if trigger_msg_ids else None
            response_msg_ids_json = json.dumps(response_msg_ids) if response_msg_ids else None
            flags_json = json.dumps(flags) if flags else None

            try:
                cursor = conn.execute(
                    """
                    INSERT INTO pairs
                    (contact_id, trigger_text, response_text, trigger_timestamp,
                     response_timestamp, chat_id, trigger_msg_id, response_msg_id,
                     trigger_msg_ids_json, response_msg_ids_json,
                     quality_score, flags_json, is_group,
                     gate_a_passed, gate_b_score, gate_c_verdict, validity_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contact_id,
                        trigger_text,
                        response_text,
                        trigger_timestamp,
                        response_timestamp,
                        chat_id,
                        trigger_msg_id,
                        response_msg_id,
                        trigger_msg_ids_json,
                        response_msg_ids_json,
                        quality_score,
                        flags_json,
                        is_group,
                        gate_a_passed,
                        gate_b_score,
                        gate_c_verdict,
                        validity_status,
                    ),
                )
                pair_id = cursor.lastrowid

                # Store artifacts in separate table if provided
                if context_json or gate_a_reason or gate_c_scores_json or raw_trigger_text:
                    conn.execute(
                        """
                        INSERT INTO pair_artifacts
                        (pair_id, context_json, gate_a_reason, gate_c_scores_json,
                         raw_trigger_text, raw_response_text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            pair_id,
                            context_json,
                            gate_a_reason,
                            gate_c_scores_json,
                            raw_trigger_text,
                            raw_response_text,
                        ),
                    )

                return Pair(
                    id=pair_id,
                    contact_id=contact_id,
                    trigger_text=trigger_text,
                    response_text=response_text,
                    trigger_timestamp=trigger_timestamp,
                    response_timestamp=response_timestamp,
                    chat_id=chat_id,
                    trigger_msg_id=trigger_msg_id,
                    response_msg_id=response_msg_id,
                    trigger_msg_ids_json=trigger_msg_ids_json,
                    response_msg_ids_json=response_msg_ids_json,
                    quality_score=quality_score,
                    flags_json=flags_json,
                    is_group=is_group,
                    gate_a_passed=gate_a_passed,
                    gate_b_score=gate_b_score,
                    gate_c_verdict=gate_c_verdict,
                    validity_status=validity_status,
                )
            except sqlite3.IntegrityError:
                # Duplicate pair
                return None
