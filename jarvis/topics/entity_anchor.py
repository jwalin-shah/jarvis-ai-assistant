"""Entity Anchor - Lightweight entity tracking for topic segmentation.

Uses a hybrid approach:
1. Contact-Aware spaCy (EntityRuler seeded from DB)
2. Noun-Phrase extraction (Syntactic anchors)
3. BGE-Similarity matching (Semantic anchors)
"""

from __future__ import annotations

import logging
import sqlite3

import spacy
from spacy.pipeline import EntityRuler

logger = logging.getLogger(__name__)


class EntityAnchorTracker:
    def __init__(self, jarvis_db_path: str | None = None) -> None:
        # 1. Load tiny spaCy model (12MB)
        try:
            self.nlp = spacy.load(
                "en_core_web_sm", disable=["ner"]
            )  # Disable default NER, we'll build a better one
        except OSError:
            # Fallback if not installed
            import subprocess  # nosec B404

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)  # nosec B603 B607
            self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

        # 2. Add custom EntityRuler for high-precision local matching
        self.ruler: EntityRuler = self.nlp.add_pipe("entity_ruler")  # type: ignore[assignment]

        # 3. Seed ruler with contact names from DB
        if jarvis_db_path:
            self._seed_from_contacts(jarvis_db_path)

    def _seed_from_contacts(self, db_path: str) -> None:
        """Load all contact names into the ruler so spaCy always finds them."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT display_name FROM contacts WHERE display_name IS NOT NULL"
            )
            patterns = []
            for (name,) in cursor.fetchall():
                if len(name) > 2:
                    # Match exact name and lowercase version
                    patterns.append({"label": "PERSON", "pattern": name})
                    patterns.append({"label": "PERSON", "pattern": name.lower()})

            self.ruler.add_patterns(patterns)
            conn.close()
            logger.info(f"Seeded EntityAnchor with {len(patterns)} contact patterns")
        except Exception as e:
            logger.debug(f"Could not seed contacts: {e}")

    def get_anchors(self, text: str) -> set[str]:
        """Extract set of entity anchors from text."""
        if not text:
            return set()

        doc = self.nlp(text)
        anchors = set()

        # 1. Add detected entities (Contacts, Orgs, etc.)
        for ent in doc.ents:
            anchors.add(ent.text.lower())

        # 2. Add Noun Chunks (Span-based keywords)
        # Filters out simple pronouns like "it", "me", "you"
        for chunk in doc.noun_chunks:
            chunk_text = chunk.root.text.lower()
            if len(chunk_text) > 2 and chunk.root.pos_ != "PRON":
                anchors.add(chunk_text)

        return anchors


_tracker = None


def get_tracker(db_path: str | None = None) -> EntityAnchorTracker:
    global _tracker
    if _tracker is None:
        # Default path
        import os

        if not db_path:
            db_path = os.path.expanduser("~/.jarvis/jarvis.db")
        _tracker = EntityAnchorTracker(db_path)
    return _tracker
