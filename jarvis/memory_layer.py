"""Adaptive Memory Layer using Mem0.

Provides long-term, self-improving user memory that persists across 
conversations and sessions.
"""

import os
from mem0 import Memory
from jarvis.config import get_config

class JARVISMemory:
    """Memory layer wrapper for JARVIS using Mem0."""
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        # Configure Mem0 for local operation
        # Note: By default it uses local storage if no API key provided
        self.memory = Memory()

    def add_interaction(self, user_msg: str, assistant_msg: str):
        """Record an interaction to learn from it."""
        # Mem0 will extract facts automatically
        self.memory.add(f"User: {user_msg}
Assistant: {assistant_msg}", user_id=self.user_id)

    def get_relevant_facts(self, query: str):
        """Retrieve learned facts relevant to the current query."""
        results = self.memory.search(query, user_id=self.user_id)
        # Extract just the fact strings
        facts = [res['fact'] for res in results]
        return facts

    def delete_all(self):
        """Clear the memory for the user."""
        self.memory.delete_all(user_id=self.user_id)

# Global memory instance
_memory = None

def get_memory():
    """Get or create singleton memory instance."""
    global _memory
    if _memory is None:
        _memory = JARVISMemory()
    return _memory
