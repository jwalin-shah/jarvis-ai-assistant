"""Snapshot tests for prompt builders.

These tests ensure that the exact text of prompts remains stable.
Any change to prompts should be intentional and verified, as it affects model behavior.
"""

from jarvis.prompts import build_reply_prompt, build_threaded_reply_prompt
from jarvis.threading import ThreadContext, ThreadTopic, ThreadState, UserRole
from contracts.imessage import Message

def test_reply_prompt_snapshot(snapshot):
    """Verify output of build_reply_prompt using syrupy snapshots."""
    context = "[10:00] John: Hello\\n[10:01] Me: Hi"
    last_message = "How are you?"
    
    prompt = build_reply_prompt(
        context=context,
        last_message=last_message,
        instruction="be nice",
        tone="casual"
    )
    
    assert prompt == snapshot

def test_threaded_reply_prompt_snapshot(snapshot):
    """Verify output of build_threaded_reply_prompt using syrupy snapshots."""
    # Setup minimal thread context
    thread_context = ThreadContext(
        messages=[
            Message(
                id=1,
                chat_id="chat1",
                sender="Alice",
                sender_name="Alice",
                text="Where are we meeting?",
                date=None,
                is_from_me=False
            ),
        ],
        topic=ThreadTopic.LOGISTICS,
        state=ThreadState.OPEN_QUESTION,
        user_role=UserRole.RESPONDER,
        confidence=1.0,
        relevant_messages=[], 
        action_items=[],
        participants_count=2
    )
    
    # Mock config
    class MockConfig:
        response_style = "concise"
        max_response_length = 50
        include_action_items = False
        suggest_follow_up = False
    
    prompt = build_threaded_reply_prompt(
        thread_context=thread_context,
        config=MockConfig(),
        instruction="confirm 5pm",
        tone="casual"
    )
    
    assert prompt == snapshot