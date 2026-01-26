    calendar_id: str = Field(
        ...,
        description="Target calendar ID",
    )
    detected_event: DetectedEventResponse = Field(
        ...,
        description="Detected event to add to calendar",
    )


class CreateEventResponse(BaseModel):
    """Response after creating a calendar event.

    Example:
        ```json
        {
            "success": true,
            "event_id": "event-789",
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "event_id": "event-789",
                "error": None,
            }
        }
    )

    success: bool = Field(
        ...,
        description="Whether the event was created successfully",
    )
    event_id: str | None = Field(
        default=None,
        description="ID of created event (if successful)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (if failed)",
=======
    imported: int = Field(
        ...,
        ge=0,
        description="Number of templates successfully imported",
    )
    skipped: int = Field(
        ...,
        ge=0,
        description="Number of templates skipped",
    )
    errors: int = Field(
        ...,
        ge=0,
        description="Number of templates that failed to import",
    )
    total_templates: int = Field(
        ...,
        ge=0,
        description="Total templates after import",
>>>>>>> origin/claude/custom-template-builder-VPeOl
    )


# =============================================================================
# Threading Schemas
# =============================================================================


class ThreadResponse(BaseModel):
    """A conversation thread grouping related messages.

    Threads are auto-detected based on time gaps, topic shifts, and explicit
    reply references. They help organize long conversations into logical
    segments.

    Example:
        ```json
        {
            "thread_id": "a1b2c3d4e5f67890",
            "messages": [12345, 12346, 12347],
            "topic_label": "Dinner Plans",
            "start_time": "2024-01-15T18:00:00Z",
            "end_time": "2024-01-15T18:30:00Z",
            "participant_count": 2,
            "message_count": 3
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thread_id": "a1b2c3d4e5f67890",
                "messages": [12345, 12346, 12347],
                "topic_label": "Dinner Plans",
                "start_time": "2024-01-15T18:00:00Z",
                "end_time": "2024-01-15T18:30:00Z",
                "participant_count": 2,
                "message_count": 3,
            }
        }
    )

    thread_id: str = Field(
        ...,
        description="Unique thread identifier",
        examples=["a1b2c3d4e5f67890"],
    )
    messages: list[int] = Field(
        default_factory=list,
        description="List of message IDs in this thread",
        examples=[[12345, 12346, 12347]],
    )
    topic_label: str = Field(
        default="",
        description="Auto-generated topic label for the thread",
        examples=["Dinner Plans", "Work Discussion", "Weekend Plans"],
    )
    start_time: datetime | None = Field(
        default=None,
        description="Timestamp of first message in thread",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Timestamp of last message in thread",
    )
    participant_count: int = Field(
        default=0,
        description="Number of unique participants in thread",
        examples=[2, 3],
        ge=0,
    )
    message_count: int = Field(
        default=0,
        description="Number of messages in thread",
        examples=[3, 10, 25],
        ge=0,
    )


class ThreadedMessageResponse(MessageResponse):
    """Message with thread information attached.

    Extends MessageResponse with thread-specific fields to enable
    threaded conversation views.

    Example:
        ```json
        {
            "id": 12345,
            "chat_id": "chat123456789",
            "sender": "+15551234567",
            "text": "Are you free for dinner?",
            "date": "2024-01-15T18:00:00Z",
            "is_from_me": false,
            "thread_id": "a1b2c3d4e5f67890",
            "thread_position": 0,
            "is_thread_start": true
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 12345,
                "chat_id": "chat123456789",
                "sender": "+15551234567",
                "sender_name": "John Doe",
                "text": "Are you free for dinner?",
                "date": "2024-01-15T18:00:00Z",
                "is_from_me": False,
                "attachments": [],
                "reply_to_id": None,
                "reactions": [],
                "is_system_message": False,
                "thread_id": "a1b2c3d4e5f67890",
                "thread_position": 0,
                "is_thread_start": True,
            }
        },
    )

    thread_id: str = Field(
        ...,
        description="ID of the thread this message belongs to",
        examples=["a1b2c3d4e5f67890"],
    )
    thread_position: int = Field(
        default=0,
        description="Position within the thread (0-indexed)",
        examples=[0, 1, 2],
        ge=0,
    )
    is_thread_start: bool = Field(
        default=False,
        description="True if this message starts a new thread",
    )


class ThreadedViewResponse(BaseModel):
    """Response containing threaded view of a conversation.

    Provides both the list of threads and the messages with thread info,
    enabling the frontend to display a threaded conversation view.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "threads": [...],
            "messages": [...],
            "total_threads": 5,
            "total_messages": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "threads": [
                    {
                        "thread_id": "a1b2c3d4e5f67890",
                        "messages": [12345, 12346],
                        "topic_label": "Dinner Plans",
                        "message_count": 2,
                    }
                ],
                "messages": [
                    {
                        "id": 12345,
                        "text": "Are you free for dinner?",
                        "thread_id": "a1b2c3d4e5f67890",
                        "thread_position": 0,
                        "is_thread_start": True,
                    }
                ],
                "total_threads": 5,
                "total_messages": 50,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation identifier",
        examples=["chat123456789"],
    )
    threads: list[ThreadResponse] = Field(
        default_factory=list,
        description="List of detected threads",
    )
    messages: list[ThreadedMessageResponse] = Field(
        default_factory=list,
        description="Messages with thread information",
    )
    total_threads: int = Field(
        default=0,
        description="Total number of threads detected",
        examples=[5, 10],
        ge=0,
    )
    total_messages: int = Field(
        default=0,
        description="Total number of messages analyzed",
        examples=[50, 100],
        ge=0,
    )


class ThreadingConfigRequest(BaseModel):
    """Configuration options for thread analysis.

    Allows customizing how threads are detected.

    Example:
        ```json
        {
            "time_gap_threshold_minutes": 30,
            "semantic_similarity_threshold": 0.4,
            "use_semantic_analysis": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time_gap_threshold_minutes": 30,
                "semantic_similarity_threshold": 0.4,
                "use_semantic_analysis": True,
            }
        }
    )

    time_gap_threshold_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Minutes of silence to start a new thread (5-1440)",
        examples=[30, 60, 120],
    )
    semantic_similarity_threshold: float = Field(
        default=0.4,
        ge=0.1,
        le=0.9,
        description="Minimum similarity to group messages (0.1-0.9)",
        examples=[0.3, 0.4, 0.5],
    )
    use_semantic_analysis: bool = Field(
        default=True,
        description="Whether to use ML-based topic detection",
    )
