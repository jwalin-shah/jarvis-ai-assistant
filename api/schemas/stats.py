"""Statistics models.

Contains schemas for conversation statistics and analytics.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TimeRangeEnum(StrEnum):
    """Time range options for statistics calculation."""

    WEEK = "week"
    MONTH = "month"
    THREE_MONTHS = "three_months"
    ALL_TIME = "all_time"


class HourlyActivity(BaseModel):
    """Hourly message activity data point.

    Represents the number of messages sent/received during a specific hour.

    Example:
        ```json
        {
            "hour": 14,
            "count": 45
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hour": 14,
                "count": 45,
            }
        }
    )

    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day (0-23)",
        examples=[9, 14, 20],
    )
    count: int = Field(
        ...,
        ge=0,
        description="Number of messages during this hour",
        examples=[45, 23, 78],
    )


class WordFrequency(BaseModel):
    """Word frequency data for conversation analytics.

    Represents how often a specific word appears in conversations.

    Example:
        ```json
        {
            "word": "hello",
            "count": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "word": "hello",
                "count": 50,
            }
        }
    )

    word: str = Field(
        ...,
        description="The word",
        examples=["hello", "thanks", "meeting"],
    )
    count: int = Field(
        ...,
        ge=0,
        description="Number of occurrences",
        examples=[50, 35, 28],
    )


class ConversationStatsResponse(BaseModel):
    """Comprehensive conversation statistics response.

    Contains analytics and insights about messaging patterns in a conversation.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "time_range": "month",
            "total_messages": 500,
            "sent_count": 250,
            "received_count": 250,
            "avg_response_time_minutes": 15.5,
            "hourly_activity": [{"hour": 9, "count": 45}],
            "daily_activity": {"Monday": 80, "Tuesday": 70},
            "message_length_distribution": {
                "short": 200,
                "medium": 200,
                "long": 80,
                "very_long": 20
            },
            "top_words": [{"word": "hello", "count": 50}],
            "emoji_usage": {"heart": 25, "smile": 20},
            "attachment_breakdown": {"images": 30, "videos": 5}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "time_range": "month",
                "total_messages": 500,
                "sent_count": 250,
                "received_count": 250,
                "avg_response_time_minutes": 15.5,
                "hourly_activity": [
                    {"hour": 9, "count": 45},
                    {"hour": 10, "count": 52},
                ],
                "daily_activity": {
                    "Monday": 80,
                    "Tuesday": 70,
                    "Wednesday": 65,
                },
                "message_length_distribution": {
                    "short": 200,
                    "medium": 200,
                    "long": 80,
                    "very_long": 20,
                },
                "top_words": [
                    {"word": "hello", "count": 50},
                    {"word": "thanks", "count": 35},
                ],
                "emoji_usage": {"❤️": 25, "😊": 20},
                "attachment_breakdown": {"images": 30, "videos": 5},
                "first_message_date": "2024-01-01T10:00:00Z",
                "last_message_date": "2024-01-15T18:30:00Z",
                "participants": ["+15551234567"],
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation identifier",
        examples=["chat123456789"],
    )
    time_range: TimeRangeEnum = Field(
        ...,
        description="Time range used for statistics",
    )
    total_messages: int = Field(
        ...,
        ge=0,
        description="Total number of messages analyzed",
        examples=[500, 1000],
    )
    sent_count: int = Field(
        ...,
        ge=0,
        description="Number of messages sent by user",
        examples=[250, 480],
    )
    received_count: int = Field(
        ...,
        ge=0,
        description="Number of messages received",
        examples=[250, 520],
    )
    avg_response_time_minutes: float | None = Field(
        default=None,
        description="Average response time in minutes (within 24h window)",
        examples=[15.5, 8.2],
    )
    hourly_activity: list[HourlyActivity] = Field(
        default_factory=list,
        description="Message count by hour of day (0-23)",
    )
    daily_activity: dict[str, int] = Field(
        default_factory=dict,
        description="Message count by day of week",
        examples=[{"Monday": 80, "Tuesday": 70}],
    )
    message_length_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of message lengths (short/medium/long/very_long)",
        examples=[{"short": 200, "medium": 200, "long": 80, "very_long": 20}],
    )
    top_words: list[WordFrequency] = Field(
        default_factory=list,
        description="Most frequently used words",
    )
    emoji_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Most frequently used emojis with counts",
        examples=[{"❤️": 25, "😊": 20}],
    )
    attachment_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Attachment counts by type (images/videos/audio/documents/other)",
        examples=[{"images": 30, "videos": 5}],
    )
    first_message_date: datetime | None = Field(
        default=None,
        description="Date of earliest message in the analyzed range",
    )
    last_message_date: datetime | None = Field(
        default=None,
        description="Date of most recent message in the analyzed range",
    )
    participants: list[str] = Field(
        default_factory=list,
        description="List of conversation participants",
        examples=[["+15551234567", "+15559876543"]],
    )
