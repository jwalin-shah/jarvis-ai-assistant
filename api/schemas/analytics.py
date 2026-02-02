"""Analytics, insights, and profile models.

Contains schemas for sentiment analysis, relationship health, conversation insights,
and relationship profiles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from api.schemas.stats import TimeRangeEnum


class SentimentResponse(BaseModel):
    """Sentiment analysis result.

    Provides sentiment score and breakdown of positive/negative signals.

    Example:
        ```json
        {
            "score": 0.45,
            "label": "positive",
            "positive_count": 120,
            "negative_count": 30,
            "neutral_count": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "score": 0.45,
                "label": "positive",
                "positive_count": 120,
                "negative_count": 30,
                "neutral_count": 50,
            }
        }
    )

    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score from -1.0 (negative) to 1.0 (positive)",
        examples=[0.45, -0.2, 0.0],
    )
    label: str = Field(
        ...,
        description="Sentiment label: 'positive', 'negative', or 'neutral'",
        examples=["positive", "negative", "neutral"],
    )
    positive_count: int = Field(
        default=0,
        ge=0,
        description="Number of positive signals detected",
        examples=[120, 45],
    )
    negative_count: int = Field(
        default=0,
        ge=0,
        description="Number of negative signals detected",
        examples=[30, 15],
    )
    neutral_count: int = Field(
        default=0,
        ge=0,
        description="Number of neutral messages",
        examples=[50, 100],
    )


class SentimentTrendResponse(BaseModel):
    """Sentiment trend data point for a time period.

    Example:
        ```json
        {
            "date": "2024-W03",
            "score": 0.35,
            "message_count": 45
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "date": "2024-W03",
                "score": 0.35,
                "message_count": 45,
            }
        }
    )

    date: str = Field(
        ...,
        description="Period identifier (YYYY-MM-DD, YYYY-WNN, or YYYY-MM)",
        examples=["2024-W03", "2024-01-15", "2024-01"],
    )
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Average sentiment score for the period",
        examples=[0.35, -0.1],
    )
    message_count: int = Field(
        ...,
        ge=0,
        description="Number of messages in this period",
        examples=[45, 120],
    )


class ResponsePatternsResponse(BaseModel):
    """Response time pattern analysis.

    Provides detailed analysis of response times between participants.

    Example:
        ```json
        {
            "avg_response_time_minutes": 15.5,
            "median_response_time_minutes": 8.0,
            "fastest_response_minutes": 0.5,
            "slowest_response_minutes": 480.0,
            "my_avg_response_time_minutes": 12.0,
            "their_avg_response_time_minutes": 18.5
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "avg_response_time_minutes": 15.5,
                "median_response_time_minutes": 8.0,
                "fastest_response_minutes": 0.5,
                "slowest_response_minutes": 480.0,
                "response_times_by_hour": {"9": 5.2, "14": 12.5, "20": 25.0},
                "response_times_by_day": {"Monday": 10.5, "Saturday": 45.0},
                "my_avg_response_time_minutes": 12.0,
                "their_avg_response_time_minutes": 18.5,
            }
        }
    )

    avg_response_time_minutes: float | None = Field(
        default=None,
        description="Average response time in minutes",
        examples=[15.5, 8.0],
    )
    median_response_time_minutes: float | None = Field(
        default=None,
        description="Median response time in minutes",
        examples=[8.0, 5.0],
    )
    fastest_response_minutes: float | None = Field(
        default=None,
        description="Fastest response time in minutes",
        examples=[0.5, 1.0],
    )
    slowest_response_minutes: float | None = Field(
        default=None,
        description="Slowest response time in minutes (within 24h)",
        examples=[480.0, 120.0],
    )
    response_times_by_hour: dict[int, float] = Field(
        default_factory=dict,
        description="Average response time by hour of day (0-23)",
        examples=[{9: 5.2, 14: 12.5, 20: 25.0}],
    )
    response_times_by_day: dict[str, float] = Field(
        default_factory=dict,
        description="Average response time by day of week",
        examples=[{"Monday": 10.5, "Saturday": 45.0}],
    )
    my_avg_response_time_minutes: float | None = Field(
        default=None,
        description="Your average response time in minutes",
        examples=[12.0, 8.0],
    )
    their_avg_response_time_minutes: float | None = Field(
        default=None,
        description="Their average response time in minutes",
        examples=[18.5, 15.0],
    )
    typical_response_length: Literal["short", "medium", "long"] = Field(
        default="medium",
        description="Typical response length category",
        examples=["short", "medium", "long"],
    )
    greeting_style: list[str] = Field(
        default_factory=list,
        description="Common greeting phrases used",
        examples=[["Hey", "Hi there", "Hello"]],
    )
    signoff_style: list[str] = Field(
        default_factory=list,
        description="Common signoff phrases used",
        examples=[["Thanks", "Cheers", "Best"]],
    )
    common_phrases: list[str] = Field(
        default_factory=list,
        description="Commonly used phrases",
        examples=[["sounds good", "let me know", "no problem"]],
    )


class FrequencyTrendsResponse(BaseModel):
    """Message frequency trend analysis.

    Provides daily, weekly, and monthly message counts with trend direction.

    Example:
        ```json
        {
            "trend_direction": "increasing",
            "trend_percentage": 25.5,
            "most_active_day": "Saturday",
            "most_active_hour": 20,
            "messages_per_day_avg": 12.5
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "daily_counts": {"2024-01-15": 25, "2024-01-16": 18},
                "weekly_counts": {"2024-W03": 120, "2024-W04": 95},
                "monthly_counts": {"2024-01": 450},
                "trend_direction": "increasing",
                "trend_percentage": 25.5,
                "most_active_day": "Saturday",
                "most_active_hour": 20,
                "messages_per_day_avg": 12.5,
            }
        }
    )

    daily_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Message count by day (YYYY-MM-DD)",
    )
    weekly_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Message count by week (YYYY-WNN)",
    )
    monthly_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Message count by month (YYYY-MM)",
    )
    trend_direction: str = Field(
        ...,
        description="Trend direction: 'increasing', 'decreasing', or 'stable'",
        examples=["increasing", "decreasing", "stable"],
    )
    trend_percentage: float = Field(
        ...,
        description="Percentage change over the analysis period",
        examples=[25.5, -10.0, 0.0],
    )
    most_active_day: str | None = Field(
        default=None,
        description="Most active day of the week",
        examples=["Saturday", "Wednesday"],
    )
    most_active_hour: int | None = Field(
        default=None,
        ge=0,
        le=23,
        description="Most active hour of the day (0-23)",
        examples=[20, 14],
    )
    messages_per_day_avg: float = Field(
        ...,
        ge=0,
        description="Average messages per day",
        examples=[12.5, 5.0],
    )


class RelationshipHealthResponse(BaseModel):
    """Relationship health score and breakdown.

    Provides a composite health score based on engagement, sentiment,
    responsiveness, and consistency factors.

    Example:
        ```json
        {
            "overall_score": 75.5,
            "health_label": "good",
            "engagement_score": 80.0,
            "sentiment_score": 72.5,
            "responsiveness_score": 70.0,
            "consistency_score": 78.0,
            "factors": {
                "engagement": "Balanced conversation with good message exchange",
                "sentiment": "Predominantly positive communication"
            }
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_score": 75.5,
                "health_label": "good",
                "engagement_score": 80.0,
                "sentiment_score": 72.5,
                "responsiveness_score": 70.0,
                "consistency_score": 78.0,
                "factors": {
                    "engagement": "Balanced conversation with good message exchange",
                    "sentiment": "Predominantly positive communication",
                    "responsiveness": "Good response time",
                    "consistency": "Very consistent communication",
                },
            }
        }
    )

    overall_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall relationship health score (0-100)",
        examples=[75.5, 85.0],
    )
    engagement_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Engagement score based on message balance and frequency",
        examples=[80.0, 65.0],
    )
    sentiment_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Sentiment score (normalized 0-100)",
        examples=[72.5, 80.0],
    )
    responsiveness_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Responsiveness score based on response times",
        examples=[70.0, 90.0],
    )
    consistency_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Consistency score based on regular communication",
        examples=[78.0, 50.0],
    )
    health_label: str = Field(
        ...,
        description="Health label: 'excellent', 'good', 'fair', 'needs_attention', 'concerning'",
        examples=["good", "excellent", "fair"],
    )
    factors: dict[str, str] = Field(
        default_factory=dict,
        description="Contributing factors with descriptions",
    )


class ConversationInsightsResponse(BaseModel):
    """Complete conversation insights response.

    Contains all analytics including sentiment, response patterns,
    frequency trends, and relationship health.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "contact_name": "John Doe",
            "time_range": "month",
            "sentiment_overall": {"score": 0.45, "label": "positive"},
            "relationship_health": {"overall_score": 75.5, "health_label": "good"},
            "total_messages_analyzed": 500
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "contact_name": "John Doe",
                "time_range": "month",
                "sentiment_overall": {
                    "score": 0.45,
                    "label": "positive",
                    "positive_count": 120,
                    "negative_count": 30,
                    "neutral_count": 50,
                },
                "sentiment_trends": [
                    {"date": "2024-W01", "score": 0.3, "message_count": 45},
                    {"date": "2024-W02", "score": 0.5, "message_count": 52},
                ],
                "response_patterns": {
                    "avg_response_time_minutes": 15.5,
                    "my_avg_response_time_minutes": 12.0,
                    "their_avg_response_time_minutes": 18.5,
                },
                "frequency_trends": {
                    "trend_direction": "stable",
                    "messages_per_day_avg": 12.5,
                },
                "relationship_health": {
                    "overall_score": 75.5,
                    "health_label": "good",
                },
                "total_messages_analyzed": 500,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation identifier",
        examples=["chat123456789"],
    )
    contact_name: str | None = Field(
        default=None,
        description="Display name of the contact",
        examples=["John Doe", "Mom"],
    )
    time_range: TimeRangeEnum = Field(
        ...,
        description="Time range used for analysis",
    )
    sentiment_overall: SentimentResponse = Field(
        ...,
        description="Overall sentiment analysis for the conversation",
    )
    sentiment_trends: list[SentimentTrendResponse] = Field(
        default_factory=list,
        description="Sentiment trends over time (weekly)",
    )
    response_patterns: ResponsePatternsResponse = Field(
        ...,
        description="Response time pattern analysis",
    )
    frequency_trends: FrequencyTrendsResponse = Field(
        ...,
        description="Message frequency trend analysis",
    )
    relationship_health: RelationshipHealthResponse = Field(
        ...,
        description="Relationship health score and breakdown",
    )
    total_messages_analyzed: int = Field(
        ...,
        ge=0,
        description="Total number of messages analyzed",
        examples=[500, 1000],
    )
    first_message_date: datetime | None = Field(
        default=None,
        description="Date of the earliest message analyzed",
    )
    last_message_date: datetime | None = Field(
        default=None,
        description="Date of the most recent message analyzed",
    )


class ToneProfileResponse(BaseModel):
    """Communication tone characteristics for a relationship.

    Example:
        ```json
        {
            "formality_score": 0.3,
            "emoji_frequency": 1.5,
            "exclamation_frequency": 0.8,
            "question_frequency": 0.2,
            "avg_message_length": 45.5,
            "uses_caps": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "formality_score": 0.3,
                "emoji_frequency": 1.5,
                "exclamation_frequency": 0.8,
                "question_frequency": 0.2,
                "avg_message_length": 45.5,
                "uses_caps": False,
            }
        }
    )

    formality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Formality score: 0.0 (very casual) to 1.0 (very formal)",
        examples=[0.3, 0.7],
    )
    emoji_frequency: float = Field(
        ...,
        ge=0.0,
        description="Average emojis per message",
        examples=[1.5, 0.2],
    )
    exclamation_frequency: float = Field(
        ...,
        ge=0.0,
        description="Average exclamation marks per message",
        examples=[0.8, 0.1],
    )
    question_frequency: float = Field(
        ...,
        ge=0.0,
        description="Average question marks per message",
        examples=[0.2, 0.5],
    )
    avg_message_length: float = Field(
        ...,
        ge=0.0,
        description="Average characters per message",
        examples=[45.5, 120.0],
    )
    uses_caps: bool = Field(
        ...,
        description="Whether the person occasionally uses ALL CAPS",
    )


class TopicDistributionResponse(BaseModel):
    """Distribution of conversation topics.

    Example:
        ```json
        {
            "topics": {
                "scheduling": 0.35,
                "food": 0.25,
                "work": 0.2
            },
            "top_topics": ["scheduling", "food", "work"]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topics": {
                    "scheduling": 0.35,
                    "food": 0.25,
                    "work": 0.2,
                },
                "top_topics": ["scheduling", "food", "work"],
            }
        }
    )

    topics: dict[str, float] = Field(
        default_factory=dict,
        description="Topic name to frequency (0.0-1.0) mapping",
        examples=[{"scheduling": 0.35, "food": 0.25}],
    )
    top_topics: list[str] = Field(
        default_factory=list,
        description="Top 3 most discussed topics",
        examples=[["scheduling", "food", "work"]],
    )


class RelationshipProfileResponse(BaseModel):
    """Complete relationship profile for a contact.

    Contains analyzed communication patterns, topic distribution,
    and response behaviors learned from message history.

    Example:
        ```json
        {
            "contact_id": "a1b2c3d4e5f6g7h8",
            "contact_name": "John Doe",
            "tone_profile": {...},
            "topic_distribution": {...},
            "response_patterns": {...},
            "message_count": 250,
            "last_updated": "2024-01-15T10:30:00",
            "version": "1.0.0"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "contact_id": "a1b2c3d4e5f6g7h8",
                "contact_name": "John Doe",
                "tone_profile": {
                    "formality_score": 0.3,
                    "emoji_frequency": 1.5,
                    "exclamation_frequency": 0.8,
                    "question_frequency": 0.2,
                    "avg_message_length": 45.5,
                    "uses_caps": False,
                },
                "topic_distribution": {
                    "topics": {"scheduling": 0.35, "food": 0.25},
                    "top_topics": ["scheduling", "food"],
                },
                "response_patterns": {
                    "avg_response_time_minutes": 15.5,
                    "typical_response_length": "medium",
                    "greeting_style": ["hey", "hi"],
                    "signoff_style": ["thanks"],
                    "common_phrases": ["sounds good"],
                },
                "message_count": 250,
                "last_updated": "2024-01-15T10:30:00",
                "version": "1.0.0",
            }
        }
    )

    contact_id: str = Field(
        ...,
        description="Hashed contact identifier",
        examples=["a1b2c3d4e5f6g7h8"],
    )
    contact_name: str | None = Field(
        default=None,
        description="Display name of the contact",
        examples=["John Doe", "Mom"],
    )
    tone_profile: ToneProfileResponse = Field(
        ...,
        description="Communication tone characteristics",
    )
    topic_distribution: TopicDistributionResponse = Field(
        ...,
        description="Topics typically discussed with this contact",
    )
    response_patterns: ResponsePatternsResponse = Field(
        ...,
        description="Response time and style patterns",
    )
    message_count: int = Field(
        ...,
        ge=0,
        description="Total messages analyzed for this profile",
        examples=[250, 50],
    )
    last_updated: str = Field(
        ...,
        description="ISO timestamp of last profile update",
        examples=["2024-01-15T10:30:00"],
    )
    version: str = Field(
        ...,
        description="Profile format version",
        examples=["1.0.0"],
    )


class StyleGuideResponse(BaseModel):
    """Natural language style guide for a relationship.

    Provides human-readable guidance on how to communicate
    with a specific contact based on their relationship profile.

    Example:
        ```json
        {
            "contact_id": "a1b2c3d4e5f6g7h8",
            "contact_name": "John Doe",
            "style_guide": "Keep it very casual and relaxed, feel free to use emojis...",
            "voice_guidance": {
                "formality": "casual",
                "use_emojis": true,
                "emoji_level": "high",
                ...
            }
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "contact_id": "a1b2c3d4e5f6g7h8",
                "contact_name": "John Doe",
                "style_guide": "Keep it casual, use emojis, keep messages brief.",
                "voice_guidance": {
                    "formality": "casual",
                    "use_emojis": True,
                    "emoji_level": "high",
                    "message_length": "short",
                    "use_exclamations": True,
                    "common_greetings": ["hey", "hi"],
                    "common_signoffs": ["thanks", "bye"],
                    "preferred_phrases": ["sounds good"],
                    "top_topics": ["scheduling", "food"],
                },
            }
        }
    )

    contact_id: str = Field(
        ...,
        description="Hashed contact identifier",
        examples=["a1b2c3d4e5f6g7h8"],
    )
    contact_name: str | None = Field(
        default=None,
        description="Display name of the contact",
        examples=["John Doe", "Mom"],
    )
    style_guide: str = Field(
        ...,
        description="Natural language style description",
        examples=["Keep it very casual and relaxed, feel free to use emojis liberally."],
    )
    voice_guidance: dict[str, object] = Field(
        ...,
        description="Structured guidance parameters for prompt building",
    )


class RefreshProfileRequest(BaseModel):
    """Request to refresh a relationship profile.

    Example:
        ```json
        {
            "message_limit": 500,
            "force_refresh": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message_limit": 500,
                "force_refresh": True,
            }
        }
    )

    message_limit: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum messages to analyze for profile building",
        examples=[500, 1000],
    )
    force_refresh: bool = Field(
        default=False,
        description="Force refresh even if profile is recent",
    )


class RefreshProfileResponse(BaseModel):
    """Response after refreshing a relationship profile.

    Example:
        ```json
        {
            "success": true,
            "profile": {...},
            "messages_analyzed": 500,
            "previous_message_count": 250
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "messages_analyzed": 500,
                "previous_message_count": 250,
            }
        }
    )

    success: bool = Field(
        ...,
        description="Whether the refresh was successful",
    )
    profile: RelationshipProfileResponse | None = Field(
        default=None,
        description="The refreshed profile (if successful)",
    )
    messages_analyzed: int = Field(
        ...,
        ge=0,
        description="Number of messages analyzed",
        examples=[500, 250],
    )
    previous_message_count: int | None = Field(
        default=None,
        description="Previous profile's message count (if existed)",
        examples=[250, None],
    )
    error: str | None = Field(
        default=None,
        description="Error message if refresh failed",
        examples=[None, "Insufficient messages for profile"],
    )
