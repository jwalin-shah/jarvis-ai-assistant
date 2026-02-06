"""Topic segmentation for conversation chunking."""

from jarvis.topics.topic_segmenter import (
    SegmentBoundary,
    SegmentBoundaryReason,
    SegmentMessage,
    TopicSegment,
    TopicSegmenter,
    get_segmenter,
    reset_segmenter,
    segment_conversation,
    segment_for_extraction,
)

__all__ = [
    "SegmentBoundary",
    "SegmentBoundaryReason",
    "SegmentMessage",
    "TopicSegment",
    "TopicSegmenter",
    "get_segmenter",
    "reset_segmenter",
    "segment_conversation",
    "segment_for_extraction",
]
