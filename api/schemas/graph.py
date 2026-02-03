"""Graph API schema definitions.

Pydantic models for graph visualization endpoints.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GraphNodeSchema(BaseModel):
    """Schema for a graph node representing a contact."""

    id: str = Field(description="Unique node identifier")
    label: str = Field(description="Display name")
    size: float = Field(default=16.0, description="Node radius based on message frequency")
    color: str = Field(default="#8E8E93", description="Node color")
    relationship_type: str = Field(default="unknown", description="Relationship category")
    message_count: int = Field(default=0, description="Total messages exchanged")
    last_contact: str | None = Field(default=None, description="ISO timestamp of last message")
    sentiment_score: float = Field(default=0.0, description="Average sentiment (-1 to 1)")
    response_time_avg: float | None = Field(default=None, description="Average response time in minutes")
    x: float | None = Field(default=None, description="X position from layout")
    y: float | None = Field(default=None, description="Y position from layout")
    cluster_id: int | None = Field(default=None, description="Community cluster assignment")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GraphEdgeSchema(BaseModel):
    """Schema for a graph edge representing interaction."""

    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    weight: float = Field(default=0.5, description="Edge weight (0-1)")
    message_count: int = Field(default=0, description="Messages between nodes")
    sentiment: float = Field(default=0.0, description="Average sentiment")
    last_interaction: str | None = Field(default=None, description="ISO timestamp of last interaction")
    bidirectional: bool = Field(default=True, description="Two-way communication")


class GraphDataSchema(BaseModel):
    """Schema for complete graph data."""

    nodes: list[GraphNodeSchema] = Field(default_factory=list, description="Graph nodes")
    edges: list[GraphEdgeSchema] = Field(default_factory=list, description="Graph edges")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Graph metadata")


class ClusterResultSchema(BaseModel):
    """Schema for community detection results."""

    clusters: dict[str, int] = Field(default_factory=dict, description="Node to cluster mapping")
    modularity: float = Field(default=0.0, description="Modularity score")
    num_clusters: int = Field(default=0, description="Number of clusters")
    cluster_sizes: dict[int, int] = Field(default_factory=dict, description="Size of each cluster")
    cluster_labels: dict[int, str] = Field(default_factory=dict, description="Cluster labels")


class NetworkGraphRequest(BaseModel):
    """Request parameters for network graph."""

    include_relationships: list[str] | None = Field(
        default=None,
        description="Filter by relationship types (family, friend, work, etc.)"
    )
    min_messages: int = Field(default=1, ge=0, description="Minimum message count to include")
    days_back: int | None = Field(default=None, ge=1, description="Only include recent messages")
    max_nodes: int = Field(default=100, ge=1, le=500, description="Maximum nodes to return")
    layout: Literal["force", "hierarchical", "radial"] = Field(
        default="force",
        description="Layout algorithm to apply"
    )
    include_clusters: bool = Field(default=True, description="Run community detection")
    width: int = Field(default=800, ge=100, le=4000, description="Layout width in pixels")
    height: int = Field(default=600, ge=100, le=4000, description="Layout height in pixels")


class EgoGraphRequest(BaseModel):
    """Request parameters for ego-centric graph."""

    depth: int = Field(default=1, ge=1, le=3, description="Hops from center")
    max_neighbors: int = Field(default=20, ge=1, le=100, description="Maximum neighbors")
    layout: Literal["force", "radial"] = Field(default="radial", description="Layout algorithm")
    width: int = Field(default=800, ge=100, le=4000, description="Layout width")
    height: int = Field(default=600, ge=100, le=4000, description="Layout height")


class GraphEvolutionRequest(BaseModel):
    """Request parameters for temporal graph evolution."""

    from_date: str = Field(description="Start date (ISO format)")
    to_date: str = Field(description="End date (ISO format)")
    interval: Literal["day", "week", "month"] = Field(
        default="week",
        description="Time interval for snapshots"
    )
    max_nodes: int = Field(default=50, ge=1, le=200, description="Maximum nodes per snapshot")


class GraphEvolutionSnapshot(BaseModel):
    """A single snapshot in graph evolution."""

    timestamp: str = Field(description="Snapshot timestamp")
    graph: GraphDataSchema = Field(description="Graph state at this time")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Snapshot metrics")


class GraphEvolutionResponse(BaseModel):
    """Response for graph evolution endpoint."""

    from_date: str
    to_date: str
    interval: str
    snapshots: list[GraphEvolutionSnapshot] = Field(default_factory=list)
    total_snapshots: int = Field(default=0)


class ExportGraphRequest(BaseModel):
    """Request parameters for graph export."""

    format: Literal["json", "graphml", "svg", "html"] = Field(
        default="json",
        description="Export format"
    )
    include_layout: bool = Field(default=True, description="Apply layout before export")
    width: int = Field(default=800, ge=100, le=4000, description="Export width")
    height: int = Field(default=600, ge=100, le=4000, description="Export height")


class ExportGraphResponse(BaseModel):
    """Response for graph export endpoint."""

    format: str
    filename: str
    data: str = Field(description="Exported data (base64 for binary)")
    size_bytes: int
