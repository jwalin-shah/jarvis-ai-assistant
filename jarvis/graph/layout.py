"""Layout algorithms for graph visualization.

Provides force-directed and hierarchical layout algorithms for
positioning nodes in relationship graphs.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Literal

from jarvis.graph.builder import GraphData

logger = logging.getLogger(__name__)

# Layout constants
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
DEFAULT_ITERATIONS = 100
REPULSION_STRENGTH = 5000
ATTRACTION_STRENGTH = 0.01
DAMPING = 0.9
MIN_DISTANCE = 30


@dataclass
class LayoutConfig:
    """Configuration for layout algorithms.

    Attributes:
        width: Canvas width in pixels
        height: Canvas height in pixels
        iterations: Number of simulation iterations
        repulsion: Repulsion force strength
        attraction: Attraction force strength
        damping: Velocity damping factor
        center_gravity: Pull toward center strength
        random_seed: Seed for reproducible layouts
    """

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    iterations: int = DEFAULT_ITERATIONS
    repulsion: float = REPULSION_STRENGTH
    attraction: float = ATTRACTION_STRENGTH
    damping: float = DAMPING
    center_gravity: float = 0.05
    random_seed: int | None = None


class LayoutEngine:
    """Engine for computing graph layouts.

    Supports force-directed and hierarchical layout algorithms
    optimized for relationship graphs.
    """

    def __init__(self, config: LayoutConfig | None = None) -> None:
        """Initialize the layout engine.

        Args:
            config: Layout configuration (uses defaults if None)
        """
        self.config = config or LayoutConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def force_directed(self, graph: GraphData) -> GraphData:
        """Apply force-directed layout to the graph.

        Uses a spring-embedder algorithm with:
        - Repulsion between all node pairs
        - Attraction along edges (spring forces)
        - Center gravity to prevent drift
        - Edge weight influences spring strength

        Args:
            graph: Input graph data

        Returns:
            GraphData with x, y positions set on nodes
        """
        if not graph.nodes:
            return graph

        # Initialize positions randomly
        center_x = self.config.width / 2
        center_y = self.config.height / 2
        radius = min(self.config.width, self.config.height) * 0.4

        positions: dict[str, tuple[float, float]] = {}
        velocities: dict[str, tuple[float, float]] = {}

        for i, node in enumerate(graph.nodes):
            # Start in a circle if node has no position
            if node.x is None or node.y is None:
                angle = 2 * math.pi * i / len(graph.nodes)
                x = center_x + radius * math.cos(angle) * random.uniform(0.8, 1.2)
                y = center_y + radius * math.sin(angle) * random.uniform(0.8, 1.2)
            else:
                x, y = node.x, node.y

            positions[node.id] = (x, y)
            velocities[node.id] = (0.0, 0.0)

        # Build adjacency for faster edge lookup
        adjacency: dict[str, list[tuple[str, float]]] = {n.id: [] for n in graph.nodes}
        for edge in graph.edges:
            if edge.source in adjacency:
                adjacency[edge.source].append((edge.target, edge.weight))
            if edge.target in adjacency:
                adjacency[edge.target].append((edge.source, edge.weight))

        # Run simulation
        for iteration in range(self.config.iterations):
            # Calculate temperature (cooling schedule)
            temp = 1.0 - iteration / self.config.iterations
            temp = max(0.01, temp)

            forces: dict[str, tuple[float, float]] = {n.id: (0.0, 0.0) for n in graph.nodes}

            # Repulsion forces (all pairs)
            for i, node1 in enumerate(graph.nodes):
                for node2 in graph.nodes[i + 1 :]:
                    x1, y1 = positions[node1.id]
                    x2, y2 = positions[node2.id]

                    dx = x2 - x1
                    dy = y2 - y1
                    dist_sq = dx * dx + dy * dy
                    dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.1

                    # Prevent division by zero and very close nodes
                    if dist < MIN_DISTANCE:
                        dist = MIN_DISTANCE
                        dist_sq = dist * dist

                    # Repulsion force (inverse square)
                    force = self.config.repulsion / dist_sq

                    fx = force * dx / dist
                    fy = force * dy / dist

                    f1x, f1y = forces[node1.id]
                    f2x, f2y = forces[node2.id]

                    forces[node1.id] = (f1x - fx, f1y - fy)
                    forces[node2.id] = (f2x + fx, f2y + fy)

            # Attraction forces (edges)
            for edge in graph.edges:
                if edge.source not in positions or edge.target not in positions:
                    continue

                x1, y1 = positions[edge.source]
                x2, y2 = positions[edge.target]

                dx = x2 - x1
                dy = y2 - y1
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < 0.01:
                    continue

                # Spring force (Hooke's law) weighted by edge weight
                force = self.config.attraction * dist * (0.5 + edge.weight * 0.5)

                fx = force * dx / dist
                fy = force * dy / dist

                f1x, f1y = forces[edge.source]
                f2x, f2y = forces[edge.target]

                forces[edge.source] = (f1x + fx, f1y + fy)
                forces[edge.target] = (f2x - fx, f2y - fy)

            # Center gravity
            for node in graph.nodes:
                x, y = positions[node.id]
                dx = center_x - x
                dy = center_y - y

                fx, fy = forces[node.id]
                forces[node.id] = (
                    fx + dx * self.config.center_gravity,
                    fy + dy * self.config.center_gravity,
                )

            # Apply forces with damping
            for node in graph.nodes:
                vx, vy = velocities[node.id]
                fx, fy = forces[node.id]

                # Update velocity with damping
                vx = (vx + fx * temp) * self.config.damping
                vy = (vy + fy * temp) * self.config.damping

                # Limit velocity
                speed = math.sqrt(vx * vx + vy * vy)
                max_speed = 50 * temp
                if speed > max_speed:
                    vx = vx * max_speed / speed
                    vy = vy * max_speed / speed

                velocities[node.id] = (vx, vy)

                # Update position
                x, y = positions[node.id]
                x += vx
                y += vy

                # Keep within bounds
                margin = 50
                x = max(margin, min(self.config.width - margin, x))
                y = max(margin, min(self.config.height - margin, y))

                positions[node.id] = (x, y)

        # Apply positions to nodes
        for node in graph.nodes:
            x, y = positions[node.id]
            node.x = round(x, 2)
            node.y = round(y, 2)

        graph.metadata["layout"] = "force_directed"
        graph.metadata["dimensions"] = {
            "width": self.config.width,
            "height": self.config.height,
        }

        return graph

    def hierarchical(
        self,
        graph: GraphData,
        root_id: str | None = None,
        direction: Literal["TB", "BT", "LR", "RL"] = "TB",
    ) -> GraphData:
        """Apply hierarchical layout to the graph.

        Arranges nodes in levels based on graph structure,
        with the root at the top (or specified direction).

        Args:
            graph: Input graph data
            root_id: ID of root node (uses highest degree if None)
            direction: Layout direction (TB=top-bottom, etc.)

        Returns:
            GraphData with x, y positions set on nodes
        """
        if not graph.nodes:
            return graph

        # Build node lookup for O(1) access
        node_lookup = {n.id: n for n in graph.nodes}

        # Find root node (highest degree if not specified)
        if root_id is None:
            degree: dict[str, int] = {n.id: 0 for n in graph.nodes}
            for edge in graph.edges:
                if edge.source in degree:
                    degree[edge.source] += 1
                if edge.target in degree:
                    degree[edge.target] += 1

            root_id = max(degree.keys(), key=lambda k: degree[k])

        # Build adjacency list
        adjacency: dict[str, set[str]] = {n.id: set() for n in graph.nodes}
        for edge in graph.edges:
            if edge.source in adjacency and edge.target in adjacency:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)

        # BFS to assign levels
        levels: dict[str, int] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(root_id, 0)])

        while queue:
            node_id, level = queue.popleft()
            if node_id in visited:
                continue

            visited.add(node_id)
            levels[node_id] = level

            for neighbor in adjacency.get(node_id, []):
                if neighbor not in visited:
                    queue.append((neighbor, level + 1))

        # Handle disconnected nodes
        for node in graph.nodes:
            if node.id not in levels:
                levels[node.id] = max(levels.values(), default=0) + 1

        # Group nodes by level
        level_groups: dict[int, list[str]] = {}
        for node_id, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node_id)

        # Calculate positions
        max_level = max(level_groups.keys(), default=0)
        level_count = max_level + 1

        for level, node_ids in level_groups.items():
            count = len(node_ids)

            for i, node_id in enumerate(node_ids):
                # Distribute horizontally
                if count == 1:
                    x_pos = 0.5
                else:
                    x_pos = (i + 0.5) / count

                # Distribute vertically by level
                if level_count == 1:
                    y_pos = 0.5
                else:
                    y_pos = (level + 0.5) / level_count

                # Apply direction
                if direction == "TB":
                    x = x_pos * self.config.width
                    y = y_pos * self.config.height
                elif direction == "BT":
                    x = x_pos * self.config.width
                    y = (1 - y_pos) * self.config.height
                elif direction == "LR":
                    x = y_pos * self.config.width
                    y = x_pos * self.config.height
                else:  # RL
                    x = (1 - y_pos) * self.config.width
                    y = x_pos * self.config.height

                # Update node position
                if node_id in node_lookup:
                    node_lookup[node_id].x = round(x, 2)
                    node_lookup[node_id].y = round(y, 2)

        graph.metadata["layout"] = "hierarchical"
        graph.metadata["direction"] = direction
        graph.metadata["dimensions"] = {
            "width": self.config.width,
            "height": self.config.height,
        }

        return graph

    def radial(self, graph: GraphData, center_id: str | None = None) -> GraphData:
        """Apply radial layout centered on a specific node.

        Arranges nodes in concentric circles based on distance
        from the center node.

        Args:
            graph: Input graph data
            center_id: ID of center node (uses "me" or highest degree if None)

        Returns:
            GraphData with x, y positions set on nodes
        """
        if not graph.nodes:
            return graph

        # Build node lookup for O(1) access
        radial_node_lookup = {n.id: n for n in graph.nodes}

        # Find center node
        if center_id is None:
            # Look for "me" node first
            for node in graph.nodes:
                if node.id == "me":
                    center_id = "me"
                    break
            else:
                # Use highest message count
                center_id = max(graph.nodes, key=lambda n: n.message_count).id

        center_x = self.config.width / 2
        center_y = self.config.height / 2
        max_radius = min(self.config.width, self.config.height) * 0.45

        # Build adjacency list
        adjacency: dict[str, set[str]] = {n.id: set() for n in graph.nodes}
        for edge in graph.edges:
            if edge.source in adjacency and edge.target in adjacency:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)

        # BFS to assign distance rings
        distances: dict[str, int] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(center_id, 0)])

        while queue:
            node_id, dist = queue.popleft()
            if node_id in visited:
                continue

            visited.add(node_id)
            distances[node_id] = dist

            for neighbor in adjacency.get(node_id, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

        # Handle disconnected nodes
        max_dist = max(distances.values(), default=0)
        for node in graph.nodes:
            if node.id not in distances:
                distances[node.id] = max_dist + 1
                max_dist = distances[node.id]

        # Group by distance
        rings: dict[int, list[str]] = {}
        for node_id, dist in distances.items():
            if dist not in rings:
                rings[dist] = []
            rings[dist].append(node_id)

        # Calculate positions
        for dist, node_ids in rings.items():
            count = len(node_ids)
            radius = max_radius * dist / max(max_dist, 1) if dist > 0 else 0

            for i, node_id in enumerate(node_ids):
                if dist == 0:
                    x, y = center_x, center_y
                else:
                    angle = 2 * math.pi * i / count - math.pi / 2  # Start at top
                    x = center_x + radius * math.cos(angle)
                    y = center_y + radius * math.sin(angle)

                # Update node position
                if node_id in radial_node_lookup:
                    radial_node_lookup[node_id].x = round(x, 2)
                    radial_node_lookup[node_id].y = round(y, 2)

        graph.metadata["layout"] = "radial"
        graph.metadata["center"] = center_id
        graph.metadata["dimensions"] = {
            "width": self.config.width,
            "height": self.config.height,
        }

        return graph


def compute_force_layout(
    graph: GraphData,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    iterations: int = DEFAULT_ITERATIONS,
) -> GraphData:
    """Convenience function for force-directed layout.

    Args:
        graph: Input graph
        width: Canvas width
        height: Canvas height
        iterations: Simulation iterations

    Returns:
        GraphData with layout applied
    """
    config = LayoutConfig(width=width, height=height, iterations=iterations)
    engine = LayoutEngine(config)
    return engine.force_directed(graph)


def compute_hierarchical_layout(
    graph: GraphData,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    root_id: str | None = None,
    direction: Literal["TB", "BT", "LR", "RL"] = "TB",
) -> GraphData:
    """Convenience function for hierarchical layout.

    Args:
        graph: Input graph
        width: Canvas width
        height: Canvas height
        root_id: Root node ID
        direction: Layout direction

    Returns:
        GraphData with layout applied
    """
    config = LayoutConfig(width=width, height=height)
    engine = LayoutEngine(config)
    return engine.hierarchical(graph, root_id=root_id, direction=direction)
