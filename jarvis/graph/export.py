"""Export functionality for graph data.

Provides export to various formats including JSON, GraphML,
interactive HTML, and SVG.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET  # nosec B405

from jarvis.graph.builder import GraphData

logger = logging.getLogger(__name__)


def export_to_json(
    graph: GraphData,
    path: Path | str | None = None,
    indent: int = 2,
) -> str:
    """Export graph to JSON format.

    Args:
        graph: Graph data to export
        path: Optional file path to write to
        indent: JSON indentation level

    Returns:
        JSON string representation
    """
    data = graph.to_dict()
    data["export_info"] = {
        "format": "json",
        "exported_at": datetime.now().isoformat(),
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
    }

    json_str = json.dumps(data, indent=indent)

    if path:
        path = Path(path)
        path.write_text(json_str, encoding="utf-8")
        logger.info("Exported graph to %s", path)

    return json_str


def export_to_graphml(
    graph: GraphData,
    path: Path | str | None = None,
) -> str:
    """Export graph to GraphML format.

    GraphML is an XML-based format supported by many graph tools
    including Gephi, yEd, and NetworkX.

    Args:
        graph: Graph data to export
        path: Optional file path to write to

    Returns:
        GraphML XML string
    """
    # Create root element
    root = ET.Element("graphml")
    root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set(
        "xsi:schemaLocation",
        "http://graphml.graphdrawing.org/xmlns "
        "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
    )

    # Define node attributes
    node_attrs = [
        ("label", "string"),
        ("size", "double"),
        ("color", "string"),
        ("relationship_type", "string"),
        ("message_count", "int"),
        ("sentiment_score", "double"),
        ("cluster_id", "int"),
        ("x", "double"),
        ("y", "double"),
    ]

    for attr_name, attr_type in node_attrs:
        key = ET.SubElement(root, "key")
        key.set("id", attr_name)
        key.set("for", "node")
        key.set("attr.name", attr_name)
        key.set("attr.type", attr_type)

    # Define edge attributes
    edge_attrs = [
        ("weight", "double"),
        ("message_count", "int"),
        ("sentiment", "double"),
    ]

    for attr_name, attr_type in edge_attrs:
        key = ET.SubElement(root, "key")
        key.set("id", attr_name)
        key.set("for", "edge")
        key.set("attr.name", attr_name)
        key.set("attr.type", attr_type)

    # Create graph element
    graph_elem = ET.SubElement(root, "graph")
    graph_elem.set("id", "G")
    graph_elem.set("edgedefault", "undirected")

    # Add nodes
    for node in graph.nodes:
        node_elem = ET.SubElement(graph_elem, "node")
        node_elem.set("id", node.id)

        for attr_name, _ in node_attrs:
            value = getattr(node, attr_name, None)
            if value is not None:
                data_elem = ET.SubElement(node_elem, "data")
                data_elem.set("key", attr_name)
                data_elem.text = str(value)

    # Add edges
    for i, edge in enumerate(graph.edges):
        edge_elem = ET.SubElement(graph_elem, "edge")
        edge_elem.set("id", f"e{i}")
        edge_elem.set("source", edge.source)
        edge_elem.set("target", edge.target)

        for attr_name, _ in edge_attrs:
            value = getattr(edge, attr_name, None)
            if value is not None:
                data_elem = ET.SubElement(edge_elem, "data")
                data_elem.set("key", attr_name)
                data_elem.text = str(value)

    # Generate XML string
    xml_str = ET.tostring(root, encoding="unicode")

    # Add XML declaration
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

    if path:
        path = Path(path)
        path.write_text(xml_str, encoding="utf-8")
        logger.info("Exported graph to %s", path)

    return xml_str


def export_to_svg(
    graph: GraphData,
    path: Path | str | None = None,
    width: int = 800,
    height: int = 600,
    include_labels: bool = True,
) -> str:
    """Export graph to SVG format.

    Creates a static SVG visualization of the graph.

    Args:
        graph: Graph data to export
        path: Optional file path to write to
        width: SVG width in pixels
        height: SVG height in pixels
        include_labels: Whether to include node labels

    Returns:
        SVG XML string
    """
    # SVG header
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "  .node { stroke: #fff; stroke-width: 2; cursor: pointer; }",
        "  .edge { stroke: #999; stroke-opacity: 0.6; }",
        "  .label { font-family: -apple-system, BlinkMacSystemFont, sans-serif; "
        "font-size: 10px; fill: #333; pointer-events: none; }",
        "</style>",
        '<g class="edges">',
    ]

    # Build node lookup for O(1) access
    node_lookup = {n.id: n for n in graph.nodes}

    # Draw edges
    for edge in graph.edges:
        source_node = node_lookup.get(edge.source)
        target_node = node_lookup.get(edge.target)

        if (
            source_node
            and target_node
            and source_node.x is not None
            and source_node.y is not None
            and target_node.x is not None
            and target_node.y is not None
        ):
            stroke_width = max(1, edge.weight * 4)
            opacity = 0.3 + edge.weight * 0.5
            svg_parts.append(
                f'  <line class="edge" x1="{source_node.x}" y1="{source_node.y}" '
                f'x2="{target_node.x}" y2="{target_node.y}" '
                f'stroke-width="{stroke_width}" stroke-opacity="{opacity}"/>'
            )

    svg_parts.append("</g>")
    svg_parts.append('<g class="nodes">')

    # Draw nodes
    for node in graph.nodes:
        if node.x is None or node.y is None:
            continue

        svg_parts.append(
            f'  <circle class="node" cx="{node.x}" cy="{node.y}" r="{node.size}" '
            f'fill="{node.color}">'
        )
        svg_parts.append(f"    <title>{html.escape(node.label)}</title>")
        svg_parts.append("  </circle>")

    svg_parts.append("</g>")

    # Draw labels
    if include_labels:
        svg_parts.append('<g class="labels">')
        for node in graph.nodes:
            if node.x is None or node.y is None:
                continue

            label_y = node.y + node.size + 12
            svg_parts.append(
                f'  <text class="label" x="{node.x}" y="{label_y}" '
                f'text-anchor="middle">{html.escape(node.label)}</text>'
            )
        svg_parts.append("</g>")

    svg_parts.append("</svg>")

    svg_str = "\n".join(svg_parts)

    if path:
        path = Path(path)
        path.write_text(svg_str, encoding="utf-8")
        logger.info("Exported graph to %s", path)

    return svg_str


def export_to_html(
    graph: GraphData,
    path: Path | str | None = None,
    title: str = "Relationship Graph",
    width: int = 960,
    height: int = 700,
) -> str:
    """Export graph to interactive HTML with D3.js.

    Creates a standalone HTML file with an interactive force-directed
    graph visualization using D3.js.

    Args:
        graph: Graph data to export
        path: Optional file path to write to
        title: Page title
        width: Canvas width in pixels
        height: Canvas height in pixels

    Returns:
        HTML string
    """
    # Prepare graph data as JSON
    graph_json = json.dumps(graph.to_dict())

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #fff;
            overflow: hidden;
        }}
        #container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        header {{
            padding: 12px 20px;
            background: #2a2a2a;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{ font-size: 18px; font-weight: 600; }}
        .controls {{ display: flex; gap: 10px; }}
        .controls button {{
            padding: 6px 12px;
            border: 1px solid #444;
            background: #333;
            color: #fff;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
        }}
        .controls button:hover {{ background: #444; }}
        #graph {{ flex: 1; }}
        svg {{ width: 100%; height: 100%; }}
        .node {{ cursor: pointer; }}
        .node:hover {{ stroke-width: 4; }}
        .link {{ stroke: #666; stroke-opacity: 0.6; }}
        .label {{
            font-size: 11px;
            fill: #fff;
            pointer-events: none;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }}
        .tooltip {{
            position: absolute;
            padding: 10px 14px;
            background: rgba(40, 40, 40, 0.95);
            border: 1px solid #444;
            border-radius: 8px;
            font-size: 13px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
            max-width: 280px;
        }}
        .tooltip.visible {{ opacity: 1; }}
        .tooltip h3 {{ font-size: 14px; margin-bottom: 6px; }}
        .tooltip p {{ color: #aaa; margin: 4px 0; }}
        .tooltip .stat {{ color: #4ecdc4; }}
        #legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(40, 40, 40, 0.9);
            border: 1px solid #444;
            border-radius: 8px;
            padding: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 4px 0;
            font-size: 12px;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div id="container">
        <header>
            <h1>{html.escape(title)}</h1>
            <div class="controls">
                <button onclick="resetZoom()">Reset Zoom</button>
                <button onclick="toggleLabels()">Toggle Labels</button>
                <button onclick="reheat()">Reheat</button>
            </div>
        </header>
        <div id="graph"></div>
    </div>
    <div id="legend"></div>
    <div class="tooltip" id="tooltip"></div>

    <script>
        const graphData = {graph_json};
        const width = {width};
        const height = {height};
        let showLabels = true;

        // Create SVG
        const svg = d3.select("#graph")
            .append("svg")
            .attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.2, 5])
            .on("zoom", (event) => g.attr("transform", event.transform));

        svg.call(zoom);

        // Create simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.edges)
                .id(d => d.id)
                .distance(d => 100 / (d.weight + 0.1))
                .strength(d => d.weight * 0.5))
            .force("charge", d3.forceManyBody()
                .strength(d => -300 - d.size * 10))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.size + 5));

        // Create links
        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(graphData.edges)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.max(1, d.weight * 4));

        // Create nodes
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(graphData.nodes)
            .join("circle")
            .attr("class", "node")
            .attr("r", d => d.size)
            .attr("fill", d => d.color)
            .attr("stroke", "#fff")
            .attr("stroke-width", 2)
            .call(drag(simulation));

        // Create labels
        const label = g.append("g")
            .attr("class", "labels")
            .selectAll("text")
            .data(graphData.nodes)
            .join("text")
            .attr("class", "label")
            .attr("text-anchor", "middle")
            .attr("dy", d => d.size + 14)
            .text(d => d.label);

        // Tooltip
        const tooltip = d3.select("#tooltip");

        node.on("mouseover", (event, d) => {{
            tooltip.classed("visible", true)
                .html(`
                    <h3>${{d.label}}</h3>
                    <p>Type: <span class="stat">${{d.relationship_type}}</span></p>
                    <p>Messages: <span class="stat">${{d.message_count.toLocaleString()}}</span></p>
                    <p>Sentiment: <span class="stat">${{
                        (d.sentiment_score * 100).toFixed(0)}}%</span></p>
                    ${{d.last_contact ? `<p>Last contact: <span class="stat">${{
                        new Date(d.last_contact).toLocaleDateString()
                    }}</span></p>` : ''}}
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mousemove", (event) => {{
            tooltip.style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", () => {{
            tooltip.classed("visible", false);
        }});

        // Update positions
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});

        // Drag behavior
        function drag(simulation) {{
            return d3.drag()
                .on("start", (event, d) => {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }})
                .on("drag", (event, d) => {{
                    d.fx = event.x;
                    d.fy = event.y;
                }})
                .on("end", (event, d) => {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }});
        }}

        // Control functions
        function resetZoom() {{
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        }}

        function toggleLabels() {{
            showLabels = !showLabels;
            label.style("display", showLabels ? "block" : "none");
        }}

        function reheat() {{
            simulation.alpha(1).restart();
        }}

        // Build legend
        const types = [...new Set(graphData.nodes.map(n => n.relationship_type))];
        const typeColors = {{}};
        graphData.nodes.forEach(n => {{ typeColors[n.relationship_type] = n.color; }});

        d3.select("#legend")
            .selectAll(".legend-item")
            .data(types)
            .join("div")
            .attr("class", "legend-item")
            .html(t => `<div class="legend-dot" style="background: ${{
                typeColors[t]}}"></div>${{t}}`);
    </script>
</body>
</html>"""

    if path:
        path = Path(path)
        path.write_text(html_template, encoding="utf-8")
        logger.info("Exported graph to %s", path)

    return html_template
