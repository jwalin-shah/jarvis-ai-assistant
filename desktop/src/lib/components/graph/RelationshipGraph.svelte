<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as d3 from "d3";
  import { api } from "../../api/client";
  import type { GraphData, GraphNode, LayoutType } from "../../api/types";
  import GraphControls from "./GraphControls.svelte";
  import NodeTooltip from "./NodeTooltip.svelte";
  import ClusterLegend from "./ClusterLegend.svelte";

  // Props
  export let width: number = 800;
  export let height: number = 600;
  export let onNodeClick: ((node: GraphNode) => void) | null = null;
  export let onNodeDoubleClick: ((node: GraphNode) => void) | null = null;

  // State
  let loading = true;
  let error: string | null = null;
  let graphData: GraphData | null = null;
  let svgElement: SVGSVGElement;
  // @ts-expect-error - used in bind:this
  let _containerElement: HTMLDivElement;
  let showLabels = true;
  let currentLayout: LayoutType = "force";
  let tooltipNode: GraphNode | null = null;
  let tooltipX = 0;
  let tooltipY = 0;

  // D3 references
  let simulation: d3.Simulation<d3.SimulationNodeDatum, undefined> | null = null;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = null;
  let g: d3.Selection<SVGGElement, unknown, null, undefined> | null = null;
  let zoom: d3.ZoomBehavior<SVGSVGElement, unknown> | null = null;

  // Pre-computed adjacency map for O(1) neighbor lookups
  let adjacencyMap: Map<string, Set<string>> = new Map();

  // Pre-computed node map for O(1) node lookups
  let nodeMap: Map<string, GraphNode> = new Map();

  function buildAdjacencyMap(edges: { source: string; target: string }[]) {
    const map = new Map<string, Set<string>>();
    for (const e of edges) {
      if (!map.has(e.source)) map.set(e.source, new Set());
      if (!map.has(e.target)) map.set(e.target, new Set());
      map.get(e.source)!.add(e.target);
      map.get(e.target)!.add(e.source);
    }
    return map;
  }

  function buildNodeMap(nodes: GraphNode[]) {
    const map = new Map<string, GraphNode>();
    for (const node of nodes) {
      map.set(node.id, node);
    }
    return map;
  }

  // Relationship colors for legend
  const relationshipColors: Record<string, string> = {
    family: "#FF6B6B",
    friend: "#4ECDC4",
    work: "#45B7D1",
    acquaintance: "#96CEB4",
    romantic: "#DDA0DD",
    professional: "#6495ED",
    unknown: "#8E8E93",
    self: "#007AFF",
  };

  async function loadGraph() {
    loading = true;
    error = null;

    try {
      graphData = await api.getNetworkGraph({
        layout: currentLayout,
        maxNodes: 100,
        includeClusters: true,
        width,
        height,
      });
      adjacencyMap = buildAdjacencyMap(graphData.edges);
      nodeMap = buildNodeMap(graphData.nodes);
      renderGraph();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load graph";
    } finally {
      loading = false;
    }
  }

  // Track if this is first render for incremental updates
  let isFirstRender = true;
  // Store references to D3 selections for updates
  let linkSelection: d3.Selection<any, any, any, unknown> | null = null;
  let nodeSelection: d3.Selection<any, any, any, unknown> | null = null;
  let labelSelection: d3.Selection<any, any, any, unknown> | null = null;

  function renderGraph() {
    if (!graphData || !svgElement) return;

    // Prepare data for D3
    const nodes = graphData.nodes.map(n => ({
      ...n,
      x: n.x ?? width / 2,
      y: n.y ?? height / 2,
    }));

    const links = graphData.edges.map(e => ({
      ...e,
      source: e.source,
      target: e.target,
    }));

    if (isFirstRender) {
      // First render: build everything from scratch
      renderInitial(nodes, links);
      isFirstRender = false;
    } else {
      // Subsequent updates: use D3's data join for efficient updates
      updateGraph(nodes, links);
    }
  }

  function renderInitial(nodes: any[], links: any[]) {
    // Clear existing (only on first render)
    d3.select(svgElement).selectAll("*").remove();

    svg = d3.select(svgElement)
      .attr("viewBox", [0, 0, width, height])
      .attr("width", "100%")
      .attr("height", "100%");

    g = svg.append("g");

    // Setup zoom
    zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 5])
      .on("zoom", (event) => {
        g!.attr("transform", event.transform);
      });

    svg.call(zoom);

    // Create simulation
    simulation = d3.forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force("link", d3.forceLink(links)
        .id((d: any) => d.id)
        .distance((d: any) => 100 / (d.weight + 0.1))
        .strength((d: any) => d.weight * 0.3))
      .force("charge", d3.forceManyBody()
        .strength((d: any) => -200 - d.size * 5))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide()
        .radius((d: any) => d.size + 5));

    // Draw links
    linkSelection = g!.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", "#666")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", (d: any) => Math.max(1, d.weight * 3));

    // Draw nodes
    nodeSelection = g!.append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d: any) => d.size)
      .attr("fill", (d: any) => d.color)
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .attr("cursor", "pointer")
      .call(drag(simulation) as any);

    // Draw labels
    labelSelection = g!.append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .attr("text-anchor", "middle")
      .attr("dy", (d: any) => d.size + 14)
      .attr("font-size", "11px")
      .attr("fill", "#fff")
      .attr("pointer-events", "none")
      .style("text-shadow", "1px 1px 2px rgba(0,0,0,0.8)")
      .style("display", showLabels ? "block" : "none")
      .text((d: any) => d.label);

    // Event handlers
    bindEventHandlers(nodeSelection!);

    // Update positions on tick
    simulation.on("tick", () => {
      linkSelection!
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      nodeSelection!
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y);

      labelSelection!
        .attr("x", (d: any) => d.x)
        .attr("y", (d: any) => d.y);
    });
  }

  function updateGraph(nodes: any[], links: any[]) {
    if (!simulation || !g) return;

    // Update simulation data
    simulation.nodes(nodes as d3.SimulationNodeDatum[]);
    (simulation.force("link") as d3.ForceLink<any, any>).links(links);

    // Update links using data join
    linkSelection = g.select(".links")
      .selectAll("line")
      .data(links, (d: any) => `${d.source.id || d.source}-${d.target.id || d.target}`)
      .join(
        enter => enter.append("line")
          .attr("stroke", "#666")
          .attr("stroke-opacity", 0.6)
          .attr("stroke-width", (d: any) => Math.max(1, d.weight * 3)),
        update => update
          .attr("stroke-width", (d: any) => Math.max(1, d.weight * 3)),
        exit => exit.remove()
      );

    // Update nodes using data join
    nodeSelection = g.select(".nodes")
      .selectAll("circle")
      .data(nodes, (d: any) => d.id)
      .join(
        enter => enter.append("circle")
          .attr("r", (d: any) => d.size)
          .attr("fill", (d: any) => d.color)
          .attr("stroke", "#fff")
          .attr("stroke-width", 2)
          .attr("cursor", "pointer")
          .call(drag(simulation!) as any),
        update => update
          .attr("r", (d: any) => d.size)
          .attr("fill", (d: any) => d.color),
        exit => exit.remove()
      );

    // Update labels using data join
    labelSelection = g.select(".labels")
      .selectAll("text")
      .data(nodes, (d: any) => d.id)
      .join(
        enter => enter.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", (d: any) => d.size + 14)
          .attr("font-size", "11px")
          .attr("fill", "#fff")
          .attr("pointer-events", "none")
          .style("text-shadow", "1px 1px 2px rgba(0,0,0,0.8)")
          .style("display", showLabels ? "block" : "none")
          .text((d: any) => d.label),
        update => update
          .attr("dy", (d: any) => d.size + 14)
          .style("display", showLabels ? "block" : "none")
          .text((d: any) => d.label),
        exit => exit.remove()
      );

    // Re-bind event handlers for new nodes
    bindEventHandlers(nodeSelection!);

    // Reheat simulation for layout adjustment
    simulation.alpha(0.3).restart();
  }

  function bindEventHandlers(node: d3.Selection<any, any, any, unknown>) {
    node.on("mouseover", (event: MouseEvent, d: any) => {
      tooltipNode = d as GraphNode;
      tooltipX = event.pageX;
      tooltipY = event.pageY;
    })
    .on("mousemove", (event: MouseEvent) => {
      tooltipX = event.pageX;
      tooltipY = event.pageY;
    })
    .on("mouseout", () => {
      tooltipNode = null;
    })
    .on("click", (_event: MouseEvent, d: any) => {
      highlightNode(d.id);
      if (onNodeClick) onNodeClick(d as GraphNode);
    })
    .on("dblclick", (_event: MouseEvent, d: any) => {
      if (onNodeDoubleClick) onNodeDoubleClick(d as GraphNode);
      zoomToNode(d);
    });
  }

  function drag(simulation: d3.Simulation<d3.SimulationNodeDatum, undefined>) {
    return d3.drag()
      .on("start", (event: any, d: any) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event: any, d: any) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event: any, d: any) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
  }

  function highlightNode(nodeId: string) {
    if (!g) return;

    // Reset all nodes
    g.selectAll("circle")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .attr("opacity", 1);

    // Reset all links
    g.selectAll("line")
      .attr("opacity", 0.6);

    // Highlight selected node
    g.selectAll("circle")
      .filter((d: any) => d.id === nodeId)
      .attr("stroke", "#FFD700")
      .attr("stroke-width", 4);

    // Highlight connected edges
    g.selectAll("line")
      .attr("opacity", (d: any) =>
        d.source.id === nodeId || d.target.id === nodeId ? 1 : 0.2
      );

    // Dim unconnected nodes (O(1) lookup via adjacency map)
    const neighbors = adjacencyMap.get(nodeId) ?? new Set();
    const connectedIds = new Set<string>(neighbors);
    connectedIds.add(nodeId);

    g.selectAll("circle")
      .attr("opacity", (d: any) => connectedIds.has(d.id) ? 1 : 0.3);
  }

  function zoomToNode(node: any) {
    if (!svg || !zoom) return;

    const scale = 2;
    const x = width / 2 - node.x * scale;
    const y = height / 2 - node.y * scale;

    svg.transition()
      .duration(500)
      .call(
        zoom.transform,
        d3.zoomIdentity.translate(x, y).scale(scale)
      );
  }

  function resetZoom() {
    if (!svg || !zoom) return;
    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
  }

  function toggleLabels() {
    showLabels = !showLabels;
    if (g) {
      g.selectAll(".labels text")
        .style("display", showLabels ? "block" : "none");
    }
  }

  function reheat() {
    if (simulation) {
      simulation.alpha(1).restart();
    }
  }

  async function exportGraph(format: "png" | "svg" | "json" | "html") {
    if (!svgElement || !graphData) return;

    if (format === "json") {
      const blob = new Blob([JSON.stringify(graphData, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `jarvis-graph-${new Date().toISOString().split("T")[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
      return;
    }

    if (format === "svg") {
      const serializer = new XMLSerializer();
      const svgStr = serializer.serializeToString(svgElement);
      const blob = new Blob([svgStr], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `jarvis-graph-${new Date().toISOString().split("T")[0]}.svg`;
      a.click();
      URL.revokeObjectURL(url);
      return;
    }

    if (format === "png") {
      const serializer = new XMLSerializer();
      const svgStr = serializer.serializeToString(svgElement);
      const canvas = document.createElement("canvas");
      canvas.width = width * 2;
      canvas.height = height * 2;
      const ctx = canvas.getContext("2d")!;
      ctx.scale(2, 2);
      const img = new Image();
      const svgBlob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(svgBlob);
      img.onload = () => {
        ctx.drawImage(img, 0, 0, width, height);
        URL.revokeObjectURL(url);
        canvas.toBlob((blob) => {
          if (!blob) return;
          const pngUrl = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = pngUrl;
          a.download = `jarvis-graph-${new Date().toISOString().split("T")[0]}.png`;
          a.click();
          URL.revokeObjectURL(pngUrl);
        }, "image/png");
      };
      img.src = url;
    }
  }

  async function changeLayout(layout: LayoutType) {
    currentLayout = layout;
    isFirstRender = true; // Force full re-render on layout change
    await loadGraph();
  }

  function filterByRelationship(types: string[]) {
    if (!g || !graphData) return;

    const typeSet = new Set(types);

    g.selectAll("circle")
      .attr("opacity", (d: any) =>
        types.length === 0 || typeSet.has(d.relationship_type) ? 1 : 0.2
      );

    g.selectAll("line")
      .attr("opacity", (d: any) => {
        const sourceId = d.source.id || d.source;
        const targetId = d.target.id || d.target;
        const sourceNode = nodeMap.get(sourceId);
        const targetNode = nodeMap.get(targetId);
        if (types.length === 0) return 0.6;
        const sourceMatch = sourceNode && typeSet.has(sourceNode.relationship_type);
        const targetMatch = targetNode && typeSet.has(targetNode.relationship_type);
        return sourceMatch && targetMatch ? 0.6 : 0.1;
      });
  }

  function searchNodes(query: string) {
    if (!g || !graphData) return;

    if (!query) {
      g.selectAll("circle").attr("opacity", 1);
      g.selectAll("line").attr("opacity", 0.6);
      return;
    }

    const lowerQuery = query.toLowerCase();
    const matchingIds = new Set(
      graphData.nodes
        .filter(n => n.label.toLowerCase().includes(lowerQuery))
        .map(n => n.id)
    );

    g.selectAll("circle")
      .attr("opacity", (d: any) => matchingIds.has(d.id) ? 1 : 0.2);
  }

  onMount(() => {
    loadGraph();
  });

  onDestroy(() => {
    if (simulation) {
      simulation.stop();
    }
  });
</script>

<div class="graph-container" bind:this={_containerElement}>
  <GraphControls
    {showLabels}
    {currentLayout}
    on:resetZoom={resetZoom}
    on:toggleLabels={toggleLabels}
    on:reheat={reheat}
    on:changeLayout={(e) => changeLayout(e.detail)}
    on:search={(e) => searchNodes(e.detail)}
    on:filterRelationships={(e) => filterByRelationship(e.detail)}
    on:export={(e) => exportGraph(e.detail)}
  />

  <div class="graph-wrapper">
    {#if loading}
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p>Loading relationship graph...</p>
      </div>
    {:else if error}
      <div class="error-state">
        <p class="error-message">{error}</p>
        <button class="retry-btn" on:click={loadGraph}>Try Again</button>
      </div>
    {:else}
      <svg bind:this={svgElement}></svg>

      {#if graphData}
        <ClusterLegend
          nodes={graphData.nodes}
          colors={relationshipColors}
          onfilter={filterByRelationship}
        />
      {/if}
    {/if}
  </div>

  {#if tooltipNode}
    <NodeTooltip
      node={tooltipNode}
      x={tooltipX}
      y={tooltipY}
    />
  {/if}
</div>

<style>
  .graph-container {
    position: relative;
    width: 100%;
    height: 100%;
    background: var(--bg-primary);
    border-radius: 12px;
    overflow: hidden;
  }

  .graph-wrapper {
    width: 100%;
    height: calc(100% - 50px);
    position: relative;
  }

  svg {
    display: block;
    width: 100%;
    height: 100%;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 16px;
    color: var(--text-secondary);
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 16px;
    text-align: center;
  }

  .error-message {
    color: var(--error-color);
    font-size: 14px;
  }

  .retry-btn {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
  }

  .retry-btn:hover {
    background: var(--bg-active);
  }
</style>
