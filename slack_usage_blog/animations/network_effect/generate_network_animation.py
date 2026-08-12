"""
Network Effect Multiplier: Network Animation HTML Generator

Generates standalone HTML file with D3.js animation showing
group explosion from one person's perspective.

Author: Adam McCabe
"""

import json
from pathlib import Path


def generate_animation_html(data_path: Path, output_path: Path) -> None:
    """
    Generate HTML file with embedded network animation.

    Args:
        data_path: Path to network_groups_data.json
        output_path: Path for output HTML file
    """
    # Load data
    with open(data_path) as f:
        data = json.load(f)

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Network Group Explosion</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #FFFFFF;
            color: #333333;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            font-size: 24px;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            font-size: 14px;
            color: #666;
            margin-bottom: 20px;
        }}
        #animation-container {{
            position: relative;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .stats-overlay {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(248, 249, 250, 0.95);
            padding: 15px 20px;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            font-size: 14px;
            line-height: 1.8;
            min-width: 180px;
        }}
        .phase-title {{
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 15px;
            color: #4C78A8;
        }}
        .counter {{
            font-size: 18px;
            font-weight: bold;
            color: #FF6B6B;
        }}
        .summary {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #E0E0E0;
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }}
        svg {{
            display: block;
        }}
        .node {{
            stroke: #FFFFFF;
            stroke-width: 2px;
        }}
        .node-you {{
            fill: #4C78A8;
        }}
        .node-other {{
            fill: #6B7F8F;
        }}
        .node-highlighted {{
            fill: #87CEEB;
        }}
        .edge {{
            stroke: #CCCCCC;
            stroke-width: 1px;
            stroke-opacity: 0.4;
        }}
        .edge-highlighted {{
            stroke: #FF6B6B;
            stroke-width: 2.5px;
            stroke-opacity: 1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div id="animation-container">
            <svg id="network-svg" width="800" height="600"></svg>
            <div class="stats-overlay" id="stats">
                <div class="phase-title" id="phase-title">Starting...</div>
                <div class="counter" id="counter">0</div>
                <div class="summary" id="summary" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        // Embedded network groups data
        const networkData = {json.dumps(data, indent=2)};

        // Animation configuration
        const width = 800;
        const height = 600;
        const nodeRadius = 14;

        // Timing: accelerate within each phase (start -> end)
        const timings = {{
            2: {{start: 100, end: 20}},   // Pairs: 100ms -> 20ms
            3: {{start: 100, end: 20}},   // Triplets: 100ms -> 20ms
            4: {{start: 100, end: 20}}    // Groups of 4: 100ms -> 20ms
        }};

        const pauseBetweenPhases = 1000; // 1 second pause
        const summaryPause = 3000; // 3 seconds on summary

        // Calculate timing for current group within phase (linear acceleration)
        function calculateTiming(phaseSize, currentIndex, totalInPhase) {{
            const config = timings[phaseSize];
            const progress = currentIndex / (totalInPhase - 1); // 0 to 1
            return config.start - (config.start - config.end) * progress;
        }}

        // Scale for positions
        const xScale = d3.scaleLinear().domain([0, 1]).range([width * 0.15, width * 0.85]);
        const yScale = d3.scaleLinear().domain([0, 1]).range([height * 0.15, height * 0.85]);

        // Setup SVG
        const svg = d3.select("#network-svg");
        const edgesGroup = svg.append("g").attr("class", "edges");
        const nodesGroup = svg.append("g").attr("class", "nodes");

        // Draw all edges (static, complete graph)
        edgesGroup.selectAll("line")
            .data(networkData.all_edges)
            .enter()
            .append("line")
            .attr("class", "edge")
            .attr("x1", d => xScale(networkData.node_positions[d[0]].x))
            .attr("y1", d => yScale(networkData.node_positions[d[0]].y))
            .attr("x2", d => xScale(networkData.node_positions[d[1]].x))
            .attr("y2", d => yScale(networkData.node_positions[d[1]].y));

        // Draw all nodes
        nodesGroup.selectAll("circle")
            .data(networkData.node_positions)
            .enter()
            .append("circle")
            .attr("class", (d, i) => i === networkData.you_node ? "node node-you" : "node node-other")
            .attr("r", (d, i) => i === networkData.you_node ? nodeRadius * 1.2 : nodeRadius)
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y));

        // Animation state
        let currentPhase = 0;
        let currentGroupIndex = 0;
        let animationTimer = null;

        // Update stats display
        function updateStats(phase, count, total) {{
            const phaseNames = {{
                0: "Groups of 2",
                1: "Groups of 3",
                2: "Groups of 4"
            }};
            document.getElementById("phase-title").textContent = phaseNames[phase];
            document.getElementById("counter").textContent = `${{count}} / ${{total}}`;
            document.getElementById("summary").style.display = "none";
        }}

        function showSummary() {{
            document.getElementById("phase-title").textContent = "You're in...";
            document.getElementById("counter").textContent = "575 groups!";
            const summary = document.getElementById("summary");
            summary.innerHTML = "15 pairs<br>105 triplets<br>455 groups of 4";
            summary.style.display = "block";
        }}

        // Highlight a group
        function highlightGroup(group) {{
            // Highlight nodes
            nodesGroup.selectAll("circle")
                .classed("node-highlighted", d => {{
                    const idx = networkData.node_positions.indexOf(d);
                    return group.nodes.includes(idx) && idx !== networkData.you_node;
                }});

            // Highlight edges
            edgesGroup.selectAll("line")
                .classed("edge-highlighted", function(edgeData) {{
                    return group.edges.some(ge =>
                        (ge[0] === edgeData[0] && ge[1] === edgeData[1]) ||
                        (ge[0] === edgeData[1] && ge[1] === edgeData[0])
                    );
                }});
        }}

        // Unhighlight all
        function unhighlightAll() {{
            nodesGroup.selectAll("circle").classed("node-highlighted", false);
            edgesGroup.selectAll("line").classed("edge-highlighted", false);
        }}

        // Animate one group
        function animateGroup() {{
            const phases = [
                {{size: 2, groups: networkData.groups["2"]}},
                {{size: 3, groups: networkData.groups["3"]}},
                {{size: 4, groups: networkData.groups["4"]}}
            ];

            const phase = phases[currentPhase];
            const group = phase.groups[currentGroupIndex];
            const timing = calculateTiming(phase.size, currentGroupIndex, phase.groups.length);

            // Update stats
            updateStats(currentPhase, currentGroupIndex + 1, phase.groups.length);

            // Highlight this group
            highlightGroup(group);

            // Schedule next group or next phase
            currentGroupIndex++;

            if (currentGroupIndex >= phase.groups.length) {{
                // Move to next phase
                unhighlightAll();
                currentPhase++;
                currentGroupIndex = 0;

                if (currentPhase >= phases.length) {{
                    // End of all phases - show summary
                    showSummary();
                    animationTimer = setTimeout(() => {{
                        // Reset and restart
                        currentPhase = 0;
                        currentGroupIndex = 0;
                        unhighlightAll();
                        runAnimation();
                    }}, summaryPause);
                }} else {{
                    // Pause before next phase
                    animationTimer = setTimeout(animateGroup, pauseBetweenPhases);
                }}
            }} else {{
                // Continue with next group in current phase
                animationTimer = setTimeout(() => {{
                    unhighlightAll();
                    animateGroup();
                }}, timing);
            }}
        }}

        // Start animation
        function runAnimation() {{
            animateGroup();
        }}

        // Begin
        runAnimation();
    </script>
</body>
</html>
"""

    # Save HTML
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"✓ Generated animation HTML: {output_path}")


def main() -> None:
    """Generate network animation HTML."""
    output_dir = Path(__file__).parent / "output"
    data_path = output_dir / "network_groups_data.json"
    html_path = output_dir / "network_animation.html"

    if not data_path.exists():
        print(f"Error: {data_path} not found. Run generate_network_data.py first.")
        return

    generate_animation_html(data_path, html_path)
    print("\n✓ Done! Open network_animation.html to view the animation.")
    print("  Animation sequence:")
    print("    - Pairs: 200ms each (~3s total)")
    print("    - Triplets: 50ms each (~5s total)")
    print("    - Groups of 4: 40ms each (~18s total)")
    print("    - Summary: 3s pause")
    print("    - Total loop: ~30 seconds")


if __name__ == "__main__":
    main()
