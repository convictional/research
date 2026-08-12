"""
Network Effect Multiplier: Combinatorial Chart

Demonstrates the combinatorial explosion of possible communication groups
as team size grows. Shows C(n,k) for different group sizes k.

Research context:
- Possible groups of size k from n people: C(n,k) = n! / (k!(n-k)!)
- Example: 10 people → 45 pairs, 120 groups of 3, 210 groups of 4
- Teams typically have 2-3× as many channels as employees

Author: Adam McCabe
"""

import json
import math
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Configuration
MAX_TEAM_SIZE = 150
MIN_TEAM_SIZE = 1
GROUP_SIZES = [2, 3, 5, 10]

# Color palette: battleship gray → sky blue gradient
COLORS = {
    2: "#6B7F8F",  # Battleship gray (darkest)
    3: "#5A92B3",  # Medium blue
    5: "#4DA6D9",  # Light blue
    10: "#87CEEB",  # Sky blue (lightest)
}

# Layout styling
BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"
GRID_COLOR = "#E0E0E0"


def calculate_combinations(n: int, k: int) -> int:
    """
    Calculate C(n,k) = n! / (k!(n-k)!)

    Returns number of ways to choose k items from n items.
    Returns 0 if k > n.
    """
    if k > n:
        return 0
    return math.comb(n, k)


def generate_data() -> Dict[str, List]:
    """
    Generate combinatorial data for all team sizes and group sizes.

    Returns:
        Dictionary with:
            - team_sizes: list of n values
            - groups: dict mapping group_size -> list of counts
    """
    team_sizes = list(range(MIN_TEAM_SIZE, MAX_TEAM_SIZE + 1))
    groups: Dict[int, List[int]] = {k: [] for k in GROUP_SIZES}

    for n in team_sizes:
        for k in GROUP_SIZES:
            count = calculate_combinations(n, k)
            groups[k].append(count)

    return {
        "team_sizes": team_sizes,
        "groups": groups,
    }


def create_chart(data: Dict[str, List]) -> go.Figure:
    """
    Create interactive Plotly chart showing combinatorial explosion.

    Args:
        data: Dictionary with team_sizes and groups data

    Returns:
        Plotly figure object
    """
    team_sizes = data["team_sizes"]
    groups = data["groups"]

    fig = go.Figure()

    # Add trace for each group size
    for k in GROUP_SIZES:
        counts = groups[k]
        fig.add_trace(
            go.Scatter(
                x=team_sizes,
                y=counts,
                mode="lines",
                name=f"Groups of {k}",
                line=dict(color=COLORS[k], width=3),
                hovertemplate=(
                    f"<b>Groups of {k}</b><br>"
                    + "Team size: %{x}<br>"
                    + "Possible groups: %{y:,}<br>"
                    + "<extra></extra>"
                ),
            )
        )

    # No annotations - clean chart

    # Layout configuration
    fig.update_layout(
        title=dict(
            text="The Network Effect Multiplier: Combinatorial Explosion of Communication Groups",
            font=dict(size=20, color=TEXT_COLOR, family="Arial, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Team Size (number of people)", font=dict(size=14, color=TEXT_COLOR)),
            range=[0, 150],
            showgrid=True,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=12, color=TEXT_COLOR),
        ),
        yaxis=dict(
            title=dict(text="Number of Possible Groups", font=dict(size=14, color=TEXT_COLOR)),
            type="linear",  # Linear scale for better intuition
            range=[0, 10000],  # 10k max - larger groups go off top
            showgrid=True,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=12, color=TEXT_COLOR),
        ),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        hovermode="closest",
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(size=12),
        ),
        width=1000,
        height=700,
        margin=dict(l=80, r=40, t=100, b=80),
    )

    return fig


def save_outputs(data: Dict[str, List], fig: go.Figure, output_dir: Path) -> None:
    """
    Save chart HTML and data JSON to output directory.

    Args:
        data: Combinatorial data
        fig: Plotly figure
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(exist_ok=True)

    # Generate base HTML
    html_base = fig.to_html(
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["pan2d", "select2d", "lasso2d", "autoScale2d"],
        },
        include_plotlyjs="cdn",
        div_id="plotly-chart",
    )

    # Prepare data for embedding
    json_data = {
        "team_sizes": data["team_sizes"],
        "groups": {str(k): [int(v) for v in vals] for k, vals in data["groups"].items()},
    }

    # Custom HTML with slider and summary card
    custom_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Network Effect Multiplier</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #FFFFFF;
            color: #333333;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            position: relative;
        }}
        .slider-container {{
            position: absolute;
            right: 0;
            top: 120px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            z-index: 1000;
        }}
        .slider-label {{
            font-size: 13px;
            font-weight: bold;
            color: #333;
            text-align: center;
            line-height: 1.3;
        }}
        .slider-value {{
            font-size: 20px;
            font-weight: bold;
            color: #FF6B6B;
            margin-top: 5px;
        }}
        input[type="range"] {{
            writing-mode: bt-lr;
            -webkit-appearance: slider-vertical;
            width: 30px;
            height: 400px;
            padding: 0;
            margin: 10px 0;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #FF6B6B;
            cursor: pointer;
        }}
        input[type="range"]::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #FF6B6B;
            cursor: pointer;
            border: none;
        }}
        .summary-card {{
            margin-top: 20px;
            padding: 20px;
            background: #F8F9FA;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
        }}
        .summary-title {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .summary-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .summary-item {{
            padding: 8px 0;
            font-size: 14px;
            border-bottom: 1px solid #E0E0E0;
        }}
        .summary-item:last-child {{
            border-bottom: none;
        }}
        .summary-label {{
            display: inline-block;
            width: 120px;
            color: #666;
        }}
        .summary-value {{
            font-weight: bold;
            color: #333;
        }}
        .off-chart {{
            color: #999;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="slider-container">
            <div class="slider-label">
                Slide to<br>your team size
            </div>
            <div class="slider-value" id="teamSizeDisplay">20</div>
            <input type="range"
                   id="teamSizeSlider"
                   min="1"
                   max="150"
                   value="20"
                   orient="vertical">
        </div>

        {html_base}

        <div class="summary-card">
            <div class="summary-title" id="summaryTitle">Your Team (20 people):</div>
            <ul class="summary-list" id="summaryList">
                <li class="summary-item">
                    <span class="summary-label">Pairs:</span>
                    <span class="summary-value" id="value-2">190</span>
                </li>
                <li class="summary-item">
                    <span class="summary-label">Groups of 3:</span>
                    <span class="summary-value" id="value-3">1,140</span>
                </li>
                <li class="summary-item">
                    <span class="summary-label">Groups of 5:</span>
                    <span class="summary-value" id="value-5">15,504</span>
                </li>
                <li class="summary-item">
                    <span class="summary-label">Groups of 10:</span>
                    <span class="summary-value" id="value-10">184,756</span>
                </li>
            </ul>
        </div>
    </div>

    <script>
        // Embedded combinatorial data
        const combinatorialData = {json.dumps(json_data)};

        // Format numbers for display
        function formatNumber(n) {{
            if (n >= 1e12) return (n/1e12).toFixed(1) + ' trillion';
            if (n >= 1e9) return (n/1e9).toFixed(1) + ' billion';
            if (n >= 1e6) return (n/1e6).toFixed(1) + ' million';
            if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
            return n.toLocaleString();
        }}

        // Update chart and summary
        function updateVisualization(teamSize) {{
            const idx = teamSize - 1;

            // Update display
            document.getElementById('teamSizeDisplay').textContent = teamSize;
            document.getElementById('summaryTitle').textContent = `Your Team (${{teamSize}} ${{teamSize === 1 ? 'person' : 'people'}}):`;

            // Update values
            const groupSizes = [2, 3, 5, 10];
            groupSizes.forEach(k => {{
                const count = combinatorialData.groups[k.toString()][idx];
                const element = document.getElementById(`value-${{k}}`);

                if (count > 10000) {{
                    element.innerHTML = `<span class="off-chart">${{formatNumber(count)}} (off chart)</span>`;
                }} else {{
                    element.textContent = formatNumber(count);
                }}
            }});

            // Update vertical line on chart
            Plotly.relayout('plotly-chart', {{
                shapes: [{{
                    type: 'line',
                    x0: teamSize,
                    x1: teamSize,
                    y0: 0,
                    y1: 10000,
                    line: {{
                        color: '#FF6B6B',
                        width: 2,
                        dash: 'dash'
                    }}
                }}]
            }});
        }}

        // Slider event listener
        document.getElementById('teamSizeSlider').addEventListener('input', (e) => {{
            updateVisualization(parseInt(e.target.value));
        }});

        // Initialize with default value (20)
        window.addEventListener('load', () => {{
            setTimeout(() => {{
                updateVisualization(20);
            }}, 100);
        }});
    </script>
</body>
</html>
"""

    # Save enhanced HTML
    html_path = output_dir / "combinatorial_chart.html"
    with open(html_path, "w") as f:
        f.write(custom_html)
    print(f"✓ Saved interactive chart: {html_path}")

    # Save data as JSON (for GCS storage or reuse)
    json_path = output_dir / "combinatorial_data.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved data JSON: {json_path}")


def main() -> None:
    """Generate combinatorial chart and save outputs."""
    print("Generating combinatorial explosion chart...")
    print(f"  Team size range: {MIN_TEAM_SIZE}-{MAX_TEAM_SIZE}")
    print(f"  Group sizes: {GROUP_SIZES}")

    # Generate data
    data = generate_data()

    # Print some example values
    print("\nExample combinations:")
    for n in [10, 50, 100, 250]:
        if n <= MAX_TEAM_SIZE:
            idx = n - MIN_TEAM_SIZE
            print(f"  {n} people:")
            for k in GROUP_SIZES:
                count = data["groups"][k][idx]
                print(f"    Groups of {k}: {count:,}")

    # Create chart
    fig = create_chart(data)

    # Save outputs
    output_dir = Path(__file__).parent / "output"
    save_outputs(data, fig, output_dir)

    print("\n✓ Done! Open combinatorial_chart.html to view the visualization.")


if __name__ == "__main__":
    main()
