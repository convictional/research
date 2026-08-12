# Network Effect Multiplier Visualization

## Overview

This visualization demonstrates the combinatorial explosion of potential communication groups as team size grows. It consists of two complementary visualizations:

1. **Combinatorial Chart**: Interactive Plotly chart showing C(n,k) growth for different group sizes
2. **Network Growth Animation**: D3.js force-directed animation showing groups emerging as people join

## Research Context

From research findings:
- Teams typically have 2-3× as many channels as employees
- 92 messages/day per user (Slack data 2025)
- Possible groups of size k from n people: C(n,k) = n! / (k!(n-k)!)
- Example: 10 people → 45 pairs, 120 groups of 3, 210 groups of 4

## Files

- `generate_combinatorial_chart.py`: Generates interactive Plotly chart
- `generate_network_data.py`: Pre-calculates network layouts and group data
- `network_animation.html`: D3.js animated visualization
- `network_animation.js`: Animation logic
- `output/`: Generated HTML and JSON files

## Usage

### Generate Combinatorial Chart

```bash
poetry run python animations/network_effect/generate_combinatorial_chart.py
```

Outputs:
- `output/combinatorial_chart.html` - Standalone interactive chart
- `output/combinatorial_data.json` - Data for GCS storage/reuse

### Generate Network Animation

```bash
# Step 1: Generate data
poetry run python animations/network_effect/generate_network_data.py

# Step 2: Open the animation
open animations/network_effect/output/network_animation.html
```

## Configuration

### Combinatorial Chart
- Team size range: 2-250 people
- Group sizes: k = [2, 3, 5, 10]
- Y-axis: Logarithmic scale
- Colors: Muted palette matching essay style guide

### Network Animation
- Team size range: 2-20 people (for visual clarity)
- Animation speed: ~1-2 seconds per new person
- Group highlighting: Brief pulse effect on new groups
- Loop: Continuous (resets after reaching 20 people)

## Key Parameters

Edit these in the scripts to adjust visualization:

**generate_combinatorial_chart.py:**
- `MAX_TEAM_SIZE`: Maximum team size to plot (default: 250)
- `GROUP_SIZES`: Which group sizes to show (default: [2, 3, 5, 10])
- `COLORS`: Color palette for traces

**generate_network_data.py:**
- `MAX_NODES`: Maximum team size for animation (default: 20)
- `LAYOUT_SEED`: Random seed for reproducible layouts
- `GROUP_SIZES`: Which group sizes to calculate

## Output Files

### combinatorial_chart.html
Standalone interactive Plotly chart. Can be embedded in essay via:
- `<iframe>` tag
- Direct HTML injection
- Link to hosted version

### combinatorial_data.json
```json
{
  "team_sizes": [2, 3, 4, ...],
  "groups": {
    "2": [1, 3, 6, ...],
    "3": [0, 1, 4, ...],
    "5": [0, 0, 0, ...],
    "10": [0, 0, 0, ...]
  }
}
```

### network_growth_data.json
```json
{
  "frames": [
    {
      "n": 2,
      "nodes": [{"id": 0, "x": 0.5, "y": 0.5}, ...],
      "edges": [[0, 1]],
      "new_groups": {"2": 1, "3": 0, "5": 0, "10": 0},
      "total_groups": {"2": 1, "3": 0, "5": 0, "10": 0}
    },
    ...
  ]
}
```

## Design Notes

### Color Palette
- Groups of 2: Muted blue (#4C78A8)
- Groups of 3: Muted green (#59A14F)
- Groups of 5: Muted orange (#F28E2B)
- Groups of 10: Muted purple (#B279A2)

### Typography
- Chart titles: Sans-serif, clear hierarchy
- Axis labels: Readable, generous spacing
- Tooltips: Concise, show exact values

## Known Limitations

- Network animation limited to 20 people for visual clarity
- Group highlighting for k>5 uses simplified convex hull (actual groups are more complex)
- Layout algorithm may occasionally produce overlapping nodes (fixed seed helps)
- Animation performance depends on browser (tested on Chrome/Safari)

## Future Enhancements

Potential improvements:
- Interactive controls: play/pause, speed slider, group size filter
- Click node to highlight all groups including that person
- Comparison view: show multiple team sizes side-by-side
- Integration with "Your Interruption Profile" calculator (visualization #2)
