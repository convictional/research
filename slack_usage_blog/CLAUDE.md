# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an interactive essay exploring the cognitive and decision-making impact of workplace communication platforms (specifically Slack) through research-based visualizations. The project creates physics-based animations and data visualizations to demonstrate attention residue, network effects, and interruption costs.

**Target Output:** Interactive HTML essay with embedded animations and visualizations
**Tech Stack:** Python (Matplotlib, Pymunk, Plotly), HTML/CSS/JavaScript for interactivity

## Build Commands

### Environment Setup
- Install dependencies: `uv sync`
- Run commands with environment: `uv run python <script>`

### Running Animations
- Run animation scripts directly: `uv run python animations/attention_residue/generate_funnel_visual.py`
- Output files are generated in `animations/attention_residue/output/` directory

### Code Quality
- Format code: `uv run ruff format .`
- Lint code: `uv run ruff check .`
- Line length limit: 119 characters (configured in pyproject.toml)

## Project Structure

```
slack_usage_blog/
├── animations/              # Animation generation scripts
│   └── attention_residue/  # Physics-based funnel visualization
│       ├── generate_funnel_visual.py  # Main animation script
│       └── output/         # Generated MP4/GIF files
├── media/                  # Manim-generated assets (intermediate)
├── essay_plan.md          # Detailed plan for interactive essay sections
├── research_findings.md   # Research data and findings
├── slack_productivity_research.md  # Literature review and citations
└── pyproject.toml         # Python dependencies
```

## Key Dependencies

- **pymunk** (^7.1.0): 2D physics engine (Chipmunk2D wrapper) for particle simulations
- **matplotlib** (^3.10.6): Animation rendering and export to MP4/GIF
- **manim** (^0.18.0): Mathematical animation engine (used for earlier prototypes)
- **plotly** (^6.0.0): Interactive charts for the final HTML essay
- **numpy** (^2.0.0): Numerical computations

## Animation System

### Attention Residue Funnel Visualization

**File:** `animations/attention_residue/generate_funnel_visual.py`

**Purpose:** Demonstrates cognitive "attention residue" through a physics-based funnel metaphor where colored particles represent different tasks mixing and draining.

**Key Parameters** (edit in script):
- `TASKS`: List of tasks with colors, particle counts, pour times, and gaps
- `FPS`: Frame rate (default: 30)
- `PARTICLE_RADIUS`, `PARTICLE_MASS`: Granular physics parameters
- Funnel geometry: `Y_TOP`, `Y_NECK_TOP`, `X_TOP_HALF`, `X_NECK_HALF`

**Physics Model:**
- Real collision detection via Pymunk's Chipmunk2D engine
- Gravity, damping, elasticity, and friction configured for realistic particle flow
- Sweeping emitter moves across funnel mouth during each pour
- Neck drainage between pours visualizes "residue" effect

**Output:**
- MP4 video (if ffmpeg available) or GIF (fallback)
- Saved to `animations/attention_residue/output/`

**Common Edits:**
- Adjust task colors/timings in `TASKS` list
- Modify funnel shape by changing geometry constants
- Tweak physics parameters for different visual effects
- Change resolution via `DPI` and `FIGSIZE` constants

## Research Context

### Key Research Findings (Quantified)

The essay is grounded in peer-reviewed research:

- **23 min 15 sec**: Time to fully refocus after interruption (Gloria Mark, UC Irvine)
- **165+ interruptions/day**: Combined email + Slack checks (estimated from research)
- **12% increase**: Diagnostic errors per interruption (AHRQ healthcare studies)
- **3-minute average**: Task switching frequency in knowledge work
- **3× daily batching**: Optimal notification frequency for reduced stress (Fitz et al., 2019)
- **40% productivity loss**: From task-switching costs (Rubinstein et al., APA)

See `slack_productivity_research.md` for full literature review and citations.

### Essay Structure (Planned)

**7 Interactive Visualizations:**
1. Network Effect Multiplier (combinatorial explosion of communication groups)
2. Your Interruption Profile (personalized calculator)
3. The Recovery Window Impossibility (timeline animation)
4. Lost Opportunity Cost Calculator (economic impact)
5. Attention Residue Accumulator (physics simulation - current focus)
6. ~~3× Daily Batching Solution~~ (integrated into #3)
7. The Cognitive Cost Surface (3D visualization)

See `essay_plan.md` for complete specification.

## Code Style Guidelines

Follow parent experiment guidelines in `/experiments/CLAUDE.md`:

- **Imports**: Standard lib → third-party → internal (grouped with blank lines)
- **Type hints**: Required for all function parameters and return values
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Line length**: 119 characters max
- **Docstrings**: Google style for complex functions

### Project-Specific Conventions

- **Physics constants**: Use descriptive UPPER_CASE names (e.g., `PARTICLE_RADIUS`, `Y_NECK_TOP`)
- **Color specifications**: Use hex strings for consistency (e.g., `"#4C78A8"`)
- **Animation timing**: Specify in seconds as floats (e.g., `pour_time: float`)
- **Coordinate system**: Physics world uses arbitrary consistent units; document scales

## Development Workflow

### Adding a New Animation

1. Create subdirectory under `animations/` for the visualization
2. Create `output/` directory for generated files
3. Follow the structure of `attention_residue/generate_funnel_visual.py`:
   - Configuration constants at top
   - Helper functions in middle
   - Main execution logic at bottom
4. Generate output to `output/` subdirectory with descriptive filenames
5. Update this CLAUDE.md with new animation parameters/usage

### Modifying Physics Simulations

**Critical Parameters:**
- `DT` (timestep): Smaller = more stable but slower (current: 1/600 seconds)
- `STEPS_PER_FRAME`: Physics steps per rendered frame (current: ~20)
- `space.iterations`: Collision solver accuracy (current: 30)
- `space.damping`: Global velocity damping (current: 0.9)

**Collision Tuning:**
- Shape `elasticity`: Bounciness (0 = inelastic, 1 = perfectly elastic)
- Shape `friction`: Surface friction (higher = more resistance)
- Segment `radius`: Wall thickness affects collision detection

**Visual Quality:**
- Increase `DPI` for higher resolution (impacts file size)
- Adjust `PARTICLE_RADIUS` scaling in scatter plot for visual clarity
- Use `to_rgba()` with alpha channel for particle transparency

## Data and Research Files

- **essay_plan.md**: Comprehensive specification for all 7 visualizations, including interactive elements, research citations, and technical implementation notes
- **research_findings.md**: Structured research data including Slack usage statistics, network dynamics, recovery curve modeling, and data gaps requiring sensitivity analysis
- **slack_productivity_research.md**: Deep literature review covering interruption costs, attention residue theory, cognitive impacts, and healthcare error studies

**Important:** All visualizations must be grounded in cited research. Make assumptions transparent when data is unavailable.

## Known Limitations and Future Work

### Current Status
- ✅ Attention residue funnel animation (physics-based)
- 🚧 Interactive HTML essay (not yet started)
- 🚧 Network visualization (combinatorial graphs)
- 🚧 Recovery timeline animation
- 🚧 Economic cost calculator

### Data Gaps
- No public data on DM spawn rates from channels
- Thread adoption rates vary widely (10-45%)
- Recovery curve functional form is modeling assumption (exponential approach)

Approach: Use sensitivity analysis and make assumptions transparent in visualizations.

## Transparency and Bias

**Important Context:**
- Authors work at Convictional, building an email-focused product
- Not Slack users; exploring questions from first principles
- Research-based approach to understand cognitive costs objectively
- Acknowledges genuine benefits of real-time communication for some work types

The essay explicitly discloses this context and focuses on presenting research findings rather than prescriptive solutions.
