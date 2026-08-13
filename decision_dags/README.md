# Decision DAGs Experiment

**Author:** Adam McCabe

A strategic planning system that builds Decision DAGs (Directed Acyclic Graphs) to explore decision spaces, evaluate strategic paths, and evolve optimal solutions.

## Documentation

📚 **[Technical Documentation](./DOCUMENTATION.md)** - Complete system architecture, implementation details, and API reference
🧠 **[Learnings & Journey](./LEARNINGS.md)** - Key insights, challenges overcome, and lessons learned during development

## Quick Start

```bash
# Initialize the database (creates database if it doesn't exist)
poetry run python -m decision_dags db init

# Create a new Decision DAG
poetry run python -m decision_dags create --problem "How to expand our business internationally?"

# List all DAGs
poetry run python -m decision_dags list

# Show details of a specific DAG
poetry run python -m decision_dags show --dag-id <uuid>
```

### Prerequisites

- PostgreSQL 15+ installed and running
- PostgreSQL user with database creation privileges (default: uses local user)
- Python 3.11+

The system will automatically create the `decision_dags_experiment` database if it doesn't exist when you run `db init`.

## Directory Structure

```
decision_dags/
├── src/                      # Main source code
│   ├── dag_builder/         # DAG construction components
│   ├── path_evolution/      # Path optimization using genetic algorithms
│   ├── persistence/         # PostgreSQL database persistence
│   ├── context/            # Database and organizational context
│   ├── hitl/              # Human-in-the-loop interfaces
│   ├── prompts/            # Jinja2 prompt templates
│   ├── utils/              # Utilities and CSV logging
│   ├── models.py          # Data models and schemas
│   ├── main.py            # Core orchestration logic
│   ├── cli.py             # Command-line interface
│   └── settings.py        # Configuration
├── tests/                   # Test suite
├── output/                  # Generated DAGs and CSV logs
└── docs/                    # Additional documentation
```

## Key Features

- **Persistent Storage**: PostgreSQL-backed storage for DAGs, nodes, and edges
- **Parallel DAG Building**: Constructs decision trees using multiple LLM agents
- **Path Evolution**: Optimizes strategic paths using genetic algorithms
- **Path Stitching**: Merges evolved paths back into unified DAGs
- **Human-in-the-Loop**: Optional interactive refinement of decisions
- **CSV Logging**: Comprehensive metrics and DAG structure logging
- **Export Capabilities**: Export DAGs to JSON, CSV, or Graphviz formats
- **Interactive Visualization**: Web-based DAG explorer with:
  - **Single Mode**: View individual DAGs with full details
  - **Comparison Mode**: Side-by-side comparison of original and evolved DAGs
  - **Evolution Journey Mode**: Three-way comparison showing the complete evolution process:
    - Original DAG → Extracted Path → Evolved Path
    - Track node count and confidence changes at each stage
    - View evolution performance metrics (fitness scores, generations)
    - Understand the impact of path extraction and evolution
  - Hierarchical layout with color-coded nodes by type and decision category
  - Hover details for nodes and edges
  - Critical path highlighting and analysis
  - Search and filter capabilities
  - Real-time DAG switching
  - Evolution statistics showing changes between DAGs
  - Export visualizations as HTML

## CLI Commands

### Database Management

```bash
# Initialize database tables
poetry run python -m decision_dags db init

# Drop all database tables (WARNING: deletes all data)
poetry run python -m decision_dags db drop
```

### DAG Operations

```bash
# Create a new Decision DAG
poetry run python -m decision_dags create --problem "Strategic problem statement" \
    --max-layers 6 \
    --timeout 180 \
    --enable-evolution \
    --stitching-strategy balanced

# List saved DAGs
poetry run python -m decision_dags list \
    --filter-by evolved \
    --limit 20 \
    --sort-by created_at

# Show DAG details
poetry run python -m decision_dags show --dag-id <uuid> \
    --include-nodes \
    --include-edges \
    --include-metrics

# Export a DAG
poetry run python -m decision_dags export --dag-id <uuid> \
    --format json \
    --output-dir ./exports

# Delete a DAG
poetry run python -m decision_dags delete --dag-id <uuid> --cascade
```

### Path Operations

```bash
# Extract strategic paths from a DAG
poetry run python -m decision_dags extract-paths --dag-id <uuid> \
    --min-length 3 \
    --max-length 10

# Evolve paths using genetic algorithms
poetry run python -m decision_dags evolve --dag-id <uuid> \
    --generations 10 \
    --population-size 8 \
    --mutation-rate 0.7 \
    --top-k-paths 5

# Stitch evolved paths back into a DAG
poetry run python -m decision_dags stitch --dag-id <uuid> \
    --strategy balanced
```

### Visualization

```bash
# Launch interactive web visualization server
poetry run python -m decision_dags visualize \
    --port 5006 \
    --host localhost

# Specify initial DAG to display
poetry run python -m decision_dags visualize --dag-id <uuid>
```

The visualization interface provides an interactive web-based DAG explorer with three distinct modes:

1. **Single Mode** (default): Browse and explore individual DAGs
   - Select any DAG from the dropdown (build, extracted, or evolved)
   - View comprehensive DAG information including creation date, parent relationships, and metadata
   - Examine node distribution statistics and decision type breakdowns
   - Toggle node labels for cleaner visualization
   - Search nodes by title or description with real-time highlighting
   - Enable critical path highlighting to identify key decision sequences
   - View detailed path analysis showing node sequences

2. **Comparison Mode**: Compare any two DAGs side-by-side
   - Select any two DAGs for comparison (not limited to parent-child relationships)
   - View both DAGs in synchronized side-by-side layout
   - Dynamic subplot titles based on DAG types (Build, Extracted, Evolved)
   - Comprehensive comparison statistics showing:
     - Node and edge count differences
     - Average confidence score changes
     - Decision type distribution comparison
     - Node type (decision vs option) changes
   - Synchronized controls for labels, search, and path highlighting

3. **Evolution Journey Mode**: Trace the complete evolution process
   - Select a build DAG that has been through the evolution pipeline
   - View count of evolvable paths (extracted paths that have evolved versions)
   - Select an extracted path from the filtered list (only shows paths with evolved children)
   - Select an evolved version of that path
   - Three-column visualization showing the complete journey:
     - Original build DAG with full decision tree
     - Extracted path showing focused strategic route
     - Evolved path showing optimized version
   - Comprehensive journey statistics including:
     - Node count evolution across all three stages
     - Confidence score progression
     - Path extraction impact (reduction in complexity)
     - Evolution impact (improvements and modifications)
     - Evolution performance metrics (generations, fitness scores, variant counts)

The interface features a clean Material Design layout with:
- Mode selector buttons for easy switching between views
- Cascading dropdowns that update based on DAG relationships
- Real-time search across all displayed DAGs
- Responsive Plotly graphs with zoom, pan, and hover details
- Collapsible side panel with detailed statistics and analysis
- Automatic height adjustment based on DAG complexity

### Command Options

#### Create Command
- `--problem`: Problem statement to solve (required)
- `--max-layers`: Maximum DAG depth (default: 6)
- `--timeout`: Agent timeout in seconds (default: 30)
- `--enable-hitl`: Enable Human-in-the-Loop workflows
- `--enable-evolution` / `--no-evolution`: Enable/disable path evolution
- `--stitching-strategy`: Path stitching strategy (balanced/conservative/aggressive)

#### List Command
- `--filter-by`: Filter by generation method (build/extracted/evolved)
- `--limit`: Maximum results to show (default: 10)
- `--offset`: Number of results to skip
- `--sort-by`: Sort field (created_at/updated_at/node_count/max_layers)
- `--ascending`: Sort in ascending order

#### Evolution Command
- `--generations`: Number of evolution generations (default: 10)
- `--population-size`: Population size (default: 8)
- `--mutation-rate`: Mutation rate (default: 0.7)
- `--top-k-paths`: Number of top paths to select for evolution based on initial fitness (default: 5, use -1 for all)

#### Visualize Command
- `--dag-id`: UUID of DAG to visualize initially
- `--port`: Port for visualization server (default: 5006)
- `--host`: Host for visualization server (default: localhost)
- `--no-browser`: Don't open browser automatically

## Configuration

Key settings in `src/settings.py`:
- `max_layers`: DAG depth (default: 6)
- `agent_timeout`: LLM timeout in seconds (default: 30)
- `enable_csv_logging`: Detailed logging (default: true)
- `local_postgres_db`: Database name (default: "decision_dags_experiment")

## Workflow Example

```bash
# 1. Initialize the database (first time only)
poetry run python -m decision_dags db init

# 2. Create a DAG with full evolution pipeline
poetry run python -m decision_dags create \
    --problem "How should we optimize our supply chain?" \
    --max-layers 8 \
    --enable-evolution

# 3. View the created DAG
poetry run python -m decision_dags list
poetry run python -m decision_dags show --dag-id <uuid> --include-nodes

# 4. Export for analysis
poetry run python -m decision_dags export --dag-id <uuid> --format json

# 5. For manual path optimization:
# Extract paths
poetry run python -m decision_dags extract-paths --dag-id <uuid>

# Evolve the paths (top 5 by default, or specify --top-k-paths)
poetry run python -m decision_dags evolve --dag-id <uuid> --generations 20 --top-k-paths 10

# Stitch back into final DAG
poetry run python -m decision_dags stitch --dag-id <uuid> --strategy aggressive

# 6. Launch visualization to explore results
poetry run python -m decision_dags visualize
```

## Output Files

The system generates CSV files in the `output/` directory:
- `orchestration_metrics_*.csv` - Overall process metrics
- `dag_building_metrics_*.csv` - Layer-by-layer building stats
- `dag_nodes_*.csv` - Complete node details
- `dag_edges_*.csv` - Edge relationships and enrichment
- `evolution_metrics_*.csv` - Path optimization results

## Documentation

- [Full Documentation](./DOCUMENTATION.md) - Detailed architecture and implementation
- [Learnings](./LEARNINGS.md) - Journey, challenges, and design decisions

## Database Setup & Troubleshooting

### Automatic Database Creation

The system will automatically create the database when you run `poetry run python -m decision_dags db init`. This requires:
- PostgreSQL user with CREATE DATABASE permission
- Connection to PostgreSQL server

### Manual Database Setup

If automatic creation fails, you can create the database manually:

```sql
-- Connect to PostgreSQL as a superuser
CREATE DATABASE decision_dags_experiment;
```

### Common Issues

1. **Connection Refused**
   ```
   Could not connect to PostgreSQL at localhost:5432
   ```
   - Ensure PostgreSQL is running: `brew services start postgresql@15` (macOS) or `sudo systemctl start postgresql` (Linux)

2. **Authentication Failed**
   ```
   Invalid password for PostgreSQL user
   ```
   - Check credentials in `src/settings.py`
   - For local development, you might need to update `pg_hba.conf` to use `trust` authentication

3. **Insufficient Privileges**
   ```
   User does not have permission to create databases
   ```
   - Grant permission: `ALTER USER your_username CREATEDB;`
   - Or create the database manually as shown above

4. **Database Already Exists**
   - This is fine! The system will use the existing database
   - To start fresh: `python -m decision_dags db drop` then `python -m decision_dags db init`
