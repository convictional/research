"""Panel web application for DAG visualization."""

import asyncio
import logging
import panel as pn
import param
from datetime import datetime
from typing import Dict, Any
from uuid import UUID
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..persistence.dag_repository import dag_repository
from ..persistence.database import init_db, close_db
from .dag_visualizer import DAGVisualizer
from ..settings import settings


logger = logging.getLogger(__name__)

# Enable Panel extensions
pn.extension("plotly", "tabulator")


class DAGVisualizationApp(param.Parameterized):
    """Interactive DAG visualization application with comparison mode."""

    comparison_mode = param.Boolean(default=False, doc="Enable comparison mode")
    show_labels = param.Boolean(default=True, doc="Show node labels")
    search_text = param.String(default="", doc="Search nodes by text")
    highlight_critical_paths = param.Boolean(default=False, doc="Highlight critical paths")

    def __init__(self, **params):
        super().__init__(**params)
        # Current DAGs and visualizers
        self.original_dag = None
        self.original_visualizer = None
        self.extracted_dag = None
        self.extracted_visualizer = None
        self.evolved_dag = None
        self.evolved_visualizer = None

        # DAG lists
        self.all_dags = []
        self.build_dags = []
        self.extracted_dags_map = {}  # Maps parent ID to list of extracted paths
        self.evolved_dags_map = {}  # Maps parent ID to list of evolved DAGs

        # UI row references (set in get_layout)
        self.evolved_row = None

        # UI components - Mode selector
        self.mode_selector = pn.widgets.RadioButtonGroup(
            name="Visualization Mode",
            value="Single",
            options=["Single", "Comparison", "Evolution Journey"],
            button_type="primary",
            width=450,
        )
        self.mode_selector.param.watch(self._on_mode_changed, "value")

        # Original DAG selector (used in both modes)
        self.original_selector = pn.widgets.Select(name="Select DAG", options={}, value=None, width=500)
        self.original_selector.param.watch(self._on_original_selected, "value")

        # Extracted path selector (only for Evolution Journey mode)
        self.extracted_selector = pn.widgets.Select(
            name="Select Extracted Path", options={}, value=None, width=500, disabled=True
        )
        self.extracted_selector.param.watch(self._on_extracted_selected, "value")

        # Evolved DAG selector (only for comparison/journey modes)
        self.evolved_selector = pn.widgets.Select(
            name="Select Evolved DAG", options={}, value=None, width=500, disabled=True
        )
        self.evolved_selector.param.watch(self._on_evolved_selected, "value")

        # Control buttons
        self.refresh_button = pn.widgets.Button(name="Refresh DAGs", button_type="primary", width=100)
        self.refresh_button.on_click(self._refresh_dags)

        # LLM analysis button (only visible in Comparison mode)
        self.analyze_diff_button = pn.widgets.Button(
            name="🤖 Analyze Differences", button_type="success", width=180, visible=False
        )
        self.analyze_diff_button.on_click(self._analyze_differences)

        self.show_labels_toggle = pn.widgets.Toggle(name="Show Labels", value=self.show_labels, width=100)
        self.show_labels_toggle.param.watch(self._update_visualization, "value")

        self.search_input = pn.widgets.TextInput(
            name="Search Nodes", placeholder="Search by title or description...", width=300
        )
        self.search_input.param.watch(self._on_search, "value")

        self.highlight_paths_toggle = pn.widgets.Toggle(
            name="Highlight Paths", value=self.highlight_critical_paths, width=120
        )
        self.highlight_paths_toggle.param.watch(self._update_visualization, "value")

        # Placeholder for visualization
        self.plot_pane = pn.pane.Plotly(config={"responsive": True, "displayModeBar": True})

        # Info panes
        self.dag_info_pane = pn.pane.Markdown("Select a DAG to view details")
        self.stats_pane = pn.pane.Markdown("No statistics available")
        self.paths_pane = pn.pane.Markdown("No paths analyzed")
        self.comparison_pane = pn.pane.Markdown("")  # For comparison statistics
        self.llm_analysis_pane = pn.pane.Markdown(
            "", styles={"background-color": "#f0f8ff", "padding": "10px", "border-radius": "5px"}
        )

        # Initialize
        self._initialize_task = asyncio.create_task(self._initialize())

    async def _initialize(self):
        """Initialize the application."""
        try:
            # Initialize database
            await init_db()

            # Initialize LLM client for analysis features
            from common.instruct_llm import set_async_instructor_client
            from .. import prompts  # This triggers prompt template registration

            # Determine which API key to use based on the model
            api_key = (
                settings.anthropic_api_key if settings.llm_model.startswith("claude") else settings.openai_api_key
            )

            set_async_instructor_client(llm_model=settings.llm_model, api_key=api_key)

            # Refresh DAGs
            await self._refresh_dags(None)
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            if pn.state.notifications:
                pn.state.notifications.error(f"Failed to initialize: {str(e)}", duration=5000)

    async def _refresh_dags(self, event):
        """Refresh the list of available DAGs."""
        try:
            # Get all DAGs
            self.all_dags = await dag_repository.list_dags(limit=1000)

            # Separate DAGs by type and create maps
            self.build_dags = []
            self.extracted_dags_map = {}
            self.evolved_dags_map = {}

            # Debug: log all generation methods
            generation_methods = set(dag.generation_method for dag in self.all_dags)
            logger.info(f"Found generation methods: {generation_methods}")

            # First pass: collect all build DAGs and initialize maps
            for dag in self.all_dags:
                if dag.generation_method == "build":
                    self.build_dags.append(dag)
                    # Initialize maps if not already present
                    if str(dag.id) not in self.extracted_dags_map:
                        self.extracted_dags_map[str(dag.id)] = []
                    if str(dag.id) not in self.evolved_dags_map:
                        self.evolved_dags_map[str(dag.id)] = []

            # Second pass: add extracted and evolved DAGs to their parents
            for dag in self.all_dags:
                if dag.parent_dag_id:
                    parent_id = str(dag.parent_dag_id)

                    if dag.generation_method == "extracted":
                        # This is an extracted path
                        if parent_id not in self.extracted_dags_map:
                            self.extracted_dags_map[parent_id] = []
                        self.extracted_dags_map[parent_id].append(dag)
                        logger.info(f"Added extracted path {dag.id} to parent {parent_id}")

                        # Also initialize evolved map for this extracted path
                        if str(dag.id) not in self.evolved_dags_map:
                            self.evolved_dags_map[str(dag.id)] = []

                    elif dag.generation_method == "evolved":
                        # This could be evolved from either a build DAG or an extracted path
                        if parent_id not in self.evolved_dags_map:
                            self.evolved_dags_map[parent_id] = []
                        self.evolved_dags_map[parent_id].append(dag)
                        logger.info(f"Added evolved DAG {dag.id} to parent {parent_id}")

            # Debug logging
            logger.info(f"Found {len(self.build_dags)} build DAGs")
            logger.info(f"Extracted paths map keys: {list(self.extracted_dags_map.keys())}")
            logger.info(f"Evolved DAGs map keys: {list(self.evolved_dags_map.keys())}")

            for parent_id, extracted in self.extracted_dags_map.items():
                if extracted:
                    logger.info(f"Parent {parent_id} has {len(extracted)} extracted paths")

            for parent_id, evolved in self.evolved_dags_map.items():
                if evolved:
                    logger.info(f"Parent {parent_id} has {len(evolved)} evolved DAGs")

            # Update selectors based on mode
            await self._update_selectors()

            if pn.state.notifications:
                pn.state.notifications.success("DAG list refreshed", duration=2000)

        except Exception as e:
            logger.error(f"Failed to refresh DAGs: {e}")
            if pn.state.notifications:
                pn.state.notifications.error(f"Failed to refresh DAGs: {str(e)}", duration=5000)

    async def _update_selectors(self):
        """Update selector options based on current mode."""
        if self.mode_selector.value in ["Single", "Comparison"]:
            # In single and comparison modes, show all DAGs
            options = {}
            for dag in self.all_dags:
                label = f"{dag.problem_statement[:50]}... ({dag.generation_method}) - {dag.created_at.strftime('%Y-%m-%d %H:%M')}"
                options[label] = str(dag.id)

            if self.mode_selector.value == "Single":
                self.original_selector.name = "Select DAG"
            else:
                self.original_selector.name = "Select First DAG"

            self.original_selector.options = options

            if options and not self.original_selector.value:
                self.original_selector.value = list(options.values())[0]
        else:
            # In Evolution Journey mode, show only build DAGs
            options = {}
            for dag in self.build_dags:
                # Count extracted paths that have evolved children
                dag_id = str(dag.id)
                extracted_paths = self.extracted_dags_map.get(dag_id, [])

                # Count how many extracted paths have evolved children
                evolvable_count = 0
                for extracted in extracted_paths:
                    extracted_id = str(extracted.id)
                    # Check map first, then direct
                    has_evolved = len(self.evolved_dags_map.get(extracted_id, [])) > 0
                    if not has_evolved:
                        has_evolved = any(
                            d
                            for d in self.all_dags
                            if d.generation_method == "evolved"
                            and d.parent_dag_id
                            and str(d.parent_dag_id) == extracted_id
                        )
                    if has_evolved:
                        evolvable_count += 1

                label = f"{dag.problem_statement[:50]}... ({evolvable_count} evolvable paths / {len(extracted_paths)} total) - {dag.created_at.strftime('%Y-%m-%d %H:%M')}"
                options[label] = str(dag.id)

            self.original_selector.name = "Select Original (Build) DAG"
            self.original_selector.options = options

            if options and not self.original_selector.value:
                self.original_selector.value = list(options.values())[0]

    async def _on_mode_changed(self, event):
        """Handle mode change between single, comparison, and evolution journey."""
        mode = event.new

        # Update selector states based on mode
        self.extracted_selector.disabled = mode != "Evolution Journey"
        # Evolved selector is only enabled in Comparison mode initially
        # In Evolution Journey, it's enabled after selecting an extracted path
        self.evolved_selector.disabled = mode != "Comparison"

        # Show/hide LLM analysis button based on mode
        self.analyze_diff_button.visible = mode == "Comparison"
        self.llm_analysis_pane.visible = False
        self.llm_analysis_pane.object = ""

        # Clear selections when switching modes
        if mode == "Single":
            self.extracted_dag = None
            self.extracted_visualizer = None
            self.evolved_dag = None
            self.evolved_visualizer = None
            self.extracted_selector.value = None
            self.evolved_selector.value = None
            self.comparison_pane.object = ""
        elif mode == "Comparison":
            self.extracted_dag = None
            self.extracted_visualizer = None
            self.extracted_selector.value = None
        elif mode == "Evolution Journey":
            # Clear evolved selections when entering Evolution Journey
            self.evolved_dag = None
            self.evolved_visualizer = None
            self.evolved_selector.value = None
            self.evolved_selector.options = {}
            self.evolved_selector.disabled = True

        # Update selectors for new mode
        await self._update_selectors()

        # Update visualization
        await self._update_visualization(None)

    async def _on_original_selected(self, event):
        """Handle original DAG selection."""
        if not event.new:
            return

        try:
            # Load the selected DAG
            dag_id = UUID(event.new)
            self.original_dag = await dag_repository.load_dag(dag_id)
            self.original_visualizer = DAGVisualizer(self.original_dag)

            # Update info
            dag_info = await dag_repository.get_dag_info(dag_id)

            # In comparison mode, show all DAGs for second selector
            if self.mode_selector.value == "Comparison":
                # Show all DAGs except the one selected as first
                comparison_options = {}
                for dag in self.all_dags:
                    if str(dag.id) != str(dag_id):  # Don't show the same DAG
                        label = f"{dag.problem_statement[:50]}... ({dag.generation_method}) - {dag.created_at.strftime('%Y-%m-%d %H:%M')}"
                        comparison_options[label] = str(dag.id)

                self.evolved_selector.name = "Select Second DAG"
                self.evolved_selector.options = comparison_options
                self.evolved_selector.disabled = len(comparison_options) == 0
                logger.info(f"Set comparison selector with {len(comparison_options)} options")

                if comparison_options and not self.evolved_selector.value:
                    self.evolved_selector.value = list(comparison_options.values())[0]
                elif not comparison_options:
                    self.evolved_selector.value = None
                    self.evolved_dag = None
                    self.evolved_visualizer = None

            # In Evolution Journey mode, update extracted path selector
            elif self.mode_selector.value == "Evolution Journey":
                extracted_options = {}
                extracted_paths = self.extracted_dags_map.get(str(dag_id), [])
                logger.info(f"Looking for extracted paths for parent {dag_id}: found {len(extracted_paths)}")

                # Only show extracted paths that have evolved children
                paths_shown = 0
                for i, extracted_dag in enumerate(extracted_paths):
                    # Check if this extracted path has any evolved children
                    extracted_id = str(extracted_dag.id)

                    # First check the map, then do a direct check
                    has_evolved = len(self.evolved_dags_map.get(extracted_id, [])) > 0
                    if not has_evolved:
                        # Double-check with direct search
                        has_evolved = any(
                            d
                            for d in self.all_dags
                            if d.generation_method == "evolved"
                            and d.parent_dag_id
                            and str(d.parent_dag_id) == extracted_id
                        )

                    if has_evolved:
                        paths_shown += 1
                        # Try to get path info from metadata
                        path_idx = extracted_dag.metadata.get("path_index", i) if extracted_dag.metadata else i
                        label = f"Path {path_idx + 1}: {extracted_dag.created_at.strftime('%Y-%m-%d %H:%M')} - ID: {str(extracted_dag.id)[:8]}..."
                        extracted_options[label] = str(extracted_dag.id)

                logger.info(
                    f"Showing {paths_shown} extracted paths with evolved children out of {len(extracted_paths)} total"
                )

                # Clear existing state before updating
                self.extracted_selector.value = None
                self.extracted_selector.options = extracted_options
                self.extracted_selector.disabled = len(extracted_options) == 0

                # Force UI update
                self.extracted_selector.param.trigger("options")
                self.extracted_selector.param.trigger("disabled")

                # Don't auto-select in Evolution Journey - let user choose
                if not extracted_options:
                    self.extracted_dag = None
                    self.extracted_visualizer = None

                # Always clear evolved selector in Evolution Journey until extracted path is chosen
                self.evolved_selector.value = None
                self.evolved_selector.options = {}
                self.evolved_selector.disabled = True

            await self._update_info_panes(dag_info, is_original=True)
            await self._update_visualization(None)

        except RecursionError as e:
            logger.error(f"Recursion error in _on_original_selected: {e}")
            if pn.state.notifications:
                pn.state.notifications.error(
                    "Recursion depth exceeded - DAG structure may be too complex", duration=5000
                )
        except Exception as e:
            logger.error(f"Failed to load original DAG: {e}")
            if pn.state.notifications:
                pn.state.notifications.error(f"Failed to load DAG: {str(e)}", duration=5000)

    async def _on_extracted_selected(self, event):
        """Handle extracted path selection."""
        if not event.new or self.mode_selector.value != "Evolution Journey":
            return

        try:
            # Load the extracted path DAG
            dag_id = UUID(event.new)
            self.extracted_dag = await dag_repository.load_dag(dag_id)
            self.extracted_visualizer = DAGVisualizer(self.extracted_dag)

            # Update evolved selector for this extracted path
            evolved_options = {}
            evolved_dags = self.evolved_dags_map.get(str(dag_id), [])
            logger.info(f"Looking for evolved DAGs for extracted path {dag_id}: found {len(evolved_dags)}")

            # Debug: check if evolved DAGs are mapped differently
            if len(evolved_dags) == 0:
                # Check if any evolved DAGs have this as parent
                direct_evolved = [
                    d
                    for d in self.all_dags
                    if d.generation_method == "evolved" and d.parent_dag_id and str(d.parent_dag_id) == str(dag_id)
                ]
                if direct_evolved:
                    logger.warning(
                        f"Found {len(direct_evolved)} evolved DAGs with direct parent check, but evolved_dags_map was empty!"
                    )
                    evolved_dags = direct_evolved

            for evolved_dag in evolved_dags:
                label = f"Evolved from Path: {evolved_dag.created_at.strftime('%Y-%m-%d %H:%M')} - ID: {str(evolved_dag.id)[:8]}..."
                evolved_options[label] = str(evolved_dag.id)

            self.evolved_selector.options = evolved_options
            self.evolved_selector.disabled = len(evolved_options) == 0

            # Update evolved row visibility when selector is enabled/disabled
            if self.evolved_row:
                self.evolved_row.visible = not self.evolved_selector.disabled

            if evolved_options:
                self.evolved_selector.value = list(evolved_options.values())[0]
            else:
                self.evolved_selector.value = None
                self.evolved_dag = None
                self.evolved_visualizer = None

            # Update visualization
            await self._update_visualization(None)

        except Exception as e:
            logger.error(f"Failed to load extracted path: {e}")
            if pn.state.notifications:
                pn.state.notifications.error(f"Failed to load extracted path: {str(e)}", duration=5000)

    async def _on_evolved_selected(self, event):
        """Handle evolved DAG selection."""
        if not event.new or self.mode_selector.value == "Single":
            return

        try:
            # Load the evolved DAG
            dag_id = UUID(event.new)
            self.evolved_dag = await dag_repository.load_dag(dag_id)
            self.evolved_visualizer = DAGVisualizer(self.evolved_dag)

            # Update comparison statistics
            await self._update_comparison_stats()

            # Update visualization
            await self._update_visualization(None)

        except Exception as e:
            logger.error(f"Failed to load evolved DAG: {e}")
            if pn.state.notifications:
                pn.state.notifications.error(f"Failed to load evolved DAG: {str(e)}", duration=5000)

    async def _update_visualization(self, event):
        """Update the visualization based on current settings."""
        if not self.original_visualizer:
            return

        try:
            # Get filter nodes based on search
            filter_nodes = None
            if self.search_input.value:
                search_lower = self.search_input.value.lower()

                # Search in original DAG
                if self.original_dag:
                    filter_nodes = [
                        node_id
                        for node_id, node in self.original_dag.all_nodes.items()
                        if search_lower in node.title.lower() or search_lower in node.description.lower()
                    ]

            # Handle different visualization modes
            if self.mode_selector.value == "Single":
                # Single DAG visualization
                highlight_paths = None
                if self.highlight_paths_toggle.value:
                    highlight_paths = self.original_visualizer.get_critical_paths(top_k=3)

                    # Update paths info
                    paths_md = "### Critical Paths\n\n"
                    for i, path in enumerate(highlight_paths, 1):
                        paths_md += f"**Path {i}:**\n"
                        for node_id in path:
                            node = self.original_dag.all_nodes[node_id]
                            paths_md += f"- {node.title}\n"
                        paths_md += "\n"
                    self.paths_pane.object = paths_md
                else:
                    self.paths_pane.object = "Enable 'Highlight Paths' to see critical paths"

                # Create single figure
                fig = self.original_visualizer.create_plotly_figure(
                    show_labels=self.show_labels_toggle.value,
                    highlight_paths=highlight_paths,
                    filter_nodes=filter_nodes,
                )

                self.plot_pane.object = fig

            elif self.mode_selector.value == "Comparison" and self.evolved_visualizer:
                # Two-way comparison visualization
                # Create figures for both DAGs
                original_fig = self.original_visualizer.create_plotly_figure(
                    show_labels=self.show_labels_toggle.value, filter_nodes=filter_nodes
                )

                evolved_fig = self.evolved_visualizer.create_plotly_figure(
                    show_labels=self.show_labels_toggle.value, filter_nodes=filter_nodes
                )

                # Create subplot figure with dynamic titles based on DAG types
                first_type = (
                    self.original_dag.generation_method.title()
                    if hasattr(self.original_dag, "generation_method")
                    else "First"
                )
                second_type = (
                    self.evolved_dag.generation_method.title()
                    if hasattr(self.evolved_dag, "generation_method")
                    else "Second"
                )

                fig = make_subplots(
                    rows=1, cols=2, subplot_titles=(f"{first_type} DAG", f"{second_type} DAG"), horizontal_spacing=0.05
                )

                # Add traces from both figures
                for trace in original_fig.data:
                    fig.add_trace(trace, row=1, col=1)

                for trace in evolved_fig.data:
                    fig.add_trace(trace, row=1, col=2)

                # Update layout
                # Get the max number of layers from both DAGs for dynamic height
                original_layers = len(set(node.layer for node in self.original_dag.all_nodes.values()))
                evolved_layers = len(set(node.layer for node in self.evolved_dag.all_nodes.values()))
                max_layers = max(original_layers, evolved_layers)
                fig_height = max(800, max_layers * 150)  # At least 150px per layer

                fig.update_layout(
                    title_text="DAG Comparison", showlegend=False, height=fig_height, width=1600, hovermode="closest"
                )

                # Update axes to match original settings
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

                self.plot_pane.object = fig

            elif (
                self.mode_selector.value == "Evolution Journey"
                and self.extracted_visualizer
                and self.evolved_visualizer
            ):
                # Three-way Evolution Journey visualization
                # Create figures for all three DAGs
                original_fig = self.original_visualizer.create_plotly_figure(
                    show_labels=self.show_labels_toggle.value, filter_nodes=filter_nodes
                )

                extracted_fig = self.extracted_visualizer.create_plotly_figure(
                    show_labels=self.show_labels_toggle.value, filter_nodes=filter_nodes
                )

                evolved_fig = self.evolved_visualizer.create_plotly_figure(
                    show_labels=self.show_labels_toggle.value, filter_nodes=filter_nodes
                )

                # Create subplot figure with 3 columns
                fig = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=("Original DAG", "Extracted Path", "Evolved Path"),
                    horizontal_spacing=0.03,
                )

                # Add traces from all three figures
                for trace in original_fig.data:
                    fig.add_trace(trace, row=1, col=1)

                for trace in extracted_fig.data:
                    fig.add_trace(trace, row=1, col=2)

                for trace in evolved_fig.data:
                    fig.add_trace(trace, row=1, col=3)

                # Update layout
                # Get the max number of layers from all DAGs for dynamic height
                original_layers = len(set(node.layer for node in self.original_dag.all_nodes.values()))
                extracted_layers = len(set(node.layer for node in self.extracted_dag.all_nodes.values()))
                evolved_layers = len(set(node.layer for node in self.evolved_dag.all_nodes.values()))
                max_layers = max(original_layers, extracted_layers, evolved_layers)
                fig_height = max(800, max_layers * 150)  # At least 150px per layer

                fig.update_layout(
                    title_text="Complete Evolution Journey",
                    showlegend=False,
                    height=fig_height,
                    width=2400,  # Wider for 3 columns
                    hovermode="closest",
                )

                # Update axes to match original settings
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

                self.plot_pane.object = fig
            else:
                # Waiting for all selections to be made
                self.plot_pane.object = go.Figure().add_annotation(
                    text="Please select all required DAGs for visualization",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20),
                )

            # Update paths based on mode and highlight toggle
            if self.highlight_paths_toggle.value:
                if self.mode_selector.value == "Comparison" and self.original_visualizer and self.evolved_visualizer:
                    original_paths = self.original_visualizer.get_critical_paths(top_k=2)
                    evolved_paths = self.evolved_visualizer.get_critical_paths(top_k=2)

                    paths_md = "### Critical Paths Comparison\n\n"

                    paths_md += "**Original DAG:**\n"
                    for i, path in enumerate(original_paths, 1):
                        paths_md += f"\nPath {i}:\n"
                        for node_id in path:
                            if node_id in self.original_dag.all_nodes:
                                node = self.original_dag.all_nodes[node_id]
                                paths_md += f"- {node.title}\n"

                    paths_md += "\n**Evolved DAG:**\n"
                    for i, path in enumerate(evolved_paths, 1):
                        paths_md += f"\nPath {i}:\n"
                        for node_id in path:
                            if node_id in self.evolved_dag.all_nodes:
                                node = self.evolved_dag.all_nodes[node_id]
                                paths_md += f"- {node.title}\n"

                    self.paths_pane.object = paths_md

                elif self.mode_selector.value == "Evolution Journey" and self.extracted_visualizer:
                    # For Evolution Journey, show the single extracted path
                    paths_md = "### Evolution Journey Path\n\n"

                    # The extracted path IS the critical path we're following
                    if self.extracted_dag:
                        extracted_nodes = list(self.extracted_dag.all_nodes.values())
                        extracted_nodes.sort(key=lambda n: n.layer)

                        paths_md += "**Complete Path:**\n"
                        for node in extracted_nodes:
                            paths_md += f"- {node.title}\n"

                        # Show evolution metrics if available
                        if self.evolved_dag and self.evolved_dag.metadata:
                            evolution_summary = self.evolved_dag.metadata.get("evolution_metrics", {}).get(
                                "evolution_summary", {}
                            )
                            if evolution_summary:
                                paths_md += f"\n**Evolution Performance:**\n"
                                paths_md += (
                                    f"- Best Fitness: {evolution_summary.get('best_fitness', 'N/A'):.3f}\n"
                                    if isinstance(evolution_summary.get("best_fitness"), (int, float))
                                    else "- Best Fitness: N/A\n"
                                )
                                paths_md += (
                                    f"- Avg Fitness: {evolution_summary.get('avg_fitness', 'N/A'):.3f}\n"
                                    if isinstance(evolution_summary.get("avg_fitness"), (int, float))
                                    else "- Avg Fitness: N/A\n"
                                )
                                paths_md += f"- Generations: {evolution_summary.get('generations_run', 'N/A')}\n"

                    self.paths_pane.object = paths_md
                else:
                    self.paths_pane.object = "Enable 'Highlight Paths' to see paths info"
            else:
                self.paths_pane.object = "Enable 'Highlight Paths' to see paths info"

        except Exception as e:
            logger.error(f"Failed to update visualization: {e}")
            if pn.state.notifications:
                pn.state.notifications.error(f"Failed to update visualization: {str(e)}", duration=5000)

    async def _on_search(self, event):
        """Handle search input."""
        await self._update_visualization(None)

    async def _update_info_panes(self, dag_info: Dict[str, Any], is_original: bool = True):
        """Update the information panes."""
        # DAG Info
        info_md = f"""### {"First" if self.mode_selector.value == "Comparison" else ""} DAG Information

**ID:** `{dag_info["id"]}`
**Problem:** {dag_info["problem_statement"]}
**Method:** {dag_info["generation_method"]}
**Created:** {dag_info["created_at"]}
**Nodes:** {dag_info["node_count"]} | **Edges:** {dag_info["edge_count"]} | **Depth:** {dag_info["max_layers"]}
"""

        if dag_info["parent"]:
            info_md += f"\n**Parent:** {dag_info['parent']['problem_statement'][:50]}..."

        if dag_info["children"]:
            info_md += f"\n**Children:** {len(dag_info['children'])} evolved DAGs"

        self.dag_info_pane.object = info_md

        # Statistics (only update if this is for the original DAG)
        if is_original and self.original_visualizer:
            stats = self.original_visualizer.get_dag_stats()
            stats_md = f"""### DAG Statistics

**Node Distribution:**
- Decision Nodes: {stats["decision_nodes"]}
- Option Nodes: {stats["option_nodes"]}

**Decision Types:**
"""
            for dt, count in stats["decision_type_distribution"].items():
                stats_md += f"- {dt}: {count}\n"

            stats_md += f"\n**Average Confidence:** {stats['avg_confidence']:.2f}"

            self.stats_pane.object = stats_md

    async def _update_comparison_stats(self):
        """Update comparison statistics between DAGs based on mode."""
        try:
            if self.mode_selector.value == "Comparison" and self.original_visualizer and self.evolved_visualizer:
                # Get stats for both DAGs
                original_stats = self.original_visualizer.get_dag_stats()
                evolved_stats = self.evolved_visualizer.get_dag_stats()

                # Calculate differences
                node_diff = evolved_stats["total_nodes"] - original_stats["total_nodes"]
                edge_diff = evolved_stats["total_edges"] - original_stats["total_edges"]
                conf_diff = evolved_stats["avg_confidence"] - original_stats["avg_confidence"]

                # Get DAG types for labels
                first_type = (
                    self.original_dag.generation_method.title()
                    if hasattr(self.original_dag, "generation_method")
                    else "First"
                )
                second_type = (
                    self.evolved_dag.generation_method.title()
                    if hasattr(self.evolved_dag, "generation_method")
                    else "Second"
                )

                # Build comparison markdown
                comparison_md = f"""### DAG Comparison

**{first_type} → {second_type} Changes:**
- Nodes: {original_stats["total_nodes"]} → {evolved_stats["total_nodes"]} ({"+" if node_diff >= 0 else ""}{node_diff})
- Edges: {original_stats["total_edges"]} → {evolved_stats["total_edges"]} ({"+" if edge_diff >= 0 else ""}{edge_diff})
- Avg Confidence: {original_stats["avg_confidence"]:.2f} → {evolved_stats["avg_confidence"]:.2f} ({"+" if conf_diff >= 0 else ""}{conf_diff:.2f})

**Node Type Distribution:**
- Decision Nodes: {original_stats["decision_nodes"]} → {evolved_stats["decision_nodes"]}
- Option Nodes: {original_stats["option_nodes"]} → {evolved_stats["option_nodes"]}
"""

                # Add decision type comparison
                original_dt = original_stats["decision_type_distribution"]
                evolved_dt = evolved_stats["decision_type_distribution"]

                all_decision_types = set(original_dt.keys()) | set(evolved_dt.keys())
                if all_decision_types:
                    comparison_md += "\n**Decision Type Distribution:**\n"
                    for dt in sorted(all_decision_types):
                        orig_count = original_dt.get(dt, 0)
                        evol_count = evolved_dt.get(dt, 0)
                        diff = evol_count - orig_count
                        comparison_md += f"- {dt}: {orig_count} → {evol_count}"
                        if diff != 0:
                            comparison_md += f" ({'+' if diff >= 0 else ''}{diff})"
                        comparison_md += "\n"

                self.comparison_pane.object = comparison_md

            elif (
                self.mode_selector.value == "Evolution Journey"
                and self.original_visualizer
                and self.extracted_visualizer
                and self.evolved_visualizer
            ):
                # Get stats for all three DAGs
                original_stats = self.original_visualizer.get_dag_stats()
                extracted_stats = self.extracted_visualizer.get_dag_stats()
                evolved_stats = self.evolved_visualizer.get_dag_stats()

                # Build journey comparison markdown
                comparison_md = f"""### Complete Evolution Journey

**Original DAG → Extracted Path → Evolved Path**

**Node Count Evolution:**
- Original: {original_stats["total_nodes"]} nodes
- Extracted: {extracted_stats["total_nodes"]} nodes ({extracted_stats["total_nodes"] - original_stats["total_nodes"]:+d})
- Evolved: {evolved_stats["total_nodes"]} nodes ({evolved_stats["total_nodes"] - extracted_stats["total_nodes"]:+d})

**Confidence Evolution:**
- Original: {original_stats["avg_confidence"]:.2f}
- Extracted: {extracted_stats["avg_confidence"]:.2f} ({extracted_stats["avg_confidence"] - original_stats["avg_confidence"]:+.2f})
- Evolved: {evolved_stats["avg_confidence"]:.2f} ({evolved_stats["avg_confidence"] - extracted_stats["avg_confidence"]:+.2f})

**Path Extraction Impact:**
- Reduced from {original_stats["total_nodes"]} to {extracted_stats["total_nodes"]} nodes
- Focus on single strategic path
- Preserved confidence: {extracted_stats["avg_confidence"]:.2f}

**Evolution Impact:**
- Path complexity: {extracted_stats["total_nodes"]} → {evolved_stats["total_nodes"]} nodes
- Confidence improvement: {evolved_stats["avg_confidence"] - extracted_stats["avg_confidence"]:+.2f}
"""

                # Add evolution performance if available
                if self.evolved_dag and self.evolved_dag.metadata:
                    evolution_summary = self.evolved_dag.metadata.get("evolution_metrics", {}).get(
                        "evolution_summary", {}
                    )
                    if evolution_summary:
                        comparison_md += f"\n**Evolution Performance:**\n"
                        comparison_md += (
                            f"- Total Variants Generated: {evolution_summary.get('total_variants', 'N/A')}\n"
                        )
                        comparison_md += (
                            f"- Best Fitness Achieved: {evolution_summary.get('best_fitness', 'N/A'):.3f}\n"
                            if isinstance(evolution_summary.get("best_fitness"), (int, float))
                            else "- Best Fitness: N/A\n"
                        )
                        comparison_md += f"- Generations Run: {evolution_summary.get('generations_run', 'N/A')}\n"

                self.comparison_pane.object = comparison_md

        except Exception as e:
            logger.error(f"Failed to update comparison stats: {e}")
            self.comparison_pane.object = "Error calculating comparison statistics"

    async def _analyze_differences(self, event):
        """Use LLM to analyze differences between two DAGs."""
        if not self.original_dag or not self.evolved_dag:
            if pn.state.notifications:
                pn.state.notifications.warning("Please select two DAGs to compare", duration=3000)
            return

        try:
            self.analyze_diff_button.loading = True
            self.llm_analysis_pane.object = "🤖 Analyzing differences..."
            self.llm_analysis_pane.visible = True

            # Import the analysis module
            from ..analysis.dag_analyzer import DAGAnalyzer

            analyzer = DAGAnalyzer()

            # Get comparison analysis
            analysis = await analyzer.compare_dags(
                self.original_dag,
                self.evolved_dag,
                context={
                    "original_type": self.original_dag.generation_method
                    if hasattr(self.original_dag, "generation_method")
                    else "original",
                    "evolved_type": self.evolved_dag.generation_method
                    if hasattr(self.evolved_dag, "generation_method")
                    else "evolved",
                },
            )

            # Format the analysis for display
            analysis_md = f"""### 🤖 AI Analysis

{analysis}

*Analysis generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

            self.llm_analysis_pane.object = analysis_md

        except Exception as e:
            logger.error(f"Failed to analyze differences: {e}")
            self.llm_analysis_pane.object = f"❌ Error analyzing differences: {str(e)}"
            if pn.state.notifications:
                pn.state.notifications.error(f"Analysis failed: {str(e)}", duration=5000)
        finally:
            self.analyze_diff_button.loading = False

    def get_layout(self):
        """Get the application layout."""
        # Control panel - adjust based on mode
        controls = [
            "## DAG Visualization",
            pn.Row(self.mode_selector, self.refresh_button),
            pn.Row(self.original_selector),
        ]

        # Add extracted selector row (visible in Evolution Journey mode)
        extracted_row = pn.Row(self.extracted_selector)
        extracted_row.visible = self.mode_selector.value == "Evolution Journey"

        # Add evolved selector row (visible in Comparison mode, and Evolution Journey after path selection)
        evolved_row = pn.Row(self.evolved_selector)
        evolved_row.visible = self.mode_selector.value == "Comparison"
        self.evolved_row = evolved_row  # Store reference for later updates

        # Watch mode changes to update visibility
        def update_selector_visibility(event):
            extracted_row.visible = event.new == "Evolution Journey"
            # Only show evolved row in Comparison mode initially
            # In Evolution Journey, it becomes visible when evolved selector is enabled
            evolved_row.visible = event.new == "Comparison" or (
                event.new == "Evolution Journey" and not self.evolved_selector.disabled
            )

        self.mode_selector.param.watch(update_selector_visibility, "value")

        controls.append(extracted_row)
        controls.append(evolved_row)

        # Add LLM analysis button (visible in Comparison mode)
        analyze_row = pn.Row(self.analyze_diff_button)
        analyze_row.visible = self.mode_selector.value == "Comparison"

        def update_analyze_visibility(event):
            analyze_row.visible = event.new == "Comparison"

        self.mode_selector.param.watch(update_analyze_visibility, "value")

        controls.append(analyze_row)

        # Add other controls
        controls.append(pn.Row(self.show_labels_toggle, self.highlight_paths_toggle, self.search_input))

        control_panel = pn.Column(*controls, width=1000)

        # Main visualization
        main_content = pn.Column(self.plot_pane, sizing_mode="stretch_both")

        # Side panel with info - adjust based on mode
        side_panels = [self.dag_info_pane, pn.layout.Divider(), self.stats_pane]

        # Add comparison pane (visible in comparison and journey modes)
        comparison_section = pn.Column(pn.layout.Divider(), self.comparison_pane)
        comparison_section.visible = self.mode_selector.value in ["Comparison", "Evolution Journey"]

        # Watch mode changes to update visibility
        def update_comparison_visibility(event):
            comparison_section.visible = event.new in ["Comparison", "Evolution Journey"]

        self.mode_selector.param.watch(update_comparison_visibility, "value")

        side_panels.append(comparison_section)

        # Add LLM analysis pane (visible when analysis is performed)
        side_panels.extend([pn.layout.Divider(), self.llm_analysis_pane])

        side_panels.extend([pn.layout.Divider(), self.paths_pane])

        side_panel = pn.Column(*side_panels, width=350, height=800, scroll=True)

        # Combine layout
        return pn.template.MaterialTemplate(
            title="Decision DAG Visualizer",
            sidebar=[side_panel],
            main=[control_panel, main_content],
            header_background="#4A90E2",
        )


def create_app() -> pn.template.MaterialTemplate:
    """Create and return the Panel application."""
    app = DAGVisualizationApp()
    return app.get_layout()


async def run_server(
    port: int = 5006, address: str = "localhost", show: bool = True, title: str = "Decision DAG Visualizer"
):
    """Run the visualization server."""
    # Create the app
    app = create_app()

    # Configure server
    pn.config.port = port
    pn.config.address = address

    # Serve the app
    if show:
        app.show(title=title)
    else:
        app.servable(title=title)

    logger.info(f"Visualization server running at http://{address}:{port}")

    # Keep server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down visualization server")
        await close_db()
