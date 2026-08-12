import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.cm import inferno
import numpy as np


# Get evenly spaced colors from inferno colormap
def get_inferno_colors(n_colors: int) -> list[tuple[float, ...]]:
    """Get n evenly spaced colors from the inferno colormap."""
    return [inferno(i) for i in np.linspace(0.2, 0.8, n_colors)]


# Update color palette with more distinct inferno colors
inferno_palette = get_inferno_colors(7)  # Get 7 colors with wider spacing
COLORS = {
    "primary": {
        "question": inferno(0.15),  # Very dark purple
        "thread": inferno(0.3),  # Purple
        "meeting": inferno(0.5),  # Orange-red
        "decision_process": inferno(0.7),  # Yellow-orange
        "user": inferno(0.85),  # Light yellow
        "collaborator": inferno(0.95),  # Very light yellow
        "organization": inferno(0.4),  # Red-orange
    },
    "accents": {
        "blue": inferno(0.2),  # Dark purple from inferno
        "red": inferno(0.8),  # Yellow from inferno
    },
    "background": "#F5F5F5",  # Keep light grey background
    "grid": "#E0E0E0",  # Keep light grey grid
}

PLOT_STYLE = {
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": COLORS["background"],
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.grid.axis": "both",
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.3,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "legend.framealpha": 0.9,
    "legend.facecolor": "white",
    "legend.edgecolor": COLORS["grid"],
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def setup_plotting_style():
    """Configure matplotlib with our custom style."""
    plt.style.use("default")  # Reset to default first
    plt.rcParams.update(PLOT_STYLE)


def create_figure(figsize=(12, 6)):
    """Create a figure with our custom style."""
    setup_plotting_style()
    return plt.subplots(figsize=figsize)


def create_subplot_figure(nrows: int, ncols: int, figsize=(12, 6)):
    """Create a figure with multiple subplots and our custom style."""
    setup_plotting_style()
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)


def create_dual_axis_figure(figsize=(12, 6)):
    """Create a figure with two y-axes and our custom style."""
    setup_plotting_style()
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    return fig, ax1, ax2


def create_row_figure(ncols: int, figsize=(20, 6)):
    """Create a figure with multiple subplots in a row and our custom style."""
    setup_plotting_style()
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    return fig, axes


def style_axis(ax, title=None, xlabel=None, ylabel=None, rotate_labels=45):
    """Apply consistent styling to an axis."""
    if title:
        ax.set_title(title, pad=15)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.tick_params(axis="x", rotation=rotate_labels)

    # Add subtle border
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(0.5)


def style_stacked_bar(ax, title=None, xlabel=None, ylabel=None):
    """Apply specific styling for stacked bar charts."""
    style_axis(ax, title, xlabel, ylabel)
    ax.grid(axis="y")  # Only show horizontal grid for stacked bars
    ax.grid(axis="x", alpha=0)  # Hide vertical grid


def get_color_with_alpha(color_key, alpha=0.7):
    """Get a color from our palette with specified alpha."""
    rgba = to_rgba(COLORS["primary"][color_key])
    return (*rgba[:3], alpha)


def add_branded_legend(ax, title=None, loc="center left", bbox_to_anchor=(1.05, 0.5), lines=None, labels=None):
    """Add a consistently styled legend.

    Args:
        ax: The axis to add the legend to
        title: Optional legend title
        loc: Legend location
        bbox_to_anchor: Legend box anchor point
        lines: Optional list of Line2D objects for custom legend entries
        labels: Optional list of labels for custom legend entries
    """
    if lines is not None and labels is not None:
        legend = ax.legend(
            lines,
            labels,
            title=title,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
    else:
        legend = ax.legend(
            title=title,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    return legend
