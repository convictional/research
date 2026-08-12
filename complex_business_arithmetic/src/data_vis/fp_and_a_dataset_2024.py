import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np

from ..settings import settings


html_path = settings.output_path / "attachment_data_plots" / "fp_and_a_dataset_2024.html"
png_path = settings.output_path / "attachment_data_plots" / "fp_and_a_dataset_2024.png"


def plot_fp_and_a_dataset_2024():
    """
    Plot data for the FP&A Dataset 2024 attachment
    Creates both a Plotly HTML version and a Matplotlib PNG version
    """
    print("Plotting FP&A Dataset 2024...")

    # Load the data
    data = pd.read_csv(settings.input_path / "attachment_files" / "fp&a_dataset_2024.csv")

    # Convert Date column to datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Get unique departments for color mapping
    departments = data["Department"].unique()

    # Define a colorblind-friendly palette
    colors = [
        "#0173B2",
        "#DE8F05",
        "#029E73",
        "#D55E00",
        "#CC78BC",
        "#CA9161",
        "#FBAFE4",
        "#949494",
        "#ECE133",
        "#56B4E9",
    ]
    dept_colors = dict(zip(departments, colors[: len(departments)]))

    # Define metrics to plot (only using raw metrics from CSV)
    # Skip 'Date' and 'Department' as they're not metrics
    metrics = [col for col in data.columns if col not in ["Date", "Department"]]

    # Fixed 3x3 grid layout
    n_cols = 3
    n_rows = 3

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)

    # PART 1: Create Plotly version (HTML output)
    plotly_fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[metric.replace("_", " ") for metric in metrics],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    # Loop through metrics and create each plot
    for i, metric in enumerate(metrics):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Plot each department as a separate line
        for dept in departments:
            dept_data = data[data["Department"] == dept]
            plotly_fig.add_trace(
                go.Scatter(
                    x=dept_data["Date"],
                    y=dept_data[metric],
                    mode="lines+markers",
                    name=dept,
                    line=dict(color=dept_colors[dept]),
                    legendgroup=dept,
                    showlegend=True,  # Show legend on all subplots
                ),
                row=row,
                col=col,
            )

    # Update layout
    plotly_fig.update_layout(
        title={
            "text": "FP&A Performance Dashboard - 2024",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=16),
        },
        height=n_rows * 300,  # Adjust height based on number of rows
        width=900,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        template="plotly_white",
    )

    # Update x-axes to show month abbreviations
    plotly_fig.update_xaxes(
        tickformat="%b",  # Month abbreviation
        tickangle=45,
        gridcolor="rgba(211,211,211,0.3)",
    )

    # Update y-axes
    plotly_fig.update_yaxes(gridcolor="rgba(211,211,211,0.3)")

    # Save interactive HTML version
    plotly_fig.write_html(html_path)
    print(f"Interactive visualization saved to: {html_path}")

    # PART 2: Create Matplotlib version (PNG output)
    # This is essentially your original code but with the color scheme from above
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Make sure axes is always a 2D array, even with a single row
    if n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()  # Flatten to make indexing easier

    # Loop through metrics and create each plot
    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break

        ax = axes[i]

        # Plot each department as a separate line
        for dept in departments:
            dept_data = data[data["Department"] == dept]
            ax.plot(dept_data["Date"], dept_data[metric], "o-", label=dept, color=dept_colors[dept])

        # Format the plot
        ax.set_title(metric.replace("_", " "), fontsize=12)
        ax.set_ylabel(metric)
        # Format x-axis to show every month
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Show legend on all subplots
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.suptitle("FP&A Performance Dashboard - 2024", fontsize=16, y=1.02)

    # Save PNG
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"PNG visualization saved to: {png_path}")

    # Close the figure to free memory
    plt.close(fig)
