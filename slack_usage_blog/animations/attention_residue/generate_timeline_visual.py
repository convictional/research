"""
Recovery Window Impossibility - Timeline Visualization

Shows how attention residue accumulates when interruptions arrive too quickly,
and how it drains during longer gaps. Uses stacked area curves to demonstrate
that incomplete recovery causes cognitive "debt" to build up.

Key insight: When a new task interrupts before previous task decays to 0,
the new attention curve layers ON TOP of residual attention, creating
visible accumulation.

Based on Gloria Mark's research: 23 min 15 sec to fully refocus after interruption.

Author: Adam McCabe
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------
# Configuration
# ----------------------------

# Recovery parameters (from Gloria Mark's research)
FULL_RECOVERY_TIME = 23.25  # minutes (23 min 15 sec)
TAU = FULL_RECOVERY_TIME / 5  # time constant ≈ 4.65 min for steeper exponential decay
# At t=23: exp(-23/4.65) = exp(-4.95) ≈ 0.007 (0.7% residual)


# Interruption schedule (time in minutes, task name, color)
@dataclass
class Interruption:
    time: float  # minutes into workday
    task: str
    color: str


# Create a schedule with varying gaps to show both accumulation and recovery
INTERRUPTIONS: List[Interruption] = [
    # Morning - rapid fire interruptions (accumulation phase)
    Interruption(5.0, "Check Slack", "#4C78A8"),
    Interruption(8.0, "Reply to email", "#E45756"),  # 3 min gap - too short
    Interruption(12.0, "Standup meeting", "#59A14F"),  # 4 min gap - too short
    Interruption(15.0, "Code review ping", "#B279A2"),  # 3 min gap - too short
    Interruption(19.0, "Quick question", "#EDC948"),  # 4 min gap - too short
    # Mid-morning - short recovery window (partial drain)
    Interruption(35.0, "Team sync", "#4C78A8"),  # 16 min gap - partial recovery
    Interruption(40.0, "Urgent Slack", "#E45756"),  # 5 min gap - too short
    # Longer gap - shows what recovery could look like
    Interruption(68.0, "Design review", "#59A14F"),  # 28 min gap - FULL recovery possible
    # Afternoon - more accumulation
    Interruption(73.0, "Email check", "#B279A2"),  # 5 min gap - too short
    Interruption(77.0, "Slack ping", "#EDC948"),  # 4 min gap - too short
    Interruption(82.0, "1-on-1 call", "#4C78A8"),  # 5 min gap - too short
]

WORK_DAY_DURATION = 100.0  # minutes to visualize
ANIMATION_DURATION = 8.0  # seconds for the animation to play
FPS = 20
TIMELINE_SAMPLES = 2000  # sample points for smooth curves

# Visual parameters
FIGSIZE = (14, 7)
DPI = 120
BG_COLOR = "#FAFAFA"
GRID_COLOR = "#E0E0E0"
TEXT_COLOR = "#333333"
ATTENTION_LIMIT_COLOR = "#FF6B6B"

# Y-axis: attention residue level (0 = full focus, higher = more cognitive load)
Y_MAX = 4.0  # max accumulated attention residue to show


# ----------------------------
# Attention Decay Model
# ----------------------------


def attention_residue(t_since_interruption: float) -> float:
    """
    Calculate attention residue at time t after an interruption.

    Based on Gloria Mark: 23 min 15 sec to fully refocus.
    Uses exponential approach: residue starts at 1.0 and decays toward 0.

    Args:
        t_since_interruption: Time in minutes since the interruption occurred

    Returns:
        Attention residue level (1.0 = just interrupted, 0.0 = fully recovered)
    """
    if t_since_interruption < 0:
        return 0.0

    # Exponential decay: starts at 1.0, approaches 0
    return math.exp(-t_since_interruption / TAU)


def calculate_total_residue(current_time: float, interruptions: List[Interruption]) -> dict:
    """
    Calculate stacked attention residue at current_time from all active interruptions.

    Returns dict mapping each interruption to its current residue contribution.
    """
    residue_by_interruption = {}

    for intr in interruptions:
        if intr.time <= current_time:
            t_since = current_time - intr.time
            residue = attention_residue(t_since)
            if residue > 0.001:  # ignore negligible residue
                residue_by_interruption[intr] = residue

    return residue_by_interruption


# ----------------------------
# Visualization
# ----------------------------


def main(output_base="recovery_timeline"):
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Sample timeline
    timeline = np.linspace(0, WORK_DAY_DURATION, TIMELINE_SAMPLES)

    # Build stacked residue data for all time points
    # Structure: list of (interruption, residue_array) tuples
    residue_layers = []

    for intr in INTERRUPTIONS:
        residue_array = np.array([attention_residue(t - intr.time) if t >= intr.time else 0.0 for t in timeline])
        residue_layers.append((intr, residue_array))

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Styling
    ax.set_xlim(0, WORK_DAY_DURATION)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel("Time (minutes into workday)", fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel("Accumulated Attention Residue", fontsize=12, color=TEXT_COLOR)
    ax.set_title(
        "The Recovery Window Impossibility\nWhy constant interruptions prevent deep focus",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=20,
    )
    ax.grid(True, alpha=0.3, color=GRID_COLOR, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add reference line for "sustainable" attention load
    sustainable_threshold = 1.2
    ax.axhline(
        y=sustainable_threshold,
        color=ATTENTION_LIMIT_COLOR,
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Sustainable threshold",
    )

    # Add 23-minute recovery annotation (will be placed later)
    recovery_annotation = ax.annotate(
        "23 min needed for\nfull recovery",
        xy=(0, 0),  # will be updated
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
        fontsize=9,
        color=TEXT_COLOR,
        visible=False,
    )

    # Initialize empty stackplot (will be updated in animation)
    # We'll use fill_between for each layer to control stacking
    fill_objects = []

    # Interruption markers (vertical lines and labels)
    interruption_markers = []  # will hold (vline, text) tuples

    # Time cursor
    time_cursor = ax.axvline(x=0, color="red", linewidth=2, linestyle="-", alpha=0.7)

    # Current time text
    # time_text = ax.text(
    #    0.02, 0.97,
    #    '',
    #    transform=ax.transAxes,
    #    fontsize=11,
    #    verticalalignment='top',
    #    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    #    color=TEXT_COLOR
    # )

    # Animation state
    total_frames = int(ANIMATION_DURATION * FPS)  # frames for the animation duration

    def init():
        """Initialize animation."""
        return []

    def update(frame):
        """Update animation frame."""
        # Calculate current time (in minutes)
        current_time = (frame / total_frames) * WORK_DAY_DURATION

        # Update cursor
        time_cursor.set_xdata([current_time])

        # Update time text
        hours = int(current_time // 60)
        mins = int(current_time % 60)
        # time_text.set_text(f'Time: {hours:02d}:{mins:02d}')

        # Add interruption markers as they occur
        for intr in INTERRUPTIONS:
            # Check if this interruption just occurred
            if intr.time <= current_time:
                # Check if we've already added this marker
                if not any(marker[0].get_xdata()[0] == intr.time for marker in interruption_markers):
                    # Add vertical line
                    vline = ax.axvline(x=intr.time, color=intr.color, linewidth=1.5, linestyle="--", alpha=0.5)

                    # Add label at top
                    text = ax.text(
                        intr.time,
                        Y_MAX * 0.98,
                        intr.task,
                        rotation=90,
                        fontsize=8,
                        color=intr.color,
                        verticalalignment="top",
                        horizontalalignment="right",
                        alpha=0.8,
                    )

                    interruption_markers.append((vline, text))

        # Clear previous fills
        for fill_obj in fill_objects:
            fill_obj.remove()
        fill_objects.clear()

        # Build stacked areas up to current time
        cumulative_bottom = np.zeros(TIMELINE_SAMPLES)

        for intr, residue_array in residue_layers:
            # Only show residue up to current time
            visible_residue = np.where(timeline <= current_time, residue_array, 0)

            # Stack on top of previous layers
            cumulative_top = cumulative_bottom + visible_residue

            # Fill area
            fill = ax.fill_between(
                timeline, cumulative_bottom, cumulative_top, color=intr.color, alpha=0.6, linewidth=0, step=None
            )
            fill_objects.append(fill)

            # Update bottom for next layer
            cumulative_bottom = cumulative_top

        # Show recovery annotation for the long gap (around 68 min)
        if 35 < current_time < 68 and len([i for i in INTERRUPTIONS if i.time <= current_time]) >= 6:
            # Position annotation at the gap
            recovery_annotation.xy = (40, 0.5)
            recovery_annotation.set_visible(True)
        else:
            recovery_annotation.set_visible(False)

        # Return all artists that need to be redrawn
        marker_objects = [obj for marker in interruption_markers for obj in marker]
        return [time_cursor, recovery_annotation] + fill_objects + marker_objects

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        blit=False,  # blit=True can cause issues with fill_between
        interval=1000 / FPS,
    )

    # Save
    use_ffmpeg = animation.writers.is_available("ffmpeg")
    writer = (
        animation.FFMpegWriter(fps=FPS, codec="libx264", bitrate=2400 * 8)
        if use_ffmpeg
        else animation.PillowWriter(fps=FPS)
    )
    ext = ".mp4" if use_ffmpeg else ".gif"
    outfile = str(output_dir / f"{output_base}{ext}")

    print(f"Generating animation... (ffmpeg: {use_ffmpeg})")
    print(f"Output: {outfile}")

    anim.save(outfile, writer=writer)

    print(f"Done: {outfile}")
    plt.close()


if __name__ == "__main__":
    main()
