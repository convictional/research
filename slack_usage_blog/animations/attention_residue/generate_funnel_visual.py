"""
Attention Residue (granular physics) — fixed funnel & emitter

- Real collisions via Pymunk (Chipmunk2D)
- Sweeping emitter forces interaction with funnel walls
- Clear neck drainage between pours (residue effect)
- Clean Matplotlib rendering to MP4 / GIF

Author: (you)
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgba

import pymunk
from pymunk import Vec2d

# ----------------------------
# Scenario parameters
# ----------------------------


@dataclass
class TaskSpec:
    name: str
    color: str  # Matplotlib color
    count: int  # particles to emit in the pour
    pour_time: float  # seconds of emission
    gap_after: float  # seconds of gap after the pour


# Palette + timings (edit these for your article)
TASKS: List[TaskSpec] = [
    TaskSpec("Email triage", "#4C78A8", 1800, 1.6, 1.0),  # longer gap -> more drain
    TaskSpec("Code review", "#59A14F", 800, 1.4, 1.2),  # short gap -> more residue
    TaskSpec("Standup", "#E45756", 500, 1.5, 2.0),  # long gap -> strong drain
    TaskSpec("Deep work", "#B279A2", 600, 1.4, 3.0),  # long gap -> strong drain
]

SEED = 7
FPS = 30

# Physics step; small for stable segment collisions
DT = 1.0 / 600.0
STEPS_PER_FRAME = int((1.0 / FPS) / DT)  # ≈20 steps/frame

# Granular parameters
PARTICLE_RADIUS = 0.025
PARTICLE_MASS = 0.05

# Funnel geometry (world units are arbitrary but consistent)
Y_TOP = 2.8
Y_MID = 1.325  # intermediate transition point
Y_NECK_TOP = -0.55
Y_NECK_BOTTOM = -1.8
X_TOP_HALF = 2.9
X_NECK_HALF = 0.15  # tighter neck shows "drain" clearly
# Calculate mid-point to maintain 22.5 deg angle from vertical
X_MID_HALF = X_NECK_HALF + (Y_MID - Y_NECK_TOP) * math.tan(math.radians(22.5))

# Wall collision “thickness” (radius for Segment shapes)
WALL_THICKNESS = 0.06

# Camera / draw limits
X_MIN, X_MAX = -3.6, 3.6
Y_MIN, Y_MAX = -2.5, 3.6

# Visuals
BG_COLOR = "#FFFFFF"
FUNNEL_COLOR = "#9AA0A6"
TEXT_COLOR = "#333333"
DPI = 150
FIGSIZE = (7.2, 8.8)
LINE_WIDTH = 3.0

# ----------------------------
# Helpers
# ----------------------------


def lerp(a, b, t):
    return a + (b - a) * t


def wall_half_width(y):
    """Half width of funnel at vertical y (three-segment profile)."""
    if y <= Y_NECK_TOP:
        return X_NECK_HALF
    elif y <= Y_MID:
        t = (y - Y_NECK_TOP) / (Y_MID - Y_NECK_TOP)
        return lerp(X_NECK_HALF, X_MID_HALF, t)
    else:
        t = (y - Y_MID) / (Y_TOP - Y_MID)
        return lerp(X_MID_HALF, X_TOP_HALF, t)


def add_funnel_walls(space: pymunk.Space):
    """Add static collision geometry matching the visual funnel."""
    static = space.static_body
    segs = []

    # Three-segment sloped walls for smooth transition
    left_top = (-X_TOP_HALF, Y_TOP)
    left_mid = (-X_MID_HALF, Y_MID)
    left_neck = (-X_NECK_HALF, Y_NECK_TOP)
    right_top = (X_TOP_HALF, Y_TOP)
    right_mid = (X_MID_HALF, Y_MID)
    right_neck = (X_NECK_HALF, Y_NECK_TOP)

    # Top segment (steeper)
    segs.append(pymunk.Segment(static, left_top, left_mid, WALL_THICKNESS))
    segs.append(pymunk.Segment(static, right_top, right_mid, WALL_THICKNESS))

    # Middle segment (22.5 deg from vertical)
    segs.append(pymunk.Segment(static, left_mid, left_neck, WALL_THICKNESS))
    segs.append(pymunk.Segment(static, right_mid, right_neck, WALL_THICKNESS))

    # Neck verticals
    segs.append(pymunk.Segment(static, (-X_NECK_HALF, Y_NECK_TOP), (-X_NECK_HALF, Y_NECK_BOTTOM), WALL_THICKNESS))
    segs.append(pymunk.Segment(static, (X_NECK_HALF, Y_NECK_TOP), (X_NECK_HALF, Y_NECK_BOTTOM), WALL_THICKNESS))

    for s in segs:
        s.elasticity = 0.08
        s.friction = 0.002
        # *** Important: no shape filters — use defaults to guarantee contacts ***

    space.add(*segs)
    return segs


def draw_funnel_outline(ax):
    """Crisp outline that mirrors the physics walls."""
    # Top slope
    ax.plot([-X_TOP_HALF, -X_MID_HALF], [Y_TOP, Y_MID], color=FUNNEL_COLOR, linewidth=LINE_WIDTH)
    ax.plot([X_TOP_HALF, X_MID_HALF], [Y_TOP, Y_MID], color=FUNNEL_COLOR, linewidth=LINE_WIDTH)
    # Middle slope (22.5 deg transition)
    ax.plot([-X_MID_HALF, -X_NECK_HALF], [Y_MID, Y_NECK_TOP], color=FUNNEL_COLOR, linewidth=LINE_WIDTH)
    ax.plot([X_MID_HALF, X_NECK_HALF], [Y_MID, Y_NECK_TOP], color=FUNNEL_COLOR, linewidth=LINE_WIDTH)
    # Neck
    ax.plot([-X_NECK_HALF, -X_NECK_HALF], [Y_NECK_TOP, Y_NECK_BOTTOM], color=FUNNEL_COLOR, linewidth=LINE_WIDTH)
    ax.plot([X_NECK_HALF, X_NECK_HALF], [Y_NECK_TOP, Y_NECK_BOTTOM], color=FUNNEL_COLOR, linewidth=LINE_WIDTH)


# ----------------------------
# Particles
# ----------------------------


class Particle:
    __slots__ = ("body", "shape", "task_idx")

    def __init__(self, body, shape, task_idx):
        self.body = body
        self.shape = shape
        self.task_idx = task_idx


def spawn_particle(space: pymunk.Space, task_idx: int, x_pos: float) -> Particle:
    """Spawn a single particle at x_pos, just above the rim."""
    r = PARTICLE_RADIUS
    m = PARTICLE_MASS
    moment = pymunk.moment_for_circle(m, 0, r)

    body = pymunk.Body(m, moment)
    x_jitter = (random.random() - 0.5) * 0.12
    y0 = Y_TOP + 0.55
    body.position = Vec2d(x_pos + x_jitter, y0)
    # Small downward bias + tiny horizontal push
    body.velocity = Vec2d((random.random() - 0.5) * 0.2, -0.35)

    shape = pymunk.Circle(body, r)
    shape.elasticity = 0.25
    shape.friction = 0.0002
    # default shape filter (collides with everything)

    space.add(body, shape)
    return Particle(body, shape, task_idx)


# ----------------------------
# Main
# ----------------------------


def main(output_base="attention_residue_particles"):
    random.seed(SEED)
    np.random.seed(SEED)

    # ---- Output setup
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # ---- Physics world
    space = pymunk.Space()
    space.gravity = (0, -9.8)
    space.damping = 0.9  # 99
    space.iterations = 30
    add_funnel_walls(space)

    # ---- Build schedule (pour/gap pairs + settle)
    schedule = []
    for i, t in enumerate(TASKS):
        schedule.append(("pour", i, t.pour_time))
        schedule.append(("gap", i, t.gap_after))
    schedule.append(("settle", None, 2.0))
    total_time = sum(d for _, _, d in schedule)

    # ---- Figure
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    draw_funnel_outline(ax)

    # One scatter per task
    scatters = []
    base_point_area = (PARTICLE_RADIUS * 45.0) ** 2
    for t in TASKS:
        sc = ax.scatter([], [], s=base_point_area, color=to_rgba(t.color, 0.85), edgecolors="none")
        scatters.append(sc)

    particles: List[Particle] = []

    # Writer
    use_ffmpeg = animation.writers.is_available("ffmpeg")
    writer = (
        animation.FFMpegWriter(fps=FPS, codec="libx264", bitrate=1800 * 8)
        if use_ffmpeg
        else animation.PillowWriter(fps=FPS)
    )
    ext = ".mp4" if use_ffmpeg else ".gif"
    outfile = str(output_dir / f"{output_base}{ext}")
    print(f"Writing to: {outfile}  (ffmpeg: {use_ffmpeg})")

    # Emission bookkeeping
    remaining_to_spawn = [t.count for t in TASKS]
    current_phase_idx, phase_elapsed = 0, 0.0
    sim_time = 0.0

    # Emitter sweep config: moves across the mouth each pour
    def emitter_x(progress, left_to_right=True):
        # keep inside walls with margin
        L = -X_TOP_HALF * 0.82
        R = X_TOP_HALF * 0.82
        return lerp(L, R, progress) if left_to_right else lerp(R, L, progress)

    def update_scatters():
        grouped = [[] for _ in TASKS]
        to_remove = []
        for p in particles:
            pos = p.body.position
            if pos.y < Y_MIN - 0.5 or pos.x < X_MIN - 1.0 or pos.x > X_MAX + 1.0:
                to_remove.append(p)
                continue
            grouped[p.task_idx].append([pos.x, pos.y])
        if to_remove:
            for p in to_remove:
                try:
                    space.remove(p.shape, p.body)
                except Exception:
                    pass
            particles[:] = [p for p in particles if p not in to_remove]

        for i, sc in enumerate(scatters):
            arr = np.asarray(grouped[i], dtype=float)
            sc.set_offsets(arr if arr.size else np.zeros((0, 2)))

    # Main write loop
    with writer.saving(fig, outfile, DPI):
        total_frames = int(math.ceil(total_time * FPS))
        left_to_right = True

        for frame_idx in range(total_frames):
            phase, task_idx, duration = schedule[current_phase_idx]

            # Start-of-phase setup
            if phase_elapsed == 0.0:
                if phase == "pour":
                    spawn_accum = 0.0

            # Emit during pour (sweeping emitter)
            if phase == "pour":
                # recompute on each frame so we can keep 'spawn_accum' in scope
                progress = min(1.0, phase_elapsed / max(TASKS[task_idx].pour_time, 1e-6))
                x_emit = emitter_x(progress, left_to_right)
                # rate-controlled spawning
                spawn_accum += (remaining_to_spawn[task_idx] / max(TASKS[task_idx].pour_time, 1e-6)) * (1.0 / FPS)
                n_new = int(spawn_accum)
                if n_new > 0:
                    spawn_accum -= n_new
                    n_new = min(n_new, remaining_to_spawn[task_idx])
                    for _ in range(n_new):
                        particles.append(spawn_particle(space, task_idx, x_emit))
                    remaining_to_spawn[task_idx] -= n_new

            # Physics sub-steps
            for _ in range(STEPS_PER_FRAME):
                space.step(DT)

            # Visual update
            update_scatters()
            writer.grab_frame()

            # Advance time
            sim_time += 1.0 / FPS
            phase_elapsed += 1.0 / FPS

            if phase_elapsed >= duration - 1e-9:
                # flip sweep direction each pour for variety
                if phase == "pour":
                    left_to_right = not left_to_right
                current_phase_idx += 1
                phase_elapsed = 0.0
                if current_phase_idx >= len(schedule):
                    break

    print(f"Done: {outfile}")


if __name__ == "__main__":
    main()
