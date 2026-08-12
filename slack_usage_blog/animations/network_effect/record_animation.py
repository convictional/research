"""
High-Quality Network Animation Recorder

Captures network_animation.html at high resolution using Playwright
and compiles frames to broadcast-quality MP4 with ffmpeg.

Usage:
    poetry run python animations/network_effect/record_animation.py

Output:
    animations/network_effect/output/network_animation_hq.mp4

Author: Adam McCabe
"""

import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright


def record_animation_high_quality(
    html_path: Path,
    output_path: Path,
    width: int = 900,
    height: int = 650,
    duration: int = 40,
    fps: int = 30,
) -> None:
    """
    Record HTML animation to high-quality MP4.

    Args:
        html_path: Path to HTML animation file
        output_path: Path for output MP4
        width: Video width in pixels
        height: Video height in pixels
        duration: Recording duration in seconds
        fps: Frames per second
    """
    frames_dir = output_path.parent / "frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"Recording animation at {width}x{height}, {fps}fps for {duration}s...")
    print(f"Capturing {duration * fps} frames...")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})

        # Load the animation
        page.goto(f"file://{html_path.absolute()}")

        # Wait for animation to start
        page.wait_for_timeout(500)

        # Capture frames
        frame_interval = 1000 / fps  # milliseconds per frame
        total_frames = duration * fps

        for i in range(total_frames):
            page.screenshot(path=frames_dir / f"frame_{i:04d}.png")
            page.wait_for_timeout(int(frame_interval))

            if (i + 1) % 100 == 0:
                print(f"  Captured {i + 1}/{total_frames} frames...")

        browser.close()

    print(f"✓ Captured {total_frames} frames")
    print("Encoding video with ffmpeg...")

    # Compile frames to MP4 with high quality settings
    subprocess.run(
        [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",  # Better quality (slower encoding)
            "-crf",
            "18",  # High quality (18 = near-lossless, default is 23)
            "-pix_fmt",
            "yuv420p",  # Compatibility
            str(output_path),
        ],
        check=True,
    )

    print(f"✓ Encoded video: {output_path}")

    # Cleanup frames
    shutil.rmtree(frames_dir)
    print("✓ Cleaned up temporary frames")

    # Show file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nOutput: {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Resolution: {width}x{height}")
    print(f"Duration: {duration}s @ {fps}fps")


def main() -> None:
    """Record network animation at high quality."""
    output_dir = Path(__file__).parent / "output"
    html_path = output_dir / "network_animation.html"
    mp4_path = output_dir / "network_animation_hq.mp4"

    if not html_path.exists():
        print(f"Error: {html_path} not found.")
        print("Run generate_network_animation.py first to create the HTML.")
        return

    record_animation_high_quality(
        html_path=html_path,
        output_path=mp4_path,
        width=900,
        height=650,
        duration=40,
        fps=30,
    )

    print("\n✓ Done! High-quality recording complete.")


if __name__ == "__main__":
    main()
