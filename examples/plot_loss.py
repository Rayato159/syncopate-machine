#!/usr/bin/env python3
"""Generate a loss plot from the training CSV.

Usage:
    python examples/plot_loss.py runs/10m-10k/loss.csv
    python examples/plot_loss.py runs/10m-10k/loss.csv --output runs/10m-10k/loss_plot.png
    python examples/plot_loss.py runs/10m-10k/loss.csv --smooth 50
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required.")
    print("Install with: pip install matplotlib numpy")
    sys.exit(1)


def read_loss_csv(path: str) -> tuple[list[int], list[float]]:
    """Read step,loss CSV and return (steps, losses)."""
    steps = []
    losses = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


def smooth(values: list[float], window: int) -> list[float]:
    """Apply moving average smoothing."""
    if window <= 1:
        return values
    arr = np.array(values, dtype=np.float64)
    kernel = np.ones(window) / window
    # Pad edges to maintain length
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return [float(x) for x in smoothed]


def plot_loss(
    steps: list[int],
    losses: list[float],
    output_path: str,
    smooth_window: int = 0,
    title: str = "Training Loss",
):
    """Generate and save a loss plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw loss
    ax.plot(
        steps, losses, alpha=0.3, linewidth=0.5, color="steelblue", label="Raw loss"
    )

    # Smoothed loss
    if smooth_window > 1:
        smoothed = smooth(losses, smooth_window)
        ax.plot(
            steps,
            smoothed,
            linewidth=2,
            color="darkblue",
            label=f"Smoothed (window={smooth_window})",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    min_loss = min(losses)
    max_loss = max(losses)
    final_loss = losses[-1]
    stats_text = (
        f"Final: {final_loss:.4f}\n"
        f"Best: {min_loss:.4f}\n"
        f"Worst: {max_loss:.4f}\n"
        f"Steps: {len(steps)}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training loss from CSV")
    parser.add_argument("csv", help="Path to loss.csv file")
    parser.add_argument(
        "--output", "-o", help="Output image path (default: <csv_dir>/loss_plot.png)"
    )
    parser.add_argument(
        "--smooth",
        "-s",
        type=int,
        default=50,
        help="Smoothing window size (0 for no smoothing, default: 50)",
    )
    parser.add_argument("--title", "-t", default=None, help="Plot title")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    output_path = args.output or str(csv_path.parent / "loss_plot.png")
    title = args.title or f"Training Loss — {csv_path.parent.name}"

    steps, losses = read_loss_csv(str(csv_path))
    if not steps:
        print("Error: CSV is empty")
        sys.exit(1)

    print(f"Loaded {len(steps)} data points")
    print(f"  Step range: {steps[0]}–{steps[-1]}")
    print(f"  Loss range: {min(losses):.4f}–{max(losses):.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")

    plot_loss(steps, losses, output_path, args.smooth, title)


if __name__ == "__main__":
    main()
