#!/usr/bin/env python3
"""Plot UniDex training loss lines from a log file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_RE = re.compile(r"step=(\d+)/(?:-?\d+)\s+loss=([0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss from UniDex finetune logs.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smooth", type=int, default=50)
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        out.append(running / min(idx + 1, window))
    return out


def main() -> None:
    args = parse_args()
    out = args.out or args.log.with_suffix(".loss.png")

    steps: list[int] = []
    losses: list[float] = []
    for line in args.log.read_text(errors="ignore").splitlines():
        match = LOSS_RE.search(line)
        if not match:
            continue
        steps.append(int(match.group(1)))
        losses.append(float(match.group(2)))

    if not steps:
        raise SystemExit(f"No loss lines found in {args.log}")

    plt.figure(figsize=(10, 5), dpi=160)
    plt.plot(steps, losses, color="#8a8f98", alpha=0.35, linewidth=0.9, label="loss")
    smoothed = moving_average(losses, args.smooth)
    if smoothed != losses:
        plt.plot(steps, smoothed, color="#0f766e", linewidth=1.8, label=f"moving avg ({args.smooth})")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("UniDex Training Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"saved {out} ({len(steps)} points)")


if __name__ == "__main__":
    main()
