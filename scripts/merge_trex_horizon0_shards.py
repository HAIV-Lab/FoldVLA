#!/usr/bin/env python3
"""Merge completed horizon-0 trajectory shards and render one full-time plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from shadow_replay.visualization import save_horizon0_error_over_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    args = parser.parse_args()

    pattern = f"shard_*/episode_{args.episode_id:04d}_horizon0_samples.npz"
    paths = sorted(args.shard_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shard artifacts matching {pattern}")
    keys = (
        "source_steps",
        "frame_indices",
        "timestamps",
        "gt_action",
        "pred_action_raw",
        "pred_action_safe",
        "pred_action_raw_chunks",
        "pred_action_safe_chunks",
        "latency_inference_ms",
    )
    combined = {}
    for key in keys:
        values = []
        for path in paths:
            with np.load(path) as shard:
                values.append(np.asarray(shard[key]))
        combined[key] = np.concatenate(values)
    order = np.argsort(combined["source_steps"])
    combined = {key: value[order] for key, value in combined.items()}
    expected = np.arange(args.expected_frames)
    if not np.array_equal(combined["source_steps"], expected):
        raise ValueError(
            "Merged shards do not cover every source frame exactly once: "
            f"got {len(combined['source_steps'])}, expected {args.expected_frames}"
        )

    stats = json.loads(args.dataset_stats.read_text(encoding="utf-8"))["action"]
    lower = np.asarray(stats["min"], dtype=np.float32)
    upper = np.asarray(stats["max"], dtype=np.float32)
    output_dir = args.shard_root / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = (
        output_dir
        / f"episode_{args.episode_id:04d}_horizon0_error_full_30hz.png"
    )
    save_horizon0_error_over_time(
        plot_path,
        timestamps=combined["timestamps"],
        gt_action=combined["gt_action"],
        pred_action_raw=combined["pred_action_raw"],
        pred_action_safe=combined["pred_action_safe"],
        dataset_lower=lower,
        dataset_upper=upper,
    )
    np.savez_compressed(
        output_dir
        / f"episode_{args.episode_id:04d}_action_chunks_full_30hz.npz",
        **combined,
    )
    clipped_error = np.abs(
        np.clip(combined["pred_action_raw"], lower, upper)
        - np.clip(combined["gt_action"], lower, upper)
    )
    summary = {
        "episode_id": args.episode_id,
        "frame_count": len(expected),
        "trajectory_duration_seconds": float(
            combined["timestamps"][-1] - combined["timestamps"][0]
        ),
        "dataset_clipped_horizon0_mae_rad": float(clipped_error.mean()),
        "dataset_clipped_horizon0_median_frame_mae_rad": float(
            np.median(clipped_error.mean(axis=1))
        ),
        "dataset_clipped_horizon0_p95_frame_mae_rad": float(
            np.percentile(clipped_error.mean(axis=1), 95)
        ),
        "mean_inference_latency_ms": float(
            np.mean(combined["latency_inference_ms"])
        ),
        "action_chunk_shape": list(combined["pred_action_safe_chunks"].shape),
        "shards": [str(path) for path in paths],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(plot_path)


if __name__ == "__main__":
    main()
