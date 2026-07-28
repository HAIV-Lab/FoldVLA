#!/usr/bin/env python3
"""Compare Shadow Replay errors before and after clipping GT/prediction to URDF limits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shadow_replay.safety import load_north_urdf_limits, north_dataset_joint_names


GROUPS = {
    "Left arm": (0, 7),
    "Left hand": (7, 29),
    "Right arm": (29, 36),
    "Right hand": (36, 58),
    "Body": (58, 65),
}


def _metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    error = pred[valid] - gt[valid]
    first_error = pred[:, 0] - gt[:, 0]
    horizon_mae = []
    horizon_count = []
    for horizon in range(pred.shape[1]):
        horizon_valid = valid[:, horizon]
        horizon_count.append(int(horizon_valid.sum()))
        if horizon_valid.any():
            value = np.abs(
                pred[horizon_valid, horizon] - gt[horizon_valid, horizon]
            ).mean()
            horizon_mae.append(float(value))
        else:
            horizon_mae.append(None)
    return {
        "valid_action_count": int(valid.sum()),
        "action_mae": float(np.abs(error).mean()),
        "action_mse": float(np.square(error).mean()),
        "action_rmse": float(np.sqrt(np.square(error).mean())),
        "first_action_mae": float(np.abs(first_error).mean()),
        "first_action_mse": float(np.square(first_error).mean()),
        "first_action_rmse": float(np.sqrt(np.square(first_error).mean())),
        "per_dimension_mae": np.abs(error).mean(axis=0).tolist(),
        "horizon_valid_count": horizon_count,
        "horizon_mae": horizon_mae,
    }


def _group_metrics(
    pred_raw: np.ndarray,
    gt_raw: np.ndarray,
    pred_clipped: np.ndarray,
    gt_clipped: np.ndarray,
    valid: np.ndarray,
) -> dict:
    result = {}
    for name, (start, stop) in GROUPS.items():
        raw_error = pred_raw[:, :, start:stop][valid] - gt_raw[:, :, start:stop][valid]
        clipped_error = (
            pred_clipped[:, :, start:stop][valid]
            - gt_clipped[:, :, start:stop][valid]
        )
        result[name] = {
            "dimensions": [start, stop],
            "raw_mae": float(np.abs(raw_error).mean()),
            "clipped_mae": float(np.abs(clipped_error).mean()),
            "mae_delta": float(
                np.abs(clipped_error).mean() - np.abs(raw_error).mean()
            ),
            "relative_mae_change_percent": float(
                (
                    np.abs(clipped_error).mean() / np.abs(raw_error).mean()
                    - 1.0
                )
                * 100.0
            ),
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-npz", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episode = np.load(args.episode_npz)
    pred_raw = np.asarray(episode["pred_action_denormalized"], dtype=np.float64)
    gt_raw = np.asarray(episode["gt_native_chunks"], dtype=np.float64)
    valid = np.asarray(episode["valid_mask"], dtype=bool)
    if pred_raw.shape != gt_raw.shape or pred_raw.shape[-1] != 65:
        raise ValueError(
            f"Expected matching (..., 65) arrays, got {pred_raw.shape} and {gt_raw.shape}"
        )

    dataset_names = north_dataset_joint_names()
    limits = load_north_urdf_limits(args.urdf, dataset_names)
    lower = limits.lower.astype(np.float64)
    upper = limits.upper.astype(np.float64)
    pred_clipped = np.clip(pred_raw, lower, upper)
    gt_clipped = np.clip(gt_raw, lower, upper)

    pred_changed = ~np.isclose(pred_raw, pred_clipped, rtol=0.0, atol=1e-12)
    gt_changed = ~np.isclose(gt_raw, gt_clipped, rtol=0.0, atol=1e-12)
    valid3 = np.broadcast_to(valid[..., None], pred_raw.shape)

    raw_metrics = _metrics(pred_raw, gt_raw, valid)
    clipped_metrics = _metrics(pred_clipped, gt_clipped, valid)
    raw_mae = raw_metrics["action_mae"]
    clipped_mae = clipped_metrics["action_mae"]
    raw_mse = raw_metrics["action_mse"]
    clipped_mse = clipped_metrics["action_mse"]

    clipping = {
        "comparison_scalar_count": int(valid.sum() * pred_raw.shape[-1]),
        "comparison_action_point_count": int(valid.sum()),
        "gt_clipped_scalar_count": int((gt_changed & valid3).sum()),
        "pred_clipped_scalar_count": int((pred_changed & valid3).sum()),
        "gt_clipped_action_point_count": int((gt_changed & valid3).any(axis=-1).sum()),
        "pred_clipped_action_point_count": int(
            (pred_changed & valid3).any(axis=-1).sum()
        ),
        "pred_clipped_action_point_count_all_predictions": int(
            pred_changed.any(axis=-1).sum()
        ),
        "pred_total_action_point_count_all_predictions": int(
            np.prod(pred_raw.shape[:-1])
        ),
        "gt_max_clip_distance": float(
            np.abs(gt_raw[valid] - gt_clipped[valid]).max(initial=0.0)
        ),
        "pred_max_clip_distance": float(
            np.abs(pred_raw[valid] - pred_clipped[valid]).max(initial=0.0)
        ),
    }
    group_metrics = _group_metrics(
        pred_raw, gt_raw, pred_clipped, gt_clipped, valid
    )
    for name, (start, stop) in GROUPS.items():
        group_valid = valid3[:, :, start:stop]
        group_metrics[name].update(
            {
                "gt_clipped_scalar_count": int(
                    (gt_changed[:, :, start:stop] & group_valid).sum()
                ),
                "pred_clipped_scalar_count": int(
                    (pred_changed[:, :, start:stop] & group_valid).sum()
                ),
            }
        )

    summary = {
        "source_episode_npz": str(args.episode_npz.resolve()),
        "urdf": str(args.urdf.resolve()),
        "joint_order": "canonical North absolute_joint65",
        "raw": raw_metrics,
        "both_urdf_clipped": clipped_metrics,
        "delta": {
            "action_mae": clipped_mae - raw_mae,
            "action_mae_relative_change_percent": (
                clipped_mae / raw_mae - 1.0
            )
            * 100.0,
            "action_mse": clipped_mse - raw_mse,
            "action_mse_relative_change_percent": (
                clipped_mse / raw_mse - 1.0
            )
            * 100.0,
        },
        "clipping": clipping,
        "groups": group_metrics,
    }
    metrics_path = args.output_dir / "urdf_clipped_metrics.json"
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    raw_per_dim = np.asarray(raw_metrics["per_dimension_mae"])
    clipped_per_dim = np.asarray(clipped_metrics["per_dimension_mae"])
    gt_clip_per_dim = (gt_changed & valid3).sum(axis=(0, 1))
    pred_clip_per_dim = (pred_changed & valid3).sum(axis=(0, 1))
    csv_path = args.output_dir / "urdf_clipped_per_joint.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "dataset_joint",
                "urdf_joint",
                "lower",
                "upper",
                "raw_mae",
                "clipped_mae",
                "mae_delta",
                "gt_clipped_count",
                "pred_clipped_count",
            ]
        )
        for index in range(65):
            writer.writerow(
                [
                    index,
                    dataset_names[index],
                    limits.names[index],
                    lower[index],
                    upper[index],
                    raw_per_dim[index],
                    clipped_per_dim[index],
                    clipped_per_dim[index] - raw_per_dim[index],
                    int(gt_clip_per_dim[index]),
                    int(pred_clip_per_dim[index]),
                ]
            )

    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "seaborn-whitegrid"
    )
    plt.style.use(style)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    axes[0, 0].bar(
        ["Raw", "Both URDF-clipped"],
        [raw_metrics["action_mae"], clipped_metrics["action_mae"]],
        color=["#64748b", "#2563eb"],
    )
    axes[0, 0].set_ylabel("MAE (rad)")
    axes[0, 0].set_title("Overall error")
    for index, value in enumerate(
        [raw_metrics["action_mae"], clipped_metrics["action_mae"]]
    ):
        axes[0, 0].text(index, value, f"{value:.5f}", ha="center", va="bottom")

    group_names = list(GROUPS)
    positions = np.arange(len(group_names))
    width = 0.36
    axes[0, 1].bar(
        positions - width / 2,
        [group_metrics[name]["raw_mae"] for name in group_names],
        width,
        label="Raw",
        color="#64748b",
    )
    axes[0, 1].bar(
        positions + width / 2,
        [group_metrics[name]["clipped_mae"] for name in group_names],
        width,
        label="Both clipped",
        color="#2563eb",
    )
    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels(group_names, rotation=20, ha="right")
    axes[0, 1].set_ylabel("MAE (rad)")
    axes[0, 1].set_title("Error by 65D group")
    axes[0, 1].legend()

    horizons = np.arange(pred_raw.shape[1])
    axes[1, 0].plot(
        horizons, raw_metrics["horizon_mae"], "o-", label="Raw", color="#64748b"
    )
    axes[1, 0].plot(
        horizons,
        clipped_metrics["horizon_mae"],
        "o-",
        label="Both clipped",
        color="#2563eb",
    )
    axes[1, 0].set_xlabel("Action chunk horizon")
    axes[1, 0].set_ylabel("MAE (rad)")
    axes[1, 0].set_title("Horizon error")
    axes[1, 0].set_xticks(horizons)
    axes[1, 0].legend()

    axes[1, 1].bar(
        np.arange(65) - width / 2,
        gt_clip_per_dim,
        width,
        label="GT",
        color="#f59e0b",
    )
    axes[1, 1].bar(
        np.arange(65) + width / 2,
        pred_clip_per_dim,
        width,
        label="Prediction",
        color="#dc2626",
    )
    axes[1, 1].set_xlabel("65D joint index")
    axes[1, 1].set_ylabel("Clipped sample count (valid comparisons)")
    axes[1, 1].set_title("URDF clipping frequency per joint")
    axes[1, 1].legend()

    figure_path = args.output_dir / "urdf_clipped_error_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    top_indices = np.argsort(
        np.maximum(gt_clip_per_dim, pred_clip_per_dim)
    )[-8:][::-1]
    flat_gt = gt_raw[valid]
    flat_pred = pred_raw[valid]
    flat_gt_clipped = gt_clipped[valid]
    flat_pred_clipped = pred_clipped[valid]
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), constrained_layout=True)
    for axis, joint_index in zip(axes.flat, top_indices):
        axis.plot(
            flat_gt[:, joint_index],
            color="#f59e0b",
            alpha=0.35,
            linewidth=1.0,
            label="GT raw",
        )
        axis.plot(
            flat_pred[:, joint_index],
            color="#dc2626",
            alpha=0.35,
            linewidth=1.0,
            label="Pred raw",
        )
        axis.plot(
            flat_gt_clipped[:, joint_index],
            color="#15803d",
            linewidth=1.4,
            label="GT clipped",
        )
        axis.plot(
            flat_pred_clipped[:, joint_index],
            color="#2563eb",
            linewidth=1.4,
            label="Pred clipped",
        )
        axis.axhline(lower[joint_index], color="black", linestyle="--", linewidth=0.8)
        axis.axhline(upper[joint_index], color="black", linestyle="--", linewidth=0.8)
        axis.set_title(
            f"{joint_index}: {dataset_names[joint_index]} "
            f"(GT {gt_clip_per_dim[joint_index]}, pred {pred_clip_per_dim[joint_index]})"
        )
        axis.set_xlabel("Valid chunk-point index")
        axis.set_ylabel("Position (rad)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    trajectory_path = args.output_dir / "urdf_clipped_top_joint_trajectories.png"
    fig.savefig(trajectory_path, dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {trajectory_path}")


if __name__ == "__main__":
    main()
