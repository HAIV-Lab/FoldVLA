"""Optional plots and annotated videos for Shadow Replay artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_episode_plots(
    path: Path,
    *,
    gt_action: np.ndarray,
    pred_action_safe: np.ndarray,
    latency_ms: Sequence[float],
    safety_valid: Sequence[bool],
    max_curve_dimensions: int = 16,
) -> None:
    """Save action curves, all-dimension error heatmap, latency, and safety timeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    gt = np.asarray(gt_action)
    pred = np.asarray(pred_action_safe)
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError(f"Episode plot expects matching [T,D] arrays, got {gt.shape}/{pred.shape}")
    count = min(gt.shape[1], max_curve_dimensions)
    time_axis = np.arange(len(gt))
    figure, axes = plt.subplots(4, 1, figsize=(14, 13), constrained_layout=True)
    for dimension in range(count):
        axes[0].plot(time_axis, gt[:, dimension], linewidth=0.8, alpha=0.7)
        axes[0].plot(time_axis, pred[:, dimension], linewidth=0.7, alpha=0.5, linestyle="--")
    axes[0].set_title(
        f"GT (solid) and safe prediction (dashed), first {count}/{gt.shape[1]} dimensions"
    )
    axes[0].set_ylabel("joint target [rad]")
    image = axes[1].imshow(
        np.abs(pred - gt).T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    axes[1].set_title("Absolute joint error, all dimensions")
    axes[1].set_ylabel("action dimension")
    figure.colorbar(image, ax=axes[1], label="absolute error [rad]")
    axes[2].plot(time_axis, np.asarray(latency_ms), color="tab:purple")
    axes[2].set_title("Total latency")
    axes[2].set_ylabel("ms")
    valid = np.asarray(safety_valid, dtype=bool)
    axes[3].step(time_axis, (~valid).astype(np.int8), where="post", color="tab:red")
    axes[3].set_title("Safety violation timeline")
    axes[3].set_ylabel("violation")
    axes[3].set_xlabel("replay step")
    axes[3].set_yticks([0, 1])
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_dataset_plots(
    directory: Path,
    *,
    horizon_mae: Sequence[Optional[float]],
    latency_ms: Sequence[float],
    stage_metrics: Mapping[str, Mapping[str, Any]],
    chunk_boundary_values: Optional[Sequence[float]] = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    horizons = np.arange(len(horizon_mae))
    values = np.asarray(
        [np.nan if value is None else float(value) for value in horizon_mae]
    )
    axis.plot(horizons, values, marker="o", linewidth=1.5)
    axis.set_xlabel("prediction horizon")
    axis.set_ylabel("MAE")
    axis.set_title("Action chunk horizon-error curve")
    axis.grid(alpha=0.3)
    figure.savefig(directory / "horizon_error.png", dpi=150)
    plt.close(figure)

    latency = np.asarray(latency_ms, dtype=np.float64)
    latency = latency[np.isfinite(latency)]
    if len(latency):
        figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        axis.hist(latency, bins=min(50, max(5, int(np.sqrt(len(latency))))), color="tab:blue")
        axis.set_xlabel("latency [ms]")
        axis.set_ylabel("count")
        axis.set_title("Inference/total latency distribution")
        figure.savefig(directory / "latency_distribution.png", dpi=150)
        plt.close(figure)

    if stage_metrics:
        names = list(stage_metrics)
        values = [
            np.nan
            if stage_metrics[name].get("action_mae") is None
            else float(stage_metrics[name]["action_mae"])
            for name in names
        ]
        figure, axis = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.5), constrained_layout=True)
        axis.bar(names, values, color="tab:green")
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylabel("action MAE")
        axis.set_title("Per-stage action error")
        figure.savefig(directory / "stage_error.png", dpi=150)
        plt.close(figure)

    if chunk_boundary_values:
        figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        axis.plot(chunk_boundary_values, color="tab:orange")
        axis.set_xlabel("chunk boundary index")
        axis.set_ylabel("MAE")
        axis.set_title("Chunk boundary discontinuity")
        figure.savefig(directory / "chunk_boundary.png", dpi=150)
        plt.close(figure)


def _format_vector(value: Optional[np.ndarray], count: int = 5) -> str:
    if value is None:
        return "N/A"
    vector = np.asarray(value).reshape(-1)
    text = ",".join(f"{number:.3f}" for number in vector[:count])
    return f"[{text}{',...' if len(vector) > count else ''}]"


def save_episode_video(
    path: Path,
    *,
    video_reader: Any,
    frame_indices: Sequence[int],
    frame_results: Sequence[Mapping[str, Any]],
    fps: float,
) -> None:
    """Overlay replay evidence on original images; no projection is fabricated."""
    if len(frame_indices) != len(frame_results):
        raise ValueError("frame_indices and frame_results lengths differ")
    if not frame_indices:
        return
    first = video_reader.read(int(frame_indices[0]))
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for index, (frame_index, result) in enumerate(zip(frame_indices, frame_results)):
            rgb = first if index == 0 else video_reader.read(int(frame_index))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = bgr.copy()
            panel_height = min(height, 230)
            cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.62, bgr, 0.38, 0.0, bgr)
            lines = [
                f"episode={result.get('episode_id')} step={result.get('step_id')} frame={frame_index}",
                f"instruction={str(result.get('instruction', ''))[:90]}",
                f"stage={result.get('stage_gt')} safety={result.get('safety_valid')} "
                f"flags={','.join(result.get('safety_flags') or [])[:80]}",
                f"GT={_format_vector(result.get('gt_action'))}",
                f"pred_safe={_format_vector(result.get('pred_action_safe'))}",
                f"joint_MAE={result.get('joint_mae', float('nan')):.4f} "
                f"latency={result.get('total_latency_ms', float('nan')):.2f}ms",
                f"stage_pred={result.get('stage_pred')} gate={result.get('gate_output')}",
            ]
            for row, text in enumerate(lines):
                cv2.putText(
                    bgr,
                    text,
                    (12, 25 + row * 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(bgr)
    finally:
        writer.release()
