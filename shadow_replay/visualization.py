"""Optional plots and annotated videos for Shadow Replay artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np

from .robot_projection import CameraCalibration, NorthUrdfKinematics


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _make_mp4_ide_compatible(path: Path) -> None:
    """Replace OpenCV's MPEG-4 Part 2 output with broadly playable H.264."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_h264_",
        suffix=".mp4",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(temporary),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg H.264 conversion failed for {path}: "
                f"{completed.stderr.strip()}"
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def save_horizon0_error_over_time(
    path: Path,
    *,
    timestamps: Sequence[float],
    gt_action: np.ndarray,
    pred_action_raw: np.ndarray,
    pred_action_safe: np.ndarray,
    dataset_lower: Optional[Sequence[float]] = None,
    dataset_upper: Optional[Sequence[float]] = None,
) -> None:
    """Plot per-frame horizon-0 action error against real trajectory time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    gt = np.asarray(gt_action, dtype=np.float64)
    raw = np.asarray(pred_action_raw, dtype=np.float64)
    safe = np.asarray(pred_action_safe, dtype=np.float64)
    time_axis = np.asarray(timestamps, dtype=np.float64)
    if gt.shape != raw.shape or gt.shape != safe.shape or gt.ndim != 2:
        raise ValueError(
            f"Horizon-0 plot expects matching [T,D] arrays, got "
            f"{gt.shape}/{raw.shape}/{safe.shape}"
        )
    if time_axis.shape != (len(gt),):
        raise ValueError(
            f"timestamps must have shape ({len(gt)},), got {time_axis.shape}"
        )
    time_axis = time_axis - time_axis[0]
    safe_absolute = np.abs(safe - gt)

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.plot(
        time_axis,
        safe_absolute.mean(axis=1),
        marker="s",
        markersize=3,
        linewidth=1.4,
        linestyle="--",
        color="tab:blue",
        label="safety-clipped horizon-0 MAE",
    )
    if dataset_lower is not None and dataset_upper is not None:
        lower = np.asarray(dataset_lower, dtype=np.float64)
        upper = np.asarray(dataset_upper, dtype=np.float64)
        if lower.shape != (gt.shape[1],) or upper.shape != lower.shape:
            raise ValueError(
                f"Dataset bounds must have shape ({gt.shape[1]},), got "
                f"{lower.shape}/{upper.shape}"
            )
        clipped_error = np.abs(
            np.clip(raw, lower, upper) - np.clip(gt, lower, upper)
        )
        axis.plot(
            time_axis,
            clipped_error.mean(axis=1),
            linewidth=2.0,
            color="tab:green",
            label="dataset-range-clipped horizon-0 MAE",
        )
    axis.set_title("Clipped horizon-0 action error over trajectory time")
    axis.set_xlabel("trajectory time [s]")
    axis.set_ylabel("absolute joint error [rad]")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=160)
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
    _make_mp4_ide_compatible(path)


def _pixel(
    point: np.ndarray,
    width: int,
    height: int,
    *,
    margin: int = 0,
) -> Optional[tuple[int, int]]:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        return None
    x, y = int(round(value[0])), int(round(value[1]))
    if not (-margin <= x < width + margin and -margin <= y < height + margin):
        return None
    return x, y


def _project_landmarks(
    kinematics: NorthUrdfKinematics,
    calibration: CameraCalibration,
    action65: np.ndarray,
    capture_link_poses: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    landmarks = kinematics.landmark_positions(action65)
    names = list(landmarks)
    image_points, _ = calibration.project(
        np.stack([landmarks[name] for name in names]),
        capture_link_poses,
    )
    return dict(zip(names, image_points))


def _draw_skeleton(
    image_bgr: np.ndarray,
    points: Mapping[str, np.ndarray],
    edges: Sequence[tuple[str, str]],
    *,
    color: tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> None:
    height, width = image_bgr.shape[:2]
    layer = image_bgr.copy()
    for start_name, end_name in edges:
        start = _pixel(points[start_name], width, height, margin=max(width, height))
        end = _pixel(points[end_name], width, height, margin=max(width, height))
        if start is None or end is None:
            continue
        visible, clipped_start, clipped_end = cv2.clipLine(
            (0, 0, width, height), start, end
        )
        if visible:
            cv2.line(
                layer,
                clipped_start,
                clipped_end,
                color,
                thickness,
                cv2.LINE_AA,
            )
    for name, point in points.items():
        if not (
            name.endswith("_wrist")
            or name.endswith("_fingertip")
            or name.endswith("arm_joint_7")
        ):
            continue
        pixel = _pixel(point, width, height)
        if pixel is not None:
            radius = (
                max(5, thickness + 3)
                if name.endswith("_wrist")
                else max(3, thickness + 1)
            )
            cv2.circle(layer, pixel, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, image_bgr, 1.0 - alpha, 0.0, image_bgr)


def _time_colors(count: int) -> list[tuple[int, int, int]]:
    if count < 1:
        return []
    values = np.linspace(20, 235, count, dtype=np.uint8).reshape(-1, 1)
    colors = cv2.applyColorMap(values, cv2.COLORMAP_TURBO).reshape(-1, 3)
    return [tuple(int(channel) for channel in color) for color in colors]


def _draw_future_trajectories(
    image_bgr: np.ndarray,
    projected_horizons: Sequence[Mapping[str, np.ndarray]],
) -> None:
    if not projected_horizons:
        return
    height, width = image_bgr.shape[:2]
    colors = _time_colors(len(projected_horizons))
    tracked = [
        f"{side}_{landmark}"
        for side in ("left", "right")
        for landmark in (
            "wrist",
            "thumb_fingertip",
            "index_fingertip",
            "middle_fingertip",
            "ring_fingertip",
            "pinky_fingertip",
        )
    ]
    for name in tracked:
        previous = None
        for horizon, points in enumerate(projected_horizons):
            current = _pixel(points[name], width, height)
            if current is None:
                previous = None
                continue
            color = colors[horizon]
            if previous is not None:
                cv2.line(
                    image_bgr,
                    previous,
                    current,
                    color,
                    2 if name.endswith("_wrist") else 1,
                    cv2.LINE_AA,
                )
            cv2.circle(
                image_bgr,
                current,
                4 if name.endswith("_wrist") else 2,
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            previous = current


def save_robot_projection_video(
    path: Path,
    *,
    video_reader: Any,
    frame_indices: Sequence[int],
    current_joint_states: np.ndarray,
    predicted_joint_chunks: np.ndarray,
    valid_mask: Optional[np.ndarray],
    fps: float,
    kinematics: NorthUrdfKinematics,
    calibration: CameraCalibration,
    max_frames: Optional[int] = None,
    output_stride: int = 1,
) -> None:
    """Overlay current/predicted North skeletons and 16-step 2-D trajectories.

    Future configurations are projected into the *current captured image*.  The
    head-camera pose is therefore computed from the current measured state and
    held fixed across the prediction horizon.
    """
    states = np.asarray(current_joint_states, dtype=np.float64)
    chunks = np.asarray(predicted_joint_chunks, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 65:
        raise ValueError(f"current_joint_states must be [T,65], got {states.shape}")
    if chunks.ndim != 3 or chunks.shape[0] != len(states) or chunks.shape[2] != 65:
        raise ValueError(f"predicted_joint_chunks must be [T,H,65], got {chunks.shape}")
    if len(frame_indices) != len(states):
        raise ValueError("frame_indices and current_joint_states lengths differ")
    if output_stride < 1:
        raise ValueError("output_stride must be positive")
    mask = (
        np.ones(chunks.shape[:2], dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if mask.shape != chunks.shape[:2]:
        raise ValueError(f"valid_mask must be {chunks.shape[:2]}, got {mask.shape}")

    selected = np.arange(0, len(states), output_stride, dtype=np.int64)
    if max_frames is not None:
        selected = selected[: int(max_frames)]
    if not len(selected):
        return
    first = video_reader.read(int(frame_indices[int(selected[0])]))
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps) / output_stride,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open robot projection video writer: {path}")
    edges = kinematics.skeleton_edges()
    try:
        for output_index, step_value in enumerate(selected):
            step = int(step_value)
            rgb = (
                first
                if output_index == 0
                else video_reader.read(int(frame_indices[step]))
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            capture_poses = kinematics.link_poses(states[step])
            current = _project_landmarks(
                kinematics, calibration, states[step], capture_poses
            )
            valid_horizons = np.flatnonzero(mask[step])
            projected = [
                _project_landmarks(
                    kinematics,
                    calibration,
                    chunks[step, int(horizon)],
                    capture_poses,
                )
                for horizon in valid_horizons
            ]
            _draw_skeleton(
                bgr,
                current,
                edges,
                color=(255, 220, 0),
                thickness=5,
                alpha=1.0,
            )
            if projected:
                _draw_skeleton(
                    bgr,
                    projected[0],
                    edges,
                    color=(30, 60, 255),
                    thickness=2,
                    alpha=0.82,
                )
                _draw_future_trajectories(bgr, projected)

            cv2.rectangle(bgr, (6, 6), (width - 6, 66), (0, 0, 0), -1)
            cv2.putText(
                bgr,
                "CURRENT skeleton",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 220, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                "T-REX h=0 skeleton | wrist/fingertip trails: near -> far",
                (14, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (30, 60, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                f"projection calibration: {calibration.source}",
                (14, 63),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            writer.write(bgr)
    finally:
        writer.release()
    _make_mp4_ide_compatible(path)
