#!/usr/bin/env python3
"""Refine T-Rex head-camera intrinsics/extrinsics from robot self-observation.

The fitter uses measured 65-D joint states, North URDF FK, and robot foreground
centerlines extracted from many head-left images.  It alternates nearest
centerline association (ICP) with robust nonlinear least squares and evaluates
on held-out frames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from shadow_replay.dataset import LeRobotEpisodeDataset
from shadow_replay.robot_projection import (
    CameraCalibration,
    NorthUrdfKinematics,
    transform,
)


class FrameSample(NamedTuple):
    step: int
    rgb: np.ndarray
    state: np.ndarray
    points_urdf: np.ndarray
    link_poses: dict[str, np.ndarray]
    mask: np.ndarray
    centerline_yx: np.ndarray
    tree: cKDTree


def _robot_mask(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    candidate = ((gray > 42) & (hsv[..., 1] < 150)).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    height, width = candidate.shape
    keep = np.zeros_like(candidate, dtype=bool)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_robot_entry = (
            y + component_height >= height - 2
            or (
                (x <= 2 or x + component_width >= width - 2)
                and y + component_height > height * 0.55
            )
        )
        if touches_robot_entry and area > 150:
            keep |= labels == label
    return keep


def _dense_skeleton_points(
    kinematics: NorthUrdfKinematics,
    state: np.ndarray,
    samples_per_edge: int = 5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    landmarks = kinematics.landmark_positions(state)
    points = []
    for start_name, end_name in kinematics.skeleton_edges():
        start, end = landmarks[start_name], landmarks[end_name]
        for alpha in np.linspace(0.0, 1.0, samples_per_edge, endpoint=False):
            points.append(start * (1.0 - alpha) + end * alpha)
    for name, point in landmarks.items():
        if name.endswith("_wrist") or name.endswith("_fingertip"):
            points.extend([point, point, point])
    return np.asarray(points, dtype=np.float64), landmarks


def _params_to_calibration(
    nominal: CameraCalibration,
    nominal_mount_xyz: np.ndarray,
    nominal_mount_rpy: np.ndarray,
    parameters: np.ndarray,
) -> CameraCalibration:
    matrix = nominal.matrix.copy()
    matrix[0, 0] *= math.exp(float(parameters[0]))
    matrix[1, 1] *= math.exp(float(parameters[1]))
    matrix[0, 2] += float(parameters[2])
    matrix[1, 2] += float(parameters[3])
    mount_xyz = nominal_mount_xyz + parameters[4:7]
    mount_rpy = nominal_mount_rpy + parameters[7:10]
    return replace(
        nominal,
        matrix=matrix,
        link_from_camera=transform(mount_xyz, mount_rpy),
        source="multiframe_robot_centerline_icp",
    )


def _project(
    calibration: CameraCalibration,
    sample: FrameSample,
) -> np.ndarray:
    projected, visible = calibration.project(sample.points_urdf, sample.link_poses)
    projected[~visible] = np.nan
    return projected


def _associate(
    calibration: CameraCalibration,
    samples: list[FrameSample],
    maximum_distance: float,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    associations = []
    for frame_index, sample in enumerate(samples):
        projected = _project(calibration, sample)
        finite = np.isfinite(projected).all(axis=1)
        height, width = sample.mask.shape
        in_image = (
            finite
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
        )
        source_indices = np.flatnonzero(in_image)
        if not len(source_indices):
            continue
        query_yx = projected[source_indices][:, ::-1]
        distance, target_indices = sample.tree.query(query_yx, k=1)
        accepted = distance < maximum_distance
        if np.any(accepted):
            associations.append(
                (
                    frame_index,
                    source_indices[accepted],
                    sample.centerline_yx[target_indices[accepted]][:, ::-1],
                )
            )
    return associations


def _fit_iteration(
    nominal: CameraCalibration,
    nominal_mount_xyz: np.ndarray,
    nominal_mount_rpy: np.ndarray,
    initial: np.ndarray,
    samples: list[FrameSample],
    associations: list[tuple[int, np.ndarray, np.ndarray]],
) -> np.ndarray:
    # Priors prevent the self-observation objective from trading arbitrary
    # focal-length changes against unobservable camera translations.
    prior_scale = np.asarray(
        [0.08, 0.08, 12.0, 12.0, 0.006, 0.006, 0.006, 0.018, 0.018, 0.018],
        dtype=np.float64,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        calibration = _params_to_calibration(
            nominal, nominal_mount_xyz, nominal_mount_rpy, parameters
        )
        values = []
        for frame_index, source_indices, targets in associations:
            projected = _project(calibration, samples[frame_index])[source_indices]
            delta = projected - targets
            delta[~np.isfinite(delta)] = 100.0
            values.append(delta.reshape(-1) / 3.0)
        values.append(parameters / prior_scale)
        return np.concatenate(values)

    lower = np.asarray(
        [-0.18, -0.16, -25.0, -25.0, -0.015, -0.015, -0.015, -0.05, -0.05, -0.05]
    )
    upper = -lower
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=80,
        verbose=0,
    )
    return result.x


def _metrics(
    calibration: CameraCalibration,
    samples: list[FrameSample],
) -> dict[str, float]:
    distances = []
    inside = []
    total = 0
    visible_in_image = 0
    for sample in samples:
        projected = _project(calibration, sample)
        finite = np.isfinite(projected).all(axis=1)
        height, width = sample.mask.shape
        in_image = (
            finite
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
        )
        total += len(projected)
        visible_in_image += int(in_image.sum())
        pixels = projected[in_image]
        if not len(pixels):
            continue
        distance, _ = sample.tree.query(pixels[:, ::-1], k=1)
        distances.extend(distance.tolist())
        rounded = np.rint(pixels).astype(np.int64)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        inside.extend(sample.mask[rounded[:, 1], rounded[:, 0]].tolist())
    values = np.asarray(distances, dtype=np.float64)
    return {
        "mean_centerline_distance_px": float(values.mean()),
        "median_centerline_distance_px": float(np.median(values)),
        "p90_centerline_distance_px": float(np.percentile(values, 90)),
        "robot_mask_inside_fraction": float(np.mean(inside)),
        "in_image_fraction": visible_in_image / total,
        "projected_point_count": int(len(values)),
    }


def _draw_diagnostics(
    path: Path,
    samples: list[FrameSample],
    nominal: CameraCalibration,
    fitted: CameraCalibration,
) -> None:
    panels = []
    for sample in samples[:12]:
        bgr = cv2.cvtColor(sample.rgb, cv2.COLOR_RGB2BGR)
        for calibration, color in (
            (nominal, (255, 220, 0)),
            (fitted, (30, 255, 30)),
        ):
            projected = _project(calibration, sample)
            for point in projected[::3]:
                if np.isfinite(point).all():
                    x, y = np.rint(point).astype(int)
                    if 0 <= x < bgr.shape[1] and 0 <= y < bgr.shape[0]:
                        cv2.circle(bgr, (x, y), 1, color, -1, cv2.LINE_AA)
        cv2.putText(
            bgr,
            f"step {sample.step}: nominal cyan / fitted green",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(bgr)
    columns = 3
    rows = math.ceil(len(panels) / columns)
    height, width = panels[0].shape[:2]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = panel
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=WORKSPACE / "configs/shadow_replay_trex.yaml")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=120)
    parser.add_argument("--icp-iterations", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    projection = config["visualization"]["robot_projection"]
    nominal = CameraCalibration.from_config(projection)
    mount = projection["mount"]
    nominal_mount_xyz = np.asarray(mount["xyz"], dtype=np.float64)
    nominal_mount_rpy = np.asarray(mount["rpy"], dtype=np.float64)
    urdf_path = Path(projection["urdf_path"])
    if not urdf_path.is_absolute():
        urdf_path = WORKSPACE / urdf_path
    kinematics = NorthUrdfKinematics(urdf_path)
    dataset = LeRobotEpisodeDataset(args.dataset_path)
    episode = dataset.load_episode(args.episode_id)
    steps = np.arange(0, len(episode), args.sample_stride, dtype=np.int64)
    if steps[-1] != len(episode) - 1:
        steps = np.append(steps, len(episode) - 1)

    samples = []
    with dataset.open_video(
        args.episode_id, config["visualization"]["camera_key"]
    ) as reader:
        for sample_number, step_value in enumerate(steps):
            step = int(step_value)
            rgb = reader.read(int(episode.frame_indices[step]))
            mask = _robot_mask(rgb)
            centerline = skeletonize(mask)
            centerline_yx = np.argwhere(centerline)
            if len(centerline_yx) < 100:
                continue
            points, _ = _dense_skeleton_points(kinematics, episode.states[step])
            samples.append(
                FrameSample(
                    step=step,
                    rgb=rgb,
                    state=episode.states[step],
                    points_urdf=points,
                    link_poses=kinematics.link_poses(episode.states[step]),
                    mask=mask,
                    centerline_yx=centerline_yx,
                    tree=cKDTree(centerline_yx),
                )
            )
            if (sample_number + 1) % 20 == 0:
                print(f"prepared {sample_number + 1}/{len(steps)} frames", flush=True)

    training = samples[::2]
    validation = samples[1::2]
    parameters = np.zeros(10, dtype=np.float64)
    fitted = nominal
    best_parameters = parameters.copy()
    best_validation = _metrics(nominal, validation)
    iteration_history = []
    for iteration in range(args.icp_iterations):
        maximum_distance = max(10.0, 32.0 - iteration * 5.0)
        associations = _associate(fitted, training, maximum_distance)
        pair_count = sum(len(item[1]) for item in associations)
        if pair_count < 500:
            raise RuntimeError(f"Too few ICP correspondences: {pair_count}")
        parameters = _fit_iteration(
            nominal,
            nominal_mount_xyz,
            nominal_mount_rpy,
            parameters,
            training,
            associations,
        )
        fitted = _params_to_calibration(
            nominal, nominal_mount_xyz, nominal_mount_rpy, parameters
        )
        train_metrics = _metrics(fitted, training)
        validation_metrics = _metrics(fitted, validation)
        iteration_history.append(
            {
                "iteration": iteration + 1,
                "pair_count": pair_count,
                "parameters": parameters.tolist(),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            }
        )
        if (
            validation_metrics["median_centerline_distance_px"]
            < best_validation["median_centerline_distance_px"]
        ):
            best_validation = validation_metrics
            best_parameters = parameters.copy()
        print(
            f"ICP {iteration + 1}/{args.icp_iterations}: "
            f"pairs={pair_count} "
            f"val_median={validation_metrics['median_centerline_distance_px']:.3f}px "
            f"params={parameters.tolist()}",
            flush=True,
        )

    parameters = best_parameters
    fitted = _params_to_calibration(
        nominal, nominal_mount_xyz, nominal_mount_rpy, parameters
    )
    nominal_train = _metrics(nominal, training)
    fitted_train = _metrics(fitted, training)
    nominal_validation = _metrics(nominal, validation)
    fitted_validation = _metrics(fitted, validation)
    fitted_xyz = nominal_mount_xyz + parameters[4:7]
    fitted_rpy = nominal_mount_rpy + parameters[7:10]
    result = {
        "method": "multiframe_robot_centerline_icp",
        "episode_id": args.episode_id,
        "sample_stride": args.sample_stride,
        "training_frames": [sample.step for sample in training],
        "validation_frames": [sample.step for sample in validation],
        "nominal": {
            "matrix": nominal.matrix.tolist(),
            "mount_xyz": nominal_mount_xyz.tolist(),
            "mount_rpy": nominal_mount_rpy.tolist(),
            "train_metrics": nominal_train,
            "validation_metrics": nominal_validation,
        },
        "fitted": {
            "matrix": fitted.matrix.tolist(),
            "distortion": fitted.distortion.tolist(),
            "mount_xyz": fitted_xyz.tolist(),
            "mount_rpy": fitted_rpy.tolist(),
            "train_metrics": fitted_train,
            "validation_metrics": fitted_validation,
        },
        "parameter_delta": parameters.tolist(),
        "iteration_history": iteration_history,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_fit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    calibration_yaml = {
        "source": "multiframe_robot_centerline_icp_episode8",
        "matrix": fitted.matrix.tolist(),
        "distortion": fitted.distortion.tolist(),
        "mount": {
            "link": fitted.link,
            "xyz": fitted_xyz.tolist(),
            "rpy": fitted_rpy.tolist(),
        },
    }
    (output_dir / "calibration_fitted.yaml").write_text(
        yaml.safe_dump(calibration_yaml, sort_keys=False),
        encoding="utf-8",
    )
    _draw_diagnostics(
        output_dir / "calibration_validation_contact_sheet.png",
        validation,
        nominal,
        fitted,
    )
    print(json.dumps(result["fitted"], indent=2), flush=True)


if __name__ == "__main__":
    main()
