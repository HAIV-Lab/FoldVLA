"""Metrics for action chunks, rotations, dexterity, smoothness, and latency."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .dataset import build_valid_chunk_mask


def _json_float(value: Any) -> Optional[float]:
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def quaternion_geodesic_deg(
    predicted: np.ndarray,
    target: np.ndarray,
    order: str = "xyzw",
) -> np.ndarray:
    """Quaternion angular error in degrees, invariant to q/-q."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.shape[-1] != 4:
        raise ValueError(f"Quaternion arrays must share shape [...,4], got {pred.shape}/{truth.shape}")
    if order not in {"xyzw", "wxyz"}:
        raise ValueError(f"Quaternion order must be xyzw or wxyz, got {order!r}")
    pred = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-12)
    truth = truth / np.maximum(np.linalg.norm(truth, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(pred * truth, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def rotation6d_to_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError(f"rotation6d must end in 6 dimensions, got {value.shape}")
    first, second = value[..., :3], value[..., 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-2)


def so3_geodesic_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """SO(3) geodesic distance for matrices ending in [3,3]."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices must share shape [...,3,3], got {pred.shape}/{truth.shape}")
    relative = pred @ np.swapaxes(truth, -1, -2)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rotation6d_geodesic_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return so3_geodesic_deg(rotation6d_to_matrix(predicted), rotation6d_to_matrix(target))


def build_ground_truth_chunks(actions: np.ndarray, chunk_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Build future labels and a mask without crossing an episode boundary."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"actions must be [T,D], got {values.shape}")
    mask = build_valid_chunk_mask(len(values), chunk_length)
    chunks = np.zeros((len(values), chunk_length, values.shape[1]), dtype=np.float32)
    for horizon in range(chunk_length):
        valid_length = max(0, len(values) - horizon)
        if valid_length:
            chunks[:valid_length, horizon] = values[horizon:]
    return chunks, mask


def _masked_errors(
    predicted: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if predicted.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {predicted.shape}/{target.shape}")
    if predicted.ndim != 3:
        raise ValueError(f"Action chunks must be [T,H,D], got {predicted.shape}")
    if mask.shape != predicted.shape[:2]:
        raise ValueError(f"Mask must be {predicted.shape[:2]}, got {mask.shape}")
    valid = np.asarray(mask, dtype=bool)
    if not valid.any():
        return np.empty((0, predicted.shape[-1])), np.empty((0, predicted.shape[-1]))
    delta = predicted - target
    return np.abs(delta[valid]), np.square(delta[valid])


def action_error_metrics(
    predicted: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> Dict[str, Any]:
    absolute, squared = _masked_errors(predicted, target, mask)
    if absolute.size == 0:
        return {"valid_action_count": 0, "action_mae": None, "action_mse": None}
    endpoint_errors = []
    for step in range(len(predicted)):
        valid_horizons = np.flatnonzero(mask[step])
        if valid_horizons.size:
            horizon = int(valid_horizons[-1])
            endpoint_errors.append(np.abs(predicted[step, horizon] - target[step, horizon]))
    endpoint = np.asarray(endpoint_errors)
    horizon_mae = []
    for horizon in range(predicted.shape[1]):
        valid = mask[:, horizon]
        horizon_mae.append(
            _json_float(np.mean(np.abs(predicted[valid, horizon] - target[valid, horizon])))
            if valid.any()
            else None
        )
    return {
        "valid_action_count": int(mask.sum()),
        "action_mae": _json_float(absolute.mean()),
        "action_mse": _json_float(squared.mean()),
        "per_dimension_mae": [_json_float(value) for value in absolute.mean(axis=0)],
        "action_chunk_mean_error": _json_float(absolute.mean()),
        "action_chunk_endpoint_mae": _json_float(endpoint.mean()),
        "horizon_mae": horizon_mae,
    }


def position_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    xyz_indices: Sequence[int],
    unit: str,
) -> Dict[str, Any]:
    indices = list(xyz_indices)
    if not indices:
        return {}
    if len(indices) % 3:
        raise ValueError("xyz_indices must contain complete XYZ triplets")
    if max(indices) >= predicted.shape[-1]:
        raise ValueError("xyz_indices exceed action dimension")
    pred = predicted[..., indices].reshape(predicted.shape[:2] + (-1, 3))
    truth = target[..., indices].reshape(target.shape[:2] + (-1, 3))
    valid_pred, valid_truth = pred[mask], truth[mask]
    delta = valid_pred - valid_truth
    euclidean = np.linalg.norm(delta, axis=-1)
    pred_norm = np.linalg.norm(valid_pred, axis=-1)
    truth_norm = np.linalg.norm(valid_truth, axis=-1)
    denominator = pred_norm * truth_norm
    direction_valid = denominator > 1e-12
    cosine = np.divide(
        np.sum(valid_pred * valid_truth, axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        where=direction_valid,
    )
    endpoint_distance = []
    for step in range(len(predicted)):
        valid_horizons = np.flatnonzero(mask[step])
        if valid_horizons.size:
            horizon = int(valid_horizons[-1])
            endpoint_distance.extend(
                np.linalg.norm(pred[step, horizon] - truth[step, horizon], axis=-1)
            )
    result: Dict[str, Any] = {
        "xyz_mae": [_json_float(value) for value in np.abs(delta).mean(axis=(0, 1))],
        "xyz_euclidean_mean": _json_float(euclidean.mean()),
        "xyz_endpoint_euclidean_mean": _json_float(np.mean(endpoint_distance)),
        "direction_cosine_similarity": (
            _json_float(cosine[direction_valid].mean())
            if direction_valid.any()
            else None
        ),
        "opposite_direction_ratio": (
            _json_float(np.mean(cosine[direction_valid] < 0.0))
            if direction_valid.any()
            else None
        ),
        "position_unit": unit,
    }
    if unit == "m":
        result["xyz_euclidean_mean_cm"] = _json_float(euclidean.mean() * 100.0)
        result["xyz_endpoint_euclidean_mean_cm"] = _json_float(
            np.mean(endpoint_distance) * 100.0
        )
    return result


def rotation_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    rotation_indices: Sequence[int],
    representation: str,
    quaternion_order: str = "xyzw",
) -> Dict[str, Any]:
    indices = list(rotation_indices)
    if not indices:
        return {}
    if max(indices) >= predicted.shape[-1]:
        raise ValueError("rotation_indices exceed action dimension")
    pred = predicted[..., indices][mask]
    truth = target[..., indices][mask]
    if representation == "rotation6d":
        if len(indices) % 6:
            raise ValueError("rotation6d indices must form groups of 6")
        pred = pred.reshape((-1, len(indices) // 6, 6))
        truth = truth.reshape((-1, len(indices) // 6, 6))
        angular = rotation6d_geodesic_deg(pred, truth)
    elif representation == "quaternion":
        if len(indices) % 4:
            raise ValueError("quaternion indices must form groups of 4")
        pred = pred.reshape((-1, len(indices) // 4, 4))
        truth = truth.reshape((-1, len(indices) // 4, 4))
        angular = quaternion_geodesic_deg(pred, truth, order=quaternion_order)
    elif representation == "matrix":
        if len(indices) % 9:
            raise ValueError("matrix indices must form groups of 9")
        angular = so3_geodesic_deg(
            pred.reshape((-1, len(indices) // 9, 3, 3)),
            truth.reshape((-1, len(indices) // 9, 3, 3)),
        )
    elif representation == "euler":
        if len(indices) % 3:
            raise ValueError("Euler indices must form XYZ triplets")
        wrapped = (pred - truth + np.pi) % (2 * np.pi) - np.pi
        return {
            "rotation_representation": "euler",
            "euler_axis_mae_deg": [
                _json_float(value)
                for value in np.degrees(np.abs(wrapped)).reshape(-1, 3).mean(axis=0)
            ],
            "warning": "Euler errors are periodic and representation-dependent.",
        }
    else:
        raise ValueError(f"Unsupported rotation representation: {representation}")
    return {
        "rotation_representation": representation,
        "rotation_geodesic_mean_deg": _json_float(angular.mean()),
        "rotation_geodesic_median_deg": _json_float(np.median(angular)),
        "rotation_geodesic_p95_deg": _json_float(np.percentile(angular, 95)),
    }


def gripper_classification_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Binary open/close metrics and transition timing for one or more channels."""
    pred = np.asarray(predicted)
    truth = np.asarray(target)
    if pred.shape != truth.shape:
        raise ValueError(f"Gripper shape mismatch: {pred.shape}/{truth.shape}")
    if pred.ndim == 1:
        pred = pred[:, None]
        truth = truth[:, None]
    pred_binary = pred >= threshold
    truth_binary = truth >= threshold
    tp = int(np.sum(pred_binary & truth_binary))
    fp = int(np.sum(pred_binary & ~truth_binary))
    fn = int(np.sum(~pred_binary & truth_binary))
    accuracy = float(np.mean(pred_binary == truth_binary)) if pred_binary.size else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    timing_errors: list[int] = []
    early_close = late_close = early_open = late_open = 0
    for channel in range(pred.shape[1]):
        pred_switch = np.flatnonzero(np.diff(pred_binary[:, channel].astype(np.int8)) != 0) + 1
        truth_switch = np.flatnonzero(np.diff(truth_binary[:, channel].astype(np.int8)) != 0) + 1
        unused = list(int(value) for value in pred_switch)
        for event in truth_switch:
            if not unused:
                break
            nearest_index = int(np.argmin(np.abs(np.asarray(unused) - event)))
            predicted_event = unused.pop(nearest_index)
            error = predicted_event - int(event)
            timing_errors.append(error)
            closes = bool(truth_binary[event, channel])
            if closes and error < 0:
                early_close += 1
            elif closes and error > 0:
                late_close += 1
            elif not closes and error < 0:
                early_open += 1
            elif not closes and error > 0:
                late_open += 1
    return {
        "gripper_accuracy": _json_float(accuracy),
        "gripper_precision": _json_float(precision),
        "gripper_recall": _json_float(recall),
        "gripper_f1": _json_float(f1),
        "gripper_switch_timing_mae_frames": (
            _json_float(np.mean(np.abs(timing_errors))) if timing_errors else None
        ),
        "early_close_count": early_close,
        "late_close_count": late_close,
        "early_release_count": early_open,
        "late_release_count": late_open,
    }


def chunk_consistency_metrics(predicted: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Compare overlapping forecasts of the same future time point."""
    prediction = np.asarray(predicted, dtype=np.float64)
    if prediction.ndim != 3 or mask.shape != prediction.shape[:2]:
        raise ValueError("Expected predicted [T,H,D] and mask [T,H]")
    errors = []
    per_horizon: Dict[int, list[float]] = defaultdict(list)
    for step in range(len(prediction)):
        for horizon in range(1, prediction.shape[1]):
            future = step + horizon
            if future >= len(prediction) or not mask[step, horizon]:
                continue
            error = float(np.mean(np.abs(prediction[step, horizon] - prediction[future, 0])))
            errors.append(error)
            per_horizon[horizon].append(error)
    boundary = []
    horizon = prediction.shape[1]
    for step in range(0, max(0, len(prediction) - horizon)):
        if mask[step, -1]:
            boundary.append(
                float(
                    np.mean(
                        np.abs(
                            prediction[step, -1]
                            - prediction[min(step + horizon, len(prediction) - 1), 0]
                        )
                    )
                )
            )
    return {
        "overlap_consistency_mae": _json_float(np.mean(errors)) if errors else None,
        "overlap_comparison_count": len(errors),
        "overlap_consistency_by_horizon": {
            str(key): _json_float(np.mean(value)) for key, value in per_horizon.items()
        },
        "chunk_boundary_discontinuity_mae": (
            _json_float(np.mean(boundary)) if boundary else None
        ),
    }


def smoothness_metrics(predicted: np.ndarray, fps: float) -> Dict[str, Any]:
    """First-action temporal derivatives plus within-chunk displacement."""
    prediction = np.asarray(predicted, dtype=np.float64)
    if prediction.ndim != 3:
        raise ValueError(f"predicted must be [T,H,D], got {prediction.shape}")
    sequence = prediction[:, 0]
    change = np.diff(sequence, axis=0)
    velocity = change * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    chunk_change = np.diff(prediction, axis=1)
    chunk_norm = np.linalg.norm(chunk_change, axis=-1)

    def mean_norm(value: np.ndarray) -> Optional[float]:
        return _json_float(np.mean(np.linalg.norm(value, axis=-1))) if len(value) else None

    return {
        "adjacent_action_change_l2": mean_norm(change),
        "predicted_velocity_l2": mean_norm(velocity),
        "predicted_acceleration_l2": mean_norm(acceleration),
        "predicted_jerk_l2": mean_norm(jerk),
        "chunk_max_single_step_l2": (
            _json_float(np.max(chunk_norm)) if chunk_norm.size else None
        ),
    }


def trajectory_derivative_metrics(
    trajectory: np.ndarray,
    mask: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    """Compute within-chunk derivatives for one trajectory source only.

    The caller must invoke this independently for VLA predictions and ground
    truth.  In particular, no ground-truth derivative is used as the initial
    velocity of a predicted chunk (or vice versa).
    """
    values = np.asarray(trajectory, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError(
            f"trajectory/mask must be [T,H,D]/[T,H], got {values.shape}/{valid.shape}"
        )
    if fps <= 0:
        raise ValueError("fps must be positive")

    velocity_mask = valid[:, 1:] & valid[:, :-1]
    velocity = np.diff(values, axis=1) * float(fps)
    valid_velocity = velocity[velocity_mask]
    acceleration_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    acceleration = np.diff(velocity, axis=1) * float(fps)
    valid_acceleration = acceleration[acceleration_mask]

    def summarize(samples: np.ndarray) -> Dict[str, Any]:
        if not samples.size:
            return {"sample_count": 0}
        absolute = np.abs(samples)
        return {
            "sample_count": int(len(samples)),
            "mean_abs": _json_float(absolute.mean()),
            "p95_abs": _json_float(np.percentile(absolute, 95)),
            "max_abs": _json_float(absolute.max()),
            "mean_l2": _json_float(np.linalg.norm(samples, axis=-1).mean()),
            "per_dimension_mean_abs": [
                _json_float(value) for value in absolute.mean(axis=0)
            ],
        }

    return {
        "velocity_rad_s": summarize(valid_velocity),
        "acceleration_rad_s2": summarize(valid_acceleration),
    }


def evaluate_dataset_clipped_predictions(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    fps: float,
) -> Dict[str, Any]:
    """Compare prediction and GT after clipping both to empirical data bounds."""
    pred = np.asarray(predicted)
    truth = np.asarray(target)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError(f"Expected matching [T,H,D] arrays, got {pred.shape}/{truth.shape}")
    if low.shape != (pred.shape[-1],) or high.shape != low.shape:
        raise ValueError(
            f"Dataset bounds must have shape ({pred.shape[-1]},), got {low.shape}/{high.shape}"
        )
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)) or np.any(low > high):
        raise ValueError("Dataset bounds must be finite and lower <= upper")
    clipped_pred = np.clip(pred, low, high)
    clipped_truth = np.clip(truth, low, high)
    result = evaluate_action_predictions(
        clipped_pred,
        clipped_truth,
        mask,
        fps=fps,
        position_unit="rad",
    )
    valid = np.asarray(mask, dtype=bool)
    result["dataset_bound_clipping"] = {
        "bound_source": "dataset action min/max",
        "prediction_clipped_element_count": int(
            np.sum(((pred < low) | (pred > high)) & valid[..., None])
        ),
        "ground_truth_clipped_element_count": int(
            np.sum(((truth < low) | (truth > high)) & valid[..., None])
        ),
        "lower": low.tolist(),
        "upper": high.tolist(),
    }
    return result


def evaluate_action_predictions(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    fps: float,
    xyz_indices: Optional[Sequence[int]] = None,
    rotation_indices: Optional[Sequence[int]] = None,
    rotation_representation: Optional[str] = None,
    quaternion_order: str = "xyzw",
    gripper_indices: Optional[Sequence[int]] = None,
    gripper_threshold: float = 0.5,
    position_unit: str = "rad",
) -> Dict[str, Any]:
    """Compute all metrics that are valid for the configured action semantics."""
    result = action_error_metrics(predicted, target, mask)
    if xyz_indices:
        result["position"] = position_metrics(
            predicted, target, mask, xyz_indices, position_unit
        )
    if rotation_indices and rotation_representation:
        result["rotation"] = rotation_metrics(
            predicted,
            target,
            mask,
            rotation_indices,
            rotation_representation,
            quaternion_order,
        )
    if gripper_indices:
        result["gripper"] = gripper_classification_metrics(
            predicted[:, 0, list(gripper_indices)],
            target[:, 0, list(gripper_indices)],
            gripper_threshold,
        )
    result["smoothness"] = smoothness_metrics(predicted, fps)
    result["trajectory_derivatives"] = {
        "vla_prediction": trajectory_derivative_metrics(predicted, mask, fps),
        "ground_truth": trajectory_derivative_metrics(target, mask, fps),
    }
    result["chunk_consistency"] = chunk_consistency_metrics(predicted, mask)
    return result


def latency_statistics(values_ms: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(values_ms, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean_ms": _json_float(np.mean(values)),
        "median_ms": _json_float(np.median(values)),
        "p90_ms": _json_float(np.percentile(values, 90)),
        "p95_ms": _json_float(np.percentile(values, 95)),
        "p99_ms": _json_float(np.percentile(values, 99)),
    }


def stage_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    stages: Sequence[Optional[str]],
    *,
    fps: float,
    safety_valid: Optional[Sequence[bool]] = None,
    latency_ms: Optional[Sequence[float]] = None,
    xyz_indices: Optional[Sequence[int]] = None,
    rotation_indices: Optional[Sequence[int]] = None,
    rotation_representation: Optional[str] = None,
    quaternion_order: str = "xyzw",
    gripper_indices: Optional[Sequence[int]] = None,
    gripper_threshold: float = 0.5,
    position_unit: str = "rad",
) -> Dict[str, Dict[str, Any]]:
    """Aggregate trusted stage/event intervals without inventing labels."""
    if len(stages) != len(predicted):
        raise ValueError("Stage labels must have one value per prediction step")
    result: Dict[str, Dict[str, Any]] = {}
    for stage in sorted(set(value for value in stages if value is not None)):
        selected = np.asarray([value == stage for value in stages])
        stage_mask = mask & selected[:, None]
        metrics = action_error_metrics(predicted, target, stage_mask)
        metrics["sample_count"] = int(selected.sum())
        if xyz_indices:
            metrics["position"] = position_metrics(
                predicted, target, stage_mask, xyz_indices, position_unit
            )
        if rotation_indices and rotation_representation:
            metrics["rotation"] = rotation_metrics(
                predicted,
                target,
                stage_mask,
                rotation_indices,
                rotation_representation,
                quaternion_order,
            )
        if gripper_indices:
            metrics["gripper"] = gripper_classification_metrics(
                predicted[selected, 0][:, list(gripper_indices)],
                target[selected, 0][:, list(gripper_indices)],
                gripper_threshold,
            )
        consistency = chunk_consistency_metrics(predicted, stage_mask)
        metrics["overlap_consistency_mae"] = consistency[
            "overlap_consistency_mae"
        ]
        metrics["chunk_boundary_discontinuity_mae"] = consistency[
            "chunk_boundary_discontinuity_mae"
        ]
        if safety_valid is not None:
            valid = np.asarray(safety_valid, dtype=bool)
            metrics["safety_violation_count"] = int(np.sum(~valid[selected]))
        if latency_ms is not None:
            metrics["latency"] = latency_statistics(np.asarray(latency_ms)[selected])
        result[str(stage)] = metrics
    return result
