"""Configurable, non-actuating safety checks for predicted actions."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


_HAND_JOINT_SUFFIXES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)


def north_dataset_joint_names() -> list[str]:
    names = [f"left_arm_j{i}" for i in range(7)]
    names += [f"left_hand_j{i}" for i in range(22)]
    names += [f"right_arm_j{i}" for i in range(7)]
    names += [f"right_hand_j{i}" for i in range(22)]
    names += [f"motor_j{i}" for i in range(7)]
    return names


def north_dataset_to_urdf_names() -> list[str]:
    names = [f"left_arm_joint_{index}" for index in range(1, 8)]
    names += [f"left_{suffix}" for suffix in _HAND_JOINT_SUFFIXES]
    names += [f"right_arm_joint_{index}" for index in range(1, 8)]
    names += [f"right_{suffix}" for suffix in _HAND_JOINT_SUFFIXES]
    names += [f"lower_body_joint_{index}" for index in range(1, 6)]
    names += ["neck_joint_1", "neck_joint_2"]
    return names


@dataclass(frozen=True)
class JointLimits:
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray


def load_north_urdf_limits(
    urdf_path: Path, dataset_names: Optional[Sequence[str]] = None
) -> JointLimits:
    """Parse only lightweight joint limits; no new robotics dependency is required."""
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    parsed: Dict[str, tuple[float, float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            continue
        parsed[str(joint.get("name"))] = (
            float(limit.get("lower")),
            float(limit.get("upper")),
            float(limit.get("velocity", "inf")),
        )
    expected_dataset_names = list(dataset_names or north_dataset_joint_names())
    if expected_dataset_names != north_dataset_joint_names():
        raise ValueError(
            "URDF mapping requires the canonical North 65-D dataset action ordering"
        )
    urdf_names = north_dataset_to_urdf_names()
    missing = sorted(set(urdf_names) - set(parsed))
    if missing:
        raise ValueError(f"URDF is missing mapped movable joints: {missing}")
    values = [parsed[name] for name in urdf_names]
    return JointLimits(
        names=tuple(urdf_names),
        lower=np.asarray([value[0] for value in values], dtype=np.float32),
        upper=np.asarray([value[1] for value in values], dtype=np.float32),
        velocity=np.asarray([value[2] for value in values], dtype=np.float32),
    )


@dataclass
class SafetyViolation:
    violation_type: str
    original_value: Any
    limit: Any
    action_rejected: bool = False
    action_clipped: bool = False
    indices: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "original_value": self.original_value,
            "limit": self.limit,
            "action_rejected": self.action_rejected,
            "action_clipped": self.action_clipped,
            "indices": self.indices,
        }


@dataclass
class SafetyCheckResult:
    action_safe: np.ndarray
    valid: bool
    rejected: bool
    clipped: bool
    violations: List[SafetyViolation] = field(default_factory=list)
    ik_success: Optional[bool] = None
    near_joint_limit: bool = False
    nan_or_inf: bool = False
    unavailable_checks: List[str] = field(default_factory=list)

    @property
    def flags(self) -> list[str]:
        flags = [violation.violation_type for violation in self.violations]
        flags.extend(f"unavailable:{name}" for name in self.unavailable_checks)
        return sorted(set(flags))


def _rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = value[..., :3]
    second = value[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first_unit = first / np.maximum(first_norm, 1e-12)
    second = second - np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    second_unit = second / np.maximum(second_norm, 1e-12)
    third = np.cross(first_unit, second_unit)
    return np.stack([first_unit, second_unit, third], axis=-2)


def _rotation_angle_deg(rot6d: np.ndarray) -> float:
    matrix = _rot6d_to_matrix(rot6d)
    cosine = np.clip((np.trace(matrix, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class ActionSafetyChecker:
    """Apply configurable safety policy and record every mutation or rejection."""

    def __init__(
        self,
        config: Mapping[str, Any],
        joint_limits: Optional[JointLimits] = None,
    ) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.behavior = str(self.config.get("violation_behavior", "reject"))
        self.joint_limits = joint_limits
        self.near_limit_margin_ratio = float(
            self.config.get("near_joint_limit_margin_ratio", 0.05)
        )

    def _mutation_flags(self) -> tuple[bool, bool]:
        return self.behavior == "reject", self.behavior == "clip"

    def _violation(
        self,
        violation_type: str,
        original_value: Any,
        limit: Any,
        indices: Optional[np.ndarray] = None,
        force_reject: bool = False,
        clip_supported: bool = True,
    ) -> SafetyViolation:
        rejected, clipped = self._mutation_flags()
        if clipped and not clip_supported:
            rejected, clipped = True, False
        if force_reject:
            rejected, clipped = True, False
        return SafetyViolation(
            violation_type=violation_type,
            original_value=original_value,
            limit=limit,
            action_rejected=rejected,
            action_clipped=clipped,
            indices=None if indices is None else [int(value) for value in indices],
        )

    def check(
        self,
        action: np.ndarray,
        robot_state: np.ndarray,
        *,
        previous_action: Optional[np.ndarray] = None,
        previous_velocity: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        eef_action: Optional[np.ndarray] = None,
        urdf_model: Any = None,
    ) -> SafetyCheckResult:
        """Check a single 65-D absolute joint target.

        ``urdf_model`` is reserved for a future full kinematics/collision service;
        parsing position/velocity limits is handled by :func:`load_north_urdf_limits`.
        """
        target = np.asarray(action, dtype=np.float32).copy()
        state = np.asarray(robot_state, dtype=np.float32)
        if target.shape != (65,) or state.shape != (65,):
            raise ValueError(
                f"Safety checker requires action/state shape (65,), got "
                f"{target.shape}/{state.shape}"
            )
        if not self.enabled:
            return SafetyCheckResult(action_safe=target, valid=True, rejected=False, clipped=False)

        violations: List[SafetyViolation] = []
        unavailable: List[str] = []
        finite = np.isfinite(target)
        nan_or_inf = not bool(finite.all())
        if nan_or_inf:
            bad = np.flatnonzero(~finite)
            violations.append(
                self._violation(
                    "nan_or_inf",
                    [str(target[index]) for index in bad],
                    "finite",
                    bad,
                    force_reject=True,
                )
            )

        if self.joint_limits is not None and finite.all():
            below = target < self.joint_limits.lower
            above = target > self.joint_limits.upper
            outside = below | above
            if outside.any():
                indices = np.flatnonzero(outside)
                violations.append(
                    self._violation(
                        "joint_limit",
                        target[indices].tolist(),
                        {
                            "lower": self.joint_limits.lower[indices].tolist(),
                            "upper": self.joint_limits.upper[indices].tolist(),
                        },
                        indices,
                    )
                )
                if self.behavior == "clip":
                    target = np.clip(
                        target, self.joint_limits.lower, self.joint_limits.upper
                    )
            span = self.joint_limits.upper - self.joint_limits.lower
            margin = span * self.near_limit_margin_ratio
            near = (target - self.joint_limits.lower <= margin) | (
                self.joint_limits.upper - target <= margin
            )
            near_joint_limit = bool(near.any())
        else:
            near_joint_limit = False
            if self.joint_limits is None:
                unavailable.append("urdf_joint_limits")

        # ``previous_action`` means the previous command from the same command
        # trajectory.  Measured state is deliberately not substituted here:
        # target-state error is a servo tracking error, not a one-frame command
        # velocity.  When no previous command is available (chunk horizon 0),
        # command step/velocity/acceleration checks are unavailable.
        reference = (
            None
            if previous_action is None
            else np.asarray(previous_action, dtype=np.float32)
        )
        delta = None if reference is None else target - reference
        max_step = self.config.get("max_joint_step")
        if (
            delta is not None
            and max_step is not None
            and np.isfinite(delta).all()
        ):
            excessive = np.abs(delta) > float(max_step)
            if excessive.any():
                indices = np.flatnonzero(excessive)
                violations.append(
                    self._violation(
                        "max_joint_step",
                        delta[indices].tolist(),
                        float(max_step),
                        indices,
                    )
                )
                if self.behavior == "clip":
                    target = reference + np.clip(delta, -float(max_step), float(max_step))
                    delta = target - reference

        velocity: Optional[np.ndarray] = None
        if (
            delta is not None
            and dt is not None
            and dt > 0
            and np.isfinite(delta).all()
        ):
            velocity = delta / float(dt)
            configured_velocity = self.config.get("max_joint_velocity")
            if configured_velocity is not None:
                velocity_limit = np.full(65, float(configured_velocity), dtype=np.float32)
            elif self.joint_limits is not None:
                velocity_limit = self.joint_limits.velocity
            else:
                velocity_limit = None
            if velocity_limit is not None:
                excessive = np.abs(velocity) > velocity_limit
                if excessive.any():
                    indices = np.flatnonzero(excessive)
                    violations.append(
                        self._violation(
                            "max_joint_velocity",
                            velocity[indices].tolist(),
                            velocity_limit[indices].tolist(),
                            indices,
                        )
                    )
                    if self.behavior == "clip":
                        clipped_velocity = np.clip(
                            velocity, -velocity_limit, velocity_limit
                        )
                        target = reference + clipped_velocity * float(dt)
                        velocity = clipped_velocity
            acceleration_limit = self.config.get("max_joint_acceleration")
            if acceleration_limit is not None and previous_velocity is not None:
                acceleration = (
                    velocity - np.asarray(previous_velocity, dtype=np.float32)
                ) / float(dt)
                excessive = np.abs(acceleration) > float(acceleration_limit)
                if excessive.any():
                    indices = np.flatnonzero(excessive)
                    violations.append(
                        self._violation(
                            "max_joint_acceleration",
                            acceleration[indices].tolist(),
                            float(acceleration_limit),
                            indices,
                        )
                    )
                    if self.behavior == "clip":
                        clipped_acceleration = np.clip(
                            acceleration,
                            -float(acceleration_limit),
                            float(acceleration_limit),
                        )
                        velocity = (
                            np.asarray(previous_velocity, dtype=np.float32)
                            + clipped_acceleration * float(dt)
                        )
                        target = reference + velocity * float(dt)

        if eef_action is not None:
            eef = np.asarray(eef_action, dtype=np.float32)
            if eef.shape != (18,):
                raise ValueError(f"eef_action must be 18-D pose9d pair, got {eef.shape}")
            xyz = np.stack([eef[0:3], eef[9:12]])
            workspace_min = self.config.get("workspace_min")
            workspace_max = self.config.get("workspace_max")
            if workspace_min is not None and workspace_max is not None:
                low, high = np.asarray(workspace_min), np.asarray(workspace_max)
                outside = np.any((xyz < low) | (xyz > high), axis=1)
                if outside.any():
                    violations.append(
                        self._violation(
                            "workspace",
                            xyz[outside].tolist(),
                            {"min": low.tolist(), "max": high.tolist()},
                            clip_supported=False,
                        )
                    )
            max_cartesian = self.config.get("max_cartesian_step")
            if max_cartesian is not None:
                distances = np.linalg.norm(xyz, axis=-1)
                outside = distances > float(max_cartesian)
                if outside.any():
                    violations.append(
                        self._violation(
                            "max_cartesian_step",
                            distances[outside].tolist(),
                            float(max_cartesian),
                            clip_supported=False,
                        )
                    )
            max_rotation = self.config.get("max_rotation_step_deg")
            if max_rotation is not None:
                angles = np.asarray(
                    [_rotation_angle_deg(eef[3:9]), _rotation_angle_deg(eef[12:18])]
                )
                outside = angles > float(max_rotation)
                if outside.any():
                    violations.append(
                        self._violation(
                            "max_rotation_step",
                            angles[outside].tolist(),
                            float(max_rotation),
                            clip_supported=False,
                        )
                    )

        for enabled_key, check_name in (
            ("check_ik", "ik"),
            ("check_singularity", "singularity"),
            ("check_self_collision", "self_collision"),
            ("check_environment_collision", "environment_collision"),
        ):
            if self.config.get(enabled_key, False):
                unavailable.append(check_name)
        if urdf_model is not None:
            unavailable.append("external_urdf_model_adapter_not_registered")

        rejected = nan_or_inf or any(item.action_rejected for item in violations)
        clipped = any(item.action_clipped for item in violations)
        if rejected:
            target = state.copy()
        valid = not violations
        return SafetyCheckResult(
            action_safe=target,
            valid=valid,
            rejected=rejected,
            clipped=clipped,
            violations=violations,
            ik_success=None,
            near_joint_limit=near_joint_limit,
            nan_or_inf=nan_or_inf,
            unavailable_checks=unavailable,
        )
