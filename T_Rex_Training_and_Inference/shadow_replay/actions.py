"""Action normalization and UniDex/North POC2.2 action adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


IDENTITY_POSE9D = np.asarray(
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    dtype=np.float32,
)


class ActionDimensionError(ValueError):
    """Raised when checkpoint, normalization, and dataset action spaces disagree."""


class ActionNormalizer:
    """Numpy implementation matching ``UniDex/src/utils/normalizers.py``."""

    def __init__(
        self,
        stats: Optional[Mapping[str, Any]] = None,
        norm_type: str = "identity",
        epsilon: float = 1e-6,
    ) -> None:
        self.stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in (stats or {}).items()
        }
        self.norm_type = str(norm_type)
        self.epsilon = float(epsilon)
        supported = {"meanstd", "std", "minmax", "identity"}
        if self.norm_type not in supported:
            raise ValueError(
                f"Unknown normalization type {self.norm_type!r}; supported={sorted(supported)}"
            )
        required = {
            "meanstd": ("mean", "std"),
            "std": ("std",),
            "minmax": ("min", "max"),
            "identity": (),
        }[self.norm_type]
        missing = [key for key in required if key not in self.stats]
        if missing:
            raise ValueError(
                f"Normalizer type {self.norm_type} is missing statistics: {missing}"
            )

    @property
    def dimension(self) -> Optional[int]:
        if not self.stats:
            return None
        return int(next(iter(self.stats.values())).shape[-1])

    def _check(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if self.dimension is not None and array.shape[-1] != self.dimension:
            raise ActionDimensionError(
                f"Normalizer expects {self.dimension} dimensions, got {array.shape[-1]}"
            )
        return array

    def normalize(self, value: np.ndarray) -> np.ndarray:
        array = self._check(value)
        if self.norm_type == "meanstd":
            return (array - self.stats["mean"]) / (self.stats["std"] + self.epsilon)
        if self.norm_type == "std":
            return array / (self.stats["std"] + self.epsilon)
        if self.norm_type == "minmax":
            minimum, maximum = self.stats["min"], self.stats["max"]
            midpoint = (maximum + minimum) / 2.0
            return (array - midpoint) / (maximum - minimum + self.epsilon) * 2.0
        return array.copy()

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        array = self._check(value)
        if self.norm_type == "meanstd":
            return array * (self.stats["std"] + self.epsilon) + self.stats["mean"]
        if self.norm_type == "std":
            return array * (self.stats["std"] + self.epsilon)
        if self.norm_type == "minmax":
            minimum, maximum = self.stats["min"], self.stats["max"]
            midpoint = (maximum + minimum) / 2.0
            return array / 2.0 * (maximum - minimum + self.epsilon) + midpoint
        return array.copy()

    @classmethod
    def from_training_config(
        cls, training_config: Mapping[str, Any], key: str
    ) -> "ActionNormalizer":
        normalizer = training_config.get("dataset", {}).get("normalizer") or {}
        stats = (normalizer.get("norm_stats") or {}).get(key)
        norm_type = (normalizer.get("norm_type") or {}).get(key, "identity")
        if stats is None and norm_type != "identity":
            raise ValueError(
                f"Training config declares {key} normalization {norm_type!r} "
                "but contains no statistics"
            )
        return cls(stats=stats, norm_type=norm_type)


@dataclass
class AdaptedAction:
    """Post-denormalized model action and its honest deployment representation."""

    robot_joint_target: np.ndarray
    eef_action: Optional[np.ndarray]
    limitations: Tuple[str, ...]


def _find_indices(names: Sequence[str], prefix: str, expected: int) -> list[int]:
    result = [index for index, name in enumerate(names) if str(name).startswith(prefix)]
    if len(result) != expected:
        raise ActionDimensionError(
            f"Expected {expected} action names starting with {prefix!r}, found {len(result)}"
        )
    return result


class NorthActionAdapter:
    """Bridge raw 65-joint North actions and the current UniDex action space.

    The repository's trained UniDex checkpoint is 82-D:

    ``right wrist pose9d, left wrist pose9d, right mapped-hand[32],
    left mapped-hand[32]``.

    The source dataset is 65-D absolute joint position control. Because there is
    no wrist-pose IK adapter in the repository, the 82->65 conversion applies
    only the predicted hand joints and holds arm/lower-body/neck joints at the
    measured observation. This limitation is returned on every conversion.
    """

    def __init__(
        self,
        dataset_action_names: Sequence[str],
        model_action_dim: int,
        adapter: str,
        hand_utils_json: Path,
    ) -> None:
        self.dataset_action_names = list(dataset_action_names)
        self.dataset_dim = len(self.dataset_action_names)
        self.model_action_dim = int(model_action_dim)
        if self.dataset_dim != 65:
            raise ActionDimensionError(
                f"North POC2.2 adapter requires 65 named dataset actions, got "
                f"{self.dataset_dim}"
            )
        if not hand_utils_json.is_file():
            raise FileNotFoundError(f"UniDex hand mapping JSON is missing: {hand_utils_json}")
        mapping = json.loads(hand_utils_json.read_text(encoding="utf-8"))
        self.mapped_joint_dim = int(mapping["mapped_joint_dim"])
        self.shadow_joint_map: Dict[str, int] = {
            str(key): int(value) for key, value in mapping["joint_map"]["Shadow"].items()
        }
        self.shadow_joint_names = list(self.shadow_joint_map)
        self.hand_joint_dim = len(self.shadow_joint_names)
        self.scale = {
            side: np.asarray(
                list(mapping["retarget_joint_map_scale"]["Shadow"][side].values()),
                dtype=np.float32,
            )
            for side in ("left", "right")
        }
        self.offset = {
            side: np.asarray(
                list(mapping["retarget_joint_map_offset"]["Shadow"][side].values()),
                dtype=np.float32,
            )
            for side in ("left", "right")
        }
        self.raw_hand_indices = {
            "left": _find_indices(self.dataset_action_names, "left_hand_", self.hand_joint_dim),
            "right": _find_indices(
                self.dataset_action_names, "right_hand_", self.hand_joint_dim
            ),
        }

        resolved = adapter
        if adapter == "auto":
            resolved = (
                "identity65"
                if self.model_action_dim == 65
                else "unidex82_to_north65"
                if self.model_action_dim == 82
                else "unsupported"
            )
        if resolved == "identity65" and self.model_action_dim != 65:
            raise ActionDimensionError(
                f"identity65 requires a 65-D model, checkpoint config says "
                f"{self.model_action_dim}"
            )
        if resolved == "unidex82_to_north65" and self.model_action_dim != 82:
            raise ActionDimensionError(
                f"unidex82_to_north65 requires an 82-D model, checkpoint config says "
                f"{self.model_action_dim}"
            )
        if resolved == "unsupported":
            raise ActionDimensionError(
                f"No safe action adapter exists for model dimension {self.model_action_dim} "
                "and North dataset dimension 65"
            )
        self.adapter = resolved

    def _map_hand(self, raw: np.ndarray, side: str) -> np.ndarray:
        transformed = raw * self.scale[side] + self.offset[side]
        mapped = np.zeros(raw.shape[:-1] + (self.mapped_joint_dim,), dtype=np.float32)
        for source_index, mapped_index in enumerate(self.shadow_joint_map.values()):
            mapped[..., mapped_index] = transformed[..., source_index]
        return mapped

    def _unmap_hand(self, mapped: np.ndarray, side: str) -> np.ndarray:
        transformed = np.stack(
            [mapped[..., mapped_index] for mapped_index in self.shadow_joint_map.values()],
            axis=-1,
        )
        scale = self.scale[side]
        if np.any(np.abs(scale) < 1e-12):
            raise ValueError(f"Cannot invert zero Shadow scale for {side} hand")
        return (transformed - self.offset[side]) / scale

    def dataset_to_model(self, action65: np.ndarray) -> np.ndarray:
        """Convert absolute 65-D labels/states to the checkpoint's native space."""
        action = np.asarray(action65, dtype=np.float32)
        if action.shape[-1] != 65:
            raise ActionDimensionError(f"Expected raw 65-D action, got {action.shape}")
        if self.adapter == "identity65":
            return action.copy()
        left = action[..., self.raw_hand_indices["left"]]
        right = action[..., self.raw_hand_indices["right"]]
        pose_shape = action.shape[:-1] + (9,)
        identity = np.broadcast_to(IDENTITY_POSE9D, pose_shape)
        return np.concatenate(
            [
                identity,
                identity,
                self._map_hand(right, "right"),
                self._map_hand(left, "left"),
            ],
            axis=-1,
        ).astype(np.float32)

    def model_to_robot(
        self,
        denormalized: np.ndarray,
        robot_state65: np.ndarray,
    ) -> AdaptedAction:
        """Adapt native predictions without hiding unsupported wrist/IK behavior."""
        prediction = np.asarray(denormalized, dtype=np.float32)
        state = np.asarray(robot_state65, dtype=np.float32)
        if prediction.shape[-1] != self.model_action_dim:
            raise ActionDimensionError(
                f"Model prediction is {prediction.shape[-1]}-D, expected "
                f"{self.model_action_dim}"
            )
        if state.shape[-1] != 65:
            raise ActionDimensionError(f"Robot state must be 65-D, got {state.shape}")
        state_prefix = np.broadcast_to(state, prediction.shape[:-1] + (65,))
        if self.adapter == "identity65":
            return AdaptedAction(
                robot_joint_target=prediction.copy(),
                eef_action=None,
                limitations=(),
            )

        target = state_prefix.copy()
        right_mapped = prediction[..., 18 : 18 + self.mapped_joint_dim]
        left_mapped = prediction[
            ..., 18 + self.mapped_joint_dim : 18 + 2 * self.mapped_joint_dim
        ]
        target[..., self.raw_hand_indices["right"]] = self._unmap_hand(
            right_mapped, "right"
        )
        target[..., self.raw_hand_indices["left"]] = self._unmap_hand(
            left_mapped, "left"
        )
        return AdaptedAction(
            robot_joint_target=target,
            eef_action=prediction[..., :18].copy(),
            limitations=(
                "unidex_wrist_pose_not_applied_no_ik",
                "arm_lower_body_neck_held_at_observed_state",
            ),
        )

    @property
    def model_representation(self) -> str:
        return (
            "absolute_joint_position_65"
            if self.adapter == "identity65"
            else "bimanual_relative_pose9d_plus_mapped_hand_joints"
        )
