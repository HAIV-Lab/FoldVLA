"""Independent observation perturbations for real-visual robustness analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    kind: str
    value: Any


def _as_values(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def build_perturbation_specs(config: Mapping[str, Any]) -> List[PerturbationSpec]:
    """Expand config into separate runs; perturbations are never stacked."""
    if not config.get("enabled", False):
        return []
    specs: List[PerturbationSpec] = []
    for key in (
        "brightness",
        "contrast",
        "gaussian_noise_std",
        "blur_kernel",
        "occlusion_fraction",
        "robot_state_noise_std",
        "drop_frame_probability",
        "camera_interval_jitter_frames",
    ):
        for value in _as_values(config.get(key)):
            if value not in (None, 0, 0.0):
                specs.append(PerturbationSpec(f"{key}={value}", key, value))
    maximum_delay = int(config.get("observation_delay_frames", 0))
    specs.extend(
        PerturbationSpec(f"observation_delay_frames={delay}", "observation_delay_frames", delay)
        for delay in range(1, maximum_delay + 1)
    )
    if config.get("action_history_missing", False):
        specs.append(PerturbationSpec("action_history_missing", "action_history_missing", True))
    return specs


class ObservationPerturber:
    """Stateful transform for one perturbation and one episode."""

    def __init__(self, spec: PerturbationSpec, seed: int) -> None:
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.pointcloud_history: List[np.ndarray] = []
        self.state_history: List[np.ndarray] = []

    @staticmethod
    def _image_grid_colors(pointcloud: np.ndarray) -> tuple[np.ndarray, int]:
        point_count = pointcloud.shape[-2]
        side = int(math.ceil(math.sqrt(point_count)))
        colors = np.zeros((side * side, 3), dtype=np.float32)
        colors[:point_count] = pointcloud[..., 3:6].reshape(-1, 3)
        return colors.reshape(side, side, 3), point_count

    @staticmethod
    def _restore_colors(
        pointcloud: np.ndarray, colors: np.ndarray, point_count: int
    ) -> np.ndarray:
        result = pointcloud.copy()
        result[..., 3:6] = colors.reshape(-1, 3)[:point_count].reshape(
            result[..., 3:6].shape
        )
        return result

    def apply(
        self,
        pointcloud: np.ndarray,
        state: np.ndarray,
        action_history: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Apply only this spec and return explicit perturbation metadata."""
        original_pcd = np.asarray(pointcloud, dtype=np.float32)
        original_state = np.asarray(state, dtype=np.float32)
        self.pointcloud_history.append(original_pcd.copy())
        self.state_history.append(original_state.copy())
        pcd, robot_state = original_pcd.copy(), original_state.copy()
        history = None if action_history is None else np.asarray(action_history).copy()
        kind, value = self.spec.kind, self.spec.value
        metadata: Dict[str, Any] = {"name": self.spec.name, "applied": True}

        if kind == "brightness":
            pcd[..., 3:6] = np.clip(pcd[..., 3:6] * float(value), 0.0, 1.0)
        elif kind == "contrast":
            pcd[..., 3:6] = np.clip(
                (pcd[..., 3:6] - 0.5) * float(value) + 0.5, 0.0, 1.0
            )
        elif kind == "gaussian_noise_std":
            noise = self.rng.normal(0.0, float(value), size=pcd[..., 3:6].shape)
            pcd[..., 3:6] = np.clip(pcd[..., 3:6] + noise, 0.0, 1.0)
        elif kind == "blur_kernel":
            kernel = int(value)
            if kernel < 1:
                raise ValueError("blur_kernel must be positive")
            if kernel % 2 == 0:
                kernel += 1
                metadata["effective_kernel"] = kernel
            colors, count = self._image_grid_colors(pcd)
            colors = cv2.GaussianBlur(colors, (kernel, kernel), 0)
            pcd = self._restore_colors(pcd, colors, count)
        elif kind == "occlusion_fraction":
            fraction = float(value)
            if not 0.0 < fraction < 1.0:
                raise ValueError("occlusion_fraction must be in (0,1)")
            colors, count = self._image_grid_colors(pcd)
            side = colors.shape[0]
            block = max(1, int(round(side * math.sqrt(fraction))))
            top = int(self.rng.integers(0, max(1, side - block + 1)))
            left = int(self.rng.integers(0, max(1, side - block + 1)))
            colors[top : top + block, left : left + block] = 0.0
            pcd = self._restore_colors(pcd, colors, count)
            metadata["occlusion_box"] = [top, left, block, block]
        elif kind == "drop_frame_probability":
            dropped = bool(self.rng.random() < float(value)) and len(self.pointcloud_history) > 1
            metadata["dropped"] = dropped
            if dropped:
                pcd = self.pointcloud_history[-2].copy()
        elif kind == "observation_delay_frames":
            delay = int(value)
            source = max(0, len(self.pointcloud_history) - 1 - delay)
            pcd = self.pointcloud_history[source].copy()
            robot_state = self.state_history[source].copy()
            metadata["effective_delay"] = len(self.pointcloud_history) - 1 - source
        elif kind == "robot_state_noise_std":
            robot_state = robot_state + self.rng.normal(
                0.0, float(value), size=robot_state.shape
            ).astype(np.float32)
        elif kind == "action_history_missing":
            if history is not None:
                history = np.zeros_like(history)
        elif kind == "camera_interval_jitter_frames":
            maximum = int(value)
            if maximum < 1:
                raise ValueError("camera_interval_jitter_frames must be positive")
            offset = int(self.rng.integers(0, maximum + 1))
            source = max(0, len(self.pointcloud_history) - 1 - offset)
            pcd = self.pointcloud_history[source].copy()
            metadata["effective_offset"] = len(self.pointcloud_history) - 1 - source
        else:
            raise ValueError(f"Unsupported perturbation kind: {kind}")
        return pcd, robot_state, history, metadata
