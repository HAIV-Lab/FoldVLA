"""North POC2.2 Isaac Lab contact-force to LeRobot tactile adapter.

The selected dataset stores tactile observations as ten ordered 6D wrenches:

    [fx, fy, fz, tx, ty, tz]

The native LeRobot feature is a flattened 60-D vector.  This module keeps the
conversion independent from an Isaac Lab environment so it can be unit-tested
without launching Isaac Sim.  Isaac Lab imports are deliberately lazy and are
only needed by :func:`make_north_contact_sensor_cfgs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch


NORTH_TACTILE_FINGERS: Tuple[str, ...] = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_little",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_little",
)

# The USD/URDF names use ``pinky`` while the dataset calls the same finger
# ``little``.  Keep this mapping explicit so the output order cannot silently
# drift when the simulator body names change.
NORTH_TACTILE_SENSOR_BODIES: Tuple[str, ...] = (
    "left_thumb_elastomer",
    "left_index_elastomer",
    "left_middle_elastomer",
    "left_ring_elastomer",
    "left_pinky_elastomer",
    "right_thumb_elastomer",
    "right_index_elastomer",
    "right_middle_elastomer",
    "right_ring_elastomer",
    "right_pinky_elastomer",
)

NORTH_TACTILE_COMPONENTS: Tuple[str, ...] = ("fx", "fy", "fz", "tx", "ty", "tz")
NORTH_TACTILE_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"{finger}_{component}"
    for finger in NORTH_TACTILE_FINGERS
    for component in NORTH_TACTILE_COMPONENTS
)
NORTH_TACTILE_DIM = len(NORTH_TACTILE_FEATURE_NAMES)


@dataclass(frozen=True)
class NorthTactileAdapterConfig:
    """Numerical conventions for converting Isaac Lab data.

    ``wrench_sign`` is useful because contact-force direction depends on
    whether the simulator reports the force acting on the sensor body or the
    equal-and-opposite force acting on the contacted object.  The default is
    +1; it should be calibrated against one real contact before using the
    dataset threshold quantitatively.

    ``force_scale`` and ``torque_scale`` are explicit calibration hooks.  They
    default to one so the first simulation only tests the control logic.
    """

    wrench_sign: float = 1.0
    force_scale: float = 1.0
    torque_scale: float = 1.0
    compute_torque: bool = True
    strict_torque: bool = False

    def __post_init__(self) -> None:
        if self.wrench_sign == 0.0:
            raise ValueError("wrench_sign must be non-zero")
        if self.force_scale < 0.0 or self.torque_scale < 0.0:
            raise ValueError("force_scale and torque_scale must be non-negative")


def _as_batch_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    """Extract the single body vector from an Isaac Lab sensor buffer."""

    if value.ndim == 2 and value.shape[-1] == 3:
        return value
    if value.ndim == 3 and value.shape[1] >= 1 and value.shape[-1] == 3:
        return value[:, 0, :]
    raise ValueError(f"{name} must have shape (N, 3) or (N, B, 3), got {tuple(value.shape)}")


def _quat_apply(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply a wxyz quaternion to 3D vectors."""

    xyz = quat_wxyz[..., 1:]
    t = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quat_wxyz[..., 0:1] * t + torch.cross(xyz, t, dim=-1)


def _quat_apply_inverse(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into a sensor frame represented by wxyz q."""

    quat_inverse = quat_wxyz.clone()
    quat_inverse[..., 1:] = -quat_inverse[..., 1:]
    norm = torch.linalg.vector_norm(quat_inverse, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return _quat_apply(quat_inverse / norm, vector)


class NorthTactileAdapter:
    """Convert ten Isaac Lab ``ContactSensor`` objects to dataset tactile data."""

    def __init__(self, config: Optional[NorthTactileAdapterConfig] = None):
        self.config = config or NorthTactileAdapterConfig()

    @staticmethod
    def _sensor_data(sensor: Any) -> Any:
        return getattr(sensor, "data", sensor)

    @staticmethod
    def _extract_contact_forces(data: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return total force and per-filter force in world coordinates.

        Isaac Lab's filtered ``force_matrix_w`` has shape
        ``(num_envs, num_bodies, num_filters, 3)``.  A separate sensor is used
        for each fingertip, so the first body dimension is selected and all
        filtered target bodies are summed.  The unfiltered net force is used as
        a fallback for adapters created without a filter expression.
        """

        force_matrix = getattr(data, "force_matrix_w", None)
        if force_matrix is not None:
            if force_matrix.ndim != 4 or force_matrix.shape[1] < 1 or force_matrix.shape[-1] != 3:
                raise ValueError(
                    "force_matrix_w must have shape (N, B, M, 3), "
                    f"got {tuple(force_matrix.shape)}"
                )
            per_filter_force = torch.nan_to_num(force_matrix[:, 0, :, :], nan=0.0, posinf=0.0, neginf=0.0)
            return per_filter_force.sum(dim=1), per_filter_force

        net_force = getattr(data, "net_forces_w", None)
        if net_force is None:
            raise ValueError("ContactSensor data must expose force_matrix_w or net_forces_w")
        force = torch.nan_to_num(_as_batch_vector(net_force, "net_forces_w"), nan=0.0, posinf=0.0, neginf=0.0)
        return force, None

    @staticmethod
    def _extract_sensor_pose(data: Any, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        position = getattr(data, "pos_w", None)
        quaternion = getattr(data, "quat_w", None)
        if position is None or quaternion is None:
            return None, None
        position = _as_batch_vector(position, "pos_w").to(device=device, dtype=dtype)
        if quaternion.ndim == 2 and quaternion.shape[-1] == 4:
            quaternion = quaternion
        elif quaternion.ndim == 3 and quaternion.shape[1] >= 1 and quaternion.shape[-1] == 4:
            quaternion = quaternion[:, 0, :]
        else:
            raise ValueError(
                "quat_w must have shape (N, 4) or (N, B, 4), "
                f"got {tuple(quaternion.shape)}"
            )
        if position.shape[0] != batch_size or quaternion.shape[0] != batch_size:
            raise ValueError("Sensor pose batch size does not match contact-force batch size")
        return position, quaternion.to(device=device, dtype=dtype)

    @staticmethod
    def _extract_torque(
        data: Any,
        per_filter_force_w: Optional[torch.Tensor],
        sensor_position_w: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if per_filter_force_w is None or sensor_position_w is None:
            return None
        contact_pos_w = getattr(data, "contact_pos_w", None)
        if contact_pos_w is None:
            return None
        if contact_pos_w.ndim != 4 or contact_pos_w.shape[1] < 1 or contact_pos_w.shape[-1] != 3:
            raise ValueError(
                "contact_pos_w must have shape (N, B, M, 3), "
                f"got {tuple(contact_pos_w.shape)}"
            )
        contact_pos_w = contact_pos_w[:, 0, :, :].to(
            device=per_filter_force_w.device, dtype=per_filter_force_w.dtype
        )
        if contact_pos_w.shape[:3] != per_filter_force_w.shape[:3]:
            raise ValueError(
                "contact_pos_w and force_matrix_w must have matching filter dimensions, "
                f"got {tuple(contact_pos_w.shape)} and {tuple(per_filter_force_w.shape)}"
            )
        valid = torch.isfinite(contact_pos_w).all(dim=-1)
        relative_position_w = torch.nan_to_num(contact_pos_w, nan=0.0) - sensor_position_w.unsqueeze(1)
        torque_per_filter_w = torch.cross(relative_position_w, per_filter_force_w, dim=-1)
        torque_per_filter_w = torch.where(valid.unsqueeze(-1), torque_per_filter_w, torch.zeros_like(torque_per_filter_w))
        return torque_per_filter_w.sum(dim=1)

    def convert_f6(self, sensors: Sequence[Any]) -> torch.Tensor:
        """Return tactile data with shape ``(num_envs, 10, 6)``.

        Every sensor should represent exactly one North elastomer body.  The
        order of ``sensors`` must match :data:`NORTH_TACTILE_SENSOR_BODIES`.
        """

        if len(sensors) != len(NORTH_TACTILE_SENSOR_BODIES):
            raise ValueError(
                f"Expected {len(NORTH_TACTILE_SENSOR_BODIES)} sensors, got {len(sensors)}"
            )

        rows: List[torch.Tensor] = []
        expected_batch: Optional[int] = None
        for sensor in sensors:
            data = self._sensor_data(sensor)
            force_w, per_filter_force_w = self._extract_contact_forces(data)
            if expected_batch is None:
                expected_batch = force_w.shape[0]
            elif force_w.shape[0] != expected_batch:
                raise ValueError("All tactile sensors must have the same environment batch size")

            sensor_position_w, sensor_quaternion_w = self._extract_sensor_pose(
                data, force_w.shape[0], force_w.device, force_w.dtype
            )
            torque_w = self._extract_torque(data, per_filter_force_w, sensor_position_w)
            if torque_w is None:
                if self.config.strict_torque and self.config.compute_torque:
                    raise ValueError(
                        "Cannot compute tactile torque: configure ContactSensor with "
                        "track_pose=True and track_contact_points=True"
                    )
                torque_w = torch.zeros_like(force_w)

            if sensor_quaternion_w is not None:
                force_out = _quat_apply_inverse(sensor_quaternion_w, force_w)
                torque_out = _quat_apply_inverse(sensor_quaternion_w, torque_w)
            else:
                force_out = force_w
                torque_out = torque_w

            force_out = force_out * (self.config.wrench_sign * self.config.force_scale)
            torque_out = torque_out * (self.config.wrench_sign * self.config.torque_scale)
            rows.append(torch.cat((force_out, torque_out), dim=-1))

        return torch.stack(rows, dim=1)

    def convert(self, sensors: Sequence[Any], flatten: bool = True) -> torch.Tensor:
        """Return dataset tactile data as ``(N, 60)`` or ``(N, 10, 6)``."""

        tactile_f6 = self.convert_f6(sensors)
        if flatten:
            return tactile_f6.reshape(tactile_f6.shape[0], NORTH_TACTILE_DIM)
        return tactile_f6


def make_north_contact_sensor_cfgs(
    robot_prim_path: str,
    filter_prim_paths_expr: Iterable[str],
    history_length: int = 1,
    max_contact_data_count_per_prim: int = 16,
) -> List[Any]:
    """Build one Isaac Lab ``ContactSensorCfg`` per North elastomer.

    Isaac Lab requires a separate sensor for each sensor-body-to-many-targets
    filtered contact view.  This helper therefore deliberately returns ten
    configurations instead of using one regex over all fingertips.
    """

    from isaaclab.sensors import ContactSensorCfg

    filters = list(filter_prim_paths_expr)
    if not filters:
        raise ValueError("At least one filtered contact target is required for tactile output")
    return [
        ContactSensorCfg(
            prim_path=f"{robot_prim_path}/{body_name}",
            history_length=history_length,
            track_pose=True,
            track_contact_points=True,
            max_contact_data_count_per_prim=max_contact_data_count_per_prim,
            filter_prim_paths_expr=filters,
        )
        for body_name in NORTH_TACTILE_SENSOR_BODIES
    ]


def tactile_feature_names() -> Tuple[str, ...]:
    """Return the exact 60 feature names from the selected dataset schema."""

    return NORTH_TACTILE_FEATURE_NAMES


__all__ = [
    "NORTH_TACTILE_COMPONENTS",
    "NORTH_TACTILE_DIM",
    "NORTH_TACTILE_FEATURE_NAMES",
    "NORTH_TACTILE_FINGERS",
    "NORTH_TACTILE_SENSOR_BODIES",
    "NorthTactileAdapter",
    "NorthTactileAdapterConfig",
    "make_north_contact_sensor_cfgs",
    "tactile_feature_names",
]
