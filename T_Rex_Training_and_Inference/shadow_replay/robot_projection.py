"""Lightweight URDF forward kinematics and calibrated camera projection.

This module intentionally avoids a heavyweight robotics dependency.  It covers
the fixed, revolute, and continuous joints needed by the North POC2.2 URDF and
keeps camera calibration explicit so a nominal mount cannot be mistaken for a
measured hand-eye calibration.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import cv2
import numpy as np

from .safety import north_dataset_to_urdf_names


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _vector(text: str | None, size: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(size, default, dtype=np.float64)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.shape != (size,):
        raise ValueError(f"Expected {size} values, got {text!r}")
    return value


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw rotation, ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rpy_matrix(rpy)
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis_array))
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    rotation_vector = axis_array / norm * float(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3], _ = cv2.Rodrigues(rotation_vector)
    return result


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class NorthUrdfKinematics:
    """Forward kinematics in the canonical North 65-D dataset joint order."""

    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        self.links = tuple(str(link.get("name")) for link in root.findall("link"))
        joints = []
        children = set()
        for element in root.findall("joint"):
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                continue
            origin_element = element.find("origin")
            axis_element = element.find("axis")
            parent = str(parent_element.get("link"))
            child = str(child_element.get("link"))
            children.add(child)
            joints.append(
                UrdfJoint(
                    name=str(element.get("name")),
                    joint_type=str(element.get("type", "fixed")),
                    parent=parent,
                    child=child,
                    origin=transform(
                        _vector(
                            None if origin_element is None else origin_element.get("xyz"),
                            3,
                        ),
                        _vector(
                            None if origin_element is None else origin_element.get("rpy"),
                            3,
                        ),
                    ),
                    axis=_vector(
                        None if axis_element is None else axis_element.get("xyz"),
                        3,
                    ),
                )
            )
        roots = sorted(set(self.links) - children)
        if len(roots) != 1:
            raise ValueError(f"URDF must have one root link, found {roots}")
        self.root_link = roots[0]
        self.joints = tuple(joints)
        self.joint_by_name = {joint.name: joint for joint in joints}
        self.dataset_joint_names = tuple(north_dataset_to_urdf_names())
        missing = sorted(set(self.dataset_joint_names) - set(self.joint_by_name))
        if missing:
            raise ValueError(f"URDF is missing North dataset joints: {missing}")

    def link_poses(self, action65: Sequence[float]) -> Dict[str, np.ndarray]:
        action = np.asarray(action65, dtype=np.float64)
        if action.shape != (65,):
            raise ValueError(f"North FK expects shape (65,), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("North FK input contains NaN/Inf")
        positions = dict(zip(self.dataset_joint_names, action.tolist()))
        poses: Dict[str, np.ndarray] = {self.root_link: np.eye(4, dtype=np.float64)}
        unresolved = list(self.joints)
        while unresolved:
            deferred = []
            progressed = False
            for joint in unresolved:
                if joint.parent not in poses:
                    deferred.append(joint)
                    continue
                motion = np.eye(4, dtype=np.float64)
                if joint.joint_type in {"revolute", "continuous"}:
                    motion = axis_rotation(joint.axis, positions.get(joint.name, 0.0))
                elif joint.joint_type == "prismatic":
                    motion[:3, 3] = (
                        joint.axis * float(positions.get(joint.name, 0.0))
                    )
                poses[joint.child] = poses[joint.parent] @ joint.origin @ motion
                progressed = True
            if not progressed:
                names = [joint.name for joint in deferred]
                raise ValueError(f"Disconnected/cyclic URDF joints: {names}")
            unresolved = deferred
        return poses

    def landmark_positions(self, action65: Sequence[float]) -> Dict[str, np.ndarray]:
        poses = self.link_poses(action65)
        landmarks: Dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            for index in range(1, 8):
                joint = self.joint_by_name[f"{side}_arm_joint_{index}"]
                landmarks[f"{side}_arm_joint_{index}"] = poses[joint.child][:3, 3]
            landmarks[f"{side}_wrist"] = poses[f"{side}_hand_base_link"][:3, 3]
            for finger in FINGERS:
                tip = f"{side}_{finger}_fingertip"
                landmarks[tip] = poses[tip][:3, 3]
            for name in self.dataset_joint_names:
                if name.startswith(f"{side}_") and "arm_joint" not in name:
                    joint = self.joint_by_name[name]
                    landmarks[name] = poses[joint.child][:3, 3]
        return landmarks

    @staticmethod
    def skeleton_edges() -> tuple[tuple[str, str], ...]:
        edges = []
        finger_suffixes = {
            "thumb": ("thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP"),
            "index": ("index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP"),
            "middle": ("middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP"),
            "ring": ("ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP"),
            "pinky": ("pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP"),
        }
        for side in ("left", "right"):
            arm = [f"{side}_arm_joint_{index}" for index in range(1, 8)]
            arm.append(f"{side}_wrist")
            edges.extend(zip(arm[:-1], arm[1:]))
            for finger, suffixes in finger_suffixes.items():
                chain = [f"{side}_wrist"]
                chain.extend(f"{side}_{suffix}" for suffix in suffixes)
                chain.append(f"{side}_{finger}_fingertip")
                edges.extend(zip(chain[:-1], chain[1:]))
        return tuple(edges)


@dataclass(frozen=True)
class CameraCalibration:
    """Camera intrinsics and camera pose relative to a moving URDF link.

    ``link_from_camera`` maps camera optical-frame coordinates into ``link``.
    ``urdf_from_data_base`` maps the documented dataset base frame into the
    URDF root frame.
    """

    matrix: np.ndarray
    distortion: np.ndarray
    link: str
    link_from_camera: np.ndarray
    urdf_from_data_base: np.ndarray
    source: str

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "CameraCalibration":
        matrix = np.asarray(config["matrix"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"camera matrix must be [3,3], got {matrix.shape}")
        distortion = np.asarray(config.get("distortion", []), dtype=np.float64)
        mount = config.get("mount", {})
        if not isinstance(mount, Mapping):
            raise ValueError("camera mount must be a mapping")
        base_alignment = config.get("urdf_from_data_base", {})
        if not isinstance(base_alignment, Mapping):
            raise ValueError("urdf_from_data_base must be a mapping")
        return cls(
            matrix=matrix,
            distortion=distortion,
            link=str(mount["link"]),
            link_from_camera=transform(
                mount.get("xyz", [0.0, 0.0, 0.0]),
                mount.get("rpy", [0.0, 0.0, 0.0]),
            ),
            urdf_from_data_base=transform(
                base_alignment.get("xyz", [0.0, 0.0, 0.0]),
                base_alignment.get("rpy", [0.0, 0.0, 0.0]),
            ),
            source=str(config.get("source", "unknown")),
        )

    def camera_from_urdf(
        self, link_poses: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        if self.link not in link_poses:
            raise KeyError(f"Camera mount link {self.link!r} is absent from URDF")
        urdf_from_camera = link_poses[self.link] @ self.link_from_camera
        return np.linalg.inv(urdf_from_camera)

    def project(
        self,
        points_urdf: np.ndarray,
        link_poses_at_capture: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_urdf, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape [N,3], got {points.shape}")
        camera_from_urdf = self.camera_from_urdf(link_poses_at_capture)
        camera = (
            camera_from_urdf[:3, :3] @ points.T
            + camera_from_urdf[:3, 3:4]
        ).T
        visible = camera[:, 2] > 1e-4
        projected = np.full((len(points), 2), np.nan, dtype=np.float64)
        if np.any(visible):
            image_points, _ = cv2.projectPoints(
                camera[visible],
                np.zeros(3),
                np.zeros(3),
                self.matrix,
                self.distortion,
            )
            projected[visible] = image_points.reshape(-1, 2)
        return projected, visible
