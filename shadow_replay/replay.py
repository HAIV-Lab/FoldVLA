"""End-to-end offline Shadow Replay orchestration and artifact generation."""

from __future__ import annotations

import csv
import contextlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

from .actions import ActionNormalizer, NorthActionAdapter
from .config import discover_training_config, select_episode_ids
from .dataset import (
    EpisodeData,
    LeRobotEpisodeDataset,
    build_valid_chunk_mask,
    image_to_pointcloud,
    iter_batches,
    make_gray_image_plane_pointcloud,
)
from .metrics import (
    build_ground_truth_chunks,
    evaluate_action_predictions,
    latency_statistics,
    stage_metrics,
)
from .model import (
    BaseModelRunner,
    build_model_runner,
    load_training_yaml,
)
from .perturbations import (
    ObservationPerturber,
    PerturbationSpec,
    build_perturbation_specs,
)
from .safety import ActionSafetyChecker, load_north_urdf_limits
from .visualization import (
    save_dataset_plots,
    save_episode_plots,
    save_episode_video,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _path_size_bytes(path: Path) -> int:
    """Return the payload size for either a checkpoint file or directory."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=_json_default)
        stream.write("\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=_json_default))
        stream.write("\n")


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty and will not be overwritten: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        if hasattr(torch, "manual_seed"):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    except (ImportError, AttributeError):
        pass


def _flatten(prefix: str, value: Mapping[str, Any], target: Dict[str, Any]) -> None:
    for key, item in value.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            _flatten(full, item, target)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            target[full] = item


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_interval_annotations(path: Optional[str]) -> Dict[int, List[Dict[str, Any]]]:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {source}")
    with source.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream) or {}
    episodes = content.get("episodes", content)
    if not isinstance(episodes, Mapping):
        raise ValueError(f"Annotation file must contain an episode mapping: {source}")
    result: Dict[int, List[Dict[str, Any]]] = {}
    for episode_id, intervals in episodes.items():
        if not isinstance(intervals, list):
            raise ValueError(f"Episode {episode_id} annotations must be a list")
        result[int(episode_id)] = [dict(interval) for interval in intervals]
    return result


def _stage_labels(
    episode_id: int,
    frame_indices: np.ndarray,
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
) -> List[Optional[str]]:
    labels: List[Optional[str]] = [None] * len(frame_indices)
    for interval in annotations.get(episode_id, ()):
        if "stage" not in interval or "start_frame" not in interval or "end_frame" not in interval:
            raise ValueError(
                "Stage intervals require stage, start_frame, and end_frame (exclusive)"
            )
        selected = (frame_indices >= int(interval["start_frame"])) & (
            frame_indices < int(interval["end_frame"])
        )
        for index in np.flatnonzero(selected):
            if labels[index] is not None and labels[index] != str(interval["stage"]):
                raise ValueError(
                    f"Overlapping stage annotations for episode {episode_id}, "
                    f"frame {frame_indices[index]}"
                )
            labels[index] = str(interval["stage"])
    return labels


def _event_labels(
    episode_id: int,
    frame_indices: np.ndarray,
    events: Mapping[int, Sequence[Mapping[str, Any]]],
    window: int,
) -> List[Optional[str]]:
    labels: List[Optional[str]] = [None] * len(frame_indices)
    for event in events.get(episode_id, ()):
        if "event" not in event or "frame" not in event:
            raise ValueError("Critical events require event and frame")
        distance = np.abs(frame_indices - int(event["frame"]))
        for index in np.flatnonzero(distance <= window):
            name = str(event["event"])
            labels[index] = name if labels[index] is None else f"{labels[index]}+{name}"
    return labels


@dataclass
class EpisodeRun:
    episode_id: int
    source_length: int
    observation_frame_indices: np.ndarray
    target_frame_indices: np.ndarray
    timestamps: np.ndarray
    instructions: List[str]
    gt_robot_actions: np.ndarray
    gt_native_chunks: np.ndarray
    gt_robot_chunks: np.ndarray
    valid_mask: np.ndarray
    pred_raw_chunks: np.ndarray
    pred_denormalized_chunks: np.ndarray
    pred_joint_target_chunks: np.ndarray
    pred_safe_chunks: np.ndarray
    pred_eef_chunks: Optional[np.ndarray]
    safety_valid: np.ndarray
    safety_flags: List[List[str]]
    stage_labels: List[Optional[str]]
    event_labels: List[Optional[str]]
    latency: Dict[str, np.ndarray]
    frame_results: List[Dict[str, Any]]
    safety_violations: List[Dict[str, Any]]
    metrics_native: Dict[str, Any]
    metrics_robot: Dict[str, Any]


class ShadowReplay:
    """Offline-only evaluator. No method in this class exposes a robot command API."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.workspace_root = Path(config["_workspace_root"])
        self.output_dir = Path(config["output_dir"])
        _prepare_output_dir(self.output_dir)
        _set_seed(int(config.get("seed", 42)))
        self.started_at = time.perf_counter()
        self.warmup_done = False

        checkpoint_value = config.get("checkpoint")
        self.checkpoint = Path(checkpoint_value) if checkpoint_value else None
        backend = config.get("model", {}).get("backend", "unidex")
        self.backend = str(backend)
        if backend == "unidex":
            if self.checkpoint is None:
                raise ValueError("checkpoint is required for the UniDex backend")
            self.training_config_path = discover_training_config(
                self.checkpoint, config.get("training_config")
            )
            self.training_config = load_training_yaml(self.training_config_path)
        else:
            self.training_config_path = None
            self.training_config = {}

        frame_stride_value = config.get("dataset", {}).get("frame_stride", "auto")
        if frame_stride_value == "auto":
            frame_stride = int(
                self.training_config.get("dataset", {}).get("sample_stride", 1)
            )
        else:
            frame_stride = int(frame_stride_value)
        optional_fields = list(
            (config.get("dataset", {}).get("fields") or {}).values()
        )
        self.dataset = LeRobotEpisodeDataset(
            Path(config["dataset_path"]),
            frame_stride=frame_stride,
            optional_fields=optional_fields,
        )
        self.selected_ids = select_episode_ids(
            self.dataset.episode_ids,
            config.get("episode_ids"),
            config.get("max_episodes"),
        )
        if not self.selected_ids:
            raise ValueError("No test episodes selected")

        self.model = build_model_runner(
            config,
            checkpoint=self.checkpoint,
            training_config_path=self.training_config_path,
            workspace_root=self.workspace_root,
        )
        hand_json = (
            self.workspace_root / "UniDex" / "src" / "assets" / "utils" / "hand_utils.json"
        )
        self.action_adapter = NorthActionAdapter(
            self.dataset.action_names,
            self.model.capabilities.action_dim,
            str(config.get("action", {}).get("adapter", "auto")),
            hand_json,
        )
        if self.training_config:
            self.action_normalizer = ActionNormalizer.from_training_config(
                self.training_config, "action"
            )
            self.state_normalizer = ActionNormalizer.from_training_config(
                self.training_config, "state"
            )
        elif backend == "trex":
            self.action_normalizer = ActionNormalizer(
                stats=getattr(self.model, "action_stats"),
                norm_type="minmax",
                epsilon=0.0,
            )
            # T-Rex's real deployment adapter performs its own clipped q01/q99
            # state normalization immediately before state_embedder.
            self.state_normalizer = ActionNormalizer()
        else:
            self.action_normalizer = ActionNormalizer()
            self.state_normalizer = ActionNormalizer()
        for normalizer, expected, name in (
            (self.action_normalizer, self.model.capabilities.action_dim, "action"),
            (self.state_normalizer, self.model.capabilities.state_dim, "state"),
        ):
            if normalizer.dimension is not None and normalizer.dimension != expected:
                raise ValueError(
                    f"Checkpoint {name} dimension is {expected}, but training "
                    f"normalizer has {normalizer.dimension} values"
                )

        self._resolve_action_semantics()
        self.point_count = (
            self._resolve_point_count()
            if self.model.capabilities.input_kind == "pointcloud"
            else None
        )
        if (
            self.model.capabilities.input_kind == "trex_multimodal"
            and int(config.get("batch_size", 1)) != 1
        ):
            raise ValueError(
                "T-Rex's stateful cascaded deployment path requires batch_size=1"
            )
        self.target_offset = self._resolve_target_offset()
        self.stage_annotations = _load_interval_annotations(
            config.get("evaluation", {}).get("stage_annotations")
        )
        self.event_annotations = _load_interval_annotations(
            config.get("evaluation", {}).get("critical_events")
        )

        safety_cfg = config.get("safety", {})
        joint_limits = None
        if safety_cfg.get("urdf_path"):
            joint_limits = load_north_urdf_limits(
                Path(safety_cfg["urdf_path"]), self.dataset.action_names
            )
        self.safety_checker = ActionSafetyChecker(safety_cfg, joint_limits)
        self.perturbation_specs = build_perturbation_specs(
            config.get("perturbation", {})
        )
        self.leakage = self._check_leakage()

    def _resolve_action_semantics(self) -> None:
        action_cfg = self.config.setdefault("action", {})
        modality_action = (
            self.dataset.modality.get("action", {}).get("joints", {})
        )
        inferred = "absolute" if modality_action.get("absolute") is True else None
        configured = action_cfg.get("type", "auto")
        if configured == "auto":
            if inferred is None:
                raise ValueError(
                    "action.type cannot be inferred from meta/modality.json; set it explicitly"
                )
            action_cfg["type_resolved"] = inferred
        else:
            action_cfg["type_resolved"] = configured
            if inferred is not None and configured != inferred:
                raise ValueError(
                    f"Configured action.type={configured} contradicts dataset modality "
                    f"absolute={modality_action.get('absolute')}"
                )
        if action_cfg["type_resolved"] != "absolute":
            raise ValueError(
                "The North deployment adapter currently supports absolute joint targets only"
            )

        if self.model.capabilities.action_dim == 82:
            if action_cfg.get("xyz_indices") is None:
                action_cfg["xyz_indices"] = [0, 1, 2, 9, 10, 11]
            if action_cfg.get("rotation_indices") is None:
                action_cfg["rotation_indices"] = list(range(3, 9)) + list(
                    range(12, 18)
                )
            if action_cfg.get("representation", "auto") == "auto":
                action_cfg["representation_resolved"] = "rotation6d"
            else:
                action_cfg["representation_resolved"] = action_cfg["representation"]
            if action_cfg.get("unit", "auto") == "auto":
                action_cfg["unit_resolved"] = "m"
            else:
                action_cfg["unit_resolved"] = action_cfg["unit"]
        else:
            configured_representation = action_cfg.get(
                "representation", "joint_position"
            )
            action_cfg["representation_resolved"] = (
                "joint_position"
                if configured_representation == "auto"
                else configured_representation
            )
            action_cfg["unit_resolved"] = (
                "rad" if action_cfg.get("unit", "auto") == "auto" else action_cfg["unit"]
            )

    def _resolve_point_count(self) -> int:
        configured = self.config.get("preprocessing", {}).get("pointcloud_size", "auto")
        if configured == "auto":
            configured = self.training_config.get("dataset", {}).get(
                "pointcloud_size", 1024
            )
        count = int(configured)
        if count < 1:
            raise ValueError("preprocessing.pointcloud_size must be positive")
        return count

    def _resolve_target_offset(self) -> int:
        value = self.config.get("action", {}).get("target_offset_steps", "auto")
        if value == "auto":
            # RealDataset builds state at start_idx-1 and labels from start_idx.
            return 1 if self.action_adapter.adapter == "unidex82_to_north65" else 0
        value = int(value)
        if value < 0:
            raise ValueError("action.target_offset_steps must be non-negative")
        return value

    def _check_leakage(self) -> Dict[str, Any]:
        leakage_cfg = self.config.get("leakage", {})
        train_ids: set[int] = set()
        evidence: List[str] = []
        configured = leakage_cfg.get("training_episode_ids")
        if configured not in (None, "auto"):
            train_ids.update(int(value) for value in configured)
            evidence.append("config:leakage.training_episode_ids")

        source_zarr = self.config.get("preprocessing", {}).get("source_zarr")
        if source_zarr and Path(source_zarr).exists():
            try:
                import zarr

                root = zarr.open(str(source_zarr), mode="r")
                if "episode_indices" in root["meta"]:
                    train_ids.update(
                        int(value)
                        for value in np.unique(root["meta"]["episode_indices"][:])
                    )
                    evidence.append(f"zarr:{source_zarr}:meta/episode_indices")
            except Exception as exc:
                evidence.append(f"zarr_episode_id_read_failed:{exc}")

        manifest_value = leakage_cfg.get("training_manifest")
        if manifest_value and Path(manifest_value).is_file():
            manifest = json.loads(Path(manifest_value).read_text(encoding="utf-8"))
            manifest_ids = manifest.get("episode_ids")
            if manifest_ids is not None:
                train_ids.update(int(value) for value in manifest_ids)
                evidence.append(f"manifest:{manifest_value}:episode_ids")

        overlap = sorted(train_ids.intersection(self.selected_ids))
        warnings: List[str] = []
        dataset_path_lower = str(self.dataset.root).lower()
        if "_train" in dataset_path_lower or dataset_path_lower.endswith("/train"):
            warnings.append(
                "The selected dataset path is explicitly named as a training split."
            )
        if leakage_cfg.get("known_training_data_contains_test", False):
            warnings.append(
                str(
                    leakage_cfg.get(
                        "known_training_data_warning",
                        "Project training provenance marks the selected data as part "
                        "of the model's training corpus.",
                    )
                )
            )
        if overlap:
            warnings.append(
                f"TEST/TRAIN EPISODE OVERLAP DETECTED: {overlap}. Metrics are not "
                "independent test estimates."
            )
        finetune_path = self.workspace_root / "UniDex" / "finetune.py"
        if (
            self.backend == "unidex"
            and finetune_path.is_file()
            and "random_split("
            in finetune_path.read_text(encoding="utf-8")
        ):
            warnings.append(
                "UniDex/finetune.py uses random_split over frame windows; neighboring "
                "frames from one episode can leak between train and validation."
            )
        result = {
            "training_episode_ids": sorted(train_ids),
            "test_episode_ids": self.selected_ids,
            "overlap_episode_ids": overlap,
            "evidence": evidence,
            "warnings": warnings,
        }
        if overlap and leakage_cfg.get("fail_on_overlap", False):
            raise RuntimeError(warnings[-1])
        return result

    def _print_and_save_startup_summary(self) -> Dict[str, Any]:
        summary = self.dataset.dataset_summary(self.selected_ids)
        max_steps = self.config.get("max_steps_per_episode")
        effective_lengths = {
            episode_id: (
                selected_length
                if max_steps is None
                else min(selected_length, int(max_steps))
            )
            for episode_id, selected_length in (
                (int(key), int(value))
                for key, value in summary["episode_lengths_selected"].items()
            )
        }
        summary["episode_lengths_selected_before_step_cap"] = dict(
            summary["episode_lengths_selected"]
        )
        summary["episode_lengths_selected"] = {
            str(key): value for key, value in effective_lengths.items()
        }
        summary["total_selected_frames_before_step_cap"] = summary[
            "total_selected_frames"
        ]
        summary["total_selected_frames"] = int(sum(effective_lengths.values()))
        summary["total_model_prediction_steps"] = int(
            sum(max(0, value - self.target_offset) for value in effective_lengths.values())
        )
        capabilities = self.model.capabilities
        summary.update(
            {
                "model_action_dim": capabilities.action_dim,
                "model_state_dim": capabilities.state_dim,
                "observation_history_length": capabilities.observation_history,
                "action_chunk_length": capabilities.chunk_length,
                "action_target_offset_steps": self.target_offset,
                "action_type": self.config["action"]["type_resolved"],
                "model_action_representation": self.action_adapter.model_representation,
                "action_adapter": self.action_adapter.adapter,
                "pointcloud_size": self.point_count,
                "model_input_kind": capabilities.input_kind,
                "tactile_history_length": capabilities.tactile_history,
                "observation_source": self.config.get("preprocessing", {}).get(
                    "observation_source",
                    self.config.get("preprocessing", {}).get(
                        "pointcloud_source", "gray_image_plane"
                    ),
                ),
                "device": str(getattr(self.model, "device", "dummy/cpu")),
                "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
                "training_config": (
                    None
                    if self.training_config_path is None
                    else str(self.training_config_path)
                ),
                "leakage": self.leakage,
            }
        )
        write_json(self.output_dir / "dataset_summary.json", summary)
        print("SHADOW_REPLAY_DATASET_SUMMARY", flush=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        for warning in self.leakage["warnings"]:
            print(f"!!! LEAKAGE WARNING: {warning}", flush=True)
        return summary

    def _pointclouds_for_episode(
        self,
        episode: EpisodeData,
        step_count: int,
    ) -> tuple[List[np.ndarray], float]:
        source = self.config.get("preprocessing", {}).get(
            "pointcloud_source", "gray_image_plane"
        )
        started = time.perf_counter()
        if source == "gray_image_plane":
            base = make_gray_image_plane_pointcloud(self.point_count)
            values = [base.copy() for _ in range(step_count)]
        elif source == "rgb_image_plane":
            camera_key = self.config.get("preprocessing", {}).get(
                "camera_key", "observation.images.wrist_right"
            )
            values = []
            with self.dataset.open_video(episode.episode_id, camera_key) as reader:
                for frame_index in episode.frame_indices[:step_count]:
                    image = reader.read(int(frame_index))
                    values.append(image_to_pointcloud(image, self.point_count))
        else:
            raise ValueError(
                "preprocessing.pointcloud_source must be gray_image_plane or "
                f"rgb_image_plane, got {source!r}"
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return values, elapsed_ms

    @staticmethod
    def _split_tactile_deform(image_rgb: np.ndarray) -> np.ndarray:
        """Match T-Rex's first-channel 5x2 deform-mosaic split."""
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Tactile deform image must be RGB [H,W,3], got {image.shape}"
            )
        channel = image[..., 0].astype(np.float32) / 255.0
        height, width = channel.shape
        if height % 2 or width % 5:
            raise ValueError(
                f"T-Rex tactile deform mosaic must be divisible by 2x5, "
                f"got {(height, width)}"
            )
        cell_height, cell_width = height // 2, width // 5
        return np.stack(
            [
                channel[
                    row * cell_height : (row + 1) * cell_height,
                    column * cell_width : (column + 1) * cell_width,
                ]
                for row in range(2)
                for column in range(5)
            ],
            axis=0,
        )

    def _trex_observations_for_episode(
        self,
        episode: EpisodeData,
        step_count: int,
    ) -> tuple[Dict[str, np.ndarray], float]:
        """Decode the exact head/wrist/tactile inputs used by T-Rex training."""
        preprocessing = self.config.get("preprocessing", {})
        camera_keys = {
            "head_rgb": preprocessing.get(
                "head_camera_key", "observation.images.head_left"
            ),
            "wrist_right_rgb": preprocessing.get(
                "wrist_right_camera_key", "observation.images.wrist_right"
            ),
            "wrist_left_rgb": preprocessing.get(
                "wrist_left_camera_key", "observation.images.wrist_left"
            ),
            "tactile_deform": preprocessing.get(
                "tactile_deform_key", "observation.images.tactile_deform"
            ),
        }
        tactile_key = preprocessing.get(
            "tactile_vector_key", "observation.tactile"
        )
        tactile = episode.optional_fields.get(tactile_key)
        if tactile is None:
            raise KeyError(
                f"T-Rex requires {tactile_key!r}; add it to dataset.fields"
            )
        tactile = np.asarray(tactile[:step_count], dtype=np.float32)
        if tactile.shape != (step_count, 60):
            raise ValueError(
                f"T-Rex tactile vector must be [{step_count},60], got "
                f"{tactile.shape}"
            )

        started = time.perf_counter()
        decoded: Dict[str, List[np.ndarray]] = {
            "head_rgb": [],
            "wrist_right_rgb": [],
            "wrist_left_rgb": [],
            "tactile_deform": [],
        }
        with contextlib.ExitStack() as stack:
            readers = {
                name: stack.enter_context(
                    self.dataset.open_video(episode.episode_id, key)
                )
                for name, key in camera_keys.items()
            }
            for frame_index in episode.frame_indices[:step_count]:
                for name, reader in readers.items():
                    image = reader.read(int(frame_index))
                    decoded[name].append(
                        self._split_tactile_deform(image)
                        if name == "tactile_deform"
                        else image
                    )

        window = int(self.model.capabilities.tactile_history)
        if window < 1:
            raise ValueError("T-Rex tactile history length must be positive")
        tactile_frames = tactile.reshape(step_count, 10, 6)
        tactile_history = []
        for step in range(step_count):
            first = max(0, step - window + 1)
            selected = tactile_frames[first : step + 1]
            if len(selected) < window:
                selected = np.concatenate(
                    [
                        np.repeat(selected[:1], window - len(selected), axis=0),
                        selected,
                    ],
                    axis=0,
                )
            tactile_history.append(selected)
        observations = {
            "head_rgb": np.stack(decoded["head_rgb"]),
            "wrist_right_rgb": np.stack(decoded["wrist_right_rgb"]),
            "wrist_left_rgb": np.stack(decoded["wrist_left_rgb"]),
            "tactile_deform": np.stack(decoded["tactile_deform"]),
            "tactile_f6_history": np.stack(tactile_history),
        }
        return observations, (time.perf_counter() - started) * 1000.0

    def _perturb_trex_inputs(
        self,
        observations: Mapping[str, np.ndarray],
        states: np.ndarray,
        episode_id: int,
        perturbation: Optional[PerturbationSpec],
    ) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        values = {key: np.asarray(value).copy() for key, value in observations.items()}
        robot_states = np.asarray(states, dtype=np.float32).copy()
        if perturbation is None:
            return values, robot_states
        rng = np.random.default_rng(
            int(self.config.get("seed", 42))
            + episode_id * 1009
            + sum(ord(character) for character in perturbation.name)
        )
        image_keys = ("head_rgb", "wrist_right_rgb", "wrist_left_rgb")
        kind, setting = perturbation.kind, perturbation.value

        if kind in {
            "brightness",
            "contrast",
            "gaussian_noise_std",
            "blur_kernel",
            "occlusion_fraction",
        }:
            for key in image_keys:
                images = values[key].astype(np.float32) / 255.0
                if kind == "brightness":
                    images *= float(setting)
                elif kind == "contrast":
                    images = (images - 0.5) * float(setting) + 0.5
                elif kind == "gaussian_noise_std":
                    images += rng.normal(0.0, float(setting), images.shape)
                elif kind == "blur_kernel":
                    kernel = int(setting)
                    kernel += int(kernel % 2 == 0)
                    images = np.stack(
                        [cv2.GaussianBlur(image, (kernel, kernel), 0) for image in images]
                    )
                else:
                    fraction = float(setting)
                    if not 0.0 < fraction < 1.0:
                        raise ValueError("occlusion_fraction must be in (0,1)")
                    height, width = images.shape[1:3]
                    block_height = max(1, int(height * np.sqrt(fraction)))
                    block_width = max(1, int(width * np.sqrt(fraction)))
                    for image in images:
                        top = int(rng.integers(0, max(1, height - block_height + 1)))
                        left = int(rng.integers(0, max(1, width - block_width + 1)))
                        image[
                            top : top + block_height,
                            left : left + block_width,
                        ] = 0.0
                values[key] = np.asarray(
                    np.clip(images * 255.0, 0.0, 255.0), dtype=np.uint8
                )
        elif kind == "robot_state_noise_std":
            robot_states += rng.normal(
                0.0, float(setting), robot_states.shape
            ).astype(np.float32)
        elif kind in {"observation_delay_frames", "camera_interval_jitter_frames"}:
            maximum = int(setting)
            for step in range(len(robot_states)):
                offset = (
                    maximum
                    if kind == "observation_delay_frames"
                    else int(rng.integers(0, maximum + 1))
                )
                source = max(0, step - offset)
                for key in image_keys:
                    values[key][step] = observations[key][source]
                if kind == "observation_delay_frames":
                    values["tactile_deform"][step] = observations[
                        "tactile_deform"
                    ][source]
                    values["tactile_f6_history"][step] = observations[
                        "tactile_f6_history"
                    ][source]
                    robot_states[step] = states[source]
        elif kind == "drop_frame_probability":
            for step in range(1, len(robot_states)):
                if rng.random() < float(setting):
                    for key in values:
                        values[key][step] = values[key][step - 1]
                    robot_states[step] = robot_states[step - 1]
        elif kind == "action_history_missing":
            # The caller reports this perturbation as not applicable because
            # T-Rex has no action-history input.
            pass
        else:
            raise ValueError(f"Unsupported T-Rex perturbation: {kind}")
        return values, robot_states

    def _predict_trex(
        self,
        episode: EpisodeData,
        observations: Mapping[str, np.ndarray],
        step_count: int,
        perturbation: Optional[PerturbationSpec],
    ) -> tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        started = time.perf_counter()
        values, states = self._perturb_trex_inputs(
            observations,
            episode.states[:step_count],
            episode.episode_id,
            perturbation,
        )
        prompt_template = self.config.get("preprocessing", {}).get(
            "prompt_template", "{instruction}"
        )
        prompts = [
            prompt_template.format(instruction=episode.instructions[step])
            for step in range(step_count)
        ]
        preprocess_latency = np.full(
            step_count,
            (time.perf_counter() - started) * 1000.0 / step_count,
            dtype=np.float64,
        )
        predictions = []
        inference_latency = np.zeros(step_count, dtype=np.float64)
        for step in range(step_count):
            observation = {
                key: value[step : step + 1] for key, value in values.items()
            }
            state = states[step : step + 1, None, :]
            if not self.warmup_done and perturbation is None:
                self.model.warmup(
                    None,
                    state,
                    prompts[step : step + 1],
                    iterations=int(self.config.get("warmup_iterations", 1)),
                    seed=int(self.config.get("seed", 42)) + 800_000,
                    observations=observation,
                )
                self.warmup_done = True
            output = self.model.predict(
                None,
                state,
                prompts[step : step + 1],
                seed=int(self.config.get("seed", 42))
                + episode.episode_id * 100_000
                + step,
                observations=observation,
            )
            predictions.append(output.action_raw)
            inference_latency[step] = output.inference_latency_ms
        return (
            np.concatenate(predictions, axis=0),
            inference_latency,
            prompts,
            preprocess_latency,
        )

    def _preprocess_inputs(
        self,
        episode: EpisodeData,
        pointclouds: Sequence[np.ndarray],
        step_count: int,
        perturbation: Optional[PerturbationSpec],
    ) -> tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        history = self.model.capabilities.observation_history
        pcd_history: List[np.ndarray] = []
        state_history: List[np.ndarray] = []
        pcd_batches: List[np.ndarray] = []
        state_batches: List[np.ndarray] = []
        timings: List[float] = []
        prompt_template = self.config.get("preprocessing", {}).get(
            "prompt_template", "Use Shadow hands to {instruction}."
        )
        prompts: List[str] = []
        perturber = (
            ObservationPerturber(
                perturbation,
                seed=int(self.config.get("seed", 42))
                + episode.episode_id * 1009
                + sum(ord(char) for char in perturbation.name),
            )
            if perturbation is not None
            else None
        )
        for step in range(step_count):
            started = time.perf_counter()
            pcd = np.asarray(pointclouds[step], dtype=np.float32)
            raw_state = np.asarray(episode.states[step], dtype=np.float32)
            if perturber is not None:
                pcd, raw_state, _, _ = perturber.apply(pcd, raw_state, None)
            native_state = self.action_adapter.dataset_to_model(raw_state)
            normalized_state = self.state_normalizer.normalize(native_state)
            pcd_history.append(pcd)
            state_history.append(normalized_state)
            first = max(0, len(pcd_history) - history)
            selected_pcd = pcd_history[first:]
            selected_state = state_history[first:]
            while len(selected_pcd) < history:
                selected_pcd.insert(0, selected_pcd[0])
                selected_state.insert(0, selected_state[0])
            pcd_batches.append(np.stack(selected_pcd, axis=0))
            state_batches.append(np.stack(selected_state, axis=0))
            instruction = episode.instructions[step]
            prompt = prompt_template.format(instruction=instruction)
            if not prompt.endswith("\n"):
                prompt += "\n"
            prompts.append(prompt)
            timings.append((time.perf_counter() - started) * 1000.0)
        return (
            np.stack(pcd_batches, axis=0),
            np.stack(state_batches, axis=0),
            prompts,
            np.asarray(timings, dtype=np.float64),
        )

    def _predict(
        self,
        episode: EpisodeData,
        pointclouds: Sequence[np.ndarray],
        step_count: int,
        perturbation: Optional[PerturbationSpec],
    ) -> tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        pcd, state, prompts, preprocess_latency = self._preprocess_inputs(
            episode, pointclouds, step_count, perturbation
        )
        predictions = []
        inference_latencies = np.zeros(step_count, dtype=np.float64)
        batch_size = int(self.config.get("batch_size", 1))
        for batch_slice in iter_batches(step_count, batch_size):
            batch_prompts = prompts[batch_slice]
            if not self.warmup_done and perturbation is None:
                self.model.warmup(
                    pcd[batch_slice],
                    state[batch_slice],
                    batch_prompts,
                    iterations=int(self.config.get("warmup_iterations", 1)),
                    seed=int(self.config.get("seed", 42)) + 800_000,
                )
                self.warmup_done = True
            output = self.model.predict(
                pcd[batch_slice],
                state[batch_slice],
                batch_prompts,
                seed=int(self.config.get("seed", 42))
                + episode.episode_id * 100_000
                + int(batch_slice.start),
            )
            predictions.append(output.action_raw)
            batch_length = int(batch_slice.stop - batch_slice.start)
            inference_latencies[batch_slice] = output.inference_latency_ms / batch_length
        return (
            np.concatenate(predictions, axis=0),
            inference_latencies,
            prompts,
            preprocess_latency,
        )

    def _run_episode(
        self,
        episode: EpisodeData,
        data_read_latency_ms: float,
        perturbation: Optional[PerturbationSpec] = None,
    ) -> EpisodeRun:
        self.model.reset_episode()
        step_count = len(episode) - self.target_offset
        if step_count <= 0:
            raise ValueError(
                f"Episode {episode.episode_id} is too short for target offset "
                f"{self.target_offset}"
            )
        if self.model.capabilities.input_kind == "trex_multimodal":
            observations, observation_read_ms = self._trex_observations_for_episode(
                episode, step_count
            )
            pred_raw, inference_latency, _, preprocess_latency = (
                self._predict_trex(
                    episode, observations, step_count, perturbation
                )
            )
        else:
            pointclouds, observation_read_ms = self._pointclouds_for_episode(
                episode, step_count
            )
            pred_raw, inference_latency, _, preprocess_latency = self._predict(
                episode, pointclouds, step_count, perturbation
            )
        expected = (
            step_count,
            self.model.capabilities.chunk_length,
            self.model.capabilities.action_dim,
        )
        if pred_raw.shape != expected:
            raise ValueError(f"Prediction shape {pred_raw.shape}, expected {expected}")

        post_started = time.perf_counter()
        pred_denormalized = self.action_normalizer.denormalize(pred_raw)
        postprocess_latency = np.full(
            step_count,
            (time.perf_counter() - post_started) * 1000.0 / step_count,
            dtype=np.float64,
        )
        target_actions = episode.actions[self.target_offset :]
        target_native = self.action_adapter.dataset_to_model(target_actions)
        gt_native_chunks, valid_mask = build_ground_truth_chunks(
            target_native, self.model.capabilities.chunk_length
        )
        gt_robot_chunks, robot_mask = build_ground_truth_chunks(
            target_actions, self.model.capabilities.chunk_length
        )
        if not np.array_equal(valid_mask, robot_mask):
            raise AssertionError("Native and robot action masks disagree")

        pred_joint = np.zeros(
            (step_count, self.model.capabilities.chunk_length, 65), dtype=np.float32
        )
        pred_safe = np.zeros_like(pred_joint)
        pred_eef = (
            np.zeros(
                (step_count, self.model.capabilities.chunk_length, 18),
                dtype=np.float32,
            )
            if self.action_adapter.adapter == "unidex82_to_north65"
            else None
        )
        safety_valid = np.ones(step_count, dtype=bool)
        safety_flags: List[List[str]] = []
        violations: List[Dict[str, Any]] = []
        safety_latency = np.zeros(step_count, dtype=np.float64)
        adapter_limitations: tuple[str, ...] = ()
        dt = 1.0 / (self.dataset.fps / self.dataset.frame_stride)

        for step in range(step_count):
            adapted = self.action_adapter.model_to_robot(
                pred_denormalized[step], episode.states[step]
            )
            adapter_limitations = adapted.limitations
            pred_joint[step] = adapted.robot_joint_target
            if pred_eef is not None and adapted.eef_action is not None:
                pred_eef[step] = adapted.eef_action
            started = time.perf_counter()
            previous = episode.states[step].copy()
            previous_velocity = None
            first_result = None
            for horizon in range(self.model.capabilities.chunk_length):
                original_target = pred_joint[step, horizon]
                result = self.safety_checker.check(
                    original_target,
                    episode.states[step],
                    previous_action=previous,
                    previous_velocity=previous_velocity,
                    dt=dt,
                    eef_action=(
                        None if pred_eef is None else pred_eef[step, horizon]
                    ),
                )
                pred_safe[step, horizon] = result.action_safe
                velocity = (result.action_safe - previous) / dt
                previous = result.action_safe
                previous_velocity = velocity
                if horizon == 0:
                    first_result = result
                for violation in result.violations:
                    record = violation.to_dict()
                    record.update(
                        {
                            "episode_id": episode.episode_id,
                            "step_id": step,
                            "frame_index": int(episode.frame_indices[step]),
                            "horizon": horizon,
                        }
                    )
                    violations.append(record)
            safety_latency[step] = (time.perf_counter() - started) * 1000.0
            assert first_result is not None
            safety_valid[step] = first_result.valid
            safety_flags.append(first_result.flags)

        stage_labels = _stage_labels(
            episode.episode_id,
            episode.frame_indices[:step_count],
            self.stage_annotations,
        )
        event_labels = _event_labels(
            episode.episode_id,
            episode.frame_indices[:step_count],
            self.event_annotations,
            int(self.config.get("evaluation", {}).get("critical_event_window", 5))
            * self.dataset.frame_stride,
        )
        per_frame_data_read = (
            data_read_latency_ms + observation_read_ms
        ) / step_count
        data_latency = np.full(step_count, per_frame_data_read, dtype=np.float64)
        total_latency = (
            data_latency
            + preprocess_latency
            + inference_latency
            + postprocess_latency
            + safety_latency
        )
        latency = {
            "data_read_ms": data_latency,
            "preprocess_ms": preprocess_latency,
            "inference_ms": inference_latency,
            "postprocess_ms": postprocess_latency,
            "safety_ms": safety_latency,
            "total_ms": total_latency,
        }

        action_cfg = self.config.get("action", {})
        native_metrics = evaluate_action_predictions(
            pred_denormalized,
            gt_native_chunks,
            valid_mask,
            fps=self.dataset.fps / self.dataset.frame_stride,
            xyz_indices=action_cfg.get("xyz_indices"),
            rotation_indices=action_cfg.get("rotation_indices"),
            rotation_representation=action_cfg.get("representation_resolved"),
            quaternion_order=action_cfg.get("quaternion_order", "xyzw"),
            gripper_indices=action_cfg.get("gripper_indices"),
            gripper_threshold=float(action_cfg.get("gripper_threshold", 0.5)),
            position_unit=action_cfg.get("unit_resolved", "rad"),
        )
        robot_metrics = evaluate_action_predictions(
            pred_safe,
            gt_robot_chunks,
            valid_mask,
            fps=self.dataset.fps / self.dataset.frame_stride,
            gripper_indices=action_cfg.get("robot_gripper_indices"),
            gripper_threshold=float(action_cfg.get("gripper_threshold", 0.5)),
            position_unit="rad",
        )
        frame_results: List[Dict[str, Any]] = []
        fields = self.config.get("dataset", {}).get("fields") or {}
        tcp_values = episode.optional_fields.get(fields.get("tcp", ""))
        for step in range(step_count):
            gt = target_actions[step]
            frame_results.append(
                {
                    "episode_id": episode.episode_id,
                    "step_id": step,
                    "frame_index": int(episode.frame_indices[step]),
                    "gt_frame_index": int(
                        episode.frame_indices[step + self.target_offset]
                    ),
                    "timestamp": float(episode.timestamps[step]),
                    "instruction": episode.instructions[step],
                    "gt_action": gt,
                    "pred_action_raw": pred_raw[step, 0],
                    "pred_action_denormalized": pred_denormalized[step, 0],
                    "pred_action_safe": pred_safe[step, 0],
                    "pred_eef_action": (
                        None if pred_eef is None else pred_eef[step, 0]
                    ),
                    "pred_joint_target": pred_joint[step, 0],
                    "robot_state": episode.states[step],
                    "eef_pose": None,
                    "tcp_state": (
                        None if tcp_values is None else tcp_values[step]
                    ),
                    "joint_state": episode.states[step],
                    "gripper_state": None,
                    "stage_gt": stage_labels[step],
                    "stage_pred": None,
                    "gate_output": None,
                    "inference_latency_ms": float(inference_latency[step]),
                    "preprocess_latency_ms": float(preprocess_latency[step]),
                    "total_latency_ms": float(total_latency[step]),
                    "safety_valid": bool(safety_valid[step]),
                    "safety_flags": safety_flags[step],
                    "ik_success": None,
                    "nan_or_inf": bool(
                        not np.isfinite(pred_raw[step]).all()
                    ),
                    "joint_mae": float(np.mean(np.abs(pred_safe[step, 0] - gt))),
                    "adapter_limitations": list(adapter_limitations),
                }
            )
        return EpisodeRun(
            episode_id=episode.episode_id,
            source_length=episode.source_length,
            observation_frame_indices=episode.frame_indices[:step_count],
            target_frame_indices=episode.frame_indices[
                self.target_offset : self.target_offset + step_count
            ],
            timestamps=episode.timestamps[:step_count],
            instructions=episode.instructions[:step_count],
            gt_robot_actions=target_actions,
            gt_native_chunks=gt_native_chunks,
            gt_robot_chunks=gt_robot_chunks,
            valid_mask=valid_mask,
            pred_raw_chunks=pred_raw,
            pred_denormalized_chunks=pred_denormalized,
            pred_joint_target_chunks=pred_joint,
            pred_safe_chunks=pred_safe,
            pred_eef_chunks=pred_eef,
            safety_valid=safety_valid,
            safety_flags=safety_flags,
            stage_labels=stage_labels,
            event_labels=event_labels,
            latency=latency,
            frame_results=frame_results,
            safety_violations=violations,
            metrics_native=native_metrics,
            metrics_robot=robot_metrics,
        )

    def _save_episode(self, run: EpisodeRun) -> None:
        result_dir = self.output_dir / "episode_results"
        result_dir.mkdir(parents=True, exist_ok=True)
        if self.config.get("save_frame_results", True):
            arrays: Dict[str, Any] = {
                "episode_id": np.asarray(run.episode_id, dtype=np.int64),
                "observation_frame_indices": run.observation_frame_indices,
                "target_frame_indices": run.target_frame_indices,
                "timestamps": run.timestamps,
                "instructions": np.asarray(run.instructions, dtype=np.str_),
                "gt_action": run.gt_robot_actions,
                "robot_state": np.stack(
                    [frame["robot_state"] for frame in run.frame_results]
                ),
                "joint_state": np.stack(
                    [frame["joint_state"] for frame in run.frame_results]
                ),
                "gt_native_chunks": run.gt_native_chunks,
                "valid_mask": run.valid_mask,
                "pred_action_raw": run.pred_raw_chunks,
                "pred_action_denormalized": run.pred_denormalized_chunks,
                "pred_joint_target": run.pred_joint_target_chunks,
                "pred_action_safe": run.pred_safe_chunks,
                "safety_valid": run.safety_valid,
                "safety_flags_json": np.asarray(
                    [json.dumps(flags) for flags in run.safety_flags], dtype=np.str_
                ),
                "stage_gt": np.asarray(
                    ["" if value is None else value for value in run.stage_labels],
                    dtype=np.str_,
                ),
                "event_labels": np.asarray(
                    ["" if value is None else value for value in run.event_labels],
                    dtype=np.str_,
                ),
                "nan_or_inf": np.asarray(
                    [frame["nan_or_inf"] for frame in run.frame_results],
                    dtype=bool,
                ),
            }
            if all(frame["tcp_state"] is not None for frame in run.frame_results):
                arrays["tcp_state"] = np.stack(
                    [frame["tcp_state"] for frame in run.frame_results]
                )
            if run.pred_eef_chunks is not None:
                arrays["pred_eef_action"] = run.pred_eef_chunks
            for key, value in run.latency.items():
                arrays[f"latency_{key}"] = value
            np.savez_compressed(
                result_dir / f"episode_{run.episode_id:04d}.npz", **arrays
            )
        if (
            self.config.get("save_frame_results", True)
            and self.config.get("save_frame_jsonl", False)
        ):
            jsonl = result_dir / f"episode_{run.episode_id:04d}.jsonl"
            for frame in run.frame_results:
                append_jsonl(jsonl, frame)
        write_json(
            result_dir / f"episode_{run.episode_id:04d}_metrics.json",
            {
                "model_space": run.metrics_native,
                "robot_joint_space": run.metrics_robot,
            },
        )
        for violation in run.safety_violations:
            append_jsonl(self.output_dir / "safety_violations.jsonl", violation)

        visualization_cfg = self.config.get("visualization", {})
        if self.config.get("save_plots", True):
            save_episode_plots(
                self.output_dir
                / "visualizations"
                / f"episode_{run.episode_id:04d}_actions.png",
                gt_action=run.gt_robot_actions,
                pred_action_safe=run.pred_safe_chunks[:, 0],
                latency_ms=run.latency["total_ms"],
                safety_valid=run.safety_valid,
                max_curve_dimensions=int(
                    visualization_cfg.get("max_curve_dimensions", 16)
                ),
            )
        if self.config.get("save_video", False):
            camera_key = visualization_cfg.get(
                "camera_key", "observation.images.head_left"
            )
            with self.dataset.open_video(run.episode_id, camera_key) as reader:
                save_episode_video(
                    self.output_dir
                    / "visualizations"
                    / f"episode_{run.episode_id:04d}.mp4",
                    video_reader=reader,
                    frame_indices=run.observation_frame_indices,
                    frame_results=run.frame_results,
                    fps=self.dataset.fps / self.dataset.frame_stride,
                )

    def _aggregate_metrics(self, runs: Sequence[EpisodeRun]) -> Dict[str, Any]:
        if not runs:
            return {}
        pred_native = np.concatenate([run.pred_denormalized_chunks for run in runs])
        gt_native = np.concatenate([run.gt_native_chunks for run in runs])
        pred_robot = np.concatenate([run.pred_safe_chunks for run in runs])
        gt_robot = np.concatenate([run.gt_robot_chunks for run in runs])
        mask = np.concatenate([run.valid_mask for run in runs])
        action_cfg = self.config.get("action", {})
        native = evaluate_action_predictions(
            pred_native,
            gt_native,
            mask,
            fps=self.dataset.fps / self.dataset.frame_stride,
            xyz_indices=action_cfg.get("xyz_indices"),
            rotation_indices=action_cfg.get("rotation_indices"),
            rotation_representation=action_cfg.get("representation_resolved"),
            quaternion_order=action_cfg.get("quaternion_order", "xyzw"),
            gripper_indices=action_cfg.get("gripper_indices"),
            gripper_threshold=float(action_cfg.get("gripper_threshold", 0.5)),
            position_unit=action_cfg.get("unit_resolved", "rad"),
        )
        robot = evaluate_action_predictions(
            pred_robot,
            gt_robot,
            mask,
            fps=self.dataset.fps / self.dataset.frame_stride,
            gripper_indices=action_cfg.get("robot_gripper_indices"),
            gripper_threshold=float(action_cfg.get("gripper_threshold", 0.5)),
            position_unit="rad",
        )
        stages = sum((run.stage_labels for run in runs), [])
        safety = np.concatenate([run.safety_valid for run in runs])
        latency = np.concatenate([run.latency["total_ms"] for run in runs])
        per_stage = stage_metrics(
            pred_native,
            gt_native,
            mask,
            stages,
            fps=self.dataset.fps / self.dataset.frame_stride,
            safety_valid=safety,
            latency_ms=latency,
            xyz_indices=action_cfg.get("xyz_indices"),
            rotation_indices=action_cfg.get("rotation_indices"),
            rotation_representation=action_cfg.get("representation_resolved"),
            quaternion_order=action_cfg.get("quaternion_order", "xyzw"),
            gripper_indices=action_cfg.get("gripper_indices"),
            gripper_threshold=float(action_cfg.get("gripper_threshold", 0.5)),
            position_unit=action_cfg.get("unit_resolved", "rad"),
        )
        events = sum((run.event_labels for run in runs), [])
        per_event = stage_metrics(
            pred_native,
            gt_native,
            mask,
            events,
            fps=self.dataset.fps / self.dataset.frame_stride,
            safety_valid=safety,
            latency_ms=latency,
            xyz_indices=action_cfg.get("xyz_indices"),
            rotation_indices=action_cfg.get("rotation_indices"),
            rotation_representation=action_cfg.get("representation_resolved"),
            quaternion_order=action_cfg.get("quaternion_order", "xyzw"),
            gripper_indices=action_cfg.get("gripper_indices"),
            gripper_threshold=float(action_cfg.get("gripper_threshold", 0.5)),
            position_unit=action_cfg.get("unit_resolved", "rad"),
        )
        return {
            "model_space": native,
            "robot_joint_space": robot,
            "stage_metrics": per_stage,
            "critical_event_metrics": per_event,
            "safety": {
                "first_action_violation_count": int(np.sum(~safety)),
                "first_action_valid_ratio": float(np.mean(safety)),
                "all_chunk_violation_count": int(
                    sum(len(run.safety_violations) for run in runs)
                ),
            },
        }

    def _latency_metrics(self, runs: Sequence[EpisodeRun]) -> Dict[str, Any]:
        if not runs:
            return {}
        result = {
            key: latency_statistics(
                np.concatenate([run.latency[key] for run in runs])
            )
            for key in runs[0].latency
        }
        total_frames = sum(len(run.timestamps) for run in runs)
        wall = time.perf_counter() - self.started_at
        result["actual_fps_wall_clock"] = total_frames / wall if wall > 0 else None
        all_total_latency = np.concatenate(
            [run.latency["total_ms"] for run in runs]
        )
        result["effective_fps_from_mean_total_latency"] = (
            float(1000.0 / np.mean(all_total_latency))
            if len(all_total_latency) and np.mean(all_total_latency) > 0
            else None
        )
        result["gpu_peak_memory_mb"] = self.model.peak_gpu_memory_mb()
        return result

    def _run_perturbations(
        self,
        episodes: Mapping[int, EpisodeData],
        baselines: Mapping[int, EpisodeRun],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for spec in self.perturbation_specs:
            if spec.kind == "action_history_missing" and not self.model.capabilities.uses_action_history:
                results[spec.name] = {
                    "status": "not_applicable",
                    "reason": "The current UniDex model has no action-history input.",
                }
                continue
            variant_runs: List[EpisodeRun] = []
            deviations: List[np.ndarray] = []
            sensitive: Optional[Dict[str, Any]] = None
            errors: List[Dict[str, Any]] = []
            for episode_id, episode in episodes.items():
                try:
                    variant = self._run_episode(
                        episode, data_read_latency_ms=0.0, perturbation=spec
                    )
                    variant_runs.append(variant)
                    baseline = baselines[episode_id]
                    deviation = np.mean(
                        np.abs(
                            variant.pred_denormalized_chunks[:, 0]
                            - baseline.pred_denormalized_chunks[:, 0]
                        ),
                        axis=-1,
                    )
                    deviations.append(deviation)
                    index = int(np.nanargmax(deviation))
                    candidate = {
                        "episode_id": episode_id,
                        "step_id": index,
                        "frame_index": int(episode.frame_indices[index]),
                        "action_deviation": float(deviation[index]),
                    }
                    if sensitive is None or candidate["action_deviation"] > sensitive["action_deviation"]:
                        sensitive = candidate
                except Exception as exc:
                    errors.append(
                        {
                            "episode_id": episode_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
            aggregate = self._aggregate_metrics(variant_runs)
            baseline_runs = [baselines[run.episode_id] for run in variant_runs]
            baseline_aggregate = self._aggregate_metrics(baseline_runs)
            variant_mae = (
                aggregate.get("model_space", {}).get("action_mae")
                if aggregate
                else None
            )
            baseline_mae = (
                baseline_aggregate.get("model_space", {}).get("action_mae")
                if baseline_aggregate
                else None
            )
            variant_safety = sum(
                int(np.sum(~run.safety_valid)) for run in variant_runs
            )
            baseline_safety = sum(
                int(np.sum(~run.safety_valid)) for run in baseline_runs
            )
            results[spec.name] = {
                "status": "completed" if variant_runs else "failed",
                "action_deviation_mean": (
                    float(np.mean(np.concatenate(deviations))) if deviations else None
                ),
                "metric_action_mae": variant_mae,
                "metric_action_mae_degradation": (
                    None
                    if variant_mae is None or baseline_mae is None
                    else float(variant_mae - baseline_mae)
                ),
                "safety_violation_count": variant_safety,
                "safety_violation_increase": variant_safety - baseline_safety,
                "stage_or_gate_change": None,
                "stage_or_gate_note": "Current UniDex checkpoint has no stage/gate head.",
                "most_sensitive": sensitive,
                "episode_errors": errors,
            }
        return results

    def _write_report(
        self,
        dataset_summary: Mapping[str, Any],
        metrics: Mapping[str, Any],
        latency: Mapping[str, Any],
        runs: Sequence[EpisodeRun],
        perturbations: Mapping[str, Any],
        episode_errors: Sequence[Mapping[str, Any]],
    ) -> None:
        worst = sorted(
            runs,
            key=lambda run: (
                run.metrics_robot.get("action_mae")
                if run.metrics_robot.get("action_mae") is not None
                else -1.0
            ),
            reverse=True,
        )[:5]
        checkpoint_info = (
            "dummy test backend"
            if self.checkpoint is None
            else f"`{self.checkpoint}` ({_path_size_bytes(self.checkpoint) / (1024**3):.2f} GiB)"
        )
        lines = [
            "# Offline Shadow Replay Report",
            "",
            "> This is offline Shadow Replay. Every next observation comes from the "
            "recorded demonstrator trajectory, not from executing the predicted action. "
            "No action was sent to a robot.",
            "",
            "## Test configuration",
            "",
            f"- Generated: {datetime.now(timezone.utc).isoformat()}",
            f"- Checkpoint: {checkpoint_info}",
            f"- Device: `{getattr(self.model, 'device', 'dummy/cpu')}`",
            f"- Episodes: {dataset_summary.get('test_episode_ids')}",
            f"- Selected frames: {dataset_summary.get('total_selected_frames')}",
            f"- Observation history: {self.model.capabilities.observation_history}",
            f"- Action chunk: {self.model.capabilities.chunk_length}",
            f"- Model action dimension: {self.model.capabilities.action_dim}",
            f"- Dataset/robot action dimension: {self.dataset.action_dim}",
            f"- Action adapter: `{self.action_adapter.adapter}`",
            f"- Observation source: `{dataset_summary.get('observation_source')}`",
            "",
            "## Data leakage",
            "",
        ]
        if self.leakage["warnings"]:
            lines.extend(f"- **WARNING:** {warning}" for warning in self.leakage["warnings"])
        else:
            lines.append("- No episode overlap was found from the available training metadata.")
        lines.extend(
            [
                "",
                "## Overall action metrics",
                "",
                "```json",
                json.dumps(
                    {
                        "model_space": metrics.get("model_space"),
                        "robot_joint_space": metrics.get("robot_joint_space"),
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                "```",
                "",
                "- `model_space` compares denormalized checkpoint outputs with recorded actions "
                "in the checkpoint-native representation. With `identity65`, this is the raw "
                "65-D robot joint target before safety filtering.",
                "- `robot_joint_space` compares safety-filtered actions with recorded actions. "
                "Under `violation_behavior: reject`, a rejected target is replaced by the "
                "measured robot state.",
                "",
                "## Stage and critical-event metrics",
                "",
                "Stage metrics are absent when no trusted dataset labels or interval annotation "
                "file is supplied; no heuristic labels are invented.",
                "",
                "```json",
                json.dumps(
                    {
                        "stages": metrics.get("stage_metrics"),
                        "critical_events": metrics.get("critical_event_metrics"),
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                "```",
                "",
                "## Safety and executability",
                "",
                f"- Summary: `{json.dumps(metrics.get('safety', {}), ensure_ascii=False)}`",
                "- URDF position and velocity limits are checked when configured.",
                "- IK, singularity, self-collision, and environment collision remain unavailable "
                "without a registered kinematics/collision implementation; their status is never "
                "fabricated.",
            ]
        )
        if self.action_adapter.adapter == "unidex82_to_north65":
            lines.append(
                "- The current 82-D UniDex checkpoint predicts wrist pose9d and hand joints, "
                "not a 65-D North joint target. The adapter applies only hand predictions and "
                "holds other joints at the observed state; wrist targets are recorded but not "
                "converted because no IK exists."
            )
        lines.extend(
            [
                "",
                "## Inference latency",
                "",
                "```json",
                json.dumps(latency, indent=2, ensure_ascii=False, default=_json_default),
                "```",
                "",
                "## Worst episodes",
                "",
            ]
        )
        lines.extend(
            f"- Episode {run.episode_id}: safe robot-joint MAE="
            f"{run.metrics_robot.get('action_mae')}"
            for run in worst
        )
        lines.extend(
            [
                "",
                "## Perturbation tests",
                "",
                "```json",
                json.dumps(
                    perturbations, indent=2, ensure_ascii=False, default=_json_default
                ),
                "```",
                "",
                "## Episode failures",
                "",
                "```json",
                json.dumps(
                    list(episode_errors),
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                "```",
                "",
                "## Boundary of this test",
                "",
                "This replay measures perception/inference plumbing, action prediction against "
                "demonstrations, timing, and static executability checks. It cannot measure "
                "closed-loop recovery, contact dynamics, compounding state distribution shift, "
                "or task success. Real closed-loop success requires a separately authorized robot "
                "test with an emergency-stop path and live state feedback.",
                "",
                "## Reuse for a future live Shadow Mode",
                "",
                "Reuse the model runner, action normalizer, North action adapter, safety checker, "
                "latency recorder, and artifact writer. Replace only the episode reader with a "
                "timestamped live observation source; keep the command sink disconnected until "
                "the adapter includes validated wrist IK and collision checking.",
                "",
            ]
        )
        (self.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> Path:
        """Execute replay, continue after per-episode failures, and write a report."""
        dataset_summary = self._print_and_save_startup_summary()
        resolved = dict(self.config)
        resolved["training_config_resolved"] = (
            None
            if self.training_config_path is None
            else str(self.training_config_path)
        )
        resolved["dataset"]["frame_stride_resolved"] = self.dataset.frame_stride
        resolved["action"]["adapter_resolved"] = self.action_adapter.adapter
        resolved["action"]["target_offset_steps_resolved"] = self.target_offset
        with (self.output_dir / "config_resolved.yaml").open(
            "w", encoding="utf-8"
        ) as stream:
            yaml.safe_dump(resolved, stream, sort_keys=False, allow_unicode=True)

        runs: List[EpisodeRun] = []
        loaded_episodes: Dict[int, EpisodeData] = {}
        errors: List[Dict[str, Any]] = []
        max_steps = self.config.get("max_steps_per_episode")
        for episode_id in self.selected_ids:
            print(f"SHADOW_REPLAY_EPISODE_START episode={episode_id}", flush=True)
            try:
                read_started = time.perf_counter()
                episode = self.dataset.load_episode(episode_id, max_steps=max_steps)
                read_latency_ms = (time.perf_counter() - read_started) * 1000.0
                loaded_episodes[episode_id] = episode
                run = self._run_episode(episode, read_latency_ms)
                runs.append(run)
                self._save_episode(run)
                print(
                    "SHADOW_REPLAY_EPISODE_DONE"
                    f" episode={episode_id}"
                    f" steps={len(run.timestamps)}"
                    f" native_mae={run.metrics_native.get('action_mae')}"
                    f" robot_mae={run.metrics_robot.get('action_mae')}"
                    f" violations={len(run.safety_violations)}",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    "episode_id": episode_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                errors.append(error)
                append_jsonl(self.output_dir / "episode_errors.jsonl", error)
                print(
                    f"SHADOW_REPLAY_EPISODE_FAILED episode={episode_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )

        overall = self._aggregate_metrics(runs)
        latency = self._latency_metrics(runs)
        perturbations = self._run_perturbations(
            {episode_id: loaded_episodes[episode_id] for episode_id in loaded_episodes if episode_id in {run.episode_id for run in runs}},
            {run.episode_id: run for run in runs},
        )
        write_json(self.output_dir / "overall_metrics.json", overall)
        write_json(self.output_dir / "latency_metrics.json", latency)
        write_json(self.output_dir / "perturbation_metrics.json", perturbations)

        metric_rows = []
        for run in runs:
            row: Dict[str, Any] = {
                "episode_id": run.episode_id,
                "source_length": run.source_length,
                "replay_steps": len(run.timestamps),
                "safety_violation_records": len(run.safety_violations),
            }
            _flatten("model", run.metrics_native, row)
            _flatten("robot", run.metrics_robot, row)
            metric_rows.append(row)
        _write_csv(self.output_dir / "metrics.csv", metric_rows)
        stage_rows = []
        for stage, values in overall.get("stage_metrics", {}).items():
            row = {"stage": stage}
            _flatten("", values, row)
            stage_rows.append(row)
        _write_csv(self.output_dir / "stage_metrics.csv", stage_rows)

        if runs and self.config.get("save_plots", True):
            all_latency = np.concatenate([run.latency["total_ms"] for run in runs])
            save_dataset_plots(
                self.output_dir / "visualizations",
                horizon_mae=overall["model_space"].get("horizon_mae", []),
                latency_ms=all_latency,
                stage_metrics=overall.get("stage_metrics", {}),
            )
        self._write_report(
            dataset_summary, overall, latency, runs, perturbations, errors
        )
        if not runs:
            raise RuntimeError(
                f"All {len(self.selected_ids)} episodes failed; see "
                f"{self.output_dir / 'episode_errors.jsonl'}"
            )
        return self.output_dir
