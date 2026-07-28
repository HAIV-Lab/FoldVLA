"""Configuration loading and validation for offline Shadow Replay."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


class ConfigError(ValueError):
    """Raised when a Shadow Replay configuration is invalid."""


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(value: Optional[str], config_dir: Path, workspace_root: Path) -> Optional[str]:
    if value in (None, ""):
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    workspace_candidate = (workspace_root / path).resolve()
    config_candidate = (config_dir / path).resolve()
    if workspace_candidate.exists() or not config_candidate.exists():
        return str(workspace_candidate)
    return str(config_candidate)


def load_config(
    config_path: Path,
    workspace_root: Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load YAML, apply CLI overrides, resolve paths, and validate the result."""
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Shadow Replay config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {config_path}")

    config = copy.deepcopy(loaded)
    if overrides:
        _deep_update(config, overrides)

    config["_config_path"] = str(config_path)
    config["_workspace_root"] = str(workspace_root.resolve())
    for key in ("checkpoint", "dataset_path", "output_dir", "training_config"):
        config[key] = _resolve_path(config.get(key), config_path.parent, workspace_root)
    for section, key in (
        ("action", "normalizer_path"),
        ("safety", "urdf_path"),
        ("leakage", "training_manifest"),
        ("preprocessing", "source_zarr"),
        ("evaluation", "stage_annotations"),
        ("evaluation", "critical_events"),
        ("model", "trex_root"),
        ("model", "stats_path"),
    ):
        if section in config and isinstance(config[section], dict):
            config[section][key] = _resolve_path(
                config[section].get(key), config_path.parent, workspace_root
            )

    validate_config(config)
    return config


def _positive_int(config: Dict[str, Any], key: str, allow_none: bool = False) -> None:
    value = config.get(key)
    if allow_none and value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{key} must be a positive integer, got {value!r}")


def _indices(values: Any, name: str) -> None:
    if values is None:
        return
    if not isinstance(values, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values
    ):
        raise ConfigError(f"{name} must be null or a list of non-negative integer indices")
    if len(set(values)) != len(values):
        raise ConfigError(f"{name} contains duplicate indices")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate fields that cannot be safely inferred at runtime."""
    for required in ("dataset_path", "output_dir"):
        if not config.get(required):
            raise ConfigError(f"Missing required configuration field: {required}")

    _positive_int(config, "batch_size")
    _positive_int(config, "num_workers", allow_none=False) if config.get("num_workers", 0) else None
    _positive_int(config, "max_episodes", allow_none=True)
    _positive_int(config, "max_steps_per_episode", allow_none=True)

    dataset_cfg = config.setdefault("dataset", {})
    frame_stride = dataset_cfg.get("frame_stride", 1)
    if frame_stride != "auto" and (
        not isinstance(frame_stride, int)
        or isinstance(frame_stride, bool)
        or frame_stride < 1
    ):
        raise ConfigError(
            f"dataset.frame_stride must be 'auto' or at least 1, got {frame_stride!r}"
        )
    episode_ids = config.get("episode_ids")
    _indices(episode_ids, "episode_ids")

    action = config.setdefault("action", {})
    for key in ("xyz_indices", "rotation_indices", "gripper_indices"):
        _indices(action.get(key), f"action.{key}")
    action_type = action.get("type", "auto")
    if action_type not in {"auto", "absolute", "delta"}:
        raise ConfigError(f"action.type must be auto, absolute, or delta, got {action_type!r}")
    adapter = action.get("adapter", "auto")
    if adapter not in {"auto", "identity65", "unidex82_to_north65"}:
        raise ConfigError(f"Unsupported action.adapter: {adapter!r}")

    model = config.setdefault("model", {})
    backend = model.get("backend", "unidex")
    if backend not in {"unidex", "trex", "dummy"}:
        raise ConfigError(f"Unsupported model.backend: {backend!r}")
    if backend == "trex":
        image_size = model.get("image_size", [384, 288])
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                for value in image_size
            )
        ):
            raise ConfigError("model.image_size must be [width, height]")
        lora = model.get("action_lora") or {}
        if int(lora.get("rank", 16)) < 1:
            raise ConfigError("model.action_lora.rank must be positive")

    evaluation = config.setdefault("evaluation", {})
    window = evaluation.get("critical_event_window", 5)
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ConfigError("evaluation.critical_event_window must be a non-negative integer")

    safety = config.setdefault("safety", {})
    behavior = safety.get("violation_behavior", "reject")
    if behavior not in {"reject", "clip", "report_only"}:
        raise ConfigError(
            "safety.violation_behavior must be reject, clip, or report_only"
        )
    for key in (
        "max_cartesian_step",
        "max_rotation_step_deg",
        "max_joint_velocity",
        "max_joint_acceleration",
    ):
        value = safety.get(key)
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ConfigError(f"safety.{key} must be null or positive")
    for key in ("workspace_min", "workspace_max"):
        value = safety.get(key)
        if value is not None and (
            not isinstance(value, list)
            or len(value) != 3
            or any(not isinstance(item, (int, float)) for item in value)
        ):
            raise ConfigError(f"safety.{key} must be null or [x, y, z]")
    if safety.get("workspace_min") is not None and safety.get("workspace_max") is not None:
        if any(
            low >= high
            for low, high in zip(safety["workspace_min"], safety["workspace_max"])
        ):
            raise ConfigError("Every workspace_min coordinate must be below workspace_max")

    perturbation = config.setdefault("perturbation", {})
    probability = perturbation.get("drop_frame_probability", 0.0)
    if not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
        raise ConfigError("perturbation.drop_frame_probability must be in [0, 1]")
    delay = perturbation.get("observation_delay_frames", 0)
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
        raise ConfigError("perturbation.observation_delay_frames must be non-negative")


def discover_training_config(checkpoint: Path, explicit: Optional[str]) -> Path:
    """Return the training YAML associated with a checkpoint."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Training config does not exist: {path}")
        return path.resolve()
    candidates = sorted(checkpoint.parent.glob("config-*.yaml"))
    if not candidates:
        raise FileNotFoundError(
            "Could not infer the UniDex training config. Set training_config explicitly; "
            f"no config-*.yaml exists beside {checkpoint}."
        )
    return candidates[-1].resolve()


def select_episode_ids(
    available: Iterable[int],
    requested: Optional[Iterable[int]],
    max_episodes: Optional[int],
) -> list[int]:
    """Select whole episodes without ever splitting or shuffling frames."""
    available_ids = sorted(int(value) for value in available)
    if requested is None:
        selected = available_ids
    else:
        requested_ids = [int(value) for value in requested]
        missing = sorted(set(requested_ids) - set(available_ids))
        if missing:
            raise ConfigError(
                f"Requested episode IDs are absent: {missing}; available={available_ids}"
            )
        selected = requested_ids
    return selected if max_episodes is None else selected[:max_episodes]
