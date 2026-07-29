#!/usr/bin/env python3
"""Create head-camera robot projection overlays from saved T-Rex replay arrays."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from shadow_replay.dataset import LeRobotEpisodeDataset
from shadow_replay.robot_projection import CameraCalibration, NorthUrdfKinematics
from shadow_replay.visualization import save_robot_projection_video


def _path(value: str, base: Path = WORKSPACE) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=WORKSPACE / "configs/shadow_replay_trex.yaml")
    parser.add_argument("--results", type=Path, required=True, help="Saved episode_XXXX.npz")
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-stride", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    projection = config.get("visualization", {}).get("robot_projection", {})
    if not projection:
        raise KeyError("Config has no visualization.robot_projection block")
    dataset_path = (
        args.dataset_path
        if args.dataset_path is not None
        else _path(str(config["dataset_path"]))
    )
    dataset = LeRobotEpisodeDataset(dataset_path)
    results_path = args.results.expanduser().resolve()
    results = np.load(results_path)
    if args.episode_id is not None:
        episode_id = args.episode_id
    elif "episode_id" in results:
        episode_id = int(results["episode_id"])
    else:
        match = re.search(r"episode_(\d+)", results_path.name)
        if match is None:
            raise ValueError(
                "Could not infer episode ID from results filename; pass --episode-id"
            )
        episode_id = int(match.group(1))

    if "observation_frame_indices" in results:
        frame_indices = results["observation_frame_indices"]
    elif "frame_indices" in results:
        frame_indices = results["frame_indices"]
    else:
        raise KeyError("Results contain neither observation_frame_indices nor frame_indices")

    if "joint_state" in results:
        joint_states = results["joint_state"]
    else:
        episode = dataset.load_episode(episode_id)
        if "source_steps" not in results:
            raise KeyError(
                "Results have no joint_state; source_steps are required to reload it"
            )
        source_steps = np.asarray(results["source_steps"], dtype=np.int64)
        if np.any(source_steps < 0) or np.any(source_steps >= len(episode)):
            raise IndexError("source_steps contain indices outside the source episode")
        joint_states = episode.states[source_steps]

    prediction_key = (
        "pred_action_safe_chunks"
        if "pred_action_safe_chunks" in results
        else "pred_action_safe"
    )
    predicted = np.asarray(results[prediction_key])
    if predicted.ndim == 2:
        predicted = predicted[:, None, :]
    valid_mask = (
        results["valid_mask"]
        if "valid_mask" in results
        else np.ones(predicted.shape[:2], dtype=bool)
    )
    camera_key = config.get("visualization", {}).get(
        "camera_key", "observation.images.head_left"
    )
    kinematics = NorthUrdfKinematics(_path(str(projection["urdf_path"])))
    calibration = CameraCalibration.from_config(projection)
    max_frames = (
        args.max_frames
        if args.max_frames is not None
        else projection.get("max_frames")
    )
    output_stride = (
        args.output_stride
        if args.output_stride is not None
        else int(projection.get("output_stride", 1))
    )
    with dataset.open_video(episode_id, camera_key) as reader:
        save_robot_projection_video(
            args.output.expanduser().resolve(),
            video_reader=reader,
            frame_indices=frame_indices,
            current_joint_states=joint_states,
            predicted_joint_chunks=predicted,
            valid_mask=valid_mask,
            fps=dataset.fps / dataset.frame_stride,
            kinematics=kinematics,
            calibration=calibration,
            max_frames=max_frames,
            output_stride=output_stride,
        )
    print(f"Wrote {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
