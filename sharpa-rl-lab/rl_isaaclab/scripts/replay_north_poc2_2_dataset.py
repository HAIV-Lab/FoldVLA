"""Replay one LeRobot trajectory on the North POC2.2 articulation and record it.

The LeRobot dataset stores joints in this order:
left arm, left hand, right arm, right hand, lower body and neck.  The USD
articulation has a different joint order, so this script maps every value by
joint name before writing it to PhysX.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATASET = (
    _WORKSPACE_ROOT
    / "dataset"
    / "Robotic_Origami_Challenge"
    / "season_POC22032_2026_05_14_19_21_01_train"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset",
    type=Path,
    default=_DEFAULT_DATASET,
    help="Dataset root (either the season directory or its lerobot3.0 directory).",
)
parser.add_argument("--episode_index", type=int, default=0, help="Episode to replay.")
parser.add_argument(
    "--state_key",
    type=str,
    default="observation.state",
    help="Parquet vector field containing measured joint positions.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output MP4 path. Defaults to sharpa-rl-lab/videos/north_poc2_2_episode_<index>.mp4.",
)
parser.add_argument("--resolution", type=str, default="1280,720", help="Video resolution as width,height.")
parser.add_argument(
    "--camera_eye",
    type=str,
    default="2.8,-3.2,1.8",
    help="World-space camera position as x,y,z.",
)
parser.add_argument(
    "--camera_lookat",
    type=str,
    default="0.0,0.0,1.0",
    help="World-space camera target as x,y,z.",
)
parser.add_argument("--fps", type=float, default=None, help="Override the dataset FPS.")
parser.add_argument(
    "--start_frame",
    type=int,
    default=0,
    help="First episode-local frame to record.",
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=None,
    help="Maximum number of frames to record (the complete episode by default).",
)
parser.add_argument(
    "--frame_stride",
    type=int,
    default=1,
    help="Record every Nth dataset frame; output FPS is divided by N.",
)
parser.add_argument(
    "--warmup_frames",
    type=int,
    default=12,
    help="Renderer warm-up frames before recording.",
)
parser.add_argument(
    "--enable_collisions",
    action="store_true",
    help="Enable mesh collisions. They are unnecessary for direct state replay and slow initial loading.",
)
parser.add_argument(
    "--fast_exit",
    action="store_true",
    help="Exit immediately after finalizing the MP4, skipping slow Kit extension cleanup.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Headless Replicator rendering requires this even though there is no Camera
# sensor in an Isaac Lab scene.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import subprocess

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from rl_isaaclab.assets import NORTH_POC2_2_CFG, NORTH_POC2_2_USD_PATH


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


def _parse_tuple(value: str, name: str, length: int, cast=float) -> tuple:
    items = tuple(cast(item.strip()) for item in value.split(","))
    if len(items) != length:
        raise ValueError(f"{name} must contain {length} comma-separated values, got: {value}")
    return items


def _find_lerobot_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = (path, path / "lerobot3.0")
    for candidate in candidates:
        if (candidate / "meta" / "info.json").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find meta/info.json and data/ under either {path} or {path / 'lerobot3.0'}"
    )


def _expected_dataset_names() -> list[str]:
    names = [f"left_arm_j{i}" for i in range(7)]
    names += [f"left_hand_j{i}" for i in range(22)]
    names += [f"right_arm_j{i}" for i in range(7)]
    names += [f"right_hand_j{i}" for i in range(22)]
    names += [f"motor_j{i}" for i in range(7)]
    return names


def _dataset_to_urdf_names() -> list[str]:
    names = [f"left_arm_joint_{i}" for i in range(1, 8)]
    names += [f"left_{suffix}" for suffix in _HAND_JOINT_SUFFIXES]
    names += [f"right_arm_joint_{i}" for i in range(1, 8)]
    names += [f"right_{suffix}" for suffix in _HAND_JOINT_SUFFIXES]
    names += [f"lower_body_joint_{i}" for i in range(1, 6)]
    names += ["neck_joint_1", "neck_joint_2"]
    return names


def _load_episode(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as stream:
        info = json.load(stream)

    features = info.get("features", {})
    if args_cli.state_key not in features:
        raise KeyError(f"State key {args_cli.state_key!r} is not present in meta/info.json")
    dataset_names = features[args_cli.state_key].get("names")
    expected_names = _expected_dataset_names()
    if dataset_names != expected_names:
        raise ValueError(
            "The dataset joint schema is not the expected North POC2.2 schema.\n"
            f"Expected: {expected_names}\n"
            f"Found:    {dataset_names}"
        )

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    table = pq.read_table(
        parquet_files,
        columns=[args_cli.state_key, "episode_index", "frame_index", "timestamp"],
    )
    episode = table.filter(pc.equal(table["episode_index"], pa.scalar(args_cli.episode_index)))
    if episode.num_rows == 0:
        available = sorted(set(table["episode_index"].to_pylist()))
        raise ValueError(f"Episode {args_cli.episode_index} is absent; available episodes: {available}")
    episode = episode.sort_by([("frame_index", "ascending")])

    frame_indices = episode["frame_index"].to_numpy(zero_copy_only=False)
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise ValueError(f"Episode {args_cli.episode_index} contains duplicate frame indices")
    states = np.asarray(episode[args_cli.state_key].to_pylist(), dtype=np.float32)
    timestamps = episode["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    if states.shape != (episode.num_rows, 65):
        raise ValueError(f"Expected episode state shape ({episode.num_rows}, 65), got {states.shape}")
    if not np.isfinite(states).all():
        raise ValueError("Episode contains non-finite joint positions")

    start = args_cli.start_frame
    if start < 0 or start >= len(states):
        raise ValueError(f"--start_frame must be in [0, {len(states) - 1}], got {start}")
    stop = len(states) if args_cli.max_frames is None else min(len(states), start + args_cli.max_frames)
    states = states[start:stop:args_cli.frame_stride]
    timestamps = timestamps[start:stop:args_cli.frame_stride]
    if len(states) == 0:
        raise ValueError("The selected frame range is empty")

    dataset_fps = float(info["fps"])
    fps = float(args_cli.fps) if args_cli.fps is not None else dataset_fps
    output_fps = fps / args_cli.frame_stride
    return states, timestamps, output_fps


def _verify_video(output_path: Path, expected_frames: int) -> None:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames:format=duration,size",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    actual_frames = int(stream["nb_read_frames"])
    if actual_frames != expected_frames:
        raise RuntimeError(f"Video contains {actual_frames} frames, expected {expected_frames}")
    print(
        "VIDEO_WITNESS"
        f" path={output_path}"
        f" frames={actual_frames}"
        f" resolution={stream['width']}x{stream['height']}"
        f" fps={stream['avg_frame_rate']}"
        f" duration={metadata['format']['duration']}"
        f" bytes={metadata['format']['size']}",
        flush=True,
    )


def main() -> Path:
    if args_cli.episode_index < 0:
        raise ValueError("--episode_index must be non-negative")
    if args_cli.frame_stride < 1:
        raise ValueError("--frame_stride must be at least 1")
    if args_cli.max_frames is not None and args_cli.max_frames < 1:
        raise ValueError("--max_frames must be at least 1")
    if args_cli.warmup_frames < 0:
        raise ValueError("--warmup_frames must be non-negative")
    if not NORTH_POC2_2_USD_PATH.is_file():
        raise FileNotFoundError(f"North POC2.2 USD not found: {NORTH_POC2_2_USD_PATH}")

    dataset_root = _find_lerobot_root(args_cli.dataset)
    states, timestamps, output_fps = _load_episode(dataset_root)
    width, height = _parse_tuple(args_cli.resolution, "--resolution", 2, int)
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("--resolution dimensions must be positive even integers for yuv420p video")
    camera_eye = _parse_tuple(args_cli.camera_eye, "--camera_eye", 3)
    camera_lookat = _parse_tuple(args_cli.camera_lookat, "--camera_lookat", 3)
    output_path = (
        args_cli.output.expanduser().resolve()
        if args_cli.output is not None
        else (
            _WORKSPACE_ROOT
            / "sharpa-rl-lab"
            / "videos"
            / f"north_poc2_2_episode_{args_cli.episode_index}.mp4"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "REPLAY_REQUEST"
        f" dataset={dataset_root}"
        f" episode={args_cli.episode_index}"
        f" frames={len(states)}"
        f" source_time={timestamps[0]:.6f}:{timestamps[-1]:.6f}"
        f" output_fps={output_fps:.6f}"
        f" usd={NORTH_POC2_2_USD_PATH}"
        f" output={output_path}",
        flush=True,
    )

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / output_fps,
        render_interval=1,
        device=args_cli.device,
        gravity=(0.0, 0.0, 0.0),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=camera_eye, target=camera_lookat)
    ground_cfg = sim_utils.GroundPlaneCfg(color=(0.18, 0.18, 0.18))
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/DomeLight", light_cfg)

    robot_cfg = NORTH_POC2_2_CFG.replace(prim_path="/World/NorthPOC2_2")
    if not args_cli.enable_collisions:
        robot_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    robot = Articulation(robot_cfg)
    sim.reset()

    if not robot.is_initialized or robot.num_joints != 65:
        raise RuntimeError(
            f"North POC2.2 must initialize as a 65-joint articulation; "
            f"initialized={robot.is_initialized}, joints={robot.num_joints}"
        )
    urdf_names = _dataset_to_urdf_names()
    missing = sorted(set(urdf_names) - set(robot.joint_names))
    if missing:
        raise RuntimeError(f"Articulation is missing mapped joints: {missing}")
    articulation_indices = torch.tensor(
        [robot.joint_names.index(name) for name in urdf_names],
        dtype=torch.long,
        device=sim.device,
    )
    if len(set(articulation_indices.tolist())) != 65:
        raise RuntimeError("Dataset-to-articulation mapping is not one-to-one")
    print(
        "ASSET_WITNESS"
        f" initialized={robot.is_initialized}"
        f" bodies={robot.num_bodies}"
        f" joints={robot.num_joints}"
        f" mapping=65/65"
        f" device={sim.device}",
        flush=True,
    )

    # Map once on the GPU and enforce the physical limits supplied by the URDF.
    dataset_states = torch.from_numpy(states).to(device=sim.device)
    articulation_states = torch.empty_like(dataset_states)
    articulation_states[:, articulation_indices] = dataset_states
    limits = robot.data.soft_joint_pos_limits[0]
    clipped_states = torch.clamp(articulation_states, min=limits[:, 0], max=limits[:, 1])
    clipped_values = int(torch.count_nonzero(clipped_states != articulation_states).item())
    max_clip = float(torch.max(torch.abs(clipped_states - articulation_states)).item())
    zero_velocity = torch.zeros((1, robot.num_joints), device=sim.device)
    print(
        "LIMIT_WITNESS"
        f" clipped_values={clipped_values}"
        f" total_values={clipped_states.numel()}"
        f" max_adjustment_rad={max_clip:.9f}",
        flush=True,
    )

    import omni.replicator.core as rep

    render_product = rep.create.render_product("/OmniverseKit_Persp", resolution=(width, height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator.attach([render_product])

    first_state = clipped_states[0].unsqueeze(0)
    robot.write_joint_state_to_sim(first_state, zero_velocity)
    for _ in range(args_cli.warmup_frames):
        sim.render()

    writer = imageio.get_writer(
        output_path,
        format="FFMPEG",
        mode="I",
        fps=output_fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart"],
    )
    try:
        progress_interval = max(1, min(300, len(clipped_states) // 20))
        for frame_number, joint_position in enumerate(clipped_states):
            robot.write_joint_state_to_sim(joint_position.unsqueeze(0), zero_velocity)
            sim.render()
            rgba = np.asarray(rgb_annotator.get_data())
            if rgba.size == 0:
                raise RuntimeError(f"Renderer returned an empty image at output frame {frame_number}")
            if rgba.shape[:2] != (height, width) or rgba.shape[2] < 3:
                raise RuntimeError(
                    f"Renderer returned shape {rgba.shape}, expected ({height}, {width}, 3 or 4)"
                )
            writer.append_data(np.ascontiguousarray(rgba[:, :, :3]))
            if frame_number == 0 or (frame_number + 1) % progress_interval == 0:
                print(
                    "REPLAY_PROGRESS"
                    f" frame={frame_number + 1}/{len(clipped_states)}"
                    f" dataset_time={timestamps[frame_number]:.6f}",
                    flush=True,
                )
    finally:
        writer.close()
        # Replicator teardown can take several minutes for this mesh-heavy asset.
        # With --fast_exit the process is about to terminate, so detaching is
        # unnecessary and would defeat the purpose of that option.
        if not args_cli.fast_exit:
            rgb_annotator.detach([render_product])

    _verify_video(output_path, expected_frames=len(clipped_states))
    return output_path


if __name__ == "__main__":
    try:
        completed_output = main()
    except BaseException:
        simulation_app.close(wait_for_replicator=False)
        raise
    print(f"[INFO] Replay complete: {completed_output}", flush=True)
    if args_cli.fast_exit:
        os._exit(0)
    simulation_app.close(wait_for_replicator=False)
