#!/usr/bin/env python3
"""Sample horizon-0 T-Rex errors across the full duration of one episode."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from shadow_replay.config import load_config
from shadow_replay.replay import ShadowReplay
from shadow_replay.visualization import save_horizon0_error_over_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--sample-every-frames", type=int, default=30)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_every_frames < 1:
        raise ValueError("--sample-every-frames must be positive")
    if args.start_step < 0:
        raise ValueError("--start-step must be non-negative")

    output_dir = args.output_dir.resolve()
    overrides = {
        "episode_ids": [args.episode_id],
        "max_episodes": 1,
        "max_steps_per_episode": None,
        "output_dir": str(output_dir),
        "save_plots": False,
    }
    if args.device is not None:
        overrides["device"] = args.device
    config = load_config(
        args.config,
        WORKSPACE,
        overrides=overrides,
    )
    replay = ShadowReplay(config)
    episode = replay.dataset.load_episode(args.episode_id)
    end_step = len(episode) if args.end_step is None else args.end_step
    if not args.start_step < end_step <= len(episode):
        raise ValueError(
            f"Expected 0 <= start < end <= {len(episode)}, got "
            f"{args.start_step}/{end_step}"
        )
    selected = np.arange(
        args.start_step,
        end_step,
        args.sample_every_frames,
        dtype=np.int64,
    )
    if selected[-1] != end_step - 1:
        selected = np.append(selected, end_step - 1)

    preprocessing = replay.config.get("preprocessing", {})
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
    tactile = np.asarray(episode.optional_fields[tactile_key], dtype=np.float32)
    tactile_frames = tactile.reshape(len(episode), 10, 6)
    history_length = int(replay.model.capabilities.tactile_history)

    decoded = {key: [] for key in camera_keys}
    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        readers = {
            name: stack.enter_context(
                replay.dataset.open_video(args.episode_id, dataset_key)
            )
            for name, dataset_key in camera_keys.items()
        }
        for sample_number, source_step in enumerate(selected):
            frame_index = int(episode.frame_indices[source_step])
            for name, reader in readers.items():
                image = reader.read(frame_index)
                decoded[name].append(
                    replay._split_tactile_deform(image)
                    if name == "tactile_deform"
                    else image
                )
            if (sample_number + 1) % 50 == 0:
                print(
                    f"decoded {sample_number + 1}/{len(selected)} sampled frames",
                    flush=True,
                )

    histories = []
    for source_step in selected:
        first = max(0, int(source_step) - history_length + 1)
        history = tactile_frames[first : int(source_step) + 1]
        if len(history) < history_length:
            history = np.concatenate(
                [
                    np.repeat(
                        history[:1], history_length - len(history), axis=0
                    ),
                    history,
                ],
                axis=0,
            )
        histories.append(history)
    observations = {
        **{key: np.stack(value) for key, value in decoded.items()},
        "tactile_f6_history": np.stack(histories),
    }
    print(
        f"decoded {len(selected)} samples across "
        f"{episode.timestamps[-1] - episode.timestamps[0]:.3f}s in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )

    replay.model.reset_episode()
    prompt_template = preprocessing.get("prompt_template", "{instruction}")
    raw_chunks = []
    safe_chunks = []
    inference_ms = []
    for sample_number, source_step in enumerate(selected):
        observation = {
            key: value[sample_number : sample_number + 1]
            for key, value in observations.items()
        }
        state = episode.states[source_step : source_step + 1, None, :]
        prompt = prompt_template.format(
            instruction=episode.instructions[source_step]
        )
        seed = (
            int(replay.config.get("seed", 42))
            + args.episode_id * 100_000
            + int(source_step)
        )
        if not replay.warmup_done:
            replay.model.warmup(
                None,
                state,
                [prompt],
                iterations=int(replay.config.get("warmup_iterations", 1)),
                seed=seed + 800_000,
                observations=observation,
            )
            replay.warmup_done = True
        output = replay.model.predict(
            None,
            state,
            [prompt],
            seed=seed,
            observations=observation,
        )
        denormalized = replay.action_normalizer.denormalize(output.action_raw)[0]
        adapted = replay.action_adapter.model_to_robot(
            denormalized, episode.states[source_step]
        )
        raw_chunks.append(adapted.robot_joint_target)
        safe_chunk = []
        previous_action = None
        previous_velocity = None
        dt = 1.0 / replay.dataset.fps
        for target in adapted.robot_joint_target:
            safety = replay.safety_checker.check(
                target,
                episode.states[source_step],
                previous_action=previous_action,
                previous_velocity=previous_velocity,
                dt=dt,
            )
            safe_chunk.append(safety.action_safe)
            velocity = (
                None
                if previous_action is None
                else (safety.action_safe - previous_action) / dt
            )
            previous_action = safety.action_safe
            if velocity is not None:
                previous_velocity = velocity
        safe_chunks.append(np.stack(safe_chunk))
        inference_ms.append(output.inference_latency_ms)
        if (sample_number + 1) % 25 == 0:
            print(
                f"inferred {sample_number + 1}/{len(selected)} samples",
                flush=True,
            )

    raw_chunks_array = np.stack(raw_chunks)
    safe_chunks_array = np.stack(safe_chunks)
    raw_horizon0 = raw_chunks_array[:, 0]
    safe_horizon0_array = safe_chunks_array[:, 0]
    gt_horizon0 = episode.actions[selected]
    timestamps = episode.timestamps[selected]
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    plot_path = (
        visualization_dir
        / f"episode_{args.episode_id:04d}_horizon0_error_full_trajectory.png"
    )
    save_horizon0_error_over_time(
        plot_path,
        timestamps=timestamps,
        gt_action=gt_horizon0,
        pred_action_raw=raw_horizon0,
        pred_action_safe=safe_horizon0_array,
        dataset_lower=replay.dataset_action_lower,
        dataset_upper=replay.dataset_action_upper,
    )
    np.savez_compressed(
        output_dir / f"episode_{args.episode_id:04d}_horizon0_samples.npz",
        source_steps=selected,
        frame_indices=episode.frame_indices[selected],
        timestamps=timestamps,
        gt_action=gt_horizon0,
        pred_action_raw=raw_horizon0,
        pred_action_safe=safe_horizon0_array,
        pred_action_raw_chunks=raw_chunks_array,
        pred_action_safe_chunks=safe_chunks_array,
        latency_inference_ms=np.asarray(inference_ms),
    )
    clipped_error = np.abs(
        np.clip(
            raw_horizon0,
            replay.dataset_action_lower,
            replay.dataset_action_upper,
        )
        - np.clip(
            gt_horizon0,
            replay.dataset_action_lower,
            replay.dataset_action_upper,
        )
    )
    summary = {
        "episode_id": args.episode_id,
        "source_frame_count": len(episode),
        "sample_count": len(selected),
        "start_step": args.start_step,
        "end_step_exclusive": end_step,
        "sample_every_frames": args.sample_every_frames,
        "sample_period_seconds": args.sample_every_frames / replay.dataset.fps,
        "trajectory_duration_seconds": float(timestamps[-1] - timestamps[0]),
        "dataset_clipped_horizon0_mae_rad": float(clipped_error.mean()),
        "dataset_clipped_per_sample_mae_rad": clipped_error.mean(axis=1).tolist(),
        "mean_inference_latency_ms": float(np.mean(inference_ms)),
        "saved_action_chunk_shape": list(raw_chunks_array.shape),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(plot_path, flush=True)


if __name__ == "__main__":
    main()
