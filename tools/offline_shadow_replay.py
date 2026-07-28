#!/usr/bin/env python3
"""Run offline-only Shadow Replay on whole robot-origami episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from shadow_replay.config import load_config
from shadow_replay.replay import ShadowReplay


def _episode_ids(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "episode IDs must be comma-separated integers"
        ) from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("episode IDs must be non-negative")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("episode IDs contain duplicates")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "shadow_replay.yaml",
        help="Shadow Replay YAML configuration.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episode-ids", type=_episode_ids, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable annotated source-camera videos.",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable PNG plots.",
    )
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for argument, key in (
        ("checkpoint", "checkpoint"),
        ("dataset_path", "dataset_path"),
        ("output_dir", "output_dir"),
        ("episode_ids", "episode_ids"),
        ("max_episodes", "max_episodes"),
        ("max_steps_per_episode", "max_steps_per_episode"),
        ("device", "device"),
        ("batch_size", "batch_size"),
        ("save_video", "save_video"),
        ("save_plots", "save_plots"),
    ):
        value = getattr(args, argument)
        if value is not None:
            values[key] = str(value) if isinstance(value, Path) else value
    return values


def main() -> int:
    args = parse_args()
    config = load_config(args.config, WORKSPACE_ROOT, overrides=_overrides(args))
    output = ShadowReplay(config).run()
    print(f"SHADOW_REPLAY_COMPLETE output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
