"""Load the North POC2.2 USD in Isaac Lab and run a finite physics smoke test."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=20, help="Number of PhysX steps to execute.")
parser.add_argument(
    "--disable_collisions",
    action="store_true",
    help="Disable mesh collisions for a fast articulation-only smoke test.",
)
parser.add_argument(
    "--fast_exit",
    action="store_true",
    help="Exit immediately after a successful witness to skip slow Kit plugin cleanup.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from rl_isaaclab.assets import NORTH_POC2_2_CFG, NORTH_POC2_2_USD_PATH


def main() -> None:
    if args_cli.steps < 1:
        raise ValueError("--steps must be at least 1")
    if not NORTH_POC2_2_USD_PATH.is_file():
        raise FileNotFoundError(f"North POC2.2 USD not found: {NORTH_POC2_2_USD_PATH}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    robot_cfg = NORTH_POC2_2_CFG.replace(prim_path="/World/NorthPOC2_2")
    if args_cli.disable_collisions:
        robot_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    print(
        "LOAD_REQUEST"
        f" usd={NORTH_POC2_2_USD_PATH}"
        f" device={args_cli.device}"
        f" collisions={not args_cli.disable_collisions}",
        flush=True,
    )
    robot = Articulation(robot_cfg)
    sim.reset()

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    print(
        "GPU_WITNESS"
        f" requested={args_cli.device}"
        f" torch_device=cuda:{device_index}"
        f" name={device_name}",
        flush=True,
    )
    print(
        "ASSET_WITNESS"
        f" usd={NORTH_POC2_2_USD_PATH}"
        f" initialized={robot.is_initialized}"
        f" bodies={robot.num_bodies}"
        f" joints={robot.num_joints}",
        flush=True,
    )

    if not robot.is_initialized or robot.num_bodies < 1 or robot.num_joints < 1:
        raise RuntimeError("North POC2.2 did not initialize as a non-empty articulation")
    if "RTX 4090" not in device_name:
        raise RuntimeError(f"Expected an RTX 4090, but PhysX/PyTorch selected {device_name}")

    sim_dt = sim.get_physics_dt()
    for _ in range(args_cli.steps):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)

    if not torch.isfinite(robot.data.root_pos_w).all():
        raise RuntimeError("Non-finite root state after physics stepping")
    print(
        "PHYSICS_WITNESS"
        f" steps={args_cli.steps}"
        f" root_shape={tuple(robot.data.root_pos_w.shape)}"
        f" finite={bool(torch.isfinite(robot.data.root_pos_w).all())}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        simulation_app.close()
        raise
    if args_cli.fast_exit:
        os._exit(0)
    simulation_app.close(wait_for_replicator=False)
