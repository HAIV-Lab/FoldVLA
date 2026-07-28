"""Isaac Lab configuration for the North POC2.2 humanoid."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
NORTH_POC2_2_USD_PATH = Path(
    os.environ.get(
        "NORTH_POC2_2_USD_PATH",
        _WORKSPACE_ROOT
        / "asserts"
        / "north_poc2_2_urdf_usd"
        / "north_poc2_2_v3_1"
        / "north_poc2_2_v3_1.usd",
    )
).expanduser().resolve()


NORTH_POC2_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(NORTH_POC2_2_USD_PATH),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        )
    },
)
