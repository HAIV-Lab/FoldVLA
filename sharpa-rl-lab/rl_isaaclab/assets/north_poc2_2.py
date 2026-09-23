"""Isaac Lab configuration for the North POC2.2 humanoid."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


def _resolve_workspace_root() -> Path:
    """Find the workspace containing the North USD asset.

    The repository is sometimes checked out as a sibling of ``asserts`` and
    sometimes as ``foldVLA/sharpa-rl-lab``.  Supporting both layouts keeps the
    Isaac Lab scripts runnable from the current workspace without requiring an
    environment variable.
    """

    source_path = Path(__file__).resolve()
    asset_relative_path = Path(
        "asserts",
        "north_poc2_2_urdf_usd",
        "north_poc2_2_v3_1",
        "north_poc2_2_v3_1.usd",
    )
    for candidate in (source_path.parents[2], source_path.parents[3]):
        if (candidate / asset_relative_path).is_file():
            return candidate
    # Preserve the historical fallback so an informative USD-not-found error
    # is raised by the caller if neither layout is present.
    return source_path.parents[3]


NORTH_POC2_2_WORKSPACE_ROOT = _resolve_workspace_root()
_WORKSPACE_ROOT = NORTH_POC2_2_WORKSPACE_ROOT
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
