from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import sys
import numpy as np
from termcolor import cprint
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading

sys.path.append(os.getcwd())
from Env_StandAlone.Fold_Tops_Env import FoldTops
from Env_Config.Utils_Project.Parse import parse_args_record

# 先加载 Galbot 到场景
galbot_usd = "/new_data/why/isaac/galbot_one_golf_description/usd/galbot_one_golf.usda"
add_reference_to_stage(usd_path=galbot_usd, prim_path="/World/Galbot")

galbot = SingleXFormPrim(
    prim_path="/World/Galbot",
    name="galbot",
    position=np.array([-1.2, 0.8, 0.0]),  # 站在机器人旁边
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
)

# 正常跑叠衣服任务（带录像）
args = parse_args_record()
pos = np.array([0.0, 0.8, 0.2])
ori = np.array([0.0, 0.0, 0.0])
usd_path = None

FoldTops(pos, ori, usd_path, args.ground_material_usd, False, True)

simulation_app.close()
