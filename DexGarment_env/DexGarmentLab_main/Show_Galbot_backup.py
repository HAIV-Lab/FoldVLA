from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "livestream": 2})

import os
import sys
import numpy as np
from termcolor import cprint
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading

sys.path.append(os.getcwd())

# 初始化世界
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 加载衣服（Fold_Tops 默认衣服）
garment_usd = os.getcwd() + "/Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_018/TCLC_018_obj.usd"
add_reference_to_stage(usd_path=garment_usd, prim_path="/World/Garment")

# 加载 Galbot 机器人（站在场景旁边）
galbot_usd = "/new_data/why/isaac/galbot_one_golf_description/usd/galbot_one_golf.usda"
add_reference_to_stage(usd_path=galbot_usd, prim_path="/World/Galbot")

# 设置 Galbot 位置（站在衣服旁边，稍微偏后）
from isaacsim.core.prims import SingleXFormPrim
galbot = SingleXFormPrim(
    prim_path="/World/Galbot",
    name="galbot",
    position=np.array([0.0, -0.5, 0.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
)

# 加载 SharpaWave 双手（挂在 Galbot 手腕位置）
sharpa_usd = "/new_data/why/isaac/sharpa-urdf-usd-xml/wave_01/dual_sharpa_wave/dual_sharpa_wave_with_wrist.usda"
add_reference_to_stage(usd_path=sharpa_usd, prim_path="/World/SharpaHands")

sharpa = SingleXFormPrim(
    prim_path="/World/SharpaHands",
    name="sharpa_hands",
    position=np.array([0.0, -0.5, 1.2]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
)

cprint("场景加载完成，等待仿真...", "green")

# 重置世界
world.reset()

# 等待场景加载
while is_stage_loading():
    world.step(render=True)

cprint("World Ready! 开始展示...", "green")

# 保持运行（WebRTC 连接用）
step = 0
while simulation_app.is_running():
    world.step(render=True)
    step += 1
    if step % 500 == 0:
        cprint(f"运行中... step {step}", "cyan")

simulation_app.close()
