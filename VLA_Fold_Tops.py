from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import sys
import numpy as np
import torch
from termcolor import cprint
from PIL import Image

sys.path.append(os.getcwd())
from Env_StandAlone.FoldTops_Env_VLA import FoldTops_Env

# ------------------------------------ #
# ---        初始化仿真环境          --- #
# ------------------------------------ #
cprint("初始化仿真环境...", "cyan")
pos = np.array([0.0, 0.8, 0.2])
ori = np.array([0.0, 0.0, 0.0])
env = FoldTops_Env(pos, ori, None, None, False)
cprint("仿真环境初始化完成", "green")

# ------------------------------------ #
# ---        加载 OpenVLA 模型      --- #
# ------------------------------------ #
cprint("加载 OpenVLA 模型...", "cyan")
from transformers import AutoModelForVision2Seq, AutoProcessor

device = "cuda:0"
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True
)
vla_model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)
cprint("OpenVLA 模型加载完成", "green")

# ------------------------------------ #
# ---        VLA 控制循环           --- #
# ------------------------------------ #
instruction = "fold the shirt"
cprint(f"指令: {instruction}", "yellow")

for step in range(50):
    # 获取摄像头图像
    rgb = env.env_camera.get_rgb()
    image = Image.fromarray(rgb.astype(np.uint8))

    # OpenVLA 推理
    inputs = processor(instruction, image, return_tensors="pt").to(device)
    with torch.no_grad():
        action = vla_model.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False
        )

    action = action.cpu().numpy()
    cprint(f"Step {step}: action={action}", "cyan")

    # 获取当前末端执行器位置
    left_pos, left_ori = env.bimanual_dex.dexleft.end_effector.get_world_pose()
    right_pos, right_ori = env.bimanual_dex.dexright.end_effector.get_world_pose()

    # 用 action 更新双手位置
    new_left_pos = left_pos + action[:3] * 0.05
    new_right_pos = right_pos + action[:3] * 0.05

    env.bimanual_dex.dense_move_both_ik(
        new_left_pos, left_ori,
        new_right_pos, right_ori,
    )

    env.step()

cprint("VLA 控制完成", "green")
simulation_app.close()
