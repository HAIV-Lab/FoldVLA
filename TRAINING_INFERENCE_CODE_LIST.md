# 机器人模型训练代码与推理代码清单

扫描根目录：`/data/why/foldVLA`

扫描日期：2026-09-21

这份文档按“训练入口 → 训练核心 → 推理入口 → 部署/评估 → 本地权重”的链路整理当前工作区。路径均相对于 `/data/why/foldVLA`，可直接在编辑器中打开。

## 0. 总览

| 项目/模型 | 训练代码 | 推理/部署代码 | 当前本地权重状态 |
|---|---|---|---|
| T-Rex Origami 65-D tactile VLA | `T-Rex/scripts/train_origami_freeze_vlm.py` 及其多 GPU/Docker 启动脚本 | `T-Rex/scripts/test.py`、`lora_test.py`、Office Zenoh 服务、机器人客户端 | 已发现 `checkpoint-0-50000`、`checkpoint-0-11407`、`checkpoint-0-7000` 等；`checkpoint-0-11407` 已确认属于 T-Rex Origami 训练链 |
| T-Rex Midtrain | 当前仓库 main 分支提供 checkpoint，完整 pretrain/midtrain 代码在上游 `full-pipeline` 分支说明中 | 由 T-Rex post-train/inference 代码加载 | 已发现 `checkpoints/T-Rex-midTrain` |
| GR00T N1.7 Origami | `Isaac-GR00T/examples/Origami/train_origami_n1d7.sh` → GR00T finetune trainer | standalone、GR00T server、open-loop/rollout、TensorRT | 已发现 GR00T N1.7 基础快照；当前工作区未发现 Origami 微调成品权重 |
| Sharpa Wave RL | `sharpa-rl-lab/rl_isaaclab/scripts/train.py` | `play.py`、`deploy.py` | 已发现 `.pth` 权重 |
| DexGarment Diffusion Policy | `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/train.py` | `DP.py`、`RobotWorkspace`、`DPRunner` | 代码存在；未发现本地 `.ckpt` |
| DexGarment Diffusion Policy 3D / DP3 | `.../Diffusion_Policy_3D/train.py` | `DP3.py`、`robot_runner.py` | 代码存在；未发现本地 `.ckpt` |
| HALO / GAM / SADP / SADP-G | 当前目录只有环境/验证包装代码和文档 | `Env_Validation/*_HALO.py`、`Env_StandAlone/*.py` 引用 `Model_HALO` | `Model_HALO` 主体目录未出现在当前工作区，不能视为完整可运行链路 |
| Shadow Replay | 无训练代码；是离线评测 | `shadow_replay/trex.py`、`replay.py` | 可读取 T-Rex backend；默认配置仍含历史 checkpoint 路径 |
| office/openpi 参考框架 | 含通用训练模块，但不是当前已确认的本地模型训练产物 | WebSocket/Zenoh policy server、client | 参考部署框架；未发现与本地 T-Rex/GR00T 权重绑定的 openpi 成品 |

状态标记：

- **已确认权重**：代码入口和本地模型文件可以对应起来。
- **代码存在/权重未发现**：训练或推理源码存在，但扫描目录没有对应成品 checkpoint。
- **引用缺失**：当前文件会 import 某个模型包，但该模型包不在当前工作区。
- **上游/参考**：用于通用训练、仿真或部署的框架代码，不代表当前工作区的某个成品模型。

## 1. 本地模型与 checkpoint 总表

### 1.1 T-Rex/Qwen 本地权重

| 路径 | 类型 | 关键内容 | 备注 |
|---|---|---|---|
| `checkpoints/Qwen3-VL-2B-Instruct/` | Qwen3-VL base model | `model.safetensors`、tokenizer、processor 配置 | T-Rex 和 GR00T Origami 集成都会用到 Qwen3-VL 相关资产；不要当作任务微调模型 |
| `checkpoints/T-Rex-midTrain/` | T-Rex midtrain | `model.pt`、`deform_encoder_from_model_pt.pth`、`training_args.json`、processor、统计量 | action dim 62；用于继续 Origami/task-specific post-train |
| `checkpoints/T-Rex-origami-posttrain/checkpoint-0-50000/` | T-Rex Origami post-train | `model.pt`、deform encoder、processor、统计量、训练参数 | action dim 65、chunk 16；对应 2026-08-19 submission manifest |
| `checkpoints/checkpoint-0-11407/` | T-Rex Origami post-train | `model.pt`、processor、统计量、训练参数 | action dim 65、chunk 16、Action-LoRA/VLM-LoRA；对应 2026-09-01 submission manifest |
| `checkpoints/checkpoint-0-7000/` | T-Rex Origami post-train | `model.pt`、processor、统计量、训练参数 | action dim 65、chunk 16、Action-LoRA/VLM-LoRA；对应 2026-09-18 submission manifest |
| `checkpoints/trex_ckpt_7000_best.tar.gz/trex_ckpt_7000_best.tar.gz` | T-Rex 历史归档 | 压缩 checkpoint | 需要先确认归档内部结构后再用于当前 `test.py` |
| `checkpoints/T-Rex-origami-m3-vlm-action-lora/` | 目录占位 | 当前扫描未见模型文件 | 不能作为可加载 checkpoint |

`checkpoint-0-11407` 的 `training_args.json` 与 T-Rex 训练器的保存格式一致，且参数包含 `action_dim=65`、`action_chunk=16`、tactile VQ-VAE、cascaded flow、Action-LoRA、VLM-LoRA 和 M3；`model.pt` 内还存在 `q_proj_action`、`q_proj_tactile`、`tacf6_vqvae_*` 等 T-Rex 专属张量名。因此该 checkpoint 属于 T-Rex Origami 训练链的证据很充分。

### 1.2 GR00T 与 Sharpa 权重

| 路径 | 类型 | 状态 |
|---|---|---|
| `checkpoints/GR00T-N1.7-3B/` | `nvidia/GR00T-N1.7-3B` 本地基础快照 | 包含两片 safetensors、processor、statistics、trainer state；不是已确认的 Origami 微调成品 |
| `sharpa-rl-lab/ckpts/cylinder_ball_cube.pth` | Sharpa Wave RL checkpoint | 已存在 |
| `sharpa-rl-lab/pretrained/0.4-0.6-8.pth` | Sharpa Wave 预训练/策略权重 | 已存在 |
| `sharpa-rl-lab/pretrained/0.5-0.5-1.pth` | Sharpa Wave 预训练/策略权重 | 已存在 |
| `DexGarment_env/DexGarmentLab_main/**` 下的 `*.ckpt/*.pth/*.pt` | DP/DP3 权重 | 当前未发现 |
| `Isaac-GR00T/**` 下的 `*.safetensors/*.pt/*.pth/*.ckpt` | GR00T Origami 微调权重 | 当前未发现 |

### 1.3 submission 与 checkpoint 对应关系

| manifest | checkpoint | action | 创建时间 |
|---|---|---|---|
| `submissions/20260819_212607/submission-manifest.json` | `checkpoint-0-50000` | 65-D，horizon 16 | 2026-08-19 |
| `submissions/20260901_162505/submission-manifest.json` | `checkpoint-0-11407` | 65-D，horizon 16 | 2026-09-01 |
| `submissions/20260918_155156/submission-manifest.json` | `checkpoint-0-7000` | 65-D，horizon 16 | 2026-09-18 |

## 2. T-Rex：训练代码

T-Rex 是当前工作区最完整的机器人模型训练与推理链。典型关系为：

```text
Origami LeRobot dataset
        ↓
train_origami_* launcher
        ↓ accelerate
scripts/train_origami_freeze_vlm.py
        ↓
qwen_vla + tactile VQ-VAE + dataset loader
        ↓
checkpoint-{epoch}-{global_step}/model.pt
```

### 2.1 训练入口和启动脚本

| 文件 | 作用 |
|---|---|
| `T-Rex/scripts/train_origami_freeze_vlm.py` | 当前 Origami 65-D 训练主程序；构造数据集、模型、优化器、LoRA/VQ-VAE/cascaded/M3，并保存 checkpoint |
| `T-Rex/scripts/train_origami_2x4090_50k_freeze_vlm.sh` | 当前 2×4090、冻结 VLM 的 Origami 启动脚本；默认 action 65、chunk 16、Action-LoRA、tactile VQ-VAE、cascaded flow |
| `T-Rex/scripts/train_origami_2x4090_3epoch_full_vlm.sh` | Origami full-VLM 训练脚本；`freeze_vlm=0`，显存要求更高 |
| `T-Rex/scripts/train_origami_docker.sh` | Docker/容器通用训练包装器；负责路径检查、GPU 数量、数据预检、resume 参数和最终 accelerate 命令 |
| `T-Rex/scripts/train_origami_docker_freeze_lora.sh` | Docker wrapper：冻结 VLM，同时训练 VLM-LoRA/Action-LoRA 的默认模式 |
| `T-Rex/scripts/train_origami_docker_vlm_action_lora.sh` | 上述 VLM-LoRA + Action-LoRA 模式的快捷入口 |
| `T-Rex/scripts/train_origami_docker_full_vlm.sh` | Docker wrapper：切换到 full-VLM 模式 |
| `T-Rex/scripts/train.py` | 通用/旧版 T-Rex post-train 主循环；支持 JSON/LeRobot 数据和传统配置 |
| `T-Rex/scripts/train_origami.py` | 旧版训练接口包装器，把 Origami LeRobot 数据集接入 `scripts/train.py` |
| `T-Rex/scripts/train.sh` | 历史 task post-train 启动脚本；包含外部数据路径，运行前必须替换路径 |
| `T-Rex/config/sft_qwen.yaml` | Qwen 版 SFT/accelerate 配置 |
| `T-Rex/config/sft_multi.yaml` | 多机/多 GPU SFT 配置 |
| `T-Rex/config/sft.yaml` | 通用 SFT 配置 |
| `T-Rex/scripts/check_accelerate_gpu_ids.py` | accelerate/GPU 可见性检查辅助脚本 |
| `T-Rex/scripts/check_trex_dataset.py` | T-Rex 数据集与视频文件预检 |

### 2.2 T-Rex 模型与数据训练核心

| 文件 | 作用 |
|---|---|
| `T-Rex/qwen_vla/modeling_vla.py` | `Qwen3VLVLAModel`；实现视觉语言主干、动作专家、触觉专家、动作 flow、触觉 flow、cascaded slow/fast 前向 |
| `T-Rex/qwen_vla/modeling_qwen3vl_mot.py` | Qwen3-VL 的多专家 MoT attention/decoder；包含 latent、action、tactile 专家分支 |
| `T-Rex/qwen_vla/diffusion.py` | 动作/触觉向量投影、时间步嵌入和输出层 |
| `T-Rex/qwen_vla/DeformAE.py` | tactile deformation encoder/decoder/inference 模型 |
| `T-Rex/qwen_vla/lerobot_dataset.py` | 原始 T-Rex LeRobot 数据读取器 |
| `T-Rex/qwen_vla/origami_lerobot_dataset.py` | Origami 65-D LeRobot loader；构造 16 步绝对关节动作 chunk、state/tactile 归一化和触觉历史 |
| `T-Rex/qwen_vla/checkpoint_restore.py` | 加载 checkpoint、恢复 Action-LoRA/VLM-LoRA、恢复统计量和嵌入 VQ-VAE |
| `T-Rex/qwen_vla/m3_masking.py` | 训练时 M3 modality masking；不是单独的推理模型 |
| `T-Rex/tactile_vqvae/models/tactile_vqvae.py` | tactile F6 VQ-VAE 主模型 |
| `T-Rex/tactile_vqvae/models/encoder.py` | VQ-VAE encoder |
| `T-Rex/tactile_vqvae/models/decoder.py` | VQ-VAE decoder |
| `T-Rex/tactile_vqvae/models/quantizer.py` | vector quantizer |
| `T-Rex/tactile_vqvae/data/dataset.py` | tactile VQ-VAE 数据集 |
| `T-Rex/tactile_vqvae/data/stats.py` | tactile 统计量计算 |
| `T-Rex/tactile_vqvae/train.py` | tactile VQ-VAE 单独训练程序 |
| `T-Rex/tactile_vqvae/eval.py` | VQ-VAE 评估程序 |
| `T-Rex/tactile_vqvae/extract_codes.py` | 从 tactile 数据提取离散 code |
| `T-Rex/tactile_vqvae/config/vqvae_f6.yaml` | F6 VQ-VAE 配置 |
| `T-Rex/tactile_vqvae/scripts/train_vqvae_f6.sh` | F6 VQ-VAE 训练启动脚本 |
| `T-Rex/tactile_vqvae/scripts/eval_vqvae_f6.sh` | F6 VQ-VAE 评估启动脚本 |
| `T-Rex/tactile_vqvae/scripts/extract_codes.sh` | VQ-VAE code 提取启动脚本 |
| `T-Rex/tactile_vqvae/scripts/inspect_exemplars.py` | VQ-VAE exemplar 检查工具 |

### 2.3 T-Rex 数据准备与 checkpoint 工具

| 文件 | 作用 |
|---|---|
| `T-Rex/utils/convert_inlab_to_lerobot.py` | 原始 in-lab 数据转换为 LeRobot v3.0 |
| `T-Rex/utils/convert_inlab_to_lerobot.sh` | 上述转换脚本的 shell 入口 |
| `T-Rex/utils/gen_json_tac_deltabase_eef_bimanual_parallel.py` | 原始数据生成训练 JSON；包含 tactile、delta-base、EEF、双臂处理 |
| `T-Rex/utils/gen_json_bimanual.sh` | JSON 数据生成启动脚本 |
| `T-Rex/utils/lerobot_common.py` | LeRobot schema、pose 计算、归一化统计量辅助函数 |
| `T-Rex/utils/encode_vqvae_codes_to_json.py` | 将 tactile VQ-VAE code 预编码到 JSON |
| `T-Rex/utils/encode_vqvae_codes_to_json.sh` | VQ-VAE code 预编码启动脚本 |
| `T-Rex/utils/merge_vqvae_into_ckpt.py` | 将独立 VQ-VAE 合并/嵌入 T-Rex checkpoint |
| `T-Rex/utils/merge_vqvae_into_ckpt.sh` | checkpoint 合并启动脚本 |
| `T-Rex/utils/analyze_episode.py` | 单 episode 数据与视频可视化/检查 |

## 3. T-Rex：推理、部署与评估代码

### 3.1 模型推理服务

| 文件 | 作用 |
|---|---|
| `T-Rex/scripts/test.py` | T-Rex 核心 ZMQ REP 推理服务器；加载 `training_args.json`、LoRA、VQ-VAE 和统计量，提供 slow/fast/slow_and_fast 三种协议 |
| `T-Rex/scripts/test.sh` | 基础 ZMQ 推理 server 启动脚本；其中部分默认路径是历史容器路径，使用前应覆盖参数 |
| `T-Rex/scripts/lora_test.py` | 严格 Action-LoRA 推理 wrapper；要求 checkpoint 中存在 Action-LoRA 并重构 LoRA 层 |
| `T-Rex/scripts/lora_test.sh` | Action-LoRA 推理启动脚本；默认值仍指向历史 `checkpoint-0-60000`，当前应改为 `checkpoint-0-11407` 或 `checkpoint-0-7000` |
| `T-Rex/docker/office_policy_server.py` | Office/Origami 生产适配器；内部调用 `scripts.test.model_load` 和 `CascadedServer`，将 Zenoh 输入转换为 T-Rex 输入并输出 65-D action chunk |
| `T-Rex/docker/README_office_inference.md` | Office inference 镜像和 checkpoint 挂载说明 |

T-Rex 的推理路径为：

```text
test.py:model_load
        ↓
CascadedServer
        ├─ slow: visual/language → split KV + partial action flow
        └─ fast: tactile F6/deform/VQ-VAE → remaining flow steps
        ↓
[action_chunk, action_dim] = [16, 65]
```

### 3.2 真实机器人客户端与动作安全

| 文件 | 作用 |
|---|---|
| `T-Rex/hardware_code/eval/eval_trex_async.py` | 真实机器人 REQ 客户端；按 chunk 首次 slow、后续 fast，从服务端取 action 并执行 |
| `T-Rex/hardware_code/eval/absolute_joint65.py` | 65-D absolute joint action 到机器人关节接口的转换 |
| `T-Rex/hardware_code/eval/force_safety.py` | 力/动作安全约束 |
| `T-Rex/hardware_code/eval/test_force_safety.py` | 安全模块测试 |
| `T-Rex/hardware_code/config/default.yaml` | server 地址、控制模式、action chunk、offset 和安全相关配置 |
| `T-Rex/hardware_code/eval/README.md` | 真实机器人推理启动流程与参数说明 |
| `T-Rex/hardware_code/camera/head_camera_receiver.py` | head camera 接收 |
| `T-Rex/hardware_code/camera/wrist_camera_receiver.py` | wrist camera 接收 |
| `T-Rex/hardware_code/camera/stream_sender_dexmate.py` | Dexmate camera 流发送 |
| `T-Rex/hardware_code/camera/stream_sender_zed_box.py` | ZED camera 流发送 |
| `T-Rex/hardware_code/camera/view_head_camera.py` | head camera 查看工具 |

`hardware_code/teleop/` 下的文件主要是遥操作、数据采集和回放，不是神经网络训练/推理核心；与训练数据生成有关的文件包括：

- `T-Rex/hardware_code/teleop/main_teleop.py`
- `T-Rex/hardware_code/teleop/arm_hand_control.py`
- `T-Rex/hardware_code/teleop/data_writer.py`
- `T-Rex/hardware_code/teleop/replay_h5.py`
- `T-Rex/hardware_code/teleop/visualize_data.py`
- `T-Rex/hardware_code/teleop/ik_utils.py`
- `T-Rex/hardware_code/teleop/teleop_targets.py`
- `T-Rex/hardware_code/teleop/config.py`

## 4. GR00T N1.7 Origami：训练代码

### 4.1 训练入口

| 文件 | 作用 |
|---|---|
| `Isaac-GR00T/examples/Origami/train_origami_n1d7.sh` | Origami GR00T N1.7 训练入口；使用本地 `checkpoints/GR00T-N1.7-3B`，65-D state/action，16 步 action horizon，支持 LoRA 或 frozen VLM |
| `Isaac-GR00T/examples/Origami/origami_config.py` | Origami embodiment/modality 配置 |
| `Isaac-GR00T/examples/Origami/README.md` | 数据路径、base checkpoint、单步 smoke test 和 200k-step 训练说明 |
| `Isaac-GR00T/gr00t/experiment/launch_finetune.py` | GR00T finetune 启动器 |
| `Isaac-GR00T/gr00t/experiment/launch_train.py` | 通用 GR00T train 启动器 |
| `Isaac-GR00T/gr00t/experiment/trainer.py` | trainer、优化器、checkpoint/resume 和训练循环 |
| `Isaac-GR00T/gr00t/experiment/experiment.py` | 实验构造和运行管理 |
| `Isaac-GR00T/gr00t/experiment/dist_utils.py` | 分布式训练辅助 |
| `Isaac-GR00T/gr00t/experiment/utils.py` | 训练工具函数 |

### 4.2 GR00T 数据、配置与模型核心

| 路径/文件 | 作用 |
|---|---|
| `Isaac-GR00T/gr00t/data/dataset/factory.py` | dataset factory |
| `Isaac-GR00T/gr00t/data/dataset/lerobot_episode_loader.py` | LeRobot episode loader |
| `Isaac-GR00T/gr00t/data/dataset/sharded_mixture_dataset.py` | sharded mixture dataset |
| `Isaac-GR00T/gr00t/data/dataset/sharded_single_step_dataset.py` | sharded single-step dataset |
| `Isaac-GR00T/gr00t/data/collator/collators.py` | batch collator |
| `Isaac-GR00T/gr00t/data/state_action/action_chunking.py` | action chunk 构造 |
| `Isaac-GR00T/gr00t/data/state_action/state_action_processor.py` | state/action 预处理 |
| `Isaac-GR00T/gr00t/data/state_action/pose.py` | pose 表示与转换 |
| `Isaac-GR00T/gr00t/data/stats.py` | 数据统计量 |
| `Isaac-GR00T/gr00t/configs/base_config.py` | 基础配置 |
| `Isaac-GR00T/gr00t/configs/finetune_config.py` | finetune 配置 |
| `Isaac-GR00T/gr00t/configs/data/data_config.py` | 数据配置 |
| `Isaac-GR00T/gr00t/configs/data/embodiment_configs.py` | embodiment 配置 |
| `Isaac-GR00T/gr00t/configs/model/gr00t_n1d7.py` | N1.7 模型配置 |
| `Isaac-GR00T/gr00t/configs/training/training_config.py` | training 配置 |
| `Isaac-GR00T/gr00t/model/base/model_pipeline.py` | 模型 pipeline 基类 |
| `Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | GR00T N1.7 主模型 |
| `Isaac-GR00T/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` | N1.7 输入/输出处理 |
| `Isaac-GR00T/gr00t/model/gr00t_n1d7/image_augmentations.py` | 图像增强 |
| `Isaac-GR00T/gr00t/model/modules/qwen3_backbone.py` | Qwen3 backbone |
| `Isaac-GR00T/gr00t/model/modules/dit.py` | action diffusion/DiT 模块 |
| `Isaac-GR00T/gr00t/model/modules/flowmatching_modules.py` | flow matching 模块 |
| `Isaac-GR00T/gr00t/model/modules/embodiment_conditioned_mlp.py` | embodiment-conditioned MLP |

当前工作区的 Origami 数据和配置由 `Isaac-GR00T/examples/Origami/README.md` 指向：

- 数据：`dataset/Robotic_Origami_Challenge/lerobot3.0`
- 基础 checkpoint：`checkpoints/GR00T-N1.7-3B`
- 65-D state/action、16-step absolute joint action、三路 RGB、任务文本

## 5. GR00T N1.7：推理、评估与 TensorRT

| 文件 | 作用 |
|---|---|
| `Isaac-GR00T/gr00t/policy/policy.py` | policy 基类 |
| `Isaac-GR00T/gr00t/policy/gr00t_policy.py` | GR00T policy；负责 observation → action chunk |
| `Isaac-GR00T/gr00t/policy/replay_policy.py` | replay/evaluation policy |
| `Isaac-GR00T/gr00t/policy/server_client.py` | policy server/client 通信 |
| `Isaac-GR00T/scripts/deployment/standalone_inference_script.py` | standalone 推理入口 |
| `Isaac-GR00T/gr00t/eval/run_gr00t_server.py` | GR00T policy server 入口 |
| `Isaac-GR00T/gr00t/eval/open_loop_eval.py` | open-loop action 评估 |
| `Isaac-GR00T/gr00t/eval/rollout_policy.py` | rollout policy 评估 |
| `Isaac-GR00T/gr00t/eval/real_robot/SO100/eval_so100.py` | SO100 真实机器人评估示例 |
| `Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py` | LIBERO 仿真评估环境 |
| `Isaac-GR00T/gr00t/eval/sim/SimplerEnv/simpler_env.py` | SimplerEnv 仿真评估环境 |
| `Isaac-GR00T/gr00t/eval/sim/wrapper/multistep_wrapper.py` | 多步仿真 wrapper |
| `Isaac-GR00T/gr00t/eval/sim/wrapper/video_recording_wrapper.py` | 视频记录 wrapper |
| `Isaac-GR00T/scripts/deployment/export_onnx_n1d7.py` | N1.7 导出 ONNX |
| `Isaac-GR00T/scripts/deployment/build_tensorrt_engine.py` | 构建 TensorRT engine |
| `Isaac-GR00T/scripts/deployment/build_trt_pipeline.py` | TensorRT pipeline 构建 |
| `Isaac-GR00T/scripts/deployment/trt_model_forward.py` | TensorRT 前向推理 |
| `Isaac-GR00T/scripts/deployment/trt_torch.py` | TensorRT/Torch 适配 |
| `Isaac-GR00T/scripts/deployment/verify_n1d7_trt.py` | TensorRT 正确性验证 |
| `Isaac-GR00T/scripts/deployment/benchmark_inference.py` | 推理性能 benchmark |
| `Isaac-GR00T/scripts/validate_origami_setup.py` | Origami 数据、checkpoint、维度和 GPU 验证 |

当前扫描未在 `Isaac-GR00T/` 下发现 Origami 微调输出，因此这些推理代码目前只能直接接入基础 GR00T snapshot 或外部提供的 finetuned checkpoint。

## 6. Sharpa Wave：强化学习训练代码

### 6.1 训练与蒸馏入口

| 文件 | 作用 |
|---|---|
| `sharpa-rl-lab/rl_isaaclab/scripts/train.py` | IsaacLab RL 训练主入口；支持 PPO 和 `ProprioAdapt` 蒸馏/适配 |
| `sharpa-rl-lab/rl_isaaclab/scripts/gen_grasp.py` | 生成 grasp cache |
| `sharpa-rl-lab/rl_isaaclab/tasks/inhand_rotate/agents/ppo_cfg.yaml` | PPO agent 配置 |
| `sharpa-rl-lab/rl_isaaclab/algo/ppo/ppo.py` | PPO 算法实现 |
| `sharpa-rl-lab/rl_isaaclab/algo/ppo/experience.py` | PPO experience buffer |
| `sharpa-rl-lab/rl_isaaclab/algo/padapt/padapt.py` | ProprioAdapt 蒸馏/部署算法 |
| `sharpa-rl-lab/rl_isaaclab/algo/models/models.py` | policy/value/model 结构 |
| `sharpa-rl-lab/rl_isaaclab/algo/models/running_mean_std.py` | observation normalization |

> 注意：`sharpa-rl-lab/rl_isaaclab/scripts/train.py` 中存在清理旧 `outputs/` 的逻辑，正式运行前应先检查输出路径和脚本参数。

### 6.2 环境与任务配置

| 文件 | 作用 |
|---|---|
| `sharpa-rl-lab/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env.py` | Sharpa Wave 训练环境 |
| `sharpa-rl-lab/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env_cfg.py` | 训练环境配置 |
| `sharpa-rl-lab/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_grasp_env.py` | grasp 环境 |
| `sharpa-rl-lab/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_grasp_env_cfg.py` | grasp 环境配置 |
| `sharpa-rl-lab/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env.py` | 部署环境 |
| `sharpa-rl-lab/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env_cfg.py` | 部署环境配置 |
| `sharpa-rl-lab/rl_isaaclab/wrapper/sharpa_wave_env_wrapper.py` | 训练环境 wrapper |
| `sharpa-rl-lab/rl_isaaclab/wrapper/sharpa_wave_deploy_env_wrapper.py` | 部署环境 wrapper |
| `sharpa-rl-lab/rl_isaaclab/wrapper/config_wrapper.py` | agent/env 配置包装 |
| `sharpa-rl-lab/rl_isaaclab/wrapper/vec_env.py` | vectorized env 适配 |

### 6.3 Sharpa 推理、可视化与真实部署

| 文件 | 作用 |
|---|---|
| `sharpa-rl-lab/rl_isaaclab/scripts/play.py` | 加载 RL checkpoint，在 IsaacLab 中可视化/测试 policy |
| `sharpa-rl-lab/rl_isaaclab/scripts/deploy.py` | 真实 Sharpa Wave 部署；加载 checkpoint 后调用 `agent.model.act_inference` |
| `sharpa-rl-lab/rl_isaaclab/scripts/replay_north_poc2_2_dataset.py` | 回放 Origami/North POC2.2 LeRobot trajectory；不是模型推理 |
| `sharpa-rl-lab/rl_isaaclab/scripts/test_north_poc2_2_asset.py` | North POC2.2 资产/仿真检查 |
| `sharpa-rl-lab/README.md` | 训练、play、蒸馏和 deploy 命令说明 |

典型流程：

```text
train.py → logs/checkpoint
play.py  → IsaacLab 可视化
deploy.py → Sharpa Wave 真实机器人
```

## 7. DexGarment：Diffusion Policy 与 DP3

### 7.1 Diffusion Policy（图像）

| 文件/目录 | 作用 |
|---|---|
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/train.py` | Hydra 训练入口，实例化 `RobotWorkspace` 并调用 `workspace.run()` |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/train.sh` | 按 task/data_num/seed 启动训练 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/data2zarr_dp.py` | demonstration 转 zarr |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/data2zarr_dp.sh` | zarr 转换启动脚本 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/workspace/base_workspace.py` | workspace 基类、保存/加载 checkpoint |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/workspace/robotworkspace.py` | RobotWorkspace 训练循环、验证和 action 预测 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/policy/base_image_policy.py` | 图像 policy 基类 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/policy/diffusion_unet_image_policy.py` | Diffusion U-Net image policy |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/env_runner/dp_runner.py` | 环境 rollout；调用 `policy.predict_action` |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/DP.py` | 任务级推理封装；从 `checkpoints/{task}_{data_num}/{checkpoint_num}.ckpt` 加载 policy |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/dataset/base_dataset.py` | zarr dataset 基类 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/dataset/robot_image_dataset.py` | 图像 robot dataset |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/model/vision/multi_image_obs_encoder.py` | 多图像 observation encoder |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/model/vision/model_getter.py` | ResNet 等视觉 encoder 构造 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/model/diffusion/conditional_unet1d.py` | 条件 1D U-Net |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/model/diffusion/transformer_for_diffusion.py` | diffusion transformer |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/model/diffusion/ema_model.py` | EMA 模型 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/model/common/normalizer.py` | observation/action normalization |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/config/robot_dp.yaml` | DP 模型、dataset、训练和 checkpoint 配置 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/diffusion_policy/config/task/default_task.yaml` | DP task 默认配置 |

相关但非模型核心的公共模块还包括 `common/checkpoint_util.py`、`common/pytorch_util.py`、`common/replay_buffer.py`、`common/normalize_util.py`、`common/sampler.py` 和 `common/env_util.py`。

### 7.2 Diffusion Policy 3D / DP3

| 文件/目录 | 作用 |
|---|---|
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/train.py` | DP3 Hydra 训练入口和 `TrainDP3Workspace` 训练循环 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/train.sh` | DP3 训练启动脚本 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/data2zarr_dp3.py` | demonstration 转 DP3 zarr |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/data2zarr_dp3.sh` | DP3 数据转换启动脚本 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/policy/dp3.py` | PointNet + diffusion 的 DP3 policy |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/policy/simple_dp3.py` | 简化版 DP3 policy |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/env_runner/robot_runner.py` | DP3 rollout/action runner |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/dataset/robot_dataset.py` | 点云/robot dataset |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/model/vision/pointnet_extractor.py` | 点云 PointNet encoder |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/model/diffusion/conditional_unet1d.py` | DP3 conditional U-Net |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/model/diffusion/ema_model.py` | EMA 模型 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/common/checkpoint_util.py` | checkpoint 管理 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/DP3.py` | 任务级 DP3 推理封装；从 task/data_num/checkpoint_num 加载 checkpoint |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/config/robot_dp3.yaml` | DP3 模型、点云、训练和 checkpoint 配置 |
| `DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/diffusion_policy_3d/config/task/default_task.yaml` | DP3 task 默认配置 |

当前工作区没有在 `DexGarment_env` 下找到 DP/DP3 的 `.ckpt`、`.pth` 或 `.pt`，所以这里应标记为“源码清单”，不能确认已有训练成品。

### 7.3 HALO/GAM/SADP 相关推理包装

DexGarment 环境和验证脚本中大量引用 `Model_HALO.GAM.GAM_Encapsulation`、`Model_HALO.SADP.SADP` 和 `Model_HALO.SADP_G.SADP_G`。但是当前扫描没有发现 `DexGarment_env/DexGarmentLab_main/Model_HALO/` 目录，因此下面这些文件是**调用包装/环境代码**，不是完整的 HALO 模型实现：

验证入口：

- `DexGarment_env/DexGarmentLab_main/Env_Validation/Fling_Dress_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Fling_Tops_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Fling_Trousers_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Fold_Dress_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Fold_Tops_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Fold_Trousers_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Hang_Coat_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Hang_Dress_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Hang_Tops_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Hang_Trousers_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Store_Tops_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Wear_Baseballcap_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Wear_Bowlhat_HALO.py`
- `DexGarment_env/DexGarmentLab_main/Env_Validation/Wear_Scarf_HALO.py`

环境/推理包装入口：

- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Fling_Dress_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Fling_Tops_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Fling_Trousers_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Fold_Dress_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Fold_Tops_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Fold_Trousers_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Hang_Coat_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Hang_Dress_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Hang_Tops_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Hang_Trousers_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Store_Tops_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Wear_Baseballcap_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Wear_Bowlhat_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Wear_Glove_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Wear_Scarf_Env.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/FoldTops_Env_VLA.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Fold_Tops_Env_with_galbot.py`
- `DexGarment_env/DexGarmentLab_main/Env_StandAlone/Fold_Tops_Env_with_galbot_sharpa.py`

配套说明文件：

- `DexGarment_env/DexGarmentLab_main/README.md`
- `DexGarment_env/DexGarmentLab_main/GAM_Usage.md`
- `DexGarment_env/DexGarmentLab_main/Validation_Core.md`
- `DexGarment_env/DexGarmentLab_main/Data_Collection.sh`
- `DexGarment_env/DexGarmentLab_main/Validation.sh`

## 8. Shadow Replay：离线推理/评测代码

`shadow_replay/` 不训练模型，也不向真实机器人发送动作；它把已记录的 episode 输入模型并保存预测、指标、曲线和可视化结果。

| 文件 | 作用 |
|---|---|
| `shadow_replay/replay.py` | 离线 replay 主流程 |
| `shadow_replay/model.py` | 通用模型后端接口 |
| `shadow_replay/trex.py` | T-Rex checkpoint adapter；复用 T-Rex 的模型加载和 slow/fast 推理路径 |
| `shadow_replay/dataset.py` | 记录数据读取 |
| `shadow_replay/actions.py` | action 表示、维度与映射 |
| `shadow_replay/metrics.py` | replay 指标 |
| `shadow_replay/perturbations.py` | 输入扰动实验 |
| `shadow_replay/safety.py` | action safety/URDF limit |
| `shadow_replay/robot_projection.py` | 机器人投影/可视化 |
| `shadow_replay/visualization.py` | 结果可视化 |
| `shadow_replay/config.py` | 配置解析 |
| `configs/shadow_replay_trex.yaml` | T-Rex backend 配置 |
| `configs/shadow_replay.yaml` | UniDex backend 配置 |
| `shadow_replay/README.md` | 安装、运行和数据泄漏说明 |

注意：`configs/shadow_replay_trex.yaml` 中默认仍有历史路径 `checkpoints/T-Rex-origami-posttrain/.../checkpoint-0-60000`，当前工作区已发现的替代 checkpoint 是 `checkpoints/checkpoint-0-11407`、`checkpoints/checkpoint-0-7000` 或 `checkpoints/T-Rex-origami-posttrain/checkpoint-0-50000`。

`configs/shadow_replay.yaml` 的 UniDex checkpoint 路径也属于历史/外部资源；当前根目录没有对应的 `UniDex/` 源码和成品权重。因此当前工作区能够明确闭环的是 Shadow Replay 的 T-Rex backend。

## 9. office/openpi 与竞赛部署参考代码

### 9.1 openpi reference framework

`office_inference/openpi-base-main/` 是通用 policy server/client 参考框架。它不等同于本地 T-Rex 或 GR00T 训练产物。

训练相关源码：

- `office_inference/openpi-base-main/src/openpi/training/config.py`
- `office_inference/openpi-base-main/src/openpi/training/checkpoints.py`
- `office_inference/openpi-base-main/src/openpi/training/droid_rlds_dataset.py`
- `office_inference/openpi-base-main/src/openpi/training/optimizer.py`
- `office_inference/openpi-base-main/src/openpi/training/sharding.py`
- `office_inference/openpi-base-main/src/openpi/training/utils.py`
- `office_inference/openpi-base-main/src/openpi/training/weight_loaders.py`

模型/策略相关源码：

- `office_inference/openpi-base-main/src/openpi/models/model.py`
- `office_inference/openpi-base-main/src/openpi/models/pi0.py`
- `office_inference/openpi-base-main/src/openpi/models/pi0_fast.py`
- `office_inference/openpi-base-main/src/openpi/models/gemma.py`
- `office_inference/openpi-base-main/src/openpi/models/gemma_fast.py`
- `office_inference/openpi-base-main/src/openpi/models/lora.py`
- `office_inference/openpi-base-main/src/openpi/models/tokenizer.py`
- `office_inference/openpi-base-main/src/openpi/policies/policy.py`
- `office_inference/openpi-base-main/src/openpi/policies/north_ces_policy.py`
- `office_inference/openpi-base-main/src/openpi/policies/aloha_policy.py`
- `office_inference/openpi-base-main/src/openpi/policies/droid_policy.py`
- `office_inference/openpi-base-main/src/openpi/policies/libero_policy.py`

推理/服务相关源码：

- `office_inference/openpi-base-main/scripts/serve_policy.py`
- `office_inference/openpi-base-main/scripts/serve_policy_zenoh.py`
- `office_inference/openpi-base-main/scripts/docker/submission_entrypoint.sh`
- `office_inference/openpi-base-main/scripts/docker/submission_zenoh_entrypoint.sh`
- `office_inference/openpi-base-main/src/openpi/serving/websocket_policy_server.py`
- `office_inference/openpi-base-main/packages/openpi-client/src/openpi_client/base_policy.py`
- `office_inference/openpi-base-main/packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- `office_inference/openpi-base-main/packages/openpi-client/src/openpi_client/action_chunk_broker.py`
- `office_inference/openpi-base-main/examples/template_policy_server.py`
- `office_inference/openpi-base-main/examples/simple_client/main.py`

### 9.2 Sharpa North CES Lite SDK

`office_inference/sharpa_north_ces_lite_sdk-main/` 主要是 Zenoh/竞赛 evaluator、协议、client 和远程 observation 工具，不包含当前已确认的机器人模型训练实现。关键文件：

- `office_inference/sharpa_north_ces_lite_sdk-main/participant_local_evaluator/__main__.py`
- `office_inference/sharpa_north_ces_lite_sdk-main/participant_local_evaluator/policy_client.py`
- `office_inference/sharpa_north_ces_lite_sdk-main/participant_local_evaluator/remote_client.py`
- `office_inference/sharpa_north_ces_lite_sdk-main/participant_local_evaluator/controller.py`
- `office_inference/sharpa_north_ces_lite_sdk-main/participant_local_evaluator/contract.py`
- `office_inference/sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py`
- `office_inference/sharpa_north_ces_lite_sdk-main/examples/remote_observation_client.py`
- `office_inference/sharpa_north_ces_lite_sdk-main/examples/check_zenoh_policy.py`

## 10. 上游框架与 vendored 代码（不要误认为当前模型训练产物）

### 10.1 IsaacLab 通用训练/评估入口

`IsaacLab/` 是上游仿真与 RL/IL 框架。当前工作区没有证据表明下面入口直接训练了 `checkpoints/` 中已确认的 T-Rex、GR00T 或 Sharpa 成品。

- `IsaacLab/scripts/reinforcement_learning/rl_games/train.py`
- `IsaacLab/scripts/reinforcement_learning/rl_games/play.py`
- `IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`
- `IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py`
- `IsaacLab/scripts/reinforcement_learning/skrl/train.py`
- `IsaacLab/scripts/reinforcement_learning/skrl/play.py`
- `IsaacLab/scripts/reinforcement_learning/sb3/train.py`
- `IsaacLab/scripts/reinforcement_learning/sb3/play.py`
- `IsaacLab/scripts/imitation_learning/robomimic/train.py`
- `IsaacLab/scripts/imitation_learning/robomimic/play.py`
- `IsaacLab/scripts/imitation_learning/robomimic/robust_eval.py`
- `IsaacLab/scripts/sim2sim_transfer/rsl_rl_transfer.py`

### 10.2 T-Rex 内置 LeRobot

`T-Rex/third_party/lerobot/` 是 vendored 通用 LeRobot，不是 T-Rex 自定义模型核心。通用训练/评估入口为：

- `T-Rex/third_party/lerobot/src/lerobot/scripts/lerobot_train.py`
- `T-Rex/third_party/lerobot/src/lerobot/scripts/lerobot_eval.py`
- `T-Rex/third_party/lerobot/examples/training/train_policy.py`
- `T-Rex/third_party/lerobot/examples/training/train_with_streaming.py`
- `T-Rex/third_party/lerobot/examples/tutorial/act/act_training_example.py`
- `T-Rex/third_party/lerobot/examples/tutorial/diffusion/diffusion_training_example.py`
- `T-Rex/third_party/lerobot/examples/tutorial/async-inf/policy_server.py`
- `T-Rex/third_party/lerobot/examples/tutorial/async-inf/robot_client.py`

T-Rex 自己的 Origami loader 是 `T-Rex/qwen_vla/origami_lerobot_dataset.py`，不要把 vendored LeRobot 的通用 policy 入口和 T-Rex 的 65-D tactile VLA 混为一谈。

## 11. 根目录其他脚本的归类

下面脚本是数据分析、轨迹采样或打包工具，不属于模型训练/推理主入口：

- `scripts/analyze_shadow_replay_episode.py`
- `scripts/analyze_urdf_clipping.py`
- `scripts/merge_trex_horizon0_shards.py`
- `scripts/sample_trex_horizon0_trajectory.py`
- `submission/build_office.sh`
- `submission/package_office.sh`

它们可以辅助验证、数据处理和提交镜像，但不会直接训练一个新的 robot policy。

## 12. 当前最直接的使用路径

### T-Rex 训练

优先查看并修改：

1. `T-Rex/scripts/train_origami_2x4090_50k_freeze_vlm.sh`
2. `T-Rex/scripts/train_origami_docker.sh`
3. `T-Rex/scripts/train_origami_freeze_vlm.py`
4. `T-Rex/qwen_vla/origami_lerobot_dataset.py`

至少需要确认 `model_path`、`lerobot_root`、resume checkpoint、GPU 数量和 output directory。已保存的 `training_args.json` 中有历史机器绝对路径，迁移到当前工作区时应使用当前路径覆盖。

### T-Rex 推理

- 本地 ZMQ：`T-Rex/scripts/test.py`
- 严格 LoRA checkpoint：`T-Rex/scripts/lora_test.py`
- Office/Zenoh：`T-Rex/docker/office_policy_server.py`
- 真实机器人客户端：`T-Rex/hardware_code/eval/eval_trex_async.py`
- 离线评测：`shadow_replay/trex.py` + `configs/shadow_replay_trex.yaml`

### GR00T 训练/推理

- 训练：`Isaac-GR00T/examples/Origami/train_origami_n1d7.sh`
- 单机推理：`Isaac-GR00T/scripts/deployment/standalone_inference_script.py`
- 服务：`Isaac-GR00T/gr00t/eval/run_gr00t_server.py`
- TensorRT：`Isaac-GR00T/scripts/deployment/build_tensorrt_engine.py` 和 `build_trt_pipeline.py`

### Sharpa RL

- 训练：`sharpa-rl-lab/rl_isaaclab/scripts/train.py`
- 仿真推理：`sharpa-rl-lab/rl_isaaclab/scripts/play.py`
- 实机部署：`sharpa-rl-lab/rl_isaaclab/scripts/deploy.py`

### DexGarment DP/DP3

- DP 训练：`DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/train.py`
- DP 推理封装：`DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy/DP.py`
- DP3 训练：`DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/train.py`
- DP3 推理封装：`DexGarment_env/DexGarmentLab_main/IL_Baselines/Diffusion_Policy_3D/DP3.py`

## 13. 结论与缺口

1. 当前最明确的“训练代码 + 推理代码 + 本地成品权重”闭环是 **T-Rex Origami**。
2. `checkpoint-0-11407`、`checkpoint-0-7000` 和 `checkpoint-0-50000` 均能在训练参数、模型张量结构和 submission manifest 中与 T-Rex Origami 链路对应。
3. GR00T 的训练/推理代码较完整，且有 N1.7 base snapshot，但当前目录没有发现 Origami finetuned checkpoint。
4. Sharpa Wave 的 PPO/ProprioAdapt 训练、仿真 play 和实机 deploy 代码以及 `.pth` 权重均存在。
5. DexGarment 的 DP/DP3 训练和推理源码存在，但当前目录没有匹配 checkpoint；HALO/SADP 主体 `Model_HALO` 目录缺失。
6. `T-Rex/third_party/lerobot`、`IsaacLab`、`office_inference/openpi-base-main` 应视为上游/参考框架，不应在模型归因时直接当作当前任务模型的训练代码。
