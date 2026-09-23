# T-Rex 训练与推理代码清单（精简版）

项目根目录：`/data/why/foldVLA/T-Rex`

整理日期：2026-09-21

本清单只保留 T-Rex 相关的训练代码、模型核心、数据处理、推理服务、真实机器人客户端及离线评测代码。

## 1. T-Rex 训练流程

```text
Origami LeRobot 数据集
        ↓
训练启动脚本
        ↓ accelerate
train_origami_freeze_vlm.py
        ↓
Qwen3-VL + T-Rex MoT + tactile VQ-VAE
        ↓
checkpoint-{epoch}-{global_step}/model.pt
```

## 2. 训练入口与启动脚本

| 文件 | 作用 |
|---|---|
| `T-Rex/scripts/train_origami_freeze_vlm.py` | 当前 Origami 65-D T-Rex 主训练程序；负责数据集、模型、优化器、LoRA、VQ-VAE、cascaded flow、M3 和 checkpoint 保存 |
| `T-Rex/scripts/train_origami_2x4090_50k_freeze_vlm.sh` | 2×4090、冻结 VLM 的 Origami 训练启动脚本 |
| `T-Rex/scripts/train_origami_2x4090_3epoch_full_vlm.sh` | full-VLM Origami 训练启动脚本；`freeze_vlm=0` |
| `T-Rex/scripts/train_origami_docker.sh` | Docker/容器通用训练包装器；检查数据、模型、GPU 和 resume 参数后调用 accelerate |
| `T-Rex/scripts/train_origami_docker_freeze_lora.sh` | 冻结 VLM、训练 VLM-LoRA/Action-LoRA 的快捷入口 |
| `T-Rex/scripts/train_origami_docker_vlm_action_lora.sh` | VLM-LoRA + Action-LoRA 训练快捷入口 |
| `T-Rex/scripts/train_origami_docker_full_vlm.sh` | full-VLM Docker 训练快捷入口 |
| `T-Rex/scripts/train.py` | 通用/旧版 T-Rex post-train 主循环 |
| `T-Rex/scripts/train_origami.py` | 将 Origami 数据接入旧版 `train.py` 的包装器 |
| `T-Rex/scripts/train.sh` | 历史 task post-train 启动脚本；使用前需替换历史绝对路径 |
| `T-Rex/config/sft_qwen.yaml` | Qwen SFT/accelerate 配置 |
| `T-Rex/config/sft_multi.yaml` | 多 GPU/多机 SFT 配置 |
| `T-Rex/config/sft.yaml` | 通用 SFT 配置 |
| `T-Rex/scripts/check_accelerate_gpu_ids.py` | GPU/accelerate 检查 |
| `T-Rex/scripts/check_trex_dataset.py` | 数据集和视频预检 |

## 3. T-Rex 模型训练核心

| 文件 | 作用 |
|---|---|
| `T-Rex/qwen_vla/modeling_vla.py` | `Qwen3VLVLAModel`；视觉语言、动作专家、触觉专家、动作 flow 和 cascaded flow |
| `T-Rex/qwen_vla/modeling_qwen3vl_mot.py` | Qwen3-VL 多专家 MoT attention/decoder |
| `T-Rex/qwen_vla/diffusion.py` | 动作/触觉投影、时间步嵌入和输出层 |
| `T-Rex/qwen_vla/DeformAE.py` | tactile deformation encoder/decoder |
| `T-Rex/qwen_vla/origami_lerobot_dataset.py` | Origami 65-D LeRobot loader；生成 16 步绝对关节 action chunk 和 tactile 历史 |
| `T-Rex/qwen_vla/lerobot_dataset.py` | 原始 T-Rex LeRobot 数据读取器 |
| `T-Rex/qwen_vla/checkpoint_restore.py` | checkpoint、Action-LoRA/VLM-LoRA、统计量和 VQ-VAE 恢复 |
| `T-Rex/qwen_vla/m3_masking.py` | 训练时 M3 modality masking |

## 4. Tactile VQ-VAE 训练代码

| 文件 | 作用 |
|---|---|
| `T-Rex/tactile_vqvae/train.py` | tactile F6 VQ-VAE 单独训练程序 |
| `T-Rex/tactile_vqvae/eval.py` | VQ-VAE 评估 |
| `T-Rex/tactile_vqvae/extract_codes.py` | 提取 tactile 离散 code |
| `T-Rex/tactile_vqvae/models/tactile_vqvae.py` | VQ-VAE 主模型 |
| `T-Rex/tactile_vqvae/models/encoder.py` | encoder |
| `T-Rex/tactile_vqvae/models/decoder.py` | decoder |
| `T-Rex/tactile_vqvae/models/quantizer.py` | vector quantizer |
| `T-Rex/tactile_vqvae/data/dataset.py` | VQ-VAE 数据集 |
| `T-Rex/tactile_vqvae/data/stats.py` | tactile 统计量 |
| `T-Rex/tactile_vqvae/config/vqvae_f6.yaml` | F6 VQ-VAE 配置 |
| `T-Rex/tactile_vqvae/scripts/train_vqvae_f6.sh` | VQ-VAE 训练启动脚本 |
| `T-Rex/tactile_vqvae/scripts/eval_vqvae_f6.sh` | VQ-VAE 评估启动脚本 |
| `T-Rex/tactile_vqvae/scripts/extract_codes.sh` | code 提取启动脚本 |
| `T-Rex/tactile_vqvae/scripts/inspect_exemplars.py` | exemplar 检查工具 |

## 5. 数据准备与 checkpoint 工具

| 文件 | 作用 |
|---|---|
| `T-Rex/utils/convert_inlab_to_lerobot.py` | 原始 in-lab 数据转 LeRobot v3.0 |
| `T-Rex/utils/convert_inlab_to_lerobot.sh` | LeRobot 转换启动脚本 |
| `T-Rex/utils/gen_json_tac_deltabase_eef_bimanual_parallel.py` | 生成 tactile/EEF/双臂训练 JSON |
| `T-Rex/utils/gen_json_bimanual.sh` | JSON 生成启动脚本 |
| `T-Rex/utils/lerobot_common.py` | LeRobot schema、pose 和归一化辅助函数 |
| `T-Rex/utils/encode_vqvae_codes_to_json.py` | 将 VQ-VAE code 写入 JSON |
| `T-Rex/utils/encode_vqvae_codes_to_json.sh` | VQ-VAE code 预编码启动脚本 |
| `T-Rex/utils/merge_vqvae_into_ckpt.py` | 将 VQ-VAE 合并到 checkpoint |
| `T-Rex/utils/merge_vqvae_into_ckpt.sh` | checkpoint 合并启动脚本 |
| `T-Rex/utils/analyze_episode.py` | episode 数据/视频分析 |

## 6. T-Rex 推理代码

### 6.1 核心推理服务

| 文件 | 作用 |
|---|---|
| `T-Rex/scripts/test.py` | 核心 ZMQ REP 推理服务器；加载 checkpoint、LoRA、VQ-VAE、统计量并提供 slow/fast/slow_and_fast 协议 |
| `T-Rex/scripts/test.sh` | ZMQ 推理服务启动脚本；默认路径包含历史容器路径 |
| `T-Rex/scripts/lora_test.py` | 严格 Action-LoRA 推理入口 |
| `T-Rex/scripts/lora_test.sh` | Action-LoRA 推理启动脚本；默认 checkpoint 是历史 `checkpoint-0-60000`，当前应覆盖为现有 checkpoint |
| `T-Rex/docker/office_policy_server.py` | Office/Zenoh 生产推理适配器；将 Zenoh observation 转成 T-Rex 输入并输出 65-D action chunk |
| `T-Rex/docker/README_office_inference.md` | Office inference 服务和 checkpoint 挂载说明 |

推理逻辑：

```text
visual/language observation
        ↓ slow
缓存 split KV + 部分 action flow
        ↓ fast
最新 tactile F6 + deform + VQ-VAE
        ↓
剩余 flow steps
        ↓
16 × 65 action chunk
```

### 6.2 真实机器人推理客户端

| 文件 | 作用 |
|---|---|
| `T-Rex/hardware_code/eval/eval_trex_async.py` | 真实机器人 REQ 客户端；chunk 开始发送 slow，中间 tick 发送 fast |
| `T-Rex/hardware_code/eval/absolute_joint65.py` | 65-D absolute joint action 转换 |
| `T-Rex/hardware_code/eval/force_safety.py` | 力/动作安全约束 |
| `T-Rex/hardware_code/eval/test_force_safety.py` | 安全模块测试 |
| `T-Rex/hardware_code/config/default.yaml` | server 地址、控制模式、chunk 和安全参数 |
| `T-Rex/hardware_code/eval/README.md` | 真实机器人推理启动说明 |

### 6.3 推理输入设备支持

| 文件 | 作用 |
|---|---|
| `T-Rex/hardware_code/camera/head_camera_receiver.py` | head camera 接收 |
| `T-Rex/hardware_code/camera/wrist_camera_receiver.py` | wrist camera 接收 |
| `T-Rex/hardware_code/camera/stream_sender_dexmate.py` | Dexmate camera 流发送 |
| `T-Rex/hardware_code/camera/stream_sender_zed_box.py` | ZED camera 流发送 |
| `T-Rex/hardware_code/camera/view_head_camera.py` | head camera 查看 |

## 7. T-Rex 相关离线推理/评测

这些代码不直接控制真实机器人，但复用了 T-Rex 推理链：

| 文件 | 作用 |
|---|---|
| `shadow_replay/trex.py` | T-Rex checkpoint offline adapter |
| `shadow_replay/replay.py` | 离线 episode replay 主流程 |
| `shadow_replay/model.py` | 模型后端接口 |
| `shadow_replay/dataset.py` | episode 数据读取 |
| `shadow_replay/actions.py` | action 维度/表示映射 |
| `shadow_replay/metrics.py` | replay 指标 |
| `shadow_replay/safety.py` | 离线动作安全检查 |
| `shadow_replay/visualization.py` | 结果可视化 |
| `configs/shadow_replay_trex.yaml` | T-Rex offline replay 配置 |
| `shadow_replay/README.md` | 离线评测说明 |

## 8. 本地 T-Rex checkpoint

| 路径 | 类型 | 主要配置 |
|---|---|---|
| `checkpoints/T-Rex-midTrain/` | T-Rex midtrain | action dim 62 |
| `checkpoints/T-Rex-origami-posttrain/checkpoint-0-50000/` | Origami post-train | action dim 65，chunk 16 |
| `checkpoints/checkpoint-0-11407/` | Origami post-train | action dim 65，chunk 16，Action-LoRA/VLM-LoRA |
| `checkpoints/checkpoint-0-7000/` | Origami post-train | action dim 65，chunk 16，Action-LoRA/VLM-LoRA |
| `checkpoints/trex_ckpt_7000_best.tar.gz/` | 历史归档 | 使用前需确认归档结构 |

关键 checkpoint 文件通常包括：

- `model.pt`
- `config.json`
- `training_args.json`
- `processor/`
- `stats_data.json`
- 部分 checkpoint 还包含 `deform_encoder_from_model_pt.pth`

`checkpoint-0-11407` 的训练参数、T-Rex 专属模型张量和 submission manifest 均与 T-Rex Origami 训练链一致。

## 9. 最直接的入口

训练：

```text
T-Rex/scripts/train_origami_2x4090_50k_freeze_vlm.sh
  → T-Rex/scripts/train_origami_freeze_vlm.py
```

本地推理：

```text
T-Rex/scripts/test.py
```

严格 LoRA 推理：

```text
T-Rex/scripts/lora_test.py
```

Office/Zenoh 推理：

```text
T-Rex/docker/office_policy_server.py
```

真实机器人执行：

```text
T-Rex/hardware_code/eval/eval_trex_async.py
```

## 10. 注意事项

1. `training_args.json` 中可能保存了旧机器的绝对路径，迁移运行时需要覆盖 `model_path`、`lerobot_root`、checkpoint 和输出目录。
2. `T-Rex/scripts/lora_test.sh` 与部分 README 仍使用历史 `checkpoint-0-60000`，当前应改为实际存在的 checkpoint。
3. T-Rex 当前 main 分支主要提供 post-train 和 inference；README 中说明的 pretrain/midtrain 完整代码位于上游 `full-pipeline` 分支，当前工作区不包含那套完整训练阶段代码。
