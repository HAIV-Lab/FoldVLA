# T-Rex Training and Inference

这是从 T-Rex 项目抽取的训练、模型推理、硬件评测和辅助工具代码快照，当前作为普通目录上传到 `HAIV-Lab/FoldVLA` 的 `feature/trex-training-inference` 分支。

本目录只保存代码、配置和依赖声明，**不包含 checkpoint、基础模型、数据集、缓存或第三方 LeRobot 源码**。

## 目录内容

- `scripts/`：训练入口、Origami 训练封装、ZMQ 推理服务、LoRA 推理入口和数据检查脚本。
- `qwen_vla/`：Qwen3-VL MoT/VLA 模型、扩散动作头、数据集适配、checkpoint 恢复和 M3 masking。
- `hardware_code/`：相机、遥操作、机器人/灵巧手硬件评测和异步执行代码。
- `shadow_replay/`：离线 shadow replay、动作投影、安全和评测组件。
- `tactile_vqvae/`：触觉 VQ-VAE 训练、评测和编码工具。
- `utils/`：数据转换、统计量处理、VQ-VAE 合并和分析工具。
- `config/`、`configs/`：训练与 shadow replay 配置。
- `docker/`：训练/Office 推理镜像定义和 Office policy server。

## 训练

推荐使用相对可移植的 Docker 训练入口。运行前需要准备：

1. 名为 `trex` 的 Python 3.10/CUDA 环境；
2. 本地 Qwen3-VL-2B-Instruct 基础模型；
3. T-Rex mid-training checkpoint 或其他兼容的恢复 checkpoint；
4. Origami LeRobot 数据集；
5. 已安装的 LeRobot，或在完整 T-Rex checkout 中提供 `third_party/lerobot/src`。

示例：

```bash
cd T_Rex_Training_and_Inference
bash scripts/train_origami_docker_vlm_action_lora.sh \
  --dataset-root /path/to/origami_lerobot_dataset \
  --model-path /path/to/Qwen3-VL-2B-Instruct \
  --resume-checkpoint /path/to/trex_checkpoint \
  --checkpoint-dir /path/to/output \
  --gpus 0,1
```

训练脚本会检查数据集、模型权重和 checkpoint；可先加 `--dry-run` 查看最终 Accelerate 命令。

`train_origami_2x4090_*.sh`、`test.sh` 和 `lora_test.sh` 是历史环境封装，仍包含原机器的绝对路径。迁移到其他机器时应改路径，或直接调用对应的 Python 文件并显式传入参数。

## 模型推理

### ZMQ 推理服务

```bash
cd T_Rex_Training_and_Inference
python scripts/test.py \
  --checkpoint_path /path/to/checkpoint \
  --base_model_path /path/to/Qwen3-VL-2B-Instruct \
  --stats_path /path/to/lerobot/meta/stats.json \
  --cuda 0 \
  --port 5555 \
  --image_size 384 288
```

LoRA checkpoint 可使用严格加载入口：

```bash
python scripts/lora_test.py \
  --checkpoint_path /path/to/action_lora_checkpoint \
  --base_model_path /path/to/Qwen3-VL-2B-Instruct \
  --stats_path /path/to/lerobot/meta/stats.json \
  --cuda 0 --port 5555
```

推理服务使用 T-Rex 的 slow/fast cascaded ZMQ 协议；客户端需要按照该协议发送图像、机器人状态和触觉数据。

### Office/Zenoh 推理

`docker/office_policy_server.py` 是 Office inference 的 Zenoh policy server，负责验证 Origami observation、适配多相机/触觉输入并调用 T-Rex 模型。`docker/Dockerfile.office` 需要完整的 BuildKit 外部 context（T-Rex 源码、基础模型、checkpoint 和统计量），不能仅凭本目录单独构建。

### 真实机器人评测

`hardware_code/eval/eval_trex_async.py` 依赖机器人、灵巧手、相机、IK 和站点 SDK，只能在配好硬件和驱动的环境运行。默认配置位于 `hardware_code/config/default.yaml`。

## 触觉与离线工具

- `tactile_vqvae/` 提供触觉 VQ-VAE 的训练、评测和 code 提取。
- `utils/` 提供 InLab/LeRobot 转换、统计量和 VQ-VAE code 处理。
- `shadow_replay/` 提供离线 replay 和安全相关组件；具体运行方式取决于外部数据与评测器。

## 重要限制

- 本目录没有上传任何 checkpoint，包括 `checkpoint-0-11407`。
- 本目录没有上传 Qwen3-VL 权重、Origami 数据集、`third_party/lerobot`、硬件 SDK 或运行缓存。
- `docker/Dockerfile` 引用了原 T-Rex checkout 中的 `docker/requirements-cu124.txt`；当前目录保留 Dockerfile 作为构建参考，若要独立构建，需要补齐该依赖文件及第三方源码。
- 使用前请检查 T-Rex 原项目和 FoldVLA 项目的许可证及第三方依赖许可证。

