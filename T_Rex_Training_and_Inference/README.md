# T-Rex Training and Inference

This directory is a snapshot of the training, model inference, hardware evaluation,
and auxiliary tools extracted from the T-Rex project. It is currently uploaded as a
regular directory to the `feature/trex-training-inference` branch of
`HAIV-Lab/FoldVLA`.

This directory contains only code, configuration files, and dependency declarations.
It **does not include checkpoints, base models, datasets, caches, or third-party
LeRobot source code**.

## Directory Contents

- `scripts/`: Training entry points, Origami training wrappers, the ZMQ inference
  server, LoRA inference launchers, and data-checking scripts.
- `qwen_vla/`: Qwen3-VL MoT/VLA models, diffusion action heads, dataset adapters,
  checkpoint restoration, and M3 masking.
- `hardware_code/`: Camera, teleoperation, robot/dexterous-hand hardware evaluation,
  and asynchronous execution code.
- `shadow_replay/`: Offline shadow replay, action projection, safety, and evaluation
  components.
- `tactile_vqvae/`: Tactile VQ-VAE training, evaluation, and encoding tools.
- `utils/`: Data conversion, statistics processing, VQ-VAE merging, and analysis
  utilities.
- `config/` and `configs/`: Training and shadow-replay configurations.
- `docker/`: Training/Office inference image definitions and the Office policy server.

## Training

The relatively portable Docker training entry point is recommended. Before running it,
prepare the following:

1. A Python 3.10/CUDA environment named `trex`;
2. A local Qwen3-VL-2B-Instruct base model;
3. A T-Rex mid-training checkpoint or another compatible resume checkpoint;
4. An Origami LeRobot dataset;
5. An installed LeRobot package, or `third_party/lerobot/src` from a full T-Rex
   checkout.

Example:

```bash
cd T_Rex_Training_and_Inference
bash scripts/train_origami_docker_vlm_action_lora.sh \
  --dataset-root /path/to/origami_lerobot_dataset \
  --model-path /path/to/Qwen3-VL-2B-Instruct \
  --resume-checkpoint /path/to/trex_checkpoint \
  --checkpoint-dir /path/to/output \
  --gpus 0,1
```

The training script checks the dataset, model weights, and checkpoint. Add `--dry-run`
first to inspect the final Accelerate command.

`train_origami_2x4090_*.sh`, `test.sh`, and `lora_test.sh` are legacy environment
wrappers that still contain absolute paths from the original machine. Update those
paths when moving to another machine, or call the corresponding Python files directly
with explicit arguments.

## Model Inference

### ZMQ Inference Server

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

Use the strict loading entry point for a LoRA checkpoint:

```bash
python scripts/lora_test.py \
  --checkpoint_path /path/to/action_lora_checkpoint \
  --base_model_path /path/to/Qwen3-VL-2B-Instruct \
  --stats_path /path/to/lerobot/meta/stats.json \
  --cuda 0 --port 5555
```

The inference server uses T-Rex's slow/fast cascaded ZMQ protocol. Clients must send
images, robot state, and tactile data according to this protocol.

### Office/Zenoh Inference

`docker/office_policy_server.py` is the Zenoh policy server for Office inference. It
validates Origami observations, adapts multi-camera/tactile inputs, and calls the T-Rex
model. `docker/Dockerfile.office` requires a complete external BuildKit context
(including the T-Rex source, base model, checkpoint, and statistics) and cannot be
built from this directory alone.

### Real-Robot Evaluation

`hardware_code/eval/eval_trex_async.py` depends on the robot, dexterous hands, cameras,
IK, and the site SDK. It can run only in an environment with the required hardware
and drivers configured. The default configuration is located at
`hardware_code/config/default.yaml`.

## Tactile and Offline Tools

- `tactile_vqvae/` provides tactile VQ-VAE training, evaluation, and code extraction.
- `utils/` provides InLab/LeRobot conversion, statistics processing, and VQ-VAE code
  utilities.
- `shadow_replay/` provides offline replay and safety-related components. The exact
  usage depends on the external data and evaluator.

## Important Limitations

- No checkpoints, including `checkpoint-0-11407`, have been uploaded to this
  directory.
- Qwen3-VL weights, the Origami dataset, `third_party/lerobot`, hardware SDKs, and
  runtime caches are not included.
- `docker/Dockerfile` references `docker/requirements-cu124.txt` from the original
  T-Rex checkout. The Dockerfile is retained here as a build reference; independent
  builds require that dependency file and the third-party source code to be restored.
- Check the licenses of the original T-Rex and FoldVLA projects, as well as the
  licenses of their third-party dependencies, before use.
