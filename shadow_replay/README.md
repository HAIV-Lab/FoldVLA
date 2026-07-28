# Offline Shadow Replay for UniDex/T-Rex Origami

This package evaluates trained VLA predictions against complete recorded
episodes. It is deliberately offline: predictions are written to artifacts and
are never exposed to a robot command API.

## Repository findings

The implementation is based on the interfaces actually present in this
workspace:

- Model: `UniDex/src/unidex/unidex.py`,
  `PointCloudUniDexTrain.infer_action(batch)`.
- Checkpoint loading: Lightning `state_dict` with a `policy.` prefix, as used by
  `UniDex/finetune.py`.
- Model input: pointcloud `[B, cond_steps, P, 6]`, normalized state
  `[B, cond_steps, D]`, and a prompt list.
- Current checkpoint configuration: `cond_steps=1`, `horizon_steps=30`,
  `action_dim=82`, `proprio_dim=82`, ten flow-matching inference steps.
- Dataset: LeRobot 3.0 parquet/video, nine episodes, 98,694 frames, 30 FPS.
  `observation.state` and `action` are 65-D. The action modality explicitly
  declares absolute joint targets.
- North action order: left arm 7, left hand 22, right arm 7, right hand 22,
  lower-body/neck motor values 7.
- Existing UniDex training conversion: 82-D
  `[right pose9d, left pose9d, right mapped hand32, left mapped hand32]`.
  `pose9d` is XYZ plus rotation6D. The converter uses identity wrist poses and
  only the 22 hand joints per side.
- Training normalization: min/max normalization saved in the training YAML and
  a matching `normalizer-*.pt`.
- Training sampling: stride 3 and a one-selected-step target offset
  (state at `start_idx-1`, chunk labels from `start_idx`).
- Robot limits: the supplied North POC2.2 URDF has 65 mapped movable joints and
  position/velocity limits.
- Missing deployment services: no validated wrist-to-arm IK, singularity
  metric, or self/environment collision service is connected to UniDex.

There is a material mismatch between the requested “65-D model output” and the
checkpoint currently on disk, whose saved training configuration is 82-D. The
framework detects dimensions before replay. For this checkpoint, the explicit
`unidex82_to_north65` adapter:

1. records the complete 82-D raw and denormalized predictions;
2. inverse-maps the predicted hand joints into the 65-D North order;
3. holds arm/lower-body/neck joints at the measured observation;
4. records wrist pose9d separately and marks it unapplied because IK is absent.

It does not silently pretend that the 82-D model is a full 65-joint controller.

The current converted zarr was produced with gray image-plane pseudo
pointclouds because the source has no depth and the then-installed decoder
could not read AV1. The default replay reproduces that preprocessing exactly.
`rgb_image_plane` can decode real RGB and build pseudo pointcloud colors, but
that is a deliberate input distribution shift, not a calibrated 3D
observation.

## T-Rex backend

`configs/shadow_replay_trex.yaml` switches the same episode-preserving
evaluator to the local T-Rex post-train checkpoint. It follows the T-Rex
deployment path rather than adapting its inputs to UniDex:

- real head-left, wrist-right, and wrist-left RGB frames;
- current robot state normalized by the aggregate post-training q01/q99 stats;
- current tactile F6 plus a dense, episode-bounded 16-frame history;
- the real 5x2 tactile-deformation mosaic split into ten fingertip images;
- Qwen3-VL processor/chat template and cascaded slow+fast flow matching;
- native `[16,65]` absolute joint chunks, with no 82-D bridge.

The source videos are AV1. This host's OpenCV build cannot decode them, so the
reader automatically uses PyAV/libdav1d for AV1. The post-train checkpoint
contains 196 action-LoRA modules, but its `training_args.json` did not persist
LoRA rank/alpha and upstream `scripts/test.py` does not reconstruct those
modules. The replay loader recovers rank 16 / alpha 32 from the recorded
training command and requires a strict state-dict match, preventing silent
weight loss.

The selected season is part of the aggregate corpus recorded by the T-Rex
post-training run. T-Rex replay numbers on this path are therefore diagnostic,
not held-out generalization estimates.

## Install

Use the existing UniDex environment and install only the supplemental
dependencies if needed:

```bash
/data/why/.conda/envs/unidex/bin/python -m pip install \
  -r requirements-shadow-replay.txt
```

No heavyweight URDF, IK, or collision dependency was added. URDF limits are
parsed with Python's standard XML library.

## Run

From the workspace root:

```bash
/data/why/.conda/envs/unidex/bin/python tools/offline_shadow_replay.py \
  --config configs/shadow_replay.yaml \
  --checkpoint UniDex/finetune_checkpoints_origami_4090_freeze_vlm_bs4_10000steps/checkpoint-epoch=29-train_loss=0.0242.ckpt \
  --dataset-path dataset/Robotic_Origami_Challenge/season_POC22032_2026_05_14_19_21_01_train \
  --output-dir outputs/shadow_replay
```

Minimal checkpoint smoke test:

```bash
/data/why/.conda/envs/unidex/bin/python tools/offline_shadow_replay.py \
  --config configs/shadow_replay.yaml \
  --max-episodes 1 \
  --max-steps-per-episode 20
```

The output directory must be absent or empty. Existing artifacts are never
overwritten. Choose a new `--output-dir` for a later run.

T-Rex smoke test:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
/data/why/.conda/envs/trex/bin/python tools/offline_shadow_replay.py \
  --config configs/shadow_replay_trex.yaml \
  --max-episodes 1 \
  --max-steps-per-episode 20 \
  --output-dir outputs/shadow_replay_trex_smoke
```

The configured path is a training split, and the converted training zarr
contains episode IDs 0–2. Selecting those IDs produces a prominent leakage
warning. For a genuine test estimate, pass a separate held-out dataset or
held-out `--episode-ids`; do not randomly split neighboring frames.

## Stage and event annotations

No stage heuristic is invented. Supply trusted interval annotations:

```yaml
episodes:
  0:
    - {stage: approach, start_frame: 0, end_frame: 300}
    - {stage: grasp, start_frame: 300, end_frame: 420}
```

Intervals are `[start_frame, end_frame)`. Critical events use:

```yaml
episodes:
  0:
    - {event: first_contact, frame: 318}
    - {event: gripper_close, frame: 344}
```

Set `evaluation.stage_annotations` and `evaluation.critical_events` to these
files. Event-window size is configured in selected replay steps and converted
to source frames using the replay stride.

## Output

```text
outputs/shadow_replay/
├── config_resolved.yaml
├── dataset_summary.json
├── overall_metrics.json
├── metrics.csv
├── stage_metrics.csv
├── latency_metrics.json
├── perturbation_metrics.json
├── safety_violations.jsonl
├── episode_errors.jsonl                 # only when an episode fails
├── episode_results/
│   ├── episode_0000.npz
│   └── episode_0000_metrics.json
├── visualizations/
│   ├── episode_0000_actions.png
│   ├── horizon_error.png
│   └── latency_distribution.png
└── report.md
```

Each `.npz` stores `gt_action`, `pred_action_raw`,
`pred_action_denormalized`, `pred_joint_target`, `pred_action_safe`, optional
`pred_eef_action`, recorded robot/joint/TCP state, invalid-number flags, the
chunk-valid mask, stages/events, safety flags, and all latency layers. JSONL
frame records can be enabled with
`save_frame_jsonl: true`; they are off by default to avoid a very large
duplicate of the compressed arrays.

Every mutation/rejection record includes episode, step, horizon, violation
type, original value, limit, and whether the action was rejected or clipped.
An episode exception is recorded and later episodes continue.

## Metrics and robustness

The evaluator reports model-space and safe robot-joint-space MAE/MSE,
per-dimension MAE, chunk endpoint error, horizon-error curves, replan overlap
consistency, boundary discontinuity, temporal velocity/acceleration/jerk,
rotation6D SO(3) error, pose XYZ error, optional gripper classification,
per-stage/event metrics, safety counts, and latency
mean/median/P90/P95/P99.

Perturbations are separate runs with the same inference seed. They are never
stacked and horizontal flipping is intentionally unsupported. Available
perturbations are brightness, contrast, RGB noise, blur, occlusion, frame drop,
1…N frame delay, robot-state noise, missing action history, and camera interval
jitter. Missing action history is reported as not applicable for the current
UniDex model.

## Test

The test suite uses standard `unittest` and needs no extra test dependency:

```bash
python -m unittest discover -s tests -p 'test_shadow_replay.py' -v
```

It covers single actions, chunks, tail masks, normalization round trips,
quaternion angular error, gripper metrics, NaN/Inf and workspace safety, empty
and differently sized episodes, checkpoint/data dimension mismatch, the 82→65
adapter, and overlap consistency.

## Offline/closed-loop boundary

Dataset frame `t+1` is the result of the demonstrator's recorded command.
It is not the result of executing the model prediction at `t`. Consequently,
this framework can test perception/action inference plumbing, imitation error,
timing, robustness, and static executability. It cannot establish closed-loop
fold success, contact stability, recovery behavior, or robustness to
model-induced state distribution shift.

For a future live but non-actuating Shadow Mode, reuse:

- `model.py` for identical preprocessing/inference timing;
- `actions.py` for normalization and typed action adaptation;
- `safety.py` for limit checks and violation records;
- `metrics.py` and artifact/report writers.

Replace `LeRobotEpisodeDataset` with a timestamped live observation buffer.
Before any actuation is considered, add and validate the missing wrist IK,
singularity, self-collision, and environment-collision adapters plus an
independent emergency-stop path.
