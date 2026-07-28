### env: env_isaaclab@d99a597b
how: conda env `env_isaaclab` at `/data/why/.conda/envs/env_isaaclab`
tier: {cpus: 80, mem_gib: 251, gpus: 1}
gpu: physical index 3, RTX 4090, UUID GPU-3f829d51-d88d-8eb4-1880-f22f67e0ee32
validated: 2026-07-27 (imports + 20-step North POC2.2 articulation witness)
invocation: `cd /data/why/foldVLA/sharpa-rl-lab && conda run --no-capture-output -n env_isaaclab env CUDA_DEVICE_ORDER=PCI_BUS_ID python -u rl_isaaclab/scripts/test_north_poc2_2_asset.py --headless --device cuda:3 --disable_collisions --fast_exit --steps 20 2>&1 | tee logs/north_poc2_2_gpu3_smoke.log`
expect: `GPU_WITNESS ... cuda:3 ... RTX 4090`; `ASSET_WITNESS ... initialized=True bodies=106 joints=65`; `PHYSICS_WITNESS ... finite=True`
log: `/data/why/foldVLA/sharpa-rl-lab/logs/north_poc2_2_gpu3_smoke.log`
gotcha: Isaac Sim reports inotify `errno=28` because host limits are 65536 watches / 128 instances; this is not disk exhaustion. Full collision cooking did not finish within 10 minutes. The exported physics layer also contains unresolved fingertip `/visuals/*_fingertip` reference paths.

### env: trex@2e070f24
how: conda env `trex` at `/data/why/.conda/envs/trex`
tier: {cpus: 80, mem_gib: 251, gpus: 1}
gpu: physical PCI index 3, RTX 4090; set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
weights: `/data/why/foldVLA/checkpoints/T-Rex-origami-posttrain/t-rex_origami_65d_freeze_vlm/t-rex_origami_65d_freeze_vlm_2x4090_3epoch_0702_145407/checkpoint-0-60000` (8.0 GiB model.pt)
validated: 2026-07-28 (imports + seeded CUDA matmul + AV1/libdav1d frame decode + shared test.py strict-LoRA 20-frame replay)
invocation: `cd /data/why/foldVLA && env CUDA_DEVICE_ORDER=PCI_BUS_ID /data/why/.conda/envs/trex/bin/python -u tools/offline_shadow_replay.py --config configs/shadow_replay_trex.yaml --max-episodes 1 --max-steps-per-episode 20 --output-dir outputs/shadow_replay_trex_lora_current_logic 2>&1 | tee outputs/shadow_replay_trex_lora_current_logic.log`
expect: `TREX_CHECKPOINT_WITNESS action_dim=65 chunk=16 lora_modules=196 strict=True`; dataset summary `device: cuda:3`; `SHADOW_REPLAY_COMPLETE`
log: `/data/why/foldVLA/outputs/shadow_replay_trex_lora_current_logic.log`
gotcha: OpenCV cannot decode the AV1 dataset videos on this host; use PyAV 15.1.0 linked with libdav1d. The post-train checkpoint omits LoRA rank/alpha metadata, so the shared `scripts/test.py` loader infers rank 16 from `lora_A`, uses the recorded alpha 32 supplied by config, and requires a strict state-dict match.
