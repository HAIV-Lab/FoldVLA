#!/usr/bin/env bash
set -euo pipefail

cd /data/why/foldVLA/UniDex

RUN_DIR=/data/why/foldVLA/UniDex/UniDex_training/origami_4090_freeze_vlm_bs4_10000steps
LOG="$RUN_DIR/train.log"
PNG="$RUN_DIR/loss.png"

mkdir -p "$RUN_DIR" finetune_checkpoints_origami_4090_freeze_vlm_bs4_10000steps

set +e
WANDB_MODE=offline \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3 \
/data/why/.conda/envs/unidex/bin/python finetune.py \
  train.load_checkpoint=null \
  +train.freeze_vlm=true \
  train.checkpoint_path=./finetune_checkpoints_origami_4090_freeze_vlm_bs4_10000steps \
  train.optimizer._target_=bitsandbytes.optim.AdamW8bit \
  'train.trainer.devices=[0,1]' \
  train.trainer.strategy=ddp_find_unused_parameters_true \
  train.trainer.max_epochs=100000 \
  +train.trainer.max_steps=10000 \
  train.trainer.precision=16-true \
  train.checkpoint.save_top_k=-1 \
  train.dataloader.batch_size=4 \
  train.dataloader.num_workers=2 \
  train.dataloader.val_ratio=0.05 \
  train.wandb.project=unidex_origami \
  train.wandb.run_name=origami_4090_freeze_vlm_bs4_10000steps \
  dataset.data_dir=../dataset/unidex_real \
  dataset.cache_dir=../dataset/unidex_real/cache_origami_4090_freeze_vlm_bs4_10000steps \
  dataset.pointcloud_size=1024 \
  dataset.use_generated_data=false \
  dataset.cache_data=true \
  dataset.use_cached_metadata=false \
  'dataset.hands=[Shadow]' \
  'dataset.data_dirs=[{relative_path:lerobot3_bimanual_shadow.zarr,hand_type:Shadow,hand_side:Both,action:"north ces task",sample_stride:3,interpolation_factor:1,generated_data_dir:null,sequence_num:null}]' \
  > "$LOG" 2>&1
status=$?

/data/why/.conda/envs/unidex/bin/python scripts/plot_loss_from_log.py "$LOG" --out "$PNG" >> "$LOG" 2>&1 || true

exit "$status"
