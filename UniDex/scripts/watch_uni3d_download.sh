#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_PATH="/data/why/foldVLA/checkpoints/unidex_download.log"
HF_ROOT="$HOME/.cache/huggingface/hub/models--BAAI--Uni3D"

now() {
    date "+%F %T"
}

log() {
    echo "[$(now)] $1" >> "$LOG_PATH"
}

find_snapshot_dir() {
    find "$HF_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1
}

have_core_models() {
    local snapshot_dir="$1"
    [ -f "$snapshot_dir/modelzoo/uni3d-ti/model.pt" ] &&
    [ -f "$snapshot_dir/modelzoo/uni3d-s/model.pt" ] &&
    [ -f "$snapshot_dir/modelzoo/uni3d-b/model.pt" ] &&
    [ -f "$snapshot_dir/modelzoo/uni3d-g/model.pt" ] &&
    [ -f "$snapshot_dir/modelzoo/uni3d-l/model.pt" ]
}

log "watcher started"

while true; do
    SNAPSHOT_DIR="$(find_snapshot_dir)"
    if [ -n "$SNAPSHOT_DIR" ] && have_core_models "$SNAPSHOT_DIR"; then
        log "Uni3D files ready, running move_pretrained_uni3d.sh"
        cd "$ROOT_DIR" || exit 1
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate unidex
        bash scripts/move_pretrained_uni3d.sh >> "$LOG_PATH" 2>&1
        log "move script finished with code $?"
        break
    fi

    log "waiting for Uni3D core checkpoints"
    sleep 60
done

tail -f "$LOG_PATH"
