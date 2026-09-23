#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${1:-fold-the-world/origami-policy:submission}"
ARCHIVE_NAME="${2:-fold-the-world-origami-policy-submission.tar.zst}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint-0-7000}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
ACTION_DIM="${ACTION_DIM:-65}"
PROTOCOL="${PROTOCOL:-origami-zenoh-v1}"

if [[ "${ARCHIVE_NAME}" != *.tar.zst ]]; then
    printf 'Archive name must end with .tar.zst: %s\n' "${ARCHIVE_NAME}" >&2
    exit 1
fi
if [[ "${ARCHIVE_NAME}" == */* ]]; then
    printf 'Archive name must be a filename, not a path: %s\n' "${ARCHIVE_NAME}" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    printf 'docker command not found\n' >&2
    exit 1
fi
if ! command -v zstd >/dev/null 2>&1; then
    printf 'zstd command not found\n' >&2
    exit 1
fi
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    printf 'Docker image not found: %s\n' "${IMAGE_NAME}" >&2
    exit 1
fi

SUBMISSIONS_DIR="${REPO_ROOT}/submissions"
mkdir -p "${SUBMISSIONS_DIR}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${SUBMISSIONS_DIR}/${RUN_TIMESTAMP}"
RUN_SUFFIX=0
while [[ -e "${RUN_DIR}" ]]; do
    RUN_SUFFIX=$((RUN_SUFFIX + 1))
    RUN_DIR="${SUBMISSIONS_DIR}/${RUN_TIMESTAMP}_${RUN_SUFFIX}"
done
mkdir -- "${RUN_DIR}"

ARCHIVE="${RUN_DIR}/${ARCHIVE_NAME}"
PARTIAL="${ARCHIVE}.partial"
CHECKSUM_FILE="${ARCHIVE}.sha256"

cleanup_partial() {
    if [[ -e "${PARTIAL}" ]]; then
        rm -f -- "${PARTIAL}"
    fi
}
trap cleanup_partial EXIT

# docker save preserves the image configuration, entrypoint, and layer metadata.
docker save "${IMAGE_NAME}" \
    | zstd -T0 -3 -f -o "${PARTIAL}"
mv -- "${PARTIAL}" "${ARCHIVE}"

zstd -t "${ARCHIVE}"
CHECKSUM_LINE="$(cd "${RUN_DIR}" && sha256sum "${ARCHIVE_NAME}")"
printf '%s\n' "${CHECKSUM_LINE}" | tee "${CHECKSUM_FILE}"

ARCHIVE_SHA256="${CHECKSUM_LINE%% *}"
ARCHIVE_SIZE="$(stat -c '%s' "${ARCHIVE}")"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE_NAME}")"
IMAGE_SIZE="$(docker image inspect --format '{{.Size}}' "${IMAGE_NAME}")"
CREATED_AT="$(date '+%Y-%m-%dT%H:%M:%S%:z')"

cat > "${RUN_DIR}/fold-the-world-submission.md" <<EOF
# Fold the World：T-Rex Office Inference 提交信息

| 字段 | 值 |
|---|---|
| 镜像标签 | \`${IMAGE_NAME}\` |
| 镜像 ID | \`${IMAGE_ID}\` |
| 推理协议 | \`${PROTOCOL}\` |
| 归档文件 | \`${ARCHIVE_NAME}\` |
| 归档 SHA-256 | \`${ARCHIVE_SHA256}\` |
| 归档大小 | \`${ARCHIVE_SIZE}\` bytes |
| Checkpoint | \`${CHECKPOINT_NAME}\` |
| 固定 action horizon | \`${ACTION_HORIZON}\` |
| action dimension | \`${ACTION_DIM}\` |
| 生成时间 | \`${CREATED_AT}\` |

提交文件：

- \`${ARCHIVE_NAME}\`
- \`${ARCHIVE_NAME}.sha256\`
- \`submission-manifest.json\`

提交前应在干净 Docker 环境中运行 \`docker load\`，并重新通过 office SDK 的
\`metadata\`、\`reset\`、\`infer\` 验证。
EOF

cat > "${RUN_DIR}/submission-manifest.json" <<EOF
{
  "submission_format": "origami-oci-archive-v1",
  "team_id": "fold-the-world",
  "image": "${IMAGE_NAME}",
  "image_id": "${IMAGE_ID}",
  "archive": "${ARCHIVE_NAME}",
  "archive_size_bytes": ${ARCHIVE_SIZE},
  "archive_sha256": "${ARCHIVE_SHA256}",
  "protocol": "${PROTOCOL}",
  "action_dim": ${ACTION_DIM},
  "action_horizon": ${ACTION_HORIZON},
  "checkpoint": "${CHECKPOINT_NAME}",
  "created_at": "${CREATED_AT}"
}
EOF

printf 'Image ID: %s\n' "${IMAGE_ID}"
printf 'Archive: %s\n' "${ARCHIVE}"
printf 'Submission directory: %s\n' "${RUN_DIR}"
