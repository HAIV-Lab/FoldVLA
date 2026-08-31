# Fold the World：T-Rex Office Inference 提交清单

本目录用于保存竞赛提交说明模板和打包脚本。脚本每次运行会在仓库根目录的
`submissions/` 下按创建时间生成独立子目录，保存该次的归档、校验文件和生成文档。归档文件名遵循
`office_inference/docs/competition_participant_complete_guide.md` 的命名示例，保留
`-submission` 后缀。

## 1. 需要提交的文件

| 文件 | 说明 |
|---|---|
| `fold-the-world-origami-policy-submission.tar.zst` | 使用 `docker save` 导出的自包含 Docker/OCI 镜像归档 |
| `fold-the-world-origami-policy-submission.tar.zst.sha256` | 归档文件的 SHA-256 校验文件 |
| `fold-the-world-submission.md` | 镜像和提交元数据说明；是否上传以竞赛页面要求为准 |

归档中应已经包含模型代码、基础模型、checkpoint、tokenizer/processor、归一化统计信息、运行时依赖和固定入口点。不需要单独上传原始 checkpoint、T-Rex 源码或主机环境配置。

## 2. 提交元数据

| 字段 | 值 |
|---|---|
| 队伍 | Fold the World |
| 镜像标签 | `fold-the-world/origami-policy:submission` |
| 镜像 ID | `sha256:<image-id-64-hex>` |
| 推理协议 | `origami-zenoh-v1` |
| 归档文件 | `fold-the-world-origami-policy-submission.tar.zst` |
| 归档 SHA-256 | `<archive-sha256-64-hex>` |
| 归档大小 | `<archive-size-bytes>` bytes |
| Checkpoint | `<checkpoint-name-or-step>` |
| 固定 action horizon | `<T>` |
| action dimension | `65` |

## 3. 镜像要求

- 镜像必须运行 `metadata`、`reset` 和 `infer` 三个 Zenoh queryable；
- 输出必须是有限的 `float32[T,65]` 绝对关节位置，单位为弧度；
- 镜像以非 root 用户运行，并支持只读根文件系统和离线运行；
- 运行时只使用组织方注入的 `ORIGAMI_ZENOH_ENDPOINT` 和 `ORIGAMI_SESSION_ID`；
- 不得包含远程观测令牌、证书私钥、机器人通信代码或其他秘密。

## 4. 归档生成与校验

```bash
submission/package_office.sh fold-the-world/origami-policy:submission
```

脚本默认将归档、`.sha256` 校验文件、提交说明和 `submission-manifest.json` 写入
`submissions/<创建时间>/` 目录。

提交前应在干净 Docker 环境中验证：

```bash
RUN_DIR='submissions/<创建时间>'
ARCHIVE="${RUN_DIR}/fold-the-world-origami-policy-submission.tar.zst"
IMAGE='fold-the-world/origami-policy:submission'

zstd -dc "$ARCHIVE" | docker load
docker image inspect "$IMAGE"
```

并使用 office SDK 再次通过 `metadata`、`reset`、`infer` 验证。

## 5. 资源信息

提交页面如要求资源申报，填写实际测量值；可参考：

- GPU：`1 × NVIDIA CUDA GPU`；
- CPU：`<vCPU>`；
- RAM：`<RAM-GiB>`；
- 共享内存：`<shm-GiB>`；
- `/tmp`：`<tmpfs-size>`；
- 冷启动时间：`<startup-seconds>`。

最终 image ID、归档 SHA-256、大小和 checkpoint 信息必须以实际生成的提交包为准。
