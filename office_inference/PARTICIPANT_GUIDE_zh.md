# Origami 参与者指南

本文档是收到此代码仓库的各团队的主要入口。每个团队必须完成以下两项任务：

1. 构建实现标准协议的自包含 Docker/OCI 推理镜像；
2. 在预约的开发时段内，通过组织方提供的公开只读接口获取实体机器人的观测，并验证本地推理适配器和最终镜像。

公开观测仅用于开发验证，不能用来控制机器人。提交的生产镜像也不能直接访问机器人；观测、动作安全检查和执行由组织方负责。

## 1. 预约前后可以做什么

### 预约前

在没有公开 IP 地址、会话或令牌时，你可以：

- 实现模型适配器；
- 构建自包含镜像；
- 使用合成观测运行公开黑盒验证器；
- 检查元数据、重置、推理、输入/输出形状和数据类型，以及动作时域；
- 导出 `.tar.zst` 镜像归档并计算其 SHA-256 校验和。

### 预约后

组织方会针对每个团队和预约的时间段分别发送以下值：

```text
ORIGAMI_REMOTE_ENDPOINT=tls/<public-host>:<port>
ORIGAMI_REMOTE_SESSION_ID=<assigned-team-session>
ORIGAMI_REMOTE_TOKEN=<assigned-team-secret>
ORIGAMI_REMOTE_TLS_CA=/path/to/organizer-ca.pem
```

如果启用了 mTLS，组织方还会发送团队专用的客户端证书和私钥。

需要本地 URDF Shadow 评估器的团队，也会通过组织方的分发渠道收到官方 North 资产包，其中包含 URDF 和网格文件。该资产包仅用于本地可视化，不得包含在生产提交镜像中。没有该资产包并不妨碍使用合成验证器或开发镜像协议。

以上值是临时开发凭据：

- 不要将它们写入 Git、Dockerfile、镜像层、源代码默认值或日志；
- 不要将它们包含在生产提交镜像中；
- 不要将它们转发给其他团队；
- 预约结束后，按照组织方的说明删除它们，或等待它们过期。

正式评估期间，组织方只会向镜像注入另一组隔离的环境变量：

```text
ORIGAMI_ZENOH_ENDPOINT=tcp/<isolated-router>:7447
ORIGAMI_SESSION_ID=<evaluation-session>
```

生产镜像不得依赖任何 `ORIGAMI_REMOTE_*` 变量。

## 2. 安装公开开发环境

```bash
cd /path/to/origami-inference-kit/sharpa_north_ces_lite_sdk-main
uv sync --frozen --no-install-project
uv run --no-sync python -m unittest discover -s tests -v
```

协议和观测适配不需要 North SDK，也不需要访问机器人的私有网络。

## 3. 标准推理输入

公开客户端返回的 `observation`，与生产镜像收到的 `origami-zenoh-v1/infer` 请求中的 `observation` 完全相同：

```python
{
    "observation/image/head_left":       uint8[224, 224, 3],
    "observation/image/head_right":      uint8[224, 224, 3],
    "observation/image/wrist_left":      uint8[224, 224, 3],
    "observation/image/wrist_right":     uint8[224, 224, 3],
    "observation/state":                 float32[65],
    "observation/state/joint_torque":    float32[65],
    "observation/tactile":               float32[60],
    "observation/image/tactile_deform":  uint8[480, 1200, 3],
    "observation/image/tactile_raw":     uint8[480, 1600, 3],  # optional
    "prompt":                            str,
}
```

四张相机图像均为 HWC、RGB、C-contiguous，取值范围为 `0..255`。组织方会将原生的 `1920x1536` 相机帧直接拉伸到 `224x224`；该操作不保持宽高比，也不会添加 padding 或 letterboxing，与训练数据中使用的方形变换一致。团队不得根据原始相机宽高比添加黑边。

`observation/state` 包含 65 维、以弧度表示的绝对关节角度。确切顺序见 `docs/robot_io_spec.md`。该文档还定义了关节力矩、触觉力和触觉图像网格。即使模型只使用其中一部分字段，图像处理器也必须接受每个字段。

本地开发流程和镜像服务应复用同一个适配器：

```python
class TeamPolicyAdapter:
    def reset(self) -> None:
        ...

    def infer(self, observation: dict) -> dict:
        model_input = self.preprocess(observation)
        model_output = self.model(model_input)
        actions = self.to_absolute_radian_actions(model_output)
        return {"actions": actions}
```

不要为公开测试和镜像服务分别维护预处理、归一化或关节映射实现。

## 4. 生产镜像输出要求

成功的推理响应必须包含：

```python
{
    "actions": np.ndarray((T, 65), dtype=np.float32)
}
```

要求：

- `T` 必须与元数据中的固定 `action_horizon` 一致；
- 每个值都必须是有限值；不允许出现 NaN 和 Inf；
- 每一行都必须是绝对关节位置目标；
- 数值单位必须是弧度；
- 65 列的顺序必须与 `observation/state` 和元数据中的 `joint_names` 完全一致；
- 模型内部使用的任何归一化、增量、速度、token 化或 padding 动作，都必须在镜像内部反归一化并完成语义转换。

例如，如果 OpenPI 内部输出 72 维，镜像必须将其映射为并返回 65 个物理维度。Padding 不得跨越协议边界。

## 5. 构建生产镜像

镜像必须自包含并包含：

- 模型代码和固定入口点；
- Checkpoint/权重；
- 归一化统计信息；
- Tokenizer/词表；
- 预处理、后处理以及 65 维关节映射；
- Zenoh、MessagePack 编解码器和所有运行时依赖；
- CUDA/框架运行时；
- 第三方许可证和声明。

运行时不会挂载团队源代码、checkpoint、配置或宿主机 Python 环境，也不会提供互联网下载。镜像必须在只读根文件系统、非 root 用户、删除能力以及隔离网络环境下运行。

启动后，镜像以 Zenoh client 模式连接 `ORIGAMI_ZENOH_ENDPOINT`，并且只声明以下接口：

```text
origami-zenoh-v1/metadata
origami-zenoh-v1/reset
origami-zenoh-v1/infer
```

镜像不需要 `EXPOSE` 端口，也不得使用 HTTP `/healthz`。镜像中也不得包含 North SDK、机器人 topic、动作发布器、远程观测令牌、SSH 密钥或云凭据。

关于生产服务器模板、通用 Dockerfile 和入口点的说明，请参阅：

- `sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py`；
- `docs/competition_participant_complete_guide.md` 的第 9–10 节；
- `openpi-base-main/`，其中包含 OpenPI 推理/模型参考实现。

OpenPI 团队应使用 `openpi-base-main/scripts/docker/submission-zenoh-bundled.Dockerfile`；其精确的命名上下文构建命令记录在 `sharpa_north_ces_lite_sdk-main/scripts/README.md` 中。

## 6. 预约前运行合成黑盒验证

首先按照 `docs/competition_participant_complete_guide.md` 第 13 节启动本地 Zenoh 路由器和最终镜像。然后运行：

```bash
cd sharpa_north_ces_lite_sdk-main
uv run --no-sync python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:7447 \
  --session-id local-contract-test \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 25
```

将 `25` 替换为镜像实际使用的固定时域。

轻量级 mock 可以使用默认的 10 秒超时。对于需要加载 GPU 权重或执行 JIT/XLA 编译的最终镜像，应根据实际冷启动时间，将超时增加到 180 秒或竞赛公布的数值。记录冷启动和稳态延迟。

验证器会检查：

- 元数据、全部 65 个关节名称以及时域；
- Reset；
- 完整的合成观测，包括四张 RGB 图像、力矩和触觉数据；
- 请求/回复封装和唯一请求 ID；
- `finite float32[T,65]`；
- 重复请求和查询延迟。

`PASS` 表明镜像符合协议，但不能证明模型在任务上的性能。你仍然必须预约实体机器人观测的访问权限进行验证。

## 7. 预约后读取公开实体机器人观测

将组织方发送的 CA 文件保存到只有当前用户可读取的位置。端点、会话和 CA 路径可以设置为环境变量。通过隐藏提示输入令牌，以避免令牌出现在 shell 历史中：

```bash
export ORIGAMI_REMOTE_ENDPOINT='tls/<public-host>:<port>'
export ORIGAMI_REMOTE_SESSION_ID='<assigned-team-session>'
export ORIGAMI_REMOTE_TLS_CA='/absolute/path/to/organizer-ca.pem'
read -rsp 'ORIGAMI remote token: ' ORIGAMI_REMOTE_TOKEN
export ORIGAMI_REMOTE_TOKEN
printf '\n'
```

将占位符替换为组织方发送的实际值，然后检查连接：

```bash
cd sharpa_north_ces_lite_sdk-main
uv run --no-sync python examples/remote_observation_client.py
```

成功输出包括：

```text
PASS: received policy-compatible observation
state=(65,)
cameras=[(224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3)]
tactile_deform=(480, 1200, 3)
```

在 Python 中直接验证标准适配器：

```python
import numpy as np

from examples.remote_observation_client import RemoteObservationClient

with RemoteObservationClient(
    endpoint=endpoint,
    session_id=session_id,
    token=token,
    tls_root_ca_certificate=ca_path,
) as client:
    observation = client.get_observation()
    result = adapter.infer(observation)

actions = np.asarray(result["actions"])
assert actions.dtype == np.float32
assert actions.ndim == 2 and actions.shape[1] == 65
assert np.isfinite(actions).all()
```

公开接口绝不会将本地预测发送给机器人。它不提供动作、预测或机器人控制 API。

## 8. 使用实体机器人观测测试最终镜像

公开包中的本地 Shadow 评估器可以将相同的公开观测发送给最终镜像：

```bash
cd sharpa_north_ces_lite_sdk-main
uv run --no-sync python -m participant_local_evaluator \
  --robot-assets-dir /absolute/path/to/organizer-north-assets
```

在浏览器中打开以下 URL：

```text
http://127.0.0.1:7861
```

在页面中：

1. 输入组织方发送的端点、会话、令牌和 TLS CA；
2. 输入本地最终镜像标签，或 `.tar.zst` 路径和 SHA-256 校验和；
3. 启动镜像并执行 reset；
4. 运行 Shadow，检查图像、状态、动作、兼容性检查结果以及 URDF 预测轨迹。

该平台仅在本地可视化预测结果，绝不会向实体机器人发送动作。公开令牌也绝不会注入团队镜像。

## 9. 导出提交包

```bash
IMAGE='team-name/origami-policy:submission'
ARCHIVE='team-name-origami-policy-submission.tar'

docker save -o "$ARCHIVE" "$IMAGE"
zstd -T0 -19 "$ARCHIVE"
sha256sum "${ARCHIVE}.zst" > "${ARCHIVE}.zst.sha256"
zstd -t "${ARCHIVE}.zst"
```

请遵循竞赛公告中的提交内容和命名规则。至少保留：

- `.tar.zst` 镜像归档；
- 其 SHA-256 校验和；
- 镜像 ID/标签；
- 动作时域；
- GPU/CPU/RAM/共享内存要求；
- 第三方许可证和声明。

提交前，请在一台干净的机器上运行 `docker load`，重启镜像，然后再次通过公开验证器。

## 10. 最终检查清单

- [ ] 本地适配器可以直接处理 `RemoteObservationClient.get_observation()`；
- [ ] 四张相机图像均以 HWC RGB `uint8[224,224,3]` 接收，且没有额外的 letterboxing；
- [ ] 状态、关节力矩、触觉数据和触觉图像网格的形状与数据类型正确；
- [ ] 状态是以弧度表示的有限 `float32[65]`；
- [ ] 动作是以弧度表示的有限 `float32[T,65]` 绝对值；
- [ ] 三个 queryable——metadata、reset 和 infer——均通过验证器；
- [ ] 镜像包含代码、权重、归一化统计信息、tokenizer 和所有依赖；
- [ ] 镜像不依赖宿主机挂载或运行时下载；
- [ ] 镜像在只读、非 root、离线环境下运行；
- [ ] 镜像不包含 `ORIGAMI_REMOTE_*` 凭据、证书私钥或机器人通信代码；
- [ ] `.tar.zst` 归档完整性和 SHA-256 校验和已经验证。

## 11. 权威文档

- 完整流程：`docs/competition_participant_complete_guide.md`
- 生产 Zenoh 线协议：`docs/participant_zenoh_submission.md`
- 观测/动作张量契约：`docs/robot_io_spec.md`
- 容器提交要求：`docs/container_submission.md`
- 公开只读接口：`docs/remote_participant_development.md`

该公开包不包含组织方的实体机器人部署、机器人的私有网络、动作发布器、历史聊天记录或旧版 WebSocket 服务代码。
