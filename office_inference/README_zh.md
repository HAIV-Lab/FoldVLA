# Origami 推理工具包

本仓库是面向 Origami 竞赛队伍的公开开发套件，可用于：

- 构建实现 `origami-zenoh-v1` 的自包含 Docker/OCI 推理镜像；
- 使用合成观测数据验证镜像协议；
- 在预约时段内通过公开只读接口从实体机器人获取观测数据；
- 在队伍的本地机器上运行只读 Shadow/URDF 镜像测试；
- 导出 `.tar.zst` 镜像归档及其 SHA-256 校验和。

请从中文版的 [`PARTICIPANT_GUIDE_zh.md`](PARTICIPANT_GUIDE_zh.md) 开始阅读。

## 公开内容

```text
PARTICIPANT_GUIDE.md
docs/
  competition_participant_complete_guide.md
  participant_zenoh_submission.md
  robot_io_spec.md
  container_submission.md
  remote_participant_development.md

openpi-base-main/                   # OpenPI 推理/模型参考源码

sharpa_north_ces_lite_sdk-main/
  examples/
    policy_server_template.py
    check_zenoh_policy.py
    remote_observation_client.py
  participant_local_evaluator/
  tests/
```

主要组件：

- `policy_server_template.py`：与框架无关的生产环境 Zenoh 服务器模板；
- `check_zenoh_policy.py`：不连接机器人的公开黑盒验证器；
- `remote_observation_client.py`：预约时段后使用的公开只读观测客户端；
- `participant_local_evaluator`：在本地将真实的只读观测发送到最终镜像，并使用 URDF
  将预测轨迹可视化；
- `openpi-base-main`：OpenPI 推理和模型适配器参考源码。使用 OpenPI 的队伍仍必须自行打包
  检查点和运行时资源。

## 快速开始

```bash
cd sharpa_north_ces_lite_sdk-main
uv sync --frozen --no-install-project
uv run --no-sync python -m unittest discover -s tests -v
```

在预约时段前，你可以构建镜像并运行合成验证器。预约后，组织方会单独发送以下值：

```text
ORIGAMI_REMOTE_ENDPOINT=tls/<public-host>:<port>
ORIGAMI_REMOTE_SESSION_ID=<assigned-team-session>
ORIGAMI_REMOTE_TOKEN=<assigned-team-secret>
ORIGAMI_REMOTE_TLS_CA=/path/to/organizer-ca.pem
```

切勿将这些凭据写入 Git、源代码、Dockerfile、镜像层或日志。

## 公开张量契约

镜像接收：

```python
{
    "observation/image/head_left":      uint8[224, 224, 3],
    "observation/image/head_right":     uint8[224, 224, 3],
    "observation/image/wrist_left":     uint8[224, 224, 3],
    "observation/image/wrist_right":    uint8[224, 224, 3],
    "observation/state":                float32[65],
    "observation/state/joint_torque":   float32[65],
    "observation/tactile":              float32[60],
    "observation/image/tactile_deform": uint8[480, 1200, 3],
    "observation/image/tactile_raw":    uint8[480, 1600, 3],  # optional
    "prompt":                           str,
}
```

镜像返回：

```python
{"actions": float32[T, 65]}
```

动作必须是有限的、以弧度表示的绝对关节位置目标，并使用 `docs/robot_io_spec.md` 中定义的固定
65 维顺序。

## 安全边界

- 公开观测接口为只读接口，无法发送动作；
- 生产镜像只能连接组织方注入的隔离 Zenoh 路由器；
- 镜像不得包含 North SDK、机器人 IP 地址/话题、动作发布器或公开开发凭据；
- 组织方负责机器人 I/O、动作安全检查、Shadow 执行以及经授权的 Live 执行。

## 许可证

本仓库中的代码采用 Apache License 2.0 许可，详见 `LICENSE`。
第三方前端依赖和运行时依赖的许可证列于 `THIRD_PARTY_LICENSES.md`。
