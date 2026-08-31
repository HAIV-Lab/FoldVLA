# 公开只读真实机器人观察开发接口

该界面允许参赛团队通过以下方式读取真实机器人的观测数据：
组织者在开发过程中提供的公共Zenoh端点，并进行验证
在他们自己的机器上运行推理代码。它使用与以下情况相同的观察对象：
`infer`输入是针对最终提交的图像接收到的。

这不是机器人控制界面。公共服务仅声明一个
可查询，但不提供任何操作、预测、机器人主题、North SDK 或其他服务。
发布权限。

## 两个独立的工作流程

### 1. 主办方本地形象评估

收到参赛者图片后，组织者会在隔离环境中运行该图片。
容器。组织者的环境提供观察结果：

-Shadow：在不发送数据的情况下，显示 URDF 上的预测轨迹。
真实机器人动作；
-直播：仅由主办方在受控条件下启用并执行
经过安全检查后。

参与者图像遵循`origami-zenoh-v1`并声明了三个
`metadata` 、 `reset`和`infer` 可查询对象。参见
`participant_zenoh_submission.md` 。

### 2. 面向参与者的公开只读开发

组织者为每个队伍分配：

```文本
ORIGAMI_REMOTE_ENDPOINT= tls /<public-host>:<port>
ORIGAMI_REMOTE_SESSION_ID=<opaque-team-session>
ORIGAMI_REMOTE_TOKEN=<per-team-secret>
ORIGAMI_REMOTE_TLS_CA=/path/to/organizer- ca.pem
```

公共端点和凭据不包含在公共代码包中。
组织者会将它们分别发送给每个团队预留的开发名额。
并可能在预订结束后撤销或更改预订。在收到之前
即使持有这些凭证，参与者仍然可以构建自己的图像并使用合成凭证。
验证器。收到凭证后，他们可以验证真实机器人。
观察结果和最终图像。

参与者通过`origami-remote-v1`获取观测数据。此界面
无法控制机器人，也不能取代正式的图像提交方式。
工作流共享完全相同的推理输入模式。

## 最低限度使用

安装完 SDK 项目的冻结依赖项后：

```python
从 examples.remote_observation_client 进口 远程观察客户端

client = RemoteObservationClient(
    "tls/<public-host>:<port>",
    session_id="<assigned-team-session>",
    token="<assigned-team-secret>",
    tls_root_ca_certificate="/secure/origami-ca.pem",
)

obs = client.get_observation ( )
result = policy.infer ( obs )
```

` obs`可以直接传递给参与者的推理代码。模板
未导入机器人 SDK，也不包含任何机器人地址、内部主题等信息。
或操作代码。

建议使用上下文管理器：

with RemoteObservationClient(
    endpoint,
    session_id=session_id,
    token=token,
    tls_root_ca_certificate=ca_path,
) as client:
    while developing:
        observation = client.get_observation()
        prediction = policy.infer(observation)
        visualize_or_record_locally(prediction)
```

`prediction`仅存在于参与者的计算机上。远程模板不会执行任何操作。
不要把它送回给机器人。

## 命令行连接性检查

不要将令牌保存到 shell 历史记录中。通过环境变量传递它：

```bash
cd sharpa_north_ces_lite_sdk-main

export ORIGAMI_REMOTE_ENDPOINT='tls/<public-host>:<port>'
export ORIGAMI_REMOTE_SESSION_ID='<assigned-team-session>'
export ORIGAMI_REMOTE_TLS_CA='/secure/origami-ca.pem'
read -rsp 'ORIGAMI remote token: ' ORIGAMI_REMOTE_TOKEN
export ORIGAMI_REMOTE_TOKEN
printf '\n'

uv sync --frozen --no-install-project
uv run --no-sync python examples/remote_observation_client.py
```

## 参与者本地Shadow图像评估器

` participant_local_evaluator`将公开的只读观测值连接到
参与者的正式提交图像保存在参与者的计算机上。它不
依赖于North SDK，无法发送真实机器人指令：

```文本
origami-remote-v1（公开，只读）
-> 本地 Python 后端
-> 隔离的 Docker 网络 + 可信的Zenoh路由器
-> 参与者图像 origami-zenoh-v1 元数据/重置/推断
-> 本地验证和 North URDF 播放器，接下来的 100 步
```

首先将主办方提供的官方北方资产包提取出来
参与者的计算机。该目录必须至少包含：

```text
<north-assets>/
  urdf/north_poc2_2_with_hand_description.urdf
  meshes/...
```

启动评估器：

```bash
cd sharpa_north_ces_lite_sdk-main
uv sync --frozen --no-install-project
uv run --no-sync python -m participant_local_evaluator \
  --robot-assets-dir /absolute/path/to/north-assets
```

默认情况下，它只监听以下地址：

```文本
http://127.0.0.1:7861
```

Web界面工作流程：

1.输入公共端点、会话、令牌和 TLS CA 路径，然后单击
“连接并读取一帧”；
2.选择比赛提交的` .tar.zst`文件，将其流式传输到本地
后端，或者输入绝对本地路径和所需的匹配 SHA-256。
图片标签会在加载完成后自动填充；
3.启动镜像。评估器会创建一个内部 Docker 网络，该网络是可信的。
   Zenoh路由器和团队容器；
4.重置后，运行 Shadow。评估器获得一个新的真实观测值，并且
调用图像的`infer` 方法；
5.查看四个 RGB 图像，当前 65 维状态，完成 Three.js。
URDF/STL 当前/预测的 3D 模型、兼容性和时间表。

如果模型的预测期短于 100，评估器将执行多次迭代。
开环式代码块。每个新代码块都使用前一个代码块的最后一步作为后续步骤。
本地“观察/状态” ，而图像和提示仍然来自同一位置。
远程快照，直到生成最多请求的 100 个步骤为止。
该过程仅存在于参与者的计算机上。

### 本地容器边界

团队容器使用：

-一个内部 Docker 网络，只有受信任的路由器才能访问该网络。
网络;
- `--read-only` 、 `--cap-drop ALL`和`no-new-privileges` ；
- `--user 65532:65532` ;
- `/ tmp`和` / run` tmpfs挂载；
- CPU、RAM、共享内存和PID限制；
-仅注入了`ORIGAMI_ZENOH_ENDPOINT`和`ORIGAMI_SESSION_ID` 。

公共端点、会话和令牌永远不会传递到团队镜像中。
令牌仅存储在本地 Python 后端的内存中。HTTP 状态，
错误信息、日志和响应中都不会回显令牌，并且页面会清除令牌。
提交后立即输入。浏览器未运行Zenoh客户端。

评估者的 POST 请求路由仅限于远程连接和存档流传输。
上传/本地路径加载、镜像启动/停止、策略重置和影子推断。
没有用于实际机器人执行或命令发布的途径。URDF 和网格
`--robot-assets - dir`下的扩展名提供；绝对路径
路径、 ` ..`和符号链接转义符将被拒绝。

### 验证级别

评估器始终严格检查输出是否为有限的`float32[T, 65]`类型，并且
每个区块地平线都与元数据匹配。如果所有 65 个合约节点都在
官方 URDF 提供可解析的 它还包含`lower` 、 `upper`和`velocity`值。
检查位置，在相邻步之间跳跃，并根据计算出的速度
页面刷新率设置。如果资源缺失或 URDF 文件不完整，则页面会出错。
明确地显示了降级后的“形状有限”级别，而不是将其呈现为
完全兼容URDF。

如果组织者启用了mTLS ，还需设置：

```bash
export ORIGAMI_REMOTE_TLS_CERT='/secure/team-a-cert.pem'
export ORIGAMI_REMOTE_TLS_KEY='/secure/team-a-key.pem'
```

` tcp /127.0.0.1:<端口>`可用于本地封闭网络调试。正式
公共部署必须使用` tls /` 、受信任的 CA 以及每个实例的单独令牌。
团队。组织者可能还会要求使用mTLS 。

## 电汇合同

每个团队都只有一个只读密钥：

```文本
origami-remote-v1/{ session_id }/observation
```

请求使用MessagePack ，并采用仓库的msgpack-numpy编码：

```python
{
    "protocol_version": "origami-remote-v1",
    "operation": "observation",
    "request_id": unique_string,
    "session_id": assigned_session_id,
    "token": assigned_token,
}
```

成功的回复反映了四个相关领域：

```python
{
    "protocol_version": "origami-remote-v1",
    "operation": "observation",
    "request_id": request_id,
    "session_id": session_id,
    "observation_timestamp": unix_seconds,
    "metadata": {
        "protocol_version": "origami-v1",
        "observation_schema": "policy-infer-input",
        "observation_fields": {
            "<field-key>": {
                "dtype": "uint8" | "float32" | "str",
                "shape": [...],
                "required": bool,
            },
            # Exact entries match docs/robot_io_spec.md.
        },
        "joint_names": [...exact 65 names...],
    },
    "observation": {
        "observation/image/head_left": uint8[224, 224, 3],
        "observation/image/head_right": uint8[224, 224, 3],
        "observation/image/wrist_left": uint8[224, 224, 3],
        "observation/image/wrist_right": uint8[224, 224, 3],
        "observation/state": float32[65],
        "observation/state/joint_torque": float32[65],
        "observation/tactile": float32[60],
        "observation/image/tactile_deform": uint8[480, 1200, 3],
        "observation/image/tactile_raw": uint8[480, 1600, 3],  # optional
        "prompt": str,
    },
}
```

此“观察”字段与“观察”字段完全相同。
形式化映像收到的`origami-zenoh-v1/infer`请求。

组织者直接从其原始尺寸拉伸了四张 RGB 图像。
将分辨率从1920x1536降至224x224 ，不保持宽高比或添加其他内容
填充/信箱式装裱。参与者将收到最终的公开观察结果，并且
不得根据原始相机宽高比添加黑边。

该模板严格验证以下内容：

-回复信封和唯一请求 ID；
-四个 RGB 和触觉变形/原始图像的形状和dtype ；
-状态、关节扭矩和形状、 dtype以及NaN /Inf的缺失
触觉向量；
-语义协议和精确联合顺序；
-观察 Unix 时间戳和新鲜度；
- 64 MiB 有效载荷限制和安全的 NumPy数据类型。

## 错误和速率限制

应用程序错误使用结构化回复：

```python
{
    # correlation envelope...
    "error": {
        "code": "UNAUTHORIZED" | "RATE_LIMITED" | "BUSY"
                | "OBSERVATION_UNAVAILABLE" | "INVALID_REQUEST",
        "message": "...",
        "retryable": True | False,
    },
}
```

`RATE_LIMITED` 、 `BUSY`和瞬态`OBSERVATION_UNAVAILABLE`错误
使用退避策略重试。请勿通过增加请求次数来绕过速率限制。
并发性。该接口有意只允许一个并发请求。
公共客户端不能干扰本地真实机器人评估。

## 安全边界

-每个队伍的Tokens必须是唯一的、可撤销的、可轮换的，并且绝不能是
写入到图像、Git 或日志中；
-组织者中继器仅在身份验证后才会读取最新观测数据
速率限制检查通过；
-该中继仅调用只读快照，既不保存也不调用任何快照。
行动出版商；
-组织者仅暴露 TLS 中继，不暴露内部机器人服务或
评估管理界面；
-生产环境部署应保留访问审计、连接限制和
中继前后的带宽限制；
-接收观察结果并不赋予参与者对真实机器人的控制权。

实施参与者唯一需要阅读的内容是
`sharpa_north_ces_lite_sdk-main/examples/remote_observation_client.py` 。
组织者负责服务器部署和机器人集成。


