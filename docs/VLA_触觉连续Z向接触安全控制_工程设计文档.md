# VLA 抓纸：触觉连续 Z 向接触安全控制工程设计文档

> 适用场景：VLA 输出关节位置 / Action Chunk，机器人具有触觉反馈和可用运动学模型。

| **字段** | **内容**                                                                 |
|----------|--------------------------------------------------------------------------|
| 文档版本 | V1.0                                                                     |
| 设计目标 | 降低抓取桌面纸张时的过度下压、持续压桌和 Action Chunk 累积下压风险       |
| 控制策略 | VLA 宏观动作 + Approach 限速 + Contact 阶段连续触觉 Z 控制 + Hard Reflex |
| 输出接口 | 安全修正后的关节目标 q_safe                                              |
| 部署原则 | 控制器独立于 VLA；可旁路、可回滚、可记录、可逐级开启                     |


# 目录

- [1. 目标、边界与设计原则](#1-目标边界与设计原则)
- [2. 系统架构](#2-系统架构)
- [3. 坐标系与信号定义](#3-坐标系与信号定义)
- [4. 状态机设计](#4-状态机设计)
- [5. 触觉信号处理](#5-触觉信号处理)
- [6. 连续 Z 向力控制器](#6-连续-z-向力控制器)
- [7. 从 Z 控制量到关节角命令](#7-从-z-控制量到关节角命令)
- [8. Action Chunk 中断与 Hard Reflex](#8-action-chunk-中断与-hard-reflex)
- [9. 软件模块与接口](#9-软件模块与接口)
- [10. 实时主循环伪代码](#10-实时主循环伪代码)
- [11. 参数标定与上线流程](#11-参数标定与上线流程)
- [12. 日志、监控与故障诊断](#12-日志监控与故障诊断)
- [13. 测试矩阵与验收标准](#13-测试矩阵与验收标准)
- [14. 常见问题与工程注意事项](#14-常见问题与工程注意事项)
- [附录 A. 推荐参数表](#附录-a-推荐参数表初始值必须通过实机标定替换)
- [附录 B. 最小实现清单](#附录-b-最小实现清单)

> **工程结论：** 第一版不需要修改 VLA 网络。VLA 继续输出关节角目标；安全层用 FK/Jacobian 将触觉控制器生成的 Z 向速度/位移注入关节空间，并在过载时立即中断剩余 Action Chunk。

# 1. 目标、边界与设计原则

## 1.1 目标

- 当手指接近桌面时自动降低向下速度，避免高速撞击。

- 当触觉检测到稳定接触后，桌面法向 Z 不再完全由 VLA 决定，而由触觉闭环连续调节。

- 当接触力超过软阈值时，禁止继续下压；超过硬阈值时立即中断当前 Action Chunk 并执行回撤。

- 尽量保持 VLA 的 X/Y、姿态、夹爪等任务意图，只最小化修改与“压桌”相关的运动分量。

- 控制器应能独立开关并记录原始动作与修正动作，便于 A/B 测试和问题定位。

## 1.2 非目标

- 本设计不负责解决纸张是否被成功分离、是否一次只抓起一张等任务语义问题。

- 本设计不替代机器人驱动层已有的关节限位、碰撞检测、急停和电机保护。

- 第一版不要求触觉必须标定成牛顿；只要触觉强度对接触压力单调可用即可。

## 1.3 核心原则

1. “低频 VLA、高频安全层”：VLA 负责目标意图，触觉控制器负责快速接触响应。

2. “进入接触后接管 Z”：触觉只接管桌面法向，不无必要地重写整个动作。

3. “软控制 + 硬反射”：连续控制解决正常接触；硬阈值解决异常和延迟。

4. “接触有滞回”：进入和退出接触使用不同阈值，避免状态抖动。

5. “所有阈值先标定后上线”：不直接套用固定牛顿数值。

# 2. 系统架构

```text
RGB / Proprioception / Tactile
│
▼
VLA
│ q_vla[k] / action chunk
▼
┌─────────────────────┐
│ Safety Supervisor   │
│ ├─ FK / Jacobian    │
│ ├─ Approach limiter │
│ ├─ Contact FSM      │
│ ├─ Force-Z control  │
│ └─ Hard reflex      │
└─────────────────────┘
│ q_safe[k]
▼
Robot Position Loop
│
tactile / q / status
└──────────────► Safety Supervisor
```

| **层级**          | **典型频率**                      | **职责**                                    | **是否允许阻断 VLA** |
|-------------------|-----------------------------------|---------------------------------------------|----------------------|
| VLA 推理层        | 5–30 Hz                           | 语义、抓取策略、生成关节目标或 Action Chunk | 否                   |
| Safety Supervisor | 建议 100–500 Hz；至少显著高于 VLA | 接触状态判断、Z 控制、动作投影、过载反射    | 是                   |
| 机器人底层位置环  | 机器人原生频率                    | 跟踪 q_safe、执行驱动器保护                 | 是（驱动器自身）     |
| 触觉采样          | 尽可能与安全层同级或更高          | 提供接触强度/法向压力                       | 不适用               |

> **重要：**如果 VLA 一次输出 8/16/32 个 step，Safety Supervisor 必须逐 step 或更高频率介入，而不是等整个 chunk 执行结束后才检查触觉。

# 3. 坐标系与信号定义

## 3.1 坐标约定

- 世界坐标系 W：桌面近似为平面。

- 桌面法向 n_table：定义为“离开桌面”的方向。对于水平桌面，可近似 $n_{table}=[0,0,1]^T$。

- 正 Z：远离桌面；负 Z：朝向桌面。

- 若桌面有倾斜，应使用标定得到的 n_table，而不是硬编码世界坐标 Z。

## 3.2 输入信号

| **变量**          | **单位/类型** | **说明**                     |
|-------------------|---------------|------------------------------|
| q                 | rad / vector  | 当前关节角                   |
| q_vla             | rad / vector  | VLA 当前希望执行的关节目标   |
| F_raw             | N 或无量纲    | 原始触觉压力/法向力/接触强度 |
| T_W_EE(q)         | SE(3)         | 末端位姿，FK 计算            |
| J(q)              | 6×N           | 末端 Jacobian                |
| z_table / n_table | m / vector    | 桌面模型                     |
| dt                | s             | 安全控制循环周期             |

## 3.3 输出信号

| **变量**    | **说明**                                         |
|-------------|--------------------------------------------------|
| q_safe      | 经触觉与安全约束修正后的关节目标                 |
| state       | FREE / APPROACH / CONTACT / OVERLOAD / RECOVERY  |
| reason_code | 本周期动作是否被限速、Z 接管、chunk 取消、回撤等 |

# 4. 状态机设计

```text
FREE
│ 距桌面 < d_approach
▼
APPROACH ── 稳定触觉 > F_enter ──► CONTACT
▲ │
│ 远离桌面 │ F > F_hard / 异常增长
│ ▼
└────────────── RECOVERY ◄── OVERLOAD
│
└─ 回撤完成且触觉 < F_exit → APPROACH / FREE
```

| **状态** | **Z 方向策略**                                         | **VLA 其他分量**   | **关键退出条件**    |
|----------|--------------------------------------------------------|--------------------|---------------------|
| FREE     | 完全跟随 VLA，但保留全局硬安全约束                     | 跟随 VLA           | 进入桌面邻域        |
| APPROACH | 限制最大向下速度                                       | 跟随 VLA           | 稳定接触 / 远离桌面 |
| CONTACT  | 触觉闭环产生 v_z_cmd，VLA 的 Z 只作为可选 feed-forward | 尽量跟随 VLA       | 触觉退出 / 过载     |
| OVERLOAD | 取消当前剩余 chunk，禁止下压                           | 冻结或保留安全方向 | 立即进入 RECOVERY   |
| RECOVERY | 沿桌面法向回撤到安全距离                               | 通常冻结任务动作   | 压力恢复且回撤完成  |

## 4.1 接触滞回

建议使用 F_enter > F_exit，并要求连续 M 个采样周期满足条件。这样可避免纸张滑动或噪声造成 CONTACT/FREE 高频切换。

```text
enter_contact = (F_filt >= F_enter) for M_enter consecutive samples
exit_contact = (F_filt <= F_exit) for M_exit consecutive samples
F_enter > F_exit
```

# 5. 触觉信号处理

## 5.1 触觉强度构造

如果传感器直接给法向力 F_n，可直接使用。若给 taxel/pressure map，可构造单调接触强度 S，再将以下所有 F 替换为 S。

```text
方案 A: S = sum(valid_taxels)
方案 B: S = mean(top_k_taxels)
方案 C: S = weighted_sum(taxels), 权重突出接触核心区域
```

## 5.2 零点与漂移补偿

- 机器人离开桌面且明确无接触时更新 baseline，接触状态下禁止快速更新 baseline。

- 使用 F_zero = EMA(F_raw) 估计零点，F = max(0, F_raw - F_zero)。

- 若传感器存在明显温漂，可加入慢时间常数 baseline；但必须在 CONTACT 时冻结或极慢更新。

## 5.3 低通滤波

第一版推荐一阶低通，不要用太长滑动窗口，以免产生大延迟。

```text
alpha = dt / (tau + dt)
F_filt[k] = F_filt[k-1] + alpha * (F[k] - F_filt[k-1])
```

工程建议：先观察原始触觉频谱和延迟，再选 tau。目标是“滤掉高频尖噪声，但不把真实接触拖后几十到数百毫秒”。

# 6. 连续 Z 向力控制器

## 6.1 控制目标

在 CONTACT 状态下，使触觉强度保持在目标 F_target 附近。定义正 Z 为离开桌面，因此压力过大时控制器必须输出正 Z 速度，压力不足时允许小幅负 Z 速度。

## 6.2 基本控制律（推荐第一版：P + deadband）

```python
e_F = F_filt - F_target

if abs(e_F) <= F_deadband:
    v_z_raw = 0.0
else:
    v_z_raw = Kp_F * e_F

v_z_cmd = clip(v_z_raw, -v_down_contact_max, +v_up_max)
```

符号检查：F_filt > F_target 时 e_F > 0，因此 v_z_cmd > 0，末端上抬；F_filt < F_target 时 v_z_cmd < 0，允许极慢继续下探。

> **推荐：**第一版先用 P 控制，不要一开始加入积分项。纸张、桌面和末端接触刚度可能很高，积分项容易在延迟和饱和下产生 wind-up。


## 6.3 可选 PI 版本

```python
e_F = F_filt - F_target
I = clamp(I + e_F * dt, I_min, I_max)
v_z_raw = Kp_F * e_F + Ki_F * I
v_z_cmd = clip(v_z_raw, -v_down_contact_max, +v_up_max)

# anti-windup
if v_z_raw != v_z_cmd:
    freeze_or_backcalculate(I)
```

## 6.4 非对称速度限制

向下应该慢，向上可以更快，因此使用非对称饱和。不要设置成对称 ±v_max。

| **参数**           | **作用**                 | **调参方向**                         |
|--------------------|--------------------------|--------------------------------------|
| v_down_contact_max | CONTACT 状态最大下探速度 | 越小越安全，但抓纸贴合更慢           |
| v_up_max           | 压力过大时最大卸载速度   | 可显著大于向下速度，但需避免机械冲击 |
| F_deadband         | 目标力附近不动作的区间   | 太小会抖动；太大力控制不精确         |
| Kp_F               | 压力误差→Z速度增益       | 从小到大增加，直到响应够快但不振荡   |

## 6.5 接近阶段限速

在 APPROACH 状态，尚未稳定接触时不使用力控制，而只限制朝桌面的法向速度：

```text
v_n_vla = n_table^T * v_vla
# 正方向离开桌面；负方向朝桌面
v_n_safe = max(v_n_vla, -v_down_approach_max)
```

若 VLA 输出的是关节位置而非速度，可用相邻目标 q_vla - q 通过 Jacobian 估算本周期法向位移/速度。

# 7. 从 Z 控制量到关节角命令

## 7.1 推荐思路：最小改动的 Jacobian 法向修正

目标不是重新求一套完整 IK，而是在 VLA 关节动作上添加最小范数修正，使桌面法向位移等于安全控制器要求。

```python
dq_vla = q_vla - q
Jp = J(q)[0:3, :]
J_n = n_table^T * Jp # 1 x N

dz_vla = J_n * dq_vla
dz_safe = v_z_cmd * dt
dz_corr = dz_safe - dz_vla

# 阻尼最小二乘法向修正
dq_corr = J_n^T * inv(J_n*J_n^T + lambda^2) * dz_corr
dq_safe = dq_vla + dq_corr
q_safe = q + dq_safe
```

因为 J_n 只有 1×N，这一步计算量很低，适合高频执行。lambda 用于降低奇异位形附近的数值爆炸。

## 7.2 保留 VLA 的切向运动

上式只修正桌面法向位移，理论上尽量保留 VLA 原来的切向动作（沿桌面 X/Y）及冗余自由度。这正适合抓纸：手指可以继续横向划入/夹取，但不会无控制地向下压。

## 7.3 关节约束与速率限制

```python
q_safe = clamp(q_safe, q_min + margin, q_max - margin)
dq_safe = clamp(q_safe - q, -dq_max*dt, +dq_max*dt)
q_safe = q + dq_safe
```

若机器人提供更严格的关节加速度/jerk 约束，应继续在底层执行。

## 7.4 多接触点 / 手指结构

如果左右手指都有触觉，可根据任务定义一个主控制量：例如 max(F_left, F_right) 用于安全卸载，mean(F_left, F_right) 用于目标贴合；同时记录左右差值用于检测偏压。第一版优先确保“任一侧过载就上抬”。

# 8. Action Chunk 中断与 Hard Reflex

## 8.1 为什么必须支持 chunk 中断

如果 VLA 已经生成一个持续向下的 action chunk，即使第 3 个 step 已经接触桌面，继续执行剩余 13 个 step 仍会累计下压。因此触觉安全层必须拥有比 chunk 更高优先级的动作覆盖权。

## 8.2 软阈值与硬阈值

| **条件**                  | **动作**                                    |
|---------------------------|---------------------------------------------|
| F_filt < F_soft          | 正常运行连续 Z 控制                         |
| F_soft ≤ F_filt < F_hard | v_z_cmd 至少不允许向下；优先卸载            |
| F_filt ≥ F_hard           | 立即取消剩余 chunk；进入 OVERLOAD；执行回撤 |
| dF/dt 异常大              | 即使未到 F_hard，也可提前触发 OVERLOAD      |

## 8.3 Hard Reflex 逻辑

```python
if F_filt >= F_hard or force_rise_rate >= dFdt_hard:
    cancel_remaining_vla_chunk()
    state = OVERLOAD
    hold_current_joint_target_for_short_guard_interval()
    execute_normal_retreat(distance=d_retreat, speed=v_retreat)
    state = RECOVERY

# RECOVERY 完成条件
if F_filt <= F_exit and retreat_progress >= d_retreat_min:
    request_new_vla_inference()
    state = APPROACH  # or FREE, depending on distance-to-table
```

> **注意：**Hard Reflex 不应等待下一次 VLA 推理。它必须由安全层本地立即执行。


# 9. 软件模块与接口

| **模块**         | **主要接口**                  | **职责**                                |
|------------------|-------------------------------|-----------------------------------------|
| TactileProcessor | `update(raw) -> TactileState`    | 零点、滤波、接触强度、dF/dt             |
| Kinematics       | fk(q), jacobian(q)            | 末端位姿与 Jacobian                     |
| ContactFSM       | `step(obs) -> state`             | FREE/APPROACH/CONTACT/OVERLOAD/RECOVERY |
| ForceZController | `step(F, state) -> v_z_cmd`       | 连续 Z 向控制                           |
| ActionProjector  | `project(q, q_vla, v_z) -> q_safe` | 将 Z 控制量注入关节空间                 |
| ChunkManager     | cancel(), next_step()         | 管理 VLA action chunk 的执行与中断      |
| SafetySupervisor | `step(all_inputs) -> Command`    | 总调度、reason_code、故障降级           |
| Logger           | log(frame)                    | 记录原始/修正动作与传感器               |

## 9.1 建议数据结构

```text
TactileState:
raw
zero
filtered
dFdt
contact_valid

SafetyCommand:
q_safe
state
chunk_cancelled
reflex_active
reason_code

ControllerConfig:
F_enter, F_exit, F_target, F_soft, F_hard
Kp_F, Ki_F, F_deadband
v_down_approach_max, v_down_contact_max, v_up_max
d_approach, d_retreat
lambda_dls
debounce_samples
```

## 9.2 实时线程建议

- VLA 推理线程：异步产生新的 action chunk。

- Safety/Execution 线程：固定周期运行；是唯一允许向机器人发送关节目标的线程。

- 触觉采样线程：若驱动支持独立高频采样，可异步更新最新值并附时间戳。

- 所有信号必须带时间戳；Safety Supervisor 应拒绝使用超时的触觉数据。

# 10. 实时主循环伪代码

```python
while robot_enabled:
    now = clock()
    q = robot.read_joint_position()
    tactile = tactile_processor.update(sensor.latest())

    # 1) 健康检查
    if tactile.is_stale or robot.has_fault():
        send_hold_or_safe_stop()
        continue

    # 2) 获取当前 VLA step（但不直接发送）
    q_vla = chunk_manager.current_target_or_hold(q)

    # 3) 运动学
    pose = kin.fk(q)
    J = kin.jacobian(q)
    dist = signed_distance_to_table(pose, table_model)

    # 4) 状态机
    state = fsm.step(dist, tactile.filtered, tactile.dFdt)

    # 5) 计算安全 Z 控制
    if state == FREE:
        q_safe = q_vla

    elif state == APPROACH:
        q_safe = projector.limit_downward_normal_speed(
            q, q_vla, J, n_table, v_down_approach_max
        )

    elif state == CONTACT:
        v_z = force_z_controller.step(tactile.filtered, dt)
        if tactile.filtered >= F_soft:
            v_z = max(v_z, 0.0)  # 不允许继续下压
        q_safe = projector.override_normal_motion(
            q, q_vla, J, n_table, v_z, dt
        )

    elif state == OVERLOAD:
        chunk_manager.cancel()
        q_safe = reflex_controller.start_or_continue_retreat(q, J, n_table)

    elif state == RECOVERY:
        q_safe = reflex_controller.continue_retreat_or_hold(q, J, n_table)

    # 6) 关节限位/速度/连续性约束
    q_safe = joint_safety_filter.apply(q, q_safe, dt)

    # 7) 发给机器人
    robot.send_joint_target(q_safe)

    # 8) 记录
    logger.log(now, q, q_vla, q_safe, tactile, dist, state)

    sleep_until_next_cycle()
```

# 11. 参数标定与上线流程

## 11.1 Phase 0：先验证运动学与坐标符号

1. 机器人悬空，给一个非常小的“正 Z”测试命令，确认末端确实远离桌面。

2. 验证 Jacobian 预测的法向位移符号与实际一致。

3. 验证桌面法向 n_table 正方向正确。

4. 确认 q_vla → FK 的单位、关节顺序和零位完全一致。

> **禁止跳过：**如果 Z 符号反了，触觉越大控制器会越往下压，这是最危险的实现错误。


## 11.2 Phase 1：触觉静态标定

1. 采集无接触 30–60 s 原始信号，得到 baseline、噪声标准差和最大漂移。

2. 用低速、人工监护方式接触桌面/纸张，采集“刚接触”“正常抓纸接触”“明显过压但仍在硬件安全范围内”的信号分布。

3. 设置 F_exit、F_enter、F_target、F_soft、F_hard，使各区间有足够裕量，不使用相邻几乎重合的阈值。

4. 验证不同手指、不同纸张、不同桌面位置的信号稳定性。

## 11.3 Phase 2：只开 APPROACH 限速

先不开 CONTACT 力控制，只验证接近桌面时最大向下速度确实被限制，且不影响自由空间动作。

## 11.4 Phase 3：开启 CONTACT P 控制

1. Kp_F 从很小开始；v_down_contact_max 设置得比 APPROACH 更小。

2. 人工缓慢把末端送入接触，观察 F_filt、v_z_cmd 和实际 Z 是否形成正确负反馈。

3. 逐步增加 Kp_F，直到力恢复速度满足要求但没有明显振荡。

4. 如果目标力附近频繁上下抖动，优先增加 deadband 或滤波，而不是立刻降低所有增益。

## 11.5 Phase 4：开启 Hard Reflex

1. 先用较保守的 F_hard，确认触发后 chunk 会立即取消。

2. 验证回撤方向一定远离桌面，且回撤距离/速度受限。

3. 验证回撤后不会自动继续旧 chunk，而是请求新的 VLA 推理。

## 11.6 Phase 5：真实抓纸 A/B 测试

- A：原 VLA；B：VLA + Safety Supervisor。

- 至少记录峰值触觉、接触持续时间、Hard Reflex 次数、抓纸成功率、额外执行时间。

- 目标不是只降低峰值力，也要确认抓纸成功率没有被过度保守的 Z 控制显著破坏。

# 12. 日志、监控与故障诊断

## 12.1 每个控制周期至少记录

| **类别** | **字段**                                             |
|----------|------------------------------------------------------|
| 时间     | timestamp、dt、sensor_age                            |
| VLA      | chunk_id、step_id、q_vla                             |
| 机器人   | q、q_safe、joint_error                               |
| 运动学   | EE pose、dist_to_table、J_n*dq_vla、J_n*dq_safe    |
| 触觉     | raw、baseline、filtered、dF/dt                       |
| 控制器   | state、F_target、v_z_cmd、saturation_flag            |
| 安全     | F_soft_hit、F_hard_hit、chunk_cancelled、reason_code |

## 12.2 推荐离线曲线

- F_filt 与 F_target / F_soft / F_hard 同图。

- v_z_vla 与 v_z_cmd / v_z_safe 同图。

- 末端到桌面距离 dist_to_table。

- chunk step 与 state 切换时间轴。

- Hard Reflex 前后 0.5–1 s 的高分辨率窗口。

## 12.3 典型故障定位

| **现象**           | **最可能原因**                          | **处理**                             |
|--------------------|-----------------------------------------|--------------------------------------|
| 压力越大越继续下压 | Z 符号/桌面法向错误                     | 立即停用控制器，重新验证坐标         |
| CONTACT 高频抖动   | 无滞回、噪声大、debounce 太短           | 拉开 enter/exit，增加轻度滤波        |
| 接触后仍持续升压   | 控制频率过低、chunk 未被覆盖、Kp 太小   | 确认执行线程优先级和 q_safe 真正下发 |
| 上下振荡           | Kp 太大、滤波延迟、机械刚度高           | 降低 Kp、扩大 deadband、减小向下速度 |
| 一碰就弹很高       | v_up_max 或 Kp 太大                     | 限制上抬速度与每周期位移             |
| 抓不到纸           | F_target 太低或 v_down_contact_max 太小 | 在安全范围内逐步增加目标接触强度     |

# 13. 测试矩阵与验收标准

## 13.1 单元测试

| **测试**            | **输入**               | **期望**                           |
|---------------------|------------------------|------------------------------------|
| Force sign          | F > F_target          | v_z_cmd > 0                       |
| Low force           | F < F_target          | v_z_cmd ≤ 0 且不超过下探限速       |
| Deadband            | abs(F-F_target) < band | v_z_cmd ≈ 0                     |
| Soft limit          | F ≥ F_soft             | 法向命令不得朝桌面                 |
| Hard limit          | F ≥ F_hard             | chunk_cancelled=True               |
| Stale tactile       | sensor_age 超限        | Hold / Safe stop，不继续执行旧动作 |
| Jacobian projection | 任意 q_vla             | J_n*dq_safe 接近 dz_safe          |

## 13.2 台架测试

1. 无纸桌面：低速接触，验证峰值触觉受控。

2. 单张纸：多个位置、不同末端姿态重复抓取。

3. 纸张边缘/褶皱：测试触觉瞬态。

4. 故意给 VLA 连续向下 chunk：验证接触后其余 step 不会继续积累下压。

5. 触觉断流/延迟注入：验证系统安全降级。

## 13.3 建议验收指标

| **指标**                  | **建议定义**                                                                 |
|---------------------------|------------------------------------------------------------------------------|
| Peak contact              | 整个抓取 episode 中最大 F_filt；应明显低于未加控制版本，并低于内部硬安全上限 |
| Soft exceed duration      | F > F_soft 的累计时间应很短                                                 |
| Hard reflex rate          | 正常成功抓取中应趋近于 0；出现时必须正确回撤                                 |
| Contact settling          | 进入 CONTACT 后，F 回到 F_target±deadband 的时间                             |
| Task success              | 抓纸成功率不应因安全层出现不可接受下降                                       |
| Action modification ratio | norm(q_safe-q_vla) / norm(q_vla-q)，用于监控安全层是否过度干预 |

# 14. 常见问题与工程注意事项

## 14.1 VLA 输出绝对关节角，为什么还能做 Z 控制？

因为可以将 q_vla - q 视为本周期期望关节位移，通过当前 Jacobian 得到其末端法向分量，再添加一个最小范数关节修正，使最终法向位移符合触觉控制器要求。

## 14.2 CONTACT 后是否完全忽略 VLA 的 Z？

第一版建议是“基本接管”。可以保留很小的 VLA Z feed-forward，但安全上限必须由触觉控制器决定。抓纸时 VLA 的 X/Y、姿态和夹爪动作仍可保留。

## 14.3 能不能只在 F > F_hard 时回撤，不做连续控制？

可以作为最小安全补丁，但体验会变成“撞上→触发→弹开”，峰值力和任务稳定性通常不如连续力控制。连续 Z 控制用于正常接触，Hard Reflex 只处理异常。

## 14.4 触觉没有牛顿标定怎么办？

只要信号随压力单调变化，就可在无量纲空间设定 F_target/F_soft/F_hard。工程上先保证相对控制正确，再做绝对力标定。

## 14.5 为什么不直接 clip 关节角？

关节角 clip 不知道哪一部分关节运动对应“向桌面压”。Jacobian 法向修正直接约束任务空间桌面法向，更符合问题本质。

## 14.6 如果腕部/夹爪外壳碰桌子，但触觉在指尖怎么办？

必须额外加入几何安全层：对 wrist、gripper body、finger side 等非预期接触部位使用 FK/碰撞模型维护最小桌面距离。触觉控制只处理允许接触的指尖区域。

# 附录 A. 推荐参数表（初始值必须通过实机标定替换）

| **参数**            | **初始工程策略**                    | **说明**                      |
|---------------------|-------------------------------------|-------------------------------|
| controller_hz       | 100–500 Hz 范围内尽量高             | 受机器人接口和触觉刷新率限制  |
| tau_filter          | 小时间常数                          | 以不引入明显接触延迟为原则    |
| F_exit              | 略高于无接触噪声上界                | 退出接触阈值                  |
| F_enter             | 高于 F_exit                         | 进入接触阈值                  |
| F_target            | 正常抓纸可接受接触水平              | 需实机标定                    |
| F_soft              | 高于 F_target                       | 触发“禁止继续向下”            |
| F_hard              | 明显高于 F_soft，但低于硬件危险水平 | 触发 chunk 取消和回撤         |
| F_deadband          | F_target 周围小区间                 | 抑制高频抖动                  |
| Kp_F                | 从小开始递增                        | 先 P 后 PI                    |
| v_down_approach_max | 低于自由空间下降速度                | 靠近桌面减速                  |
| v_down_contact_max  | 显著低于 approach                   | 接触后只允许极慢下探          |
| v_up_max            | 可高于向下速度                      | 用于快速卸载，但限制冲击      |
| d_approach          | 进入桌面邻域的距离                  | 根据 FK 精度和 VLA 误差确定   |
| d_retreat           | Hard Reflex 回撤距离                | 足够解除接触但不过大          |
| lambda_dls          | 小正数                              | Jacobian 阻尼                 |
| sensor_timeout      | 略大于正常触觉采样周期的若干倍      | 超时则降级为 Hold / Safe stop |

> **参数关系约束：**F_exit < F_enter < F_target < F_soft < F_hard；同时 CONTACT 的最大下探速度应小于 APPROACH 阶段。


# 附录 B. 最小实现清单

| **优先级** | **实现项**                         | **完成标准**                     |
|------------|------------------------------------|----------------------------------|
| P0         | 确认桌面法向与 Jacobian Z 符号     | 人工小步测试方向 100% 正确       |
| P0         | 触觉滤波 + enter/exit 滞回         | 无接触时不误触发，接触时可靠进入 |
| P0         | APPROACH 向下限速                  | VLA 大幅向下命令可被限速         |
| P0         | CONTACT P 控制                     | 压力大自动上抬，压力小仅极慢下探 |
| P0         | F_soft 禁止下压                    | 超过软阈值后 J_n*dq_safe ≥ 0    |
| P0         | F_hard chunk 取消 + 回撤           | 无需等待新 VLA 推理              |
| P1         | 完整日志与 reason_code             | 可复盘每次安全层干预             |
| P1         | 触觉断流/超时降级                  | 不使用陈旧触觉继续压桌           |
| P1         | 非触觉部位桌面几何约束             | gripper/wrist 不碰桌             |
| P2         | PI / 自适应目标力 / 左右指压力平衡 | 在 P 控制稳定后再增加            |

# 实施建议总结

建议按以下顺序上线：①运动学符号验证 → ②触觉滤波与状态机 → ③Approach限速 → ④Contact P 控制 → ⑤Soft/Hard 阈值 → ⑥Action Chunk 取消与回撤 → ⑦A/B 抓纸测试。第一版的成功标准不是“精确恒力”，而是“VLA 保留任务能力，同时不再出现持续、不可控地压桌”。
