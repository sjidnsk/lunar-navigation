# 正式能力内建化与飞跃式无累计燃料设计

日期：2026-08-08

状态：已按用户“已经设定的就是正式的；本项目不需要担心燃料问题”批准

## 1. 修订结论

本设计修正两项已经偏离当前项目边界的旧假设：

1. `ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml`
   中已批准的三平台 capability v2 就是正式运动能力，不再等待另一份外部
   `lunar-training-capability-freeze/v1`，也不要求用 URDF/mesh 打包来再次证明这些数值。
2. 飞跃式平台在本项目中没有跨跳燃料预算。质量、比冲和参考推进剂只定义每次独立规划所用的
   固定单跳弹道包络；连续探索不累计扣减燃料，不因历史跳跃次数改变下一目标的可达性。

本设计覆盖并取代以下旧规则：

- `2026-08-08-sensor-observation-capability-design.md` 中“仍需外部正式能力 closure”的部分；
- `2026-08-07-hopper-repeat-goal-fuel-design.md` 中提交和递减剩余燃料的部分；
- 旧 Volume 3 训练门中把外部 capability bundle、URDF 和 mesh closure 当作正式训练前置条件的部分。

其余现行权威不变：当前 C++ v3 路径规划算法、三平台 capability v2、30 m/360° 保守观测、
飞跃式单次安全凸落区与简单抛物线认证继续有效。

## 2. 正式能力身份

正式能力由仓库内三个受版本控制的输入共同组成：

- `platform_capability_schema_v2.yaml`：允许字段、类型和 retired 字段规则；
- `three_platform_capability_freeze_v1.yaml`：三平台已批准值和逐字段来源；
- `training_semantics.py`：共享 `30 m/360°` 传感器训练语义。

`three_platform_capability_freeze_v1.yaml` 的 `platforms` 规范化摘要必须等于文件内
`freeze_digest_sha256`。正式训练身份再绑定该摘要、观测语义摘要和当前规划器源码提交。任一值变化
都要求新能力版本、新摘要以及重新生成 cache/checkpoint；不允许静默沿用旧 Volume 3 产物。

训练包提供一个只读的“项目正式能力适配器”，直接把上述 YAML 转换成 C++ v3 bridge 的三类
typed capability。正式 `train`、`resume`、`evaluate` 和传感器性能基准都调用同一个适配器，
不再接受可替换正式权威的外部 capability lock。

URDF 和 mesh 仍可用于 Isaac Sim、RViz 外观、碰撞代理核对及设备集成，但它们不是探索模型训练的
运动能力门。当前规划器实际使用的外形、足迹、净空、落区支撑半径和飞行管半径已经明确冻结在
capability v2 中；训练不因缺少与这些数值重复的资源包而被阻塞。

## 3. 飞跃式单跳能力语义

飞跃式每次动作仍执行以下认证：

1. 目标是安全凸落区而不是多跳路线；
2. 从当前静止起飞位姿到目标落区求解简单月面抛物线；
3. 使用正式 capability 中的质量、比冲、参考推进剂、月面重力和安全裕度，计算本次规划可使用的
   固定单跳 `delta-v` 包络；
4. 验证整条飞行管、目标落区坡度、平面残差、支撑半径和横向安全裕度；
5. 通过后输出本跳名义抛物线和落区证据。

参考推进剂的含义是“每次独立规划的能力标定条件”，不是任务资源。它不随动作递减，也不表示
平台真实油箱。由此，同一平台连续选择多个安全且在固定单跳包络内的目标时，每一跳面对相同的
可达性边界。

## 4. 状态、奖励与接口边界

探索策略输入中不增加燃料通道；候选特征、价值网络和 PPO loss 不读取燃料。奖励不奖励节油、
不惩罚推进剂等价值，也不因“燃料耗尽”终止 episode。

C++ 规划器不得要求新鲜 `HopperPropellantState` 才能规划。该 Topic 从本项目的活动输入基线和
`PlanMotion` 快照构建中移除；消息定义可以暂留为未使用的兼容 schema，但不会再创建订阅或参与
输入时序校验。

`HopSegment` 不再输出 `ideal_fuel_required_kg`、`certified_fuel_required_kg` 或
`expected_remaining_usable_fuel_kg`。对外只保留 `required_delta_v_mps` 和
`available_delta_v_mps`，分别说明本跳带安全裕度的需求和固定正式单跳包络。内部可用火箭方程把
正式参考质量、比冲和参考推进剂换算成 `available_delta_v_mps`，但不产生任何可提交的剩余量。
`HOPPER_FUEL_INSUFFICIENT` 由 `HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED` 取代。

仓库外 Isaac/ROS/RViz 测试桥停止执行 `0.200 -> 0.189 -> ...` 的递减链，不再发布推进剂 Topic、
提交落地后的剩余量或因提交失败阻止下一次 Goal。第二跳只依赖新着陆位姿、新地图和同一个正式
单跳能力。

## 5. 数据流

```text
项目正式 capability v2 YAML
  -> 摘要与 schema 校验
  -> 三平台 typed capability
  -> 当前 C++ v3 规划器

HOPPER 当前位姿 + 安全凸落区
  -> 固定正式单跳弹道包络
  -> 抛物线/飞行管/落区认证
  -> 执行反馈与真实观测 reveal
  -> 下一探索决策（仍使用同一单跳包络）
```

没有“上一跳剩余燃料 -> 下一跳规划”的边。

## 6. 失败语义

飞跃式只因输入、地图、落区安全、抛物线解或飞行管认证失败而拒绝。有限地图的候选真正耗尽后才
能返回无安全落区；取消、内存失败和外部截止时间仍是搜索未完成。燃料余量、历史跳跃次数和累计
等效推进剂消耗都不能产生不可达或 episode 终止。

正式能力 YAML 的 schema、批准值或摘要错误必须在创建训练 worker、CUDA 上下文和 artifact 前
fail closed。这里拒绝的是能力文件篡改，而不是等待另一资料提供方。

## 7. 验证标准

- 正式能力适配器读取当前 YAML，三平台 typed capability 与规划器资格基准逐字段一致；
- 修改任一正式能力值但不更新版本/摘要时，正式入口在副作用前拒绝；
- 正式性能工具无需外部 capability lock，并把当前 capability v2 摘要写入报告；
- 两次及更多连续飞跃规划使用相同的固定单跳能力，均不携带上一跳的剩余燃料；
- 缺失 `HopperPropellantState` 时飞跃规划正常工作，ROS 节点也不订阅或等待该 Topic；
- `HopSegment` 对外只携带单跳 required/available delta-v，不携带消耗量或剩余量；
- 飞跃式策略观测、奖励、checkpoint 和 episode 终止原因均不存在累计燃料状态；
- 质量、比冲或参考标定值改变时，单跳可达性边界按新的正式能力版本变化；
- 当前路径规划完整搜索、三平台正反例、30 m/360° 观测闭环和性能门全部回归通过；
- 外部 RViz 连续选择两个飞跃目标时，第二跳不因第一跳的诊断等效推进剂而失败。

## 8. 非目标

- 不模拟真实推进系统的油箱、补给、热状态或姿态控制；
- 不训练燃料最优策略；
- 不为飞控生成真实推力时序；
- 不改变三平台已批准尺寸、速度、坡度、越障、落区或传感器数值；
- 不恢复旧 Volume 3 路径算法、平台代理或固定搜索资源上限。
