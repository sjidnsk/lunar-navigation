# 传感器观测能力与探索闭环 Ubuntu 资格报告

日期：2026-08-08

## 结论

批准的系统级保守观测能力 `30 m / 360°` 已接入探索候选、真实观测揭示、宏步状态更新、
三平台动作语义和正式训练前置门。当前功能分支在 Ubuntu amd64 Release 环境通过了跨层
正确性、ROS/C++、性能和仓库边界回归，可以合入 `integration` 继续准备训练。

本报告不表示正式 PPO 训练已经开始。正式运动能力 closure 已由仓库内批准的三平台
capability v2 闭合；`30 m/360°` 观测语义和 Release 性能门也已闭合。正式训练不再等待
外部 `lunar-training-capability-freeze/v1`，也不把 URDF/mesh 资源包当作运动能力的重复证明。
当前正式 `sensor-observation-performance/v1` 报告已经生成并通过，训练代码已具备进入下一项
独立数据/运行环境门的资格，但尚不具备启动 formal rollout 的完整公开命令链。正式 cache、
环境 builder、calibrate/train/resume/evaluate 闭环的复审结论见
`docs/validation/2026-08-08-formal-training-preflight-review.md`。

## 权威基线和历史边界

本次资格只认以下现行基线：

- 路径规划：`integration` 上当前 C++ v3 分层全局规划、有限地图完整搜索、平台局部物理
  认证、滚动衔接和当前 ROS Action 行为；本功能分支的共同基点为
  `b33d437cec5a58d4dfce215a120d45eef6a638b8`。
- 平台运动能力：能力 schema v2 和
  `ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml`，规范化
  SHA-256 为 `60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`。
- 传感器观测：三平台共享 `sensor_range_m=30.0`、`sensor_fov_deg=360.0`；训练语义版本为
  `lunar-training-semantics/sensor-30m-360-theta-mask/v1`，SHA-256 为
  `739ce3e1f6eaab4ee44a0136ff848f246d158a15ebb7f8b6c8b2c7ec1bca8cb5`。

从旧 Volume 3 合入的 policy、数据、PPO、checkpoint 和训练编排只是可继续使用的软件骨架。
其中旧路径算法、旧平台代理/能力字段、旧 search limit、旧观测默认值和由它们生成的 cache、
checkpoint 或性能结论均不再具有现行权威。训练环境目录名中出现 `volume3` 也只表示 Python
依赖环境的历史位置，不改变上述权威关系。

Isaac/ROS 外部仓的 `30 m/120°` `capability_provenance.json` 继续冻结为历史验证证据，不能
重新定基线，也不能转换成 30 m/360° 正式能力。正式入口已取消可替换权威的
`--capability-lock`；它们只读取并验证上述仓库内批准能力。

## 已实现语义

- `TruthWorld` 与策略可见的 `ObservedWorld` 使用不同类型；候选构造、策略和奖励不能读取
  truth。只有到达观测边界后执行的真实 reveal 会更新 `valid_mask`、
  `observation_age_s`、`observation_quality` 和 `observation_count`。
- 30 m 只限制 LOS 增益估算和实际 reveal，不会把半径内栅格自动标成已知。任意正的物理
  障碍比例阻挡后方视线，首个阻挡栅格仍可见，`forbidden` 本身不阻挡 LOS。
- C++ Release 可见性核一次批量处理全部候选，并保持确定性 Bresenham 语义；360° 使用与
  yaw 无关的快速路径。
- 探索宏步严格执行“选择候选 -> 当前规划器规划 -> 外部执行反馈到达边界 -> reveal ->
  构建下一观测/奖励”。同一请求、同一世界和同一能力贯穿该边界。
- `theta` 表示到达候选后的绝对 `map` 机体 yaw，不是传感器视线。轮式和足式保留
  `theta` 的运动学学习信号；飞跃式在当前项目中只选择安全且简单抛物线可达的落区，
  `theta` 被 mask，且不进入 PPO loss。
- 飞跃式每次动作都使用 capability v2 的固定参考质量、比冲和参考推进剂计算同一个单跳
  `delta-v` 包线。该参考推进剂是标定条件，不是 episode 状态；规划、观测、奖励和终止条件
  均不累计燃料消耗，连续规划不会因前一跳而缩小可达范围。
- 正式 `train`、`resume` 和 `evaluate` 在创建 worker、CUDA 上下文或输出 artifact 前，
  必须同时校验项目正式能力摘要、训练语义和同主机/同源码 Release 性能报告。

## 验证主机和源码身份

- 分支：`feature/sensor-observation-capability-design`；
- 当前正式性能证据所绑定源码提交：`44df8814e6e4d29dca41134354f86bc1ac3577af`；
- OS：Ubuntu 22.04.5 LTS，Linux `6.8.0-124-generic`，`x86_64`；
- ROS：ROS 2 Humble；
- 编译器：GCC 11.4.0，CMake 3.22.1，Python 3.10.12；
- CPU：Intel Core i7-14700KF，20 核、28 线程；内存 62 GiB；
- GPU：NVIDIA GeForce RTX 4080 SUPER，16,376 MiB，驱动 595.84。

可见性和规划基准均为 CPU 测试；GPU 仅用于记录训练主机身份。

## 正确性与构建结果

本轮无燃料状态闭包使用的仓库外 Release 安装目录为：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/final-33df16e/install
```

资格结果：

- Release 构建 8 个 ROS 包成功，包括两组消息、配置、core、model contract、Nav2 adapter、
  training bridge 和 ROS server；
- core 聚焦回归：24 tests，0 errors，0 failures；
- message/core/bridge/ROS 干净结果：36 tests，0 errors，0 failures；
- `model_contract/tests`、完整训练测试、差分和性能测试：645 passed，16 skipped；
- 原生/参考可见性随机小图等价回归：9 passed；
- 性能门、sensor 和 CLI 聚焦回归：57 passed；
- 仓库边界检查：`repository boundaries: OK`，专项测试 14 passed；
- 能力 schema v2 冻结校验：通过，规范化 SHA-256 与本报告记录一致；
- UTF-8 读取和 `git diff --check integration...HEAD`：通过。

跳过项是需要显式设备或资格开关的测试，不被当成通过项；24-worker 资格用例已另行显式
打开并执行。

## Release 性能证据

正式报告和原生 runner：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/sensor-performance.json
/home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install/lib/lunar_planner_training_bridge/lunar_training_visibility_benchmark
```

固定 50 次预热、200 次测量的结果：

| 工作负载 | fixture | p50 | p95 | 门限 | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| 候选信息增益 | 256×256，4.0 m，30 m，64 candidates | 0.118651 ms | 0.128879 ms | 5 ms | 通过 |
| 实际观测 reveal | 320×320，0.2 m，30 m | 1.079292 ms | 1.206533 ms | 2 ms | 通过 |

原生程序自报 `build_type=Release`、GCC 11.4.0 和 schema
`native-visibility-benchmark/v1`。此前稳定性资格还连续执行了 8 轮相同 50/200 fixture，
reveal p95 范围为 1.289822–1.555860 ms，8/8 均低于门限。

24-worker 回归使用当前 `PlannerBridge` 和当前 C++ v3 规划器，三平台各 8 worker；全局图
为通用多分辨率 `640×640 @ 1.6 m`，局部图为 `320×320 @ 0.2 m`。每个 worker 的规划输入
通过 `Frozen*Capability.to_bridge_capability()` 注入 schema-valid v2 能力。结果为：

| 指标 | 结果 |
| --- | ---: |
| 禁用观测闭环 | 14.307369 step/s |
| 启用观测闭环 | 14.055330 step/s |
| 吞吐降幅 | 1.761604% |
| 门限 | <= 10% |

该 24-worker 用例直接通过项目正式能力适配器加载当前 capability v2；工作负载标识为
`cpp-v3-current-capability-v2`。报告内摘要为
`22e258b2f8d4da517acbcc414e5eb3d5ea9b4b3a57b0856f0fd39e58ed744f45`，外部 JSON 文件
SHA-256 为 `5d1cf5b390cb1d5e3985d56cfbf18ff61c260c8af6934fff3ba55db74426e690`，结果
`passed: true`。

## 正式训练后续边界

运动能力、观测语义和性能门已经闭合；正式数据/cache/环境/CLI 链尚未闭合。后续按独立流程完成：

1. 实现并生成项目正式三平台 traversability/candidate cache，绑定正式能力、数据、训练语义和
   当前规划器身份；旧 Volume 3 cache/checkpoint 不得续用。
2. 闭合公开 `calibrate/train/resume/evaluate` 的正式环境 builder 与观测模板注入。
3. 使用本报告的正式性能 JSON 执行进程级 formal preflight，通过后冻结 worker/micro-batch 并
   启动 seed 4080。
4. ONNX、TensorRT、AGX 和实际
   平台验收仍属于后续独立门。

正式性能命令为：

```bash
source /opt/ros/humble/setup.bash
source /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install/setup.bash
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"

/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  training/tools/benchmark_sensor_observation.py \
  --output /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/sensor-performance.json \
  --native-benchmark /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install/lib/lunar_planner_training_bridge/lunar_training_visibility_benchmark \
  --workers 24
```

正式 `train/resume/evaluate` 使用该外部 JSON 路径进行性能预检，同时自行加载仓库内正式
capability v2。本轮完成的是训练资格闭包，未启动正式 PPO 训练。

## 外部 Isaac/ROS/RViz 桥同步

独立外部仓
`/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression` 使用分支
`feature/hopper-no-fuel-budget-sync` 完成同步，当前提交为 `dbfdb43`。该分支不再发布
`/platform/hopper_propellant_state`，不含燃料提交 API 或燃料提交失败状态；外部 hopper
capability 使用 `20.0 kg` 参考总质量和 `0.2 kg` 参考推进剂计算每次相同的单跳包线。

跨仓 Release 构建使用本报告中的主仓 install，结果为：

- 外部源码 pytest：608 passed，12 skipped；
- 2 个外部包构建成功，colcon 结果 513 tests、0 errors、0 failures、2 skipped；
- 50×50 m 合成图真实 ROS 进程回归：1 passed in 361.24 s；
- 飞跃式目标 A 和目标 B 均完成抛物线与落地稳定；两次 `active_plan_id`、
  `active_segment_id` 均不同，目标 A 之后无陈旧 `CANCELED`，ROS graph 中不存在推进剂 Topic。

外部仓保持独立 Git 根，本轮未将其源码、历史或运行 artifact 导入主仓，也未自动合并或推送。
