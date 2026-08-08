# 传感器观测能力与探索闭环 Ubuntu 资格报告

日期：2026-08-08

## 结论

批准的系统级保守观测能力 `30 m / 360°` 已接入探索候选、真实观测揭示、宏步状态更新、
三平台动作语义和正式训练前置门。当前功能分支在 Ubuntu amd64 Release 环境通过了跨层
正确性、ROS/C++、性能和仓库边界回归，可以合入 `integration` 继续准备训练。

本报告不表示正式 PPO 训练已经开始。正式运动能力 closure 已由仓库内批准的三平台
capability v2 闭合；`30 m/360°` 观测语义和 Release 性能门也已闭合。正式训练不再等待
外部 `lunar-training-capability-freeze/v1`，也不把 URDF/mesh 资源包当作运动能力的重复证明。
当前正式 `sensor-observation-performance/v1` 报告已经生成并通过；正式 cache、统一环境
builder、calibrate/train/resume/evaluate 和 formal preflight 也已闭合。训练前状态见
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

- 分支：`feature/formal-training-environment-closure`；
- 当前正式性能证据所绑定传感/训练源码提交：
  `e7c0c3b0c1563c99ec8d6b59454c42aa8e58aa98`；
- OS：Ubuntu 22.04.5 LTS，Linux `6.8.0-124-generic`，`x86_64`；
- ROS：ROS 2 Humble；
- 编译器：GCC 11.4.0，CMake 3.22.1，Python 3.10.12；
- CPU：Intel Core i7-14700KF，20 核、28 线程；内存 62 GiB；
- GPU：NVIDIA GeForce RTX 4080 SUPER，16,376 MiB，驱动 595.84。

可见性和规划基准均为 CPU 测试；GPU 仅用于记录训练主机身份。

## 正确性与构建结果

本轮无燃料状态闭包使用的仓库外 Release 安装目录为：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-install
```

资格结果：

- Release 构建 7 个 ROS 包成功，包括两组消息、配置、core、Nav2 adapter、training bridge 和
  ROS server；`colcon test-result` 为 302 tests、0 errors、0 failures、0 skipped；
- `model_contract/tests` 与完整训练测试：690 passed、1 skipped；
- 规划、ROS、差分和性能仓库级回归：179 passed、2 skipped；
- 仓库边界检查：`repository boundaries: OK`，专项测试 14 passed；
- 能力 schema v2 冻结校验：通过，规范化 SHA-256 与本报告记录一致；
- UTF-8 读取和 `git diff --check integration...HEAD`：通过。

跳过项是需要显式设备或资格开关的测试，不被当成通过项；24-worker 资格用例已另行显式执行。

## Release 性能证据

正式报告和原生 runner：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/e7c0c3b/sensor-performance.json
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-install/lib/lunar_planner_training_bridge/lunar_training_visibility_benchmark
```

固定 50 次预热、200 次测量的结果：

| 工作负载 | fixture | p50 | p95 | 门限 | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| 候选信息增益 | 256×256，4.0 m，30 m，64 candidates | 0.119669 ms | 0.132299 ms | 5 ms | 通过 |
| 实际观测 reveal | 320×320，0.2 m，30 m | 1.212102 ms | 1.381889 ms | 2 ms | 通过 |

原生程序自报 `build_type=Release`、GCC 11.4.0 和 schema
`native-visibility-benchmark/v1`。此前稳定性资格还连续执行了 8 轮相同 50/200 fixture，
reveal p95 范围为 1.289822–1.555860 ms，8/8 均低于门限。

24-worker 回归使用当前 `PlannerBridge` 和当前 C++ v3 规划器，三平台各 8 worker；全局图
为通用多分辨率 `256×256 @ 4.0 m`，局部图为 `320×320 @ 0.2 m`。每个 worker 的规划输入
通过 `Frozen*Capability.to_bridge_capability()` 注入 schema-valid v2 能力。结果为：

| 指标 | 结果 |
| --- | ---: |
| 禁用观测闭环 | 15.471201 step/s |
| 启用观测闭环 | 14.403742 step/s |
| 吞吐降幅 | 6.899653% |
| 门限 | <= 10% |

该 24-worker 用例直接通过项目正式能力适配器加载当前 capability v2；工作负载标识为
`cpp-v3-current-capability-v2`。报告内摘要为
`95ba584aa18b9f57ec087f295215ecc6c185dd0a3ad725ebb27ecbb525a9a1d4`，外部 JSON 文件
SHA-256 为 `c919f9beb2c9d255d1c599b134561f69c9f485f520009e3a8593a655d3f952ff`，结果
`passed: true`。

## 正式训练边界

运动能力、观测语义、性能门和正式数据/cache/环境/CLI 链均已闭合。当前状态是
`formal-training-ready / training-not-started`；本报告不授权或声称 seed 4080 已启动。
ONNX、TensorRT、AGX 和实际平台验收仍属于后续独立门。

正式性能命令为：

```bash
source /opt/ros/humble/setup.bash
source /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-install/setup.bash
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"

/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  training/tools/benchmark_sensor_observation.py \
  --output /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/e7c0c3b/sensor-performance.json \
  --native-benchmark /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-install/lib/lunar_planner_training_bridge/lunar_training_visibility_benchmark \
  --workers 24
```

正式 `train/resume/evaluate` 使用该外部 JSON 路径进行性能预检，同时自行加载仓库内正式
capability v2。本轮没有启动正式 PPO 训练。

## 外部 Isaac/ROS/RViz 桥同步

独立外部仓
`/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression` 已将同步结果快进合入
`main`，当前提交为 `649818b`。该主线不再发布
`/platform/hopper_propellant_state`，不含燃料提交 API 或燃料提交失败状态；外部 hopper
capability 使用 `20.0 kg` 参考总质量和 `0.2 kg` 参考推进剂计算每次相同的单跳包线。

跨仓 Release 构建使用本报告中的主仓 install，结果为：

- 外部源码 pytest：608 passed，12 skipped；
- 2 个外部包构建成功，colcon 结果 513 tests、0 errors、0 failures、2 skipped；
- 50×50 m 合成图真实 ROS 进程回归：1 passed in 361.24 s；
- 飞跃式目标 A 和目标 B 均完成抛物线与落地稳定；两次 `active_plan_id`、
  `active_segment_id` 均不同，目标 A 之后无陈旧 `CANCELED`，ROS graph 中不存在推进剂 Topic。

外部仓保持独立 Git 根，本轮未将其源码、历史或运行 artifact 导入主仓；只完成本地 `main`
快进合并，未向远端推送。
