# 三平台正式能力实现 Ubuntu 资格报告（历史燃料链已取代）

> **2026-08-08 修订：** 本报告的三平台几何、运动能力、落区、抛物线、飞行管和性能证据继续
> 有效；实时剩余燃料输入、认证燃料/剩余燃料输出及其连续递减结论已经失效。现行实现不接收
> `HopperPropellantState`，只输出每次可重复的 required/available Δv。当前资格结果见
> `sensor-observation-capability-qualification.md`，现行训练启动状态见
> `2026-08-08-formal-training-preflight-review.md`。

日期：2026-08-07

## 结论与适用边界

轮式、足式和飞跃式平台的批准工程基线已接入规划核心、ROS 2 Action 输出和 RViz 证据层，
并在 Ubuntu amd64 Release 环境完成自动化资格验证。三平台能力冻结、输入快照、规划约束、
正反例结果、单次飞跃 required/available Δv、弹道与飞行管证据和性能门限均由测试约束。

本报告只证明当前工程基线在 Ubuntu 仿真规划栈中的实现一致性，不代表：

- Isaac Sim 刚体、接触、轮胎、足端或推进器动力学验收；
- 实际轮式底盘、Yobotics Quad48 或飞跃平台的现场能力验收；
- 正式 PPO 训练完成或旧 checkpoint 与新能力兼容；
- Jetson AGX Orin 原生构建、性能、功耗或稳定性验收。

因此当前状态是“批准工程基线已实现并通过 Ubuntu 自动化资格”，不是物理平台的最终能力
定型。正式实测值后续只能通过新的能力冻结版本原子替换，不能静默修改本基线。

## 能力权威与固定语义

能力 schema 为
`ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml`，批准冻结为
`ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml`，规范化
SHA-256 为：

```text
60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95
```

测试专用能力文件继续声明 `authority: test-only/non-authoritative` 和
`runtime_eligible: false`，不能作为运行时能力来源。

本次实现固定以下核心语义：

- 轮式平台按 `1.182×0.818×1.29996 m` 机体、`0.319 m` 轮外径、`0.148 m` 单轮宽度、
  `0.8175 m` 轴距、`0.210 m` 最小离地间隙、`20°` 爬坡、`0.20 m` 越障和
  `0.20 m` 障碍净空规划；转向、倒车、原地转向和停止换向使用显式原语。
- 足式平台按 Quad48 `0.68×0.33×0.35 m`、`15.89 kg`、`10 kg` 最大载荷、`30°`
  爬坡、`0.50 m` 越障、`0.30 m` 跨沟和 `0.30 m` 最小身体净空规划。越障是连续身体
  规划中的可通行能力，不另建高层特殊技能；粗糙度只作为诊断和代价证据，不因其本身阻断
  足式平台。
- 飞跃平台只规划一次从静止起飞到精确目标的弹道，不构建多跳路线，也不替换目标。
  冻结能力中的参考总质量、参考推进剂量、`301 s` 比冲和月球重力只用于计算每次相同、
  不递减的单跳 Δv 包络；不存在上游运行时燃料输入或跨 Goal 库存。落点必须拥有完整
  `0.65 m` 横向安全支撑半径，飞行管半径为 `0.75 m`，输出一个正面积安全凸区域、名义
  抛物线以及 required/available Δv 证据。
- 三类有限地图搜索不以固定展开数、候选数或 Open 表数量作为“不可行”判据。性能门限仅
  用于 Release 回归报警，不是生产搜索终止条件。

## 验证主机

- OS：Ubuntu 22.04.5 LTS，Linux `6.8.0-124-generic`，`x86_64`；
- ROS：ROS 2 Humble；
- 编译器：GCC 11.4.0，CMake 3.22.1，Python 3.10.12；
- CPU：Intel Core i7-14700KF，20 核、28 线程；
- 内存：62 GiB；
- GPU：NVIDIA GeForce RTX 4080 SUPER，16,376 MiB，驱动 595.84。

基准为 CPU 规划测试；记录 GPU 仅用于复现整机环境，不表示规划器使用 GPU。

## Release 性能资格（2026-08-08 现行 4 m 修订）

基准 schema 为 `lunar-three-platform-qualification/v1`。每个用例预热 1 次、测量 20 次，
并要求规划结果、原因码、模式、路线哈希、展开计数和峰值计数在重复运行间一致。L0 基础
分辨率为 `0.20 m`；现行通用尺度因子为 `[1,2,4,8,16,20]`，并选择每轴不超过目标
`256` 格的最细层。千米用例因此使用 `250×250 @ 4.0 m` 的 L5 全局图，同时保留
L0 `0.20 m` 局部图。该选择来自统一公式，不是千米范围特判。

足式 50 m 和两个千米用例的 `3.0 s` 门限由首个验证通过的 Release 基线冻结；其他门限
来自批准实施计划。所有门限都只用于自动化回归。

| 用例 | 全局地图 | p50 | p95 | max | P95 门限 | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `wheel_50m_l0` | 50×50 m，0.20 m，L0 | 0.444684 s | 0.454381 s | 0.454786 s | 2.0 s | 通过 |
| `wheel_local_frontier_stress` | 50×50 m，0.20 m，L0 | 0.435236 s | 0.436207 s | 0.437474 s | 2.0 s | 通过 |
| `legged_50m_l0` | 50×50 m，0.20 m，L0 | 0.948130 s | 0.964901 s | 0.973578 s | 3.0 s | 通过 |
| `wheel_1km_l5` | 1000×1000 m，4.00 m，L5 | 0.194370 s | 0.195302 s | 0.196432 s | 3.0 s | 通过 |
| `legged_1km_l5` | 1000×1000 m，4.00 m，L5 | 1.125778 s | 1.164988 s | 1.173191 s | 3.0 s | 通过 |
| `hopper_direct_100m` | 112×12 m，0.80 m，L2 | 0.003447 s | 0.003482 s | 0.003496 s | 1.0 s | 通过 |
| `hopper_alternate_time_100m` | 112×12 m，0.80 m，L2 | 0.012670 s | 0.012820 s | 0.012953 s | 2.0 s | 通过 |
| `hopper_complete_blocked_100m` | 112×12 m，0.80 m，L2 | 0.001830 s | 0.001864 s | 0.001881 s | 5.0 s | 通过 |

开阔 100 m 用例产生 `11.111111013 s` 的认证弹道；中部 `35 m` 障碍使规划器选择更高的
`13.304123168 s` 弹道；`100 m` 高的完整阻断柱返回
`HOPPER_ALL_FLIGHT_TUBES_BLOCKED`，不产生引用。完整 JSON 同时记录各阶段耗时、展开数、
Open 峰值、工作内存、进程峰值 RSS、认证尝试数、模式和原因码。

权威性能证据位于仓库外：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/360fe26/evidence/three-platform-release-benchmark.json
SHA-256 9887c4e5cdf774d720651b1570949abba73578ae9bb3d01d91169c99787b87e9
```

## 自动化结果

- 外部干净目录 Release 构建：7 个包完成；
- `colcon test-result --verbose`：302 tests，0 errors，0 failures，0 skipped；
- `tests/foundation tests/ros tests/differential tests/performance`：179 passed、2 skipped；
- 能力冻结 schema 与规范化 SHA-256 检查：通过；
- ROS 外部接口实际安装前缀检查：通过；
- 仓库边界检查及 13 项专项测试：通过；
- `git diff --check`：通过。

差分基线保留旧规划器的可达、安全、目标和执行语义，但成本使用当前分层规划含义：足式成本
反映 `0.20 m` 身体原语，飞跃成本为带 10% 裕度的 `required_delta_v_mps`。若全局路线存在而
最后的平台连接不可行，有限的全局路线成本只作为诊断，不表示目标可达。

## 复现命令

所有构建、安装、日志和性能输出均写入仓库外：

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble

colcon --log-base /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-log build \
  --merge-install \
  --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-install \
  --packages-select lunar_planning_msgs lunar_navigation_msgs lunar_navigation_config \
    lunar_planner_core lunar_nav2_adapter lunar_planner_ros lunar_planner_training_bridge \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON

source /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-install/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-test-log test \
  --merge-install \
  --build-base /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-install \
  --packages-select lunar_planning_msgs lunar_navigation_msgs lunar_navigation_config \
    lunar_planner_core lunar_nav2_adapter lunar_planner_ros lunar_planner_training_bridge
colcon test-result \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-build \
  --verbose
```

运行仓库级 pytest 前设置三个 runner：

```bash
export LUNAR_PLANNER_FACADE_SUMMARY=/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-build/lunar_planner_core/lunar_planner_facade_summary
export LUNAR_HIERARCHICAL_PLANNER_BENCHMARK=/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-build/lunar_planner_core/lunar_hierarchical_planner_benchmark
export LUNAR_PLANNER_CORE_BENCHMARK=/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/34679db/native-build/lunar_planner_core/lunar_planner_core_benchmark
python3 -m pytest -q tests/foundation tests/ros tests/differential tests/performance
```

本机证据目录不是部署输入，也不得提交到仓库。AGX 和 Isaac Sim 的后续验收必须另外生成
各自环境的证据，不得覆写本报告的 Ubuntu 结论。
