# 分层全局规划 Ubuntu 自动化验证

日期：2026-08-05

结论：通用 L0–L4 多分辨率地图、三平台分层全局规划、局部物理认证、ROS 诊断与 RViz
六层证据已完成 Ubuntu 自动化验证。Jetson AGX Orin 性能和现场 Isaac Sim/RViz 操作尚未
验证，不得由本报告替代。

## 1. 版本与主机基线

- 主仓分支：`hierarchical-global-planning`
- 主仓已验证实现基线：`9fd1862cff6f42496b76dcee99dbbeb8577637d5`
- 外部地图金字塔提交：`c4e04f2e019ae4b7d59dfcab8eb164f4beef0a7a`
- 外部 RViz 分层证据提交：`467d25201253bf95a591df851a5afc84063fa1ca`
- OS：Ubuntu 22.04，内核 `6.8.0-124-generic`，架构 `x86_64`
- ROS：ROS 2 Humble
- CPU：Intel Core i7-14700KF，20 核、28 线程
- GPU：NVIDIA GeForce RTX 4080 SUPER，16,376 MiB，驱动 `595.84`
- 内存：62 GiB

本报告所在提交包含最终资格测试和证据文档；其提交 ID 由 Git 历史确定，不能在同一提交
内容中自引用。

## 2. 固定回归覆盖

`lunar_planner_core_hierarchical_regression_test` 使用固定坐标，不在运行时扫描或重采样终点。
7 项测试覆盖：

- 远距离轮式和腿式成功，以及完整全局墙阻断；
- 同一坡面上轮式失败、腿式通过；
- 跳跃式三跳名义链、断链、分辨率不足和图资源不足；
- 全局成功但 L0 局部覆盖不足；
- 固定 `map_from_odom` 平移和旋转；
- L4 仍不可表达的地图规模、预取消；
- 相同输入的结果、路线、局部轨迹、代价、警告和搜索计数确定性。

结果：7 项通过，0 失败。

## 3. 性能基准

基准 schema 为 `lunar-hierarchical-benchmark/v1`。每个规模档位覆盖 `open`、
`fixed-obstacle`、`narrow-channel` 和 `no-route`，预热后各测量 30 次。`global_p95_s`
来自生产规划诊断的 `hierarchical.global_elapsed`；展开状态、Open 峰值和内存来自规划器搜索
计数，不使用进程 RSS 推算。

以下为各档位四类 fixture 中最慢的 p95；三档最慢项均为 `fixed-obstacle`：

| 栅格数 | 全局 p95 | Ubuntu 门槛 | 完整核心 p95 | Ubuntu 门槛 | 最大规划器工作内存 |
|---:|---:|---:|---:|---:|---:|
| 65,536 | 0.078916 s | 0.5 s | 0.088118 s | 2.0 s | 1,164,032 B |
| 262,144 | 0.327965 s | 1.0 s | 0.336826 s | 3.0 s | 4,557,848 B |
| 1,048,576 | 1.364447 s | 2.0 s | 1.374577 s | 4.0 s | 18,027,952 B |

12 组结果均通过 Ubuntu 门槛且 30 次路线哈希稳定。AGX 档位只写出设备门槛，不在 Ubuntu
上求值：全局 p95 为 `1/2/4 s`，完整核心 p95 为 `3/4/6 s`。

完整 JSON 位于仓库外：

```text
/home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/benchmarks/ubuntu-hierarchical-benchmark.json
SHA-256 db64977089262ee3f0dde9b49c32f981844740c9f128324a593bddda31dd641e
```

权威性能构建显式使用 `-DCMAKE_BUILD_TYPE=Release`。一次无优化的空 `CMAKE_BUILD_TYPE`
试跑在发现配置不适合作为性能证据后终止，未计入上述结果。

## 4. PPO 行为兼容性

结论：`retraining_required`，旧 checkpoint 策略为 `do_not_relabel`。

资格时并行 PPO 分支 `f8cea6a8fa76e4d0c9e6c050f7b7f04da14425af` 的训练桥仍构造旧
`PlannerInput.goal`，并把规划器身份覆盖为 `cpp_v3`；当前分层实现要求 `goal_map` 和
`cpp_v3_hierarchical`。三个安全黄金案例的规划代价也已改变，宏步规划输入不是精确等价。
当前分支不包含 PPO 策略动作黄金 fixture，因此不能证明观测和动作逐项完全一致；兼容性门
按失败关闭，不把旧 checkpoint 宣称为分层规划兼容。冻结结论记录在
`tests/differential/ppo_behavior_compatibility.json`。

## 5. 自动化结果

主仓：

- Release 全工作区构建：6 个包通过；
- `colcon test-result --verbose`：29 tests，0 errors，0 failures，0 skipped；
- `tests/foundation tests/differential tests/performance`：141 passed，13 skipped；
  其中 runner 依赖测试已由 CTest 注入真实可执行文件执行；
- 仓库边界专项：12 passed；
- 外部接口检查：通过；
- `git diff --check`：通过。

外部 ROS/RViz 仓：

- 源码 pytest：508 passed，2 skipped；
- `lunar_isaac_validation` 与 `lunar_isaac_rviz_plugins`：416 tests，0 errors，
  0 failures，2 skipped。

外部仓已有 `build/`、`install/`、`log/` 历史树，因此源码 pytest 使用
`--ignore=install --ignore=build --ignore=log`，避免重复收集过期安装副本。

## 6. 权威命令

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/log build \
  --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/install/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/log test \
  --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/install \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/test-results
colcon test-result \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/test-results \
  --verbose
python3 -m pytest -q tests/foundation tests/differential tests/performance
python3 tools/check_external_interfaces.py \
  --config ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml \
  --expected-lunar-navigation-prefix \
  /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/install/lunar_navigation_msgs
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git diff --check
```

`lunar_navigation_msgs` 使用 colcon 隔离安装，因此检查器参数必须是实际包前缀
`.../install/lunar_navigation_msgs`，不能传安装根目录。

## 7. 未验证边界

| 项目 | 状态 | 说明 |
|---|---|---|
| AGX performance | 未验证 | 必须在 Jetson AGX Orin 原生 Release 构建上执行设备档位基准 |
| live Isaac Sim/RViz operator run | 未验证 | 自动化已验证 ROS/RViz 逻辑，不代表现场图形会话验收 |
