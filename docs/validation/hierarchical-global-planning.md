# 分层全局规划 Ubuntu 自动化验证

日期：2026-08-05

结论：通用 L0–L4 多分辨率地图、三平台分层全局规划、局部物理认证、ROS 诊断与 RViz
六层证据已完成 Ubuntu 自动化验证。Jetson AGX Orin 性能和现场 Isaac Sim/RViz 操作尚未
验证，不得由本报告替代。

## 1. 版本与主机基线

- 主仓分支：`hierarchical-global-planning`
- 主仓已验证实现基线：`b691a15`（有界平滑与性能资格提交在其后）
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
10 项测试覆盖：

- 远距离轮式和腿式成功，以及完整全局墙阻断；
- 同一坡面上轮式失败、腿式通过；
- 跳跃式三跳名义链、断链、分辨率不足和图资源不足；
- 全局成功但 L0 局部覆盖不足；
- 固定 `map_from_odom` 平移和旋转；
- L4 仍不可表达的地图规模、预取消；
- 相同输入的结果、路线、局部轨迹、代价、警告和搜索计数确定性。
- 50×50 m、0.2 m 全局图上三类平台的远目标正例；
- 同一 50 m 固定图上的全局墙、跳跃式落区面积不足和图资源耗尽；
- 轮式/足式真实起点连接碰撞，以及优化失败时的认证离散回退和强制平滑失败。

结果：10 项通过，0 失败。

## 3. 性能基准

当前基准 schema 为 `lunar-hierarchical-benchmark/v2`。五个 fixture 都固定为
50×50 m、0.2 m、250×250 栅格，预热后测量 30 次；起点、终点、障碍和安全落区均为字面
坐标，运行时不扫描或重采样目标。能力文档位于 `tests/fixtures/capabilities/test-only/`，统一
声明 `authority: test-only/non-authoritative` 和 `runtime_eligible: false`，不得替换生产能力。

Ubuntu amd64 Release 结果：

| fixture | 完整核心 p50 | 完整核心 p95 | 门槛 | 展开节点 | 候选边 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| wheel_positive | 0.080197 s | 0.081283 s | 2.0 s | 39,257 | 0 | 通过 |
| legged_positive | 0.126760 s | 0.129120 s | 2.0 s | 34,157 | 0 | 通过 |
| hopper_direct_positive | 0.658525 s | 0.669625 s | 1.0 s | 1 | 1 | 通过 |
| hopper_multihop_positive | 0.881612 s | 0.906888 s | 5.0 s | 5 | 5 | 通过 |
| hopper_complete_negative | 0.073390 s | 0.079129 s | 5.0 s | 740 | 32 | 通过 |

所有用例的 30 次路线哈希、结果类别、原因码和诊断计数一致。报告同时记录完整核心以及
`global_search`、`local_planning`、`landing_field`、`spatial_index`、`ballistic_solve`、
`flight_tube_certification` 各阶段的 p50/p95/最大值，并分别保存规划器估算工作内存和 Linux
进程峰值 RSS。诊断计数只在搜索/认证之后累加和上报，不参与终止判断或候选裁剪。

完整 JSON 位于仓库外：

```text
/home/kai/CodexDownloads/lunar_navigation/task9-rolling-NVeOXrwF/task10-release-benchmark.json
SHA-256 fec795506a0be0170e572d2521d23893d30877446828c1363fdd7000add56a9b
```

权威性能构建显式使用 `-DCMAKE_BUILD_TYPE=Release`。一次无优化的空 `CMAKE_BUILD_TYPE`
试跑在发现配置不适合作为性能证据后终止，未计入上述结果。

标准 `/diagnostics` 及 RViz 交互状态面板同时镜像全局/局部耗时、落区场、空间索引、弹道
求解、飞行管认证、展开/Open 峰值、安全落点、候选/粗筛/完整认证/失效/缓存计数、路线复用
与游标、滚动请求计数，以及活动 plan/segment 和执行状态。旧的
`hierarchical_*` 键继续保留，避免破坏已有观察工具。

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

## 7. 完整搜索与滚动规划最终资格（2026-08-06）

本节覆盖取消固定全局搜索截断、三平台滚动协调和 RViz 完成态显示后的最终增量资格；前面
2026-08-05 的结果保留为历史基线。实现提交为：

- 主仓实现：`73128ecff737b362243067a9ecfd3ffc13059a83`；
- 外部 RViz/Action 实现：`a14745b580c285ee0d878019d0288848b9137414`；
- 外部操作文档：`f7dc9dfb63932707c07d746d0df375d2e223c6d9`。

验证记录所在的主仓文档提交不能在自身内容中自引用，以 Git 历史为准。生产能力 YAML 和三类
平台运动能力数值未改；`maximum_authorized_hops=1`、Action/`MotionReference` schema 与外部
地图所有权均保持不变。

### 7.1 新鲜 Release 构建与测试

所有产物位于仓库外：

```text
/home/kai/CodexDownloads/lunar_navigation/complete-rolling-final-Nzp9RH
```

主仓从空 `build/install/test-results` 构建，`CMAKE_BUILD_TYPE=Release`：

- 6 个包构建通过；
- `colcon test-result --verbose`：34 tests，0 errors，0 failures，0 skipped；
- `tests/foundation`：127 passed；
- 外部接口和仓库边界检查通过；
- `git diff --check` 通过。

外部 overlay run id 为 `20260806T111057997490Z`，`build.json` 同时记录 `Release`、本次主仓
worktree 和独立 install 路径：

- 源码 pytest：568 passed，2 skipped；历史快照契约通过显式
  `LUNAR_INTERACTIVE_SNAPSHOT_MANIFEST` 读取仓库外保留快照；
- merged overlay CTest：502 tests，0 errors，0 failures，2 skipped；
- merged install 中仅排除不适用于混合头目录的
  `lunar_planner_core_public_header_boundary`，随后把安装出的 9 个核心头复制到隔离视图，使用
  原测试脚本验证 1/1 通过，且安装/隔离逐文件 SHA-256 完全一致。

### 7.2 三平台固定 Action 正反例与滚动完成

固定地图为 `synthetic-ad8056e0fcc3a848`：50×50 m、0.2 m/cell、250×250 栅格、seed
`20260805`。进程级测试对 wheel、legged、hopper 依次执行障碍目标负例和远目标正例，并要求
每个平台都从首个局部授权继续滚动到 `ROLLING_GOAL_COMPLETE`；最后还验证服务器已接收请求
后的取消路径。结果为 1 passed，耗时 52.68 s，日志保存在：

```text
/home/kai/CodexDownloads/lunar_navigation/complete-rolling-final-Nzp9RH/external/artifacts/20260806T111057997490Z/synthetic-process
```

地面平台的下一段只在 bridge 回显同一代 canonical odometry 后发起；取消等待中的续段会清空
待发命令，迟到回显不能启动新 Action。轮式转角局部目标使用能力运动原语推导的航向容差，旋转
矩形碰撞复检使用真实 footprint 与栅格相交，不再把包围盒角落误判为碰撞。

### 7.3 30 次 Release 性能结果

报告及 SHA-256：

```text
/home/kai/CodexDownloads/lunar_navigation/complete-rolling-final-Nzp9RH/main/benchmark/complete-rolling-release.json
69cd118cd44a83a6b79224e3b16fd08181d2b4742f7d9348904ffa1ff59bfa34
```

| fixture | p50 | p95 | 门槛 | 结果 |
|---|---:|---:|---:|---|
| wheel_positive | 0.077295 s | 0.083870 s | 2.0 s | 通过 |
| legged_positive | 0.114696 s | 0.116464 s | 2.0 s | 通过 |
| hopper_direct_positive | 0.601238 s | 0.634274 s | 1.0 s | 通过 |
| hopper_multihop_positive | 0.877334 s | 0.884558 s | 5.0 s | 通过 |
| hopper_complete_negative | 0.073642 s | 0.074597 s | 5.0 s | 通过 |

五个 fixture 均运行 30 次，路线哈希和结果确定；能力来源为
`test-only/non-authoritative`，不能替代正式平台参数。

### 7.4 真实 RViz 窗口证据

在 X11 RViz2 会话 `20260806T112050570664Z` 中，三平台都使用远目标滚动到完成态；面板同时
显示 `FEASIBLE`、`COMPLETED` 和 `ROLLING_GOAL_COMPLETE`。另分别捕获规划路线、飞跃式当前
授权弹道/飞行管，以及障碍目标 `INFEASIBLE / GOAL_OBSTACLE`。九张规范截图和逐文件
SHA-256 位于：

```text
/home/kai/CodexDownloads/lunar_navigation/complete-rolling-final-Nzp9RH/external/screenshots
```

文件为 `wheel-completed.png`、`legged-completed.png`、`hopper-completed.png`、
`wheel-route.png`、`legged-route.png`、`hopper-route-ballistic.png`，以及三份
`<platform>-negative-goal-obstacle.png`。会话通过启动器
自有 PID 正常清理；controller/RViz 日志未发现 traceback、fatal 或崩溃。该证据使用独立合成
地图，不连接 Isaac Sim，也不代表真实平台执行性能。

## 8. 未验证边界

| 项目 | 状态 | 说明 |
|---|---|---|
| AGX performance | 未验证 | 必须在 Jetson AGX Orin 原生 Release 构建上执行设备档位基准 |
| live synthetic RViz operator run | 已验证 | 覆盖三平台路线、弹道/飞行管、完成态和障碍目标反例 |
| live Isaac Sim scene replay | 未验证 | 本轮按批准范围与 Isaac Sim 无关，不能替代当前 USD 场景回放验收 |
