# Pure Planner Task 16 验证报告

日期：2026-08-23
状态：`PASS_HUMBLE_OFFLINE_WITH_NOT_RUN_BOUNDARIES`

本报告只证明 Ubuntu 22.04 amd64 + ROS 2 Humble 中的 Release 构建、离线算法、ROS
Action/graph、接口合同和仓库边界。没有真实 rosbag、在线全局图、生产 ROS domain 切换或 Jetson
AGX Orin 证据，因此不声明真实 `LAVA_TUBE`、`LUNAR_SURFACE` 端到端或设备发布就绪。

## 1. 源码与证据目录

生产源码/前置测试基线：

```text
branch: feat/pure-planner-isolated
HEAD:   e93a4685ed4cebbe06bc2983b6ea3e03159554c5
tree:   d80de1853a72ac626716acf09d93c6522210818c
```

验证分三层执行：

| 层 | 源码 | 仓外 artifact | 用途 |
| --- | --- | --- | --- |
| primary | `77a5b89008a56ce9f97e6a4afe1d7876181a1948`，tree `d5d6a96b981f10577ee391b40cb3cb2ec35bd3a5` | `/home/kai/CodexDownloads/lunar_navigation/pure_planner/task16-77a5b89008a56ce9f97e6a4afe1d7876181a1948-4GTaiX` | fresh Release 三包构建、core/ROS 全测、矩阵、重复性能、固定 domain、installed consumer 和二进制闭包 |
| final recheck | `e93a4685ed4cebbe06bc2983b6ea3e03159554c5`，tree `d80de1853a72ac626716acf09d93c6522210818c` | `/home/kai/CodexDownloads/lunar_navigation/pure_planner/task16-e93a4685ed4cebbe06bc2983b6ea3e03159554c5-recheck-4i3MEh` | 生产源码/前置测试基线的完整 foundation、pure、边界、live checker、UTF-8 和 artifact 复验 |
| Task 16 change set | 本报告所在 Task 16 change set，验证时尚未提交 | `/home/kai/CodexDownloads/lunar_navigation/pure_planner/task16-e93a4685ed4cebbe06bc2983b6ea3e03159554c5-fix1-19iwVj` | 当时实际 collect 58 项的全 `pure_planner/tests`、README/isolation、RED/GREEN、UTF-8、artifact 和 boundary 证据 |

`77a5b89..e93a468` 只修改以下两个 foundation 测试，没有修改
`pure_planner/`、`lunar_planning_msgs` 或任何生产源码：

```text
M tests/foundation/test_planner_search_semantics.py
M tests/foundation/test_planning_message_package.py
```

因此 primary 的生产构建、性能与 ROS 二进制证据适用于生产源码/前置测试基线；final recheck 又重新运行了所有
受变更影响的测试及最终证据门。primary 旧日志中的 foundation 结果不再代表当前状态；前置测试基线
的权威结果是 `148 passed`。

Task 16 change set 只修改 `pure_planner/README.md`、本报告与
`pure_planner/tests/test_launch_contract.py`；没有修改生产源码、算法、阈值或 installed
产物。因此第三层使用 primary 的只读生产 install，但从只读当前工作树采集测试和文档。
验证发生时该 change set 尚未提交，报告不循环嵌入自身提交 SHA；在提交后的 checkout 中使用
`git log -1 -- pure_planner/VERIFICATION.md` 定位实际承载本报告的提交。

## 2. 权威环境和依赖

primary、final recheck 与 Task 16 change set 都使用一次性
`osrf/ros:humble-desktop-full-jammy` 容器，镜像 digest 为
`sha256:076d67e751982ceafd25a873266a5e3e0979635ddc777f6301e4aec6f5068dca`。容器内记录为：

```text
Ubuntu 22.04
ROS_DISTRO=humble
ROS_VERSION=2
arch=x86_64
```

源码只读挂载到 `/workspace`，所有 build/install/log/test artifact 写到上述仓外目录。显式安装并记录
版本的依赖为：

```text
binutils 2.38-4ubuntu2.12
cmake 3.22.1-1ubuntu1.22.04.2
g++ 4:11.2.0-1ubuntu1
libyaml-cpp-dev 0.7.0+dfsg-8build1
nlohmann-json3-dev 3.10.5-2
python3-pytest 6.2.5-1ubuntu2
python3-yaml 5.4.1-1ubuntu1
ros-humble-grid-map-msgs 2.0.1-1jammy.20260726.110828
```

## 3. 主要命令与结果

### 3.1 Release 顺序构建和全量测试

primary 容器中的关键命令为：

```bash
source /opt/ros/humble/setup.bash
colcon --log-base /artifacts/log build \
  --executor sequential \
  --base-paths pure_planner/ros2_ws/src ros2_ws/src/lunar_planning_msgs \
  --build-base /artifacts/build --install-base /artifacts/install \
  --packages-select \
    lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

colcon --log-base /artifacts/log-test test \
  --executor sequential \
  --build-base /artifacts/build --install-base /artifacts/install \
  --packages-select lunar_pure_planner_core lunar_pure_planner_ros \
  --return-code-on-test-failure --ctest-args --output-on-failure

colcon --log-base /artifacts/log-test-result test-result \
  --test-result-base /artifacts/build --verbose
```

结果：

| 门 | 结果 | 状态 |
| --- | --- | --- |
| fresh Release sequential build | 3 packages，`39.8 s` | PASS |
| core + ROS `colcon test-result` | `283 tests, 0 errors, 0 failures, 0 skipped` | PASS |
| 全 `pure_planner/tests`，primary | `54 passed in 22.07 s` | PASS |
| 全 `pure_planner/tests`，final recheck | `54 passed in 23.57 s` | PASS |
| 全 `pure_planner/tests`，Task 16 change set | `58 passed in 23.55 s` | PASS |
| selected foundation，primary | `67 passed` | PASS |
| 两个定向 foundation，final recheck | `2 passed` | PASS |
| 全 foundation，final recheck | `148 passed in 1.15 s` | PASS |
| repository checker | `repository boundaries: OK` | PASS |
| repository boundary tests，final recheck | `14 passed` | PASS |
| Task 16 change set README + isolation contract | `33 passed in 0.05 s` | PASS |
| Task 16 change set UTF-8 / artifact / bytecode scan | `198 files` / `OK` / `OK` | PASS |

首次 fresh build 完成后，脚本在测试启动前因 `set -u` 与 Humble overlay 的
`COLCON_CURRENT_PREFIX` 初始化顺序停止；fresh build 原始结果保存在
`build-release-sequential-first-fresh.log`。临时关闭 nounset 完成 overlay source 后，测试才开始。
live checker 的首次调用还把非 merge-install 根误作 package prefix；改用实际
`/artifacts/install/lunar_planning_msgs` 后两套 checker 均通过。这些是命令初始化修正，没有修改
源码、算法、阈值或安装产物。

### 3.2 重复 wall-time

所有数字都是 Release 二进制的 GTest real wall duration；每个关键用例 fresh 重复 5 次且
`--gtest_break_on_failure`。原始记录位于 primary artifact 的 `perf-*-repeat5.log`。

| 用例 | 5 次 wall time (ms) | min / median / max | 门 | 结果 |
| --- | --- | --- | --- | --- |
| 2048×2048 Surface global projection | `149, 145, 145, 144, 145` | `144 / 145 / 149` | global real-wall `<=150 ms`；测试同时断言 `global_elapsed<=150 ms` | PASS |
| wheel 320×320 high branching | `27, 27, 27, 27, 27` | `27 / 27 / 27` | `<1000 ms` | PASS |
| wheel 320×320 narrow corridor | `66, 67, 64, 64, 64` | `64 / 64 / 67` | `<1000 ms` | PASS |
| legged 320×320 far plan | `153, 147, 145, 145, 143` | `143 / 145 / 153` | `<1000 ms` | PASS |
| hopper 320×320 direct | `49, 46, 46, 46, 46` | `46 / 46 / 49` | `<1000 ms` | PASS |
| hopper formal medium two-hop | `901, 867, 865, 869, 870` | `865 / 869 / 901` | `<1000 ms` | PASS |
| hopper unaligned landing | `729, 716, 719, 719, 721` | `716 / 719 / 729` | `<1000 ms` | PASS |
| hopper bounded 1 m goal region | `51, 50, 50, 50, 50` | `50 / 50 / 51` | `<1000 ms` | PASS |

上表证明这些离线关键算法工作负载的严格 `<1 s` 门。Action 强制超时生命周期不是成功性能样本：
它返回 `TIMEOUT`，诊断记录 `total_elapsed_ms=1000.50011`，即有 `0.50011 ms` 调度/收尾越界；对应
测试按冻结设计的“1 秒预算加调度误差”通过并有界结束。这里明确保留该差异，不把它写成严格
`<1000 ms` 成功。

## 4. 三平台、双模式与失败矩阵

本轮采用冻结 checklist 允许的组合覆盖，不声称执行了 24 个独立的真实 Action E2E 单元格：

| 层 | wheel | legged | hopper | 证据 |
| --- | --- | --- | --- | --- |
| 非平凡成功 | off-grid 精确目标 | off-grid 精确位姿/yaw | 真实 hop 到 off-grid 目标 | 三个 backend GTest |
| `NO_PATH` | 窄一格 choke | 超能力台阶 | 占据落点 | 三个 backend GTest |
| `TIMEOUT` | immediate/搜索 deadline | immediate/搜索 deadline | immediate 与未穷尽 landing scan | 三个 backend GTest |
| cancel | immediate/重建/搜索 | immediate/重建/搜索 | immediate/重建/优先级 | 三个 backend GTest |
| `LUNAR_SURFACE` 编排 | 三平台默认 backend 均成功；global=1、local=1 | 同左 | 同左 | `DefaultBackendsSolveAllThreePlatformsInBothModes` 与 Surface global failure 表 |
| `LAVA_TUBE` 编排 | 三平台默认 backend 均成功；global=0、local=1 | 同左 | 同左 | 同一双模式测试与 Lava local failure 表 |

`PreservesGlobalFailureAndExceptionSemantics` 对 Surface 的
`NO_PATH/TIMEOUT/REQUEST_CANCELED` 做表驱动编排验证；
`PreservesCanceledNoPathAndExceptionSemantics` 和
`CaveLocalStageMayUseTheFullNineHundredFiftyMilliseconds` 对 Lava 做对应验证。三个 backend 的状态合同
与模式编排接口相同，因此这是一项 backend × orchestration 的组合证明。真实 provider/bag 缺失，故不把
组合证明升级为三平台真实输入 E2E。

额外矩阵日志：dual mode 6/6、wheel 4/4、legged 4/4、hopper 4/4、ROS Action/diagnostics
11/11，全部 PASS。

## 5. fixed-domain、Action、diagnostics 与删除准入

installed overlay 下重复运行：

```bash
python3 -m pytest -q -p no:cacheprovider \
  pure_planner/tests/launch/test_pure_plan_motion_server.py \
  -k 'fixed_domain_runs_all_platforms_and_mismatch_without_graph_leaks or installed_production_closure_excludes_test_seams'
```

结果为 `2 passed, 7 deselected`。fixed domain 依次启动 wheel、legged、hopper，逐次 drain ROS graph，
随后验证 legged+wheel.yaml mismatch fail-closed。动态 graph 和 C++ Action 测试共同证明：

- 恰好 4 个业务订阅：global overview、local grid map、Odometry、`/tf`；
- `/Car/T4/plan_motion` Action 的 5 个内部 service 存在；
- `/Car/T4/planning/diagnostics` 恰有 1 个 publisher；
- 每个 accepted request 恰有一条、恰含 10 个键的 diagnostics；
- Surface/Lava 成功、`TIMEOUT`、client cancel、replace 和异常恢复均有界结束；
- wheel、legged、hopper 安装配置选择正确，错配返回 `PLANNER_ERROR`。

组合回归使用的 ROS fixture 同时具备：goal z=`NaN`、消息 stamp=0、Odometry pose/twist covariance
首元素=`1000000.0`、无 runtime revision/status/task/feedback 订阅；Surface 和 Lava 请求仍进入注入的
planner 并成功。另有 `LastArrivalWinsRegardlessOfStamp` 覆盖 stamp `100 -> 0`，以及 state adapter
大协方差测试。它们证明这些字段不构成准入，不能解释为真实地图质量证据。

调用次数和计时的直接证据：

| 路径 | global calls/time | local calls/time | total | 结果 |
| --- | --- | --- | --- | --- |
| Lava 编排 fake clock | `0 / 0 ms` | `1 / 30 ms` | `30 ms` | PASS |
| Surface 编排 fake clock | `1 / 20 ms` | `1 / 30 ms` | `50 ms` | PASS |
| ROS diagnostics timing fixture | `1 / 2 ms` | `1 / 3 ms` | `8.127783 ms` wall | PASS |
| ROS cancel fixture | 保留已发生的计数合同 | 保留已发生的计数合同 | `1.124787 ms` | `REQUEST_CANCELED` |
| ROS timeout fixture | 保留已发生的计数合同 | 保留已发生的计数合同 | `1000.50011 ms` | `TIMEOUT` |

## 6. installed consumer 与生产闭包

仓外 consumer 从 fresh install prefix 包含全部 7 个公开 ROS 头，经独立 CMake configure、Release
compile、强制 link 和执行，输出 `installed headers consumer: OK`。

`nm -D -C --defined-only`、`strings -a` 和 `ldd` 同时扫描 core/ROS installed shared library；下列
test seam 与 legacy token 均无命中：

```text
ServerExecutionHooks
PurePlanMotionServerTestFactory
pure_plan_motion_server_test_seam
before_terminal_primitive
after_terminal_primitive
before_cancel_response_return
lunar_planner_core
lunar_planner_ros
luna_t3_map_adapter
```

三个私有测试头也未安装。pure v1 live checker 和 legacy v5 live checker 分别绑定唯一 fresh provider，
均输出 `OK`。这证明安装闭包隔离，不证明生产 ROS domain 已完成原子切换。

## 7. rosbag 与真实输入

在 primary 和 final recheck 都执行了：

```bash
find /home/kai -maxdepth 8 -type f -name metadata.yaml -print 2>&1 | sort
```

两次原始输出都为空，`METADATA_COUNT=0`。没有候选可供 `ros2 bag info` 分类，因此没有自行构造输入，
也没有回放任何 bag：

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| metadata 定位 | `SEARCHED_ZERO_CANDIDATES` | 两次都已执行搜索，候选数为 0 |
| Lava 真实 bag 子集 | `NOT_RUN_REAL_BAG_NOT_LOCATED` | 无 metadata，未回放，不能声明真实 `LAVA_TUBE` |
| Surface 真实全局+局部 E2E | `NOT_RUN_MISSING_GLOBAL_INPUT` | 本轮未接入/未验证在线 producer，也无含 `/Car/T3/mapping/global_overview` 的完整 bag |
| synthetic global | `FORBIDDEN` | 禁止使用，且本轮没有生成、零填充或用 local map 假冒全局图 |

## 8. Final Review Checklist

| checklist | 状态 | fresh 证据/边界 |
| --- | --- | --- |
| Action 合同与调用端原子重建 | `PASS_BUILD_SCOPE / NOT_RUN_PRODUCTION_CUTOVER` | 消息、pure server 同次 Release 构建且 live checker 只有一个 provider；外部生产调用端和同 domain 切换未执行 |
| 无旧十层 MapSnapshot、Lifecycle、旧 planner/adapter 生产依赖 | PASS | no-legacy CTest、installed consumer、nm/strings/ldd 闭包 |
| global int8 与 local occupancy/elevation 语义 | PASS | map/projection/adapter 全量 CTest |
| 三平台×双模式×成功/NO_PATH/TIMEOUT/cancel | `PASS_COMPOSITIONAL` | backend × orchestration 组合证明；不冒充真实 E2E |
| off-grid 精确目标和真实足迹窄通道 | PASS | wheel/legged/hopper backend 全测 |
| hopper 7×7 按需障碍高度与垂直柱回退 | PASS | height estimator、hopper integrated test |
| ARA* 四档 epsilon、OPEN/CLOSED/INCONS、deadline incumbent 语义 | PASS | ARA* 13 项测试及全 core CTest |
| 昂贵边最多检查一次；调用次数与 wall time | PASS | edge cache/backend metrics、dual-mode timing、ROS diagnostics |
| 旧/零 stamp、大 covariance、缺 runtime revision/status/task/feedback 仍进入算法 | PASS | ROS 组合 fixture、InputStore/StateAdapter、精确四订阅 graph |
| Humble 构建、全测、边界、UTF-8、artifact | PASS | core+ROS `283/283`；pure：primary `54/54`、final recheck `54/54`、Task 16 change set `58/58`；foundation `148/148`；boundary `14/14`；checker/scans |
| 当前 bag 只声明可验证子集；Surface 无全局时保留非目标 | PASS | 本轮连历史子集 metadata 也未定位，因此 Lava 与 Surface 都明确 NOT_RUN |

## 9. Readiness 分级

| 级别 | 状态 | 可声明范围 |
| --- | --- | --- |
| source / contract | PASS | 生产源码/前置测试基线合同、隔离和 repository boundaries |
| Ubuntu Humble Release | PASS | amd64 三包构建与 core/ROS/foundation 全测；pure 分层为 primary 54/54、final recheck 54/54、Task 16 change set 58/58 |
| offline algorithm + ROS fixture | PASS | 三平台 backend、双模式编排、Action/diagnostics/graph 与 installed closure |
| real Lava bag | `NOT_RUN_REAL_BAG_NOT_LOCATED` | 不可声明 |
| real Surface global+local | `NOT_RUN_MISSING_GLOBAL_INPUT` | 不可声明 |
| production atomic domain cutover | `NOT_RUN` | 外部 producer/client/server 未同次 rebuild/install/restart |
| Jetson AGX Orin | `NOT_RUN` | 未做 aarch64 原生构建、性能、功耗或稳定性验收 |

## 10. 完成前源码安全检查

final recheck 已通过 UTF-8 读取 `197` 个相关文本文件、仓内
`build/install/log` 扫描、`*.pyc/*.pyo` 扫描和两套接口 checker。Task 16 change set 在报告修改后的门为：

```bash
git diff --check
python3 -c 'from pathlib import Path; [p.read_text(encoding="utf-8") for p in (Path("pure_planner/README.md"), Path("pure_planner/VERIFICATION.md"))]'
python3 -m pytest -q -p no:cacheprovider pure_planner/tests
python3 -m pytest -q -p no:cacheprovider pure_planner/tests/test_launch_contract.py \
  pure_planner/tests/test_isolation_contract.py
```

实际结果是完整 pure `58/58`、README + isolation `33/33`、UTF-8 `198`
文件、artifact/bytecode scan `OK`、repository checker `OK` 和 boundary `14/14`。本报告不授权
push、PR、生产切换或设备激活。

第一次 Task 16 change-set final 命令的 58/58、33/33 和 UTF-8 已通过，但仓内 bytecode
扫描如实返回 `REPO_BYTECODE_SCAN=FAILED`：发现 7 个 host CPython 3.12/pytest 7.4.4
产物。它们按明确路径逐个移入 Task 16 change-set artifact 的 `quarantine/`，并保留 SHA-256
和恢复路径；没有删除。相同 Humble final 命令随后重跑，原始输出
`quality-fix-final-after-quarantine.log` 为 58/58、33/33、`REPO_ARTIFACT_SCAN=OK` 和
`REPO_BYTECODE_SCAN=OK`。
