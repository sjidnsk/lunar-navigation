# Pure Frontier Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在物理隔离的纯算法目录中实现一个基于任务多边形、WFD 前沿、候选观测位姿、90 度/10 米信息增益和真实全局路径代价的 ROS 2 Humble 探索节点，并仅在任务区域内不存在可达前沿时正常结束。

**Architecture:** 探索核心保持 ROS 无关，负责地图分类、任务区域栅格化、前沿检测、候选生成、可见未知量、排序、失败记忆、覆盖率和状态转换；ROS 适配层只负责订阅课题三接口、解析最新 TF、调用纯规划器 Action、发布执行参考及诊断。纯探索器读取纯规划器拥有并安装的唯一平台配置，不移动或复制配置，也不复制旧探索策略、PPO、训练运行时或旧安全准入逻辑。

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, C++20, `ament_cmake`, `rclcpp`, `rclcpp_action`, `nav_msgs`, `grid_map_msgs`, `tf2_msgs`, `diagnostic_msgs`, `visualization_msgs`, `yaml-cpp`, GoogleTest, `launch_testing`, Python 3 repository contract tests.

**Spec:** `docs/superpowers/specs/2026-08-22-pure-frontier-exploration-design.md`

## Global Constraints

- 本计划与 `docs/superpowers/plans/2026-08-22-dual-mode-anytime-pure-planner.md` 并行执行：Task 1--9 可基于已验证的规划 ROS 预集成提交实施；用户已授权 Task 9 与最终规划器集成并行。Task 10--16 必须等待规划器 Task 10、14 和最终接口审核完成，并通过下述 `bf1bb79` gate 后才能实施或集成验证。
- 本计划从包含纯规划 ROS 接口、配置和诊断转换的 `6a94182` 创建独立工作树；不得直接在 `/home/kai/lunar-navigation-orin` 当前脏工作区实施，也不得修改任何规划器任务工作树。
- 新功能限定在 `pure_planner/`、本计划文档及必要的仓库边界检查白名单内；不得修改或复制旧 PPO、训练、旧探索运行时。
- 课题三输入保持原生语义：全局 `OccupancyGrid` 使用 `-1`、`0..100`，局部 `GridMap` 使用 `NaN`、`0.0..1.0`；探索器只订阅全局图，局部图由规划器消费。
- 不校验消息时间戳、地图版本、协方差、地图新鲜度或观测时间；只保留消息尺寸、数组长度、数值有限性、四元数可归一化、任务多边形合法性等结构校验。
- 所有项目 Topic 使用绝对 `/Car/T4` 前缀，课题三输入使用绝对 `/Car/T3` 前缀；ROS 时间只原样携带在消息中，耗时和卡滞检测只用 `std::chrono::steady_clock`。
- 正常完成条件唯一：当前任务区域内不存在可达前沿。覆盖率只统计，不参与终止。
- 候选位置探测数、候选视点数、碰撞检查工作量和可见性工作量四个上限由调用方显式传入正数，
  核心和默认 YAML 不提供生产默认；超限为 `length_error`/ROS `ERROR`，绝不能作为空候选或零
  增益完成证据。
- 跨周期前沿/承诺相等比较完整 canonical key；同周期候选身份通过同一冻结 span 的
  `frontier_index` 解析完整 frontier key，再与 CandidateKey 组合比较。FNV-1a-64 只用于显示，
  任何算法判断不得只信任哈希；失败记忆按批准合同只使用 CandidateKey。
- 每一任务先写失败测试，再写最小实现；每个任务只提交本任务文件，不纳入用户现有改动或运行 artifact。
- 构建、测试和日志写入 `/home/kai/CodexDownloads/lunar_navigation/pure_frontier_exploration`，不得把 `build/`、`install/`、`log/` 写入仓库。
- 所有 ROS 2/C++、`colcon`、接口生成、launch 和运行验证必须在 Ubuntu 22.04 + ROS 2 Humble
  容器中执行；当前权威容器为 `ros2-humble`（镜像
  `osrf/ros:humble-desktop-full-jammy`）。宿主只做 Git、UTF-8、文档和静态仓库检查，不能替代容器
  构建证据。

## Dependency Gate

执行本计划前，在纯规划器工作树中运行：

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
test -f pure_planner/ros2_ws/src/lunar_pure_planner_core/package.xml
test -f pure_planner/ros2_ws/src/lunar_pure_planner_ros/package.xml
test -f pure_planner/config/wheel.yaml
test -f pure_planner/config/pure_planner.yaml
rg -n "environment_mode" ros2_ws/src/lunar_planning_msgs/action/PlanMotion.action
python3 -m pytest -q pure_planner/tests/test_isolation_contract.py \
  pure_planner/tests/test_action_contract.py \
  pure_planner/tests/test_launch_contract.py \
  pure_planner/tests/test_external_interface_contract.py
```

预期：全部命令退出码为 0；若任一文件或测试不存在，返回纯规划器计划，不得在本计划中补写规划器算法。

纯规划器还必须已经提供：

- `/Car/T4/plan_motion`，类型 `lunar_planning_msgs/action/PlanMotion`；
- `/Car/T4/planning/diagnostics`，每个 `request_id` 可关联全局规划和局部规划耗时；
- `/Car/T4/execution/motion_reference` 可消费的 `lunar_planning_msgs/msg/MotionReference`；
- `PlanMotion.environment_mode`，探索请求固定发送 `LUNAR_SURFACE`；
- 单 worker 合同：候选请求串行发送且 `replace_active_request=false`；
- `wheel.yaml` 中唯一的平台 footprint、clearance 和 base frame 定义。

上述预集成门允许不消费最终 Action server 的 Task 1--9 开始。开始 Task 10 前必须把规划器最终
Task 10/14 提交（当前已核对基线 `bf1bb79`，或合同等价后继）集成到本分支，并重新通过规划器的
完整构建、Action 生命周期、typed-result、diagnostics-before-terminal 和一秒 deadline 测试。

## Target File Structure

```text
pure_planner/
├── README.md
├── config/
│   ├── pure_planner.yaml
│   └── pure_exploration.yaml
├── launch/
│   ├── pure_planner.launch.py
│   └── pure_exploration.launch.py
├── ros2_ws/src/
│   ├── lunar_pure_exploration_msgs/
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   └── msg/{PureExplorationTask,PureExplorationStatus}.msg
│   ├── lunar_pure_exploration_core/
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   ├── include/lunar_pure_exploration_core/
│   │   │   ├── types.hpp
│   │   │   ├── occupancy_grid.hpp
│   │   │   ├── task_raster.hpp
│   │   │   ├── frontier_detector.hpp
│   │   │   ├── candidate_generator.hpp
│   │   │   ├── information_gain.hpp
│   │   │   ├── candidate_ranker.hpp
│   │   │   ├── failure_memory.hpp
│   │   │   ├── coverage.hpp
│   │   │   ├── exploration_state_machine.hpp
│   │   │   └── progress_monitor.hpp
│   │   ├── src/
│   │   └── test/
│   └── lunar_pure_exploration_ros/
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── include/lunar_pure_exploration_ros/
│       │   ├── platform_config_loader.hpp
│       │   ├── pose_resolver.hpp
│       │   ├── planner_client.hpp
│       │   ├── planner_timing_accumulator.hpp
│       │   ├── marker_builder.hpp
│       │   └── exploration_node.hpp
│       ├── src/
│       └── test/
└── tests/
    ├── exploration/test_exploration_isolation.py
    ├── exploration/test_interface_contract.py
    ├── launch/test_pure_exploration_launch.py
    └── scenarios/test_synthetic_exploration.py
```

## Execution Bootstrap

- [x] 从规划 ROS 预集成提交创建独立工作树和分支：

```bash
cd /home/kai/lunar-navigation-orin
git worktree add /home/kai/WS/lunar-navigation-orin-pure-exploration \
  -b feat/pure-frontier-exploration 6a94182
cd /home/kai/WS/lunar-navigation-orin-pure-exploration
git rev-parse --show-toplevel
git status --short
```

预期 Git 根为 `/home/kai/WS/lunar-navigation-orin-pure-exploration`，状态为空。

- [ ] 创建仓库外构建目录，并在 Humble 容器内固定环境：

```bash
mkdir -p /home/kai/CodexDownloads/lunar_navigation/pure_frontier_exploration/{build,install,log,test-results}
docker exec ros2-humble mkdir -p \
  /root/lunar-navigation-orin-pure-exploration \
  /root/CodexDownloads/lunar_navigation/pure_frontier_exploration
docker cp /home/kai/WS/lunar-navigation-orin-pure-exploration/. \
  ros2-humble:/root/lunar-navigation-orin-pure-exploration/
docker exec ros2-humble bash -lc '
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
export COLCON_BUILD_BASE=/root/CodexDownloads/lunar_navigation/pure_frontier_exploration/build
export COLCON_INSTALL_BASE=/root/CodexDownloads/lunar_navigation/pure_frontier_exploration/install
export COLCON_LOG_BASE=/root/CodexDownloads/lunar_navigation/pure_frontier_exploration/log
cd /root/lunar-navigation-orin-pure-exploration
pwd
'
docker cp ros2-humble:/root/CodexDownloads/lunar_navigation/pure_frontier_exploration/. \
  /home/kai/CodexDownloads/lunar_navigation/pure_frontier_exploration/
```

后续代码块中的 `source /opt/ros/humble/setup.bash`、`colcon`、`ros2` 和 `launch_testing` 命令均
表示在该容器内执行；不得因宿主缺少 `/opt/ros/humble` 而跳过或降级验证。
容器没有宿主 bind mount，因此每个任务在 RED/GREEN 前用 `docker cp` 同步当前源快照，验证后只把
仓库外 artifact 复制回宿主；不得把容器的 build/install/log 复制进 Git 工作树。

---

## Task 1: Freeze the Planner/Explorer Compatibility Contract

**Files:**

- Create: `pure_planner/tests/exploration/test_exploration_isolation.py`

**Interfaces:**

- Consumes planner-owned files under `share/lunar_pure_planner_ros/config/` without linking planner algorithms.
- `pure_planner/config/wheel.yaml` remains the single source for footprint, `minimum_clearance_m` and base frame.
- Freezes `PlanMotion.environment_mode`, exact planner diagnostics keys and the single-worker request contract used by later tasks.

- [ ] Write the failing isolation/config test.

```python
from pathlib import Path

import pytest
import yaml

@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[3]

def test_platform_config_has_one_source(repo_root):
    source = repo_root / "pure_planner/config/wheel.yaml"
    assert source.is_file()
    assert not list((repo_root / "pure_planner/ros2_ws/src").glob("**/config/wheel.yaml"))

def test_wheel_geometry_contract(repo_root):
    data = yaml.safe_load(
        (repo_root / "pure_planner/config/wheel.yaml")
        .read_text(encoding="utf-8")
    )

    def unique_value(node, key):
        values = []
        if isinstance(node, dict):
            values.extend(value for name, value in node.items() if name == key)
            for value in node.values():
                values.extend(unique_value(value, key))
        elif isinstance(node, list):
            for value in node:
                values.extend(unique_value(value, key))
        return values

    assert data["platform"] == "wheel"
    assert data["base_frame_id"] == "base_footprint"
    capability = data["capability"]
    assert capability["footprint_xy_m"] == [
        [0.591, 0.409], [0.591, -0.409],
        [-0.591, -0.409], [-0.591, 0.409],
    ]
    clearances = [value for value in unique_value(data, "minimum_clearance_m")
                  if isinstance(value, (int, float))]
    assert clearances == [0.2]
```

- [ ] Extend the failing test to parse `PlanMotion.action` and `external_interfaces.yaml`. Require
  `LUNAR_SURFACE=1`, `LAVA_TUBE=2`, `uint8 environment_mode`, the exact ten planner diagnostic keys,
  `/Car/T4/plan_motion` and `/Car/T4/planning/diagnostics`. Assert no exploration package contains a copied
  wheel YAML and no production exploration source depends on old planner/PPO packages.

- [ ] Run the test and confirm RED because the exploration isolation/compatibility test is new.

```bash
python3 -m pytest -q pure_planner/tests/exploration/test_exploration_isolation.py
```

- [ ] Add only the repository contract test; do not move planner config, edit planner launch, or modify planner
  packages. Rerun the planner static tests together with the new compatibility test.

```bash
python3 -m pytest -q \
  pure_planner/tests/test_action_contract.py \
  pure_planner/tests/test_launch_contract.py \
  pure_planner/tests/test_external_interface_contract.py \
  pure_planner/tests/exploration/test_exploration_isolation.py
```

- [ ] Commit only the compatibility contract.

```bash
git add pure_planner/tests/exploration/test_exploration_isolation.py
git commit -m "test(pure-exploration): freeze planner compatibility"
```

## Task 2: Define the Exploration Task and Status Interfaces

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_msgs/CMakeLists.txt`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_msgs/package.xml`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_msgs/msg/PureExplorationTask.msg`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_msgs/msg/PureExplorationStatus.msg`
- Create: `pure_planner/tests/exploration/test_interface_contract.py`

**Interfaces:**

- `/Car/T4/exploration/task` → `PureExplorationTask`.
- `/Car/T4/exploration/status` → `PureExplorationStatus`.
- Task commands: `START=1`, `PAUSE=2`, `RESUME=3`, `CANCEL=4`.
- Normal terminal state: `COMPLETED` with reason `COMPLETED_NO_REACHABLE_FRONTIER`.
- Structural or internal numeric failure state: `ERROR`; cancellation returns the node to `IDLE`.

- [ ] Write a failing contract test that parses the message files and asserts every field and enum value.

```python
TASK_FIELDS = {
    "uint8 START=1", "uint8 PAUSE=2", "uint8 RESUME=3", "uint8 CANCEL=4",
    "std_msgs/Header header", "string task_id", "uint8 command",
    "geometry_msgs/Polygon boundary",
}

STATUS_FIELDS = {
    "uint8 IDLE=0", "uint8 WAITING_FOR_INPUT=1", "uint8 SELECTING_FRONTIER=2",
    "uint8 PLANNING=3", "uint8 EXECUTING=4", "uint8 REPLANNING=5",
    "uint8 PAUSED=6", "uint8 COMPLETED=7", "uint8 ERROR=8",
    "std_msgs/Header header", "string task_id", "uint8 state", "string reason_code",
    "float64 polygon_area_m2", "float64 task_raster_area_m2",
    "float64 known_free_area_m2", "float64 known_occupied_area_m2",
    "float64 unknown_area_m2", "float64 outside_map_area_m2",
    "float64 coverage_ratio", "uint32 frontier_cluster_count",
    "uint32 candidate_count", "uint32 reachable_candidate_count",
    "uint32 failed_candidate_count", "uint32 completed_goal_count",
    "uint32 replan_count", "string current_plan_id",
    "geometry_msgs/Pose current_goal", "float64 active_elapsed_s",
}
```

- [ ] Run and observe missing-file failure.

```bash
python3 -m pytest -q pure_planner/tests/exploration/test_interface_contract.py
```

- [ ] Create both message definitions exactly as asserted and configure `rosidl_generate_interfaces` with `std_msgs` and `geometry_msgs` dependencies. `header.frame_id` for START must be `map`; the timestamp remains display/transport metadata and must not appear in any admission rule.

- [ ] Build and inspect generated interfaces.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build \
  --base-paths src ../../ros2_ws/src/lunar_planning_msgs \
  --build-base "$COLCON_BUILD_BASE" \
  --install-base "$COLCON_INSTALL_BASE" \
  --packages-select lunar_pure_exploration_msgs
source "$COLCON_INSTALL_BASE/setup.bash"
ros2 interface show lunar_pure_exploration_msgs/msg/PureExplorationTask
ros2 interface show lunar_pure_exploration_msgs/msg/PureExplorationStatus
cd ../../
python3 -m pytest -q pure_planner/tests/exploration/test_interface_contract.py
```

- [ ] Commit the interface package and contract test.

```bash
git add pure_planner/ros2_ws/src/lunar_pure_exploration_msgs \
        pure_planner/tests/exploration/test_interface_contract.py
git commit -m "feat(pure-exploration): define task and status interfaces"
```

## Task 3: Implement Native Occupancy Semantics and Task Rasterization

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/package.xml`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/types.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/occupancy_grid.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/task_raster.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/occupancy_grid.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/task_raster.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_task_raster.cpp`

**Interfaces:**

```cpp
namespace lunar::pure_exploration {
enum class CellState : std::uint8_t {
  kOutsideTask, kOutsideMap, kUnknown, kFree, kOccupied
};
struct Vec2 { double x; double y; };
struct Pose2 { double x; double y; double yaw; };
struct GridIndex { std::int32_t x; std::int32_t y; auto operator<=>(const GridIndex&) const = default; };
struct GridGeometry {
  std::uint32_t width;
  std::uint32_t height;
  double resolution;
  double origin_x;
  double origin_y;
  double origin_yaw;
};
struct Polygon2 { std::vector<Vec2> vertices; };

class OccupancyGridView {
 public:
  OccupancyGridView(GridGeometry geometry, std::span<const std::int8_t> data,
                    std::int8_t occupied_threshold);
  CellState Classify(GridIndex index) const;
  bool Contains(GridIndex index) const;
  std::optional<GridIndex> WorldToCell(Vec2 point) const;
  Vec2 CellCenter(GridIndex index) const;
  const GridGeometry& geometry() const;
};

class TaskRaster {
 public:
  struct Limits {
    std::size_t maximum_raster_cell_count{1048576U};
  };
  static TaskRaster Build(const OccupancyGridView& map, const Polygon2& polygon,
                          Limits limits = {});
  CellState Classify(GridIndex logical_index) const;
  bool ContainsCellCenter(GridIndex logical_index) const;
  bool IsMapBacked(GridIndex logical_index) const;
  Vec2 CellCenter(GridIndex logical_index) const;
  const GridGeometry& geometry() const;
  double polygon_area_m2() const;
  std::span<const GridIndex> task_cells() const;
  std::span<const GridIndex> map_backed_cells() const;
};
}
```

`OccupancyGridView` copies and owns the supplied data; it must not retain a caller-owned span. The constructor
and `TaskRaster::Build` throw `std::invalid_argument` for invalid structural contracts and
`std::overflow_error` for checked size/index/bounding-box arithmetic overflow. A zero resource limit is invalid;
a finite bbox above `maximum_raster_cell_count` throws `std::length_error` before allocation or iteration.
`WorldToCell` returns the signed
logical map index even when it lies outside the backing array; it returns `std::nullopt` only for non-finite or
non-representable coordinates. `Contains` is the separate map-backed predicate. `TaskRaster` owns its cell and
classification storage.

- [ ] Write tests for native map classification and concave polygon rasterization. The cases must assert: `-1`
  is UNKNOWN, `0` and `49` are FREE, `50` and `100` are OCCUPIED, and every other signed int8 value
  (including `< -1` and `101..127`) is UNKNOWN. Cells inside the task polygon but outside the current map are
  OUTSIDE_MAP, a concave notch is OUTSIDE_TASK, and world/cell conversion works for a finite rotated map origin.
  UNKNOWN and OUTSIDE_MAP remain nontraversable and are grouped only for potential-gain accounting.

- [ ] Add structural failure tests for zero resolution, `width*height != data.size()`, non-finite origin XY/yaw,
  fewer than three distinct polygon vertices, self-intersection, zero area, and non-finite coordinates. Quaternion
  normalization belongs to the ROS OccupancyGrid adapter in Task 11; do not invent a quaternion in the ROS-free
  `GridGeometry` API.

- [ ] Add lifetime and arithmetic tests: mutate/destroy the caller data after constructing `OccupancyGridView`
  and assert classifications remain stable; distinguish `invalid_argument` from `overflow_error`; cover
  `INT32_MIN/MAX` conversion bounds and checked `width*height`/polygon bounding-box loops. Use inverse rotation
  plus `floor` (never integer truncation) and round-trip both 90-degree and non-axis-aligned origins. Do not clip
  the task bounding box to the map. Boundary inclusion must use point-on-segment before half-open even-odd
  classification with a translation-invariant tolerance based on local edge vectors, never absolute world
  coordinates; reject duplicate vertices and collinear overlapping/self-intersecting edges. Test the default
  `1048576` bbox-cell limit and a `100000 x 100000` finite bbox that fails with `length_error` before allocation.

- [ ] Run the test target and observe link or missing-header failure.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" test \
  --build-base "$COLCON_BUILD_BASE" \
  --install-base "$COLCON_INSTALL_BASE" \
  --packages-select lunar_pure_exploration_core \
  --ctest-args -R test_task_raster --output-on-failure
```

- [ ] Implement strict structural validation, boundary-inclusive even-odd point-in-polygon classification at cell centers, signed logical indices beyond the map array, and direct native occupancy classification without `0..1` conversion.

- [ ] Build and rerun the package tests.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build \
  --base-paths src ../../ros2_ws/src/lunar_planning_msgs \
  --build-base "$COLCON_BUILD_BASE" \
  --install-base "$COLCON_INSTALL_BASE" \
  --packages-select lunar_pure_exploration_core
colcon --log-base "$COLCON_LOG_BASE" test \
  --build-base "$COLCON_BUILD_BASE" \
  --install-base "$COLCON_INSTALL_BASE" \
  --packages-select lunar_pure_exploration_core \
  --event-handlers console_direct+
colcon test-result --test-result-base "$COLCON_BUILD_BASE/lunar_pure_exploration_core" --verbose
```

- [ ] Commit the core package skeleton and raster logic.

```bash
git add pure_planner/ros2_ws/src/lunar_pure_exploration_core
git commit -m "feat(pure-exploration): rasterize task region on native occupancy map"
```

## Task 4: Detect and Cluster Reachable-Space Frontiers with WFD

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/frontier_detector.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/frontier_detector.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_frontier_detector.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt`

**Interfaces:**

```cpp
namespace lunar::pure_exploration {
struct FrontierCluster {
  std::uint64_t id;
  std::vector<GridIndex> cells;
  struct InterfaceEdge {
    GridIndex free_cell;
    GridIndex unknown_cell;
    std::uint8_t direction;
    Vec2 midpoint;
  };
  std::vector<InterfaceEdge> interface_edges;
  std::vector<std::int64_t> canonical_key;
  Vec2 centroid;
  double length_m;
};
enum class FrontierDetectionReason : std::uint8_t {
  kOk,
  kNoReachableFreeStart,
};
struct FrontierDetection {
  std::vector<FrontierCluster> clusters;
  std::uint32_t reachable_free_cell_count;
  bool has_reachable_free_start;
  FrontierDetectionReason reason;
};
struct FrontierParameters {
  double minimum_cluster_length_m;
};
class FrontierDetector {
 public:
  explicit FrontierDetector(FrontierParameters parameters);
  FrontierDetection Detect(const TaskRaster& raster, GridIndex robot_cell) const;
};
}
```

- [ ] Write failing tests for a corridor with one frontier, two disconnected unknown boundaries, an unknown island behind an occupied wall, a frontier clipped by a concave task boundary, and no frontier in a fully known region.

- [ ] Assert the exact WFD rules in tests: free-space BFS uses 4-neighbor adjacency; a frontier cell is
  robot-reachable FREE and has at least one 4-neighbor `kUnknown` task cell; `kOutsideMap` never creates a
  frontier; frontier clustering alone uses 8-neighbor adjacency; the minimum cluster length passed to the
  detector is `max(platform_width_m, 2*resolution_m)`; shorter clusters are dropped.

- [ ] Run the focused test and confirm failure.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" test \
  --build-base "$COLCON_BUILD_BASE" \
  --install-base "$COLCON_INSTALL_BASE" \
  --packages-select lunar_pure_exploration_core \
  --ctest-args -R test_frontier_detector --output-on-failure
```

- [ ] Implement deterministic breadth-first search with fixed neighbor enumeration. If the robot cell is not
  free, inspect only its four adjacent FREE task cells by increasing cell-center world distance and world `x/y`
  lexicographic order. If none is free, return an empty detection with
  `reason=kNoReachableFreeStart`; valid occupied/unknown content is not a structural ERROR.

- [ ] Build one unique directed interface edge for every frontier FREE cell to a four-neighbor UNKNOWN cell.
  Define `length_m = interface_edges.size() * resolution_m` and centroid as the length-weighted mean of interface
  midpoints. Encode logical-grid directions as `+x=0`, `+y=1`, `-x=2`, `-y=3`; canonically sort cells and
  interface edges by FREE-cell world x/y, direction, then UNKNOWN world x/y. Quantize midpoint x/y to signed
  millimetres with `std::llround` (half away from zero). Define `canonical_key` as the flattened signed-int64
  triples `[midpoint_x_mm, midpoint_y_mm, direction]` in edge order. For `id`, serialize each midpoint int64 as
  fixed-width two's-complement little-endian bytes followed by the one-byte direction and apply fixed FNV-1a-64.
  Cross-cycle equality uses the full key, while `id` is only a
  compact display field. Test reversed discovery order and equivalent independently rebuilt rasters; ROS stamps
  are absent from this API. Also construct two cluster values with the same display `id` but different
  `canonical_key` values to prove callers have enough data to reject a hash-only match.

- [ ] Rerun all core tests and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core \
  --event-handlers console_direct+
cd ../../
git add pure_planner/ros2_ws/src/lunar_pure_exploration_core
git commit -m "feat(pure-exploration): detect WFD frontier clusters"
```

## Task 5: Generate Platform-Safe Candidate Viewpoints

**Files:**

- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/occupancy_grid.hpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/occupancy_grid.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_occupancy_grid.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/task_raster.hpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/task_raster.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_task_raster.cpp`
- Create: `pure_planner/tests/exploration/test_coordinate_implementation_contract.py`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/candidate_generator.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/candidate_generator.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_candidate_generator.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt`

**Interfaces:**

```cpp
namespace lunar::pure_exploration {
class OccupancyGridView {
 public:
  // Geometry-taking overloads are the only coordinate-math implementation.
  static std::optional<Vec2> WorldToGrid(const GridGeometry& geometry,
                                         Vec2 world_point);
  static std::optional<Vec2> GridToWorld(const GridGeometry& geometry,
                                         Vec2 grid_point);
  static std::optional<GridIndex> WorldToCell(const GridGeometry& geometry,
                                              Vec2 world_point);
  static std::array<Vec2, 4> CellCornersInGrid(GridIndex logical_index);
  // Existing instance API delegates with geometry_.
  std::optional<Vec2> WorldToGrid(Vec2 world_point) const;
  std::optional<Vec2> GridToWorld(Vec2 grid_point) const;
  std::optional<GridIndex> WorldToCell(Vec2 world_point) const;
  Vec2 CellCenter(GridIndex index) const;  // calls static GridToWorld
};
class TaskRaster {
 public:
  // Existing Task 3 API remains. These call OccupancyGridView's static
  // geometry overloads with geometry_; no map data is duplicated.
  std::optional<Vec2> WorldToGrid(Vec2 world_point) const;
  std::optional<Vec2> GridToWorld(Vec2 grid_point) const;
  std::optional<GridIndex> WorldToCell(Vec2 world_point) const;
  std::array<Vec2, 4> CellCornersInGrid(GridIndex logical_index) const;
  Vec2 CellCenter(GridIndex index) const;  // calls the same static GridToWorld
};

struct PlatformGeometry {
  std::string platform_id;
  std::string platform_type;
  std::string base_frame_id;
  std::vector<Vec2> footprint_vertices;
  double minimum_clearance_m;
};
struct CandidateParameters {
  std::array<double, 5> yaw_offsets_rad;
};
struct CandidateKey {
  std::int64_t x_mm;
  std::int64_t y_mm;
  std::int64_t yaw_tenth_deg;
  auto operator<=>(const CandidateKey&) const = default;
};
struct CandidateView {
  std::uint64_t id;                            // display only
  std::uint64_t frontier_id;                   // display only
  std::size_t frontier_index;                  // validated against indexed span
  CandidateKey key;
  Pose2 pose;
  double frontier_distance_m;
  std::shared_ptr<const std::vector<std::int64_t>> frontier_canonical_key;
};
class CandidateGenerator {
 public:
  struct Limits {
    std::size_t maximum_position_probes;
    std::size_t maximum_candidate_views;
    std::size_t maximum_collision_work_units;
  };
  CandidateGenerator(PlatformGeometry platform, CandidateParameters parameters,
                     Limits limits);
  std::vector<CandidateView> Generate(const TaskRaster& raster,
                                      std::span<const FrontierCluster> frontiers) const;
};
}
```

- [ ] Extend `OccupancyGridView` tests first. Freeze its `WorldToGrid`/`GridToWorld` as the only world/origin-yaw
  mathematics, `WorldToCell` as inverse rotation plus `floor`, and `CellCornersInGrid({x,y})` as the four closed
  square corners `{(x,y),(x+1,y),(x+1,y+1),(x,y+1)}`. Then test that every TaskRaster method delegates and is
  bit-for-bit equivalent without a second transform implementation. Cover 0/30/90-degree origins, map-exterior
  signed indices, `INT32_MIN/MAX`, round trips, non-finite/nonrepresentable values, and source translation.
  Task 5/6 consume only these helpers.

- [ ] Modify the existing `OccupancyGridView::CellCenter` in `occupancy_grid.cpp` and
  `TaskRaster::CellCenter` in `task_raster.cpp`: each forms `{x+0.5,y+0.5}` and delegates to the exact static
  `OccupancyGridView::GridToWorld(geometry, point)` primitive. Add dynamic equivalence tests across translated,
  30/90-degree rotated and boundary indices. Add a source mutation guard that extracts both CellCenter bodies,
  requires the shared GridToWorld call, rejects duplicated `sin/cos`/origin/resolution transform expressions,
  and proves the guard fails against locally mutated source strings; it must not rewrite repository files.

- [ ] Write failing constructor/derivation tests with 0.2 m resolution and the exact planner-owned wheel footprint
  `[[0.591,0.409],[0.591,-0.409],[-0.591,-0.409],[-0.591,0.409]]` plus 0.2 m clearance. Assert length
  1.182 m, width 0.818 m, circumscribed radius about 0.71872 m, minimum spacing 1.182 m, minimum standoff
  about 0.91872 m and maximum extra search 2.364 m. Reject non-WHEELED/empty IDs, concave, self-intersecting,
  duplicate, degenerate or non-finite footprint, negative/NaN clearance, and yaw offsets which are non-finite,
  repeated or not strictly ascending. The deployment order remains exactly the approved five values. Reject zero
  values in every required CandidateGenerator Limits field. For V vertices, independently compute the checked
  static validation bound `V*(V-1)/2 + V*(V-3)/2 + V = V*(V-1)`; the four-vertex bound is exactly 12, so
  collision limit 12 reaches normal geometry validation while 11 throws `length_error` before any vertex/pair
  validation loop.
  Arithmetic overflow or a large-V bound above the limit has the same early `length_error` result. After resource
  admission, malformed geometry keeps its specified `invalid_argument` result.

- [ ] Freeze representative selection without inventing a boundary chain. For the Task 4 canonical
  `interface_edges` sequence of length `N`, select `ceil(q*N)-1` for `q={0.25,0.50,0.75}`, deduplicate the
  selected complete edge, use its midpoint as representative, and average every four-neighbor `kUnknown` center
  of its `free_cell` for the local unknown centroid. Test N=1/2/3/4, straight/L/ring/branch fixtures, multiple
  UNKNOWN neighbors, symmetric zero-vector fallback to the selected edge direction, reversed cluster input and
  an independently rebuilt raster. Derive centroid/normal/fallback/search direction in logical-grid coordinates,
  convert the final center once with shared GridToWorld, and derive world yaw from local yaw plus origin yaw.
  Equivalent 0/30/90-degree origins with translations 0 and 1e9 must preserve the local-grid normal, relative
  candidate/yaw and collision-cell set; no subtraction of large world coordinates is allowed. Duplicate full
  cluster canonical keys are `invalid_argument`.

- [ ] Freeze search and position groups. Use `d(k)=minimum_standoff_m+k*resolution_m` with integer `k` and
  `k*resolution_m <= 2*platform_length_m`; for the 0.2 m baseline assert only k=0..11 are probed. Test the first
  probe, only the last probe, a success one cell beyond the limit which must not be seen, and repeated centers
  mapping to one logical cell. Minimum spacing is only between XY position groups: `<1.182 m` rejects and exact
  equality accepts, while one safe XY retains every safe one of the five yaw variants. If all five yaw fail,
  continue to the next distance; once any yaw succeeds, preserve all safe yaw and stop that representative.
  Freeze `maximum_search_step` to double `extra=2.0*platform_length_m`, double `ratio=extra/resolution_m`, checked
  finite/size_t admission, then `floor(ratio)` followed by double-product correction to the greatest step satisfying
  `step*resolution_m<=extra`, with no epsilon or long-double recomputation. With exactly representable
  extra=2.0, resolution 0.2 gives 10, `nextafter(0.2,+inf)` gives 9 and
  `nextafter(0.2,0)` gives 10. Invalid non-finite/nonpositive inputs throw `invalid_argument`; finite inputs whose
  derived dimensions/radius/standoff/extra/spacing/ratio/center become non-finite or unrepresentable throw
  `overflow_error` before iteration or cast.

- [ ] Define the inflated collision oracle exactly. Normalize either input winding for a finite nondegenerate
  simple convex footprint. Transform the rotated footprint to the logical grid frame using TaskRaster helpers.
  A closed cell square whose grid-unit minimum distance to the original footprint is
  `<=minimum_clearance_m/resolution_m` is touched by the closed-disc Minkowski sum, including tangency, and must
  be both `kFree` and map-backed. Test 0/45/90-degree
  candidate yaw, 30/90-degree map origin yaw, footprint corner/edge/rounded-clearance tangency against each of
  OCCUPIED, UNKNOWN, OUTSIDE_MAP and OUTSIDE_TASK, a free center with body outside, and large-world translation.
  Use checked signed AABB/index arithmetic. Before reserve/transform, require the Generate collision budget to
  have at least V remaining units; charge one unit immediately before each vertex transform, then one before each
  AABB cell classification/contact test. A budget failure takes precedence over a later transform failure.

- [ ] Test snapshot-scoped identity and deterministic output. Normalize yaw to `[-pi,pi)` (`+pi -> -pi`), then
  quantize XY to millimetres and yaw to 0.1 degree with half-away-from-zero. Reject non-finite quantization input
  with `invalid_argument`; for a finite scaled value outside int64, throw `overflow_error` before `llround`.
  Hash the input cluster's full frontier canonical key followed by CandidateKey as fixed little-endian two's-
  complement int64 bytes for display only. For each frontier that emits at least one view, create exactly one
  `shared_ptr<const vector<int64_t>>` full-key copy and share it across all that frontier's CandidateViews; never
  copy the variable key per view. Cover ±half
  boundaries, `-0.0`, `+pi/-pi`, large finite coordinates, independently rebuilt snapshots, same frontier
  display hash with different full keys, and same candidate display hash with different CandidateKeys. Preserve
  `frontier_index` as the original input-span index while processing/output order follows canonical keys. Assert
  same-frontier views share the exact owner and different full keys do not. `frontier_distance_m` is
  center-to-selected-edge-midpoint distance.

- [ ] Add resource tests. The same positive `maximum_collision_work_units` independently admits one constructor
  static-footprint bound and caps each Generate call; constructor admission never reduces a later Generate budget,
  and each Generate starts from zero. Count every representative/distance center as one position probe, every safe
  yaw about to be emitted as one candidate view, every transformed footprint vertex, and every logical cell taken
  from an inflated-footprint AABB before classification/geometric contact testing. Use the uniform rule “if
  used==limit throw, else increment”. For a free collision fixture independently derive
  `N=V+enumerated_AABB_cells` for each attempted yaw and assert limit N succeeds/N-1 throws; in the approved
  five-yaw wheel fixture this is 20 vertex transforms + 463 AABB cells = 483, so 483 succeeds and 482 throws.
  Also prove the next CollisionFree call sees the cumulative remaining budget. For each CandidateGenerator budget
  assert exact-limit success, one-under-limit
  `std::length_error`, no reserve/transform/final large-vector allocation before preflight rejection, deterministic
  finite exit at tiny resolution and the Task 3 raster cap, and no interpretation as an empty valid result.

- [ ] Run the focused test and observe failure.

- [ ] Implement one shared OccupancyGridView transform and TaskRaster delegation, canonical-edge representative
  generation, integer-bounded search,
  position-group spatial buckets, exact closed-polygon/closed-cell distance collision, full candidate identity and
  explicit work budgets. Preflight the checked static-footprint validation bound before all vertex/pair loops;
  separately preflight/charge every CollisionFree vertex transform and charge every AABB cell. The standard yaw
  points from the final candidate center to the selected local UNKNOWN centroid; a zero unknown vector falls back
  to that selected edge. Compute this geometry in local grid space, freeze double-authority maximum-search steps,
  and distinguish invalid inputs from finite derived overflow before sorting or iteration.

- [ ] Run the core suite and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core \
  --event-handlers console_direct+
cd ../../
git add pure_planner/ros2_ws/src/lunar_pure_exploration_core
git commit -m "feat(pure-exploration): generate footprint-safe viewpoints"
```

## Task 6: Compute 90-Degree, 10-Meter Visible Information Gain

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/information_gain.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/detail/visibility_traversal.hpp` (inline implementation; non-installed)
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/information_gain.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_information_gain.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_visibility_traversal.cpp`
- Create: `pure_planner/tests/exploration/test_visibility_implementation_contract.py`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt`

**Interfaces:**

```cpp
namespace lunar::pure_exploration {
struct SensorModel {
  double range_m;
  double field_of_view_rad;
};
struct GainEvaluation {
  std::uint32_t visible_unknown_cells;
  double visible_unknown_area_m2;
};
class InformationGainEvaluator {
 public:
  struct Limits {
    std::size_t maximum_visibility_work_units;
  };
  InformationGainEvaluator(SensorModel model, Limits limits);
  GainEvaluation Evaluate(const TaskRaster& raster, const CandidateView& candidate) const;
};
}
```

`InformationGainEvaluator::Evaluate` above remains the only stable public API. Add a non-installed,
non-exported, test-linkable seam in `src/detail/visibility_traversal.hpp`; the exact type spelling may follow
project conventions, but it must preserve these semantics:

```cpp
namespace lunar::pure_exploration::detail {
enum class TraceControl { kContinue, kStop };
struct VisibilityWorkBudget {
  std::size_t limit;
  std::size_t used;
};
void ConsumeVisibilityWork(VisibilityWorkBudget& budget);
struct TraceSummary {
  std::size_t visited_cell_count;
};
using TraceGroupVisitor = std::function<TraceControl(
    long double first_contact_t, std::span<const GridIndex> canonical_group)>;
TraceSummary TraceClosedSegment(Vec2 start_grid, Vec2 end_grid,
                                VisibilityWorkBudget& budget,
                                const TraceGroupVisitor& visitor);
enum class PreTargetContact { kPass, kOccupied, kOutsideTask };
PreTargetContact ClassifyPreTargetContact(CellState state, GridIndex visited,
                                         GridIndex target);
void CheckedVisibleIncrement(std::uint32_t& count);
}
```

The header is private implementation detail: CMake exposes its directory only to the core target and focused
test target, never installs/exports it, and Task 7/8/ROS code must not include it. The complete
`TraceClosedSegment` and `ConsumeVisibilityWork` implementations are inline in this exact header so the source
contract checks the code production executes; they may not delegate traversal to an unscanned cpp/helper.
`TraceClosedSegment` accepts
arbitrary finite continuous grid endpoints, streams complete t-groups in increasing t, orders each group by
`(x,y)`, charges every unique cell before invoking the visitor, and returns the per-ray unique visited count.
The visitor may stop only after receiving a complete tie group; stopped traversal neither generates nor charges
later groups. Production `Evaluate` must call this seam and the checked increment/contact helpers rather than
duplicate their semantics; its AABB cell path must use the same `ConsumeVisibilityWork` helper.

- [ ] **Oracle 1 — construction:** reject NaN/Inf/zero/negative range; NaN/Inf/zero/negative FOV and
  `nextafter(2*pi,+inf)`; accept FOV exactly `2*pi`; reject visibility limit zero.

- [ ] **Oracle 2 — candidate input:** each NaN/Inf pose x/y/yaw throws `invalid_argument`; failure of shared
  WorldToGrid representation throws `overflow_error`, never `{0,0.0}`.

- [ ] **Oracle 3 — range authority:** compute double `range_m/resolution_m` before long-double promotion. At
  resolution 1.0 test nextafter-inside/exact 10 m/nextafter-outside; separately assert 50 cells at 0.2 m and
  100 cells at 0.1 m lie on the included boundary, with no epsilon.

- [ ] **Oracle 4 — FOV authority:** compute double atan2/remainder and double half-FOV before promotion. Test
  ±45° inside/exact/outward-nextafter, yaw wrap across ±pi, and FOV=2*pi including the reverse target.

- [ ] **Oracle 5 — frames:** use `yaw_grid=wrap(candidate.pose.yaw-origin_yaw)` and TaskRaster transforms. Assert
  equivalent visible sets at origin yaw 0/30/90 degrees and under an exactly representable large translation such
  as 1e9; do not use a translation where resolution is below the double ULP.

- [ ] **Oracle 6 — five yaw/single-result API:** the approved five yaw values at one XY produce at least three
  different gains; arbitrary Evaluate call order leaves each candidate result unchanged; zero gain returns exactly
  `{0,0.0}` and is not filtered by Task 6.

- [ ] **Oracle 7 — native states:** carry `-1/0/49/50/100/101/-128` through OccupancyGridView and TaskRaster;
  only target `kUnknown`/task-internal `kOutsideMap` contributes, with no 0..1/NaN conversion.

- [ ] **Oracle 8 — blocker matrix:** middle OCCUPIED blocks far UNKNOWN; middle FREE/UNKNOWN/OUTSIDE_MAP does
  not; OCCUPIED target does not launch a ray or contribute. Directly test the detail contact classifier: every
  state at `visited==target` is pass, while non-target OUTSIDE_TASK/OCCUPIED remain distinct for tie precedence.

- [ ] **Oracle 9 — corner supercover:** a diagonal crossing one grid corner emits x-side, y-side and diagonal as
  the same t group; either side OCCUPIED blocks, both FREE leave the far target visible. Call
  `detail::TraceClosedSegment` with arbitrary continuous grid endpoints and assert the exact groups/count directly.

- [ ] **Oracle 10 — grid-line supercover:** horizontal and vertical segments exactly along an integer grid line
  enumerate both closed-cell sides for the entire segment without epsilon offsets; an OCCUPIED cell on either side
  blocks. Verify the raw streamed groups/count through the detail seam, not only the public gain result.

- [ ] **Oracle 11 — t=0/tie groups:** start at an ordinary point, one grid line and a grid corner and assert
  initial groups of 1/2/4 unique cells. A later equal-t group containing both OUTSIDE_TASK and OCCUPIED consumes
  and classifies the whole group before applying `OUTSIDE_TASK > OCCUPIED`; repeated runs have identical visits.
  The detail visitor records exact t-groups, canonical `(x,y)` order and returned visited count.

- [ ] **Oracle 12 — concave task:** candidate and target are task-internal but their segment touches an
  OUTSIDE_TASK concave notch and later re-enters; the target is not visible.

- [ ] **Oracle 13 — target-only dedup:** two or more collinear UNKNOWN targets are each counted once even though
  farther rays cross nearer UNKNOWN; intermediate UNKNOWN/OUTSIDE_MAP never contributes opportunistically.

- [ ] **Oracle 14 — resolution goldens:** aligned UNKNOWN rectangle `x=[2,3), y=[-0.4,0.4)` gives 20 cells at
  0.2 m and 80 at 0.1 m, bitwise-equal area under the same double expression. Independent integer-sector oracle
  gives 1996 cells/79.84 m² at 0.2 m and 7924 cells/79.24 m² at 0.1 m; production evaluator cannot self-generate
  expected values.

- [ ] **Oracle 15 — output arithmetic:** every fixture asserts the frozen order
  `double(count)*(resolution*resolution)`; check UINT32_MAX before increment, preflight area in long double and
  throw `overflow_error` instead of wrapping count or returning infinite area. Directly seed
  `CheckedVisibleIncrement` at UINT32_MAX-1/MAX so the overflow branch is reachable without an impossible raster.

- [ ] **Oracle 16 — exact 11/10 budget:** for a 3x3, resolution=10, range=10 fixture, the tight AABB consumes 9
  units and the unique adjacent gain ray consumes origin+target 2 units. Limit 11 succeeds and 10 throws
  `length_error`.

- [ ] **Oracle 17 — blocker budget:** lock a ray with a middle blocking t group; every unique cell in that group
  is consumed/classified, then no later group is generated or charged. Assert exact success/failure limits.

- [ ] **Oracle 18 — extremes/complexity:** cover candidate/AABB/DDA near INT32_MIN/MAX, side expansion as virtual
  OUTSIDE_TASK, rotated origin and large translation. Tiny resolution whose theoretical tight AABB exceeds
  remaining budget throws before allocation/loop. At the 1,048,576-cell TaskRaster cap, prove repeated bounded
  results. Add a source-contract checker plus mutation fixtures: production `Evaluate` must call
  `ConsumeVisibilityWork`, `detail::TraceClosedSegment`, `ClassifyPreTargetContact` and
  `CheckedVisibleIncrement`; the checker must also
  extract the actual inline `TraceClosedSegment` body from `src/detail/visibility_traversal.hpp`, reject any
  two-dimensional ray bbox/slab scan, require `ConsumeVisibilityWork` on every cell-traversal path, and reject
  theoretical two-dimensional bbox-area `vector(width*height)`, `reserve(width*height)`, `resize(width*height)` or
  equivalent allocation even when no nested scan exists. Removing an Evaluate seam call, replacing the
  TraceClosedSegment body with bbox double loops, deleting its consume call, or inserting only such bbox-area
  allocation must each make the checker fail.

- [ ] Run the focused test and confirm failure.

- [ ] Implement the tight AABB exactly as
  `ceil(sx-r-0.5)..floor(sx+r-0.5)` for x/y, intersect safely with int32, enumerate y-major then x-major, and
  preflight theoretical AABB work against remaining budget. Charge before range/FOV/state checks; launch a ray
  only for qualifying UNKNOWN/OUTSIDE_MAP targets.

- [ ] Implement streaming O(K) closed-grid DDA with per-ray checked-index dedup, t=0 closed groups, three-cell
  corner events, explicit double-sided grid-line traversal and monotonic `[0,1]` t groups. Charge/classify the
  complete tie group before OUTSIDE_TASK/OCCUPIED precedence; exclude target from blockers and stop generating/
  charging after a block. Implement it once in the private `detail::TraceClosedSegment` seam and make production
  `Evaluate` consume that seam. Never scan/reserve a 2-D ray bbox before budget admission.

- [ ] Implement the frozen double-then-promote numeric authority, yaw_grid/wrap/zero-distance/2pi behavior,
  target-aware contact classifier, checked uint32 increment and area overflow checks. Consume only delegated
  TaskRaster transforms, preserve native states, and keep Task 6 a single-candidate evaluator with no sorting/
  filtering or cross-snapshot CandidateView cache. The detail header remains non-installed/non-exported.

- [ ] Implement `test_visibility_implementation_contract.py` as an executable source-contract validator, not a
  comment grep: validate the real `Evaluate` body and the actual inline `TraceClosedSegment` body at the frozen
  detail-header path. Exercise mutant source strings that remove each Evaluate seam call, replace the traversal
  body with a ray-bbox/slab nested loop, remove `ConsumeVisibilityWork` from a cell path, or add bbox-area-backed
  vector construction/reserve/resize without a nested loop. Each of these four mutation classes must be rejected
  while both production bodies pass.

- [ ] Run all core tests and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core \
  --event-handlers console_direct+
cd ../../
python3 -m pytest -q pure_planner/tests/exploration/test_visibility_implementation_contract.py
git add pure_planner/ros2_ws/src/lunar_pure_exploration_core \
        pure_planner/tests/exploration/test_visibility_implementation_contract.py
git commit -m "feat(pure-exploration): evaluate sensor-visible unknown gain"
```

## Task 7: Rank Candidates, Track Failures, and Calculate Coverage

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/candidate_ranker.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/failure_memory.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/coverage.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/candidate_ranker.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/failure_memory.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/coverage.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/detail/coverage_statistics.hpp` (inline private seam; non-installed)
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_candidate_ranker.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_failure_memory.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_coverage.cpp`
- Create: `pure_planner/tests/exploration/test_task7_implementation_contract.py`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt`

**Interfaces:**

```cpp
namespace lunar::pure_exploration {
struct CandidateGain {
  std::size_t candidate_index;
  double information_gain_m2;
};
struct PlannedCandidate {
  std::size_t candidate_index;
  double path_length_m;
};
struct RankedCandidate {
  std::size_t candidate_index;
  double information_gain_m2;
  double euclidean_distance_m;
  double path_length_m;
  double heading_change_rad;
  double revisit_penalty;
  double rank_value;
};
struct ScoreWeights {
  double information_gain = 0.60;
  double path_cost = 0.30;
  double heading_change = 0.05;
  double revisit = 0.05;
};
class CandidateRanker {
 public:
  CandidateRanker(ScoreWeights weights, double platform_length_m);
  std::vector<RankedCandidate> CoarseRank(
      std::span<const CandidateView> frozen_candidates,
      std::span<const FrontierCluster> frozen_frontiers,
      std::span<const CandidateGain> gains,
      Pose2 frozen_robot_pose) const;
  std::vector<RankedCandidate> FinalRank(
      std::span<const CandidateView> frozen_candidates,
      std::span<const FrontierCluster> frozen_frontiers,
      std::span<const CandidateGain> gains,
      std::span<const PlannedCandidate> planned,
      Pose2 frozen_robot_pose,
      std::span<const Vec2> completed_goal_positions,
      double resolution_m) const;
};
enum class PersistentFailureReason : std::uint8_t {
  kExecutionReplansExhausted,
};
struct FailureMemoryLimits {
  std::size_t maximum_entries;
  std::size_t maximum_patch_cells_per_entry;
  std::size_t maximum_total_patch_cells;
};
class FailureMemory {
 public:
  FailureMemory(double platform_length_m, FailureMemoryLimits limits);
  void BeginTask(std::string task_id);
  void RecordPersistentFailure(const CandidateView& candidate,
                               PersistentFailureReason reason,
                               const TaskRaster& raster);
  bool IsSuppressed(const CandidateView& candidate, const TaskRaster& raster);
  std::size_t size() const;
};
struct CoverageStats {
  double polygon_area_m2;
  double task_raster_area_m2;
  double known_free_area_m2;
  double known_occupied_area_m2;
  double unknown_area_m2;
  double outside_map_area_m2;
  double coverage_ratio;
};
CoverageStats CalculateCoverage(const TaskRaster& raster);
}
```

- [ ] Write candidate ranking tests for the exact coarse utility
  `information_gain_m2 / (euclidean_distance_m + platform_length_m)`. Task 7, not Task 6, removes only exact
  zero-gain candidates. `CandidateGain` contains only owning `candidate_index` and gain; CoarseRank must derive
  Euclidean distance and heading internally from one frozen robot pose. Freeze heading as
  `abs(remainder(normalized_target_yaw-robot_yaw,2*pi))`, including robot yaw near +pi, targets near -pi,
  +pi/-pi equivalence and negative-zero normalization. Freeze the tie-break order as higher gain, shorter
  Euclidean distance, smaller heading, lexicographically smaller world XY, then smaller normalized yaw, followed
  by full identity. Assert CoarseRank returns the complete positive-gain order for 33 candidates; Task 11 chunks
  this as 16/16/1 with a monotonic cursor and no repeated first batch or omitted tail. Every coarse output fixes
  `path_length_m=0.0` and `revisit_penalty=0.0`; assert all output scalars are initialized and finite.

- [ ] CandidateGain is the sole gain authority: its rows must be unique and cover the complete frozen candidate
  span; PlannedCandidate carries only candidate index/path length and is a unique reachable subset. Reject
  missing/duplicate/out-of-bounds candidate indices, out-of-bounds
  `candidate.frontier_index`, invalid frontier association and duplicate full
  `(frontier.canonical_key,CandidateKey)` identity. Both rank methods borrow the same frozen candidate/frontier
  spans, resolve the complete identity for the final total-order tie and return only index/scalars; they do not
  retain CandidateView/span/reference across snapshots. Permuting input records must not change output. Artificial
  equal candidate/frontier display hashes with different full keys remain distinct; display hashes never order,
  deduplicate or associate candidates.

- [ ] Write final-ranking tests for normalization by the reachable-set maximum: gain divided by maximum gain, path
  divided by maximum path length, heading change internally recomputed from the same frozen robot pose and divided
  by pi, and revisit internally derived from completed-goal XY plus current resolution. Revisit is 1 at distance
  `<=max(2*platform_length_m,2*resolution_m)` and 0 at `nextafter(radius,+inf)`. Verify the exact default score
  `0.60*gain_norm - 0.30*path_norm - 0.05*heading_norm - 0.05*revisit`; tie-break by higher gain, shorter real
  path, smaller heading, world XY, normalized yaw, then full identity. Empty planned input returns empty. Gain/path
  maximum zero contributes normalized 0 without division.

- [ ] Add scalar-domain tests before sorting: platform length and resolution finite positive; robot/candidate/
  completed-goal pose fields finite; gain/path finite nonnegative; every weight finite nonnegative. Use long double
  to require a strictly positive weight sum, while accepting a non-1 positive sum without renormalizing; reject
  all-zero weights. Do not write an impossible finite-double/long-double-sum-overflow RED and do not require the
  sum to fit double. Default weights sum exactly to 1.0. Require finite positive coarse denominator and finite
  utility/components/final score; use two DBL_MAX-level same-sign nonzero penalty components to make the final
  score exceed double range and deterministically throw `std::overflow_error`. NaN never reaches a comparator.

- [ ] Write persistent failure-memory tests for candidates that exhausted execution replans. Key entries with the
  exact Task 5 CandidateKey and never with display ID; the only writable typed reason is
  `kExecutionReplansExhausted`. Candidate-evaluation NO_PATH, TIMEOUT, cancel, rejection and transport/contract
  failures have no corresponding PersistentFailureReason; Record rejects unknown enum values. `BeginTask` rejects
  empty ID and unconditionally clears on every accepted
  START, including reuse of the same ID; PAUSE/RESUME/map recomputation never call it. Duplicate CandidateKey
  atomically replaces its entry without growing size; artificial equal display IDs with different CandidateKeys
  never match.

- [ ] Freeze failure patch geometry. Record the first finite candidate world XY, complete finite GridGeometry and
  radius `max(platform_length_m,2*recording_resolution_m)`. Freeze double-then-promote exactly: compute double
  `twice_resolution`, radius and `radius_cells=radius/resolution`, require every result finite positive, and obtain
  double `(cx,cy)` only through TaskRaster::WorldToGrid. Promote these double results to long double, then use
  `xmin=ceil(cx-radius_cells-0.5)`, `xmax=floor(cx+radius_cells-0.5)` and the same y formula. Do not copy map
  transform math or recompute the radius directly in long double.

- [ ] Before int32 conversion/clipping, compare the complete uncut theoretical AABB width/height/product against
  `maximum_patch_cells_per_entry`; an over-limit AABB throws `length_error` before allocation/iteration even if it
  also crosses int32. After resource admission, any bound outside int32 throws `overflow_error`; never clip, wrap or
  create virtual cells. Enumerate the admitted AABB in `(y,x)` order and charge every cell before classification.
  Membership uses logical long double `(x+0.5,y+0.5)` distance to the promoted double center and
  `<=promoted_radius_cells^2`, with no epsilon; tangent is included and double-nextafter outside is excluded.
  Save all five CellState values. Add golden REDs for resolution 0.2, rotated origin, tangent/nextafter, exact N/N-1
  uncut work, and both exception priorities at an int32 side.

- [ ] Later checks use the saved world center/double radius and the identical arithmetic. With identical geometry,
  distant classification changes retain suppression while any patch index/state/length change erases the entry and
  returns false. Changing width, height, resolution, origin X/Y or yaw erases the entry rather than comparing the
  same GridIndex in a different physical map.

- [ ] Add `FailureMemoryLimits` tests. All three values must be positive. Tight-AABB theoretical work is checked
  with the frozen uncut long-double authority against `maximum_patch_cells_per_entry` before int32 conversion,
  allocation or iteration; admitted int32/index/size_t operations remain checked, and every classification consumes
  one unit. Test exact N success/N-1 `length_error`, exact/over maximum entries, checked
  maximum total patch cells, and a tiny-resolution resource attack. No eviction or false unsuppression is allowed;
  failed new/replace operations leave entries, size and total unchanged.

- [ ] Write coverage tests asserting `(free + occupied) / all task cells`, with separate areas for free, occupied,
  in-map unknown and outside-map unknown. Add the non-installed/non-exported
  `src/detail/coverage_statistics.hpp` seam with `CoverageCounts`, `AccumulateCoverageState`,
  `CheckedCoverageArea` and `FinalizeCoverage`; it is visible only to coverage.cpp and its white-box test, never to
  installed headers, ROS or Task 8+. Inject kOutsideTask directly into the private accumulator and require
  `logic_error`. Verify
  polygon area uses the stored shoelace result while coverage uses deterministic raster count/area. A valid nonzero
  polygon with no cell center returns zero raster/class areas and ratio 0.0, never NaN. Freeze
  `area_ld=long double(count)*long double(resolution)*long double(resolution)`; inject a positive resolution for
  which `area_ld>0` but its double cast is 0 and require `overflow_error`, and separately test positive overflow.
  Nonempty ratio uses integer counts and remains in `[0,1]`.

- [ ] Add an API/source-contract test that RankedCandidate contains only owning index/scalars, rank/failure code does
  not compare display hashes or retain CandidateView, and CoverageStats/source contains no completion/success/
  terminal/threshold field or ratio comparison. Fully known coverage 1.0 and partial coverage both remain statistics;
  neither returns or triggers a completion state. Extract the real `CalculateCoverage` and inline private seam
  bodies: production must call `AccumulateCoverageState` and `FinalizeCoverage`; mutation fixtures that remove a
  shared call, treat OutsideTask as a normal state, or delete the positive-area cast-to-zero check must each fail.
  Reject installing/exporting the detail header; do not enlarge TaskRaster or coverage stable public APIs.

- [ ] Run the three focused C++ tests plus `test_task7_implementation_contract.py` and confirm failure.

- [ ] Implement owning-index complete ranking, internally derived distance/heading/revisit, wrapped-heading and
  full-identity total ties, checked maximum normalization and the complete numeric-domain rules above. Implement the
  geometry-aware closed-circle failure patch, typed persistent reason, unconditional START scope reset, three
  fail-closed limits and strong exception guarantee. Implement five-state/empty-denominator/representable-area
  coverage. Failure memory and ranking must not store or compare display hashes, message stamps or revisions;
  coverage must never participate in completion.

- [ ] Run the core suite and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core \
  --event-handlers console_direct+
cd ../../
python3 -m pytest -q pure_planner/tests/exploration/test_task7_implementation_contract.py
git add pure_planner/ros2_ws/src/lunar_pure_exploration_core \
        pure_planner/tests/exploration/test_task7_implementation_contract.py
git commit -m "feat(pure-exploration): rank candidates and track exploration statistics"
```

## Task 8: Implement Goal Commitment and Steady-Clock Progress Monitoring

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/exploration_state_machine.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/progress_monitor.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/exploration_state_machine.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/progress_monitor.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_exploration_state_machine.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_progress_monitor.cpp`
- Create: `pure_planner/tests/exploration/test_task8_implementation_contract.py`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/candidate_generator.hpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/src/candidate_generator.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/test/test_candidate_generator.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt`

**Interfaces:**

```cpp
namespace lunar::pure_exploration {
enum class ExplorationState : std::uint8_t {
  kIdle, kWaitingForInput, kSelectingFrontier, kPlanning, kExecuting,
  kReplanning, kPaused, kCompleted, kError
};
enum class GoalReleaseReason : std::uint8_t {
  kArrived, kNoPath, kCandidateInvalid, kFrontierDisappeared,
  kInformationGainZero
};
enum class ReplanCause : std::uint8_t { kRollingSegment, kStuckRecovery };
enum class ReplanResult : std::uint8_t { kStarted, kExhausted };
class ActiveGoal {
 public:
  ActiveGoal(ActiveGoal&& other) noexcept;
  ActiveGoal& operator=(ActiveGoal&&) = delete;
  ActiveGoal(const ActiveGoal&) = delete;
  ActiveGoal& operator=(const ActiveGoal&) = delete;
  std::uint64_t candidate_id() const;
  std::uint64_t frontier_id() const;
  std::span<const std::int64_t> frontier_canonical_key() const;
  const CandidateKey& candidate_key() const;
  const Pose2& target() const;
  const std::string& request_id() const;
  std::uint8_t replan_count() const;
  bool MatchesAnyFrontier(
      std::span<const FrontierCluster> current_frontiers) const;

 private:
  friend ActiveGoal MakeActiveGoal(const CandidateView&,
                                   std::span<const FrontierCluster>,
                                   std::string);
  friend class ExplorationStateMachine;
  ActiveGoal(std::uint64_t candidate_id, std::uint64_t frontier_id,
             std::vector<std::int64_t> frontier_canonical_key,
             CandidateKey candidate_key, Pose2 target,
             std::string request_id);
  std::uint64_t candidate_id_;
  std::uint64_t frontier_id_;
  std::vector<std::int64_t> frontier_canonical_key_;
  CandidateKey candidate_key_;
  Pose2 target_;
  std::string request_id_;
  std::uint8_t replan_count_{0U};
  bool owns_payload_{true};
};
ActiveGoal MakeActiveGoal(const CandidateView& candidate,
                          std::span<const FrontierCluster> frozen_frontiers,
                          std::string request_id);
class ExplorationStateMachine {
 public:
  explicit ExplorationStateMachine(std::uint8_t maximum_replans);
  void Start(std::string task_id);
  void WaitForInput();
  void BeginSelection();
  void BeginPlanning();
  void Pause();
  void Resume();
  void Cancel();
  void CommitGoal(ActiveGoal&& goal);
  ReplanResult BeginReplanning(ReplanCause cause);
  void ResumeExecution();
  void ReleaseGoal(GoalReleaseReason reason);
  void CompleteNoReachableFrontier();
  void Fail(std::string reason_code);
  bool MayReplaceGoalForMapUpdate() const;
  ExplorationState state() const;
  const std::string& task_id() const;
  const std::string& reason_code() const;
  const std::optional<ActiveGoal>& active_goal() const;
};
struct ProgressParameters {
  std::size_t maximum_executable_path_points;
  std::chrono::steady_clock::duration timeout{std::chrono::seconds{30}};
  double minimum_progress_m{0.2};
};
class ProgressMonitor {
 public:
  explicit ProgressMonitor(ProgressParameters parameters);
  void Reset(std::chrono::steady_clock::time_point now,
             std::span<const Vec2> path_polyline, Vec2 position);
  bool Update(std::chrono::steady_clock::time_point now, Vec2 position,
              bool paused);
  double historical_max_arc_length_m() const;
};
}
```

- [ ] Write state-machine tests first. Freeze construction as IDLE/empty-task/empty-reason/no-goal. Use a
  table-driven `(event, every ExplorationState)` matrix for every public transition: each legal cell asserts the
  literal destination reason, task retention/clearing, complete ActiveGoal contents and retry count; every illegal
  cell requires `std::logic_error` and a byte-for-behavior snapshot of state, task id, reason, full frontier key,
  CandidateKey, target, request id and retry count. Cover `Start`, `Cancel` and `Fail` from every state including
  REPLANNING/PAUSED/COMPLETED/ERROR: START requires nonempty task and replaces atomically; CANCEL is idempotent,
  returns IDLE and clears task/goal; FAIL requires nonempty reason, retains task and clears goal. Completion is
  legal only from SELECTING with no goal, retains task and reports `COMPLETED_NO_REACHABLE_FRONTIER`; PAUSE
  clears a committed goal and RESUME returns to WAITING. Empty START/FAIL input throws `invalid_argument` with
  the same complete strong guarantee. Assert `MayReplaceGoalForMapUpdate()` is false while a goal exists and true
  otherwise.

- [ ] Test CandidateView provenance, goal factory and release authority. CandidateGenerator makes exactly one
  immutable shared full-key owner per frontier with emitted views, and all of that frontier's views share it.
  `MakeActiveGoal` rejects empty request id, out-of-range `frontier_index`, non-finite pose,
  non-finite/negative frontier distance, null/empty shared key, display-id mismatch and full-key mismatch against
  the indexed cluster. Candidate A crossed with span B having the same display hash but a different full key must
  reject. It deep-copies request id, target, CandidateKey and the full frontier key; mutate/destroy the input span,
  shared owner and original storage afterward and require unchanged accessors. Assert ActiveGoal is move-constructible
  but neither copyable, move-assignable, aggregate nor default-constructible, and CommitGoal takes only
  `ActiveGoal&&`. After one successful commit, release and legally return to PLANNING; resubmitting that same
  moved-from object must throw `invalid_argument` with complete state-machine strong guarantee, independent of
  moved-from string/vector contents. Moving a fresh factory goal through one intermediate owner still succeeds.
  Same display id/different
  full keys in separate valid spans both succeed and `MatchesAnyFrontier` distinguishes them only by full key;
  artificial equal candidate/frontier display hashes do not create identity. `ReleaseGoal` accepts only final
  arrival, current candidate/footprint invalidation, full-key frontier disappearance, zero gain and conclusive
  `NO_PATH`, and enters SELECTING after clearing the goal. A forged `GoalReleaseReason(255)` in a legal source
  throws `invalid_argument` with complete strong guarantee.

- [ ] Test typed replanning with `maximum_replans=2`: rolling segment enters REPLANNING with
  `ROLLING_SEGMENT`, preserves both complete keys and count zero; `ResumeExecution` returns to EXECUTING. The
  first and second stuck recovery calls return `kStarted`, report `STUCK_RETRY`, increment to one and two, and can
  resume. The third returns `kExhausted`, reports `STUCK_RETRIES_EXHAUSTED`, clears the goal and enters SELECTING.
  `maximum_replans=0` exhausts on the first stuck event. The state machine is the only retry-count authority;
  ProgressParameters contains no retry field. After one stuck recovery, interleave rolling replan and require the
  count remains one, then the next stuck becomes two. With `maximum_replans=UINT8_MAX`, drive exact 254/255 and
  require the next call exhausts without wrap. Every resume preserves full key, CandidateKey, target, request id
  and count. A forged `ReplanCause(255)` in EXECUTING throws `invalid_argument` with complete strong guarantee.

- [ ] Write progress tests with manually supplied `steady_clock::time_point` values and literal hand-derived
  arcs. Constructor rejects nonpositive timeout, non-finite/nonpositive minimum progress and zero
  `maximum_executable_path_points`. Before Reset, Update and historical-max access throw `logic_error`. Reset
  rejects empty/non-finite path or position and over-limit points before allocation/work; exact limit succeeds.
  It owns the path: mutation/destruction of the input container cannot change later results. Freeze these literal
  fixtures: single point `{(1,1)}` projects any finite position to `0.0`; repeated
  `{(0,0),(1,0),(1,0),(1,1)}` at `(1,0.5)` gives `1.5`; nonconsecutive repeated points stay valid; self-crossing
  `{(0,0),(2,2),(0,2),(2,0)}` at `(1,1)` chooses `2+3*sqrt(2)`, not `sqrt(2)`; a separate bent path checks an
  unambiguous interior projection: `{(0,0),(2,0),(2,2)}` at `(1,0.5)` uniquely selects `(1,0)` with arc `1.0`,
  not distance to an
  endpoint. Assert these literal results with fixed double absolute/relative tolerances independent of production
  projection/arc helpers.

- [ ] Add numeric/lifecycle REDs. A finite path whose one segment or cumulative length exceeds `DBL_MAX` throws
  `overflow_error` with the old monitor behavior unchanged; a representable path plus DBL_MAX/-DBL_MAX-scale
  finite position still projects correctly through long-double intermediates. Reset is legal before first use and
  from active/stuck, clears the stuck latch and window, and requires `now >= last_accepted_time`; it is illegal
  while paused. Failed/backward Reset preserves owned path/window/pause/latch/history. Use
  `steady_clock::time_point::min()/max()` to prove a huge forward elapsed comparison does not overflow, and a
  reverse extreme rejects without mutation. A stuck result remains latched until successful Reset or resume.

- [ ] Assert 29 seconds without 0.2 m progress is not stuck and 30 seconds is. Test
  `nextafter(0.2,0)` does not reset, exact `0.2` and `nextafter(0.2,+inf)` do; for each branch verify the old
  t=30 boundary and the new window t=29/t=30 boundary. Backward motion never reduces historical maximum. Update
  rejects NaN/Inf position and backward time with behaviorally proven strong guarantee.

- [ ] Test pause semantics separately: first paused update ignores its position; movement and duration while
  paused do not change progress or trigger stuck; first unpaused update rebuilds historical maximum, baseline and
  timeout window from the current position, clears a pre-existing stuck latch and returns false; 29 seconds after
  resume is false and 30 seconds is true. Inject backward time independently into active, first-pause,
  already-paused and first-resume branches; each rejects with strong guarantee. In those same four branches,
  independently inject x/y NaN and positive/negative infinity; each throws `invalid_argument` before any
  pause-position ignore/resume rebuild, and subsequent finite historical-max plus 29/30-second behavior must prove
  path, pause, latch, baseline and last accepted time unchanged. Ignoring the first paused position applies only
  to finite positions. ROS/GPS/system timestamps and
  retry counts are absent from ProgressMonitor's API. Final-candidate position/yaw arrival remains explicitly
  owned by Task 12's execution-monitor tests, not Task 8; Task 8 only types `ReleaseGoal(kArrived)`.

- [ ] Add `test_task8_implementation_contract.py` with mutation-proven checks that both sources are linked into
  the core library, both gtests are registered, CandidateView retains one shared const key owner rather than a
  per-view vector, ActiveGoal stays one-shot move-constructible/non-copyable/non-move-assignable/nonaggregate/
  non-default with a custom source-invalidating move, ProgressParameters has exactly one required
  path-point resource limit and no retry field, and progress code contains no `system_clock`, ROS/rclcpp/GPS time,
  stamp/revision/version/freshness or hidden retry authority. Each source/CMake/static rule needs at least one
  built-in mutant that demonstrably fails validation.

- [ ] Run the focused tests and confirm failure.

- [ ] Implement only enough to pass the tests. All state transitions not listed in the spec matrix throw
  `std::logic_error` before mutation; forged enum values in a legal state throw `std::invalid_argument`. Extend
  CandidateGenerator with one shared immutable full-key copy per frontier, never one vector per view, and require
  exact key provenance at the ActiveGoal factory. Candidate cost uses the complete global `path_preview`, but execution
  progress uses the current executable `trajectory` transform sequence (wheel/legged) or `hops` (hopper); never
  infer executable extent from the complete preview. ProgressMonitor owns at most
  `maximum_executable_path_points`, projects with long-double intermediate arithmetic and finite checked double
  arc results, and for exactly equal nearest distance chooses the greater arc length. On the approved signed-integral
  steady-clock ABI, compare elapsed ticks by ordered epoch counts converted to the corresponding unsigned rep and
  subtracted modulo, never by potentially overflowing signed duration subtraction. Use `>=` for both
  time and progress boundaries. Paused positions are ignored and resume rebuilds the active window from the
  current position.

- [ ] Run all core tests and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_core \
  --event-handlers console_direct+
cd ../../
git add pure_planner/ros2_ws/src/lunar_pure_exploration_core
git commit -m "feat(pure-exploration): enforce committed goals and steady progress"
```

## Task 9: Load Platform Geometry and Resolve Latest Task3 Pose

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/package.xml`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/platform_config_loader.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/pose_resolver.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/src/platform_config_loader.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/src/pose_resolver.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/test/test_platform_config_loader.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/test/test_pose_resolver.cpp`

**Interfaces:**

```cpp
namespace lunar::pure_exploration_ros {
struct LoadedPlatformConfig {
  lunar::pure_exploration::PlatformGeometry geometry;
};
LoadedPlatformConfig LoadPlatformConfig(const std::filesystem::path& yaml_path,
                                        std::string_view platform_selector);

class PoseResolver {
 public:
  void UpdateOdometry(const nav_msgs::msg::Odometry& odometry);
  void UpdateTransforms(const tf2_msgs::msg::TFMessage& transforms);
  std::optional<lunar::pure_exploration::Pose2> LatestPoseInMap() const;
};
}
```

- [ ] Write a failing config test against the planner-installed wheel YAML under
  `share/lunar_pure_planner_ros/config/`. Require `platform=wheel`, unique nonempty `platform_id`,
  `platform_type=WHEELED`, `base_frame_id=base_footprint`, finite polygonal
  `capability.footprint_xy_m` and nonnegative `capability.minimum_clearance_m`. Assert the exact four
  vertices `[[0.591,0.409],[0.591,-0.409],[-0.591,-0.409],[-0.591,0.409]]` and `0.2 m` clearance.
  Freeze the adapter boundary: ROS/launch `platform_selector=wheel` selects the file/profile and maps to core
  `PlatformGeometry.platform_type=WHEELED`; the lowercase selector is never copied into the uppercase payload.
  Assert legged/hopper or mismatched selector/payload are rejected rather than silently coerced to wheel.

- [ ] Write pose tests using the approved upstream data shape: odometry in `odom` supplies the sole
  `odom -> base_link` state, and `/tf` supplies the latest direct `map -> odom`. Assert composition ignores all
  timestamps and that an `odom -> base_link` transform present in `/tf` never overrides Odometry.

- [ ] Add structural tests for malformed received transforms, zero-norm quaternion and non-finite translation. Missing map/odom/base transform content returns `std::nullopt` so the node enters `WAITING_FOR_INPUT`; it is not an ERROR. Do not reject covariance values, including the observed `1000000.0` diagonal.

- [ ] Run and observe missing-target failure.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" test \
  --build-base "$COLCON_BUILD_BASE" \
  --install-base "$COLCON_INSTALL_BASE" \
  --packages-select lunar_pure_exploration_ros \
  --event-handlers console_direct+
```

- [ ] Implement YAML loading from the planner-owned package share, quaternion normalization, planar transform composition and latest-value replacement without timestamp ordering. Do not link
  `lunar_pure_planner_core` or `lunar_pure_planner_ros`; the YAML file is the only shared platform input.

- [ ] Build, run both ROS adapter tests, and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" \
  --packages-up-to lunar_pure_exploration_ros
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros \
  --event-handlers console_direct+
cd ../../
git add pure_planner/ros2_ws/src/lunar_pure_exploration_ros
git commit -m "feat(pure-exploration): load platform geometry and resolve task3 pose"
```

## Task 10: Implement the PlanMotion Client and Timing Correlation

**Implementation gate:** **BLOCKED.** 当前 `feat/pure-frontier-exploration` 尚未包含最终
`PurePlanMotionServer` 集成；已核对的最终实现位于 `feat/pure-planner-isolated@bf1bb79`。只有
`bf1bb79` 或保持下述 Action/诊断合同的后继提交进入当前分支，并重新通过最终 planner 的构建、
Action 生命周期、typed-result 和 diagnostics-before-terminal 验证后，才允许开始本任务实现。
仅 fake server RED/GREEN 不能解除该 gate，也不能声明 Task 10 完成或最终 planner 兼容。

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/planner_client.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/planner_timing_accumulator.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/src/planner_client.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/src/planner_timing_accumulator.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_client.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_timing_accumulator.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/package.xml`

**Interfaces:**

```cpp
namespace lunar::pure_exploration_ros {
enum class PlannerEvaluationKind : std::uint8_t {
  kReachable,
  kExhaustiveNoPath,
  kRetryable,
  kCanceled,
  kContractError,
};
struct PlannerEvaluation {
  std::string request_id;
  std::uint64_t candidate_id;  // display only; never used for association
  PlannerEvaluationKind kind;
  std::string reason_code;
  std::optional<double> path_length_m;
  std::optional<lunar_planning_msgs::msg::MotionReference> reference;
};
struct PlannerClientParameters {
  std::size_t maximum_path_preview_poses;
  std::chrono::steady_clock::duration result_timeout;
};
class PlannerClient {
 public:
  using Completion = std::function<void(PlannerEvaluation)>;
  using SteadyNow =
      std::function<std::chrono::steady_clock::time_point()>;
  PlannerClient(rclcpp::Node& node, std::string action_name,
                PlannerClientParameters parameters,
                SteadyNow now = [] { return std::chrono::steady_clock::now(); });
  ~PlannerClient() noexcept;
  PlannerClient(const PlannerClient&) = delete;
  PlannerClient& operator=(const PlannerClient&) = delete;
  PlannerClient(PlannerClient&&) = delete;
  PlannerClient& operator=(PlannerClient&&) = delete;
  static std::string MakeRequestId(std::string_view task_id,
                                   std::uint64_t sequence);
  void Evaluate(std::string task_id, std::string request_id,
                const lunar::pure_exploration::CandidateView& candidate,
                double position_tolerance_m, double yaw_tolerance_rad,
                Completion completion);
  void CancelActive();
  void PollTimeout();  // production steady timer and deterministic tests share this seam
};
struct PlannerTiming {
  std::string request_id;
  double global_elapsed_ms;
  std::uint64_t global_call_count;
  double local_elapsed_ms;
  std::uint64_t local_call_count;
  double total_elapsed_ms;
};
enum class TimingIngestResult : std::uint8_t {
  kAccepted,
  kDuplicate,
  kRejected,
};
class PlannerTimingAccumulator {
 public:
  explicit PlannerTimingAccumulator(std::size_t capacity);
  TimingIngestResult Ingest(
      const diagnostic_msgs::msg::DiagnosticArray& message);
  std::optional<PlannerTiming> Find(std::string_view request_id) const;
};
}
```

- [ ] Write a fake `/Car/T4/plan_motion` Action server test. Assert `MakeRequestId` deterministically returns
  `task_id + "/candidate/" + sequence` for the node-lifetime monotonic sequence. The caller registers that exact
  ID in its owning batch before calling `Evaluate`, eliminating a callback-before-map-insert race. Assert
  `mission_id=task_id`; `mission_revision=0` and ignored; `environment_mode=LUNAR_SURFACE`;
  `replace_active_request=false`; and at most one outstanding Action request. Freeze the nested `GoalRegion`
  exactly: `header.frame_id="map"`, deterministic nonempty
  `goal_id=request_id + "/goal"`, `goal_type=POINT`, `point.x/y=candidate.pose.x/y`, finite `point.z=+0.0`,
  empty `planar_region`, `position_tolerance_m` equal to the supplied finite nonnegative value,
  `has_yaw_constraint=true`, `yaw_rad=candidate.pose.yaw`, and `yaw_tolerance_rad` equal to the supplied finite
  nonnegative value. Explorer 侧 canonical outgoing `point.z` 必须逐位表现为正零，Task 10 client RED 不得
  改写或扰动 z。Goal header stamp 和其他 message stamp 才可设为零、旧值或相互乱序，且不影响准入与
  关联。另以绑定最终 server source SHA 的独立 adapter 合同证明：外来 GoalRegion 的任意 finite z 在
  final server adapter 中被忽略；该兼容性不得放宽 explorer client 的 canonical `+0.0` 输出。

- [ ] Freeze one exhaustive wrapper-plus-payload classifier with table-driven REDs. A non-null typed payload is
  classified before interpreting `WrappedResult.code`; `ABORTED` is not intrinsically retryable because the final
  server uses it for all ordinary business failures. The only accepted combinations are:

  | Action wrapper and typed payload | `PlannerEvaluationKind` | Required semantics |
  |---|---|---|
  | `SUCCEEDED` + `NEW_REFERENCE_AVAILABLE/ACTIVATE_NEW_REFERENCE/has_reference=true` | `kReachable` | nonempty admitted preview and unchanged reference |
  | `ABORTED` + `GOAL_INFEASIBLE/NO_SAFE_REFERENCE/has_reference=false` with reason `NO_PATH` or `GOAL_OUTSIDE_LOCAL_MAP` | `kExhaustiveNoPath` | the only task-exhaustion evidence |
  | `ABORTED` + `RESOURCE_EXHAUSTED/NO_SAFE_REFERENCE/has_reference=false/reason_code=TIMEOUT` | `kRetryable` | the final server's only typed resource/deadline result |
  | locally requested cancel + `CANCELED` wrapper and typed `CANCELED/NO_SAFE_REFERENCE/has_reference=false/REQUEST_CANCELED` | `kCanceled` | task-control cancellation only |
  | server unavailable, Goal rejection, or any transport terminal without a typed result | `kRetryable` | no exhaustion evidence |

  `INVALID_REQUEST`、`NUMERICAL_FAILURE`、`STALE_INPUT`、`ACTIVE_REFERENCE_INVALIDATED`、
  `SAFE_FRONTIER_REFERENCE_AVAILABLE`、`NO_KNOWN_SAFE_ROUTE`、未知 outcome/directive 底层值、
  outcome 不支持的 reason（包括任何非 `TIMEOUT` 的 `RESOURCE_EXHAUSTED` reason）、
  非本客户端请求的 typed `CANCELED`，以及 wrapper/outcome/directive/has-reference/reference/path 的任一
  矛盾组合都固定为 `kContractError`。合法 typed failure 即使由 `ABORTED` 包裹仍按 payload 分类；
  typed success 被 `ABORTED` 包裹、typed failure 被 `SUCCEEDED` 包裹、空 reason 或不受支持的
  `GOAL_INFEASIBLE` reason 都是合同错误。五种 kind 是唯一互斥状态；不得恢复
  `reachable/exhaustive_unreachable_evidence/retryable` 三个可矛盾 bool。联合不变量固定为：只有
  `kReachable` 同时携带 `reference` 和 finite nonnegative `path_length_m`；其余四种 kind 的两项都
  必须为 `nullopt`，失败态不得用伪造的 `0.0` 路径长度冒充已验证路径。

- [ ] Freeze `path_preview` numeric and resource admission. `maximum_path_preview_poses` is required, positive
  and checked before traversal/copy; over-limit is `kContractError`. Empty preview is a contract error; one finite
  pose is valid and yields exact `path_length_m=+0.0`. Only pose XY participates: promote finite input XY to
  `long double` before subtracting, require every `hypot` segment and cumulative sum finite, reject a cumulative
  result not representable as finite double, and normalize a final negative zero to `+0.0`. Z, orientations,
  `path_preview.header.stamp`, per-pose stamps, Result
  `global_map_stamp/local_map_stamp/state_stamp` and every `MotionReference` header/input/pose stamp do not affect
  classification or length. On reachable success, preserve the complete `MotionReference` byte-for-field
  unchanged, including all stamps, trajectory and hops; validation must not rewrite the saved reference.

- [ ] Construct the client with a positive steady `result_timeout`; Task 14 supplies the runtime default
  `planner_result_timeout_s=2.0`. `Evaluate` snapshots the injected `SteadyNow` deadline before sending and never
  blocks an executor callback. A production steady timer and tests call `PollTimeout`; although the final server has
  a one-second deadline, reaching the client's closed two-second deadline wins the terminal claim under the state
  mutex, moves out the optional already-accepted Goal handle, and clears the active request, Goal handle, cancel
  intent and deadline, then unlocks. If and only if that optional handle existed at the claim, the winner sends one
  best-effort `async_cancel_goal(old_handle)` scoped to that old Goal before invoking the one local Completion. If no
  handle existed, it sends no cancel at all and invokes Completion directly. Production code MUST NOT call
  `async_cancel_all_goals`: a global cancel may be processed after Completion starts a new generation and cancel the
  new Goal. Completion is exactly one `kRetryable/CLIENT_RESULT_TIMEOUT` and may immediately start the next
  `Evaluate`. Any old scoped-cancel response is bound to the terminal old generation and cannot touch the new one.
  This is a transport watchdog covering late Goal response, accepted Goal with no Result, server crash and DDS loss;
  it never reads ROS time or creates a message-freshness rule. Late goal-response, result, cancel-response, feedback
  or watchdog callbacks after a terminal claim are ignored.

- [ ] Add a complete cancel-race matrix: cancel before Goal response records intent and sends exactly one per-handle
  cancel if an accepted handle later arrives **and that evaluation is still active/non-terminal**; repeated
  `CancelActive` is idempotent; Goal rejection always claims one
  `kRetryable` transport terminal if it wins, including when cancel intent was already recorded, and Task 11 uses
  that completion only to close its pending cancel-and-wait gate; Goal rejection and
  cancel response racing each other still have one winner; rejected cancel response is not an Action terminal and
  the client continues waiting for typed Result or watchdog; accepted cancel with a late/no Result likewise waits
  for Result/watchdog; Result before cancel response wins and the late response is ignored; an actual locally
  requested typed canceled Result yields only `kCanceled`; an unsolicited typed canceled Result is
  `kContractError`. `async_cancel_goal` response alone must never release Task 11's cancel-and-wait gate. Freeze the
  exact RED sequence `CancelActive -> PollTimeout(deadline) -> late accepted Goal response`: watchdog terminal claim
  wins without an accepted handle, sends neither per-handle nor global cancel, clears cancel intent, invokes the
  single retryable Completion, and permits that Completion to start a new evaluation; the late Goal response cannot
  complete again, change kind, or send any cancel. Add three generation-safety REDs: (a) handle-before-timeout moves
  that old handle and sends only `async_cancel_goal(old_handle)` before Completion; (b) timeout-before-handle sends
  no cancel, never calls `async_cancel_all_goals`, and the late response is a no-op; (c) after Completion starts a
  new Goal, neither an old scoped-cancel response nor any other old-generation callback can cancel or mutate the new
  generation. The remote old accepted Goal in case (b) may keep the single worker briefly busy and a new Goal may be
  rejected; that rejection remains `kRetryable`. `PlanMotion` only computes a reference and does not publish/execute
  it, so this transport race cannot command the controller. Also RED the normal inverse ordering: if the accepted
  handle arrives before watchdog wins while cancel intent remains active, exactly one control-requested per-handle
  cancel is sent; watchdog cleanup remains scoped to that same old handle if timeout later wins.

- [ ] Freeze callback ownership and reentrancy. All Action, cancel and watchdog callbacks capture only a
  `weak_ptr` to shared `CallbackState`, never raw `this` and never a `FrozenPlanningCycle` directly. The
  `CallbackState` active request strongly owns its Completion; Task 11 supplies a Completion closure that strongly
  owns the `FrozenPlanningCycle`, which transitively owns its batch. Thus the only callback-to-cycle ownership path
  is `weak transport callback -> CallbackState active Completion -> FrozenPlanningCycle` while the request remains
  active. `~PlannerClient() noexcept` locks the state mutex, wins/records teardown, clears the active Completion,
  Goal handle, cancel intent and deadline without invoking user completion, and then releases client ownership.
  Copy and move construction/assignment are deleted. A callback that weak-locks state before destruction but obtains
  the mutex only after teardown must observe teardown and no-op; a callback already holding the mutex is serialized
  against the destructor. Every other
  terminal source competes through one single-winner claim. The winner moves Completion and owning captures to a
  local object, clears the member active state under the mutex, releases the mutex, and only then invokes
  completion. Completion may immediately call `Evaluate` for the next serialized Goal; if it throws, the exception
  is contained at the callback boundary and cannot undo cleanup or cause a second completion. Test reentrant
  Evaluate, throwing completion, timeout/result/cancel races, nothrow/noncopyable/nonmovable traits, and delayed
  callbacks during client/node destruction, including the weak-lock-before-teardown/mutex-after-teardown ordering.
  Feedback is deliberately not consumed and cannot affect result, timing, cancel, watchdog or freshness state.

- [ ] Write diagnostic tests using two interleaved request IDs. `PlannerTimingAccumulator` requires a positive
  capacity supplied from Task 14's `maximum_candidate_views`. Each `Ingest` accepts atomically only one
  `DiagnosticStatus` containing exactly these ten unique keys and no unknown/missing/duplicate key:
  `request_id`, `platform_type`, `environment_mode`, `planning_outcome`, `reason_code`,
  `global_elapsed_ms`, `global_call_count`, `local_elapsed_ms`, `local_call_count`, `total_elapsed_ms`;
  `request_id/reason_code` are nonempty, `platform_type` is exactly `WHEELED/LEGGED/HOPPER`,
  `environment_mode` is exactly decimal `1/2`, and `planning_outcome` is one defined decimal PlanMotion outcome;
  enum/count strings are strict full-string unsigned decimal values in range, elapsed strings are strict
  full-string finite nonnegative decimal values, and uint64 overflow,
  whitespace/sign/trailing characters, NaN/Inf or negative elapsed are rejected. `count>0,elapsed=0` is valid.
  Header/status stamps are ignored.

- [ ] Freeze cache determinism: first valid record for a request ID wins **while that record remains resident**;
  later valid duplicates or conflicts do not replace it and do not refresh its age，`Ingest` 返回 `kDuplicate`。
  New unique valid IDs are stored in FIFO
  arrival order、返回 `kAccepted`，and insertion beyond capacity evicts the oldest unique record; malformed
  messages 返回 `kRejected` and change neither records nor FIFO order（strong guarantee）。
  RED a capacity-one sequence: ingest `A1 -> kAccepted`; ingest `B -> kAccepted` and evict A; reingest a new `A2`
  as a new unique `kAccepted`, evict B, then require `Find(A)==A2` and `Find(B)==nullopt`. The bounded cache keeps no
  tombstone or unbounded seen-ID set after eviction.
  Diagnostics may arrive after Action completion and remain queryable by request ID. An Action result with no
  diagnostics retains its typed classification; rejected/missing diagnostic fields leave timing unavailable as a
  whole and global/local values are never synthesized from `total_elapsed_ms`. Diagnostics are statistics only and
  cannot invoke or change planning completion.

- [ ] Give two strictly serialized evaluations the same artificial `candidate_id` display value but different
  request IDs. Assert PlannerEvaluation preserves both fields, but documents/tests that only request_id is usable
  for owning-batch association; no PlannerClient consumer may bind by candidate hash/ID. Destroy/mutate the
  caller's CandidateView storage after Evaluate returns and prove the already-built Goal/callback display data do
  not retain a reference to it.

- [ ] Run the tests and confirm failure.

- [ ] Implement the asynchronous Action client and bounded timing cache. The client must never block an executor
  callback waiting for the Action result; completion is delivered only by the single-winner terminal path after
  active state is cleared. It copies candidate pose into the exact GoalRegion and candidate_id into display output,
  but treats caller-provided request_id as the sole correlation key and never retains the CandidateView reference
  past `Evaluate` return. Implement manifest/CMake dependency consistency for `rclcpp_action`,
  `diagnostic_msgs`, `lunar_planning_msgs` and fake-server `action_msgs` test support. Preserve returned references
  unchanged and keep all transport callbacks weak-owned.

- [ ] After fake-server GREEN, run at least one lightweight test against the integrated final
  `PurePlanMotionServer` (or an equivalent static contract tied to its integrated source SHA) proving
  `ABORTED+NO_PATH`, `ABORTED+TIMEOUT`, typed invalid/contract failure and diagnostics-before-terminal behavior.
  If `bf1bb79` or a verified successor is still not an ancestor of the current branch, stop as BLOCKED; do not run
  the commit step below and do not count fake-only tests as completion.

- [ ] Build, test and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros \
  --event-handlers console_direct+
cd ../../
git add pure_planner/ros2_ws/src/lunar_pure_exploration_ros
git commit -m "feat(pure-exploration): evaluate candidates through pure planner"
```

## Task 11: Build the Exploration ROS Node and Planning Cycle

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/exploration_node.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/src/main.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/package.xml`

**Interfaces consumed:**

| Topic/Action | Type | Role |
|---|---|---|
| `/Car/T3/mapping/global_overview` | `nav_msgs/msg/OccupancyGrid` | 全局探索图，原生 `-1/0..100` |
| `/Car/T3/localization/odometry` | `nav_msgs/msg/Odometry` | 最新车体位姿输入 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 最新直接 `map -> odom`；`odom -> base` 只取 Odometry |
| `/Car/T4/exploration/task` | `PureExplorationTask` | 任务及控制命令 |
| `/Car/T4/plan_motion` | `PlanMotion` | 候选真实可达性和路径代价 |
| `/Car/T4/planning/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 全局/局部规划耗时 |

**Interfaces produced:**

| Topic | Type | QoS |
|---|---|---|
| `/Car/T4/execution/motion_reference` | `lunar_planning_msgs/msg/MotionReference` | reliable, keep last 1 |
| `/Car/T4/execution/cancel` | `std_msgs/msg/String` | reliable, keep last 10; payload is active `plan_id` |
| `/Car/T4/exploration/status` | `PureExplorationStatus` | reliable, transient local, keep last 1 |

**Owning cycle interface:**

```cpp
namespace lunar::pure_exploration_ros {
struct FrozenGlobalMapContent {
  lunar::pure_exploration::GridGeometry geometry;
  std::vector<std::int8_t> data;
  bool EqualsGeometryAndData(const nav_msgs::msg::OccupancyGrid& latest) const;
};
class FrozenCandidateBatch {
 public:
  FrozenCandidateBatch(
      FrozenGlobalMapContent global_map_content,
      std::vector<lunar::pure_exploration::FrontierCluster> frontiers,
      std::vector<lunar::pure_exploration::CandidateView> candidates);
  std::span<const lunar::pure_exploration::FrontierCluster> frontiers() const;
  std::span<const lunar::pure_exploration::CandidateView> candidates() const;
  void RegisterRequest(std::string request_id, std::size_t candidate_index);
  std::optional<std::size_t> CandidateIndexForRequest(
      std::string_view request_id) const;
  bool GlobalMapContentEquals(
      const nav_msgs::msg::OccupancyGrid& latest) const;

 private:
  const FrozenGlobalMapContent global_map_content_;
  const std::vector<lunar::pure_exploration::FrontierCluster> frontiers_;
  const std::vector<lunar::pure_exploration::CandidateView> candidates_;
  std::unordered_map<std::string, std::size_t> request_to_candidate_index_;
};
using FrozenCandidateBatchPtr = std::shared_ptr<FrozenCandidateBatch>;
struct FrozenReachableCandidate {
  lunar::pure_exploration::PlannedCandidate metrics;
  lunar_planning_msgs::msg::MotionReference reference;
};
class FrozenPlanningCycle {
 public:
  FrozenPlanningCycle(
      FrozenCandidateBatchPtr batch,
      lunar::pure_exploration::Pose2 frozen_robot_pose,
      double frozen_resolution_m,
      std::vector<lunar::pure_exploration::CandidateGain> complete_gains,
      std::vector<lunar::pure_exploration::RankedCandidate> coarse_order);
  const FrozenCandidateBatchPtr& batch() const;
  lunar::pure_exploration::Pose2 frozen_robot_pose() const;
  double frozen_resolution_m() const;
  std::span<const lunar::pure_exploration::CandidateGain> complete_gains() const;
  std::span<const lunar::pure_exploration::RankedCandidate> coarse_order() const;
  std::size_t coarse_cursor() const;
  std::vector<std::size_t> TakeNextCandidateIndices();  // at most 16
  void AddReachable(FrozenReachableCandidate result);
  std::span<const FrozenReachableCandidate> reachable() const;

 private:
  const FrozenCandidateBatchPtr batch_;
  const lunar::pure_exploration::Pose2 frozen_robot_pose_;
  const double frozen_resolution_m_;
  const std::vector<lunar::pure_exploration::CandidateGain> complete_gains_;
  const std::vector<lunar::pure_exploration::RankedCandidate> coarse_order_;
  std::size_t coarse_cursor_{0U};
  std::vector<FrozenReachableCandidate> reachable_;
};
using FrozenPlanningCyclePtr = std::shared_ptr<FrozenPlanningCycle>;
}
```

The content comparison excludes header stamp and has no revision/version field. `CandidateView.frontier_index`
indexes only `batch.frontiers()`; construction rejects any out-of-range candidate/frontier association before
moving the vectors into immutable storage. `RegisterRequest` validates a unique nonempty request_id and in-range candidate
index under the node state mutex before each request is sent. Callback association uses only
`CandidateIndexForRequest(returned_request_id)`.
`FrozenPlanningCycle` 是 ROS 包内部 owning context，不属于 core 稳定 API。构造时验证 batch 非空、
pose/resolution 有限有效、complete gains 唯一完整覆盖 batch candidates、coarse order 是其全部正增益
owning indices 的唯一排列；cursor/reachable 只在节点状态 mutex 下修改。它共同拥有 batch、唯一 gain
authority、冻结 pose/resolution、完整 coarse order/cursor 和 reachable path/reference rows；这些内容
不能放在 latest cache 或会随 active batch 替换而销毁的旁路容器。

Task 11 只按 `PlannerEvaluationKind` 推进：`kReachable` 才加入 reachable rows；
`kExhaustiveNoPath` 才推进当前候选的任务穷尽证据；`kRetryable` 保留任务、当前 cycle 和 cursor 位置，
等待重试且不得完成；`kCanceled` 只完成已经在等待的 PAUSE/CANCEL/replacement START 控制过渡；
`kContractError` 进入稳定合同 `ERROR`。节点不得从 `reason_code` 重新推断 kind，也不得恢复三个 bool。
在当前月表合同中，穷尽 kind 的两个且仅两个 typed reason 是 `NO_PATH` 和
`GOAL_OUTSIDE_LOCAL_MAP`。若 PAUSE/CANCEL/replacement START 已先进入 cancel-and-wait，则该旧请求的
任一 single-winner terminal completion（包括 Goal reject 或 client watchdog 的 `kRetryable`）只关闭
transport wait gate，再完成控制过渡；不得重新排队旧任务候选或让 late callback 影响新任务。

- [ ] Write a node test with in-process publishers and fake planner server. Feed one valid START task, map,
  odometry and TF, then assert the node moves the complete frontier/candidate vectors and frozen global-map
  geometry/data into one owning FrozenCandidateBatch, then moves that batch, complete CandidateGain table, frozen
  robot pose/resolution and complete coarse order into one FrozenPlanningCycle. It sends at most 16 planner goals
  and publishes one selected motion reference. Every CandidateView.frontier_index must resolve within
  `cycle.batch()->frontiers()`.

- [ ] Add independent missing-input cases for no global map, no odometry and no usable `map -> odom -> base_link` composition. Each case must publish `WAITING_FOR_INPUT`, retain the task indefinitely without age checks, and never report completion or ERROR.

- [ ] Add task-structure cases: START requires nonempty task ID, `header.frame_id == "map"` and a valid simple boundary; PAUSE/RESUME/CANCEL ignore boundary contents. Add OccupancyGrid adapter cases requiring finite origin position and a finite nonzero-norm normalized quaternion, then convert it to planar yaw. Add a map-geometry case proving a changed origin, dimensions or resolution rebuilds the logical task raster from content without consulting a revision or timestamp.

- [ ] Assert the exact cycle order: freeze latest map/pose content; rasterize task; detect frontiers; generate
  candidates; calculate one gain row per owning candidate index; create the owning batch; coarse-rank the complete
  positive-gain index set against that same batch, complete gains and frozen pose; create one FrozenPlanningCycle
  owning all of them plus frozen resolution; use its monotonic cursor to take the next
  `[cursor,min(cursor+16,count))` indices; compute request_id; insert through
  `cycle->batch()->RegisterRequest(request_id, candidate_index)`; request real paths strictly serially; associate callback by
  request_id; continue with the next 16 after a conclusive all-`kExhaustiveNoPath` batch; final-rank reachable
  index/path rows
  against `cycle->batch()/complete_gains()/frozen_robot_pose()/frozen_resolution_m()` and completed-goal positions;
  build ActiveGoal from that same batch; publish its unchanged saved reference. Assert 33 candidates produce
  16/16/1 with no repeat/omission and
  unselected references are comparison-only. The current Action has no FAST/ANYTIME field; do not invent one and
  use the planner's fixed one-second anytime behavior. The explorer's separate
  `planner_result_timeout_s=2.0` steady transport watchdog remains active around that server behavior and is not a
  freshness timeout.

- [ ] Add lifetime/race tests with a delayed fake server and `weak_ptr<FrozenPlanningCycle>`. Transport callbacks
  capture only `weak_ptr<CallbackState>`; the active Completion stored in CallbackState strongly captures the same
  cycle, which transitively owns its FrozenCandidateBatch. Reset the node's active cycle while that request remains
  active and assert both cycle and batch remain live through this indirect ownership path; after terminal Completion
  returns, or after client teardown clears the active Completion, assert both expire. Inspect callback captures so
  no transport lambda directly owns a cycle. Map/
  odometry callbacks may replace latest cache but may not mutate cycle batch/gains/pose/resolution/order/cursor/
  reachable rows or rebind an in-flight CandidateView to latest content. Exercise PlannerClient's reentrant
  completion path: after it has cleared its active state and released its mutex, the node may synchronously send the
  next candidate while the previous completion's local cycle owner remains alive. A throwing completion or
  client/node destruction cannot revive state; late goal/result/cancel/watchdog callbacks are ignored.

- [ ] Add a frozen-authority race RED: after the first planner result and before the second, publish a different
  odometry pose and a map with different resolution/content. Let both callbacks finish and assert FinalRank receives
  the original cycle `complete_gains()`, `frozen_robot_pose()` and `frozen_resolution_m()`, and uses the original
  coarse order/cursor plus accumulated reachable rows. Latest cache is used only to decide whether a new cycle must
  be built before completion; it never changes ranking inputs inside the in-flight cycle.

- [ ] Add request-association tests with duplicate/artificially colliding candidate display IDs and a mapping
  order intentionally different from candidate-vector order. Resolve only `PlannerEvaluation.request_id` through
  `cycle->batch()->CandidateIndexForRequest`; verify the index and then read CandidateView/frontier from that cycle
  batch.
  Unknown request_id is a contract ERROR. Never associate by PlannerEvaluation.candidate_id, frontier/candidate
  hash, callback order, current candidate array, or current latest-map content.

- [ ] Add tests showing that ordinary maps and a newly higher score do not interrupt the committed goal, while
  occupied goal/footprint, frontier disappearance, zero recomputed gain and conclusive `kExhaustiveNoPath` do.
  Assert PAUSE/CANCEL/replacement START use two distinct cancellation surfaces: publish execution cancel for an
  already issued `plan_id`, call `CancelActive` for any in-flight Action and wait for the PlannerClient single-winner
  terminal completion, then transition or send the next Goal. Goal-response-before/after-cancel, repeated cancel,
  cancel response rejected/accepted, result-before-cancel-response and no-result watchdog cases all obey Task 10's
  matrix; an `async_cancel_goal` response is never treated as terminal. PAUSE preserves task statistics and
  persistent failures; RESUME rebuilds from current content; CANCEL returns IDLE; replacement START clears
  old-task statistics/failures only after cancellation completes.

- [ ] Add candidate-exhaustion tests: after all candidates in the current owning batch have conclusive
  `kExhaustiveNoPath` (`NO_PATH` or `GOAL_OUTSIDE_LOCAL_MAP`),
  evaluate the next 16; only after every positive-gain candidate in that batch has conclusive no-route evidence
  or valid persistent failure suppression may the node consider completion. TIMEOUT, cancel, planner/transport
  error, invalid response and unavailable server remain non-terminal; contract errors enter `ERROR` rather than
  retry or completion. Immediately before
  `COMPLETED_NO_REACHABLE_FRONTIER`, compare latest global-map geometry/data by value against
  `cycle->batch()->GlobalMapContentEquals(latest)`; if false, discard the conclusion and build a new owning cycle
  from latest cache.
  Stamp/revision/version never participates.

- [ ] Add failure-memory integration cases. Only exhausted execution-stage stuck replans call
  `RecordPersistentFailure(...,kExecutionReplansExhausted,...)`; candidate-evaluation NO_PATH remains batch-local,
  and TIMEOUT/cancel/reject/
  transport/contract failures never persist. Every accepted START invokes `BeginTask` once after cancel-and-wait and
  clears entries even when task_id is reused. PAUSE/RESUME and ordinary map updates retain the same scope. A
  geometry-changed raster conservatively invalidates matching entries; a same-geometry distant data change does not.

- [ ] Publish coverage only from `CalculateCoverage`. Assert empty task-cell denominator publishes finite 0.0 and
  fully known coverage publishes 1.0 without either value causing a state transition. The only normal completion
  gate remains exhaustive positive-gain candidate evidence/persistent suppression plus latest-map content equality;
  no ROS callback compares coverage to a threshold.

- [ ] Validate every reachable reference before storing or publishing it. Task 10's required
  `maximum_path_preview_poses` bounds the complete preview; the existing required
  `maximum_executable_path_points` independently bounds `trajectory.points` for WHEELED/LEGGED and `hops` for
  HOPPER before copying, iteration or ProgressMonitor reset. Exact limit succeeds, over-one maps to resource
  `ERROR`; empty trajectory/hops remains governed by the platform/reference contract. The saved and published
  `MotionReference` remains exactly the PlannerClient copy, including header, input time, path pose, trajectory
  and hop stamps. Feedback is ignored.

- [ ] Inject `std::length_error` independently from task rasterization, candidate position probes, candidate
  views, collision work, visibility work, path-preview poses, executable trajectory/hops, failure-entry count,
  per-entry failure patch work and total retained failure patch cells. Each maps to ROS state `ERROR` with a stable
  resource reason and
  never to an empty candidate set, zero gain, candidate exhaustion or `COMPLETED_NO_REACHABLE_FRONTIER`.

- [ ] Run the test and observe missing-node failure.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros \
  --ctest-args -R test_exploration_node --output-on-failure
```

- [ ] Implement `ExplorationNode` with a mutually exclusive state mutex, asynchronous queue and shared owning
  FrozenPlanningCycle. Subscription callbacks replace latest cached content immediately without timestamp
  comparisons and do not mutate any in-flight cycle or its batch. Register request association through
  `cycle->batch()` before `PlannerClient::Evaluate`; pass a Completion closure that strongly owns the cycle into
  `PlannerClient`, whose transport callbacks remain weak-only. Only START creates a failure-memory scope; PAUSE
  retains it; CANCEL discards the task after cancel-and-wait completes.

- [ ] Run ROS package tests and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros \
  --event-handlers console_direct+
cd ../../
git add pure_planner/ros2_ws/src/lunar_pure_exploration_ros
git commit -m "feat(pure-exploration): run frontier selection and planning cycle"
```

## Task 12: Add Arrival, Stuck Recovery, and Replanning Behavior

**Files:**

- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/exploration_node.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/test/test_execution_monitor.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt`

**Behavior contract:**

```text
position_tolerance = max(0.5 * global_map_resolution,
                         0.25 * platform.platform_width_m)
yaw_tolerance = 11.25 degrees
stuck_window = 30 steady-clock seconds
minimum_progress = 0.2 meters
maximum_replans_per_candidate = 2
```

- [ ] Write a failing execution-monitor test with an injectable steady-clock function. Assert arrival only when
  both position and yaw tolerances pass, including wrapped yaw across `-pi/pi`. Use two literal
  `(global_resolution_m, platform_width_m)` fixtures so `max(0.5*resolution,0.25*width)` is won once by each
  branch. For each, hand-supply position error exactly equal to the resulting tolerance and
  `nextafter(tolerance,+inf)`. Hand-supply yaw error exactly `11.25 deg` and
  `nextafter(11.25 deg,+inf)`, plus one position-only failure and one yaw-only failure. Assert the exact cases
  release ActiveGoal and every outer/one-failed case preserves it; expected values must not call the production
  tolerance helper.

- [ ] Assert the first and second stuck detections cancel the current `plan_id` and replan the same committed candidate; the third marks the candidate failed, releases commitment and selects another candidate.

- [ ] Build references whose global `path_preview` reaches the final candidate but whose executable trajectory
  ends at an intermediate local-segment endpoint. Assert reaching the segment endpoint enters rolling replan for
  the same committed candidate, does not increment `replan_count` for stuck recovery, and only final-candidate
  position/yaw arrival releases commitment. Assert execution cancel, Action cancel-and-wait and the next
  `replace_active_request=false` Goal occur in that order.

- [ ] Assert map content invalidation cancels immediately, while a changed message stamp with identical map content does nothing. Assert odometry covariance has no effect.

- [ ] Run the test and confirm failure.

- [ ] Integrate `ProgressMonitor`, executable-segment endpoint extraction and final-candidate arrival checks into
  the node. Reset the progress window on each accepted reference and on each qualifying 0.2 m increase in
  historical executable-segment arc length. Keep the candidate committed across rolling replans and its two
  stuck-recovery replans.

- [ ] Run all ROS tests and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros \
  --event-handlers console_direct+
cd ../../
git add pure_planner/ros2_ws/src/lunar_pure_exploration_ros
git commit -m "feat(pure-exploration): monitor arrival and recover from stuck motion"
```

## Task 13: Publish Current Goal, Frontiers, Coverage, and Planner Timing

**Files:**

- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/marker_builder.hpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/src/marker_builder.cpp`
- Create: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_outputs.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/exploration_node.hpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt`

**Additional outputs:**

| Topic | Type | Contents |
|---|---|---|
| `/Car/T4/exploration/current_goal` | `geometry_msgs/msg/PoseStamped` | 当前承诺候选位姿 |
| `/Car/T4/exploration/frontiers` | `visualization_msgs/msg/MarkerArray` | 所有前沿簇、候选和选中状态 |
| `/Car/T4/exploration/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 覆盖率、候选统计、失败原因、规划耗时 |

- [ ] Write a failing output test. Assert status publishes after task command, map recomputation, candidate selection, replan and terminal transition; `coverage_ratio` remains statistical and can be below 1.0 at normal completion.

- [ ] Assert every status message contains the full frozen message contract: polygon/raster/free/occupied/unknown/outside-map areas, coverage, frontier/candidate/reachable/failed/completed/replan counts, current plan ID, current goal and pause-adjusted active elapsed time.

- [ ] Assert diagnostic key/value output includes `task_id`, `state`, `reason_code`, `active_candidate_id`, `request_id`; call count, last and accumulated duration for frontier detection and information gain; coarse/final ranking durations; snapshot-to-goal decision duration; candidate-planning, rolling-replanning and stuck-replanning counts; and the correlated planner keys `global_elapsed_ms`, `global_call_count`, `local_elapsed_ms`, `local_call_count`, `total_elapsed_ms`.

- [ ] Assert planner timings are copied only from a strictly valid ten-key diagnostic record with the matching
  request ID and accumulated separately for candidate validation versus execution replanning. A missing/rejected/
  evicted whole record leaves planner timing `unavailable`; missing global/local keys reject the record atomically,
  are never synthesized from total time, and do not invalidate an otherwise valid planning result.

- [ ] Run the output test and confirm failure.

- [ ] Implement transient-local status/current-goal publishers, marker namespaces with stable IDs, and a bounded diagnostic history. Use received message headers for display only; timing values remain steady-clock measurements supplied by the planner.

- [ ] Run ROS tests and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros
colcon --log-base "$COLCON_LOG_BASE" test --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" --packages-select lunar_pure_exploration_ros \
  --event-handlers console_direct+
cd ../../
git add pure_planner/ros2_ws/src/lunar_pure_exploration_ros
git commit -m "feat(pure-exploration): publish exploration state and planner timings"
```

## Task 14: Add Runtime Configuration and Launch Integration

**Files:**

- Create: `pure_planner/config/pure_exploration.yaml`
- Create: `pure_planner/launch/pure_exploration.launch.py`
- Create: `pure_planner/tests/launch/test_pure_exploration_launch.py`
- Modify: `pure_planner/README.md`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt`
- Modify: `pure_planner/ros2_ws/src/lunar_pure_exploration_ros/package.xml`

**Runtime defaults:**

```yaml
pure_exploration:
  ros__parameters:
    global_occupied_threshold: 50
    maximum_task_raster_cells: 1048576
    sensor_range_m: 10.0
    sensor_fov_deg: 90.0
    yaw_offsets_deg: [-45.0, -22.5, 0.0, 22.5, 45.0]
    candidates_per_planning_batch: 16
    score_weights:
      information_gain: 0.60
      global_path_length: 0.30
      heading_change: 0.05
      revisit: 0.05
    stuck_timeout_s: 30.0
    minimum_progress_m: 0.2
    maximum_replans_per_candidate: 2
    goal_yaw_tolerance_deg: 11.25
    planner_result_timeout_s: 2.0
    global_map_topic: /Car/T3/mapping/global_overview
    odometry_topic: /Car/T3/localization/odometry
    tf_topic: /tf
    task_topic: /Car/T4/exploration/task
    planner_action: /Car/T4/plan_motion
    planner_diagnostics_topic: /Car/T4/planning/diagnostics
    motion_reference_topic: /Car/T4/execution/motion_reference
    execution_cancel_topic: /Car/T4/execution/cancel
```

`maximum_position_probes`、`maximum_candidate_views`、`maximum_collision_work_units`、
`maximum_visibility_work_units`、`maximum_path_preview_poses`、`maximum_executable_path_points`、
`maximum_failure_entries`、`maximum_failure_patch_cells_per_entry` 和
`maximum_failure_total_patch_cells` 不出现在默认 YAML，也没有代码默认值。launch 将九者声明为
无默认值的必填正整数参数并传给节点；缺失时 launch 明确
报配置 `ERROR`，零值/负值时节点进入
配置 `ERROR`，两者都不得继续启动探索周期。测试可为其小型固定 fixture 显式
传入 `64`、`64`、`4096`、`4096`、`64`、`64`、`8`、`256`、`1024`，这些值只属于测试，不是生产或 Orin
能力声明。生产值必须在
Humble 压力测试和 Orin 时延/内存验收后由批准的部署 profile 显式传入。
`maximum_path_preview_poses` 独立约束完整全局 preview；`maximum_executable_path_points` 复用于
WHEELED/LEGGED 的 `trajectory.points` 和 HOPPER 的 `hops`，不另设两个容易漂移的执行序列预算。
`PlannerTimingAccumulator` 的 FIFO capacity 复用 `maximum_candidate_views`，不增加第十个资源参数。
`planner_result_timeout_s` 是有限正数，仓库运行时默认精确为 `2.0`；它只驱动 Task 10 的客户端
steady transport watchdog，不表示 Goal、Result、地图、状态、diagnostics 或 reference freshness。
批准的部署 profile 必须显式导出全部九个正整数资源值，包括
`PURE_EXPLORATION_MAXIMUM_PATH_PREVIEW_POSES` 与
`PURE_EXPLORATION_MAXIMUM_EXECUTABLE_PATH_POINTS`；二者缺失都属于部署前置条件失败。
visibility budget 是每次 Evaluate 上限；Task 14 的 profile/schema 和 README 还必须要求整周期证据：
以受检整数计算 `maximum_candidate_views * maximum_visibility_work_units`，在同一组值下记录完整候选
周期的最大 elapsed time 与 peak RSS。测试 fixture 的 `64*4096` 只验证接线，不能充当生产验收或
仓库默认。三个独立 failure fixture 按设计第13节固定 schema 生成仓库外、发布后不可变的 JSON
artifact。profile 除原 validated-limit/elapsed/RSS/over-rejected 字段外，还必须提供：

```text
PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_PATH
PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_SHA256
PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_EVIDENCE_PATH
PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_EVIDENCE_SHA256
PURE_EXPLORATION_FAILURE_TOTAL_PATCH_EVIDENCE_PATH
PURE_EXPLORATION_FAILURE_TOTAL_PATCH_EVIDENCE_SHA256
PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_ENVIRONMENT
PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_IMAGE
```

三条 path 必须是仓库外绝对文件路径，三个 SHA-256 是对应实际 JSON bytes 的小写十六进制 digest，
path/digest 均唯一。entry fixture 以小 patch 触达 entry exact/over；per-entry fixture 以一个未裁剪
AABB 触达 N/N-1；total fixture 以多个已准入 patch 触达 total exact/over。每个 artifact 都包含完整
三 limit 配置、固定 fixture_kind/target 映射、reached/over-one/rejected、独立 elapsed/RSS、authority
environment/image 和 generated_at 记录。profile 导出的 validated limit、elapsed/RSS/rejected/
authority 必须与相应 artifact 逐值相等。generated_at 不导出为准入值，不解析日期且不检查年龄。

- [ ] Write a failing launch test that starts a fake planner Action server plus `pure_exploration.launch.py` with
  `platform_selector:=wheel` and all nine test-only positive resource limits, waits for every expected
  subscription/publisher/action client, and asserts no old training/PPO packages are loaded.

- [ ] Add configuration validation tests: score weights are finite and nonnegative and sum to 1.0 for the
  shipped defaults; sensor range/FOV and tolerances are finite positive; `maximum_task_raster_cells` is exactly
  the approved positive default `1048576`; batch size is 16; yaw offsets contain exactly the approved five
  values; all Topic names are absolute and match the table. Assert footprint, clearance, platform dimensions,
  derived frontier length, candidate spacing, standoff, revisit radius, failure radius, planner goal tolerance and
  all nine candidate/gain/preview/progress/failure-memory resource limits do not appear in exploration YAML.
  Assert `planner_result_timeout_s` is exactly `2.0`, finite and positive, is wired through steady
  `PollTimeout`, and no source compares it with a ROS/message stamp. Assert omitted
  arguments emit explicit launch configuration `ERROR`, zero/negative values put the node in configuration
  `ERROR`, and explicit
  positive test-only values are accepted; no test value is documented as a production default. Validate the
  external deployment-profile schema requires a checked visibility-product value plus measured whole-cycle
  elapsed/peak-RSS evidence matching the exact candidate-view and per-Evaluate limits. It must also require all
  three validated failure-limit fields, three artifact absolute-path/SHA pairs, exact configured-limit triples,
  strict JSON schema/kind/target/reached/over semantics, three independent finite-positive saturation elapsed/RSS
  pairs, three true over-limit-rejected flags and one shared nonempty authority environment/image pair. Read each
  artifact's bytes once, validate its SHA before strict JSON decode, reject duplicate/unknown/missing keys and
  duplicate path/digest. Mutation REDs must reject: swapping entry/per-entry elapsed+RSS, replacing any measurement
  with an old positive value, swapping complete artifact path/hash records, swapping path without hash, changing
  bytes without digest, and stale configured limits. An old/non-current generated_at string with otherwise valid
  bytes must still pass, proving generation time is record-only and not a freshness gate.

- [ ] Run the launch test and confirm failure.

```bash
source /opt/ros/humble/setup.bash
source "$COLCON_INSTALL_BASE/setup.bash"
python3 -m pytest -q pure_planner/tests/launch/test_pure_exploration_launch.py
```

- [ ] Implement the launch file using installed package shares; declare `platform_selector`, `platform_config`,
  `exploration_config`, `use_sim_time` plus the nine no-default required resource-limit arguments, and start exactly
  one exploration node. Document launch and task-publish commands in UTF-8 README, requiring measured positive
  work-limit values, whole-cycle product evidence and all three external JSON artifact path/SHA pairs without
  publishing unverified production numbers or copying generated evidence into the repository.

- [ ] Build, source, run the launch test and commit.

```bash
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build --base-paths src ../../ros2_ws/src/lunar_planning_msgs --build-base "$COLCON_BUILD_BASE" --install-base "$COLCON_INSTALL_BASE" \
  --packages-up-to lunar_pure_exploration_ros
source "$COLCON_INSTALL_BASE/setup.bash"
cd ../../
python3 -m pytest -q pure_planner/tests/launch/test_pure_exploration_launch.py
git add pure_planner/config/pure_exploration.yaml \
        pure_planner/launch/pure_exploration.launch.py \
        pure_planner/tests/launch/test_pure_exploration_launch.py \
        pure_planner/README.md \
        pure_planner/ros2_ws/src/lunar_pure_exploration_ros
git commit -m "feat(pure-exploration): add isolated runtime launch"
```

## Task 15: Verify Synthetic End-to-End Exploration and Completion Semantics

**Files:**

- Create: `pure_planner/tests/scenarios/test_synthetic_exploration.py`
- Create: `pure_planner/tests/scenarios/fixtures/concave_region.yaml`
- Create: `pure_planner/tests/scenarios/fixtures/unreachable_frontier.yaml`
- Create: `pure_planner/tests/scenarios/fixtures/map_growth.yaml`
- Modify: `pure_planner/README.md`

**Scenario contract:**

- Concave task region with reachable frontier produces a reference and remains EXECUTING.
- A frontier connected through a point-free but platform-infeasible narrow passage survives WFD and candidate prefiltering but is rejected by the real planner; later candidates continue to be evaluated.
- Map growth near a failed candidate changes local content and permits reevaluation.
- Fully known task region completes with coverage 1.0.
- Partly unknown region whose remaining frontiers are all unreachable completes below coverage 1.0 only after exhaustive planner evidence.
- Timestamp-only, map-version-only and covariance-only mutations produce identical behavior.

- [ ] Write the synthetic test harness with an in-process deterministic fake `PlanMotion` server whose response table is keyed by candidate goal cell. Record all Action requests, references, cancellation messages and status transitions. Model success, conclusive `NO_PATH`, `TIMEOUT`, cancel, invalid success payload and a two-segment lunar reference.

- [ ] Run it before adding fixtures and observe fixture-not-found failure.

```bash
source /opt/ros/humble/setup.bash
source "$COLCON_INSTALL_BASE/setup.bash"
python3 -m pytest -q pure_planner/tests/scenarios/test_synthetic_exploration.py
```

- [ ] Add the three fixed fixtures and assertions for all six scenario contracts. Add explicit checks that TIMEOUT
  cannot complete a task and that two local execution segments reach one committed final candidate without
  consuming stuck retries. Require every test to finish within 20 steady-clock seconds and print the ordered
  candidate/request IDs on failure.

- [ ] Run the scenario suite three consecutive times to prove deterministic selection.

```bash
for run_index in 1 2 3; do
  python3 -m pytest -q pure_planner/tests/scenarios/test_synthetic_exploration.py
done
```

- [ ] Document the observed completion semantics, coverage fields and planner timing fields in the README, clearly separating synthetic evidence from vehicle evidence.

- [ ] Commit scenario evidence sources only; do not commit generated logs.

```bash
git add pure_planner/tests/scenarios pure_planner/README.md
git commit -m "test(pure-exploration): verify deterministic exploration scenarios"
```

## Task 16: Run Final Repository, Build, Interface, and Source Audits

**Files:**

- Modify only if a check identifies a task-related defect; otherwise no source changes.

- [ ] Run repository boundary checks required by `AGENTS.md`.

```bash
cd /home/kai/WS/lunar-navigation-orin-pure-exploration
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
```

- [ ] Build every pure package in the authoritative Humble environment.

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
cd pure_planner/ros2_ws
colcon --log-base "$COLCON_LOG_BASE" build \
  --base-paths src ../../ros2_ws/src/lunar_planning_msgs \
  --build-base "$COLCON_BUILD_BASE" \
  --install-base "$COLCON_INSTALL_BASE" \
  --packages-up-to lunar_pure_exploration_ros \
  --event-handlers console_direct+
```

- [ ] Run all pure package tests and inspect every result.

```bash
colcon --log-base "$COLCON_LOG_BASE" test \
  --build-base "$COLCON_BUILD_BASE" \
  --install-base "$COLCON_INSTALL_BASE" \
  --packages-select lunar_pure_exploration_msgs \
                    lunar_pure_exploration_core \
                    lunar_pure_exploration_ros \
                    lunar_pure_planner_core \
                    lunar_pure_planner_ros \
  --event-handlers console_direct+
colcon test-result --test-result-base "$COLCON_BUILD_BASE" --verbose
```

- [ ] Run isolation, interface, launch and scenario tests.

```bash
cd /home/kai/WS/lunar-navigation-orin-pure-exploration
python3 -m pytest -q \
  pure_planner/tests/exploration \
  pure_planner/tests/launch \
  pure_planner/tests/scenarios
```

- [ ] Audit forbidden validation and old-runtime coupling. The first command must return no matches; the second may match documentation assertions only, not production dependencies.

```bash
rg -n "stamp.*(reject|stale)|map_revision|covariance.*reject|freshness|observation_time" \
  pure_planner/ros2_ws/src/lunar_pure_exploration_*/include \
  pure_planner/ros2_ws/src/lunar_pure_exploration_*/src
rg -n "PPO|policy_runtime|training|lunar_exploration|lunar_navigation_learning" \
  pure_planner/ros2_ws/src/lunar_pure_exploration_*/include \
  pure_planner/ros2_ws/src/lunar_pure_exploration_*/src \
  pure_planner/launch/pure_exploration.launch.py
```

- [ ] Audit the exact Topic graph and message types from a bounded launch with the real
  `lunar_pure_planner_node` and exploration client. Assert exactly one `/Car/T4/plan_motion` server, matching
  `lunar_planning_msgs` provider prefix/Action definition, ten-key diagnostics, fixed lunar environment Goal,
  cancel-and-wait and serialized request behavior. Keep fake-server scenarios for determinism only; they do not
  replace this real compatibility test. This launch has no repository work-limit defaults: before running it,
  the operator must point `PURE_EXPLORATION_DEPLOYMENT_PROFILE` to an external profile produced by completed
  Orin resource validation. The profile must explicitly mark that evidence and export all nine positive resource
  limits;
  it must also export the checked visibility-product and the measured whole-cycle maximum elapsed time and peak
  RSS for those exact limits. Absence is a failed prerequisite, not an invitation to invent values. A passing
  single-candidate `Evaluate` benchmark is not whole-cycle certification: the Orin acceptance run must exercise
  one complete candidate cycle with up to `maximum_candidate_views` evaluations, each bounded by
  `maximum_visibility_work_units`. Failure memory uses three independent saturation fixtures with other limits set
  compatibly: one reaches maximum entries, one reaches the per-entry uncut-AABB limit, and one reaches total retained
  patch cells. Each proves exact success and over-one rejection and records its own elapsed/peak retained RSS in a
  separate immutable external JSON artifact; no single fixture is required to reach three potentially incompatible
  limits simultaneously. The profile carries each artifact's absolute path plus SHA-256 of its actual bytes. The
  validator binds the slot, kind, all three configured limits, reach semantics and measurements from those bytes;
  profile-only positive measurements are not evidence.

```bash
source "$COLCON_INSTALL_BASE/setup.bash"
: "${PURE_EXPLORATION_DEPLOYMENT_PROFILE:?set the approved Orin profile path}"
test -f "$PURE_EXPLORATION_DEPLOYMENT_PROFILE"
set -a
. "$PURE_EXPLORATION_DEPLOYMENT_PROFILE"
set +a
test "${PURE_EXPLORATION_ORIN_VALIDATED:-}" = 1
for required_limit in \
  PURE_EXPLORATION_MAXIMUM_POSITION_PROBES \
  PURE_EXPLORATION_MAXIMUM_CANDIDATE_VIEWS \
  PURE_EXPLORATION_MAXIMUM_COLLISION_WORK_UNITS \
  PURE_EXPLORATION_MAXIMUM_VISIBILITY_WORK_UNITS \
  PURE_EXPLORATION_MAXIMUM_PATH_PREVIEW_POSES \
  PURE_EXPLORATION_MAXIMUM_EXECUTABLE_PATH_POINTS \
  PURE_EXPLORATION_MAXIMUM_FAILURE_ENTRIES \
  PURE_EXPLORATION_MAXIMUM_FAILURE_PATCH_CELLS_PER_ENTRY \
  PURE_EXPLORATION_MAXIMUM_FAILURE_TOTAL_PATCH_CELLS; do
  test "${!required_limit:-0}" -gt 0
done
for required_evidence in \
  PURE_EXPLORATION_VALIDATED_VISIBILITY_PRODUCT \
  PURE_EXPLORATION_WHOLE_CYCLE_MAX_ELAPSED_MS \
  PURE_EXPLORATION_WHOLE_CYCLE_PEAK_RSS_MIB \
  PURE_EXPLORATION_VALIDATED_FAILURE_MAXIMUM_ENTRIES \
  PURE_EXPLORATION_FAILURE_ENTRIES_SATURATION_ELAPSED_MS \
  PURE_EXPLORATION_FAILURE_ENTRIES_SATURATION_PEAK_RSS_MIB \
  PURE_EXPLORATION_FAILURE_ENTRIES_OVER_LIMIT_REJECTED \
  PURE_EXPLORATION_VALIDATED_FAILURE_MAXIMUM_PATCH_CELLS_PER_ENTRY \
  PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_SATURATION_ELAPSED_MS \
  PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_SATURATION_PEAK_RSS_MIB \
  PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_OVER_LIMIT_REJECTED \
  PURE_EXPLORATION_VALIDATED_FAILURE_MAXIMUM_TOTAL_PATCH_CELLS \
  PURE_EXPLORATION_FAILURE_TOTAL_PATCH_SATURATION_ELAPSED_MS \
  PURE_EXPLORATION_FAILURE_TOTAL_PATCH_SATURATION_PEAK_RSS_MIB \
  PURE_EXPLORATION_FAILURE_TOTAL_PATCH_OVER_LIMIT_REJECTED \
  PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_PATH \
  PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_SHA256 \
  PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_EVIDENCE_PATH \
  PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_EVIDENCE_SHA256 \
  PURE_EXPLORATION_FAILURE_TOTAL_PATCH_EVIDENCE_PATH \
  PURE_EXPLORATION_FAILURE_TOTAL_PATCH_EVIDENCE_SHA256 \
  PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_ENVIRONMENT \
  PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_IMAGE; do
  test -n "${!required_evidence:-}"
done
python3 - <<'PY'
import hashlib
import json
import math
import os
import re
from pathlib import Path


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def require_exact_keys(value, expected, context):
    assert isinstance(value, dict), context
    assert set(value) == set(expected), context


def require_positive_integer(value, context):
    assert isinstance(value, int) and not isinstance(value, bool), context
    assert value > 0, context


def require_positive_finite_number(value, context):
    assert isinstance(value, (int, float)) and not isinstance(value, bool), context
    converted = float(value)
    assert math.isfinite(converted) and converted > 0.0, context
    return converted

candidate_views = int(os.environ["PURE_EXPLORATION_MAXIMUM_CANDIDATE_VIEWS"])
visibility_units = int(os.environ["PURE_EXPLORATION_MAXIMUM_VISIBILITY_WORK_UNITS"])
validated_product = int(os.environ["PURE_EXPLORATION_VALIDATED_VISIBILITY_PRODUCT"])
elapsed_ms = float(os.environ["PURE_EXPLORATION_WHOLE_CYCLE_MAX_ELAPSED_MS"])
peak_rss_mib = float(os.environ["PURE_EXPLORATION_WHOLE_CYCLE_PEAK_RSS_MIB"])
failure_entries = int(os.environ["PURE_EXPLORATION_MAXIMUM_FAILURE_ENTRIES"])
failure_patch_per_entry = int(
    os.environ["PURE_EXPLORATION_MAXIMUM_FAILURE_PATCH_CELLS_PER_ENTRY"]
)
failure_total_patch = int(
    os.environ["PURE_EXPLORATION_MAXIMUM_FAILURE_TOTAL_PATCH_CELLS"]
)
validated_failure_entries = int(
    os.environ["PURE_EXPLORATION_VALIDATED_FAILURE_MAXIMUM_ENTRIES"]
)
validated_failure_patch_per_entry = int(
    os.environ[
        "PURE_EXPLORATION_VALIDATED_FAILURE_MAXIMUM_PATCH_CELLS_PER_ENTRY"
    ]
)
validated_failure_total_patch = int(
    os.environ["PURE_EXPLORATION_VALIDATED_FAILURE_MAXIMUM_TOTAL_PATCH_CELLS"]
)
assert candidate_views > 0 and visibility_units > 0 and validated_product > 0
assert validated_product == candidate_views * visibility_units
assert math.isfinite(elapsed_ms) and elapsed_ms > 0.0
assert math.isfinite(peak_rss_mib) and peak_rss_mib > 0.0
assert validated_failure_entries == failure_entries
assert validated_failure_patch_per_entry == failure_patch_per_entry
assert validated_failure_total_patch == failure_total_patch

configured_limits = {
    "maximum_entries": failure_entries,
    "maximum_patch_cells_per_entry": failure_patch_per_entry,
    "maximum_total_patch_cells": failure_total_patch,
}
profile_authority = {
    "environment": os.environ[
        "PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_ENVIRONMENT"
    ],
    "image": os.environ["PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_IMAGE"],
}
assert all(isinstance(value, str) and value for value in profile_authority.values())
fixture_slots = {
    "entries": {
        "kind": "entry",
        "target": "maximum_entries",
        "validated": validated_failure_entries,
        "path": os.environ["PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_PATH"],
        "sha256": os.environ["PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_SHA256"],
        "elapsed": os.environ[
            "PURE_EXPLORATION_FAILURE_ENTRIES_SATURATION_ELAPSED_MS"
        ],
        "rss": os.environ[
            "PURE_EXPLORATION_FAILURE_ENTRIES_SATURATION_PEAK_RSS_MIB"
        ],
        "rejected": os.environ[
            "PURE_EXPLORATION_FAILURE_ENTRIES_OVER_LIMIT_REJECTED"
        ],
    },
    "per_entry": {
        "kind": "per_entry",
        "target": "maximum_patch_cells_per_entry",
        "validated": validated_failure_patch_per_entry,
        "path": os.environ[
            "PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_EVIDENCE_PATH"
        ],
        "sha256": os.environ[
            "PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_EVIDENCE_SHA256"
        ],
        "elapsed": os.environ[
            "PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_SATURATION_ELAPSED_MS"
        ],
        "rss": os.environ[
            "PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_SATURATION_PEAK_RSS_MIB"
        ],
        "rejected": os.environ[
            "PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_OVER_LIMIT_REJECTED"
        ],
    },
    "total": {
        "kind": "total",
        "target": "maximum_total_patch_cells",
        "validated": validated_failure_total_patch,
        "path": os.environ[
            "PURE_EXPLORATION_FAILURE_TOTAL_PATCH_EVIDENCE_PATH"
        ],
        "sha256": os.environ[
            "PURE_EXPLORATION_FAILURE_TOTAL_PATCH_EVIDENCE_SHA256"
        ],
        "elapsed": os.environ[
            "PURE_EXPLORATION_FAILURE_TOTAL_PATCH_SATURATION_ELAPSED_MS"
        ],
        "rss": os.environ[
            "PURE_EXPLORATION_FAILURE_TOTAL_PATCH_SATURATION_PEAK_RSS_MIB"
        ],
        "rejected": os.environ[
            "PURE_EXPLORATION_FAILURE_TOTAL_PATCH_OVER_LIMIT_REJECTED"
        ],
    },
}
repo_root = Path.cwd().resolve()
seen_paths = set()
seen_digests = set()
approved_artifacts = {}
top_level_keys = {
    "schema",
    "fixture_kind",
    "configured_limits",
    "target_limit_name",
    "reached_value",
    "over_one_attempted_value",
    "over_one_rejected",
    "elapsed_ms",
    "peak_retained_rss_mib",
    "authority",
    "generated_at",
}
for slot_name, slot in fixture_slots.items():
    artifact_path = Path(slot["path"])
    assert artifact_path.is_absolute(), slot_name
    resolved_path = artifact_path.resolve(strict=True)
    assert resolved_path.is_file(), slot_name
    assert resolved_path != repo_root and repo_root not in resolved_path.parents, slot_name
    assert resolved_path not in seen_paths, slot_name
    seen_paths.add(resolved_path)

    expected_digest = slot["sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", expected_digest), slot_name
    assert expected_digest not in seen_digests, slot_name
    seen_digests.add(expected_digest)
    artifact_bytes = resolved_path.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == expected_digest, slot_name
    artifact = json.loads(
        artifact_bytes.decode("utf-8", errors="strict"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite_constant,
    )
    require_exact_keys(artifact, top_level_keys, slot_name)
    assert artifact["schema"] == (
        "lunar-pure-exploration/failure-saturation-evidence/v1"
    ), slot_name
    assert artifact["fixture_kind"] == slot["kind"], slot_name
    require_exact_keys(artifact["configured_limits"], configured_limits, slot_name)
    for limit_name, configured_value in artifact["configured_limits"].items():
        require_positive_integer(configured_value, f"{slot_name}:{limit_name}")
        assert configured_value == configured_limits[limit_name], slot_name
    assert artifact["target_limit_name"] == slot["target"], slot_name
    require_positive_integer(artifact["reached_value"], slot_name)
    require_positive_integer(artifact["over_one_attempted_value"], slot_name)
    assert artifact["reached_value"] == configured_limits[slot["target"]], slot_name
    assert artifact["reached_value"] == slot["validated"], slot_name
    assert artifact["over_one_attempted_value"] == artifact["reached_value"] + 1, slot_name
    assert artifact["over_one_rejected"] is True, slot_name
    assert slot["rejected"] == "1", slot_name

    artifact_elapsed = require_positive_finite_number(
        artifact["elapsed_ms"], f"{slot_name}:elapsed_ms"
    )
    artifact_rss = require_positive_finite_number(
        artifact["peak_retained_rss_mib"], f"{slot_name}:peak_retained_rss_mib"
    )
    profile_elapsed = require_positive_finite_number(
        float(slot["elapsed"]), f"{slot_name}:profile_elapsed_ms"
    )
    profile_rss = require_positive_finite_number(
        float(slot["rss"]), f"{slot_name}:profile_peak_retained_rss_mib"
    )
    assert artifact_elapsed == profile_elapsed, slot_name
    assert artifact_rss == profile_rss, slot_name

    require_exact_keys(artifact["authority"], profile_authority, slot_name)
    assert artifact["authority"] == profile_authority, slot_name
    assert isinstance(artifact["generated_at"], str), slot_name
    assert artifact["generated_at"], slot_name
    # generated_at is record-only: deliberately do not parse it or compare it to a clock.
    approved_artifacts[slot_name] = expected_digest
print(
    "approved whole-cycle visibility evidence:",
    f"product={validated_product}",
    f"max_elapsed_ms={elapsed_ms}",
    f"peak_rss_mib={peak_rss_mib}",
)
print(
    "approved failure saturation evidence:",
    f"entries={validated_failure_entries}",
    f"per_entry={validated_failure_patch_per_entry}",
    f"total={validated_failure_total_patch}",
    f"artifacts={sorted(approved_artifacts.items())}",
)
PY
timeout 20s ros2 launch lunar_pure_exploration_ros pure_exploration.launch.py \
  platform_selector:=wheel \
  maximum_position_probes:="$PURE_EXPLORATION_MAXIMUM_POSITION_PROBES" \
  maximum_candidate_views:="$PURE_EXPLORATION_MAXIMUM_CANDIDATE_VIEWS" \
  maximum_collision_work_units:="$PURE_EXPLORATION_MAXIMUM_COLLISION_WORK_UNITS" \
  maximum_visibility_work_units:="$PURE_EXPLORATION_MAXIMUM_VISIBILITY_WORK_UNITS" \
  maximum_path_preview_poses:="$PURE_EXPLORATION_MAXIMUM_PATH_PREVIEW_POSES" \
  maximum_executable_path_points:="$PURE_EXPLORATION_MAXIMUM_EXECUTABLE_PATH_POINTS" \
  maximum_failure_entries:="$PURE_EXPLORATION_MAXIMUM_FAILURE_ENTRIES" \
  maximum_failure_patch_cells_per_entry:="$PURE_EXPLORATION_MAXIMUM_FAILURE_PATCH_CELLS_PER_ENTRY" \
  maximum_failure_total_patch_cells:="$PURE_EXPLORATION_MAXIMUM_FAILURE_TOTAL_PATCH_CELLS" &
launch_pid=$!
sleep 3
ros2 node info /pure_exploration
ros2 topic type /Car/T4/exploration/status
ros2 topic type /Car/T4/exploration/current_goal
ros2 topic type /Car/T4/exploration/frontiers
ros2 topic type /Car/T4/exploration/diagnostics
wait "$launch_pid" || test $? -eq 124
```

- [ ] Extract the embedded Python validator and run it against three temporary, repository-external JSON artifacts.
  The unmodified triplet must pass. Independent mutation REDs named `swapped_measurements` and
  `generic_stale_measurements` must respectively swap the entry/per-entry elapsed+RSS exports and replace each
  exported measurement in turn with a different old positive value; every mutation must fail against the unchanged
  artifact bytes. Also require failures for swapped complete artifact path/hash records, a swapped path without its
  hash, duplicate artifact path/digest, changed bytes with the old digest, and stale configured limits. Re-hash an
  otherwise identical artifact whose `generated_at` is an arbitrary old nonempty string and require it to pass;
  neither this test nor the deployment validator may parse that field or compare it with a clock.

- [ ] Check UTF-8 documents, clean source tree and diff scope.

```bash
python3 - <<'PY'
from pathlib import Path
for path in [
    Path("docs/superpowers/specs/2026-08-22-pure-frontier-exploration-design.md"),
    Path("docs/superpowers/plans/2026-08-22-pure-frontier-exploration.md"),
    Path("pure_planner/README.md"),
]:
    path.read_text(encoding="utf-8")
PY
git status --short
git diff --check 6a94182...HEAD
git diff --stat 6a94182...HEAD
```

- [ ] If final verification required a task-related fix, commit that fix separately; otherwise create no empty commit. Record in the handoff: exact tested commit SHA, build/test commands, test counts, synthetic evidence boundary, and the remaining requirement for Jetson AGX Orin vehicle validation.

## Spec Coverage Review

- [ ] Confirm every approved input appears exactly once in the node contract: global map, odometry, `/tf`, task command, planner Action and planner diagnostics.
- [ ] Confirm the explorer does not subscribe to local `GridMap`; local elevation/occupancy handling remains inside the pure planner.
- [ ] Confirm Task 6 implements the frozen double-then-promote range/FOV arithmetic, tight row-major AABB,
  streaming closed-grid DDA and exact work accounting covered by all eighteen visibility oracles; no ray bounding-box
  scan, tolerance expansion or uncharged classified cell remains.
- [ ] Confirm the approved Orin deployment profile binds all nine required resource limits, includes a checked
  `maximum_candidate_views * maximum_visibility_work_units` value and matching complete-cycle elapsed/RSS
  evidence, and binds each failure fixture to a unique repository-external JSON absolute-path/SHA pair. Confirm the
  validator checks the actual bytes, strict schema, fixed slot/kind, all three current limits, exact reach/over-one
  semantics, authority and exact profile elapsed/RSS/rejected exports. Swapped or old positive measurements,
  artifact/hash swaps and stale limits must fail; `generated_at` remains a nonempty record only and is never a
  freshness gate. Per-candidate timing alone is not accepted and no measured value becomes a repository default.
- [ ] Confirm task polygon supports concave shapes and task cells outside the current map are classified OUTSIDE_MAP, counted with potential unknown gain, and never rejected as invalid.
- [ ] Confirm unknown is never traversable, including footprint validation and global path evaluation.
- [ ] Confirm the only normal completion transition is exhaustive absence of reachable frontier.
- [ ] Confirm coverage is reported but never compared to a completion threshold.
- [ ] Confirm coverage defensive state and positive-area-underflow branches are exercised only through the private,
  non-installed detail seam and mutation checker; TaskRaster/coverage stable public APIs are unchanged.
- [ ] Confirm failure patches use the frozen double-then-promote tight-AABB/closed-circle authority, uncut work
  admission before int32 conversion, and exact length_error/overflow_error priority at GridIndex boundaries.
- [ ] Confirm goal commitment, five yaw offsets, 10 m range, 90 degree FOV, batch 16, scoring weights, 30 s/0.2 m stuck rule and two replans are all tested as exact constants.
- [ ] Confirm global and local planner timings are correlated by `request_id` and remain separate values.
- [ ] Confirm only typed `kExhaustiveNoPath` with reason `NO_PATH` or `GOAL_OUTSIDE_LOCAL_MAP` is conclusive
  unreachable evidence; TIMEOUT/client deadline/cancel/error/invalid response cannot complete.
- [ ] Confirm complete global preview and current executable segment are distinct, with rolling replan to the same committed candidate.
- [ ] Confirm Odometry exclusively supplies `odom -> base` while `/tf` supplies direct `map -> odom`.
- [ ] Confirm the real planner and explorer use one Action provider and the same planner-owned platform config.
- [ ] Confirm OccupancyGridView/TaskRaster CellCenter both delegate the one static GridToWorld primitive and the
  mutation guard rejects duplicated coordinate formulas.
- [ ] Confirm the indirect ownership graph: every asynchronous planner/cancel transport callback captures only weak
  CallbackState; its resident active Completion strongly owns the FrozenPlanningCycle and therefore the same batch,
  complete gain authority, frozen pose/resolution, coarse order/cursor and reachable rows. CandidateView indices
  resolve only inside that cycle batch, and request_id is the sole PlannerEvaluation association key.
- [ ] Confirm launch maps `platform_selector=wheel` to core `platform_type=WHEELED`, and all nine resource limits
  come from an externally approved Orin deployment profile with no repository production defaults.
- [ ] Confirm no timestamp, map version, covariance, freshness or observation-time admission exists.
- [ ] Confirm every implementation step has concrete values and a resolved interface choice, with no unfinished-work markers.

## Completion Boundary

This plan is complete when Tasks 1–16 pass on Ubuntu 22.04 + ROS 2 Humble and the resulting branch contains source, tests and documentation only. That proves deterministic algorithm behavior and ROS integration in the authoritative host/container environment. It does not by itself prove Jetson AGX Orin timing, vehicle motion safety, controller compatibility, power behavior or field exploration success; those require a separate on-vehicle validation plan and evidence set.
