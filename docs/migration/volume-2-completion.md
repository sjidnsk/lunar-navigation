# Volume 2 规划器与 ROS 2 Action 交接记录

## 结论

Volume 2 已在 `volume-2-planner-ros` 分支完成冻结范围内的实现与验证。交接回退点为
`foundation-v1`，完成标签为 `planner-action-v1`。本卷交付 C++ v3 共享规划核心、轮式/
足式/跳跃适配、`PlanMotion` ROS 2 Action 生命周期服务和可选 Nav2 轮式适配器；未改变
`docs/interfaces/external-input-baseline.md` 冻结的外部输入所有权和原子切换规则。

`planner-action-v1` 只表示 Ubuntu amd64 上的源码、接口和构建测试基线完成，不表示 AGX
设备性能、功耗或四小时稳定性验收完成，也不替代后续卷的
`model-package-ready -> device-verified -> policy-pipeline-v1` 状态链。

## 交付位置

| 内容 | 位置 |
| --- | --- |
| 稳定、无 ROS 依赖的公共 C++ API | `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/planner.hpp` |
| 共享算法（ARA*、安全投影、凸走廊、有限 QP 等） | `ros2_ws/src/lunar_planner_core/src/shared/` |
| 轮式规划算法 | `ros2_ws/src/lunar_planner_core/src/wheel/` |
| 足式规划算法 | `ros2_ws/src/lunar_planner_core/src/legged/` |
| 跳跃规划、弹道校验和承诺状态机 | `ros2_ws/src/lunar_planner_core/src/hopper/` |
| 外部输入适配和 `PlanMotion` 生命周期服务 | `ros2_ws/src/lunar_planner_ros/` |
| `PlanMotion` Action 与规划消息 | `ros2_ws/src/lunar_planning_msgs/` |
| 可选轮式 Nav2 插件 | `ros2_ws/src/lunar_nav2_adapter/` |

规划核心只接收拥有值语义的 `PlannerInput` 快照。ROS Topic 数据、能力资料和临时同名
schema 仍遵守外部输入基线：上游未定义期间由本仓暂定提供，上游就绪后只能原子切换，
不得让两个同名消息包同时出现在运行环境中。

## 迁移提交

| 任务 | 提交 | 结果 |
| --- | --- | --- |
| Task 1 | `db90c0b` | 导入固定 C++ v3 源码快照 |
| Task 2 | `8537a7d` | 定义简化、类型化规划核心 API |
| Task 3 | `8dabf8d` | 用私有 facade 隔离旧 v3 |
| Task 4 | `a207c8e` | 迁移共享规划核心 |
| Task 5 | `e1d1c8c` | 迁移轮式规划器 |
| Task 6 | `eae1abd` | 迁移足式规划器 |
| Task 7 | `6777244` | 迁移跳跃规划器和承诺状态机 |
| Task 8 | `b50067c` | 从运行时移除旧规划契约 |
| 边界修正 | `8e69998` | 消除 URI 文本的边界检查误报 |
| Task 9 | `4097e58` | 将外部输入适配为规划快照 |
| Task 10 | `36f4c00` | 增加生命周期 `PlanMotion` Action 服务 |
| Task 11 | `d0ef562` | 增加默认关闭的轮式 Nav2 适配器 |
| Task 12 | `planner-action-v1` | 固定差分、性能、边界与交接证据 |

旧源提交和文件哈希仍以 `docs/migration/source-baseline.md`、
`migration/source_inventory.yaml` 和 `migration/fixture_inventory.yaml` 为唯一授权来源；本卷没有
回写旧仓。

## 差分基线

迁移期 Python A* 只作为离线测试参考。测试使用
`dev_platform_constraints@61e9fa8afd09db83632456bdcf181c222ee13513` 的
`src/dev_platform_constraints/path_planning/astar.py`，固定副本为
`tests/differential/fixtures/legacy_astar_61e9fa8.py`，SHA-256 为
`33bf574cfe0f3eaa0fbfbd59127339d5bc23c9f35baf608567c991bcc66ae701`。

差分仅检查轮式固定案例的可达性、路径安全、目标到达和宽松成本合理性，不要求新旧算法
轨迹逐点相同。`tests/differential/**` 和该旧 Python 文件均不进入生产安装包。

## Ubuntu amd64 资格验证

验证日期为 2026-08-03，权威环境为 Ubuntu 22.04.5 LTS amd64、Linux
`6.8.0-124-generic`、ROS 2 Humble、GCC 11.4.0。主机为 Intel Core i7-14700KF
（28 个逻辑 CPU）、64 GiB 内存、NVIDIA GeForce RTX 4080 SUPER 16376 MiB、
compute capability 8.9、驱动 595.84、CUDA 13.2。环境指纹状态为 ready，JSON 的
SHA-256 为 `06eda481a6e18e33a67c394e31a9ed6b8cedb0459225c30d26b27fdad0ed0a04`。

验证结果：

- 从干净外部 build/install 目录构建
  `lunar_navigation_msgs`、`lunar_planning_msgs`、`lunar_planner_core`、
  `lunar_planner_ros`：4 个包完成。
- `lunar_planner_core` 14 个 CTest 与 `lunar_planner_ros` 6 个 CTest：合计
  97 个测试用例，0 error、0 failure、0 skip。
- 冻结差分和 `planner_ros` 集成 pytest：26 passed，0 skip。
- foundation 回归：113 passed；仓库边界和外部接口现场检查通过。
- ASan/UBSan 独立构建与核心测试：14 个 CTest、58 个测试用例全部通过，无 sanitizer
  报告。
- 可选 Nav2 适配器：9 个测试用例全部通过；生命周期立即 configure/cleanup 压力回归
  200/200 次通过。默认运行时构建和安装中不存在 `lunar_nav2_adapter`。

主要外部证据目录：

```text
/home/kai/CodexDownloads/lunar_navigation/volume2-planner-ros/task12-final-build-20260803
/home/kai/CodexDownloads/lunar_navigation/volume2-planner-ros/task12-final-install-20260803
/home/kai/CodexDownloads/lunar_navigation/volume2-planner-ros/task12-qualification-20260803
/home/kai/CodexDownloads/lunar_navigation/volume2-planner-ros/task12-sanitizer-only-build-20260803
/home/kai/CodexDownloads/lunar_navigation/volume2-planner-ros/task11-final-build-20260803
/home/kai/CodexDownloads/lunar_navigation/volume2-planner-ros/task11-default-runtime-20260803
```

这些目录是本机验证 artifact，不进入 Git，也不是部署输入。

## 固定性能基线

`planner_core_benchmark` 固定为 12 x 8、1 m 分辨率的轮式平坦地图，起点
`[2.5, 3.5, 0.0]`，目标 `[4.5, 3.5, 0.0]`，随机种子 `549924309330`，
10 次 warmup、100 次 measured。运行绑定提交
`d0ef562ee1b43f66d4ffcaf8ea82482de12d56ad` 和上述设备指纹哈希，结果为：

| count | median | p95 | max | expanded states |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1.914368 ms | 2.035795 ms | 2.056194 ms | 8 |

这是 CPU 规划核心在 Ubuntu 开发机上的回归参考值；设备指纹记录 RTX 4080 SUPER 是为了
复现整机环境，不表示该 benchmark 使用 GPU。JSON 中 `release_gate` 为 `false`。AGX 原生
环境的 `P95 < 1 s`、功耗和四小时稳定性仍由 Volume 4 单独验收，不能用本结果替代。

## ARM64 编译通道

使用 Ubuntu 上的 AArch64 交叉编译器，以 `BUILD_TESTING=OFF` 成功生成
`liblunar_planner_core.so`；`file`/ELF 头确认其为 64 位 little-endian ARM AArch64 共享库。
证据目录为：

```text
/home/kai/CodexDownloads/lunar_navigation/volume2-planner-ros/task12-arm64-build-20260803
/home/kai/CodexDownloads/lunar_navigation/volume2-planner-ros/task12-arm64-install-20260803
```

这只证明核心的 ARM64 编译可行性，不是 AGX 原生构建、TensorRT、性能或设备验收。

## 生产边界与发布排除项

生产源已确认不含 `ContentRef`、`ContractObjectRegistry`、`ReferenceBundle`、旧 JSON/schema
codec、旧 `path_planner.search` 或 `AStarPlanner` 运行时引用；
`lunar_planner_core` 不包含 ROS、geometry、nav 或 tf2 头文件。

以下内容不得进入默认 build/install 或设备部署包：

- `tests/differential/**`，包括固定旧 Python A*；
- `tests/performance/**` 和 benchmark 输出；
- 旧仓、旧 Python 包、源码库存工作副本和任何嵌套 Git 元数据；
- 默认关闭的 `lunar_nav2_adapter`，除非明确选择 Nav2 集成构建；
- 本机 build、install、log、测试结果、环境指纹和 benchmark artifact。

## 集成、回退与后续门槛

本分支从 `foundation-v1` 指向的提交 `8e64e107514de686ad0df474eee36352a2a119c7`
继续开发。集成时应把 `planner-action-v1` 合并回当前 `volume-1-foundation`，并保留该分支上
已经存在的跨卷协调提交 `4d8588e`；该提交只调整 Volume 3、Volume 4、总路线图和 AGX Git
安装设计，没有改变本卷冻结任务。

若集成或后续平台迁移失败，回退到 `foundation-v1` 或本仓上一个通过提交；继续使用私有
facade 做离线对照，不修改冻结 tag、不回写旧 v3 仓。下一阶段按 Volume 3 进入策略模型
流水线；AGX 不可用时不得提前产生 `device-verified` 或 `policy-pipeline-v1`。Volume 4 的
设备交付继续遵守 `installed -> qualified -> activated`：Git 安装默认不启动，独立验收后
再手动激活，U 盘部署仍不在范围内。
