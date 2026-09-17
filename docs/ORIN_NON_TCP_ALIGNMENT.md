# 104/216 非 TCP 对齐记录（2026-09-16）

开发基线：integration/pure-planner-orin `5c23c13c`；工作分支
`feat/orin-nontcp-alignment`。本记录不代表已合入 `orin-humble`。

## 对齐范围与职责

| 层级 | 本次对齐 | 保留边界 |
| --- | --- | --- |
| P3 输入适配 | `/tf_static` 直接 map→odom 原样重发到私有输入；不刷新源时间戳 | 不修改 P3，不生成位姿/单位变换 |
| 导航核心 | 与 216 同源的局部目标区域、等待、全局路线复用、可通行认证 | TCP 不进入算法；UNKNOWN 补全不覆盖 BLOCKED |
| 导航 ROS | 可配置坐标系及跨帧校验、局部细图、正式调试命名空间、参数化到点容差 | 导航负责路径与地图，不发布车辆命令 |
| 探索 | 覆盖率只作统计；无前沿或无候选结束；采用现有等待/取消/暂停状态机 | 不引入第二套 PlanMotion 编排；本轮不启动探索任务 |
| 执行 | 与 216 共用 PathExecutor；命令自身限加速度；纵向有符号反馈 | 停稳仍检查横向速度；仅控制器拥有车辆命令 |
| 实机配置 | 盲区余量 2.0 m、能力速度 0.2 m/s、容差 0.1 m / 0.05 rad | 保留实机 P3 输入、传感器配置和物理通过性限制 |

`wheel_orin.yaml` 的盲区余量增加在车体外接圆半径上，总半径约 2.719 m，
不是把半径 2 m 范围全部标成可通行。已有障碍、真实出口与支撑平面认证仍有效。
`orin_controller.yaml` 仅提供将来经授权启动控制器时的匹配配置，本轮不运行；
其 `platform_config` 须由控制器启动入口提供部署目录下 `wheel_orin.yaml` 的绝对路径。

## 源码比较与归档

比对路径：104 原部署 `orin-humble-04782a82-20260916`、216 现场导出、
本地 integration 和候选工作树。核心导航、探索核心、规划消息接口与 216 对应文件一致；
导航/探索 ROS 共用头文件与源码、控制器 Python 共用源码也一致。
本地额外保留已有 demo、legacy 和测试，不复制 216 的测试裁剪/部署打包变更；
实机 launch 不启动这些 demo/legacy 节点。TCP、UE、仿真标定及虚构反馈未引入。

外部归档目录：`/tmp/p4-alignment-audit-20260916-2126`（不提交）。

- 104 修改前归档 SHA256：`91ab9deca235e04711a806067495417e0390e1fb828713b34d3ac4e891b4d944`
- 216 参考归档 SHA256：`2c89b81b3205284d77bf190b872f59ff8b49c68b1f16045e8944b96ad8a5c51d`
- 首次候选源码归档 SHA256：`7a607fe337fac019414a4ae6acb57f8ac5545c48fb43fef924bb8e863aef6737`

首次候选归档含全部本次生产源码，不含随后补充的本记录/说明文档，
也不含随后修正的控制器测试定时器隔离。
它是未提交工作树快照，不应标称为基线提交本身。

## 验证证据与未完成边界

- 本机 Jazzy：相关 5 包串行测试汇总 999 tests，0 errors，0 failures，0 skipped。
- 本机 Jazzy：20 项接口/硬件配置/真实 DDS 静态 TF 测试通过；控制器独立回归 153 项通过。
- 新增行为先观察失败再修改：覆盖率不中止、控制执行限加速度/横向停稳、无效容差、窗口外 UNKNOWN 图及正式调试命名空间。
- 原基线有一项过时源码断言失败：误要求只能存在一个 OccupancyGrid 发布器。
  已改为保留接口/默认值约束，发布行为由原生测试验证，未删除局部细图功能。
- 104 Humble 原生构建：9 个包全部 rc=0；测试未全通过，详见下方失败记录。
- 104 Humble：20 项配置/接口/静态 TF DDS 测试通过，控制器 153 项测试通过。
  系统 pytest 与用户目录 anyio 插件不兼容，设置 `PYTHONNOUSERSITE=1` 隔离后运行，未修改系统包。
  控制器首次运行 152 通过、1 失败：手动周期测试中真实定时器追加消息，
  固定计数断言可能读取上一周期；仅对该测试取消定时器，保留生产代码与停控断言，重跑 153 通过。
- 104 新候选实际输入规划：NOT_RUN；切换前检查发现 P3 原进程已正常退出，
  当前无里程计样本。P3 由另一会话负责恢复；用户随后明确要求不等待联调、直接切换 P4。
  本轮不重启 P3，切换仅允许导航与静态 TF 适配器，不发目标或启动控制。
  已有近车 BLOCKED 问题不能靠扩大 UNKNOWN 余量消除。
- 车辆运动/导航到达闭环：NOT_RUN，未授权发布控制。

### 原生回归未通过项（不得标为验收完成）

104 原生 C++ 首轮三包测试均有失败，证据在候选目录 `native-pretest-log`。

- `PlanningSnapshotContractTest.PartialBoundaryTilesKeepCellsOutsideOldBoundsUnobserved`：
  默认 8 MiB 栈发生 SIGSEGV；仅将该测试进程栈改为 64 MiB 后单例通过。生产配置未修改。
- `FineTraversabilityBuilder.RejectsInvalidEnvelopeAndSoftWeights`：
  无效包络异常导致 SIGABRT；64 MiB 栈下仍复现，待进一步定位异常传播。
- `CandidateGeneratorTest.ClosedTangencyMatrixRejectsEveryNonfreeState`：
  部分圆角间隙相切用例未抛预期异常，待定位 ARM 数值/契约边界；未放宽认证或修改断言。
- `incremental_navigation_node_test`：Cyclone 隔离测试超时。
  手动补齐安装前缀与动态库路径、切换 Fast DDS 后 35 项中 34 通过，
  调试发布断言读到 `PLAN_FOUND` 而非 `EXECUTING`；此结果不替代 Cyclone 验收。

因此，即使候选导航能够启动并正确加载参数，本候选仍是调试版本，而非已验收发布。

候选目录：`/home/yanfa/P4/orin-nontcp-candidate-20260916`。
原部署及其 `navigation-restart.gpke78a9/run_state.json` 保留用于回退；
切换前后须检查域 19/59 的车辆命令发布者均为零，并取消本次测试创建的目标。

## 已执行的候选切换

2026-09-16 22:07（设备显示时间），仅停止身份校验通过的旧 P4 navigation/tf_input
进程组，新适配器进程组 PID 26843、新导航启动器 PID 26844、导航节点 PID 26896。
运行记录：候选目录 `runtime-switch-20260916/run_state.json`，内含原命令与回退环境。

运行参数服务已验证私有 TF 输入、实际 P3 输入、`wheel_orin.yaml` 绝对路径、
`enable_tracking_feedback=true`、局部细图开启及 `0.1 m / 0.05 rad` 容差。
本地与远端平台配置 SHA256 一致，盲区余量为 2.0 m。
切换后域 19/59 车辆命令发布者为零，无跟踪反馈发布者、路径执行订阅者或活动目标；
未观察到 P3 输入样本，未执行实时规划测试。

需要回退时，在 104 上运行已交付的运行工具（非源码包的一部分）：

```bash
bash /home/yanfa/P4/orin-nontcp-candidate-20260916/rollback_candidate.sh
```

工具会先检查无控制/无活动目标，仅停止记录中身份匹配的 P4 进程并恢复旧命令；
不操作 P3。若 PID 身份变化或已回退，应先人工检查，而非强杀。
