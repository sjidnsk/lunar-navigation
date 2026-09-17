# 104 地图范围与 3 m 盲区余量同步

## 范围

用户确认同步 216 已部署的未知地图边界修复和 3 m 起点盲区余量。只修改 P4，不引入 TCP，不修改 P3、program、车辆接口、坡度/高差阈值或 UNKNOWN 传播规则。

- PersistentElevationMap 保留输入地图完整矩形，包括 NaN 边框。地图滚动时沿用历史范围并集；未知输入不删除历史有限高程。
- 全未知首帧可建立地图几何，但不分配高度 tile、不编造高度。
- 纯范围变化产生 revision。ElevationPipeline 与导航工作线程按快照变化派生，而不是只看高程脏格；避免边界更新漏传。
- 104 `config/wheel_orin.yaml` 的 `start_blind_zone_margin_m` 从 2.0 改为 3.0。加车体半径后总补偿半径约 3.718722 m，仍须观测支持、障碍认证和可达出口；不保证任意目标可规划。
- 细图显示仍为原有局部窗口，粗图保留完整地图范围；范围扩大不代表增加有效观测。

保留 104 导航节点已有实机适配，仅定点修改 RunFineDerivationWorker 的早退条件。未整文件覆盖为 216 的导航节点。

## 验证与产物

本地分支 `fix/orin-terrain-sync`，未提交、合并或推送。原 `orin-nontcp-alignment` 和 `wheel-terrain-basic` 工作树未修改。

104 候选目录 `/home/yanfa/P4/bounds-sync-20260917`，使用 104 已部署 core/ROS 源码为基线。源代码快照保存在 `candidate.tgz`，旧基线在 `/home/yanfa/P4/terrain-sync-20260917/bounds-baseline.tgz`。

- 测试先行：旧算法在全未知范围、稀疏观测不裁边、纯未知扩图、旋转平移边界四项新增用例失败。
- 本机 Jazzy：完整核心 CTest 17/17 通过；地图流水线测试通过。
- 104 Humble：本轮完整核心 CTest（候选库优先，64 MiB 栈）16/17；保留既有 `RejectsInvalidEnvelopeAndSoftWeights` 异常退出，未删除断言或掩盖失败。本次相关的高程、地形、起点补全目标通过。
- ROS 原生完整构建成功；地图流水线、地图适配、探索投影、地图发布 4/4 CTest 目标通过，ldd 确认测试加载候选 core/ROS 库。实机配置 Python 测试 3/3 通过；本机 3 m 物理能力保持测试通过。

原生测试必须显式将候选 `build-core`、`build-ros` 或候选 install/lib 放在 LD_LIBRARY_PATH 首位，避免已有 ROS overlay 使测试误加载旧部署库。

## 部署与现场结果

2026-09-17 15:54 左右只重启 P4，备份目录：`/home/yanfa/P4/bounds-sync-20260917/deployment-sk0i_369/backup`；同级 deployment.json 记录新旧哈希与 P3 会话。替换三份生产源码、三份 C++ 测试、实机配置及配置测试、core/ROS 两个共享库。未替换其他配置、静态 TF 适配或原有地形分类实现。

导航节点 PID 13561，通过 /proc/maps 核查已加载部署路径的新库：

- core SHA256 `5e159a09c4dc194f3263ff2cd22aff63ae71baf8f91f02386e152df8b2dba4c2`
- ROS SHA256 `f66623c86400266406a594fce92eedc4f1fa5850a43b127d59a8ebc2b7d4e027`

采样时 P3 为 320×320、0.2 m 格，范围 [-32,32)×[-32,32)。P4 粗图为 64×65、1 m 格，范围 [-32,32)×[-33,32)，多出南侧一行来自地图范围历史并集及粗格对齐。P4 局部细图为 140×140、0.2 m 格，范围约 [-14,14)²。起点 (0.0015,0.0012) 已位于细图内，状态 UNKNOWN；未虚构车底高度。细图采样为 3 FREE / 620 BLOCKED / 18977 UNKNOWN。图像在本机 `/tmp/P4-bounds104/live/p3_elevation_current.png`。

随后对两个当时的 FREE 候选 (4.1,1.3)、(4.5,-0.9) 做规划单测：均被接收，均有 AVAILABLE 全局引导，但返回 `START_BLIND_ZONE_UNRESOLVED`，PathReference 数量 0，假设格 719，目标均终止 ABORTED。未获得 PLAN_FOUND；不能把 216 的成功结果套用于 104 当前地图。

测试前后 domain 59/19 车辆控制发布者均为 0，跟踪反馈发布者、路径执行消费者均为 0，无活动目标。测试后的 8 秒检查未收到实时里程计（节点仍可见），因此不再继续发目标；不由进程存在推断输入连续可用。P3 已由其他现场操作变为 `run_20260917_155323` 会话，本轮没有修改或重启 P3。

原生核心异常处理测试问题仍保留；ROS 全部测试目标未全量执行，仅上述受影响目标完成；车辆运动为 NOT_RUN。源码分支仍未提交、合并或推送。
