# 净空简化器与净空代价的分支、训练环境同步

日期：2026-09-19；承接同日净空权重标定报告。用户授权同步集成主线、训练运行环境、实车和 TCP 仿真分支。

## 源码范围

| 目标分支 | 算法同步提交 |
|---|---|
| `integration/pure-planner-orin` | 简化器 `9a5a23f7`，净空代价 `7b529b42` |
| `app/orin-real-vehicle` | 简化器 `0eb2cb80`，净空代价 `5bc459b0` |
| `app/orin-unreal-simulation` | 简化器 `3ff77535`，净空代价 `d20c0caf` |
| `feat/drl-exploration-redesign` | 简化器 `0f236ad`，净空代价 `1fe5914`，测试和标定材料 `aac4e38` |

各分支简化器源码 SHA-256 完全一致：
`46801b6203665ede4174a9ff2be8d142aa134a42b84241540e4e82e769802794`。

轮式默认 `clearance_weight=2.0`，足式默认保持 0，允许启动时覆盖。保留应用/DRL 分支已有的目标位置和朝向容差参数。同步时这些相邻代码产生合并冲突，按保留双方功能解决，没有覆盖原有参数。

TCP 分支原有 5 个未提交文件、DRL 工作树原有 62 个修改/未跟踪文件的内容哈希均保持不变；其中已授权的两个简化器文件现已提交，其余既有修改未被纳入同步提交。DRL 的 README 和操作文档已有其他工作，本轮保留原文件内容，使用本独立文档记录运行环境变更。

未推送远程、未修改只读 import 分支、未连接设备或发布车辆命令。

## 训练安装环境

同步时系统没有存活的训练或 Actor 实验进程，因此没有终止训练。旧配对实验状态是 interrupted，本次不自动重启或修改旧实验记录，不清理模型和回放。

已执行 DRL 工作树的 `scripts/drl/build.sh`，6 个依赖包构建及安装成功。正式训练入口继续使用：
`/home/kai/.cache/lunar-drl-redesign/jazzy/install/local_setup.bash`。

核心库和导航 ROS 库均已进入该 install，不依赖此前临时实验库或 `LUNAR_PROBE_CLEARANCE_WEIGHT`。构建库与安装 ROS 库 Build ID 均为
`869ff07b9ce8d27b7aabd332e27b643ca2a6bb88`；安装过程去除 RUNPATH，文件整体哈希因此不同。核心库构建/安装哈希一致。

实际启动隔离训练导航进程后，通过 GetParameters 查询到 `clearance_weight=2.0`，并通过 `/proc/<pid>/maps` 确认加载的是正式训练 install 中的两份库。

## 验证结果

- 训练核心 CTest：16/16 通过。
- 训练 ROS CTest：14/14 通过。首轮测试误用了 domain 233，Fast DDS 端口超范围，改为 229 后完整重跑通过；没有修改算法或测试断言绕过失败。
- 实车分支 ROS 包：独立本机 Jazzy 构建，13/13 CTest 通过。
- TCP 分支 ROS 包：另行独立本机 Jazzy 构建，13/13 CTest 通过。两条分支核心源码相同，ROS 生产源码相同，但安装列表和既有测试不同，因此分别测试。
- 正式训练 install 的轻量运动学闭环：固定原洞穴失败场景，原失败目标和前方目标各重复三次，6/6 `GOAL_REACHED`，0 次碰撞，前缀全部到达；每次查询运行时权重均为 2。没有写入学习回放。

上述是本机 Jazzy 和轻量运动学证据；Humble、Orin、TCP 动力学及实车验证均为 NOT_RUN。实车/TCP 的“同步”指本地开发分支，不表示已部署到设备。

证据目录：`/home/kai/.cache/lunar-clearance-rollout-20260919/`，包含 preflight/postflight、构建与测试日志、`runtime-smoke/run.py` 和 `runtime-smoke/closed-loop.json`。

后续训练将采用新导航行为。旧配对实验使用旧导航版本，其结果应保留原版本标签，不能与本次更新后数据混作同一导航条件下的 Actor 对照。
