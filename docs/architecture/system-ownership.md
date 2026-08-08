# 系统所有权与仓库边界

## 仓库所有权

本仓是唯一的 `lunar_navigation` Git 根。不得复制旧仓 `.git`、`path-planner` 或 `dev-platform-constraints` gitlink；运行时也不得通过相邻目录导入旧仓。Task 1 冻结的来源清单是后续受控迁移的唯一入口。

## 运行职责

| 所有者 | 职责 | 不承担的职责 |
|---|---|---|
| Windows 开发机 | 编辑、审阅、提交、生成迁移清单、发起远程任务 | 权威 ROS/Linux/AGX 构建物 |
| Ubuntu 22.04 amd64 + RTX 4080 SUPER | ROS 集成、C++20 构建、PPO 训练、ONNX 验证、rosbag 回放 | AGX 发布验收替代品 |
| Jetson AGX Orin 64GB | aarch64 原生构建、TensorRT engine、设备验收 | PPO 训练、使用 amd64 engine |
| 外部项目 | 地图、定位、TF、任务、能力资料及其 Topic 数据 | 本仓内部 `lunar_planning_msgs` |
| 本项目 | C++ v3、`PlanMotion`、PPO 推理调度、内部配置与外部输入适配，以及上游未定义期间的暂定 `lunar_navigation_msgs` schema | 外部 Topic 数据和执行器命令 |

能力资料的“外部所有权”表示未来正式 provider 和运行时数据生产职责，不表示当前训练等待外部
closure。本仓批准的 capability v2 freeze 是现阶段规划与正式训练的唯一数值权威；未来 provider
只能在逐字段等价后原子接管发布，不能静默改变能力版本或摘要。

## 外部接口边界

外部 Topic、字段和资料包的接收基线见 [`../interfaces/external-input-baseline.md`](../interfaces/external-input-baseline.md)。该文件是暂定消息字段与语义的唯一权威基线；上游未定义期间，本仓提供同名 `lunar_navigation_msgs` schema，但 Topic 数据生产者仍由外部项目拥有。不得与上游同名包共存，未来只能原子切换。

## 迁移过渡

- 冻结 tag 和 Task 1 迁移清单不可修改。
- 旧仓 `legacy-maintenance` 仅处理紧急缺陷与安全修复，并逐项记录是否已经迁入本仓。
- 所有新功能只进入本仓。
- AGX 观察期通过后，旧仓进入只读归档。

批准设计最初在 `47716ebc81b5b7d76bcc396563d98946abf101bc` 提交；本仓从冻结提交 `7309e93fdb85c60ff3736efe1a7f3c7eb640ee78` 迁入其同一设计 blob、总路线图和四卷计划。
