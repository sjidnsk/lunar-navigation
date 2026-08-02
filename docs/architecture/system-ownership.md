# 系统所有权与仓库边界

## 仓库所有权

本仓是唯一的 `lunar_navigation` Git 根。不得复制旧仓 `.git`、`path-planner` 或 `dev-platform-constraints` gitlink；运行时也不得通过相邻目录导入旧仓。Task 1 冻结的来源清单是后续受控迁移的唯一入口。

## 运行职责

| 所有者 | 职责 | 不承担的职责 |
|---|---|---|
| Windows 开发机 | 编辑、审阅、提交、生成迁移清单、发起远程任务 | 权威 ROS/Linux/AGX 构建物 |
| Ubuntu 22.04 amd64 + RTX 4080 | ROS 集成、C++20 构建、PPO 训练、ONNX 验证、rosbag 回放 | AGX 发布验收替代品 |
| Jetson AGX Orin 64GB | aarch64 原生构建、TensorRT engine、设备验收 | PPO 训练、使用 amd64 engine |
| 外部项目 | 地图、定位、TF、任务、能力资料及其 ROS 消息定义 | 本仓内部 `lunar_planning_msgs` |
| 本项目 | C++ v3、`PlanMotion`、PPO 推理调度、内部配置与外部输入适配 | 外部同名消息定义和执行器命令 |

## 外部接口边界

外部 Topic、字段和资料包的接收基线见 [`../interfaces/external-input-baseline.md`](../interfaces/external-input-baseline.md)。该文件不是消息定义源；外部消息的版本和源码由外部项目发布。本仓不得复制外部 `.msg` 定义、重发同名消息或为同一消息建立重复 JSON Schema。

## 迁移过渡

- 冻结 tag 和 Task 1 迁移清单不可修改。
- 旧仓 `legacy-maintenance` 仅处理紧急缺陷与安全修复，并逐项记录是否已经迁入本仓。
- 所有新功能只进入本仓。
- AGX 观察期通过后，旧仓进入只读归档。

批准设计最初在 `47716ebc81b5b7d76bcc396563d98946abf101bc` 提交；本仓从冻结提交 `7309e93fdb85c60ff3736efe1a7f3c7eb640ee78` 迁入其同一设计 blob、总路线图和四卷计划。
