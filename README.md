# Lunar Navigation

`lunar_navigation` 是面向 ROS 2 Humble 与 Jetson AGX Orin 的单一 Git 根仓库。它承载 C++ v3 规划、PPO 模型发布与设备推理；不包含父仓、gitlink、嵌套 Git 仓库或相邻目录导入。

## 三机职责

- Windows：源码审阅、编辑、提交和发起远程任务；不是 ROS、Linux wheel、C++ 发布包或 TensorRT 的权威构建环境。
- Ubuntu 22.04 amd64 + RTX 4080：ROS 集成、C++20 构建、训练、ONNX 等价验证和 rosbag 回放的权威环境。
- Jetson AGX Orin 64GB（aarch64，最低 L4T R36.0.0）：原生构建、TensorRT engine 生成、Action/性能/功耗/稳定性验收。

外部 ROS 消息和静态能力资料由外部项目拥有；本仓只声明依赖、订阅、适配和运行时校验。接收字段基线见 [`docs/interfaces/external-input-baseline.md`](docs/interfaces/external-input-baseline.md)。

## 基础检查

```bash
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
```

## 迁移与过渡政策

- 冻结 tag 和 Task 1 迁移清单不可变。
- 旧仓 `legacy-maintenance` 只接受紧急缺陷或安全修复；每一项都必须记录是否迁入本仓。
- 新功能只能进入本仓。
- AGX 观察期通过后，旧仓转为只读归档。

批准架构设计的原始来源提交为 `47716ebc81b5b7d76bcc396563d98946abf101bc`；本次迁入的冻结设计、路线图和四卷计划来源提交为 `7309e93fdb85c60ff3736efe1a7f3c7eb640ee78`。
