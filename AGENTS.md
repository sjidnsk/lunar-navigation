# Agent Notes

本文件保存 `lunar_navigation` 的长期操作规则、验证入口和当前主线边界；详细设计、迁移清单和执行计划以 `docs/` 下的冻结文档为准。

## 沟通、编码与文件安全

- 默认使用中文回答；包含中文的 `.md`、`.json`、`.py`、`.txt` 文件必须使用 UTF-8 编码。
- 修改中文文件优先使用补丁式编辑，并在完成后显式按 UTF-8 读取核验。
- 禁止递归或批量删除：不得使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 删除文件时只能删除一个明确路径；批量删除必须停止并请求用户手动确认。
- 下载、数据集、模型、缓存、导出物和运行时 artifact 不得写入仓库：Windows 默认写入 `D:/CodexDownloads` 或 `D:/CodexDownloads/lunar_navigation/<stage_short>`；Ubuntu 默认写入 `~/CodexDownloads` 或 `~/CodexDownloads/lunar_navigation/<stage_short>`。不要提交训练输出、checkpoint、job state 或设备生成物。

## Git 与用户改动保护

- 工作区已有改动默认视为用户或其他任务产生；先查看状态，保留无关改动。
- 不得执行 `git reset --hard`、`git checkout -- <path>`、大范围 revert 或批量清理未跟踪文件。
- 提交时只纳入当前任务相关文件；任何回退、覆盖或删除已有改动都需要用户明确要求。
- 本仓库必须保持单一 Git 根：不得引入父仓、gitlink、嵌套 Git 仓库或相邻目录导入。
- `integration` 是本仓当前集成主线；通过本分支资格检查的功能分支才可合入，能够快进时使用
  `git merge --ff-only`，本地集成不隐含远端推送或发布。
- `volume-*` 分支保留为阶段里程碑，不再用后续跨卷功能推进其分支指针；新工作使用仓库根目录
  之外的独立工作树和功能分支，合入 `integration` 并完成合并后验证后再收尾。
- 仓库外的 Isaac/ROS/RViz 验证工具继续由其独立 Git 根管理，并合入该外部仓自己的 `main`；
  不得把外部仓源码、历史或运行 artifact 导入本仓来代替跨仓构建验证。

## 文档、计划与验证

- 长期规则写入本文件；详细设计、路线图、迁移清单和执行计划写入 `docs/`，不要把完整计划塞进 `AGENTS.md`。
- 修改仓库边界、外部接口或迁移规则后，至少运行：

  ```bash
  python3 tools/check_repository_boundaries.py .
  python3 -m pytest -q tests/foundation/test_repository_boundaries.py
  ```

- 涉及外部 ROS 消息或接口适配时，先核对 `docs/interfaces/external-input-baseline.md`、`ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml` 及对应检查脚本。
- 当前工作主机为 Ubuntu 22.04 LTS amd64 + ROS 2 Humble；执行 ROS 命令前先 `source /opt/ros/humble/setup.bash`，并确认 `ROS_DISTRO=humble`。不要混用其他 ROS 发行版生成的 `build/`、`install/`、`log/` 或接口产物。
- Windows 适合源码审阅、编辑、提交和发起远程任务；不是 ROS 集成、Linux wheel、C++ 发布包或 TensorRT 的权威构建环境。
- Ubuntu 22.04 + ROS 2 Humble 是 ROS 集成、C++20 构建、ONNX 等价验证和 rosbag 回放的权威环境；涉及训练的验证需要使用项目指定的 RTX 4080 SUPER 环境。原生构建、TensorRT engine 生成以及设备性能/功耗/稳定性验收以 Jetson AGX Orin 为权威环境。

## 当前主线与硬边界

- 本仓是面向 ROS 2 Humble 与 Jetson AGX Orin 的单一 Git 根仓库，承载 C++ v3 规划、PPO 模型发布与设备推理。
- 外部 Topic 数据和静态能力资料仍由外部项目拥有；上游未定义期间本仓按批准设计暂定提供同名 schema；不得与上游同名包共存，未来只能原子切换。本仓只声明依赖、订阅、适配和运行时校验，不复制或接管外部项目的数据所有权。
- 冻结 tag 和 Task 1 迁移清单不可变；旧仓 `legacy-maintenance` 只接受紧急缺陷或安全修复，并记录是否迁入本仓。
- 新功能只能进入本仓；AGX 观察期通过后，旧仓转为只读归档。
- 未经明确要求，不改变冻结设计、迁移边界、设备职责或外部接口基线，不把临时验证结果写成正式能力。
