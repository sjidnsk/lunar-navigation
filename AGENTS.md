# Pure Planner Orin Agent Rules

本文件适用于当前仓库根目录及其全部子目录，并针对 pure-planner 导入历史覆盖父目录中面向
`lunar-runtime` 的分支、环境和验证规则。详细设计、操作步骤和验证证据仍以 `docs/`、
`README.md` 和 `docs/操作指令.md` 为准。

## 分支职责

- `import/pure-planner-orin-main` 是从原仓库导入的只读来源快照，固定基线为提交
  `200c1ea4188b772ee8589708473eee1391fa950d`、tree
  `dc7789c0131a523d110be6faae54f212e9bc2c7c`。不得在该分支开发、提交、合并、rebase、
  force-push 或移动分支指针。
- `integration/pure-planner-orin` 是 pure-planner 的集成主线，不是
  `import/pure-planner-orin-main` 的待回合分支。任何开发结果都不得反向合入 import 分支。
- 新功能和缺陷修复应从 `integration/pure-planner-orin` 创建命名清楚的 `feat/*` 或 `fix/*`
  分支，并使用独立 worktree。验证通过后再合入 integration。
- 不把本历史合入 `lunar-runtime` 的其他主线、开发分支或远程默认分支，除非用户明确批准仓库级
  整合方案。
- Humble/Orin 稳定或发布分支只能在用户明确要求时从 integration 创建；发布修复中的通用修改
  必须同步回 integration，不得只留在部署分支。
- 未经用户明确要求，不合并、不推送、不创建 PR、不设置 upstream，也不删除分支或 worktree。

## 多会话与工作树安全

- 每次开始工作先运行 `git branch --show-current`、`git status --short` 和
  `git worktree list`，确认自己位于预期分支和 worktree。
- `integration/pure-planner-orin` 主要用于集成和验证；不要让多个会话在同一个 integration
  worktree 中并行实现功能。新任务应先创建独立功能分支/worktree。
- 已存在的修改、未跟踪文件、提交和分支移动默认属于用户或其他会话。不得清理、覆盖、暂存、
  amend、rebase 或提交不属于当前任务的内容。
- 如果进入时 integration worktree 已脏，不得为了创建分支而 stash、reset、checkout 或移动
  他人文件；停止写入并报告实际状态，改用干净的独立 worktree。
- 提交时使用显式路径，只纳入当前任务文件。禁止 `git add .`、`git add -A` 和会吞入其他会话
  变更的批量提交。
- 禁止 `git reset --hard`、强制 checkout、强制删除 worktree，以及递归删除仓库内容。

## Jazzy、Humble 与 Orin 边界

- 本机 ROS 2 Jazzy 用于隔离算法、单元测试、RViz demo 和开发调试；Jazzy 结果不能证明
  Humble、Jetson AGX Orin、DDS、真实 rosbag、控制器或实车就绪。
- 生产接口与部署目标是 Ubuntu 22.04、ROS 2 Humble 和 Jetson AGX Orin。部署前必须在对应
  Humble/Orin 环境重新构建并执行目标验证。
- Jazzy、Humble、Orin、DDS、rosbag、RViz 和实车证据必须分别记录。未实际执行的层级明确标为
  `NOT_RUN`，不得由较低层级结果推断通过。
- 不得混用不同 ROS 发行版生成的 `build/`、`install/`、`log/` 或 overlay。每个构建通道使用
  独立目录，并在执行前确认 `ROS_DISTRO`。
- `/lunar_demo/*` 仅用于隔离 Jazzy 演示；未经明确范围批准，不得接入 `/Car/T5/Car_Cmd_Vel`
  或增加会改变车辆行为的连续控制。

## 验证与完成声明

- 修改前先阅读与任务直接相关的设计、计划、接口基线和操作文档；文档中的历史结果不能代替
  当前工作树的重新验证。
- Python 仓库测试从仓库根目录使用 `python3 -m pytest`，避免通过裸 `pytest` 丢失根目录模块
  搜索路径。
- 含 750 m 滚动规划性能场景的 CTest 应按包串行运行，避免并行 CPU 竞争制造假超时。
- 修改 ROS 接口、launch、配置或包依赖后，至少执行相关静态检查、受影响包构建和包测试；准备
  声明完成前再运行与风险相称的完整测试。
- 只有 `planning_outcome: 0`、`reason_code: PLAN_FOUND` 且 `has_reference: true` 才能作为正式
  规划成功证据；Action `SUCCEEDED`、泛化的 `COMPLETED` 或仅有路径可视化均不充分。
- 报告必须区分源码静态检查、本机 Jazzy、Humble 容器/主机和 Orin/实车证据，并列出仍然
  `NOT_RUN` 的边界。

## 文件与产物

- 源码、脚本、测试和正式文档可以提交；构建目录、日志、缓存、下载依赖、bag、模型、导出物和
  设备运行 artifact 不得提交。
- 临时依赖和验证产物写入仓库外路径；若工具必须在工作树内产生临时内容，只能放入已忽略的
  专用目录，并在交付时说明。
- 中文 Markdown、Python、JSON 和文本文件使用 UTF-8；修改后显式读取并运行 `git diff --check`。
