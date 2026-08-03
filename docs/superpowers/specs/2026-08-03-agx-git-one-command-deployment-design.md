# Jetson AGX Orin Git 一键安装设计

**状态：** 已批准

**日期：** 2026-08-03

**适用范围：** Ubuntu 22.04 amd64 开发集成环境与 Jetson AGX Orin ROS 2 Humble 原生安装链

## 1. 背景

当前 Jetson AGX Orin 暂不可用，项目开发、ROS 2 Humble 适配和自动化测试先在 Ubuntu 22.04 amd64 上完成。设备恢复可用后，需要从固定 Git 发布标签在 AGX 上以一条命令完成预检、aarch64 原生构建、版本化安装和快速校验。

卷四现有计划已经拆分出发布候选、AGX 安装、安装校验、设备门槛、激活和回退脚本，但操作者仍需手工串联多个入口。本设计增加一个薄的 Git 部署编排器，统一调用底层工具，不复制其校验和安装逻辑。

本设计中的“一键部署”特指“一键安装”，不包含服务激活，也不包含性能、功耗或四小时稳定性验收。

## 2. 目标

- Ubuntu amd64 完成 ROS 2 Humble 适配、测试和固定发布标签。
- AGX 通过一条命令从 Git 获取固定标签并完成本机原生安装。
- 安装成功后不切换 `current`，不启动或重启 systemd 服务。
- 手动激活保持为独立的一条命令，并沿用原子切换、健康检查和失败回退语义。
- AGX 不可用期间，能在 Ubuntu 使用 fixture、临时目录和命令注入验证部署编排。
- 安装失败不影响当前运行版本，并保留可诊断的阶段状态和日志。

## 3. 非目标

- 本阶段不实现 U 盘或完全离线部署。
- 不在 Ubuntu 生成可部署到 AGX 的 amd64 或 aarch64 生产二进制。
- 不在 Ubuntu 生成 AGX TensorRT engine。
- 不把 ARM64 交叉编译或模拟测试声明为 AGX 实机验收。
- 不把性能、功耗、异常矩阵或四小时稳定性门槛合并到安装命令。
- 不改变卷四对生产激活所需 qualified manifest 的要求。
- 不自动删除旧版本、失败版本、构建缓存或日志。

## 4. 权威平台与制品边界

Ubuntu 22.04 amd64 是源码适配、ROS 集成测试和发布标签生成环境。Jetson AGX Orin 是 aarch64 原生构建、TensorRT engine 生成和设备安装的权威环境。

Git 只传递源码和发布身份。不得通过 Git 部署流程向 AGX 传递：

- amd64 ELF、Linux wheel 或 Ubuntu 构建目录；
- RTX 主机生成的 TensorRT engine；
- checkpoint、优化器状态、训练数据或训练日志；
- 仓库内的 `build/`、`install/` 或 `log/`。

卷二规划运行时只需要固定 Git 标签。卷三模型接入后，四文件模型候选继续遵守既有 release candidate 合同；编排器通过可选的绝对路径 `--candidate` 把候选交给底层安装器，不把模型提交到 Git。若某一发布 manifest 声明模型为必需但未提供候选，预检必须在写系统目录前失败。

跨卷接口以 [`卷三计划`](../plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md)、[`卷四计划`](../plans/2026-08-02-lunar-navigation-volume-4-integration-cutover.md) 和 [`总路线图`](../plans/2026-08-02-lunar-navigation-greenfield-roadmap.md) 为共同约束：卷三在 AGX 不可用时只交付 `model-package-ready` 候选和 pending 报告，不创建临时替代 tag；卷四可以先完成 Ubuntu 模拟工具，但真实安装、验收、激活和最终标签必须等待设备。

## 5. 总体架构

```text
Ubuntu 22.04 amd64
  ROS 2 Humble 适配 + 测试 + annotated Git tag
                         |
                         v
Jetson AGX Orin bootstrap checkout
  deploy_agx_from_git.sh
    -> GitSourceResolver
    -> DevicePreflight
    -> NativeInstaller (install_agx.sh)
    -> InstallVerifier (verify_agx_install.sh)
    -> DeploymentReport
                         |
                         v
  /opt/lunar_navigation/releases/<release-id>
  current 不变，systemd 不启动
```

Git 编排器是薄入口。环境识别、原生构建、模型安装、engine 生成和安装清单校验仍由既有或卷四计划中的底层工具承担。

## 6. 操作入口

AGX 首次部署前需要进行一次 bootstrap checkout。此后每个版本只执行一条安装命令：

```bash
./scripts/deploy_agx_from_git.sh \
  --repo <git-repository-url> \
  --ref <annotated-release-tag>
```

全系统发布需要本地 release candidate 时使用：

```bash
./scripts/deploy_agx_from_git.sh \
  --repo <git-repository-url> \
  --ref <annotated-release-tag> \
  --candidate <absolute-candidate-path>
```

接口规则：

- `--repo` 必须是明确的 Git 仓库地址。
- `--ref` 必须解析为 annotated tag，且标签名只允许 `[A-Za-z0-9._-]`；拒绝分支、浮动 `HEAD`、路径分隔符和不存在的引用。
- 没有 release manifest 时，`release-id` 等于标签名；存在 manifest 时，其中的 release ID 必须与标签名一致，否则失败。操作者不得重复输入另一份版本号。
- `--candidate` 必须是绝对路径；是否必需由发布 manifest 决定。
- 可选 `--work-root` 必须是仓库外的绝对路径；默认位于当前用户的 `CodexDownloads/lunar_navigation/agx_git_deploy`。
- 测试专用的临时根和命令替身只能通过显式测试接口注入，生产模式不得接受隐式模拟结果。

成功输出必须清楚显示：

```text
Installed: /opt/lunar_navigation/releases/<release-id>
Service state: not activated
Deployment report: <absolute-report-path>
To activate after qualification:
  sudo ./scripts/activate_agx_release.sh \
    --release-id <release-id> \
    --qualified-manifest <absolute-qualified-manifest-path>
```

激活工具要求显式的 qualified manifest 绝对路径，并验证其中的 release ID；安装报告不能替代设备验收报告。独立 release gate 通过后应输出完整激活命令，操作者不需要自行拼接路径。

## 7. 安装状态机

部署按以下顺序执行：

1. **Acquire lock**：取得全局部署锁，拒绝并发安装。
2. **Resolve source**：在仓库外创建暂存目录，拉取并检出精确标签。
3. **Verify identity**：验证 annotated tag、提交、候选 manifest 和文件 hash 一致。
4. **Preflight**：检查 aarch64、Ubuntu 22.04、ROS Humble、AGX 型号、L4T 基线、磁盘、Git、rosdep、编译器和外部 ROS 包。
5. **Resolve dependencies**：使用固定源码树执行 rosdep；系统包操作可以申请 sudo，但不以 root 运行编译器。
6. **Native build**：在仓库外的显式 build/log 根执行 aarch64 原生 `colcon build`，默认跳过训练目录和 `lunar_nav2_adapter`。
7. **Versioned install**：写入 `/opt/lunar_navigation/releases/<release-id>`，不覆盖其他版本。
8. **Model and engine**：发布需要模型时，验证四文件候选并在 AGX 本机生成、校验 TensorRT engine。
9. **Quick verify**：验证安装清单、ELF 架构、ROS 包、配置、模型和快速启动冒烟。
10. **Report**：写入部署 JSON 报告，释放锁，并输出独立激活命令。

任何阶段失败都必须停止后续阶段。安装器不得在此状态机中调用激活脚本或设备 release gate。

## 8. 权限模型

部署入口以普通用户运行。它可以提前执行一次 sudo 凭据检查，但只允许在以下操作使用提权：

- 创建单个明确的版本安装目录；
- 设置该目录的最小必要所有权和权限；
- 安装或更新明确的 systemd unit 文件；
- 写入版本化模型、engine cache 和系统日志目录。

`git`、`rosdep` 解析、CMake、编译器、测试和 TensorRT 构建进程不得整体以 root 身份运行。脚本不得放宽 `/opt/lunar_navigation`、`/var/lib/lunar_navigation`、`/var/cache/lunar_navigation` 或 `/var/log/lunar_navigation` 的全局权限。

## 9. 幂等性与失败处理

- 相同 `release-id` 已存在且安装 manifest/hash 完全一致时，返回“已安装”并重新执行只读快速校验。
- 相同 `release-id` 已存在但内容不同，立即失败，不能覆盖或修补原目录。
- Git、网络、依赖、构建、模型、engine 或快速校验失败时，`current` 和正在运行的服务保持不变。
- 失败暂存目录和日志保留，用于诊断；脚本不执行递归删除。
- 部署锁覆盖从源码解析到报告落盘的完整区间。
- 中断或异常必须在报告中留下最后完成阶段；重新执行遵守相同的版本冲突规则。

部署报告写入 `/var/log/lunar_navigation/deploy/<release-id>/<run-id>/deploy-report.json`，至少包含：schema、release ID、Git tag/commit、设备指纹 hash、候选 hash、各阶段状态、安装目录、日志目录、开始/结束时间和最终结果。Ubuntu 临时根中的模拟报告必须显式标记 `simulated: true`，不能被激活工具或 release gate 接受。

## 10. 激活与回退边界

安装成功后保持：

- `/opt/lunar_navigation/current` 不变；
- 模型 `current` 不变；
- `lunar-navigation.service` 不启动、不重启；
- 新版本只存在于版本目录中。

生产激活继续使用独立命令：

```bash
sudo ./scripts/activate_agx_release.sh \
  --release-id <release-id> \
  --qualified-manifest <absolute-qualified-manifest-path>
```

激活脚本必须重新验证安装目录、模型、engine 和 qualified manifest，记录上一 release/model，原子更新两个 `current` 链接，启动或重启服务并执行快速健康检查。失败时恢复原链接和旧服务；首次部署没有旧版本时，停止服务并撤销新建的 `current`。

一键安装与手动激活的分离不降低生产发布门槛。性能、功耗、异常和四小时稳定性验收仍由 `run_agx_release_gate.sh` 独立执行。

## 11. 无 AGX 条件下的验证

Ubuntu amd64 阶段通过以下方式验证工具设计：

- 临时 Git 仓库覆盖 annotated tag、错误 tag、提交不匹配和获取失败；
- 固定 fixture 覆盖合格与不合格 AGX 指纹；
- 命令注入替代 rosdep、colcon、TensorRT、systemd 和 sudo，校验顺序及参数；
- 临时安装根模拟 `/opt`、模型、cache 和日志路径；
- 覆盖依赖失败、构建失败、hash 不符、重复部署、版本冲突、部署锁和快速检查失败；
- 断言成功安装后 `current` 未变化、服务未启动且激活提示正确；
- 执行 shell 语法检查、pytest、仓库边界测试和 ARM64 编译预警。

Ubuntu 阶段的唯一允许结论为：

```text
AGX deployment tooling: simulated-ready
AGX native deployment: pending device verification
```

## 12. AGX 恢复后的短验收

AGX 可用后执行一次不含完整 release gate 的短验收：

1. 从真实 annotated tag 执行 Git 一键安装。
2. 检查 aarch64 原生文件和本机 TensorRT engine（如该发布需要模型）。
3. 确认安装完成后服务未自动启动，两个 `current` 链接未改变。
4. 直接针对版本化安装目录执行非 systemd 快速健康检查。
5. 记录设备指纹、安装报告以及“等待独立 release gate”的状态。

短验收不能替代规划性能、推理余量、功耗状态、故障矩阵或四小时稳定性门槛。只有独立 release gate 生成 qualified manifest 后，才能执行生产激活和回退验证。

## 13. 验收标准

- 从固定 Git 标签到版本化安装只需要一条部署命令。
- AGX 上所有生产二进制和 TensorRT engine 均为本机生成。
- 部署命令成功后服务保持未激活状态。
- 相同版本重复部署是幂等的，冲突内容不能覆盖。
- 任一失败都不改变当前运行版本，并生成可定位阶段的报告。
- Ubuntu 模拟测试能覆盖获取、预检、构建、安装、校验和失败边界。
- 没有 AGX 实测时，任何文档和报告均不声明设备已通过。
- U 盘部署不进入本次实现范围，但 Git 获取器与后续来源适配器之间保持单向、可替换边界。

## 14. 与现有计划的关系

本设计与卷三设备交接、卷四 Task 5–8 和总路线图的 AGX 顺序已协调：新增 Git 一键编排入口，复用 `install_agx.sh`、`verify_agx_install.sh`、`activate_agx_release.sh` 和 `run_agx_release_gate.sh`，并统一 `installed → qualified → activated`。它不改变 Ubuntu/AGX 权威职责、不改变版本目录和 `current` 回退模型，也不把 AGX 不可用从发布阻塞条件改写为已通过状态。

后续实施计划应先在 Ubuntu 完成编排器、fixture 和临时根测试；真实 AGX 命令只在设备恢复后执行。
