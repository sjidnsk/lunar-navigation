# 强化学习开发线整合验证

## 结果与范围

按用户“先整合出一条强化学习开发分支，然后再进行实验”的要求，复用
`drl-exploration-redesign` 工作树，并将原 `feat/drl-exploration-redesign` 改名为
`dev/drl-exploration`，作为唯一活动强化学习开发整合线。层级为集成主线 → 开发整合线 → 专项分支；
后续专项代码在子分支实现后合回，不要求全部直接在开发线实现。
当前入口是 [DRL开发入口](../DRL开发入口.md)。没有创建新的工作树或复制构建/模型目录。

整理前 HEAD=`5a208212`，当前工作树有60个修改或未跟踪文件。本次将已确认并实现的
路径尺度决策图、候选对应 Critic、有界 Actor 可选实验、配对驱动、终端显示、恢复支持、
配置及对应测试/文档固定成一个可复查的代码基线。该60文件中的代码在本次整理期间没有再次修改；
仅更新入口文档、历史实验状态说明和过期导出/推理命令。

公共导航修正已在此线提交：`0f236adc`（路径简化器）、`1fe59144`（净空代价）、
`aac4e38f`（测试与证据）、`5a208212`（实际安装运行验证）。简化器源文件与集成主线逐字节相同，
轮式默认净空权重为2.0。无需从历史试验工作树注入动态库。本次在固定模块基线后合并
`integration/pure-planner-orin` 的 `6868236f`，建立实际祖先关系；共享修正已经等效存在，
合并不重复引入旧训练算法。集成主线自身不移动，不把 DRL 专用代码反向合入主线。

## 历史来源与保留检查

| 工作树 | HEAD | 本次开始时逐文件未提交项 | 处理 |
| --- | --- | ---: | --- |
| drl-exploration-training | de54573a | 81 | 保留旧循环 SAC 实现与工作区，不向新图策略混入旧协议 |
| observability-coverage | df469d16 | 173 | 保留旧覆盖实现与工作区，不把祖先 HEAD 误当成全部改动已合入 |
| drl-exploration-complete-design | a0f190c8 | 0 | 保留早期设计历史 |
| exploration-coverage-definition | 036ad64a | 0 | 保留早期覆盖定义历史 |

计数使用 `git status --porcelain=v1 -z --untracked-files=all`，因此与折叠未跟踪目录的普通
`git status --short` 项数可能不同。整合时重新核对以上 HEAD、状态列表以及未提交文件 SHA256，
均未改变。本次不是将所有历史分支无差别 merge；旧模型语义不被重新启用，也不声称旧工作区全部合入。
其他集成、实车、TCP 工作树不属于本次写入范围。

## 本次重新执行的验证

环境：本机 x86_64、ROS 2 Jazzy；统一缓存 `/home/kai/.cache/lunar-drl-redesign/jazzy`。

- `scripts/drl/build.sh`：6包构建通过。
- 训练 venv、统一 overlay 下执行 `python -m pytest -q -p no:cacheprovider ros2_ws/src/lunar_drl_exploration/test`：**281 passed，0 skipped，71.38秒**。
  显式启用 `LUNAR_DRL_RUN_ROS=1`，包括两个真实原生导航/公共控制器/运动学测试；GPU同步审计也执行。
  使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、OMP/OpenBLAS/MKL线程4。
- core CTest：**16/16通过**，4.71秒；ROS CTest：**14/14通过**，3.01秒。两个包依次 `ctest --output-on-failure -j1`，ROS域229。
- `scripts/drl/train.sh --help` 与 `scripts/drl/compare_actor.sh --help`：实际统一安装入口可加载。
- UTF-8读取、`git diff --check`、文档相对链接及提交范围检查。

本次未启动训练或新的算法对照；无运行中的训练/配对驱动。既有
`training-output/actor-bounded-20260919/paired-state.json` 为 interrupted，旧比较未完成。
导航条件已更新，新比较另用输出目录；旧检查点和回放不删除。约7.5 GiB原训练目录未搬迁，
不生成重复模型或回放副本。

证据目录：`/home/kai/.cache/lunar-drl-consolidation-20260919/`，包含分支/文件快照、
`build.log`、`drl-tests.log`、`navigation-tests.log` 与模型文件位置/大小/时间清单。
这些审计日志不提交；源码、配置、测试和正式文档提交到统一开发线。

## 验证边界

此处通过意味着代码可构建、接口和行为回归测试通过，并已固定统一开发入口。
两节点循环改善、长期收敛、跨尺度泛化及新的公平配对结果仍待实验。
Humble、Orin、实车、TCP物理动态闭环：本次 `NOT_RUN`。
