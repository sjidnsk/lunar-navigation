# Coverage Completion Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; final independent review after implementation and tests.

**Goal:** 实现并验证统一的测量侧覆盖下界结束条件，在采用前给出实际收益与限制。

**Architecture:** 新增纯测量剩余需求计算；TaskAnalyzer 给出面积下界及 completed/exhausted 两个状态；训练和部署共用，真值仅评估。

**Tech Stack:** Python/NumPy/SciPy、现有原生射线、ROS 2 Jazzy，既有训练 venv。

**Spec:** ../specs/2026-09-20-coverage-completion.md

## Global Constraints

- 独立测试分支从 dev/drl-exploration 派生，不合并、不推送，不改正式训练产物。
- 真实 E 和实际观测 K 不变；Actor、Critic、导航、控制和奖励权重不变。
- 默认目标 1.0；试验显式 0.99；旧回放不跨终止语义静默恢复。
- 临时日志/构建写 `/home/kai/.cache/lunar-drl-coverage-completion/`，无完整检查点副本。

## Review Focus

- 未知绕行途经工作区外，不能因有限工作区被当成不可达。
- 占据格不可站立但可能被观测，不能用 M 代替 B 遮挡。
- 小孔满足比例而仍有前沿，completed 与 exhausted 必须分开。
- 无观测且剩余需求未知、或原生起点不可用，不能由 0/0 变成完成；已证明空需求可耗尽结束，但百分比为 None。
- 旧评估把 terminated 称为 exhausted，需要更正报告而不改真值指标。

## Tasks

- [x] 1. 在 `test/test_coverage_completion.py` 先写比例结束、遮挡排除、未知外部连通与逐站位真值并集包含测试，验证旧版缺少功能而失败。
- [x] 2. 新增 `remaining.py` 消费潜在站位、输出 U；TaskAnalyzer 复用输出并增加 `remaining_area_upper_m2`、`coverage_lower_bound`、`coverage_target` 和 `completed`。默认保留严格耗尽。
- [x] 3. `DecisionCore`、`runtime.py`、`ros_env.py`、`worker.py`、配置/CLI 和评估端接入同一 completed；保留 exhausted 含义、预算截断；恢复身份绑定目标及算法版本。编写真实生命周期/转移测试后实现。
- [x] 4. 独立构建受影响包，运行包测试；在 `scripts/drl/` 提供可重放快照审核脚本，输出小 JSON；执行真实闭环对照及新增时延测量。
- [x] 5. 在 `docs/validation/2026-09-20-coverage-completion.md` 记录数据、失败、性能、适用边界与是否建议采用；完成独立复审与 UTF-8/diff 检查。

## Execution ledger

- 2026-09-20：读取源码/接口/操作文档，确认开发线 clean，创建独立 test worktree。用户已授权实施与验证，直接顺序执行，不增加审批轮次。
- 裁决：固定奖励与 gamma，完成奖金不在本轮落地。原因是先隔离统一判据影响；后续奖励对照须根据该判据验证结果另做。
- 裁决：`completed` 是正常结束的并集；A=0、S=0 可按严格耗尽结束，但不声称覆盖达标。该边界在报告中保留 None 百分比和独立的 reached_99；若误解命名会夸大空任务成功率，因此补齐文档及断言。
- 步骤 1–3：先观察缺少 API 的测试失败，完成纯测量 U、统一报告及训练/部署接入；全包初跑 308 通过、3 个旧测试约定失败、2 跳过。已更新终止/耗尽区别的测试约定，23 个相关测试通过。
- 额外边界：面积判据已满足时不依赖残余前沿的稀疏见证可用性。先写反例（A=99、S=1、见证缺失）观察失败，再修正报告可用性；不绕过缺失原生起点。
- 步骤 4：8 份冻结工作区、80 个随机小世界和 118 个实际 Jazzy 目标后状态均未发现真值剩余遗漏。闭环月表 true99.8055%/lower96.6918%，洞穴 true69.1996%/lower19.6399%，均未结束。
- 性能：同输入预热后3次计时，中位数月表100.49→125.28ms，洞穴6168.75→6113.74ms；未解决洞穴原有分析瓶颈。
- Final: fixed 碰撞与覆盖同时达标会计入正常完成率/显示覆盖达标 — 两项回归先失败后通过，保留空间事实、RL terminal/bootstrap 与现有物理重置。
- Final: Ruling: 评估JSON遗漏目标值升为复现正确性问题并补齐；否则脱离命令记录无法区分99%与严格耗尽实验。
- Final: minor (deferred): 快照审计真值缓存未绑定传感器身份；本轮全部10m/90°，混合传感器目录不在已验证范围。
- 适用前提：任意历史地图不能自动保证K属于当前E。审查给出跨阻挡带导入旧测量的反例；本轮不新增来源准入机制、不声称部署百分比已普遍成立。
- 最终验证：独立Jazzy包构建通过；完整包315 passed/2 opt-in skipped；独立安装后两项原生测试2 passed。独立审查完成；结果与采用限制写入validation。
- 采用裁决：实际失败收尾未改善，保持试验分支，不切换正式训练，不合并或推送。后续应先定位与收紧多算剩余区域。
