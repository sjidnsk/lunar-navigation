# Ground Global Opportunity Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 WHEELED/LEGGED 从整幅 observed-only 物理可达区域选择远端正收益目标，并锁定该目标滚动执行到达，不再受传感器 `30 m` 或固定 64-reference 隐式范围限制。

**Architecture:** 保留 C++ 地面整图 connected-component authority 和 4 m 候选位置；Python 用无漏检的粗几何预筛缩小远端站位集合，再用既有 0.2 m visibility kernel 精确计算收益。独立 Oracle 扫描同一整图物理域但独立估算收益。地面宏动作以 `while` 滚动到目标，并用 observation identity、route cursor 和 reference endpoint 的重复签名检测真实卡死。

**Tech Stack:** Python 3.10、NumPy、pytest、C++20、pybind11、ROS 2 Humble、colcon/CMake、Git。

## Global Constraints

- 不读取、修改或暂存已有 dirty 文件 `docs/superpowers/plans/2026-08-11-physical-opportunity-planner-failure-semantics.md`。
- `sensor_range_m` 保持 `30.0 m`；只删除地面候选/Oracle 的目标距离截断。
- 4 m global known、0.2 m detail LOS、固定 2 m corridor/search-domain 和 Hopper 单跳合同不变。
- 地面普通零收益站位不得进入 universe/reserve/batch；只保留 `zero_gain_count`。
- 不逐候选调用 Planner；visibility 必须批处理，并先做无漏检的 unknown-ROI 距离预筛。
- WHEELED/LEGGED 单次规划继续满足 `< 1 s`；timeout 不得当作合法探索终止。
- 修改中文 Markdown 后按 UTF-8 读取；每次提交前运行 `git diff --check`。

---

### Task 1: 全局地面候选与独立 Oracle

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py`
- Test: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Test: `training/lunar_policy_training/tests/test_frontier_oracle.py`

**Interfaces:**
- Consumes: `PhysicalReachabilityResult.physical_observation_pose_mask` 和 `observation_positions_m`，地面 mask 已表示整幅 start-connected physical component。
- Produces: 地面全局正收益 `PhysicalCandidateUniverse`；Oracle 对整幅地面物理域的独立 `FrontierOracleResult`。

- [ ] **Step 1: 写远端机会 RED**

在 candidate/oracle 测试中构造机器人距正收益站位 `>30 m` 的 WHEELED/LEGGED fixture；断言 production universe 和 Oracle 均包含该机会。再构造近端全零、远端正收益，断言不会 `ZERO_GAIN`。

- [ ] **Step 2: 运行 RED**

Run:

```bash
cd training/lunar_policy_training
python -m pytest -q \
  tests/test_candidate_builder_v2.py -k 'ground_global or remote_ground' \
  tests/test_frontier_oracle.py -k 'ground_global or remote_ground'
```

Expected: 当前 `_candidate_within_sensor()` 与 Oracle `within_range` 过滤导致远端断言失败。

- [ ] **Step 3: 实现无漏检粗预筛与精确收益**

在 `candidate_builder.py` 增加私有函数，输入 coarse `unknown_roi`、分辨率和传感器半径，返回“圆盘可能接触未知 ROI”的布尔 mask。偏移半径必须包含 coarse 半对角线，保证只排除必然零收益位置。地面 `raw_anchors` 使用整图 physical-safe 候选与该 mask 的交集，不再调用相对机器人 30 m 过滤；Hopper 路径保持不变。

将 `_qualify_anchors(... include_zero_gain=platform_type == "HOPPER")`，保证地面全零站位只累加 `zero_gain_count`。Oracle 独立构造其粗预筛并删除 `distance_m <= 30 m`，之后继续独立调用 visibility estimator。

- [ ] **Step 4: 运行定向 GREEN 与候选全文件测试**

Run:

```bash
cd training/lunar_policy_training
python -m pytest -q tests/test_candidate_builder_v2.py tests/test_frontier_oracle.py
```

Expected: 全部通过；Hopper remote transit 用例不回归。

- [ ] **Step 5: 提交**

```bash
git add training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py \
  training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_frontier_oracle.py
git diff --cached --check
git commit -m "fix(training): expose global ground opportunities"
```

### Task 2: 无固定 reference 上限的锁定远目标

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Test: `training/lunar_policy_training/tests/test_v3_environment.py`

**Interfaces:**
- Consumes: `PlannerOutput.continuation`、`diagnostics.hierarchical.route_cursor`、`MotionReference` endpoint 和最新 `ObservationIdentity`。
- Produces: 一个 policy action 对应一个可超过 64 references 的聚合 `PlannerTransition`；重复结构签名抛 `EnvironmentInvariantError("GROUND_OPTION_STALLED")`。

- [ ] **Step 1: 写长目标与卡死 RED**

新增用例：65 个有效 reference 后第 66 次进入目标容差，断言只消费一次 policy decision、汇总全部 gain/cost/time 且不报 64 上限。新增重复 identity + route cursor + endpoint 用例，断言 fail-closed 为 `GROUND_OPTION_STALLED`。保留合法 detour 距离暂时增加的既有测试。

- [ ] **Step 2: 运行 RED**

Run:

```bash
cd training/lunar_policy_training
python -m pytest -q tests/test_v3_environment.py \
  -k 'more_than_64 or stalled or detour'
```

Expected: 长目标报 `ground option did not finish within 64 references`；卡死测试尚无对应异常。

- [ ] **Step 3: 实现滚动循环和重复签名保护**

删除 `_MAX_GROUND_OPTION_REFERENCES`，把 `_advance_ground_option()` 改为 `while True`。每次成功执行后构造包含最新 observation identity、route cursor、reference endpoint 和目标 candidate identity 的不可变签名；只有完整签名在无状态推进时重复才抛 `GROUND_OPTION_STALLED`。不得恢复逐段欧氏距离单调约束。

- [ ] **Step 4: 运行完整 v3 环境测试**

Run:

```bash
cd training/lunar_policy_training
python -m pytest -q tests/test_v3_environment.py
```

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py \
  training/lunar_policy_training/tests/test_v3_environment.py
git diff --cached --check
git commit -m "fix(training): execute locked ground goals to completion"
```

### Task 3: 身份升级与 fail-closed 迁移

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/coverability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify only where constants are consumed: `training/lunar_policy_training/lunar_policy_training/formal_preflight.py`, `training/lunar_policy_training/lunar_policy_training/closed_loop_gate.py`, `training/lunar_policy_training/lunar_policy_training/cli.py`
- Test only matching identity assertions under `training/lunar_policy_training/tests/`

**Interfaces:**
- Produces: `semantics-v12`、formal cache `v7`、checkpoint `v8` 及引用这些版本的 manifest/preflight/gate 合同。

- [ ] **Step 1: 把身份断言更新为新版本并验证 RED**

将版本单元测试期望更新为 `v12/v7/v8`，并新增旧 `v11/v6/v7` fail-closed 断言。

- [ ] **Step 2: 运行身份 RED**

Run:

```bash
cd training/lunar_policy_training
python -m pytest -q \
  tests/test_coverability.py tests/test_formal_cache.py \
  tests/test_formal_preflight.py tests/test_training_smoke.py \
  tests/test_closed_loop_gate.py
```

Expected: 旧生产常量与新断言不一致。

- [ ] **Step 3: 最小升级生产常量和消费者**

只修改版本常量、严格 loader/preflight/gate 比较和相应 manifest 输出；不改变 capability schema、ROS 消息或 Hopper 算法身份。

- [ ] **Step 4: 运行身份 GREEN**

重复 Step 2 命令，Expected: 全部通过且旧版本被明确拒绝。

- [ ] **Step 5: 提交**

```bash
git add training/lunar_policy_training/lunar_policy_training \
  training/lunar_policy_training/tests
git diff --cached --check
git commit -m "feat(training): advance global ground target identities"
```

提交前必须用 `git diff --cached --name-only` 核对仅包含本任务身份文件和测试，不得纳入 protected plan 或无关测试。

### Task 4: 必要回归与闭环资格

**Files:**
- No production changes unless a load-bearing gate exposes a directly相关 root cause。
- Artifacts: repository external `final/` build/cache/gate roots。

**Interfaces:**
- Produces: fresh ROS、Python、cache、replay、三平台闭环和 deterministic gate 证据。

- [ ] **Step 1: fresh Release ROS/C++ build 与全量测试**

Source `/opt/ros/humble/setup.bash`，在 repo 外 fresh build/install/log 构建必要 messages、planner core、bridge、ROS 包；运行全部 ROS tests 和 `colcon test-result --verbose`，要求 errors/failures 为零。

- [ ] **Step 2: Python 全量 regression**

使用项目冻结 venv、fresh bridge 和正式 PYTHONPATH 运行完整 `pytest`；不允许 skip 新增 load-bearing 用例。

- [ ] **Step 3: fresh cache/identity/historical replay**

按新 `cache-v7` 生成 fresh cache，验证 schema/manifest SHA、strict identity 和 historical replay；旧 cache 必须 fail-closed。

- [ ] **Step 4: 三平台独立完整闭环**

WHEELED、LEGGED、HOPPER 使用独立 OS 进程；某平台失败只停止该平台，siblings 继续。记录 coverage 曲线、terminal reason、candidate/Oracle mismatch、planner/execution/safety counters。地面每次规划必须 `<1 s`。

- [ ] **Step 5: 两次 deterministic gate**

在 clean linked worktree 对同一 HEAD、同一 cache、同一配置运行两次正式 gate，比较报告和关键 artifact hash。任何真实失败均先系统定位，禁止阈值放宽、skip、旧 artifact 或 watchdog 伪通过。

- [ ] **Step 6: 最终提交审计**

运行 `git status --short`、`git diff --check`、UTF-8 文档读取和提交列表核对；确认工作树只剩用户受保护 dirty plan，并汇报所有 commit、测试总数、artifact SHA 和尚未通过的真实 gate（如有）。
