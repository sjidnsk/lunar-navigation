# Coverage-First Reward V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mis-scaled V2 PPO reward with the approved coverage-only V3 contract and invalidate V2 formal checkpoints.

**Architecture:** Keep all planner transition telemetry intact and make `reward.py` the only production behavior boundary that changes. Read final mission coverage from the already-frozen `pose_features[0, 4]`; encode the V3 schema and active weights in the reward hash so checkpoint identity changes without changing shared-memory transport.

**Tech Stack:** Python 3.10, PyTorch tensors, pytest, existing ROS-independent pybind bridge.

## Global Constraints

- Work only in `feature/formal-training-environment-closure`, which is already an isolated linked worktree.
- Do not modify `parallel_pool.py`, the planner, sensor geometry, platform capability, PPO architecture, gamma, or rollout horizon.
- Preserve path cost, macro time, fuel, and priority as telemetry; do not use them in reward.
- Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and source the matching ROS/bridge install before pytest.
- Do not resume a formal V3 run from a V2 checkpoint.

Use this environment for every Python test command in this plan:

```bash
source /opt/ros/humble/setup.bash
source /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/df27f02/native/install/setup.bash
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
```

---

### Task 1: Freeze Reward V3 behavior with RED tests

**Files:**
- Modify: `training/lunar_policy_training/tests/test_reward.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/reward.py`

**Interfaces:**
- Consumes: `PlannerTransition.next_observation.pose_features[0, 4]`, coverage deltas and planner outcomes.
- Produces: `RewardWeightsV3`, `RewardInputsV3`, `compute_reward()`, `compute_transition_reward()`, `reward_weights_sha256()`.

- [ ] **Step 1: Replace V2 assertions with hand-derived V3 behavior tests**

```python
assert compute_reward(_inputs(mission_observed_delta=0.02)) == 2.0
assert compute_reward(_inputs(mission_observed_delta=0.0)) == -0.10
assert compute_reward(_inputs(
    mission_observed_ratio_after=0.40,
    episode_ended_without_success=True,
    terminated=True,
)) == -0.70
```

Also assert telemetry invariance, `+5` first success, small planner rejection penalties,
invalid resource/safety samples, and one platform-independent V3 hash.

- [ ] **Step 2: Run RED**

```bash
"$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_reward.py
```

Expected: fail because `RewardWeightsV3` and V3 behavior do not exist.

- [ ] **Step 3: Implement the minimal V3 reward**

```python
REWARD_SCHEMA_VERSION = "lunar-reward/v3"

@dataclass(frozen=True, slots=True)
class RewardWeightsV3:
    mission_observed_delta: float = 100.0
    executed_without_new_mission_coverage: float = 0.10
    goal_infeasible: float = 0.20
    no_known_safe_route: float = 0.30
    unsuccessful_remaining_coverage: float = 1.00
    success_first_crossing: float = 5.00
```

Reject non-learnable outcomes and hard-constraint violations before computing the approved
formula. Include schema plus weights in the SHA-256 payload.

- [ ] **Step 4: Run GREEN**

Run the exact Task 1 Step 2 command. Expected: all reward tests pass.

### Task 2: Verify checkpoint identity and direct consumers

**Files:**
- Modify only if a direct assertion requires it:
  `training/lunar_policy_training/tests/test_training_smoke.py`
- Verify: `training/lunar_policy_training/tests/test_formal_evaluation.py`
- Verify: `training/lunar_policy_training/tests/test_formal_preflight.py`
- Verify: `training/lunar_policy_training/tests/test_parallel_pool.py`

**Interfaces:**
- Consumes: `reward_weights_sha256()` and `compute_transition_reward()`.
- Produces: evidence that existing run/checkpoint gates pick up V3 without transport changes.

- [ ] **Step 1: Run direct integration tests**

```bash
"$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_reward.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_formal_evaluation.py \
  training/lunar_policy_training/tests/test_formal_preflight.py \
  training/lunar_policy_training/tests/test_training_smoke.py
```

- [ ] **Step 2: Run repository checks and diff validation**

```bash
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git diff --check
```

- [ ] **Step 3: Record the old run boundary and start no V3 formal run until verification is green**

The preserved V2 artifact ends at update 906. A future V3 launch must create a new artifact
root and initialize model/optimizer state instead of using its `latest.pt`.
