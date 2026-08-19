# Fine Ground Candidate Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the ground PPO interface at at most three candidates per frontier segment while removing 4 m parent-cell rejection and recovering only a failed segment's complete frontier witnesses.

**Architecture:** The native bridge owns an immutable ground global-cost-tree context and answers exact endpoint batches without a second global search. `CandidateBuilderV2` first certifies the three-anchor safe-belt witnesses, then invokes a separate complete-segment provider only for segments with zero qualified positive witnesses. There is deliberately no task-wide residual or paged scan.

**Tech Stack:** C++20, pybind11, Python 3.10, NumPy, pytest, ROS 2 Humble.

**Spec:** `docs/superpowers/specs/2026-08-19-fine-ground-candidate-completeness-design.md`

## Global Constraints

- No Oracle, coverability denominator, unknown truth, PPO-shape, Reward V4, HOPPER, sensor-range or planner-safety changes.
- Exactly one global cost tree per ground snapshot; endpoint queries are batched against its immutable context.
- A segment emits at most three candidates; level two is only a complete scan of a level-one-zero-positive segment.
- No full-task residual scan, no page cursor and no new checkpoint state for one.
- Build/install/test artifacts stay under `/home/kai/CodexDownloads/lunar_navigation/`, never in the repository.

---

### Task 1: Native exact endpoint context

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/reachability_projection.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/shared/reachability_projection.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/reachability_projection_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/request.hpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/conversions.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Test: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`

**Interfaces:**

```cpp
GroundEndpointReachabilityContextResult ProjectGroundEndpointReachabilityContext(
    const PlannerInput& input, double maximum_edge_distance_m);
GroundExactEndpointProjectionResult QueryGroundExactEndpoints(
    const GroundEndpointReachabilityContext& context,
    std::span<const Vec3> positions_map, double tolerance_m);
```

- [ ] **Step 1: Write RED core and bridge tests**

```cpp
const auto context = ProjectGroundEndpointReachabilityContext(input, 30.0);
const auto endpoints = QueryGroundExactEndpoints(
    *context.context, {{.x = 4.9, .y = 3.5, .z = 0.0}}, 0.2);
EXPECT_EQ(endpoints.projection->reachable.at(0), 1U);
EXPECT_TRUE(std::isfinite(endpoints.projection->minimum_cost_m.at(0)));
```

```python
context = bridge.project_ground_endpoint_context(request, 30.0)
query = bridge.query_ground_exact_endpoints(context, positions, 0.2)
assert query.reachable.tolist() == [True, False]
```

- [ ] **Step 2: Run RED tests**

Run the named native and bridge tests. Expected failure: the context/query symbols do not exist.

- [ ] **Step 3: Implement minimal immutable context**

Construct one map, safety projection and `SearchGlobalGridCostTree`; retain them in the context. For each point, enumerate only global cells whose closed rectangles meet the 0.2 m target disk, select a hard-feasible finite-cost cell deterministically by `(cost, cell_index)`, and return an explicit code for outside, unsafe or unreachable points.

- [ ] **Step 4: Verify GREEN and commit**

Build `lunar_planner_core` and `lunar_planner_training_bridge` in an external Release prefix; run the focused core and bridge tests. Commit only the six source/test files with `feat(training): query exact ground endpoints in one tree`.

### Task 2: Use exact endpoint authority in first-level candidate qualification

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/platform_reachability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Test: `training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`
- Test: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`

**Interfaces:**

```python
def query_exact_ground_endpoints(targets_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return C-contiguous bool [N] and float64 [N] from the native context."""
```

- [ ] **Step 1: Write RED regression**

```python
def test_detail_witness_uses_exact_endpoint_query_not_unreachable_4m_parent():
    universe = builder.build_physical_universe(
        ..., ground_detail_candidate_provider=detail_provider,
        ground_exact_endpoint_reachability=exact_query,
    )
    assert [c.target_position_m for c in universe.candidates] == [fine_target]
```

The fixture must set the old `physical_observation_pose_mask` and sampled 4 m cost false/infinite while the exact query accepts the fine target.

- [ ] **Step 2: Run RED**

Run the named test. Expected failure: current code rejects the detail witness before it can invoke the exact query.

- [ ] **Step 3: Implement minimal source fix**

Retain the legacy coarse gate only for legacy coarse providers. For detailed witnesses, call the native exact endpoint batch, then existing observed-only 0.2 m endpoint check, then exact gain. Use query costs in candidate features; validate shapes, contiguity, finite/`+inf` conventions and fail closed on malformed native output.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused candidate suites against the external native prefix and commit the scoped Python source/tests with `fix(training): certify fine ground endpoints exactly`.

### Task 3: Complete-frontier recovery for zero-positive segments

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Test: `training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`
- Test: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`

**Interfaces:**

```python
ground_detail_segment_recovery_provider(
    segments: list[list[tuple[int, int]]],
    world: ObservedWorld,
    zero_positive_segment_ids: frozenset[int],
) -> Collection[tuple[int, _RawFrontierCandidate]]
```

- [ ] **Step 1: Write RED regressions**

```python
def test_complete_segment_recovery_runs_only_after_three_anchor_zero_positive():
    assert recovery_calls == [frozenset({0})]
    assert len(segment_zero_positive_candidates) == 3

def test_positive_first_level_segment_never_enters_complete_segment_recovery():
    assert recovery_calls == []
```

- [ ] **Step 2: Run RED**

Run both tests. Expected failure: no recovery provider exists and a positive witness outside three fixed anchors is omitted.

- [ ] **Step 3: Implement minimal stage-two provider**

Keep first-level provider unchanged. When its batch contains no positive witness for a segment, call the recovery provider only for that segment; `FormalEpisode` enumerates every coarse cell of that segment through the same bounded 0.2 m strip selector. Batch-certify its results through the already-created exact context, endpoint callback and gain evaluator; then apply the existing deterministic max-three compression.

- [ ] **Step 4: Add no-level-three guard and verify GREEN**

Add a regression that monkeypatches the removed full-task exhaustion scanner to throw, then execute the normal and stage-two paths. Run focused candidate/formal-builder tests. Commit with `fix(training): recover complete zero-positive frontier segments`.

### Task 4: Minimal qualification and atomic training recovery

**Files:**
- Modify: the specification and plan above only to record exact test/build evidence.
- Runtime: external build/install and training service files only.

- [ ] **Step 1: Build external Release prefix**

Build only core and bridge into a new `/home/kai/CodexDownloads/lunar_navigation/fine-ground-endpoints-{build,install}` prefix. Do not reuse a mismatched source/build CMake cache.

- [ ] **Step 2: Run bounded verification**

Run focused core reachability, bridge, candidate and formal-builder tests; run `python3 -m py_compile` for touched Python files and `git diff --check`. Do not launch a full-scene gate.

- [ ] **Step 3: Cut over atomically**

Record HEAD and external install SHA, point the training runtime at the new source and native prefix at an already committed checkpoint boundary, then start fresh worker episodes. Verify service PID, resolved source SHA, restored global step and first worker heartbeats; retain the prior runtime pointer for rollback.

- [ ] **Step 4: Commit evidence-only documentation update**

Stage only source, tests and the two documentation files. Do not commit cache, build, checkpoint, journal or runtime artifacts.
