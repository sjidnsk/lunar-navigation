# Incremental Complex-Terrain Performance Optimization Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task by task. Use `superpowers:test-driven-development` for every production change and `superpowers:verification-before-completion` before claiming success.

**Goal:** Improve the four measured hot paths without changing planning safety semantics or moving responsibilities between planner modules, and retain an optimization only when an identical representative workload demonstrates a stable performance gain.

**Architecture:** Keep the existing `FineTraversabilityBuilder`, phase-aware simplifier, `LeggedDirectedEdgeCache`, and `GlobalRoutePlanner::Impl` boundaries. Optimize request-local data layout, reduce redundant work before existing safety certification, and extend the existing route cache with a validated suffix reuse case. Do not change map resolution, UNKNOWN/blocked treatment, vehicle envelopes, planner heuristics, coordinator behavior, or ROS interfaces.

**Tech Stack:** C++20, CMake/CTest, GoogleTest, ROS 2 Jazzy development environment. Builds and measurement artifacts stay outside the repository.

**Approved design:** `docs/validation/2026-09-04-incremental-complex-terrain-benchmark.md`, section “按收益和风险排序的优化方案”. The user's “按建议优化” is the implementation approval for that design.

## Measurement and retention gate

- Freeze `/tmp/lunar-complex-final.uyoh2c/build/lunar_incremental_navigation_core_complex_terrain_benchmark` as the before binary; never rebuild that directory.
- Build optimized code in a distinct external `RelWithDebInfo` directory.
- Run each targeted scenario serially with identical CLI arguments and at least three iterations; compare p50, not the best sample.
- Require unchanged scenario/status and relevant safety/path contracts. Require at least 10% targeted p50 improvement, supported by a deterministic work-count or cache-reuse change where one exists. If the result is borderline, repeat five times. Rework or revert an optimization that fails this gate.
- Record raw CSV, command, host boundary, p50, delta, semantic counters, and retained/reverted decision in the validation document.

### Task 0: Freeze and verify the benchmark baseline

**Files:**
- Modify: `docs/validation/2026-09-04-incremental-complex-terrain-benchmark.md`
- Existing benchmark: `ros2_ws/src/lunar_incremental_navigation_core/benchmark/complex_terrain_benchmark_main.cpp`

- [x] Re-run package CTest against the frozen fresh build and verify the worktree diff.
- [x] Collect three-run 640x640/0.1 baseline CSV for `risk-unknown`, `dead-ends`, and `legged-step-gap` without parallel test load.
- [x] Extract targeted p50 and correctness counters into a machine-readable scratch summary outside the repository.
- [ ] Commit only the already-validated benchmark harness and its documentation with explicit paths.

### Task 1: Replace fine-cell ordered-map scratch cache

**Files:**
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/include/lunar_incremental_navigation_core/fine_traversability_builder.hpp`
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/include/lunar_incremental_navigation_core/traversability_snapshot.hpp`
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/src/map/platform_elevation_physics.cpp`
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/src/map/fine_traversability_builder.cpp`
- Test: `ros2_ws/src/lunar_incremental_navigation_core/test/fine_traversability_builder_test.cpp`

- [ ] RED: add a behavior/metric test proving one dense derive reuses a bounded number of request-local cache tiles while examining the same unique elevation cells.
- [ ] GREEN: replace per-cell `std::map<GridIndex, IntrinsicTraversalEvaluation>` lookups with lazily allocated tile-indexed scratch storage and a populated bitmap; retain exact evaluator calls and state/cost behavior.
- [ ] Run fine builder tests plus complex-terrain scenario tests.
- [ ] Compare frozen versus optimized `risk-unknown` full/single-cell/patch p50 and updated/examined-cell counts. Retain only if the gate passes.
- [ ] Commit the fine optimization and its regression test with explicit paths.

### Task 2: Compress collinear vertices before phase-aware LOS simplification

**Files:**
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/src/local/phase_aware_path_simplifier.cpp`
- Test: `ros2_ws/src/lunar_incremental_navigation_core/test/phase_aware_path_simplifier_test.cpp`

- [ ] RED: add a real-path test with long collinear runs and blocked shortcuts; assert endpoints, phase boundary, safe bend retention, and cancellation/deadline fail-closed behavior.
- [ ] GREEN: within each phase, linearly remove only same-direction collinear middle vertices, then run the existing farthest-visible greedy and existing supercover/`Allowed()` certification.
- [ ] Run simplifier and wheel planner tests plus complex-terrain scenario tests.
- [ ] Compare frozen versus optimized 640 `dead-ends` wheel total/postprocess p50, raw/final point counts, and status. Retain only if the gate passes.
- [ ] Commit the simplifier optimization and regression test with explicit paths.

### Task 3: Add cheap gates before legged directed-edge certification

**Files:**
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/include/lunar_incremental_navigation_core/legged_local_planner.hpp`
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/src/legged/legged_local_planner.cpp`
- Test: `ros2_ws/src/lunar_incremental_navigation_core/test/legged_local_planner_v2_test.cpp`

- [ ] RED: add a deterministic large-grid test proving far terminal probes and non-improving/closed neighbors do not multiply primitive evaluation work while the returned path remains executable.
- [ ] GREEN: precompute legal translation/spin capability summaries, range-reject impossible terminal edges, construct/check neighbor state and a geometric lower bound before calling the existing directed-edge cache, and use allocation-free visitation for short neighbor segments where semantics remain identical.
- [ ] Run all legged directed-edge, start-prefix, terminal-yaw, planner, and complex-terrain scenario tests.
- [ ] Compare frozen versus optimized 640 `legged-step-gap` legged p50, evaluated transitions, expanded/generated states, raw/final point counts, and status. Retain only if the gate passes.
- [ ] Commit the legged optimization and regression test with explicit paths.

### Task 4: Reuse a validated global-route suffix for moved starts

**Files:**
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/src/global/global_route_planner.cpp`
- Test: `ros2_ws/src/lunar_incremental_navigation_core/test/global_route_planner_v2_test.cpp`
- Test: `ros2_ws/src/lunar_incremental_navigation_core/test/complex_terrain_benchmark_test.cpp`

- [ ] RED: change/add cache tests so a moved start on the cached route must reuse the suffix with zero expanded states, while off-route start and on-route revision changes still force search.
- [ ] GREEN: after matching goal/profile/geometry and validating revision influence, locate the new start in cached cells and build from that suffix; keep the full-search fallback unchanged.
- [ ] Run global route/cache tests plus complex-terrain benchmark tests.
- [ ] Compare frozen versus optimized 640 `dead-ends` moved-start p50, `cache_reused`, expanded states, and route endpoints. Retain only if the gate passes.
- [ ] Commit the cache optimization and regression test with explicit paths.

### Task 5: Integrated validation and evidence update

**Files:**
- Modify: `docs/validation/2026-09-04-incremental-complex-terrain-benchmark.md`
- Modify as needed: `README.md`, `docs/操作指令.md`

- [ ] Run `git diff --check`, the package CTest suite serially, and relevant Python static contracts.
- [ ] Repeat the complete 320 matrix and targeted 640 benchmarks from the final optimized binary.
- [ ] Verify formal result/status contracts and report inherited unrelated failures separately.
- [ ] Document each optimization's before/after p50, percentage, deterministic counters, and retained/reverted decision; mark Humble, Orin, DDS, rosbag, and vehicle validation `NOT_RUN`.
- [ ] Review the final diff for boundary violations, then use `superpowers:finishing-a-development-branch` without merging or pushing unless the user explicitly requests it.
