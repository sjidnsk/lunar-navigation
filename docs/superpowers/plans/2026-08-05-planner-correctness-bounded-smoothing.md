# Planner Correctness and Bounded Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct hopper landing-area semantics, anchor wheel and legged references to the true state, add bounded locally certified smoothing, and publish explicit execution evidence without changing the ROS Action or MotionReference schemas.

**Architecture:** Hopper global planning builds one conservative O(N) landing-support distance field and discovers at most 128 reachable landing nodes lazily; the first hop remains subject to full L0 certification. Wheel and legged searches start from one virtual continuous state whose first primitive is sweep-certified before entering the lattice. Smoothing is restricted to the already-certified 3–4 m local segment and is accepted only after bounded continuous revalidation; otherwise the planner either exposes a certified discrete fallback or fails when smoothed execution is required. Internal diagnostics are mapped to standard `/diagnostics` key/value pairs.

**Tech Stack:** C++20, ROS 2 Humble, ament/colcon, GoogleTest, Python 3.10, pytest, deterministic grid algorithms.

## Global Constraints

- Preserve `Planner::Plan(const PlannerInput&) noexcept`, `PlanMotion.action`, `MotionReference.msg`, public Topic names, and the separation between map-frame preview and odom-frame execution.
- Never use a single cell's area as evidence for `minimum_landing_region_area_m2`; certify a continuous support radius `sqrt(area / pi)` against all unsafe cells.
- Keep hopper preprocessing O(N) time and O(N) memory, with cancellation checks in every long pass. Candidate discovery remains bounded by 128 nodes and out-degree 8.
- `position_tolerance_m` constrains the hopper aim/landing point, not the full landing-region polygon.
- Wheel and legged references begin at the true frozen pose and yaw. A first connector that has not passed the existing continuous sweep validator must never be emitted.
- Smooth only the current local execution segment. Enforce 64 control points, 512 samples, 128 optimizer iterations, 8 trust-region reductions, and 32 continuous-validation subdivisions as hard ceilings.
- Default `require_smoothed_execution` is false. A failed optimizer may only fall back to a previously certified discrete reference, and the fallback must be visible in warnings and diagnostics.
- Do not relax the established p95 gates or the 256 MiB global work-memory limit. Do not use wall-clock timeouts to change a planning conclusion.
- Source ROS with nounset disabled: `set +u; source /opt/ros/humble/setup.bash; test "$ROS_DISTRO" = humble`.
- Keep build, install, test-result, benchmark, and evidence artifacts under `/home/kai/CodexDownloads/lunar_navigation/planner_correctness/`, never in the repository.
- Preserve unrelated user changes and the original repository's untracked `.vscode/` directory.

## Stable Interface Additions

Add these internal/public-core values without changing ROS messages:

```cpp
enum class TrajectoryMode {
  kStationary,
  kOptimized,
  kDiscreteFallback,
  kCertifiedHop,
};

enum class CollisionValidation {
  kCertified,
  kNotApplicable,
};

struct LocalTrajectoryDiagnostics final {
  TrajectoryMode trajectory_mode{TrajectoryMode::kStationary};
  double start_anchor_error_m{0.0};
  double endpoint_error_m{0.0};
  double maximum_curvature_per_m{0.0};
  CollisionValidation collision_validation{CollisionValidation::kNotApplicable};
  double smoothing_elapsed_s{0.0};
  double landing_field_elapsed_s{0.0};
};
```

Configuration defaults:

```cpp
std::size_t maximum_smoothing_control_points{64U};
std::size_t maximum_smoothing_samples{512U};
bool require_smoothed_execution{false};
```

---

### Task 1: Replace the hopper single-cell area check with a conservative landing-support field

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/landing_support_field.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/landing_support_field.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_route_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`

**Interfaces:**
- Consume: `shared::MapSnapshot`, hopper capability limits, map safety settings, and `std::stop_token`.
- Produce: row-major `LandingSupportField` containing `landing_base_safe`, squared distance-to-unsafe, `landing_center_safe`, elapsed time, and deterministic status/reason.

- [ ] **Step 1: Add focused failing tests for physical support area**

Add fixtures at 0.2 m resolution for: an open region supporting 1.327322 m²; a safe single cell surrounded by unsafe cells; a support disk cut by one obstacle; start-cell support insufficiency; goal candidates inside 0.05 m tolerance; and repeated deterministic output. Assert the open map succeeds even though `resolution * resolution < minimum_landing_region_area_m2`.

- [ ] **Step 2: Prove the regression with the existing implementation**

Run:

```bash
set +u
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/log build \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/install \
  --cmake-args -DBUILD_TESTING=ON
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/log test \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/install \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/test-results \
  --ctest-args -R hopper_route_planner
```

Expected: the new open-region assertion fails because the current code compares one cell's 0.04 m² with 1.327322 m².

- [ ] **Step 3: Implement the base-safe mask and exact two-pass squared Euclidean distance transform**

Use the Felzenszwalb-Huttenlocher one-dimensional lower-envelope transform for rows and columns, with fixed row-major traversal and preallocated vectors. Treat the map exterior and every invalid, obstacle, forbidden, terrain-infeasible, or clearance-infeasible cell as unsafe. Check `stop_token` at least once per row/column and return the existing cancel result without partial success.

- [ ] **Step 4: Convert support area to a conservative center-safe decision**

Compute `required_radius_m = sqrt(minimum_landing_region_area_m2 / std::numbers::pi)`. Convert center-to-center distance to a lower bound on center-to-unsafe-square distance by subtracting `resolution_m * sqrt(0.5)` and clamp at zero. Mark a center safe only when that lower bound is at least `required_radius_m`. Remove the `resolution² >= area` predicate completely.

- [ ] **Step 5: Map stable failures and expose stage elapsed time**

Return `HOPPER_START_REGION_AREA_INSUFFICIENT` when the start cell is base-safe but not center-safe, retain existing terrain/clearance reasons when base safety itself fails, and use `HOPPER_GLOBAL_GOAL_INFEASIBLE` when no center-safe goal candidate exists.

- [ ] **Step 6: Run the focused tests and commit**

Run the Task 1 focused build/test command, then:

```bash
git add ros2_ws/src/lunar_planner_core
git commit -m "fix: certify hopper landing support area"
```

### Task 2: Discover hopper nodes from the reachable frontier and fix local target semantics

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/landing_region.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/landing_region.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_route_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_fault_matrix_test.cpp`

- [ ] **Step 1: Add failing candidate-order and narrow-tolerance tests**

Cover a map where the first 128 row-major safe cells are irrelevant but a reachable chain exists, a graph that genuinely exhausts 128 nodes, and a local target with 0.05 m position tolerance whose certified region extends beyond that circle. Assert the aim point is within 0.05 m and the full polygon remains terrain-safe.

- [ ] **Step 2: Confirm the row-major and goal-clipping failures**

Run the hopper route/planner/fault targets. Expected: the first case reports no path/resource ambiguity and the narrow-tolerance case rejects a physically valid landing region.

- [ ] **Step 3: Implement lazy reachable-frontier discovery**

Seed the start and all stable goal-mask cells explicitly. Expand reachable landing centers in stable `(estimated_total_cost, row_major_id)` order, generate only centers inside the physical hop envelope, and keep at most eight stable lowest-cost outgoing edges per expanded node. Stop at 128 nodes. If the cap is reached while the frontier remains non-empty, return `HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT`; return no-path only after exhausting the bounded frontier with proof.

- [ ] **Step 4: Separate local region certification from aim selection**

Build `safe_mask` for the entire local map. Restrict only the set of aim seeds to the Action goal tolerance. Grow/certify the region against the full safe mask, allow its polygon to cross the tolerance circle, and finally verify the chosen aim point is inside both the region and the goal tolerance.

- [ ] **Step 5: Verify deterministic topology and all hopper faults**

Run the three focused test targets twice, compare route node IDs and warning order, then commit:

```bash
git add ros2_ws/src/lunar_planner_core/src/hierarchical ros2_ws/src/lunar_planner_core/src/hopper ros2_ws/src/lunar_planner_core/test
git commit -m "fix: bound hopper frontier and target landing semantics"
```

### Task 3: Anchor wheeled planning to a certified continuous start state

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_types.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_sweep_validator.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_sweep_validator.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/wheel_fault_matrix_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp`

- [ ] **Step 1: Add true-start and unsafe-connector failing tests**

Use starts offset from the grid center in both x/y and yaw. Assert the first trajectory sample equals the input pose within `1e-9`, the first preview pose equals the map-transformed true pose, and an obstacle intersecting only the true-to-lattice connector returns `WHEEL_START_CONNECTOR_INFEASIBLE` with no reference.

- [ ] **Step 2: Verify the current snap-to-cell behavior fails**

Run `wheel_planner`, `wheel_fault_matrix`, and `hierarchical_regression`; preserve the failing numerical output as test evidence outside the repository.

- [ ] **Step 3: Add one virtual start node**

Represent state zero as a virtual state carrying the true x, y, yaw and motion mode. Generate its successors by applying the normal wheel motion primitives in continuous coordinates and resolving only each successor into a lattice state. Do not add a raw line segment to an already-generated path.

- [ ] **Step 4: Certify every virtual-start transition**

Pass explicit source/target continuous poses to `WheelSweepValidator`; retain its footprint, curvature, boundary, terrain, and subdivision checks. Map a frontier with no valid first transition to `WHEEL_START_CONNECTOR_INFEASIBLE`.

- [ ] **Step 5: Preserve exact start through reconstruction and preview composition**

Reconstruct the discrete reference with the true virtual source as sample zero, avoid duplicate near-equal samples, and ensure global preview simplification cannot replace its first map-frame pose with a grid center without a safe supercover.

- [ ] **Step 6: Run focused tests and commit**

```bash
git add ros2_ws/src/lunar_planner_core/src/wheel ros2_ws/src/lunar_planner_core/src/hierarchical ros2_ws/src/lunar_planner_core/test
git commit -m "fix: certify wheeled true-start connector"
```

### Task 4: Anchor legged planning to the true body pose and height

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_types.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_terrain.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_terrain.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/legged_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/legged_fault_matrix_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp`

- [ ] **Step 1: Add true body-state and blocked-first-sweep tests**

Assert exact x/y/yaw/body-height anchoring, successful connection from an off-center state, failure on a body-sweep-only obstacle, and failure when the true height cannot enter any reachable height interval. Require `LEGGED_START_CONNECTOR_INFEASIBLE` and no trajectory for all first-connector failures.

- [ ] **Step 2: Confirm current center/height snapping fails the tests**

Run the three focused test targets and record only test logs under `/home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/`.

- [ ] **Step 3: Implement a continuous virtual legged start**

Carry true position, yaw, and body height in a virtual node. Apply the normal body primitives from that state, intersect successor terrain reachability with configured body-height limits, then enter the existing discrete lattice only after a successful transition.

- [ ] **Step 4: Reuse full body-sweep certification and reconstruct exactly**

Extend `ValidateLeggedBodySweep` to accept explicit continuous endpoints without weakening any footprint, clearance, vertical reach, yaw, or subdivision rule. Reconstruct sample zero from the true body state and map exhausted first successors to the stable connector failure.

- [ ] **Step 5: Run focused tests and commit**

```bash
git add ros2_ws/src/lunar_planner_core/src/legged ros2_ws/src/lunar_planner_core/test
git commit -m "fix: certify legged true-start connector"
```

### Task 5: Add bounded smoothing policy and explicit trajectory diagnostics

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_spline_optimizer.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_spline_optimizer.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_timing.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_spline_optimizer.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_spline_optimizer.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_timing.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/public_api_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/legged_planner_test.cpp`

- [ ] **Step 1: Add failing configuration and diagnostics contract tests**

Assert exact defaults, enum-to-string stability, stationary/optimized/discrete/certified-hop modes, finite error/curvature/timing values, and that public structs remain copy/move safe. Add a constructed optimizer failure that returns a discrete reference only when `require_smoothed_execution == false`.

- [ ] **Step 2: Add bounded curve acceptance tests**

For wheel, cover a curved corridor, reverse/forward switch with zero transition speed, curvature violation, and collision discovered only between control points. For legged, cover lateral translation with independent yaw, body-height interval violation, and between-sample body collision. Assert no case exceeds 64 retained controls, 512 samples, 128 iterations, 8 trust reductions, or 32 validation subdivisions.

- [ ] **Step 3: Implement deterministic control-point reduction and piecewise clamped cubic sampling**

Keep true start, requested terminal, direction switches, stop points, and corridor turns. Deterministically remove the lowest-deviation remaining points until 64 remain. Sample by curvature and obstacle proximity without exceeding 512 samples. Segment wheel curves at motion-mode switches; smooth legged x/y, z, and yaw independently.

- [ ] **Step 4: Revalidate every candidate curve**

Run finite/frame/time checks, corridor containment, the existing wheel/legged continuous sweep validators, terrain and boundary checks, curvature/velocity/acceleration checks, exact start anchoring, and terminal/yaw tolerance checks. Do not mark a curve optimized before all checks pass.

- [ ] **Step 5: Implement explicit fallback and required-smoothing failures**

On a safe optimizer/revalidation failure, return the certified discrete primitive sequence plus `WHEEL_OPTIMIZATION_DISCRETE_FALLBACK` or `LEGGED_OPTIMIZATION_DISCRETE_FALLBACK` and mode `kDiscreteFallback`. When required, return `WHEEL_SMOOTHED_EXECUTION_REQUIRED` or `LEGGED_SMOOTHED_EXECUTION_REQUIRED` with no reference. Stationary references and hops use their dedicated modes.

- [ ] **Step 6: Compute evidence metrics from the emitted reference**

Measure start/endpoint Euclidean errors, maximum wheel curvature (zero/not-applicable for stationary and hopper), collision-validation state, smoothing elapsed time, and hopper landing-field elapsed time. Reject non-finite diagnostic values instead of publishing misleading evidence.

- [ ] **Step 7: Run core planner tests and commit**

```bash
set +u
source /opt/ros/humble/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/log test \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/install \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/test-results
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/test-results --verbose
git add ros2_ws/src/lunar_planner_core
git commit -m "feat: add bounded certified local smoothing"
```

### Task 6: Publish the planner evidence through standard ROS diagnostics

**Files:**
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/message_conversion_test.cpp`

- [ ] **Step 1: Add failing diagnostic key/value tests**

Require exactly named values for `trajectory_mode`, `start_anchor_error_m`, `endpoint_error_m`, `maximum_curvature_per_m`, `collision_validation`, `smoothing_elapsed_s`, `landing_field_elapsed_s`, and comma-separated `warning_codes`. Verify Action warning codes remain unchanged and no message definition changes.

- [ ] **Step 2: Implement stable formatting**

Map enums to uppercase stable strings, format finite doubles with locale-independent round-trip-safe conversion, preserve warning order, and include the fields on success and safe fallback. Keep existing hierarchical keys and reason code.

- [ ] **Step 3: Build and run ROS tests**

Run a clean selected-package build/test for `lunar_planner_core lunar_planner_ros`, then:

```bash
git add ros2_ws/src/lunar_planner_ros
git commit -m "feat: publish local trajectory diagnostics"
```

### Task 7: Protect deterministic, memory, and performance contracts

**Files:**
- Modify: `tests/performance/hierarchical_planner_benchmark.cpp`
- Modify: `tests/performance/test_hierarchical_planner_benchmark.py`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp`
- Modify: `docs/validation/hierarchical-global-planning.md`

- [ ] **Step 1: Extend the benchmark contract before changing the runner**

Require stage timings for landing field and smoothing, peak work memory, trajectory mode counts, deterministic fingerprints, and the unchanged three cell-count tiers. Keep 30 measured runs after warm-up and existing Ubuntu/AGX thresholds.

- [ ] **Step 2: Add 50 m deterministic positive and negative fixtures**

Cover all three far-goal positives plus hopper support-area insufficiency, hopper resource exhaustion, wheel connector collision, legged connector collision, and forced optimizer fallback/required failure. Do not depend on a live ROS graph or random seed outside the fixture.

- [ ] **Step 3: Update the benchmark runner minimally**

Reuse the request's landing support field and safe projection, report phase timings without changing conclusions, and make any resource overflow a stable failure rather than an allocation retry.

- [ ] **Step 4: Run the benchmark contract and repository boundaries**

```bash
set +u
source /opt/ros/humble/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/log build \
  --base-paths ros2_ws/src --packages-select lunar_planner_core lunar_planner_ros \
  --build-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/install \
  --cmake-args -DBUILD_TESTING=ON
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/log test \
  --base-paths ros2_ws/src --packages-select lunar_planner_core lunar_planner_ros \
  --build-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/install \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/test-results
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/test-results --verbose
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git diff --check
```

Then run the benchmark executable to a JSON file under `/home/kai/CodexDownloads/lunar_navigation/planner_correctness/main/evidence/` and assert the 62,500-cell Ubuntu full-Action p95 is at most 2.0 s. Label AGX gates as pending device measurement unless run on AGX.

- [ ] **Step 5: Record evidence and commit**

Document commands, host identity, p50/p95/max, peak memory, stage timing, and explicit Ubuntu-only ownership. Do not commit raw logs or generated JSON.

```bash
git add tests/performance ros2_ws/src/lunar_planner_core/test docs/validation/hierarchical-global-planning.md
git commit -m "test: qualify planner correctness and bounded smoothing"
```

## Completion Gate

- [ ] Run all core and ROS package tests from a ROS 2 Humble-only install prefix.
- [ ] Run repository-boundary and external-interface checks.
- [ ] Confirm `git diff --check` and UTF-8 reads for modified Chinese documentation.
- [ ] Confirm no changes under any Action or message definition directory.
- [ ] Confirm no Nav2/Python fallback entered the production core.
- [ ] Confirm every successful wheel/legged reference starts at the true state and every hopper success has a locally certified first hop.
- [ ] Confirm optimized versus discrete fallback is explicit, and benchmark gates remain unchanged.
