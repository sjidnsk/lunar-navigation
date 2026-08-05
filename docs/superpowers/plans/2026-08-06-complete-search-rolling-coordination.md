# Complete Search and Rolling Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed-budget global termination with complete finite-map search, certify complete hopper routes with nominal parabolas and flight tubes, and continuously roll all three platforms through ROS/RViz without changing formal platform capability values.

**Architecture:** Keep `Planner::Plan(const PlannerInput&)` pure and pass an opaque immutable continuation between calls. Ground planners use complete grid A*, hopper uses a landing-field spatial index plus complete lazy A* and route-edge certification, ROS owns continuation storage/execution feedback/visualization output, and the external RViz harness owns the rolling Action-client and simulated executor.

**Tech Stack:** C++20, CMake 3.22, ROS 2 Humble/ament/colcon, GTest, Python 3.10/pytest, RViz2 `MarkerArray`, YAML/JSON interface contracts.

## Global Constraints

- Work only in `/mnt/data/WS/lunar-navigation/.worktrees/hierarchical-global-planning` and `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling`; preserve `/mnt/data/WS/lunar-navigation/.vscode` and all unrelated user changes.
- Source `/opt/ros/humble/setup.bash` and verify `ROS_DISTRO=humble` before every ROS build or test.
- Store build, test, benchmark, and RViz artifacts under `/home/kai/CodexDownloads/lunar_navigation`; never under either Git repository.
- Use `-DCMAKE_BUILD_TYPE=Release` for performance evidence; non-Release runs report `PERFORMANCE_BUILD_NOT_RELEASE` and cannot qualify a threshold.
- Keep the global map contract at at most 1,048,576 cells and 4,096 cells per axis; the finite valid map is the global search boundary.
- Never stop global feasibility search because expanded, reopened, generated, Open, landing-node, or out-degree counters reach configured values.
- Keep `maximum_authorized_hops=1`, `stop_token`, input validation, map-size limits, and numerical convergence bounds; none may be reported as physical no-route.
- Return `GLOBAL_NO_KNOWN_SAFE_ROUTE` only after a complete Open-set exhaustion. Cancellation, allocation failure, stale inputs, and numerical indeterminacy remain search-incomplete outcomes.
- Do not edit or promote current wheel, legged, or hopper proxy capability values. New capability fixtures must be visibly marked `test-only/non-authoritative`.
- Keep `PlanMotion.action` and `MotionReference.msg` unchanged. Add execution feedback only to provisional `lunar_navigation_msgs` and update the external ownership baseline atomically.
- Every production behavior change follows strict RED → verify RED → GREEN → verify GREEN → refactor; each task ends in an independently testable commit.

---

### Task 1: Make Release status observable and enforceable

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `tests/performance/hierarchical_planner_benchmark.cpp`
- Modify: `tests/performance/test_hierarchical_planner_benchmark.py`
- Create: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/build_record.py`
- Create: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/test/test_build_record.py`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/scripts/build_external.sh`

**Interfaces:**
- Produces: benchmark JSON field `build_type: "Release"` and `write_build_record(path, run_id, install_path, main_repository, build_type)`.
- Produces: external `build.json` with `run_id`, `install_path`, `main_repository`, and `build_type`.
- Consumes: CMake definition `LUNAR_BUILD_TYPE` set from `CMAKE_BUILD_TYPE`.
- Consumes: `build_external.sh --main-repository PATH --artifact-root PATH`; the selected main worktree and all build products are explicit.

- [ ] **Step 1: Add failing benchmark and build-record tests**

```python
assert report["build_type"] == "Release"

record = write_build_record(
    output, "20260806T010203000004Z", install, main_repository, "Release")
assert record == {
    "build_type": "Release",
    "install_path": str(install),
    "main_repository": str(main_repository),
    "run_id": "20260806T010203000004Z",
}
assert json.loads(output.read_text(encoding="utf-8")) == record
```

- [ ] **Step 2: Verify RED**

Run the existing Release benchmark contract and external unit test. Expected: the benchmark lacks `build_type`; importing `build_record` fails.

- [ ] **Step 3: Implement Release evidence**

Add this target definition and serialize it in benchmark output:

```cmake
target_compile_definitions(lunar_hierarchical_planner_benchmark PRIVATE
  LUNAR_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
```

Implement the external writer with an exact Release guard:

```python
def write_build_record(path: Path, run_id: str, install_path: Path,
                       main_repository: Path,
                       build_type: str) -> dict[str, str]:
    if build_type != "Release":
        raise ValueError("PERFORMANCE_BUILD_NOT_RELEASE")
    record = {
        "build_type": build_type,
        "install_path": str(install_path),
        "main_repository": str(main_repository),
        "run_id": run_id,
    }
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    return record
```

Pass `-DCMAKE_BUILD_TYPE=Release` from `build_external.sh`, require the selected
main repository to be a Git worktree containing `ros2_ws/src`, write all
`build/install/log/artifacts` trees below the explicit artifact root, invoke the
writer, and stop using the inline Python record writer.

- [ ] **Step 4: Verify GREEN and real build behavior**

Run both tests, execute `build_external.sh --main-repository
/mnt/data/WS/lunar-navigation/.worktrees/hierarchical-global-planning
--artifact-root /home/kai/CodexDownloads/lunar_navigation/complete-search-rolling-builds`
with a fresh run ID, and verify the generated CMake cache and `build.json` both
say Release and name the feature worktree.

- [ ] **Step 5: Commit each repository**

```bash
git commit -m "test: require release planning benchmarks"
git commit -m "build: force release validation builds"
```

### Task 2: Replace bounded ground global search with complete finite-map A*

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/grid_search.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/grid_search.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/global_grid_search_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_types_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp`
- Modify: `tests/performance/hierarchical_planner_benchmark.cpp`

**Interfaces:**
- Produces: `GlobalSearchConfig` containing preview/weight settings only, without `SearchResourceLimits`.
- Produces: `GlobalSearchStatus::{kSolved,kNoPath,kCanceled,kAllocationFailed,kInvalidProblem}`.
- Preserves: `SearchResourceLimits` under local `AraStarConfig` for local high-dimensional backends.

- [ ] **Step 1: Replace old resource-limit tests with failing completeness tests**

```cpp
TEST(GlobalGridSearch, CrossesMoreCellsThanTheFormerExpansionLimit) {
  auto problem = LongSerpentineProblem(/* safe_cells = */ 140'000U);
  const auto result = SearchGlobalGrid(problem);
  ASSERT_EQ(result.status, GlobalSearchStatus::kSolved);
  EXPECT_GT(result.expanded_states, 100'000U);
}

TEST(GlobalGridSearch, ReportsNoPathOnlyAfterFiniteOpenSetExhaustion) {
  const auto result = SearchGlobalGrid(CompletelySeparatedProblem());
  EXPECT_EQ(result.status, GlobalSearchStatus::kNoPath);
  EXPECT_EQ(result.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_EQ(result.expanded_states, ReachableSafeCellCount());
}
```

Delete assertions that configure or expect `GLOBAL_SEARCH_RESOURCE_LIMIT`.

- [ ] **Step 2: Verify RED**

Build and run `lunar_planner_core_global_grid_search_test`. Expected: compilation fails because the desired config/status does not exist or the old cap stops the serpentine case.

- [ ] **Step 3: Implement complete grid A***

Remove counter/memory checks from neighbor relaxation. Preallocate per-cell `g`, `parent`, and closed/reopen state using `map.cell_count()`, retain cancellation checks, and keep deterministic priority ordering:

```cpp
using QueueKey = std::tuple<double, double, double, std::size_t>;
// f, h, g, row_major_id
```

Catch `std::bad_alloc` at the search boundary and return
`kAllocationFailed/GLOBAL_SEARCH_ALLOCATION_FAILED`. Keep memory estimates as diagnostics only.

- [ ] **Step 4: Update callers and fixtures**

Map allocation failure to `RESOURCE_EXHAUSTED`; map only `kNoPath` to
`GLOBAL_NO_KNOWN_SAFE_ROUTE`. Remove assignments to `global_search.resources` from all fixtures and benchmark setup.

- [ ] **Step 5: Verify GREEN**

Run global search, global route, hierarchical regression, and benchmark contract tests in Release.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: complete finite-map ground search"
```

### Task 3: Solve hopper ballistic feasibility over the complete time interval

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/src/hopper/ballistic_envelope.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hopper/ballistic_envelope.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/ballistic_envelope_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hop_certifier.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp`

**Interfaces:**
- Produces: `BallisticEnvelopeResult SolveBallisticEnvelope(Vec3 launch, Vec3 landing, Vec3 initial_velocity, const HopperCapability&, double minimum_attitude_time_s, std::stop_token)`.
- Result contains `status`, optional best `BallisticArc`, `examined_intervals`, and stable `reason_code`.
- Removes feasibility dependence on `maximum_nominal_aim_points_per_region` and `maximum_certification_attempts`.

- [ ] **Step 1: Add failing literal physics tests**

```cpp
TEST(BallisticEnvelope, FindsNarrowFeasibleWindowBetweenOldSamples) {
  const auto result = SolveBallisticEnvelope(
      {0, 0, 0.5}, {9.55, 0, 0.5}, {0, 0, 0}, TestOnlyHopperCapability(),
      0.0, {});
  ASSERT_EQ(result.status, BallisticEnvelopeStatus::kSolved);
  EXPECT_NEAR(result.arc->landing_position_m.x, 9.55, 1e-9);
  EXPECT_LE(Norm(result.arc->launch_velocity_mps),
            TestOnlyHopperCapability().maximum_launch_speed_mps + 1e-9);
}

TEST(BallisticEnvelope, DistinguishesInfeasibleCanceledAndIndeterminate) {
  EXPECT_EQ(PhysicallyImpossible().status, BallisticEnvelopeStatus::kInfeasible);
  EXPECT_EQ(PreCanceled().status, BallisticEnvelopeStatus::kCanceled);
  EXPECT_EQ(NonFiniteInput().status, BallisticEnvelopeStatus::kInvalid);
}
```

- [ ] **Step 2: Verify RED**

Run the new GTest target. Expected: missing header/symbol.

- [ ] **Step 3: Implement interval feasibility**

Use `u=T²` to intersect launch-speed, impulse, landing-speed, minimum downward-speed, flight-time, and attitude-time intervals. The squared speed constraints use:

```text
|v_launch|² = |d|²/u - d·g + 0.25|g|²u
|v_land|²   = |d|²/u + d·g + 0.25|g|²u
```

Evaluate analytic interval boundaries plus the stationary point of physical score; use bounded bisection only to classify round-off neighborhoods. If the convergence bound cannot classify a boundary, return
`HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE`, not infeasible.

- [ ] **Step 4: Replace sampled certifier logic**

Call `SolveBallisticEnvelope` once per source/target pair. Remove candidate-count early exits and keep only `maximum_flight_tube_sections` and `maximum_authorized_hops` in `HopperPlannerConfig`.

- [ ] **Step 5: Verify GREEN**

Run ballistic, hopper planner, hopper fault-matrix, and public API tests.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: solve complete hopper ballistic envelope"
```

### Task 4: Build a complete landing spatial index and lazy hopper graph

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/landing_spatial_index.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/landing_spatial_index.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/landing_spatial_index_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/landing_support_field.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/landing_support_field.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.hpp`
- Replace: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_route_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp`

**Interfaces:**
- Produces: `LandingSpatialIndex(const LandingSupportField&, double maximum_reach_m)` and `Query(Vec2 source) -> std::vector<LandingNodeId>` in row-major order.
- Produces: `NominalHopEdge {source_id,target_id,cost,BallisticArc}` and `HopperRoutePlanResult.nominal_hops`.
- Removes: graph truncation fields/metrics and `HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT` production output.

- [ ] **Step 1: Add failing spatial-index equivalence tests**

```cpp
TEST(LandingSpatialIndex, MatchesBruteForceReachableCentersExactly) {
  const auto field = LiteralLandingField();
  LandingSpatialIndex index(field, 4.0);
  EXPECT_EQ(index.Query({2.0, 3.0}),
            BruteForceIdsWithinRadius(field, {2.0, 3.0}, 4.0));
}
```

Add hopper tests showing a path that needs more than 128 nodes and an edge to an already discovered non-goal node succeeds, while a complete broken chain returns only
`GLOBAL_NO_KNOWN_SAFE_ROUTE`.

- [ ] **Step 2: Verify RED**

Run new index and hopper route tests. Expected: missing index and old graph resource result.

- [ ] **Step 3: Expose all safe centers and build the index once**

Store row-major safe center IDs in `LandingSupportField`. Bucket by a deterministic integer bucket coordinate; query all intersecting buckets, exact-filter by radius, sort unique IDs row-major, and check `stop_token` during field/index construction.

- [ ] **Step 4: Implement direct-first complete lazy A***

Represent every safe center by its row-major ID and the real start by one special ID. Before multi-hop search, query goal cells within reach and try them in stable cost order. During A*, stream every neighbor returned by the index, including previously discovered non-goal nodes. Do not truncate candidates, nodes, or out-degree.

Use an admissible heuristic; when no strictly positive lower-bound cost exists, use zero. Retain only certified/invalid edge cache entries, not a dense adjacency matrix.

- [ ] **Step 5: Verify GREEN and deterministic repetition**

Run landing index, hopper route, hierarchical regression, cancellation, and repeated-input determinism tests.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: add complete lazy hopper route search"
```

### Task 5: Certify every hopper route edge and publish immutable continuation data

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/route_continuation.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/route_continuation.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/flight_tube_certifier.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_route_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp`

**Interfaces:**
- Adds forward declaration `class RouteContinuation;` and `std::shared_ptr<const RouteContinuation> continuation` to planner input/output.
- Adds runtime `position_uncertainty_m` and `velocity_uncertainty_mps` to `PlannerInput`; unit fixtures set them explicitly and ROS populates them in Task 8.
- Adds `CertifiedHopPreview {segment_id, launch_pose_map, landing_pose_map, launch_velocity_mps, flight_time, flight_tube_radius_m, landing_region_map, promotion_region_map}` to `PlannerOutput`.
- `HopperRoutePlanResult` returns a complete certified hop vector and metrics for coarse rejects, full certifications, invalidations, and cache hits.

- [ ] **Step 1: Add failing route-certification tests**

```cpp
TEST(HopperRoutePlanner, InvalidatesBlockedCandidateEdgeAndFindsAlternative) {
  const auto result = PlanHopperGlobalRoute(MapWithBlockedShortestArc());
  ASSERT_TRUE(result.ok());
  EXPECT_GE(result.full_edges_invalidated, 1U);
  EXPECT_EQ(result.certified_hops.size(), result.route_hops);
}

TEST(HopperRoutePlanner, CertifiesMidArcRockAgainstFullPlatformTube) {
  const auto result = PlanHopperGlobalRoute(MapWithMidArcRock());
  EXPECT_EQ(result.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(result.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
}
```

Add numerical-indeterminate and promotion-region contraction tests using literal geometry.

- [ ] **Step 2: Verify RED**

Run hopper route/planner tests. Expected: current global search only performs low-cost tube checks and returns one locally certified hop.

- [ ] **Step 3: Implement route-wide lazy certification**

After each optimistic candidate chain, certify every edge with `SolveBallisticEnvelope` and `CertifyFlightTube` against the immutable conservative global map. On a proven collision/capability violation, cache `(source_id,target_id)` as invalid and rerun shortest path while reusing field/index/neighbor and certificate caches. On numerical indeterminacy, abort with the stable numerical reason.

- [ ] **Step 4: Add conservative promotion regions**

Contract the target landing polygon by
`position_uncertainty_m + global_map.resolution_m` and reject an empty result.
Expand the next hop tube by the maximum displacement inside that promotion
region plus `velocity_uncertainty_mps * next_flight_time`; existing capability
clearance and map elevation variance remain part of `CertifyFlightTube`.
Store the exact uncertainty values used, the landing polygon, and the promotion
polygon in the preview/continuation. These runtime uncertainty inputs do not
modify any formal platform capability value.

- [ ] **Step 5: Make HopperPlanner authorize from certified route data**

Transform only the first certified map-frame hop into the existing odom-frame `HopSegment`; retain all certified previews and the opaque continuation. Keep `hops.size()==1` regardless of route length.

- [ ] **Step 6: Verify GREEN**

Run hopper route, planner, flight-tube, hierarchical planner/regression, and public-header tests.

- [ ] **Step 7: Commit**

```bash
git commit -m "feat: certify complete hopper routes"
```

### Task 6: Reuse ground routes and promote hopper continuations safely

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/route_continuation.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/route_continuation.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/route_continuation_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_commitment_test.cpp`

**Interfaces:**
- Adds planner identity fields: `mission_id`, `mission_revision`, `platform_id`, `capability_version`, `global_map_generation`, `local_map_generation`, and `map_from_odom_generation`.
- Produces: `TryReuseGroundRoute(input, continuation)` and `TryPromoteHopperHop(input, continuation)`.
- Preserves: pure core; ROS stores but never mutates continuations.

- [ ] **Step 1: Add failing reuse and promotion tests**

```cpp
TEST(RouteContinuation, ReusesGroundRouteForSmallRealPoseDeviation) {
  const auto first = planner.Plan(InitialGroundInput());
  auto next = AdvancedGroundInput(/* deviation_m = */ 0.1);
  next.continuation = first.continuation;
  const auto second = planner.Plan(next);
  ASSERT_TRUE(second.reference.has_value());
  EXPECT_TRUE(second.diagnostics.hierarchical->route_reused);
  EXPECT_EQ(second.reference->preview.poses_map.front().position_m,
            ActualMapPose(next));
}

TEST(RouteContinuation, PromotesOnlyStableLandingInsidePromotionRegion) {
  EXPECT_TRUE(Promote(LandedHoldInside()).ok());
  EXPECT_EQ(Promote(LandedHoldOutside()).reason_code,
            "HOP_LANDING_DEVIATION_REPLAN_REQUIRED");
}
```

Add mismatched mission, goal, platform, capability, map identity, and large TF delta cases.

- [ ] **Step 2: Verify RED**

Run route continuation tests. Expected: missing continuation API and every call recomputes the route.

- [ ] **Step 3: Implement exact identity checks**

Store immutable route identity and compare every field before reuse. Ground reuse projects the real map-frame pose onto the unexecuted certified corridor and advances the cursor; failure falls back to full global planning. Hopper promotion requires `kLandedHold`, matching active IDs, fresh stable state inside `promotion_region`, and unchanged global/capability identity.

- [ ] **Step 4: Integrate with Planner**

On valid ground continuation, skip only global A* and rerun local frontier/physics from the actual state. On valid hopper continuation, expose exactly the next pre-certified hop. On any mismatch, discard the candidate and perform a full plan from the real state to the same final goal.

- [ ] **Step 5: Verify GREEN**

Run continuation, hierarchical planner, hopper commitment, start anchoring, and deterministic tests.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: reuse certified rolling routes"
```

### Task 7: Add the provisional external execution-feedback contract

**Files:**
- Create: `ros2_ws/src/lunar_navigation_msgs/msg/MotionExecutionFeedback.msg`
- Modify: `ros2_ws/src/lunar_navigation_msgs/CMakeLists.txt`
- Modify: `docs/interfaces/external-input-baseline.md`
- Modify: `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`
- Modify: `tools/check_external_interfaces.py`
- Modify: `tools/check_repository_boundaries.py`
- Modify: `tests/foundation/test_navigation_message_package.py`
- Modify: `tests/foundation/test_external_interface_config.py`
- Modify: `tests/foundation/test_documentation_baseline.py`

**Interfaces:**
- Produces: `/execution/motion_feedback` of type `lunar_navigation_msgs/msg/MotionExecutionFeedback`, externally owned.
- Schema contains exact platform/state constants, `Header`, monotonic per-plan sequence, `plan_id`, `segment_id`, and `reason_code` from the approved design.

- [ ] **Step 1: Add failing schema and ownership tests**

```python
assert declarations == (
    "uint8 WHEELED=1", "uint8 LEGGED=2", "uint8 HOPPER=3",
    "uint8 IDLE=0", "uint8 ACCEPTED=1", "uint8 EXECUTING=2",
    "uint8 SEGMENT_COMPLETE=3", "uint8 LANDED_HOLD=4",
    "uint8 FAILED=5", "uint8 CANCELED=6", "std_msgs/Header header",
    "uint64 sequence", "uint8 platform_type", "string plan_id",
    "string segment_id", "uint8 state", "string reason_code",
)
assert config["topics"]["motion_execution_feedback"]["owner"] == "external"
```

- [ ] **Step 2: Verify RED**

Run the three focused foundation tests. Expected: message/topic/schema absent.

- [ ] **Step 3: Add schema and atomically update the baseline**

Register the message with `rosidl_generate_interfaces`, document validation/QoS/ownership, and teach both checkers to require the exact declaration and topic configuration.

- [ ] **Step 4: Verify GREEN**

Run external interface checker, repository boundary checker, and all foundation tests.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: define motion execution feedback input"
```

### Task 8: Store continuations, validate execution feedback, and publish certified markers in ROS

**Files:**
- Create: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/execution_feedback_tracker.hpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/execution_feedback_tracker.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/test/execution_feedback_tracker_test.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/route_marker_publisher.hpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/route_marker_publisher.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/test/route_marker_publisher_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/snapshot_store.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/snapshot_store.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/snapshot_builder.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/snapshot_builder.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_ros/package.xml`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/snapshot_builder_test.cpp`

**Interfaces:**
- Produces: `ExecutionFeedbackTracker::Accept(message, expected, now)` and `context()`.
- Produces: planner-owned transient-local `/planning/certified_route_markers`.
- Produces: exact map-content generations while adapting each accepted GridMap, plus conservative position/velocity uncertainty scalars.

- [ ] **Step 1: Add failing feedback tests**

```cpp
TEST(ExecutionFeedbackTracker, AcceptsStrictPerPlanSequenceAndMapsGroundIds) {
  tracker.SetExpected(GroundExpected("wheel/request-2"));
  EXPECT_TRUE(tracker.Accept(GroundExecuting(1), now).ok());
  const auto context = std::get<GroundExecutionContext>(*tracker.context());
  EXPECT_EQ(context.active_plan_id, "wheel/request-2");
  EXPECT_EQ(context.active_segment_id, "wheel/request-2");
}

TEST(ExecutionFeedbackTracker, RejectsStaleWrongPlatformAndWrongSegment) {
  EXPECT_EQ(tracker.Accept(WrongSegment(), now).reason_code,
            "EXECUTION_FEEDBACK_SEGMENT_MISMATCH");
}
```

- [ ] **Step 2: Add failing server/marker tests**

Verify the planner dependency receives a non-null matching `previous_execution`, continuation is passed on the second same-goal request, marker output contains future hopper arc/tube/promotion namespaces, and deactivate publishes DELETE only for owned IDs.

- [ ] **Step 3: Verify RED**

Run the new ROS GTests and focused server tests. Expected: missing types/subscription/publisher; existing server passes `nullopt`.

- [ ] **Step 4: Implement feedback and content identity**

Use reliable/volatile/depth-10 feedback QoS. Hash canonical adapted planning
layers in the same O(N) conversion pass, excluding Header stamp but retaining
independent freshness checks; equal content retains its generation and changed
content increments it. Increment `map_from_odom_generation` when the accepted
transform changes beyond the exact stored transform identity. Derive
`position_uncertainty_m` and `velocity_uncertainty_mps` as three times the
square root of the largest finite position or linear-velocity covariance
diagonal respectively; invalid covariance continues to fail snapshot
validation. Cache one immutable continuation for the configured platform and
clear it on target/revision/platform/capability/map generation/Lifecycle
changes.

- [ ] **Step 5: Integrate server and marker publisher**

Replace `.previous_execution = std::nullopt` with validated tracker context, pass identity/continuation into `GoalRequest`, store a returned continuation only after a valid successful result, set expected execution IDs from the issued reference, and publish map/odom markers from `CertifiedHopPreview`.

- [ ] **Step 6: Verify GREEN**

Run feedback tracker, snapshot builder, marker publisher, ReferenceGuard, message conversion, and PlanMotion server tests.

- [ ] **Step 7: Commit**

```bash
git commit -m "feat: coordinate rolling plans in ROS"
```

### Task 9: Implement the external rolling Action client and simulated executor

**Files:**
- Create: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/rolling_execution.py`
- Create: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/test/test_rolling_execution.py`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_bridge_node.py`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/rviz_evidence.py`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/constants.py`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/test/test_rviz_evidence.py`

**Interfaces:**
- Produces: `RollingExecutionSession` retaining final goal, request generation, current reference, actual pose, and platform state.
- Produces: dynamic Odometry plus exact `MotionExecutionFeedback` transitions.
- Consumes: repeated one-result `PlanMotion` calls and planner-owned certified marker Topic.
- Uses test-only transport `/lunar_isaac_validation/simulated_odometry` (`nav_msgs/msg/Odometry`) from the controller to the bridge; the bridge remains the sole publisher of canonical `/localization/odometry`.

- [ ] **Step 1: Add failing pure state-machine tests**

```python
def test_ground_segment_completion_requests_same_final_goal_from_actual_pose():
    session = GroundRollingFixture()
    session.accept_reference(first_result)
    command = session.advance_to_segment_end()
    assert command.kind == "PLAN_NEXT"
    assert command.final_goal == original_final_goal
    assert command.start_pose == session.actual_pose

def test_hopper_never_replans_in_flight_and_promotes_after_landed_hold():
    session = HopperRollingFixture()
    assert session.on_tick(IN_FLIGHT).kind == "NO_ACTION"
    assert session.on_tick(LANDED_HOLD).kind == "PLAN_NEXT"
```

Add stale Action result, wrong plan/segment feedback, cancellation, map-change, and goal-replacement cases.

- [ ] **Step 2: Verify RED**

Run the new pure tests. Expected: module absent.

- [ ] **Step 3: Implement simulated execution**

Ground execution interpolates the returned trajectory from the current actual pose and triggers next planning at segment complete or configured remaining-arc threshold. Hopper evaluates the returned nominal parabola, publishes committed/in-flight/landed-hold transitions, and never sends a planning request while committed/in-flight.

- [ ] **Step 4: Integrate dynamic bridge/controller**

Replace frozen platform Odometry/Marker state with the rolling session pose.
`interactive_node` publishes test-only simulated Odometry and canonical execution
feedback; `interactive_bridge_node` validates/subscribes that simulated state and
remains the sole canonical Odometry/TF publisher. Retain a single RViz goal as
the final goal, create unique per-segment request IDs, set ground
`replace_active_request=true`, and publish matching execution feedback sequence
numbers.

- [ ] **Step 5: Verify GREEN**

Run rolling, interactive controller/bridge, RViz evidence, and synthetic interactive integration tests without launching Isaac Sim.

- [ ] **Step 6: Commit external repository**

```bash
git commit -m "feat: roll rviz planning sessions"
```

### Task 10: Add complete diagnostics and all-platform Release performance regression

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`
- Modify: `tests/performance/hierarchical_planner_benchmark.cpp`
- Modify: `tests/performance/test_hierarchical_planner_benchmark.py`
- Create: `tests/fixtures/capabilities/test-only/wheeled.yaml`
- Create: `tests/fixtures/capabilities/test-only/legged.yaml`
- Create: `tests/fixtures/capabilities/test-only/hopper.yaml`
- Modify: `docs/validation/hierarchical-global-planning.md`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`

**Interfaces:**
- Adds hierarchical metrics for safe landing nodes, candidate/coarse/full edges, invalidations, certificate-cache hits, stage timings, route reuse, route cursor, rolling request count, and execution IDs/state.
- Produces 30-run p50/p95/max JSON for fixed 50 m × 50 m, 0.2 m fixtures.

- [ ] **Step 1: Add failing diagnostics and performance assertions**

```python
assert report["runs"] == 30
assert report["wheel_positive"]["p95_s"] <= 2.0
assert report["legged_positive"]["p95_s"] <= 2.0
assert report["hopper_direct_positive"]["p95_s"] <= 1.0
assert report["hopper_multihop_positive"]["p95_s"] <= 5.0
assert report["hopper_complete_negative"]["p95_s"] <= 5.0
assert report["build_type"] == "Release"
```

Verify diagnostics count work but cannot alter outcomes by rerunning identical inputs.

- [ ] **Step 2: Verify RED**

Run focused diagnostics and performance tests. Expected: fields/cases absent or current hopper cases exceed/fail resource limits.

- [ ] **Step 3: Populate metrics without control-flow coupling**

Measure landing field, index, ballistic solving, tube certification, global, and local stages independently. Publish standard diagnostic key/value pairs and mirror them in the RViz panel. No diagnostic count may appear in a termination branch.

- [ ] **Step 4: Add test-only capability fixtures and benchmark cases**

Mark every fixture document with `authority: test-only/non-authoritative`. Use fixed literal start/goal/obstacle maps; never scan for a successful target at runtime.

- [ ] **Step 5: Verify GREEN on a quiet Release run**

Run 30 iterations, save JSON outside the repository, and check p95 thresholds and deterministic route hashes.

- [ ] **Step 6: Commit main and external repositories**

```bash
git commit -m "test: qualify complete rolling planner performance"
git commit -m "feat: show rolling planner diagnostics"
```

### Task 11: Remove old production paths and run atomic end-to-end qualification

**Files:**
- Modify: `ros2_ws/src/lunar_planner_ros/config/planner_ros.schema.json`
- Modify: `docs/validation/hierarchical-global-planning.md`
- Modify: any checked-in planner YAML still containing retired global/hopper search caps
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/ros2_ws/src/lunar_isaac_validation/config/rviz/lunar_interactive_planning.rviz`
- Modify: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/.worktrees/complete-search-rolling/README.md`

**Interfaces:**
- Retires production outputs `GLOBAL_SEARCH_RESOURCE_LIMIT` and `HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT`.
- Preserves `maximum_authorized_hops=1`, Action schema, MotionReference schema, and external map ownership.

- [ ] **Step 1: Add failing compatibility checks**

Add behavioral configuration tests that reject retired YAML keys, run known old-cap hopper targets successfully, and assert new production results never emit either retired reason code.

- [ ] **Step 2: Verify RED**

Run focused config, regression, and external tests. Expected: old keys remain accepted or old reason codes remain reachable.

- [ ] **Step 3: Remove old branches/configuration and update operator docs**

Remove retired fields from schemas/YAML, remove old reason-code branches, document the Release build/start commands and RViz rolling controls, and configure displays for platform, global route, local segment, certified arcs/tubes/promotion regions, obstacles, and status.

- [ ] **Step 4: Run complete main-repository qualification**

Using a fresh repository-external Release build/install/log/test-result directory:

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
qualification_root="$(mktemp -d /home/kai/CodexDownloads/lunar_navigation/complete-rolling-final-XXXXXX)"
colcon --log-base "$qualification_root/log-build" build \
  --base-paths "$PWD/ros2_ws/src" \
  --build-base "$qualification_root/build" \
  --install-base "$qualification_root/install" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
colcon --log-base "$qualification_root/log-test" test \
  --build-base "$qualification_root/build" \
  --install-base "$qualification_root/install" \
  --test-result-base "$qualification_root/test-results"
colcon test-result --test-result-base "$qualification_root/test-results" --verbose
python3 tools/check_external_interfaces.py --config ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation
```

Expected: all commands exit 0, with zero failed tests.

- [ ] **Step 5: Run complete external qualification**

Build from the main worktree plus external validation worktree in Release, run the full external pytest suite, run fixed three-platform positive/negative Action regressions, and launch the synthetic RViz session long enough to capture one completed rolling goal per platform. Save logs, benchmark JSON, and screenshots under a fresh `/home/kai/CodexDownloads/lunar_navigation/complete-rolling-final-*` directory.

- [ ] **Step 6: Audit scope and commits**

Run `git diff --check`, UTF-8-read every changed Chinese text file, confirm no capability production values changed, confirm both worktrees are clean after commits, and record main/external commit hashes in the validation document.

- [ ] **Step 7: Commit final integration records**

```bash
git commit -m "docs: record complete rolling planner qualification"
git commit -m "docs: document rolling rviz operation"
```
