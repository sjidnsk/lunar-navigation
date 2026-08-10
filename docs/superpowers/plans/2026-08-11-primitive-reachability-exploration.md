# Primitive-Reachability Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ground connected-component reachability and frontier-first filtering with one platform-motion-primitive graph authority for frozen coverability and observed-only candidate generation, then launch a new formal PPO run from step 0.

**Architecture:** C++ owns platform state expansion, primitive certification, forward/returnable graph labels, canonical identity, and incremental edge reuse. A frozen truth instance materializes the cache-v5 denominator; a per-episode observed-only instance exposes exact recoverable observation states to Python, where the existing information-gain and `[64,12]` policy boundary remain. The final Planner still certifies the selected exact target, while WHEELED/LEGGED may target any recoverable state within 30 m and HOPPER may target only a certified direct successor.

**Tech Stack:** C++20, ROS 2 Humble, GoogleTest, pybind11, Python 3.10, NumPy, PyTorch PPO, pytest, canonical SHA-256 artifacts.

## Global Constraints

- Work only in `/mnt/data/WS/.lunar-navigation-worktrees/formal-training-environment-closure` on `feature/formal-training-environment-closure`; preserve unrelated worktree changes.
- Keep build, install, log, cache, report, checkpoint, and run artifacts under `/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10`; do not write generated artifacts into the repository.
- Source `/opt/ros/humble/setup.bash`, require `ROS_DISTRO=humble`, and use external `--merge-install` roots for all native builds.
- Use `/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for training tests.
- Every behavior task follows RED, observed intended failure, minimal GREEN, focused regression, and a scoped commit.
- Freeze cache schema as `lunar-formal-training-cache/v5` and training semantics as `lunar-training-semantics/sensor-30m-360-platform-primitive-coverable-detail95-observed-incremental-primitive-candidates-option-path-observation-auditable-failure/v10`.
- Freeze algorithm IDs as `cpp-wheel-motion-primitive-recoverable-graph/v1`, `cpp-legged-motion-primitive-recoverable-graph/v1`, and `cpp-hopper-certified-recoverable-state-graph/v4`.
- Do not change the PPO topology, seven observation names, `[64,12]` candidate tensor, `[64]` candidate mask, `1024 m` scene, `30 m / 360 deg` sensor, `0.2 m` reveal, source/split, capability v2, or unbounded episode lifetime.
- Truth graph data may affect cache qualification, coverage, reward, success, and diagnostics only. Candidate generation and oracle use observed-only graph data only.
- `mission_coverable_fraction >= 0.95` remains scene/platform feasibility and exact coverable coverage `>=0.95` remains episode success.
- Training remains stopped until Task 10 passes source, native, preflight-cache, closed-loop, full-cache, calibration, and formal-preflight gates. Launch is fresh: no resume and no warm start.

---

### Task 1: Add the canonical primitive graph contract and graph kernel

**Files:**

- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/primitive_reachability.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/shared/primitive_reachability_graph.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/shared/primitive_reachability_graph.cpp`
- Create: `ros2_ws/src/lunar_planner_core/src/shared/sha256.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/shared/sha256.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/primitive_reachability_graph_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_core/test/public_header_boundary_test.py`

**Interfaces:**

- Produces the public immutable result types below and a move-only `PrimitiveReachabilityEngine` whose private implementation owns reusable edge evidence.
- `recoverable == forward_reachable && returnable`; two-dimensional `reachable` is only the OR projection of recoverable observation states.
- State IDs are assigned after canonical tuple sorting; edge IDs sort by `(source_state_id, primitive_id, target_state_id)`.

```cpp
struct PrimitiveReachabilityState final {
  std::uint64_t state_id{};
  Vec3 position_m;
  double yaw_rad{};
  std::int32_t cell_x{};
  std::int32_t cell_y{};
  std::int32_t yaw_bin{};
  std::int32_t motion_mode{};
  Interval body_z_m;
  double path_cost{};
  std::uint8_t forward_reachable{};
  std::uint8_t returnable{};
  std::uint8_t observation_state{};
  std::uint8_t direct_successor{};
};

struct PrimitiveReachabilityEdge final {
  std::uint64_t source_state_id{};
  std::uint64_t target_state_id{};
  std::uint32_t primitive_index{};
  std::string primitive_id;
  double cost{};
};

struct PrimitiveReachabilitySnapshot final {
  PlatformType platform_type{};
  std::size_t width{};
  std::size_t height{};
  std::vector<PrimitiveReachabilityState> states;
  std::vector<PrimitiveReachabilityEdge> edges;
  std::vector<std::uint8_t> reachable;
  std::string algorithm_id;
  std::string state_schema;
  std::string primitive_set_sha256;
  std::string graph_sha256;
  std::uint64_t revision{};
  std::size_t invalidated_edge_count{};
  std::size_t revalidated_edge_count{};
};

struct PrimitiveReachabilityResult final {
  std::optional<PrimitiveReachabilitySnapshot> snapshot;
  std::string reason_code;
  [[nodiscard]] bool ok() const noexcept;
};

class PrimitiveReachabilityEngine final {
 public:
  PrimitiveReachabilityEngine();
  ~PrimitiveReachabilityEngine();
  PrimitiveReachabilityEngine(PrimitiveReachabilityEngine&&) noexcept;
  PrimitiveReachabilityEngine& operator=(PrimitiveReachabilityEngine&&) noexcept;
  PrimitiveReachabilityEngine(const PrimitiveReachabilityEngine&) = delete;
  PrimitiveReachabilityEngine& operator=(const PrimitiveReachabilityEngine&) = delete;
  [[nodiscard]] PrimitiveReachabilityResult Update(
      const PlannerInput& input,
      std::optional<double> maximum_action_distance_m);
  void Reset() noexcept;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
```

The private header defines `shared::PrimitiveGraphBuildResult` as canonical platform states, certified/rejected potential edges with swept tile dependencies, safe-anchor key, algorithm ID, state schema, and primitive-set canonical bytes. Tasks 2-4 return this exact type to the shared engine.

- [x] **Step 1: Write graph-label, projection, hash, and failure tests**

Add literal directed graphs proving: forward-only nodes are excluded; a node with a reverse path to anchor is included; same cell with different yaw/mode remains distinct; projected cells OR only recoverable observation states; SHA-256 matches the empty string and `abc` standard vectors; `-0.0` hashes like `+0.0`; non-finite graph fields fail.

- [x] **Step 2: Run the RED test**

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
test "$ROS_DISTRO" = humble
NATIVE_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/task1-native
colcon --log-base "$NATIVE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base "$NATIVE_ROOT/build" --install-base "$NATIVE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_core" \
  -R primitive_reachability_graph --output-on-failure
```

Expected: compile fails because the new public contract and graph kernel do not exist.

- [x] **Step 3: Implement canonical graph labels and identity**

Implement deterministic forward BFS, reverse BFS from the safe anchor, projection, canonical binary serialization, and the small internal SHA-256 implementation. Reject invalid indices, duplicate canonical keys, negative/non-finite costs, non-finite poses, and canceled updates.

- [x] **Step 4: Run GREEN and commit**

```bash
colcon --log-base "$NATIVE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base "$NATIVE_ROOT/build" --install-base "$NATIVE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_core" \
  -R 'primitive_reachability_graph|shared_core' --output-on-failure
git add ros2_ws/src/lunar_planner_core
git commit -m "feat(planner): add canonical primitive reachability graph"
```

---

### Task 2: Reuse WHEELED planner primitives for bulk recoverable reachability

**Files:**

- Create: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_primitive_expansion.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_primitive_expansion.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/shared/primitive_reachability_graph.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/primitive_reachability_graph_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`

**Interfaces:**

- Extracts the existing `PoseKey`, mode legality, primitive application, sweep validation, and edge cost without changing planner behavior.
- Produces `BuildWheelPrimitiveGraph(...)`; state key is exactly `(cell_x, cell_y, yaw_bin, WheelMotionMode)` and continuous poses remain attached to the least-cost deterministic representative.

```cpp
shared::PrimitiveGraphBuildResult BuildWheelPrimitiveGraph(
    const WheeledState& current_state,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);
```

- [x] **Step 1: Write WHEELED RED tests**

Create one 2-D connected fixture where available primitives cannot turn into a side corridor, one fixture where the same cell has reachable and unreachable yaw/mode states, and one directed primitive fixture whose endpoint cannot return. Assert the old connected-component mask would include each target while the new recoverable projection excludes it.

- [x] **Step 2: Run RED**

```bash
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_core" \
  -R 'primitive_reachability_graph|wheel_planner' --output-on-failure
```

- [x] **Step 3: Extract one primitive expansion authority and build the full finite graph**

Use a deterministic Dijkstra worklist over the existing dense state index. Record every certified directed edge, re-open a state when a lower-cost representative changes, and derive direct successors from the current exact start. Do not use a goal heuristic or the connected-component label as acceptance authority.

- [x] **Step 4: Run GREEN plus established wheel tests and commit**

```bash
colcon --log-base "$NATIVE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base "$NATIVE_ROOT/build" --install-base "$NATIVE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_core" \
  -R 'primitive_reachability_graph|wheel_planner|wheel_fault_matrix' \
  --output-on-failure
git add ros2_ws/src/lunar_planner_core
git commit -m "feat(planner): project wheel primitive reachability"
```

---

### Task 3: Reuse LEGGED height/support state propagation

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/shared/primitive_reachability_graph.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/primitive_reachability_graph_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/legged_planner_test.cpp`

**Interfaces:**

- Adds a goal-independent graph builder that reuses the exact existing `ApplyPrimitive`, terrain support, body-height interval intersection, stable edge index, and edge cost.

```cpp
LeggedLatticeBuildResult BuildLeggedPrimitiveGraph(
    const LeggedState& current_state,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);
```

- [x] **Step 1: Add RED fixtures for body-height and returnability**

Assert a 2-D connected target is excluded when the reachable body-height interval becomes empty, support rejects an edge, or only a directed outbound chain exists. Assert the same cell/yaw with different interval state is not silently merged.

- [x] **Step 2: Run RED, implement goal-independent expansion, and run GREEN**

```bash
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_core" \
  -R 'primitive_reachability_graph|legged_planner' --output-on-failure
colcon --log-base "$NATIVE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base "$NATIVE_ROOT/build" --install-base "$NATIVE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_core" \
  -R 'primitive_reachability_graph|legged_planner|legged_fault_matrix' \
  --output-on-failure
```

- [x] **Step 3: Commit**

```bash
git add ros2_ws/src/lunar_planner_core
git commit -m "feat(planner): project legged primitive reachability"
```

---

### Task 4: Put HOPPER certified edges behind the common graph contract

**Files:**

- Create: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_reachability_graph.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_reachability_graph.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/shared/reachability_projection.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/shared/primitive_reachability_graph.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/reachability_projection_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/primitive_reachability_graph_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`

**Interfaces:**

- Moves landing-evidence validation and directed landing-region edge certification into a reusable graph builder.
- Truth projection may traverse multiple certified hops. Runtime `direct_successor` is true only for a certified edge whose source is the current exact landing state.
- The old `ProjectReachability` wrapper may delegate for compatibility, but formal cache/runtime code must use `PrimitiveReachabilityEngine`.

- [ ] **Step 1: Add RED tests for multi-hop truth versus direct runtime candidates**

Use three landing patches A-B-C where A-B and B-C certify but A-C does not. Assert truth recoverable projection includes C, A's direct successors include B but not C, and any forward/reverse landing, delta-v, ballistic envelope, flight tube, forbidden, clearance, or support failure rejects the edge without turning numerical failure into ordinary infeasibility.

- [ ] **Step 2: Implement common HOPPER states/edges and run GREEN**

```bash
colcon --log-base "$NATIVE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base "$NATIVE_ROOT/build" --install-base "$NATIVE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_core" \
  -R 'primitive_reachability_graph|reachability_projection|hopper' \
  --output-on-failure
```

- [ ] **Step 3: Commit**

```bash
git add ros2_ws/src/lunar_planner_core
git commit -m "feat(planner): unify hopper primitive reachability"
```

---

### Task 5: Expose a stateful incremental engine through the training bridge

**Files:**

- Modify: `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/request.hpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/conversions.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/python/lunar_planner_training_bridge/__init__.py`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`

**Interfaces:**

```cpp
class TrainingPrimitiveReachabilityEngine final {
 public:
  PrimitiveReachabilityResult Update(
      const TrainingPlanRequest& request,
      std::optional<double> maximum_action_distance_m) noexcept;
  void Reset() noexcept;
 private:
  PrimitiveReachabilityEngine engine_;
};
```

- Python exposes immutable contiguous arrays for state IDs, positions, yaw, state fields, reachability flags, direct-successor flags, path costs, edge source/target IDs, and projected mask plus graph identity/diagnostics.
- The engine compares world geometry/generation and per-tile content fingerprints. A changed tile invalidates cached accepted and rejected primitive edges whose swept AABB intersects it; unaffected edge certificates remain reusable. Forward/reverse labels are recomputed every revision.

- [ ] **Step 1: Write bridge RED tests**

Assert array shapes/dtypes and read-only ownership, deterministic repeated identity, no-change update with zero invalidations, a new obstacle with positive invalidation count and removed state, newly observed safe evidence with positive revalidation count and added state, platform/capability/schema drift reset, and hard failure propagation.

- [ ] **Step 2: Run RED**

```bash
BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/task5-bridge
colcon --log-base "$BRIDGE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$BRIDGE_ROOT/build" --install-base "$BRIDGE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon --log-base "$BRIDGE_ROOT/log-test" test --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$BRIDGE_ROOT/build" --install-base "$BRIDGE_ROOT/install" \
  --merge-install --event-handlers console_direct+
colcon test-result --test-result-base "$BRIDGE_ROOT/build" --verbose
```

- [ ] **Step 3: Implement bindings, run GREEN, and commit**

```bash
colcon --log-base "$BRIDGE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$BRIDGE_ROOT/build" --install-base "$BRIDGE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon --log-base "$BRIDGE_ROOT/log-test" test --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$BRIDGE_ROOT/build" --install-base "$BRIDGE_ROOT/install" \
  --merge-install --event-handlers console_direct+
colcon test-result --test-result-base "$BRIDGE_ROOT/build" --verbose
git add ros2_ws/src/lunar_planner_training_bridge
git commit -m "feat(training): expose incremental primitive reachability"
```

---

### Task 6: Materialize truth primitive coverability in cache v5

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/coverability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_start_qualification.py`
- Modify: `training/lunar_policy_training/tests/test_coverability.py`
- Modify: `training/lunar_policy_training/tests/test_formal_cache.py`
- Create: `training/lunar_policy_training/tests/test_formal_start_qualification.py`

**Interfaces:**

- `PlatformCoverability` adds `qualified_start_state`, `primitive_state_count`, `certified_edge_count`, `recoverable_state_count`, `primitive_state_schema`, `primitive_set_sha256`, and `reachability_graph_sha256`.
- Canonical starts are exact cell-center `x/y`, terrain `z`, `yaw=0`; WHEELED uses START mode, LEGGED uses the capability body-height midpoint and interval, and HOPPER uses the certified landing aim pose.
- `_build_scene_platform_coverability` creates a fresh truth engine, computes the complete multi-step graph without a 30 m graph cutoff, and passes exact recoverable observation positions into tile-wise LOS union.
- Cache v5 stores graph identity/counts and the projected bit mask; it does not persist the complete edge list.
- Start qualification uses the initial reveal's observed-only graph and requires at least one legal graph-derived candidate.

- [ ] **Step 1: Write cache-v5 RED tests**

Assert old v3/v4 manifests fail, graph identity fields are mandatory and hash-validated, the same 2-D map produces different WHEELED/LEGGED/HOPPER denominators when primitive capabilities differ, a sensor-visible non-standable target remains coverable, truth states beyond 30 m contribute after multiple primitives, and changing only truth graph topology cannot affect an observed candidate fixture.

- [ ] **Step 2: Run RED**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/task5-bridge
set +u
source /opt/ros/humble/setup.bash
source "$BRIDGE_ROOT/install/setup.bash"
set -u
export PYTHONPATH="$BRIDGE_ROOT/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_coverability.py \
  training/lunar_policy_training/tests/test_formal_cache.py \
  training/lunar_policy_training/tests/test_formal_start_qualification.py
```

- [ ] **Step 3: Implement cache-v5 payload and exact-state LOS union**

Use the C++ graph's exact map-frame positions. Continue bit-packing the `5120 × 5120` coverable mask and computing the numerator by bit intersection. Never derive visibility from 4 m cell centers when an exact state pose is available.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/task5-bridge
set +u
source /opt/ros/humble/setup.bash
source "$BRIDGE_ROOT/install/setup.bash"
set -u
export PYTHONPATH="$BRIDGE_ROOT/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_coverability.py \
  training/lunar_policy_training/tests/test_formal_cache.py \
  training/lunar_policy_training/tests/test_formal_start_qualification.py
git add training/lunar_policy_training
git commit -m "feat(training): materialize primitive coverability cache v5"
```

---

### Task 7: Maintain the observed-only graph in each formal episode

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/environment/primitive_reachability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py`
- Create: `training/lunar_policy_training/tests/test_primitive_reachability.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_formal_resume_state.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class ObservedPrimitiveSnapshot:
    revision: int
    graph_sha256: str
    state_ids: np.ndarray
    positions_m: np.ndarray
    yaw_rad: np.ndarray
    cells: np.ndarray
    path_cost: np.ndarray
    recoverable: np.ndarray
    direct_successor: np.ndarray
    edge_source_ids: np.ndarray
    edge_target_ids: np.ndarray
```

- Each `FormalEpisodeWorker` owns one native engine. Every reveal updates it before candidate construction; continuation replans refresh maps but cannot reuse a snapshot whose map revision differs.
- Replay rebuilds the same graph sequence from reveal history and checks final graph SHA/revision. Environment state schema becomes `lunar-formal-environment-state/v5`.

- [ ] **Step 1: Write observed graph and replay RED tests**

Prove no truth object is accepted by the wrapper, revisions are strictly monotonic, no-change reveals preserve graph hash, changed evidence updates candidates, exact current state survives a partially observed 4 m global cell through local primitives, and snapshot/replay graph identity is byte-identical.

- [ ] **Step 2: Run RED, implement episode ownership, and run GREEN**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/task5-bridge
set +u
source /opt/ros/humble/setup.bash
source "$BRIDGE_ROOT/install/setup.bash"
set -u
export PYTHONPATH="$BRIDGE_ROOT/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_primitive_reachability.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py
```

- [ ] **Step 3: Commit**

```bash
git add training/lunar_policy_training
git commit -m "feat(training): maintain observed primitive graphs"
```

---

### Task 8: Generate candidates and oracle opportunities from primitive states

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Modify: `training/lunar_policy_training/tests/test_frontier_oracle.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`

**Interfaces:**

- Remove `PlatformCandidateReachability.filter` from the formal path. CandidateBuilder receives one `ObservedPrimitiveSnapshot` and enumerates recoverable observation states first.
- WHEELED/LEGGED may use any recoverable state with exact displacement `<=30.0 m`; HOPPER requires `direct_successor=true`.
- `CandidateBatch` adds exact `target_positions_m [64,3]`, `target_yaw_rad [64]`, `primitive_state_ids [64]`, and `primitive_graph_revision`; padded entries use mask false and state ID zero.
- Gain remains observed-only at 0.2 m: unknown rays are transparent and known obstacles stop them. Frontier cells are optional search hints only.
- Oracle independently recomputes forward/returnable labels from snapshot edges and enumerates information/transit opportunities without calling production sorting.

- [ ] **Step 1: Add graph-first RED tests**

Assert every emitted state is recoverable in the current snapshot; an unreachable high-gain frontier is absent; a reachable positive-gain non-anchor state is emitted; HOPPER multi-hop non-successor is absent; more than 64 states truncate stably; fewer than 64 use only false padding; sparse positive sets scan further distance layers; zero-gain transit appears only when needed; truth mutation leaves output byte-identical; and `production empty + oracle opportunity` raises the invariant error.

- [ ] **Step 2: Run RED, implement graph-first enumeration, and run GREEN**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/task5-bridge
set +u
source /opt/ros/humble/setup.bash
source "$BRIDGE_ROOT/install/setup.bash"
set -u
export PYTHONPATH="$BRIDGE_ROOT/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_frontier_oracle.py \
  training/lunar_policy_training/tests/test_formal_builder.py
```

- [ ] **Step 3: Commit**

```bash
git add training/lunar_policy_training
git commit -m "feat(training): generate candidates from primitive states"
```

---

### Task 9: Freeze diagnostics, termination, metrics, and v10 identity

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/macro_step.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/training_metrics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/training_semantics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/tests/test_training_metrics.py`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_parallel_pool.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`
- Modify: `training/lunar_policy_training/tests/test_closed_loop_gate.py`

**Interfaces:**

- `CandidateDiagnostics` contains the exact counts frozen in design section 7, including primitive states, forward/returnable/recoverable states, positive/transit/emitted states, planner rejections, and invalidated/revalidated edges.
- Terminal reasons are `SUCCESS`, `NO_RECOVERABLE_OBSERVATION_STATE`, `VISITED_EXHAUSTED`, `ZERO_GAIN`, `NO_TRANSIT_OPPORTUNITY`, `PLANNER_REJECTED_ALL`, `HARD_FAILURE`, and `CANCELED`.
- Metrics and JSON reports serialize every field explicitly; missing or negative values fail closed. Cache v5, environment-state v5, and training-semantics v10 reject all old checkpoints/manifests.

- [ ] **Step 1: Write RED identity/diagnostic tests**

Assert exact terminal classification, diagnostics aggregation across workers, JSON round-trip, coverage monotonicity with a frozen denominator, old v9/v4 identities rejected, and a fresh run manifest has `global_step=0`, `resume_parent=null`, and `warm_start_parent=null`.

- [ ] **Step 2: Run RED, implement identity propagation, and run GREEN**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/task5-bridge
set +u
source /opt/ros/humble/setup.bash
source "$BRIDGE_ROOT/install/setup.bash"
set -u
export PYTHONPATH="$BRIDGE_ROOT/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_training_metrics.py \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_cli.py \
  training/lunar_policy_training/tests/test_closed_loop_gate.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py
```

- [ ] **Step 3: Commit**

```bash
git add training/lunar_policy_training
git commit -m "feat(training): freeze primitive exploration semantics v10"
```

---

### Task 10: Verify, rebuild cache v5, pass gates, and launch fresh training

**Files:** Verify all Task 1-9 files. Generated outputs remain under `/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10`.

- [ ] **Step 1: Run clean native Release build and all native tests**

```bash
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
VERIFY_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/verify-${SOURCE_COMMIT:0:12}
set +u
source /opt/ros/humble/setup.bash
set -u
test "$ROS_DISTRO" = humble
colcon --log-base "$VERIFY_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$VERIFY_ROOT/build" --install-base "$VERIFY_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon --log-base "$VERIFY_ROOT/log-test" test --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$VERIFY_ROOT/build" --install-base "$VERIFY_ROOT/install" \
  --merge-install --event-handlers console_direct+
colcon test-result --test-result-base "$VERIFY_ROOT/build" --verbose
```

- [ ] **Step 2: Run full Python and repository verification against that bridge**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
VERIFY_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/verify-${SOURCE_COMMIT:0:12}
BRIDGE_PY="$VERIFY_ROOT/install/local/lib/python3.10/dist-packages"
source "$VERIFY_ROOT/install/setup.bash"
export PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q training/lunar_policy_training/tests
python3 tools/check_repository_boundaries.py .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q tests/foundation/test_repository_boundaries.py
python3 - <<'PY'
from pathlib import Path
for path in (
    Path('docs/superpowers/specs/2026-08-10-platform-coverable-exploration-design.md'),
    Path('docs/superpowers/plans/2026-08-11-primitive-reachability-exploration.md'),
):
    path.read_text(encoding='utf-8')
PY
git diff --check
```

- [ ] **Step 3: Generate current source-bound sensor performance evidence**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
VERIFY_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/verify-${SOURCE_COMMIT:0:12}
BRIDGE_PY="$VERIFY_ROOT/install/local/lib/python3.10/dist-packages"
RUN_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/run-${SOURCE_COMMIT:0:12}
mkdir -p "$RUN_ROOT"
source "$VERIFY_ROOT/install/setup.bash"
PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" training/tools/benchmark_sensor_observation.py \
  --native-benchmark "$VERIFY_ROOT/install/lib/lunar_planner_training_bridge/lunar_training_visibility_benchmark" \
  --output "$RUN_ROOT/sensor-performance.json" --workers 24
```

Require `passed=true`, current source commit/capability/semantics identities, candidate p95 below the frozen threshold, and no non-finite values.

- [ ] **Step 4: Build preflight cache v5 and run the one-scene × three-platform natural-terminal gate**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
VERIFY_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/verify-${SOURCE_COMMIT:0:12}
BRIDGE_PY="$VERIFY_ROOT/install/local/lib/python3.10/dist-packages"
RUN_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/run-${SOURCE_COMMIT:0:12}
source "$VERIFY_ROOT/install/setup.bash"
PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" -m lunar_policy_training.cli prepare-data \
  --source-lock /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json \
  --split-manifest /home/kai/CodexDownloads/lunar_navigation/volume3/data/splits/polar_split_v2.json \
  --cache-root "$RUN_ROOT/preflight-cache-v5" --materialization preflight \
  --preflight-scenario-limit 128
PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" -m lunar_policy_training.cli closed-loop-gate \
  --cache-manifest "$RUN_ROOT/preflight-cache-v5/cache-manifest.json" \
  --artifact-root "$RUN_ROOT/closed-loop-startup" \
  --minimum-scenes 1 --max-workers 3
```

Require exact graph identities, zero oracle contradictions/safety violations/invalid actions/execution failures, at least one real reference per platform, and a terminal consistent with its final coverage. Do not replace the frozen scene based on outcome.

- [ ] **Step 5: Materialize full cache v5, calibrate, and run formal preflight**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
VERIFY_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/verify-${SOURCE_COMMIT:0:12}
BRIDGE_PY="$VERIFY_ROOT/install/local/lib/python3.10/dist-packages"
RUN_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/run-${SOURCE_COMMIT:0:12}
source "$VERIFY_ROOT/install/setup.bash"
PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" -m lunar_policy_training.cli prepare-data \
  --source-lock /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json \
  --split-manifest /home/kai/CodexDownloads/lunar_navigation/volume3/data/splits/polar_split_v2.json \
  --cache-root "$RUN_ROOT/full-cache-v5" --materialization full
PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" -m lunar_policy_training.cli calibrate \
  --config training/configs/rtx4080_super_v3_joint.yaml \
  --artifact-root "$RUN_ROOT/formal-run" \
  --cache-manifest "$RUN_ROOT/full-cache-v5/cache-manifest.json" \
  --sensor-performance-report "$RUN_ROOT/sensor-performance.json"
PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" -m lunar_policy_training.cli formal-preflight \
  --config training/configs/rtx4080_super_v3_joint.yaml \
  --cache-manifest "$RUN_ROOT/full-cache-v5/cache-manifest.json" \
  --artifact-root "$RUN_ROOT/formal-preflight" \
  --calibration-root "$RUN_ROOT/formal-run" \
  --sensor-performance-report "$RUN_ROOT/sensor-performance.json"
```

Require every split/platform feasibility row and exclusion total, resume-equivalence proof for the new identity, and no checkpoint under the preflight root.

- [ ] **Step 6: Launch detached fresh step-0 training and verify first evidence**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
VERIFY_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/verify-${SOURCE_COMMIT:0:12}
BRIDGE_PY="$VERIFY_ROOT/install/local/lib/python3.10/dist-packages"
RUN_ROOT=/home/kai/CodexDownloads/lunar_navigation/primitive_reachability_v10/run-${SOURCE_COMMIT:0:12}
source "$VERIFY_ROOT/install/setup.bash"
PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" - "$RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'run-manifest.json').read_text(encoding='utf-8'))
assert manifest['global_step'] == 0
assert manifest.get('resume_parent') is None
assert manifest.get('warm_start_parent') is None
PY
nohup env \
  PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" -m lunar_policy_training.cli train \
  --config training/configs/rtx4080_super_v3_joint.yaml \
  --artifact-root "$RUN_ROOT/formal-run" \
  --sensor-performance-report "$RUN_ROOT/sensor-performance.json" \
  >"$RUN_ROOT/formal-train.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" >"$RUN_ROOT/formal-train.pid"
for attempt in $(seq 1 120); do
  kill -0 "$TRAIN_PID" 2>/dev/null || break
  if test -s "$RUN_ROOT/run-manifest.json" \
      && test -s "$RUN_ROOT/metrics/train.jsonl" \
      && find "$RUN_ROOT/checkpoints" -maxdepth 1 -type f -name '*.pt' -print -quit \
         | grep -q .; then
    break
  fi
  sleep 5
done
PYTHONPATH="$BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" - "$RUN_ROOT" "$TRAIN_PID" <<'PY'
import json, math, os, sys
from pathlib import Path
from lunar_policy_training.checkpoint import load_checkpoint
root = Path(sys.argv[1])
pid = int(sys.argv[2])
os.kill(pid, 0)
manifest = json.loads((root / 'run-manifest.json').read_text(encoding='utf-8'))
metrics = json.loads(
    (root / 'metrics/train.jsonl').read_text(encoding='utf-8').splitlines()[0]
)
def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True
def contains_key(value, name):
    if isinstance(value, dict):
        return name in value or any(contains_key(item, name) for item in value.values())
    if isinstance(value, list):
        return any(contains_key(item, name) for item in value)
    return False
assert finite(metrics)
for name in ('primitive_state_count', 'recoverable_observation_state_count',
             'invalidated_edge_count', 'revalidated_edge_count'):
    assert contains_key(metrics, name)
checkpoint_path = next((root / 'checkpoints').glob('*.pt'), None)
assert checkpoint_path is not None
checkpoint = load_checkpoint(checkpoint_path, run_kind='formal')
assert checkpoint.source_commit == manifest['source_commit']
assert checkpoint.run_identity.to_dict() == manifest['run_identity']
assert checkpoint.global_step <= manifest['global_step']
PY
```

If any gate fails, stop before launch or return to its owning RED test; do not weaken coverage, capability, network, sensor, or scene contracts.

## Completion Evidence

1. Tasks 1-9 each have RED/GREEN evidence and scoped commits.
2. Truth and observed-only graphs share the frozen platform algorithm/state schema while remaining information-isolated.
3. Every valid policy candidate references a recoverable observed graph state; HOPPER candidates are direct successors only.
4. Cache-v5 coverage denominator is exact, platform/start-specific, content-addressed, and episode-frozen.
5. Full Python, native Release, ROS bridge, deterministic, UTF-8, and repository-boundary checks pass at the final source commit.
6. The frozen one-scene × three-platform gate reaches auditable natural terminals without infrastructure/safety failure.
7. Full cache, calibration, and formal-preflight identities agree.
8. A randomly initialized formal run is alive from step 0 with finite first metrics and a valid first checkpoint; the repository remains clean and no generated artifact is tracked.
