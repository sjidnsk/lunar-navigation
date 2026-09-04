# Orin Humble Incremental Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the reviewed incremental exploration stack and deliver an `orin-humble` branch containing the Humble/Orin runtime, incremental RViz/demo, and six operator scripts.

**Architecture:** General launch and operator-interface changes are made on a feature branch and merged into `integration/pure-planner-orin`. `orin-humble` is then cut from that integration baseline and pruned to runtime source, configuration, incremental RViz/demo, and operational scripts. Its build always uses `ros2_ws/build`, `ros2_ws/install`, and `ros2_ws/log`.

**Tech Stack:** ROS 2 Humble, `ament_cmake`, ROS 2 launch Python, Bash, `colcon`, `rsync`, Python `pytest` static contract tests.

**Spec:** `docs/validation/2026-09-03-incremental-exploration-navigation.md` plus the user-confirmed deployment scope from 2026-09-04.

## Global Constraints

- `import/pure-planner-orin-main` remains read-only.
- First fast-forward the reviewed incremental branch into `integration/pure-planner-orin`; do not merge unrelated feature branches by name.
- Keep incremental RViz/demo and both wheel and legged configuration files.
- Do not change the wheel or legged 8-neighbor A* algorithms, production sensor settings, or exploration completion policy.
- Do not add readiness polling, controller startup, RViz startup, or map/TF safety gates to the Orin scripts.
- Final `orin-humble` contains no `docs/`, `tests/`, package `test/` directories, build output, or legacy pure-planner/controller runtime.
- Final build uses only `ros2_ws/build`, `ros2_ws/install`, and `ros2_ws/log`; do not create temporary build directories.
- Humble/Orin validation remains distinct from local Jazzy/static validation.

---

### Task 1: Consolidate the reviewed baseline and create the isolated implementation branch

**Files:**
- Modify: branch pointer `integration/pure-planner-orin`
- Create: worktree `.worktrees/orin-humble-deployment` on `feat/orin-humble-deployment`

**Interfaces:**
- Consumes: clean `fix/incremental-exploration-navigation-review` at `a056a69`
- Produces: clean feature branch rooted at the reviewed incremental implementation

- [x] **Step 1: Verify branch ancestry and clean worktrees**

Run: `git merge-base --is-ancestor integration/pure-planner-orin fix/incremental-exploration-navigation-review`

Expected: exit code `0`; both integration and review worktrees are clean.

- [x] **Step 2: Run the baseline static interface contract**

Run: `python3 -m pytest tests/test_incremental_navigation_interface_contract.py -q`

Expected: `10 passed` before the deployment-interface changes.

- [x] **Step 3: Fast-forward and create the feature worktree**

Run: `git merge --ff-only fix/incremental-exploration-navigation-review`, then `git worktree add -b feat/orin-humble-deployment .worktrees/orin-humble-deployment integration/pure-planner-orin`.

Expected: integration and the feature branch point at `a056a69` before later commits.

### Task 2: Define red tests for the split Orin operator interface

**Files:**
- Create: `tests/test_orin_humble_operator_scripts.py`
- Test: `tests/test_orin_humble_operator_scripts.py`

**Interfaces:**
- Consumes: repository root and the formal `exploration_navigation.launch.py`
- Produces: a static contract for six scripts and the two launch selection flags

- [x] **Step 1: Write failing script and launch contract tests**

```python
def test_navigation_start_script_selects_only_navigation() -> None:
    script = _read("scripts/orin/start_navigation.sh")
    assert "start_navigation:=true" in script
    assert "start_exploration:=false" in script


def test_exploration_start_script_selects_only_exploration() -> None:
    script = _read("scripts/orin/start_exploration.sh")
    assert "start_navigation:=false" in script
    assert "start_exploration:=true" in script


def test_build_script_uses_only_fixed_workspace_outputs() -> None:
    script = _read("scripts/orin/build.sh")
    assert "--build-base" not in script
    assert "--install-base" not in script
    assert "--log-base" not in script
    assert "mktemp" not in script
```

- [x] **Step 2: Run the new test and observe the missing-script failure**

Run: `python3 -m pytest tests/test_orin_humble_operator_scripts.py -q`

Expected: fail because `scripts/orin/*.sh` and the launch flags do not yet exist.

### Task 3: Implement the split start interface and six minimal scripts

**Files:**
- Modify: `launch/exploration_navigation.launch.py`
- Create: `scripts/orin/deploy.sh`
- Create: `scripts/orin/build.sh`
- Create: `scripts/orin/start_navigation.sh`
- Create: `scripts/orin/start_exploration.sh`
- Create: `scripts/orin/publish_navigation_goal.sh`
- Create: `scripts/orin/publish_exploration_task.sh`

**Interfaces:**
- Consumes: `lunar_planning_msgs/action/NavigateToPose`, `lunar_pure_exploration_msgs/msg/PureExplorationTask`, and installed `exploration_navigation.launch.py`
- Produces: separate navigation/exploration startup and manual navigation/exploration task publication

- [x] **Step 1: Add `start_navigation` and `start_exploration` launch arguments**

Use `IfCondition(LaunchConfiguration(...))` on the two `Node` actions. Both defaults are `true`, so the current aggregate launch retains its behavior.

- [x] **Step 2: Add minimal scripts**

`build.sh` sources `/opt/ros/humble/setup.bash`, changes to `ros2_ws`, and runs:

```bash
colcon build --merge-install --packages-up-to \
  lunar_incremental_navigation_ros lunar_pure_exploration_ros \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
```

`start_navigation.sh` and `start_exploration.sh` source the fixed installed overlay and call the formal launch with mutually exclusive flags. `publish_navigation_goal.sh` calls `/Car/T4/navigation/navigate_to_pose` using `NavigateToPose`; `publish_exploration_task.sh` publishes a rectangular `PureExplorationTask` in `map`.

- [x] **Step 3: Verify the tests and shell syntax**

Run: `python3 -m pytest tests/test_orin_humble_operator_scripts.py -q`

Expected: all tests pass.

Run: `bash -n scripts/orin/*.sh`

Expected: exit code `0`.

### Task 4: Merge the general interface changes and cut the deployment branch

**Files:**
- Modify: branch pointer `integration/pure-planner-orin`
- Create: worktree `.worktrees/orin-humble` on `orin-humble`

**Interfaces:**
- Consumes: verified `feat/orin-humble-deployment`
- Produces: deployment branch rooted at the integration commit that contains the common scripts and launch interface

- [ ] **Step 1: Commit the feature branch with explicit paths**

Run: `git add launch/exploration_navigation.launch.py scripts/orin tests/test_orin_humble_operator_scripts.py docs/superpowers/plans/2026-09-04-orin-humble-deployment.md`.

Expected: only the listed general interface, test, and plan files are staged.

- [ ] **Step 2: Merge the feature branch into integration with `--ff-only`**

Run: `git merge --ff-only feat/orin-humble-deployment` from the clean integration worktree.

Expected: integration receives the common implementation without a merge commit.

- [ ] **Step 3: Create `orin-humble` from integration**

Run: `git worktree add -b orin-humble .worktrees/orin-humble integration/pure-planner-orin`.

Expected: deployment worktree starts at the integrated feature commit.

### Task 5: Prune `orin-humble` to the incremental runtime and demo closure

**Files:**
- Delete: `docs/`, `tests/`, all package `test/` directories, legacy source packages, legacy configs, legacy launches, legacy scripts, and obsolete tools
- Modify: `launch/exploration_navigation.launch.py`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/package.xml`
- Modify: package manifests/CMake files only where legacy build targets are removed

**Interfaces:**
- Consumes: incremental navigation/exploration packages and incremental RViz/demo package assets
- Produces: a source-only deployment branch that builds the incremental stack and demo without legacy package closure

- [ ] **Step 1: Remove only explicit, verified legacy paths**

Keep `lunar_incremental_navigation_core`, `lunar_incremental_navigation_ros`, `lunar_pure_exploration_core`, `lunar_pure_exploration_msgs`, `lunar_pure_exploration_ros`, `lunar_planning_msgs`, `config/exploration_navigation.yaml`, `config/incremental_navigation_interfaces.yaml`, `config/wheel.yaml`, `config/legged.yaml`, the two incremental launch files, the incremental RViz file, and `scripts/orin/`.

- [ ] **Step 2: Make the formal launch incremental-only**

Remove the `legacy` dispatch and its imports from the deployment copy while keeping `start_navigation` / `start_exploration` selection.

- [ ] **Step 3: Remove legacy explorer build targets only**

Make `lunar_pure_exploration_ros` build and install `incremental_exploration_node` and its incremental library only. Do not alter the 8-neighbor planner implementation or the retained demo targets.

- [ ] **Step 4: Verify deployment tree and build configuration**

Run: `git ls-files docs tests | wc -l`

Expected: `0`.

Run: `git ls-files | rg '(^|/)test/'`

Expected: no output.

Run: `bash -n scripts/orin/*.sh && git diff --check`

Expected: exit code `0`.

### Task 6: Verify and commit the deployment branch

**Files:**
- Modify: branch pointers `integration/pure-planner-orin` and `orin-humble`

**Interfaces:**
- Consumes: source-only `orin-humble` tree
- Produces: explicit integration and deployment commits with a recorded validation boundary

- [ ] **Step 1: Build on a Humble/Orin target using fixed directories**

Run: `scripts/orin/build.sh`

Expected: generated output appears only in `ros2_ws/build`, `ros2_ws/install`, and `ros2_ws/log`.

- [ ] **Step 2: Record unavailable target validation accurately**

If no Humble/Orin target is available in this session, do not claim target build success; record it as `NOT_RUN` while completing local static validation.

- [ ] **Step 3: Commit with explicit paths**

Run: `git add -u` is not permitted. Stage each retained changed path explicitly, review `git diff --cached --check`, and commit `chore: create Orin Humble incremental deployment branch`.
