# Workspace And Branch Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a recoverable, tested `integration` production branch with the optimized wheeled controller and dual executable/observable planner outputs while retaining exploration as a separate feature branch.

**Architecture:** Preserve both dirty trees first, then consolidate in a temporary linked worktree. Keep `MotionReference` as the execution contract and add `Path`/`TimedPath` as observation contracts; merge the resulting production branch into the exploration branch after production verification.

**Tech Stack:** Git worktrees and bundle, ROS 2 Jazzy/Humble interfaces, C++20, Python 3, CMake/colcon, pytest, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-08-24-workspace-branch-consolidation.md`

## Global Constraints

- Preserve every existing tracked and untracked user change before integration.
- Do not push: this repository has no configured remote.
- Do not remove `build`, `install`, `log`, `car_orin`, or parent scratch directories.
- Keep `MotionReference` as the controller input and Path messages as observation outputs.
- Keep `feat/jazzy-300m-exploration` separate from the production branch.
- Do not force-delete non-ancestor branch refs without a later explicit `discard` authorization.

---

### Task 1: Preserve dirty worktrees and create the archive

**Files:**
- Create: `docs/superpowers/specs/2026-08-24-workspace-branch-consolidation.md`
- Create: `docs/superpowers/plans/2026-08-24-workspace-branch-consolidation.md`
- Archive: `/home/kai/CodexDownloads/lunar_navigation/archives/lunar_pure_planner_orin-pre-consolidation-20260824.bundle`

**Interfaces:**
- Consumes: current `integration` and `feat/jazzy-300m-exploration` worktrees.
- Produces: `wip/integration-runtime-20260824`, `wip/exploration-readiness-20260824`, and a verified all-ref bundle.

- [ ] **Step 1: Verify both diffs are structurally valid**

Run `git diff --check` in both worktrees. Expected: exit 0 with no diagnostics.

- [ ] **Step 2: Commit the production checkpoint**

Create `wip/integration-runtime-20260824`, stage all current files, and commit
with `chore: checkpoint integration runtime work`.

- [ ] **Step 3: Commit the exploration checkpoint**

Create `wip/exploration-readiness-20260824`, stage the complete current dirty
set, and commit with `chore: checkpoint exploration readiness work`. This is a
preservation commit, not an assertion that every contained change is complete.

- [ ] **Step 4: Create and verify the archive**

Run `git bundle create ... --all`, then `git bundle verify` and
`git bundle list-heads`. Expected: both WIP refs and all existing feature refs
are present.

### Task 2: Create the production consolidation worktree

**Files:**
- Worktree: `/home/kai/CodexDownloads/lunar_navigation/.worktrees/integration-consolidation`

**Interfaces:**
- Consumes: `integration` and `wip/integration-runtime-20260824`.
- Produces: clean linked worktree on `integration` containing the checkpoint.

- [ ] **Step 1: Add the linked worktree**

Add the existing `integration` branch at the exact worktree path and verify
`GIT_DIR != GIT_COMMON` with no submodule superproject.

- [ ] **Step 2: Fast-forward the preserved production checkpoint**

Run `git merge --ff-only wip/integration-runtime-20260824`. Expected:
`integration` points at the checkpoint and the worktree remains clean.

- [ ] **Step 3: Run focused baseline checks**

Run:

```bash
python3 -m pytest -q -p no:cacheprovider \
  ros2_ws/src/lunar_pure_wheeled_controller/test \
  tests/test_launch_contract.py \
  tests/launch/test_lunar_surface_demo_contract.py \
  tests/test_car_orin_bundle.py
```

Record the exact baseline failures before adding controller behavior.

### Task 3: Restore the executable controller contract with TDD

**Files:**
- Modify: `config/pure_planner.yaml`
- Modify: `launch/lunar_surface_rviz_demo.launch.py`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/package.xml`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/launch/pure_wheeled_controller.launch.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/node.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/reference.py`
- Create: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/tracking.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_node.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_reference.py`
- Create: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_tracking.py`
- Modify: `tests/test_launch_contract.py`

**Interfaces:**
- Consumes: `MotionReference.trajectory`, `MotionReference.path_preview`, and the existing `Path`/`TimedPath` observation publishers.
- Produces: `wheeled_reference_topic`, `wheeled_path_topic`, and `wheeled_timed_path_topic`; controller commands on `/Car/T5/Car_Cmd_Vel`.

- [ ] **Step 1: Add a failing dual-output planner test**

Extend `pure_plan_motion_server_test.cpp` so a successful wheel plan requires a
non-empty executable `MotionReference`, matching non-empty `Path`, and a
`TimedPath` with the same path. Run that focused test and confirm it fails
because the checkpoint no longer creates the executable publisher.

- [ ] **Step 2: Add failing controller motion-mode tests**

Apply the exploration branch controller tests for trajectory parsing, reverse
translation, in-place spin, ordered cursor progression, cancellation, and clear
reset. Run pytest and confirm the checkpoint implementation fails those tests.

- [ ] **Step 3: Restore optimized controller implementation**

Apply commits `e06d074^..f1d1467`, resolving conflicts so the controller reads
`MotionReference` and retains all optimized trajectory tracking behavior.

- [ ] **Step 4: Restore the executable planner publisher**

Add `wheeled_reference_topic` alongside—not instead of—the Path topics. Publish
the same successful wheel result through all three outputs and publish empty
failure notifications consistently.

- [ ] **Step 5: Verify green and commit**

Run the focused C++ and Python tests, followed by launch-contract tests. Expected:
all pass. Commit with `fix: preserve executable wheel references with path outputs`.

### Task 4: Verify production and synchronize exploration

**Files:**
- Modify if conflicted: `.superpowers/sdd/2026-08-24-jazzy-exploration-closed-loop/progress.md`
- Modify if conflicted: `README.md`

**Interfaces:**
- Consumes: consolidated `integration` and reviewed changes from
  `wip/exploration-readiness-20260824`.
- Produces: updated `feat/jazzy-300m-exploration` containing both histories.

- [ ] **Step 1: Run the production verification matrix**

Run:

```bash
git diff --check
python3 -m pytest -q -p no:cacheprovider \
  tests/test_external_interface_contract.py \
  tests/test_launch_contract.py \
  tests/launch/test_lunar_surface_demo_contract.py \
  tests/test_car_orin_bundle.py
bash -n scripts/start_all.sh scripts/send_goal.sh
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-consolidation-20260824/production/log build \
  --base-paths ros2_ws/src \
  --build-base /tmp/lunar-planner-consolidation-20260824/production/build \
  --install-base /tmp/lunar-planner-consolidation-20260824/production/install \
  --packages-up-to lunar_pure_planner_ros lunar_pure_wheeled_controller \
  --event-handlers console_direct+
source /tmp/lunar-planner-consolidation-20260824/production/install/setup.bash
colcon --log-base /tmp/lunar-planner-consolidation-20260824/production/test-log test \
  --base-paths ros2_ws/src \
  --build-base /tmp/lunar-planner-consolidation-20260824/production/build \
  --install-base /tmp/lunar-planner-consolidation-20260824/production/install \
  --packages-select lunar_pure_planner_core lunar_pure_planner_ros \
    lunar_pure_wheeled_controller --event-handlers console_direct+
colcon test-result \
  --test-result-base /tmp/lunar-planner-consolidation-20260824/production/build \
  --all --verbose
```

Expected: every command exits zero and `colcon test-result` reports zero failed
tests on the exact `integration` tree.

- [ ] **Step 2: Verify and fast-forward the exploration checkpoint**

First run the exploration build and package tests against the WIP checkpoint.
Only if they pass, switch the exploration worktree back to
`feat/jazzy-300m-exploration` and run
`git merge --ff-only wip/exploration-readiness-20260824`.

- [ ] **Step 3: Merge production into exploration**

Merge `integration` into `feat/jazzy-300m-exploration`, resolving only genuine
README/controller overlap while retaining exploration-only simulation files.

- [ ] **Step 4: Run focused exploration verification**

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-consolidation-20260824/exploration/log build \
  --base-paths ros2_ws/src \
  --build-base /tmp/lunar-planner-consolidation-20260824/exploration/build \
  --install-base /tmp/lunar-planner-consolidation-20260824/exploration/install \
  --packages-up-to lunar_pure_exploration_sim lunar_pure_exploration_ros \
    lunar_pure_planner_ros lunar_pure_wheeled_controller \
  --event-handlers console_direct+
source /tmp/lunar-planner-consolidation-20260824/exploration/install/setup.bash
colcon --log-base /tmp/lunar-planner-consolidation-20260824/exploration/test-log test \
  --base-paths ros2_ws/src \
  --build-base /tmp/lunar-planner-consolidation-20260824/exploration/build \
  --install-base /tmp/lunar-planner-consolidation-20260824/exploration/install \
  --packages-select lunar_pure_exploration_core lunar_pure_exploration_ros \
    lunar_pure_exploration_sim lunar_pure_wheeled_controller \
  --event-handlers console_direct+
colcon test-result \
  --test-result-base /tmp/lunar-planner-consolidation-20260824/exploration/build \
  --all --verbose
python3 -m pytest -q -p no:cacheprovider \
  tests/launch/test_pure_exploration_launch.py \
  tests/launch/test_jazzy_300m_exploration_sim.py -k static
```

Expected: every command exits zero and both CTest/pytest result sets contain
zero failures.

### Task 5: Organize redundant worktrees and report retained refs

**Files:**
- Remove only clean linked worktree directories registered for `feat/planner-core-opt` and `feat/planner-opt-*`.

**Interfaces:**
- Consumes: verified bundle and clean redundant worktrees.
- Produces: compact `git worktree list`; preserved branch refs and explicit cleanup report.

- [ ] **Step 1: Re-check every cleanup candidate**

Run `git status --porcelain -uall` in each candidate. Expected: empty output.

- [ ] **Step 2: Remove clean linked worktrees without force**

Run `git worktree remove` for each exact registered path, then
`git worktree prune`. Stop if Git reports uncommitted content.

- [ ] **Step 3: Remove fully merged refs and retain non-ancestor refs**

Delete only branches accepted by `git branch -d`. Rename remaining worker refs
under `archive/planner-opt-20260824/` so their commits remain named and
recoverable without implying that they still need integration.

- [ ] **Step 4: Final audit**

Report branch topology, worktree paths, bundle verification, exact test counts,
and any retained archive refs. Leave generated build/install/log directories
untouched.
