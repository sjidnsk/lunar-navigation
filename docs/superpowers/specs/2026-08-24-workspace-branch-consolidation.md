# Workspace And Branch Consolidation Design

## Goal

Consolidate the production planner work onto `integration`, preserve the Jazzy
300 m exploration feature as a separate branch, restore the optimized wheeled
controller execution contract, and reduce redundant planner optimization
worktrees without losing recoverability.

## Current state

- `integration` contains the completed planner-core optimization history and
  has uncommitted runtime hardening, local-start, traversability, path-output,
  controller, and Orin bundle changes.
- `feat/jazzy-300m-exploration` is a descendant of `integration` with 34 unique
  commits. Its live worktree contains additional uncommitted readiness and
  global-goal-feasibility work; the complete live set must be checkpointed
  without assuming it is ready to integrate.
- `feat/planner-core-opt` is an ancestor of `integration`.
- The `feat/planner-opt-*` worker branches are patch-equivalent to, or
  superseded by, the consolidated planner-core history. They are not all Git
  ancestors, so their refs must not be force-deleted without a recoverable
  archive and explicit final cleanup authorization.
- There is no configured Git remote.

## Production branch boundary

`integration` remains the production base. Exploration simulation sources,
operator tooling, RViz configuration, and acceptance reports stay on
`feat/jazzy-300m-exploration`.

The planner must expose two distinct contracts:

1. `/Car/T4/planning/wheeled_reference` carries `MotionReference` and remains
   the executable controller input. Its trajectory retains signed velocity and
   yaw-rate information needed for reverse translation and in-place rotation.
2. `/Car/T4/planning/wheeled_path` and
   `/Car/T4/planning/wheeled_path_timing` carry `Path` and `TimedPath` for RViz,
   rosbag, and external observation. These topics must not replace the
   executable reference.

The optimized controller implementation from the exploration branch is folded
into `integration`; the exploration branch then merges the updated production
base so both branches use the same controller and planner runtime.

## Preservation and cleanup

Before integration, each dirty worktree receives a named checkpoint commit. A
Git bundle containing all refs is created outside the repository and verified.
No ignored build, install, log, deployment bundle, or parent-directory scratch
artifact is removed as part of this change.

Clean redundant linked worktrees may be removed after their branch refs are
preserved. Non-ancestor worker refs are retained under an archive namespace
unless the user separately authorizes their permanent deletion.

## Acceptance

- Both original dirty states are reachable from named commits and the verified
  bundle.
- `integration` publishes executable `MotionReference` plus observable
  `Path`/`TimedPath` outputs.
- The wheeled controller executes forward, reverse, and in-place-spin trajectory
  modes and remains the sole publisher of `/Car/T5/Car_Cmd_Vel`.
- Planner, ROS adapter, controller, launch-contract, and bundle tests pass on
  the consolidated production tree.
- The exploration branch contains the production consolidation plus only the
  checkpointed exploration changes that pass focused review and tests.
- Redundant planner optimization worktrees are removed only when clean.
