# Closed-loop Task 5 implementation report — operator workflow and full-run acceptance

## Scope and commit

- Start HEAD: `6a18b75`.
- Operator workflow commit: `ffd60e9` (`docs: add 300m Jazzy exploration workflow`).
- Added strict shell entry `scripts/run_jazzy_300m_exploration_sim.sh`, reusable Python
  operator `scripts/jazzy_300m_operator.py`, 12 operator contract/behavior tests, and a
  Chinese UTF-8 README section.
- The operator sources Jazzy plus an explicit overridable external overlay, validates
  installed packages/executables/launch assets, reserves a candidate empty ROS domain,
  creates a unique external run directory, launches RViz by default, and accepts only
  an exact successful terminal summary with three consistent nonempty artifacts.
- Teardown reuses the independently reviewed Task 4 identity model: complete
  `(pid,start_time,pgid,sid)` identities, bounded same-group SIGINT, then same-group
  SIGTERM only for residual identities. It contains no recursive deletion, broad process
  match, broad kill, or unbounded wait.

## TDD and verification evidence

- RED: all 12 new operator tests failed because both entry files were absent.
- GREEN: `12 passed in 0.29s`, including strict summary/CSV acceptance and a synthetic
  leader-exits/child-ignores-SIGINT teardown case.
- A real `--help` invocation then exposed Jazzy setup reading an unset internal variable
  under `set -u`. A regression was added first; setup sourcing now temporarily disables
  nounset and immediately restores it. The shell remains `set -euo pipefail` for all
  operator work.
- Fresh runtime closure build with `BUILD_TESTING=OFF`: 8 packages built.
- Fresh focused simulation test build: 8/8 CTest wrappers, 73 tests, 0 failures.
- Launch plus operator tests: 23 passed, 1 opt-in live test skipped.
- Fresh opt-in live smoke: 1 passed, 11 deselected in 3.80 s.
- UTF-8 reads for README and both scripts passed; `git diff --check` passed.
- A whole-workspace `BUILD_TESTING=ON` build remains blocked before this task by unchanged
  `anytime_hopper_planner_test.cpp` using `std::ranges::any_of` without the required
  standard header. Task 5 did not modify planner, incremental optimization, or SE(2)
  files.

## Full-run attempt 1 — default unoptimized build

Run directory:

`/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/run-20260824-062821-seed-20260824-c390f2e2`

- RViz requested and started; OpenGL 4.6 initialized and RViz attempted a 300 by 300 Map.
  Mesa then reported a GLSL sampler link error, so startup is proven but correct rendering
  of every display is not claimed.
- The vehicle moved 4.199363644 m and coverage reached
  `0.0010582639714625446`.
- 286 global and 286 local planner calls were recorded. After the first movement, the
  current candidate repeatedly consumed the local fixed deadline at about 0.883 s and
  returned `TIMEOUT`; pose and coverage stopped changing.
- The run was operator-interrupted after 231.30 wall s rather than left in a deterministic
  retry loop. Recorder summary correctly says `success=false`, `SHUTDOWN`, and zero
  completed goals. All nine exact process identities disappeared on SIGINT, no SIGTERM
  was needed, and domain 44 was empty.

## Full-run attempt 2 — RelWithDebInfo build

Run directory:

`/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/run-20260824-063501-seed-20260824-66ff4b6e`

- Rebuilt the same runtime closure with `-DCMAKE_BUILD_TYPE=RelWithDebInfo`; no exploration
  or planning algorithm was changed.
- RViz again initialized OpenGL 4.6 and attempted the 300 by 300 Map; the same GLSL sampler
  error remained.
- The vehicle moved 3.199708548 m and coverage reached
  `0.00085612366230677765`.
- 106 global and 106 local calls were recorded: 60 `PLAN_FOUND`, then 46 consecutive
  `TIMEOUT` responses for the same current candidate. Optimization reduced global calls
  from about 68 ms to about 18–19 ms, but the difficult local search still consumed about
  932–935 ms and hit the fixed request budget.
- The run was interrupted after 53.71 operator wall s once the invariant retry was proven.
  Recorder summary correctly says `success=false`, `SHUTDOWN`, zero completed goals,
  wall elapsed 52.4468 s and simulated elapsed 1048.2362 s. All nine exact identities
  disappeared on SIGINT, no SIGTERM was needed, and domain 34 was empty.
- Focused simulation timing evidence remains `complete_tick_average_ms=16.2297`,
  `complete_tick_max_ms=16.7649`, `deadline_misses=0`. Runtime plant deadline-miss count is
  not exported in the full-run result schema, so no unsupported full-run count is claimed.

## Acceptance result and blocker

Task 5 full acceptance is **not complete**. Neither attempt reached `COMPLETED` with
`COMPLETED_NO_REACHABLE_FRONTIER`; neither had a completed goal, and both summaries
correctly remained unsuccessful.

The reproducible blocker is an existing semantic interaction: planner `TIMEOUT` is
classified as retryable during candidate validation, so the explorer intentionally keeps
the same candidate and does not advance or mark it unreachable. With a stationary vehicle,
the map does not change and retrying produces the same timeout indefinitely. Treating
timeout as no path would violate the frozen rule that timeout cannot establish “no reachable
frontier,” so Task 5 does not weaken that rule or fabricate completion.

Completion now depends on making the difficult local request produce a real result within
the fixed algorithm budget (for example, after the separately scoped incremental planner
optimization is integrated and reviewed), then rerunning this same strict operator entry.
