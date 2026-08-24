# Stop-Gated Local-Segment Exploration Design

## Status

Approved in chat on 2026-08-24. This document records the design for review
before implementation.

## Goal

Make exploration use each frontier candidate as a direction and information-gain
guide, not as a waypoint that the vehicle must physically reach. For each
exploration cycle, the planner will compute a global route toward the selected
frontier and one bounded local executable trajectory. The controller will track
that complete local trajectory without stopping between motion primitives. At
the local trajectory endpoint, or after an execution invalidation, the vehicle
will stop before the explorer starts the next planning cycle.

The design prioritizes planning success. It does not change the global or local
planning algorithms, vehicle footprint, obstacle inflation, sweep validation,
motion primitives, frontier scoring, candidate planning count, or exploration
completion rule.

## Confirmed Semantics

The three path concepts have different responsibilities:

- A **frontier candidate** indicates a promising exploration direction and
  sensor viewpoint. It can move, split or disappear as the map changes.
- The **global route preview** establishes reachability and path cost from the
  current pose toward that frontier.
- The **local executable trajectory** is the bounded trajectory the controller
  actually follows during the current exploration cycle.

Reaching the endpoint of a local executable trajectory is a successful rolling
exploration step. It is not necessary to reach the original frontier position.
After the step, the explorer updates coverage, detects frontiers again and may
select the same direction or a different one.

## Existing Planner Contract Reused

No new planning message is required. `MotionReference` already separates:

- `path_preview`, which contains the global route preview; and
- `trajectory`, which contains the executable wheel trajectory.

The normal lunar-surface `PlanMotion` path already performs a global search,
selects local portal goals and plans one local trajectory. When the requested
frontier lies outside the certified local map, `SelectSurfaceLocalGoals()` uses
an 8 m rolling horizon. `ComposeSurfaceReference()` retains the full global
route as the preview while keeping only the locally planned execution data in
the trajectory.

The 300 m simulation will keep `rolling_surface_enabled: false`. The planner
server's internal continuous rolling loop is not used because it replans from
live odometry while the vehicle may still be moving. Exploration owns the
stop-plan-execute cycle instead.

## Root Cause Addressed

The observed early candidate exhaustion was not evidence that all candidate
directions were spatially unreachable. Candidate planning and execution
replanning could be invoked while the controller was still tracking a previous
reference. The planner then received a nonzero initial velocity and curvature
that did not match the first available discrete motion primitive. The resulting
`NO_PATH` was treated as a persistent candidate failure and could prematurely
remove otherwise reachable frontiers.

Obstacle inflation and footprint sweep validation are not relaxed. They remain
necessary safety checks and do not resolve a dynamically incompatible planning
start.

## Chosen Architecture

### 1. Stationary Planning Admission Gate

The exploration ROS runtime will track the latest odometry twist in addition to
the existing pose. A new internal stationary gate controls admission to every
candidate-planning batch and execution-replan request.

If an active execution reference exists when planning is required, exploration
publishes its `plan_id` on `/Car/T4/execution/cancel`. The existing controller
clears the matching reference, publishes a zero `Twist`, and continues
publishing zero commands while inactive.

Planning begins only after odometry is within configurable stationary limits
for three consecutive samples. Initial defaults are:

```yaml
stop_before_planning: true
stationary_linear_speed_mps: 0.01
stationary_angular_speed_radps: 0.02
stationary_confirmation_samples: 3
stop_wait_diagnostic_s: 5.0
```

The gate uses the received twist values only. It does not validate odometry
timestamps, covariance or freshness. `stop_wait_diagnostic_s` uses the local
steady clock only to publish a diagnostic; it does not declare a frontier
unreachable or terminate exploration. If the threshold is exceeded, the
explorer remains stopped and waiting.

No stationary-speed normalization is included in the first implementation. The
planner continues to consume the real odometry supplied by Task 3. A separate,
explicit input deadband may be designed later only if stationary vehicle data
on the target platform exhibits residual velocity noise that prevents planning.

### 2. Candidate Planning While Parked

After the stationary gate opens, exploration captures a new planning epoch and
rebuilds the frontier and candidate batch from the current map and pose. The
existing candidate count and evaluation order remain unchanged.

Every candidate is still submitted to `PlanMotion`. The returned global route
preview supplies the path-length term for ranking, and the returned local
trajectory is retained as the executable reference for that candidate. All
candidate evaluations in the batch complete while the controller remains
inactive.

Only after final ranking does exploration publish the selected candidate's
`MotionReference` to `/Car/T4/execution/motion_reference`.

### 3. Continuous Execution of One Local Trajectory

The controller tracks the complete selected local trajectory continuously.
Forward, reverse, circular-arc and in-place-spin primitives within the same
trajectory are not separate exploration cycles and do not introduce stop-plan
boundaries.

Map, pose and coverage updates continue to be received during execution, but
the explorer does not start a background candidate-planning batch merely
because a new map message arrived. A map update may mark a rebuild pending.
Planning starts only after one of the stop conditions below has occurred.

### 4. Local Segment Completion and Replanning

A new exploration planning cycle is triggered when:

- the local executable trajectory reaches its endpoint;
- execution validation determines the remaining trajectory is blocked;
- the vehicle is stuck or has exceeded the allowed path deviation;
- the task is paused or canceled; or
- the active reference is otherwise invalidated.

For normal local-segment completion, the controller finishes the trajectory and
publishes zero velocity. Exploration enters `REPLANNING` with reason code
`WAITING_FOR_STOP`, waits for stationary confirmation, releases the old active
goal without recording a failure, and builds a fresh frontier batch.

For blocked, stuck or invalidated execution, exploration first cancels the
active reference, then follows the same stationary gate. It does not call the
planner from the moving state that triggered the recovery.

The existing public exploration-status message remains unchanged. The
`REPLANNING` state plus reason code communicates the internal stop wait.

## Frontier and Failure-Memory Semantics

Frontier lifetime and reachability failure are separated:

- Reaching a local segment endpoint without reaching the frontier is success,
  not `NO_PATH`.
- A frontier that disappears after new observations is released normally, not
  added to failure memory.
- A frontier whose position changes is regenerated as a new candidate from the
  current map.
- Planner results belonging to an older planning epoch, or to requests issued
  before stationary confirmation, are canceled or ignored and never stored as
  candidate failures.
- Only an exhaustive `NO_PATH` returned from a stationary, current-epoch
  candidate request enters failure memory.
- Retryable timeout, cancellation, contract or resource failures retain their
  existing non-exhaustive handling.

Exploration completes only after a fresh stationary planning cycle finds that
the task area contains no reachable frontier. Coverage remains a reported
statistic and is not an exploration termination threshold.

## Timing and Diagnostics

Existing planner diagnostics remain authoritative for each candidate request:

- global-planning elapsed time and call count;
- local-goal construction time;
- local-search elapsed time and call count;
- certification and total planning time;
- planning outcome and reason code.

Exploration adds separate orchestration fields for:

- stop-wait elapsed time;
- speed observed when the stop gate was entered;
- speed observed when stationary confirmation completed;
- number of stationary confirmation samples; and
- planning epoch.

Stop-wait time is not included in global, local or total planner algorithm
timing. Planner timing continues to use the local monotonic clock and remains
suitable for deployment on the vehicle.

## Code Boundaries

Expected production changes are limited to exploration orchestration:

- `lunar_pure_exploration_ros/src/exploration_node.cpp`
  - cache odometry linear and angular speed;
  - add the stationary planning-admission gate;
  - retain the active `plan_id` until stop confirmation;
  - rebuild a fresh candidate batch after local segment completion;
  - reject stale planning completions by epoch;
  - preserve planner timing while recording stop-wait diagnostics.
- `lunar_pure_exploration_ros` configuration and launch files
  - expose the stationary-gate parameters;
  - keep planner-internal rolling disabled for the exploration simulation.

The controller cancel behavior already meets the required command contract.
Controller production changes are not expected unless testing finds a concrete
cancel or zero-command defect.

No change is planned for:

- `PlanMotion.action` or `MotionReference.msg`;
- global or local planner algorithms;
- motion-primitive generation or timing;
- obstacle inflation, unknown-cell handling or footprint sweep validation;
- Task 3 input topics or Task 4 output topic names.

## Test-First Acceptance Criteria

Implementation begins with failing exploration tests for the moving-start
regression.

Required automated evidence:

1. While odometry is above either stationary threshold, no candidate or replan
   request is sent to `PlanMotion`.
2. If a reference is active, entering the planning gate publishes exactly its
   `plan_id` to the execution-cancel topic and the controller publishes zero
   velocity.
3. Fewer than three stationary samples do not open the gate; the third
   consecutive valid sample opens it exactly once.
4. A moving sample resets the consecutive stationary count.
5. A stop wait exceeding the diagnostic duration does not add a failure-memory
   entry and does not complete the task.
6. Candidate requests and ranking remain unchanged after the gate opens.
7. The selected `MotionReference.path_preview` remains the global route used for
   cost, while its complete executable `trajectory` is published once and
   followed without stop events between primitives.
8. Local trajectory endpoint completion releases the old active goal without a
   failure, waits for stop, refreshes the map/frontiers and begins a new batch.
9. A frontier that disappears during execution is not recorded as unreachable.
10. Only current-epoch stationary `NO_PATH` results contribute to exhaustive
    candidate failure and no-reachable-frontier completion.
11. Existing global/local planner timing records survive candidate failures,
    selection and execution transitions; stop-wait timing remains separate.
12. Existing exploration core, planner client, synthetic scenario, controller
    and planner regression suites pass.

Required runtime evidence:

1. Re-run the 300 m smoke scenario that previously produced moving-start
   failures for candidates 6 and 7.
2. Show from odometry and request logs that each candidate batch starts while
   stationary.
3. Show at least one selected local trajectory containing multiple primitives
   executes continuously to its local endpoint.
4. Show a second frontier batch is built after endpoint stop and map refresh.
5. Verify planner diagnostics contain global, local and total timing for every
   candidate request.
6. Run the complete 300 m exploration and accept completion only for
   `COMPLETED_NO_REACHABLE_FRONTIER`; report, but do not threshold, coverage.

## Non-Goals

- No requirement to drive to the exact frontier candidate coordinate.
- No stopping between motion primitives in one local trajectory.
- No concurrent candidate planning while the vehicle is moving.
- No new DWA, TEB, MPC, kinodynamic lattice or moving-start connector.
- No reduction in candidate planning calls.
- No coverage-ratio completion threshold.
- No direct exploration publisher on `/Car/T5/Car_Cmd_Vel`.
- No timestamp, map-version, covariance or input-freshness admission checks.

## Risks and Mitigations

- **Stationary odometry noise:** keep thresholds configurable and collect target
  platform evidence before considering an explicit stopped-state deadband.
- **Cancel/odometry delay:** retain the active plan identity until stationary
  confirmation and never infer stop from publication of cancel alone.
- **Exploration oscillation:** rebuild candidates after every local segment but
  retain existing information-gain, heading, revisit and failure-memory scoring.
- **Premature completion:** require one complete current-epoch candidate batch
  planned from stationary state before declaring no reachable frontier.
- **Long planning pauses:** preserve all candidate calls as requested and expose
  stop-wait separately so planning computation and orchestration delay can be
  evaluated independently.
