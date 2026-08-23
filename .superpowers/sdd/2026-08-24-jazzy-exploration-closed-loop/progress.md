# SDD ledger — plan: docs/superpowers/plans/2026-08-24-jazzy-exploration-closed-loop.md

Start HEAD: 5ac588b

## Preflight scan

| Scope | Producer / consumer check | Finding |
| --- | --- | --- |
| Simulation/controller plans -> Task 1 | Task 3 maps/pose plus planner Action readiness gate the coordinator | Consistent; this plan starts only after dependencies finish |
| Task 1 -> Task 3 | Coordinator executable and START Topic are composed by launch | Consistent |
| Simulation plan -> Task 2 | Odometry and `sim_elapsed` feed recorder/HUD | Consistent |
| Task 2 -> Task 3 | Recorder/HUD executables and planned path are composed/displayed | Consistent |
| Task 3 -> Task 4 | Installed launch and exact Topics are consumed by smoke test | Consistent |
| Task 3/4 -> Task 5 | Same launch and external artifacts are used for full-run acceptance | Consistent |
| Tasks 1/2/3 | New package CMake is shared sequentially | Requires serial task dispatch; no interface conflict |
| Task 1 internal | Readiness test matches maps, pose, TF, status and Action contract | Consistent |
| Task 2 internal | Terminal result, timings, coverage and persistence tests match implementation | Consistent |
| Task 3 internal | Car Topics, capacities, controller override, optional RViz match tests | Consistent |
| Task 4 internal | Smoke proves progress but explicitly cannot claim completion | Consistent |
| Task 5 internal | Script succeeds only on exact terminal state/reason and artifact agreement | Consistent |

Ruling: The operator script derives a candidate ROS domain from process/time data and rejects it if `ros2 node list --no-daemon` is nonempty before launch; absolute proof of global DDS nonuse is unavailable — this minimizes local collision without changing ROS interfaces — if wrong, an unrelated process could contaminate the test graph.

Task 1 review at `a9b798d`: changes requested. Important startup race: initial status endpoint matching does not guarantee the explorer's volatile task subscription is matched; a one-shot transient-local START may be lost. Add task-subscriber connectivity before MarkStarted/publish and a real node-level one-shot test.

Task 1: complete (commits 5ac588b..1f46e04, independent re-review clean). Final evidence includes real node/QoS/Action one-shot delivery and 52/52 tests passing.

Task 2 review at `1d4e1b5`: changes requested. Important: terminal status currently finalizes before the exploration node's later diagnostics; actual `IDLE+CANCELED` is not wired; exact-success can be spoofed through the public finalize reason; installed CMake export references nlohmann without dependency closure.

Task 2 re-review at `a8baf9a`: all functional findings addressed; changes requested solely because the test RAII helper uses forbidden recursive `std::filesystem::remove_all`. Replace it with explicit known-file removal and non-recursive empty-directory removal with visible failure.

Task 2: complete (commits 1f46e04..7a83dab, final independent re-review clean). Final evidence: 67/67 tests, fresh installed consumer, exact success semantics, bounded terminal diagnostic drain, and non-recursive cleanup.

Task 3 review at `2d0412e`: changes requested. Important: START can precede controller MotionReference subscription; coordinator must require exactly one T5 publisher plus an observed periodic controller command. Output validation must reject paths inside any Git repo/worktree by inspecting the output path's ancestors, not only known current roots.

Task 3: complete (commits 7a83dab..685d484, independent re-review clean). Final evidence: 9/9 launch behavior tests, real controller-readiness node test, and exact composed parameter inspection.

Task 4 review at `eed2743`: changes requested. Important teardown gap: if the launch leader exits after SIGINT while a same-PGID child remains, `process.poll()` skips SIGTERM and the final ROS graph check can miss a non-ROS child. Teardown must inspect the exact PGID/recorded child PIDs and prove all are absent.

Task 4 teardown fix implemented in working tree: process identity is `(pid,start_time,pgid,session)`, all exact same-session/group members are accumulated from `/proc`, SIGTERM escalation depends on residual group members rather than leader state, and live evidence records every member. Synthetic leader-exits/child-survives regression and fresh live smoke pass; pending commit and independent re-review.

Task 4 re-review at `336d45b`: normal teardown addressed, but capture/readline exceptions occurred before cleanup `finally`. Follow-up wraps every post-Popen operation immediately, uses the exact `start_new_session` PID=PGID=SID fallback, compares full process identity, and adds a leader-exits-before-capture regression. Fresh static/synthetic and live evidence pass; pending final short review.

Task 4 final line-level review at `3cdee13`: fallback construction still performed `/proc` I/O before entering `try`. The fallback is now a pure no-I/O PID=PGID=SID descriptor; all observation/capture work occurs inside cleanup protection. Static/synthetic `11/11` and fresh live pass; pending final review.

Task 4: complete in working tree pending commit. The opt-in live smoke uses a
locked candidate domain, localhost-only/no-daemon preflight, exact process-group
cleanup, and external artifacts. Seven bounded RED/diagnostic iterations led to
minimum startup stabilization, strict derived observation envelopes, a
forward-only startup prior, and an observable unlocked `PLANNING` snapshot;
planner/selector algorithms remain unchanged. Fresh final run
`run-ea44a6ccafaa485dbb6d3e792ee4a38a` passed in 3.52 s with states
`[0,1,2,3,4]`, one task, 385 reference points, nonzero T5, 0.200004 m motion,
coverage `0 -> 0.000475624`, and real global/local `PLAN_FOUND` timing. Domain
133 was empty after teardown. Detailed failures, rulings, regression evidence,
and limitations are in `task-4-report.md`.
