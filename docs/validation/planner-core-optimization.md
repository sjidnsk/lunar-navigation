# Planner Core Incremental Optimization Validation

## Frozen Task 1 baseline — 2026-08-24

This is a reproducible baseline, not an optimization result.  The 750 m
scenario retains its 1000 x 1000 map, `0x5EED1234` seed, `(800, 500)` final
goal, 8 m rolling horizon, project wheel primitives, 20 s global deadlines,
and 1 s local deadline.

### Host and build

- Host: `hlr` (`Linux 7.0.0-30-generic x86_64`)
- Compiler: `/usr/bin/c++`, GCC 13.3.0
- Requested build type: `RelWithDebInfo`
- Build command: the Task 1 `/tmp/lunar-planner-opt/baseline/{log,build,install}`
  colcon command.
- Build result: `FAILED` outside this task's owned files.  The unrelated
  `test/anytime_hopper_planner_test.cpp:362` uses `std::ranges::any_of`, which
  is not available from its current includes.  The wheel target had already
  linked and was used for the two required baseline invocations.

### Recorder TDD evidence

- RED: `python3 -m pytest -q tests/test_measure_planner_performance.py`
  failed during collection with `ImportError: cannot import name
  'measure_planner_performance' from 'tools'`; the recorder did not exist.
- GREEN: the same command passed `6 passed in 0.01s` after implementation.
- Recorder contract: one `subprocess.run(..., check=False, text=True,
  capture_output=True)` per repetition; a nonzero child exit is an error even
  when the child emitted metrics.  The JSON report keeps every sample and
  summarizes each numeric field using nearest-rank p50/p95/max.

### Wheel baseline evidence

- Non-750 m invocation: `54` tests ran and `54` passed in `1713 ms`.
- 750 m invocation: one test ran and failed as the frozen baseline at rolling
  segment `21`; this is expected and intentionally not repaired here.
- Metrics line from that failed run:

```json
{"scenario":"750m","success":false,"failed_segment":21,"elapsed_ms":2769.5,"expanded_states":117435,"edge_evaluations":31787,"state_labels":15241,"sweep_cell_checks":31876140}
```

The fixture emits exactly one such line before the GoogleTest assertion.  Its
run record additionally retains global/local elapsed time, generated state
count, final pose, and rolling segment count for later optimization gates.

## Gate 1 — timing, global safety and sparse anchor ARA* — 2026-08-24

### Integrated build and contract evidence

- Jazzy `RelWithDebInfo`, `BUILD_TESTING=ON`: `lunar_planning_msgs`,
  `lunar_pure_planner_core` and `lunar_pure_planner_ros` all built successfully
  in `/tmp/lunar-planner-opt/gate1`.
- Frozen interface hashes remain unchanged:
  - `PlanMotion.action`: `36e8077bb384deb15ee77a3c0768d1c1ccb3ac52a7db765db90a21e79ffdd55c`
  - `PlannerDiagnostics.msg`: `379904270f20164112a60cfba95ceb719d115c0e6bf8a2386ad74171584ea900`
- Standalone static contracts pass `45/45`: action, six-interface ROS surface
  and launch configuration.  The exported repository was missing the matching
  pure-planner checker and retained parent-repository path assumptions; Gate 1
  restored the existing checker and corrected only those standalone paths.

### Timing and search behavior

- One immutable request clock now classifies a certified result as on-target
  below 1 s, late-success from 1 s to below 2 s, business failure from 2 s to
  below 3 s, and hard stop at 3 s.  Cancellation remains higher priority than
  the hard stop.  Rolling cold global and first local work share the same cycle
  clock; conversion/publication is inside that boundary.
- ARA* OPEN ordering is exactly `(anchor_key, path_cost, guidance_cost,
  state, sequence)`.  State records are sparse and touched-state based; tests
  that intentionally exercise later epsilon rounds explicitly disable the
  production first-solution policy.
- Exact occupied-cell-area clearance and strict `< inflation` blocking are
  active globally; unknown and out-of-range cells remain hazardous.

### Test evidence and explicit Wave 2 boundary

- Non-wheel core CTest: `16/16` passed, including shared ARA*, global search,
  legged, hopper and native boundary tests.
- Wheel unaffected set: `51/51` passed.  Four named cases were deliberately
  excluded from this Gate 1 count: trajectory-reconstruction fake-clock
  adaptation, occupied-wall detour, the greater-than-300-m case, and the frozen
  750-m rolling scenario.  The first three belong to the Task 7 Hybrid-state
  redesign; the 750-m production portal integration belongs to Task 10.
- ROS CTest: `12/12` passed after sourcing the Gate 1 overlay; the launch test
  itself passed `9/9`.  Earlier loader failures from an unsourced overlay are
  discarded as invalid environment evidence.

Gate 1 therefore validates the completed Wave 1 contracts while keeping the
known wheel search-capability failures visible and blocking final acceptance
until Gate 2/Task 10 makes them green.

## Gate 2 — rolling portals, Hybrid labels and linear projection — 2026-08-24

### Integrated build and compatibility evidence

- Formal integration HEAD `a45de62b8e37dac47a7a3cdb784baf1451e5f3a3`
  built `lunar_pure_planner_core` under ROS 2 Jazzy with
  `BUILD_TESTING=ON` and `RelWithDebInfo` in the isolated
  `/tmp/lunar-planner-opt/gate2-formal` tree.  The build completed in
  `15.44 s`; peak build RSS was `512676 KiB`.
- Non-wheel core CTest passed `17/17` in `6.25 s`.  The wheel binary excluding
  only the frozen 750 m cross-module scenario passed `71/71` in `2.57 s` with
  peak RSS `144772 KiB`.  Shared ARA* passed `25/25`.
- Standalone no-legacy, action, external-interface and launch contracts passed
  `48/48` in `0.26 s`.
- `PlanMotion.action`, `GoalRegion.msg`, `MotionReference.msg` and
  `PlannerDiagnostics.msg` are byte-identical to frozen baseline `45106ad`.
  Their SHA-256 prefixes are respectively `36e8077b`, `2952e5ca`,
  `b0472444` and `37990427`.
- Both staged and formal worktrees passed `git diff --check`; the production
  debug-marker scan was empty.  Formal verification made no source changes.

### Wheel search behavior and measured scope

- Wheel states now keep exact primitive endpoints behind request-scoped SE(2)
  quantized keys, with at most four active continuous labels per key.  Interior
  edges remain complete supplied capability primitives; only the existing
  exact terminal connector may be scaled and every returned edge is evaluated
  by the existing footprint, terrain, clearance and dynamics checks.
- The occupied-wall and greater-than-300-m regressions are GREEN.  In the
  latter, timing only the `PlanWheel()` call measured `393 ms`, leaving
  `607 ms` against its explicit 1 s assertion.  The complete test including a
  2500 x 1000 fixture takes about `1.25 s`.
- That 300 m result exercises an optional certified preferred-candidate path:
  a geometrically recognized straight/mirrored-arc S-shift is fully certified
  first and supplied to ARA* as an incumbent.  With first-solution policy
  enabled, ARA* therefore reports zero OPEN-state expansions.  This metric does
  not mean zero planning work and is not a universal Hybrid A* latency bound.
- The preferred builder is not tied to the fixture's coordinates or primitive
  IDs, but its applicability is deliberately limited to compatible aligned
  goals, obstacle geometry and primitive families.  Unsupported geometry,
  overflow, control interruption or any failed edge certificate falls back to
  the same continuous-label Hybrid A* search.  Separate regressions exercise
  real state expansion, label retention, terminal-label completeness,
  cancellation and the 3 s hard stop.

Gate 2 validates the local-search components and their integration.  It does
not yet claim end-to-end acceptance: the 750 m scenario remains excluded until
Task 10 replaces independent portal attempts with one production multi-goal
search and runs that scenario unchanged.
