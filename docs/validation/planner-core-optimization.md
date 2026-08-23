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
