# Lunar runtime source package

This package is the deployable runtime for lunar exploration planning. It is
intended for a separate Ubuntu 22.04 + ROS 2 Humble amd64 host or a Jetson AGX
Orin R36 / JetPack 6 aarch64 host. Build and device evidence are native to the
target: an amd64 build is not evidence for Orin, and this package does not make
cross-compilation claims.

```text
ROS map / odometry / TF / mission -> lunar_planner_ros -> PlanMotion action
                                    -> diagnostics and route markers
```

The runtime operates in **fallback** mode by default. A validated model can be
staged with `luna model install`, but it is deliberately not connected to a ROS
policy adapter yet. Therefore a non-fallback `luna start` refuses with
`POLICY_RUNTIME_UNBOUND`; it never silently falls back while claiming that a
model is in use.

## Five-minute first run

```bash
./luna init --profile ubuntu22-humble-amd64
./luna prepare --dry-run
./luna prepare --apply --yes
./luna build
./luna start
```

For Orin, replace the profile with `jetson-orin-r36` and perform the same
sequence on the device. `prepare --dry-run` is read-only. Only
`prepare --apply --yes` may use sudo; it installs the target lock's ordinary
development dependencies. It never changes NVIDIA drivers, CUDA, TensorRT,
Jetson firmware, or flash state.

## External interfaces

All names are configured in `config/runtime.yaml`, but must remain absolute:

| Input/output | Default name | Owner / frame expectation |
|---|---|---|
| Global map | `/environment/map_global` | map provider, global map frame |
| Local map | `/environment/map_local` | perception, local planning frame |
| Odometry | `/localization/odometry` | localization, configured robot frame |
| Localization status | `/localization/status` | localization health publisher |
| TF | `/tf` | TF broadcaster, consistent map/robot transforms |
| Exploration task | `/mission/exploration_task` | mission supervisor, task ROI/frame |
| Motion feedback | `/execution/motion_feedback` | execution controller |
| Plan action | `/plan_motion` | planner server action endpoint |
| Diagnostics | `/diagnostics` | planner health output |
| Certified markers | `/planning/certified_route_markers` | planner visual output |
| Provisional markers | `/planning/provisional_route_markers` | planner visual output |

Safety projection, footprint, clearance, endpoint feasibility, path
certification, and input freshness checks stay enabled. They are runtime safety
constraints, not optional qualification gates. This package does not run formal
training preflight or closed-loop training evaluation.

The future extension points are named `map_pipeline` and `path_tracking`.
They remain disabled unless their declared ROS package is installed; enabling
one cannot create a missing map publisher or controller.

## Task3 direct-input mode

For Task3 integration, select `deployment/config/task3-adapted.runtime.yaml`.
This mode starts three project-owned adapters plus the planner and consumes the
following Task3-owned inputs without rewriting them:

| Input | Use |
|---|---|
| `/Car/T3/mapping/grid_map` | 0.2 m `odom` local map, converted to the canonical ten planner layers without resampling |
| `/Car/T3/semantic/current_pose` | direct `odom -> base_link` Odometry for the planner and localization-status check |
| `/tf` | Task3-owned `map -> odom -> base_link` transform chain |
| `/Car/T3/mapping/global_map_revision` | revision used for read-only global SQLite evidence lookup |

The local adapter reads `map -> odom` only to locate matching global evidence.
It never changes the local GridMap frame, orientation, resolution, dimensions,
or timestamp. A malformed or incomplete input is withheld rather than made
safe-looking.

`/plan_motion` returns a `MotionReference` when a route is available. The
external controller—not this runtime—executes it and publishes an identity
matched `MotionExecutionFeedback` on `/execution/motion_feedback`. It must
preserve the returned `plan_id`, `segment_id`, platform type, increasing
sequence, and current state. Wrong, stale, duplicate, or mismatched feedback is
rejected by the planner. There is no generic controller command topic in this
repository.

The repository ships three reviewed numeric platform capability documents:
`deployment/config/wheel.yaml`, `deployment/config/legged.yaml`, and
`deployment/config/hopper.yaml`. The latter two are the Quad48
`legged/legged-v1` and lunar `hopper/hopper-v1` parametric baselines; neither
requires URDF or mesh assets. Install exactly one as
`/opt/luna/capabilities/platform.yaml` before launch: only that selected file
becomes active. An observation capability remains independently required.
