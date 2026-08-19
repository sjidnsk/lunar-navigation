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
