# Luna operator commands

Run all operations through `./luna`; no manual ROS sourcing, colcon command,
or lifecycle request is required. Each command prints JSON. A nonzero exit
code with a `reason` or `reasons` field is an actionable refusal.

## Runtime configuration and environment

```bash
./luna init --profile ubuntu22-humble-amd64
./luna prepare --dry-run
./luna prepare --apply --yes
./luna doctor
./luna doctor --live
./luna config check
./luna build
```

`luna prepare --dry-run` only reports locked dependencies; it never writes or
invokes sudo. `luna prepare --apply --yes` is the only sudo-capable path.
Typical errors are `ROSDEP_NOT_INITIALIZED` or `DEPENDENCY_LOCK_DRIFT`; correct
the host/environment lock rather than bypassing it. `luna doctor --live` may
return `WAITING_FOR_EXTERNAL_INPUT`: the build is usable, but required map,
localization, TF, task, or feedback publishers are not connected yet.

`runtime.yaml` is the one supported configuration surface:

```yaml
profile: ubuntu22-humble-amd64
interfaces: {map_global: /environment/map_global, map_local: /environment/map_local, odometry: /localization/odometry, localization_status: /localization/status, tf: /tf, exploration_task: /mission/exploration_task, motion_feedback: /execution/motion_feedback, plan_motion: /plan_motion, diagnostics: /diagnostics, certified_route_markers: /planning/certified_route_markers, provisional_route_markers: /planning/provisional_route_markers}
capabilities: {platform_file: /opt/luna/capabilities/platform.yaml, observation_file: /opt/luna/capabilities/observation.yaml}
planner: {enable_nav2_adapter: false, snapshot_policy: {global_map_max_age: 1.0, local_map_max_age: 1.0, odometry_max_age: 1.0, localization_status_max_age: 1.0, tf_max_age: 1.0, max_pairwise_skew: 1.0}}
policy: {mode: fallback, model_id: null}
extensions: {map_pipeline: false, path_tracking: false}
input_adapters: {mode: external_canonical, task3_config_file: null}
runtime: {log_level: INFO}
```

Select one reviewed platform baseline on the deployment host:

```bash
sudo install -D -m 0644 deployment/config/wheel.yaml /opt/luna/capabilities/platform.yaml
sudo install -D -m 0644 deployment/config/legged.yaml /opt/luna/capabilities/platform.yaml
sudo install -D -m 0644 deployment/config/hopper.yaml /opt/luna/capabilities/platform.yaml
sudo install -D -m 0644 deployment/config/observation.yaml /opt/luna/capabilities/observation.yaml
```

Run only one command for the active platform: each overwrites the active
selection, so wheel, legged, and hopper cannot be active together. The second
selects the Quad48 `legged/legged-v1` parameter envelope; the third selects the
lunar `hopper/hopper-v1` single-hop envelope. Neither requires URDF/mesh, and
neither creates the separately owned observation capability file. The final
command installs the reviewed deployment observation baseline: 10 m maximum
range and 90 degree field of view.

## Lifecycle and logs

```bash
./luna start
./luna status
./luna logs
./luna stop
```

`status` reports the managed PID/lifecycle state. Fallback needs no model
backend. `policy.mode: onnx` or `tensorrt` is refused with
`POLICY_RUNTIME_UNBOUND` until an approved ROS policy adapter exists; a staged
model has `model_binding: staged_not_connected`.

## Task3 adapted start and one safe request

Keep the generic `runtime.yaml` in `external_canonical` mode unless the Task3
publishers, TF chain, and read-only SQLite evidence are available. On the Task3
host, copy the two templates, update only local paths/topics if needed, then
use the Task3 runtime configuration:

```bash
sudo install -D -m 0644 deployment/config/task3-adapters.default.yaml /etc/luna/task3-adapters.yaml
sudo install -D -m 0644 deployment/config/task3-adapted.runtime.yaml /etc/luna/task3-adapted.runtime.yaml
./luna config check --config /etc/luna/task3-adapted.runtime.yaml
./luna build --config /etc/luna/task3-adapted.runtime.yaml
./luna start --config /etc/luna/task3-adapted.runtime.yaml
```

The Task3 configuration directly consumes `/Car/T3/mapping/grid_map`,
`/Car/T3/semantic/current_pose`, and the `map -> odom -> base_link` TF chain.
It remains fallback-only; a non-fallback model remains refused with
`POLICY_RUNTIME_UNBOUND`.

To make one non-executing point request, substitute an approved short safe
point from the active Task3 mission:

```bash
ros2 run luna_t3_map_adapter luna_plan_smoke_client.py \
  --mission-id "$MISSION_ID" --mission-revision "$MISSION_REVISION" \
  --goal-id short-safe-point --x "$SAFE_X" --y "$SAFE_Y" --tolerance 0.2
```

The client only requests `/plan_motion` and prints its result. It never sends a
controller command. If it returns a `MotionReference`, the external controller
must use its `plan_id` and `segment_id` in matching `MotionExecutionFeedback`
messages on `/execution/motion_feedback`; invalid identity or sequence is
rejected by the planner.

## Optional WHEELED controller

Automatic chassis control is disabled by default. Keep this setting for map,
planning, and smoke-client checks:

```yaml
controller:
  wheeled:
    enabled: false
```

To enable it only after confirming that no `keyboard_teleop.py` process owns
the chassis command Topic, set `controller.wheeled.enabled: true` in a copied
runtime configuration, then restart with that file:

```bash
./luna config check --config /etc/luna/task3-adapted.runtime.yaml
./luna build --config /etc/luna/task3-adapted.runtime.yaml
./luna start --config /etc/luna/task3-adapted.runtime.yaml
```

The coordinator consumes `/mission/execution_goal`; the controller consumes
`/execution/wheeled_reference` and publishes only `geometry_msgs/msg/Twist` to
`/Car/T5/Car_Cmd_Vel`. It sends `MotionExecutionFeedback` on
`/execution/motion_feedback`. `STALE_INPUT`, `PATH_DEVIATION`,
`INVALID_REFERENCE`, and controller stop all command zero velocity before the
feedback. Source-level tests do not authorize vehicle movement; retain
`enabled: false` until the real chassis command direction and emergency stop
path have been validated.

## Model artifacts

```bash
./luna model install /absolute/path/model.tar.gz
./luna model activate demo-v4
./luna model status
./luna model rollback
```

A model package is exactly:

```text
model-manifest.json
policy.onnx
normalization.npz
```

Install validates the manifest, v4 observation contract, v2 action contract,
and SHA-256 hashes, then stages an immutable content-addressed copy under
`$LUNA_HOME/models/<model-id>/<model-sha256>/`. Activation atomically writes
`active-model.json`; rollback restores `previous-model.json`, or returns to
fallback if no previous model exists. Failed validation leaves the active
pointer unchanged.

## Optional extensions

```bash
./luna extension
./luna extension enable map_pipeline
./luna extension disable map_pipeline
```

Each entry reports `{enabled, package, status}` where status is
`not_installed`, `disabled`, or `enabled`. Enabling a package that is absent
returns `EXTENSION_PACKAGE_NOT_INSTALLED`; it does not manufacture a publisher
or controller.

## Make a source bundle

```bash
./luna bundle --target ubuntu22-humble-amd64 --output /absolute/output-directory
./luna bundle --target jetson-orin-r36 --output /absolute/output-directory
```

The bundle contains the current allowlisted source, one target profile, and
only `README.md` plus `COMMANDS.md`. Do native builds separately on amd64 and
Orin; a successful build or model probe on one is not evidence for the other.
