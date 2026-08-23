# Request-Scoped SE(2) Wheel Lattice Design

## Status

Approved in chat on 2026-08-24. This document records the design for review before implementation.

The later approved
`2026-08-24-planner-core-incremental-optimization-design.md` supersedes this
document only where it requires canonical interior primitive endpoints and
fixed 64/128 yaw-bin counts. The request-scoped exact-start SE(2) frame and the
existing certified scaled-primitive terminal connector remain in force.

## Goal

Allow the wheeled local planner to start from any finite, in-bounds, terrain-supported and collision-free continuous `(x, y, yaw)` pose without requiring that pose to align with the local map origin, map cell boundary, map cell centre, or a globally fixed yaw bin.

The planner must retain its bounded canonical-state search, existing motion primitives, exact certified goal connector, terrain and footprint validation, deterministic behavior for identical inputs, and all existing ROS interfaces.

## Root Cause

The current wheel graph uses the map raster origin and resolution as the origin and phase of the vehicle state lattice. It then stores the exact continuous start pose under a quantized key derived from that fixed map lattice.

On the first expansion, a real motion primitive is applied to the exact start. Its physical endpoint is quantized again, replaced by a fixed-map canonical pose, and checked by `PrimitiveReachesTarget()` with a physical tolerance of `1e-6 m`. If the external start has a different translation or yaw phase from the fixed lattice, the primitive cannot physically reach the substituted canonical pose and every first edge can be rejected.

This is not a map-origin, cell-centre, QoS, TF, or obstacle-generation problem. It is a coordinate-coupling error between:

- the map raster, which samples occupancy and elevation; and
- the vehicle state lattice, which discretizes kinematically reachable poses.

## Chosen Architecture

Each `WheelSearchGraph` owns one immutable `PlanningLatticeFrame` constructed from the exact request start pose. The frame is private implementation state and has the same lifetime as one `PlanWheel()` search graph.

Let the exact start pose be `(p0, yaw0)`. World poses are converted into the planning lattice frame as:

```text
p_lattice = R(-yaw0) * (p_world - p0)
yaw_lattice = normalize(yaw_world - yaw0)
```

Canonical lattice poses are converted back as:

```text
p_world = p0 + R(yaw0) * [key_x * xy_resolution,
                          key_y * xy_resolution]
yaw_world = normalize(yaw0 + key_yaw * yaw_resolution)
```

The start therefore has key `(0, 0, 0, start_mode, narrow_flag)` for every valid translation and yaw. The graph never moves or snaps the start pose. The lattice frame remains fixed during the search; it is not recomputed per expanded node or as the vehicle moves.

## State Quantization

`Quantize()` will:

1. Transform the world position into `PlanningLatticeFrame` coordinates.
2. Select the existing wide or narrow resolution using `IsNarrow()` at the world pose.
3. Quantize local `x` and `y` using the selected resolution.
4. Quantize yaw relative to `yaw0`, using the existing 64 wide bins or 128 narrow bins.
5. Preserve the existing motion mode and narrow flag in `WheelStateKey`.

`CanonicalPose()` will perform the inverse rigid transform, sample elevation at the resulting world position, reconstruct world yaw by adding `yaw0`, and retain the existing narrow-band consistency check.

Wide and narrow lattices share the same request frame. Because narrow translation resolution is exactly half the wide resolution and narrow yaw bins are exactly twice the wide count, wide nodes remain a subset of the narrow lattice.

## Search and Motion Edges

Motion primitives remain platform-relative and are still applied to the world pose stored in each node. Only key quantization and canonical-pose reconstruction change.

The following behavior remains unchanged:

- `PrimitiveReachesTarget()` must prove that the selected primitive reaches the canonical target within the existing physical tolerance.
- `MotionKindMatches()` must reject invented lateral, reverse, spin, or direction changes.
- `Evaluate()` must validate footprint sweep, occupancy, height support, slope, roughness, clearance, curvature, initial velocity, acceleration and braking.
- invalid edges are not interned as graph states;
- ARA* state limits, deadlines, cancellation, stable edge ordering, edge caching and deterministic tie-breaking remain unchanged.

A rigid SE(2) transform preserves primitive length, yaw change, curvature and relative motion. It therefore changes the search coordinate representation without changing vehicle kinematics or safety semantics.

## Start and Goal Handling

### Start

The exact request start is the origin of the planning lattice. There is no physical start connector, no synthetic straight segment and no coordinate snap. `ValidateStart()` continues to reject starts that are outside the map, unsupported, in collision, non-finite, or incompatible with the platform state.

Because the exact start is canonical in the request frame, a sub-resolution primitive whose endpoint quantizes back to the start key does not allocate a second representative state. This replaces the old fixed-map behavior in which an exact noncanonical start and a separate canonical pose could share the same quantized translation bucket.

“Any start pose” means any continuous pose that is structurally valid and physically feasible. It does not mean that a pose inside an obstacle, outside the map, unsupported by terrain, or dynamically impossible must produce a path.

### Goal

The existing certified goal connector remains unchanged. Canonical search states approach the goal, then `MatchingPrimitiveScale()`, `ScaledPrimitive()` and `AppendGoalTerminal()` may shorten a real platform primitive to enter the requested position and yaw tolerance or reach the exact endpoint.

If the capability cannot reach the requested goal using its supplied primitives, the planner continues to return `NO_PATH`; no direct geometric goal segment is introduced.

## Map and Frame Contracts

The map raster remains expressed in its existing frame and keeps its existing origin semantics. Map origin and resolution are used only by map lookup functions such as `PositionToCell()`, `SampleElevationBilinear()` and terrain validation.

No fields or flags are added to `GridMap`. In particular, there is no `cell_center` mode, phase flag, start offset, or ROS-specific map behavior.

The following public contracts remain unchanged:

- `PlanMotion.action`;
- `WheelPlanRequest` and `WheelPlanResult`;
- odometry, TF, OccupancyGrid and GridMap topics;
- map and odom frame conventions;
- RViz goal and path topics;
- platform capability YAML and motion primitive definitions.

## Global and Rolling Planning

Global occupancy search remains unchanged. It searches map cells and emits an exact start, map-cell waypoints and an exact goal as a coarse geometric route.

Each local `WheelSearchGraph` created during rolling planning owns one fixed frame derived from that local call's exact start. Graph states never cross graph boundaries; rolling continuation exchanges world poses and route progress, not `WheelStateKey` values. Re-anchoring a new local graph is therefore explicit and deterministic.

## Code Structure

Primary production change:

- `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
  - add a small private `PlanningLatticeFrame` value type;
  - construct it from the request start pose before the start key is quantized;
  - update `Quantize()` and `CanonicalPose()` to use its forward and inverse transforms.

Primary regression changes:

- `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`
  - add translated off-phase start coverage over multiple primitives;
  - add arbitrary-yaw start coverage over multiple primitives;
  - add a rigid-transform equivalence test;
  - retain existing exact goal, safety, narrow-lattice and state-limit tests.

Documentation will describe the internal request-scoped frame and explicitly state that no map or odometry alignment is required beyond valid frame transforms.

## Test-First Acceptance Criteria

Implementation must begin with failing tests that reproduce the current first-expansion `NO_PATH`. A test goal must be farther than one primitive reach so that the existing one-edge goal connector cannot mask the defect.

Required automated evidence:

1. A free-map plan succeeds from a translated start whose `x` and `y` phases differ from the map lattice.
2. A free-map plan succeeds from an arbitrary start yaw and travels for multiple primitives along that yaw.
3. The first trajectory pose equals the exact request start; no snap or inserted geometric segment is present.
4. A baseline request and its rigidly translated and rotated equivalent produce equivalent trajectories in their respective request frames.
5. Existing footprint, obstacle, slope, clearance, dynamics, narrow-lattice, goal-connector and bounded-state tests retain their behavior.
6. The wheel test target is run in full. The already observed 750 m random rolling timeout is reported separately unless a fresh unmodified baseline proves it is caused by this change.
7. ROS 2 Jazzy production build succeeds with `-DBUILD_TESTING=OFF`.
8. After restarting the demo with rebuilt binaries, the default Action result reports `planning_outcome: 0` and `has_reference: true`; RViz receives a non-empty path with valid frame IDs.

## Non-Goals

- No path smoothing, spline fitting or controller changes.
- No global-route algorithm change.
- No map, odometry or TF rewriting.
- No relaxation of physical tolerances or safety gates.
- No continuous-state graph for the full route.
- No synthetic start or goal teleportation.
- No attempt to guarantee a path from physically infeasible poses.

## Risks and Mitigations

- **Yaw wrapping:** use normalized relative yaw and modulo binning; test values on both sides of `-pi/pi`.
- **Wide/narrow transitions:** use one shared frame and the existing exact 2:1 resolution/bin ratios; retain narrow-key regression tests.
- **Rolling behavior:** run the full rolling tests and compare failures against the known unmodified baseline instead of treating a timeout as proof of a semantic regression.
- **Floating-point drift:** keep all canonical reconstruction derived directly from integer keys and the immutable frame, rather than repeatedly integrating poses.
- **Map boundaries after rotation:** canonical world poses continue through the existing map bounds and elevation checks; out-of-map nodes are rejected.
