# Local Wheel Planning Efficiency and Unknown-Clearance Design

## Status

Approved in chat on 2026-08-24. The user explicitly excluded reducing the
number of exploration-candidate planning calls from this scope. This document
records the approved planner-core and diagnostics design for review before an
implementation plan is written.

This design is a focused amendment to
`2026-08-24-planner-core-incremental-optimization-design.md`. It does not
replace that design's request timing, primitive-based search, exact terminal
connector, rolling-goal, or cache decisions. The request-scoped SE(2) frame is
already present in the current source and is verification-only here.

## Goal

Raise the wheeled local planner's first-certified-path success rate within the
existing three-second request deadline by removing false clearance caused by
unknown cells and by eliminating provably unnecessary exact sweep work.

The change must retain the supplied 0.0-to-1.0 local occupancy contract, where
`NaN` means unknown, `0.0` means free and `1.0` means occupied. It must retain
the existing 0.2 m local map resolution, vehicle footprint, motion primitives,
minimum obstacle clearance, wheel support, slope, roughness, local relief,
underbody, curvature, dynamics, cancellation and deadline checks.

## Measured Problem

The fixed Jazzy exploration smoke trace contained eleven local-planning calls:
eight reached the approximately three-second deadline, two returned certified
paths in approximately 230 ms and 352 ms, and one returned `NO_PATH` in
approximately 49 ms. Global planning consumed approximately 57 ms in total.
The dominant failure is therefore local-search work that does not finish by the
deadline, not global search time.

The current local implementation amplifies that work in two ways:

1. `BuildLocalTerrainProjection()` marks non-finite occupancy as a hazard and
   builds obstacle clearance from that hazard mask. The wheel graph also builds
   its hazard row index from `free_with_height == 0`, combining measured
   obstacles, unknown occupancy and missing elevation into one clearance
   source.
2. The wheel graph classifies every finite elevation value other than exactly
   `0.0F` as complex terrain. A level surface at a non-zero odometry height
   therefore cannot use the existing flat-terrain fast path.

The exact evaluator scans a wide clearance neighborhood for every sweep sample
because minimum clearance participates in edge cost and label guidance. Dense
unknown cells in a 64 m local map therefore create both false clearance walls
and repeated polygon-to-cell distance work near the 10 m, 90-degree observed
sector.

## Chosen Architecture

Keep one exact certified Hybrid A* wheel planner and split local-map semantics
inside its immutable terrain projection:

```text
occupancy/elevation input
  -> hard support feasibility
  -> measured occupied-source mask
  -> occupied-only clearance transform
  -> wheel broad phase
  -> exact occupied clearance + exact support/terrain certification
```

Unknown cells remain individually non-traversable. They do not consume the
vehicle's configured clearance around their boundary. Known occupied cells
remain both non-traversable and clearance sources.

No approximate path is publishable. Every returned edge retains a complete
certificate for footprint support, known-obstacle clearance, terrain and
dynamics.

## Local Terrain Projection Contract

`LocalTerrainProjection` retains its current `free_with_height` vector to avoid
unnecessary call-site churn and adds one explicit occupied-source vector:

```cpp
struct LocalTerrainProjection final {
  std::shared_ptr<const MapSnapshot> map;
  std::vector<std::uint8_t> free_with_height;
  std::vector<std::uint8_t> occupied;
  std::vector<float> clearance_m;
  std::vector<float> narrow_band_distance_m;
  std::vector<float> slope_rad;
  std::vector<float> roughness_m;
};
```

The fields have these exact meanings:

- `free_with_height[index] == 1` only when occupancy is finite, lies in
  `[0.0, occupancy_threshold)`, and elevation is finite;
- `occupied[index] == 1` only when occupancy is finite, lies in `[0.0, 1.0]`,
  and is greater than or equal to `occupancy_threshold`;
- non-finite or out-of-range occupancy and non-finite elevation remain hard
  infeasible but never set `occupied`;
- `clearance_m` is the existing exact cell-area transform of `occupied`, not of
  `!free_with_height`;
- `narrow_band_distance_m` preserves the current search-refinement behavior by
  measuring distance to occupancy cells that are not known free, including
  unknown occupancy; it is used only to select wide versus narrow state-key
  resolution and is never a collision, clearance or cost certificate;
- when there is no measured occupied cell, every clearance value is positive
  infinity even though unknown cells may exist.

This mirrors the already implemented global projection distinction between
hard feasibility and measured obstacle inflation while preserving the native
local float occupancy representation.

## Wheel Broad-Phase Contract

The wheel graph replaces its combined hazard index with measured-occupied
indices:

- `hazards_by_row_` becomes `occupied_by_row_`;
- `hazard_integral_` becomes `occupied_integral_`;
- clearance and polygon-distance rejection consult only these occupied
  structures;
- `IsNarrow()` consults `narrow_band_distance_m`, so removing unknown-cell
  inflation does not silently coarsen the existing search lattice near an
  occupancy frontier;
- the existing `free_with_height` checks remain authoritative for whether the
  actual footprint is supported by known free terrain.

Bounds and clearance are separated:

1. The unexpanded transformed footprint must remain within the closed physical
   extent of the local raster, using the existing geometry tolerance
   symmetrically on all four sides. Exact contact with the physical extent is
   allowed; crossing it by more than tolerance remains invalid. Cell lookup
   continues to use the map's existing half-open index domain.
2. The known-obstacle lookup window expands by `minimum_clearance_m` but is
   clipped to the map. The map boundary itself is not treated as an occupied
   cell and does not receive clearance inflation.
3. A broad occupied-cell rejection may use the existing footprint inset disk
   plus minimum clearance.
4. Unknown or invalid terrain may be rejected cheaply only when direct
   footprint intersection is proven. No broad unknown-cell clearance radius is
   allowed.

Broad-phase acceptance remains provisional and cannot emit a path certificate.

## Exact Sweep Contract

For each interpolated primitive pose, the exact evaluator performs two
independent checks:

1. Scan only `occupied_by_row_` in the clearance window, compute exact distance
   from the oriented footprint polygon to each measured occupied cell area,
   and reject collision or distance below `minimum_clearance_m`.
2. Enumerate map cells intersected by the true footprint polygon and reject any
   cell for which `free_with_height == 0`, elevation/slope/roughness is invalid,
   slope exceeds capability, local relief exceeds capability, underbody
   clearance fails, or neighboring elevation discontinuity exceeds the
   existing continuous-step bound.

The existing strict boundary remains authoritative: known-obstacle clearance
equal to the configured minimum is accepted within geometry tolerance, while a
smaller value is rejected. Unknown occupancy or elevation is not accepted
merely because its distance from the vehicle centre is large; all cells used by
the footprint and by bilinear wheel-support samples must remain known and
finite.

The first check enforces distance from measured obstacles. The second prevents
the vehicle from entering unknown or unsupported space. Their order may be
chosen for runtime efficiency but their returned validity must be identical.

## Elevation-Offset-Invariant Fast Path

Absolute elevation is not a measure of terrain complexity. Remove only
`elevations[index] != 0.0F` from the complex-terrain classification.

The initial change does not add a tolerance for non-zero slope or roughness.
The fast-path proof domain is the union of the rotated footprint and all four
wheel contact points, expanded by the one-cell halo required by bilinear
elevation sampling. The fast path is allowed only when every cell in that proof
domain is hard feasible, elevation is finite, and the existing derived slope
and roughness values are exactly the flat values already required by the fast
path. Any non-zero or non-finite slope/roughness, or any wheel contact outside
the proven domain, continues through full wheel-support and terrain
certification.

Adding a constant vertical offset to every finite elevation must preserve:

- planning status and reason code;
- selected goal;
- XY/yaw trajectory and motion modes;
- normalized cost components;
- obstacle and terrain validity.

Trajectory Z values may shift by the same constant. No other output may change.

## Far-Clearance Exact-Work Elision

The existing broad phase produces a `clearance_proven` bit but the exact phase
currently ignores it. The bit may skip occupied-cell polygon-distance work only
when it proves the exact evaluator would find no occupied cell inside its
entire clearance scan window.

For every sweep sample, the proof threshold is conservative with respect to:

```text
footprint circumscribed radius
+ exact evaluator clearance scan margin
+ maximum centre displacement within one raster cell
```

The final term uses the half-cell diagonal required by the current
centre-to-occupied-cell-area clearance representation. The proof must hold for
every interpolated sample. If it holds, the evaluator assigns the same
`narrow_threshold` clearance value that the existing no-nearby-hazard path
assigns and continues all footprint support, terrain and dynamics stages.

This optimization must not cap, approximate or otherwise change a clearance
value that the current exact scan would observe inside its window. If an
equivalence test cannot prove the same edge validity, cost components and
label ordering, exact occupied-cell scanning remains mandatory.

## Diagnostics and Evidence

The current `WheelPlanResult` already contains most work counters, but
`PlanLocalDefault()` forwards only expanded states and cost. Extend internal
planner result types so every solved, `NO_PATH`, timeout and cancellation result
created after wheel-graph construction carries an optional wheel diagnostics
record. Preserve that record through both the normal `Planner::Plan()` path and
the rolling surface path, which currently reconstructs planning results by
hand. The record includes:

- `edge_validation_evaluations`;
- `edge_validation_cache_hits`;
- `broad_phase_rejects`;
- `full_certifications`;
- `full_invalidations`;
- `sweep_cell_checks`;
- `quantized_state_count`;
- `maximum_active_labels_per_key`;
- `expanded_states`.

It also preserves the already available ARA invocation count, state-reuse and
endpoint-alias counts, narrow-resolution flag, finest XY resolution, maximum
yaw bins, returned-certificate confirmations, mode-switch/reverse edge counts,
preferred-candidate counters and wheel cost components/scales.

Add rejection counters for at least:

- direct unknown/unsupported footprint intersection;
- measured-obstacle collision or clearance;
- slope/roughness;
- local relief/underbody;
- dynamics or primitive-shape rejection;
- deadline/cancellation interruption.

Publish numeric values as `wheel_*` key/value fields on the existing
`/Car/T4/planning/diagnostics` topic. Add `wheel_metrics_available`; it is true
only when `PlanWheel()` constructed a graph and supplied its record. Non-wheel
requests and failures before wheel-graph construction publish false and omit
the conditional wheel fields rather than inventing zero work. Keep
`PlanMotion.action`, `PlannerDiagnostics.msg` and
`PlannerDiagnostics.warning_codes` unchanged. Continue recording global,
local-goal, local-search, certification and total elapsed time with the steady
request clock; no input timestamp, map-version or freshness admission check is
introduced.

Failure results must not collapse performed work to zero. Tests and operator
documentation that currently assert an exact diagnostic key count must be
changed to assert the required common-key subset and the conditional wheel-key
set.

## Relationship to Request-Scoped SE(2)

`PlanningLatticeFrame` is already present in the current wheel source and is
constructed from the request start. Quantization and canonical key operations
already use its forward and inverse transforms. Commit `32903e8` is an ancestor
of the current branch and the arbitrary-translation, arbitrary-SE(2) and rigid
transform regression tests are already present.

This change does not modify that frame, yaw bins, primitive endpoints, goal
connector or public interface. The existing request-scoped SE(2) plan remains
responsible only for completing its focused regression, production build and
ROS demonstration evidence. Its unchecked task boxes are not evidence that the
frame must be implemented again.

## Parallel Implementation Boundaries

After the implementation plan is approved, two lanes may start in parallel:

- **Diagnostics lane:** internal metric propagation and ROS diagnostic
  key/value tests. It must not alter planning decisions.
- **Local semantics lane:** projection masks, occupied-only clearance and the
  associated unit tests.

After the local semantics lane lands, the wheel-evaluation lane may change the
occupied row index, boundary handling, elevation-offset fast path and
far-clearance proof. This dependency prevents simultaneous incompatible edits
to `anytime_wheel_planner.cpp`.

The final integration and Jazzy benchmark lane starts only after both prior
lanes pass their focused tests. Exploration candidate generation, candidate
yaw ordering and candidate retry policy are outside all lanes.

## Test-First Acceptance

### Projection Semantics

1. A local row `[free, unknown, free]` keeps the unknown cell hard infeasible
   while every obstacle-clearance value is positive infinity when no occupied
   cell is present; its narrow-band distance still records the occupancy
   frontier for search refinement.
2. Replacing the unknown cell with threshold-occupied produces exact cell-area
   clearance values and zero clearance at the obstacle.
3. An out-of-range occupancy value and a finite-free cell with `NaN` elevation
   are hard infeasible but do not become occupied clearance sources.
4. Unknown occupancy retains the current narrow-resolution selection without
   imposing minimum obstacle clearance on adjacent known-free cells.
5. Cancellation and deadline checkpoints remain effective during projection
   and the distance transform.

### Wheel Safety

1. A path whose footprint or bilinear wheel-support domain overlaps unknown
   occupancy or `NaN` elevation is rejected.
2. A path whose footprint remains on known free cells may pass adjacent to an
   unknown cell without an artificial minimum-clearance rejection.
3. A footprint fully inside the map or exactly touching its closed physical
   extent may pass adjacent to the map boundary; crossing it by more than the
   geometry tolerance is rejected on all four sides.
4. Existing occupied-cell collision, exact 0.2 m clearance boundary, corridor,
   slope, roughness, relief, underbody and dynamics regressions retain their
   behavior.
5. All returned edges retain exact certificates and returned-edge certificate
   confirmations equal the returned path edge count.

### Fast Paths

1. Constant-elevation maps at `0.0 m`, `10.0 m` and `-3.0 m` produce equivalent
   XY/yaw plans and costs, with only trajectory Z shifted.
2. A non-zero slope, rough patch, positive relief and elevation discontinuity
   each force the full terrain path and preserve the existing rejection.
3. Far-clearance proof and forced-exact evaluation produce identical edge
   validity, cost components, minimum-clearance value used by the planner and
   deterministic label ordering.
4. Near-obstacle edges never use the far-clearance skip.

### Diagnostics and Runtime

1. Solved, `NO_PATH` and timeout fixtures expose non-zero work counters when
   work occurred; cancellation preserves work completed before interruption.
   Pre-graph and non-wheel paths publish `wheel_metrics_available=false` and no
   conditional wheel values.
2. Core result tests cover normal and rolling propagation. ROS
   diagnostic-array tests verify the common-key subset and conditional wheel
   keys without changing `PlanMotion.action` or `PlannerDiagnostics.msg`.
3. All stable wheel, global, legged and hopper regressions pass in the Jazzy
   test build. The existing long-range baseline is classified separately if it
   still has a previously recorded deterministic timeout.
4. A fixed corpus reproducing the eleven smoke candidate requests is run
   before and after the change. The after run must retain both formerly solved
   requests, introduce no new safety-oracle acceptance, and reduce local
   deadline results from eight to no more than two.
5. The 300 m, global-1 m/local-0.2 m exploration run uses the same map seed,
   platform capability and 20x simulation setting. Coverage is recorded but is
   not a completion threshold; completion still requires no reachable frontier.

The numeric timeout target is an acceptance target, not permission to alter
the three-second deadline. If it is missed, diagnostics determine the next
bounded optimization; safety thresholds remain unchanged.

## Files Expected to Change

- `ros2_ws/src/lunar_pure_planner_core/src/shared/local_terrain_projection.hpp`
- `ros2_ws/src/lunar_pure_planner_core/src/shared/local_terrain_projection.cpp`
- `ros2_ws/src/lunar_pure_planner_core/test/local_terrain_projection_test.cpp`
- `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.hpp`
- `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`
- internal planning-result/diagnostics types used by `planner.cpp`
- `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- `ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp`
- corresponding core and ROS diagnostic tests
- planner documentation describing local unknown and measured-obstacle
  semantics

## Non-Goals

- Reducing exploration candidate positions, yaw variants or planner calls.
- Changing frontier scoring, retry policy or exploration completion logic.
- Changing `PlanMotion.action`, ROS topics, TF, odometry or map wire formats.
- Adding timestamp, covariance, map-version, freshness or observation-age
  admission checks.
- Changing the global planner, controller or vehicle motion primitives.
- Lowering map resolution, shrinking the footprint or reducing minimum
  clearance.
- Treating unknown occupancy or missing elevation as drivable.
- Replacing exact wheel certification with centre-point collision checking.
- Re-implementing the already present request-scoped SE(2) frame.

## Risks and Mitigations

- **Unknown adjacency becomes less conservative:** only adjacency changes;
  actual footprint overlap remains fail-closed and has dedicated tests.
- **Map-edge behavior changes unintentionally:** separate raw-footprint bounds
  from occupied-clearance windows and test inside-versus-crossing cases.
- **Fast path hides height hazards:** remove only the absolute elevation-zero
  condition initially; retain all derived slope, roughness, relief and support
  gates.
- **Clearance proof changes ranking:** require forced-exact differential tests;
  retain exact scanning if the fallback value is not identical.
- **Metrics inflate public API:** propagate them only through internal structs
  and diagnostic key/value fields; keep action messages unchanged.
- **Parallel edit conflicts:** diagnostics and projection semantics may run in
  parallel; serialize edits to the wheel evaluator behind projection semantics.
