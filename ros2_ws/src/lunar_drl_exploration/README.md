# DRL exploration contracts and offline terrain

The deployed policy reads native policy-map exports. Offline terrain construction
is explicitly separate: importing `contracts`, `config`, `geometry` or `sensor`
does not construct a Scene or CoverageReference. No learner, simulator, controller
or runtime integration is implemented by this package's terrain layer.

## Native ownership and interfaces

- `load_platform_config(path=None)` loads the installed
  `lunar_incremental_navigation_ros/config/wheel.yaml`, or this checkout's
  `config/wheel.yaml` when no installed package is available. An explicit path
  works without ROS. `capability` exposes that file's full mapping;
  `actor_context` retains the existing eight-scalar API.
- `Scene(seed, family, extent_m, resolution_m, platform)` supports `moon` and
  `cave`, 20–1000 m nominal extent. Generator version 4 draws a seeded convex
  task quadrilateral inside that extent. `scene_id` hashes generator version, seed,
  geometry settings and capability. `task` is canonical `TaskSpec`; `task_polygon` is its float32
  polygon. `height_tile(x0, y0, width, height)` evaluates bounded analytic geometry.
  Raster extent adds 12 m context around the nominal extent, rounded up to cells;
  this finite provided context is not a physical wall or an infinite-world claim.
- `TerrainGrid.from_scene(scene)` produces compact `heights[H,W]` float32,
  `stats[H,W,4]` float32, `intrinsic[H,W]` uint8 B and `navigation[H,W]` uint8 M.
  State values are native UNKNOWN=0, FREE=1, BLOCKED=2. Stats order is slope_rad,
  relief_m, positive_rise_m, neighborhood_complete (0/1). `from_heights` also
  accepts effective stats, with NaN center heights representing missing evidence.
- Terrain statistics come from public native `MeasureLocalTerrain`, are rounded
  to float32, and only then classified with `PlatformElevationEvaluator::Evaluate`
  and `FineCellEvaluator::Evaluate`. There is no Python terrain classifier.
  Fine caches are discarded per native 256-cell tile, with its native halo.
  The hard footprint is the circumscribed radius of the canonical wheel polygon;
  minimum_clearance_m is a soft preference, not an extra hard inflation radius.
- `scene.initial_pose(terrain)` samples uniformly from the main native M FREE
  component using a deterministic start RNG stream. Moon uses the largest native
  FREE component; cave uses the component anchored to its structural main room.
  Four-connected labels exactly partition native no-corner-cut components: every
  allowed diagonal has a FREE cardinal intermediate. Uniform selection uses row
  counts and one selected row, avoiding a full array of every stance coordinate.
  Polygon and start RNG streams do not consume the original terrain/yaw stream.
  No view/FOV/known-map support filter is used; original random yaw is unchanged.
  Cave starts are on the main floor, not elevated wall plateaus. Generated cave
  loops include an external route, narrow spurs and a disconnected chamber. Free islands remain
  intrinsically free even when not reachable from the chosen start.
- `SensorModel.observe(terrain, pose, sensor)` returns immutable sparse
  `VisibleMeasurements`: row/column indices, row-major linear indices, center
  heights and the four stats. `.mask` materializes a boolean grid only on request.
  Hidden neighbors are never emitted. Native scratch buffers are range bounded.
  The raw hit `.mask` may include finite centers with native B UNKNOWN (partial
  support); these remain publishable raw measurements but do not count as
  classified effective coverage.
- Deployment-safe `sensor.visible_cells(intrinsic, origin, resolution_m, pose,
  sensor)` returns row/column indices using measured B only. `geometry` supplies
  `world_to_cell`, `cell_center` and concave, boundary-inclusive `polygon_mask`.
  Observation origin uses floor((world-origin)/resolution) then the stance-cell
  center; yaw remains actual continuous yaw plus mounting yaw. Nonzero virtual
  sensor translation is rejected until its reference mapping is supported.
- `CoverageReference.build(terrain, start, task, sensor)` computes the native M
  connected component including task-external cells, with no diagonal corner
  cuts, then the exact union of center-ray visibility from every legal stance
  over all attainable headings. Full-angle union is a fixed denominator, not a
  360° instantaneous episode observation. Supercover rays include both corner
  cells; the first-hit obstacle is visible, cells behind it are not. The effective
  coverage set contains only finite native-classified centers (B FREE or BLOCKED);
  B UNKNOWN never enters its mask, area denominator or packed intersection.
- The reference uses a square-dilation candidate superset, exact native range
  checks, first-visible-source early exit and interior-blocker rejection. It does
  not use a ring-only approximation. `packed_mask` and `reachable_bits` use
  **little bit order and row-major linear index y*width+x**. `pack`, `unpack`,
  `linear_index` and `covered_area` share this convention. `coverage_ratio` is
  zero when the coverable area is zero. Policy failure never shrinks the set.

## Build and verify

Build the current branch's `lunar_incremental_navigation_core` first, source its
matching overlay, and build this package. The new CPython buffer-protocol module
is named `lunar_drl_terrain_native`; it links that core and uses no old extension.
All build/install/log and test artifacts should remain outside the repository.

```bash
source /opt/ros/jazzy/setup.bash
source /home/kai/.cache/lunar-drl-redesign/jazzy/install/setup.bash
colcon --log-base /home/kai/.cache/lunar-drl-redesign/jazzy/log build \
  --base-paths ros2_ws/src --packages-select lunar_drl_exploration \
  --build-base /home/kai/.cache/lunar-drl-redesign/jazzy/build \
  --install-base /home/kai/.cache/lunar-drl-redesign/jazzy/install
source /home/kai/.cache/lunar-drl-redesign/jazzy/install/setup.bash
python3 -m pytest ros2_ws/src/lunar_drl_exploration/test
```

Tests include an independent continuous-segment/closed-cell-square exhaustive
visibility oracle; random maps; exact corners; fractional origins; ring-only
failure; task-external standpoints; visible nonstandable cells; float32 threshold
neighbors; and an independently compiled persistent producer/fine-builder
roundtrip comparing B, M and observed separately. That roundtrip first applies a
translated bootstrap patch, so it compares world centers rather than assuming
that the producer and scene integer indices have the same origin.

Historical **generator version 3** local x86_64 Jazzy-core initialization at
0.2 m, seed 20260915, 10 m sensor range, separate sequential processes (2026-09-15).
This table predates version 4 polygon/start randomization and the classified-center
coverage correction; it is not a version 4 performance claim:

| Task extent | Moon total / peak RSS | Cave total / peak RSS |
| --- | --- | --- |
| 40 m | 0.83 s / 66.5 MiB | 1.14 s / 66.8 MiB |
| 80 m | 1.85 s / 76.9 MiB | 2.49 s / 77.2 MiB |
| 150 m | 4.60 s / 88.1 MiB | 7.17 s / 88.3 MiB |
| 300 m | 13.67 s / 127.5 MiB | 24.15 s / 127.4 MiB |
| 1000 m | 130.97 s / 720.8 MiB | 230.91 s / 721.3 MiB |

The 1 km raster is 5120×5120 including context. Its retained terrain arrays occupy
550 MiB and the two packed reference masks 6.25 MiB. Heights used to assemble tiles
are not retained by the reference. These timings are initialization evidence for
one seed per family, not training throughput or an admission gate. They precede
the NaN-padding bounds correction and version 4 review fixes. Version 4 main-region
selection additionally uses temporary compact component labels; its 1 km peak and
timing have not been remeasured. Cave occlusion search was the scaling bottleneck
in version 3. Version 4 passed the package's 52 behavioral tests, including repeated
seeded main-region starts, independent native connectivity checks, and exclusion of
boundary/interior-insufficient-support B UNKNOWN from coverage.
Humble, Orin, DDS, rosbag, closed-loop controller and vehicle evidence: `NOT_RUN`.

## Observed task graph and joint pose actions (Task 4)

`TaskAnalyzer(task, sensor).update(snapshot)` returns a frozen `TaskReport`.
Known area counts effective classified centers, including BLOCKED centers once;
M never supplies the observed predicate or an optical blocker. Potential motion
uses M != BLOCKED after removing native reachable R. Unknown task demands seed
potential components; shared native B rays add cross-component observation
witnesses, including nonstandable demands. The raster includes all measured
evidence plus task/range margin, and its outer boundary joins a single external
UNKNOWN component. An interface cannot recruit an unrelated branch by searching
through R. `frontier_cells[N,2]` and aligned `witnesses[N,2]` use **snapshot global
integer x,y cell indices**. Frontiers identify first pending center measurements,
including legitimate task-external transit interfaces. Missing classified support
at a native FREE task center remains pending. Each retained interface has a native
R observation witness; it is not an executable UNKNOWN stance.

`DecisionCore(task, sensor, config=None).observe(snapshot, velocity=(v,w))`
returns `(DecisionObservation, TaskReport)`. No ROS, torch, Scene or reference
module is imported by this observed path. A missing native start connection is
`available=False, reason_code='INPUT_UNAVAILABLE', exhausted=False`; callers must
gate execution on report availability. The observation still contains the actual
anchor and its eight yaw choices. Task 7 owns live cold-start support integration.

`GraphBuilder(GraphConfig()).build(snapshot, report, history, task, sensor, velocity)`
uses native FREE row-run rectangles, physical portals and inserted observation
witnesses. Sparse rectangle-local Euclidean trees remove empty-interior sampling
cycles while retaining narrow corridors, physical obstacle loops and all witnesses.
Edges are undirected `[E,2]` index pairs; edge lengths include compressed known
routes. There is no graph node crop. Actual start connections attach an exact
continuous anchor. Already-satisfied short portals are traversed before exposing
moving actions. Overfull local arrival regions use deterministic world-aligned
physical relay endpoints; no co-located dummy nodes or discarded unique branches
are used. Each action is an adjacent endpoint with one of eight world vehicle yaws
`k*pi/4`, including the actual anchor; at most 20 positions / 160 joint actions.
Executable frozen `goals` are **float64**, even though network positions/features
and action yaw metadata remain float32.

The 19 node features are `(world XY - actual XY)/10`, task membership, eight
visible-frontier center counts divided by the fixed **100 cells**, and eight actual
observation-direction bits. A single sparse geometric visibility relation supplies
all eight FOV counts; empty interiors do not cast full disks. The utility uses actual
SensorModel origin/range/FOV/mounting-yaw rules, not optimistic unknown area.
Polygon corners are relative XY/10; the existing eight-scalar platform/sensor
context is unchanged. Zero-utility transit remains selectable.

Call `core.record_observation(actual_pose, observed_cells)` after real sensor
measurements, or `DirectionHistory.record(pose, observed_cells, sensor)` directly.
The stored direction is the actual optical world heading, rounded to the nearest
of eight sectors. History matches executed world positions within the exported
native position tolerance (GraphConfig's 0.1 m fallback when unavailable), survives
node resampling and clears on an epoch change. Issuing a goal does not mark a visit.

`GraphBuilder.build_truth(terrain, reference)` is the explicit training-only entry
point. It produces immutable `PrivilegedScene`: `scene_id`, static `positions`,
`edges`, `edge_lengths`, `packed_reference`, `reference_shape`, and the CSR arrays
`reference_offsets[N+1]` / `reference_indices`. Reference cells are assigned to the
nearest static node; each CSR slice contains row-major linear indices usable for
packed observed-fraction pooling. Packed masks retain little bit order. The scene
owns these arrays and recursively frozen generator metadata. `TerrainGrid.from_scene`
retains version, seed, family, extent, resolution, scene ID and canonical capability
JSON for regeneration. Arbitrary `from_heights` terrain instead supplies its terrain
identity/shape/origin/resolution descriptor; it does not claim to recover missing
source heights. No truth data enters `DecisionObservation`.

Task 4 local x86_64 Jazzy-core graph growth, 2026-09-15, synthetic measured open
M/B/K FREE squares at 0.2 m with a 2 m pending task band outside each square and
10 m / 90 degree sensor. Native 256-cell tiles own independent arrays. Timings
include task analysis, sparse graph, features and frozen actions, excluding fixture
construction; numeric threads=1, separate sequential processes. All four boundary
interfaces are retained, and the graph remains non-exhausted.

| Known square | Nodes / edges | Frontier centers | Graph build | Frozen graph | Process peak RSS |
| --- | --- | --- | --- | --- | --- |
| 100 m | 1,999 / 1,998 | 2,000 | 0.118 s | 0.214 MiB | 64.0 MiB |
| 300 m | 5,999 / 5,998 | 6,000 | 0.486 s | 0.641 MiB | 133.7 MiB |
| 1000 m | 19,999 / 19,998 | 20,000 | 3.705 s | 2.137 MiB | 878.7 MiB |

The 1 km input tiles alone occupy 275 MiB; analysis still has raster-sized scratch.
These measurements demonstrate perimeter graph growth and retained witnesses,
not 2 Hz / 30x throughput at 1 km, terrain-generation performance, training, live
navigation, Humble, Orin, DDS or vehicle readiness. Those runtime layers remain
`NOT_RUN` for Task 4. Regressions also explore all 60 branches of an adversarial
large arrival-region fixture after off-center actual arrivals and revision rebuilds;
all graph degrees remain <=19. The separate normal 0.30 m arrival test proves
that unchanged actual poses do not repeatedly expose already-satisfied portals.
