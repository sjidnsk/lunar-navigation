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
  `cave`, 20–1000 m. `scene_id` hashes generator version, seed, geometry settings
  and capability. `task` is canonical `TaskSpec`; `task_polygon` is its float32
  polygon. `height_tile(x0, y0, width, height)` evaluates bounded analytic geometry.
  Raster extent adds 12 m context on each side of the task, rounded up to cells;
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
- `scene.initial_pose(terrain)` chooses native M FREE support near the generated
  main terrain. It neither alters elevations nor resamples yaw. Cave starts are
  on the main floor, not elevated wall plateaus. Generated cave loops include an
  external route, narrow spurs and a disconnected chamber. Free islands remain
  intrinsically free even when not reachable from the chosen start.
- `SensorModel.observe(terrain, pose, sensor)` returns immutable sparse
  `VisibleMeasurements`: row/column indices, row-major linear indices, center
  heights and the four stats. `.mask` materializes a boolean grid only on request.
  Hidden neighbors are never emitted. Native scratch buffers are range bounded.
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
  cells; the first-hit obstacle is visible, cells behind it are not.
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

Measured local x86_64 Jazzy-core initialization at 0.2 m, seed 20260915, 10 m
sensor range, separate sequential processes (2026-09-15):

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
the final NaN-padding bounds correction, which leaves these fully finite scene
classifications unchanged. Cave occlusion search remains the scaling bottleneck.
Humble, Orin, DDS, rosbag, closed-loop controller and vehicle evidence: `NOT_RUN`.
