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
  `cave`, 20–1000 m nominal extent. Generator version 5 draws a seeded convex
  task quadrilateral inside that extent. `scene_id` hashes generator version, seed,
  geometry settings and capability. `task` is canonical `TaskSpec`; `task_polygon` is its float32
  polygon. `height_tile(x0, y0, width, height)` evaluates bounded analytic geometry.
  Raster bounds contain all complete analytic features (including crater support
  at twice its radius) plus 12 m context, extending the original world lattice by
  whole cells. Geometry/task/start/yaw RNG streams retain their seeded rules;
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
  cuts, then the exact union of finite beam-prefix visibility from every legal stance
  over all attainable headings. Full-angle union is a fixed denominator, not a
  360° instantaneous episode observation. Every beam points to an integer center
  in nominal range; shorter collinear tips are redundant. It emits every traversed
  cell and the complete simultaneous first-hit group. At a grid corner both side
  contacts are emitted before diagonal entry; either side blocker stops the beam.
  FOV selects actual beam angles, never the effective hit-center bearing. The effective
  coverage set contains only finite native-classified centers (B FREE or BLOCKED);
  B UNKNOWN never enters its mask, area denominator or packed intersection.
- The reference uses a square-dilation candidate superset, exact native range
  checks, forward first-visible-source early exit and a conservative optical
  component prefilter (UNKNOWN transmits; first-hit blockers border the transmitting
  component). Exact beam queries still determine every retained target. It does
  not use a ring-only approximation. `packed_mask` and `reachable_bits` use
  **little bit order and row-major linear index y*width+x**. `pack`, `unpack`,
  `linear_index` and `covered_area` share this convention. `coverage_ratio` is
  zero when the coverable area is zero. Policy failure never shrinks the set.

The stable semantic identifier is `sensor.OBSERVATION_MODEL_VERSION =
"finite_center_tip_prefix_v1"`. Scene IDs/reference IDs and the retained generator
metadata bind this identifier; generator descriptors also contain `provided_bounds`.
Future checkpoint semantic configuration must bind both versions without adding an
Actor feature. Native visibility owns an immutable prefix/event trie and reverse
target-offset lookup. Its process LRU retains at most eight templates / 128 MiB;
`native.visibility_cache_info()` reports retained bytes, beams and prefix nodes.
The key includes nominal range and capped map dimensions. Only impossible prefix
displacements are trimmed for small provided rectangles; **all nominal beam
angles remain**. Oversized templates fail explicitly with a resource error, never
reduce beam density or change range. Queries release the GIL and retain shared
ownership across eviction. No sparse query allocates a whole-world observation.

`target_visibility` queries full heading support; `source_visibility` evaluates
FORWARD from each candidate source to one demand. `directional_visibility` returns
[N,8] deduplicated target membership for eight optical headings.
`first_pending_cells(..., range_cells)` returns the first actual UNKNOWN hit on a
successful same-model demand beam. It is not a separate center-line raycaster.

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
coverage correction; it is not a version 4/5 or beam-prefix performance claim:

| Task extent | Moon total / peak RSS | Cave total / peak RSS |
| --- | --- | --- |
| 40 m | 0.83 s / 66.5 MiB | 1.14 s / 66.8 MiB |
| 80 m | 1.85 s / 76.9 MiB | 2.49 s / 77.2 MiB |
| 150 m | 4.60 s / 88.1 MiB | 7.17 s / 88.3 MiB |
| 300 m | 13.67 s / 127.5 MiB | 24.15 s / 127.4 MiB |
| 1000 m | 130.97 s / 720.8 MiB | 230.91 s / 721.3 MiB |

That historical 1 km raster was 5120×5120 including context. Its retained terrain arrays occupy
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

## Current finite-beam closure validation (2026-09-15)

Generator **v5**, observation model **finite_center_tip_prefix_v1**, seed20260915,
0.2m cells, 10m sensor. Sequential independent processes, OpenBLAS/OMP threads1;
peak memory is Linux VmHWM. Times include scene, native terrain, initial stance
and exact reference; the reference column isolates its cost. No stance sampling
or beam-density reduction was used.

| Family | Extent | Raster H×W | Native terrain | Exact reference | Total init | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| moon | 100m | 661×659 | 2.088s | 0.016s | 2.118s | 70.6MiB |
| moon | 300m | 1706×1636 | 13.552s | 0.074s | 13.650s | 139.4MiB |
| moon | 1000m | 5750×5306 | 147.826s | 0.715s | 148.705s | 1037.5MiB |
| cave | 100m | 620×663 | 1.841s | 0.019s | 1.870s | 70.1MiB |
| cave | 300m | 1620×1698 | 12.616s | 0.218s | 12.846s | 127.7MiB |
| cave | 1000m | 5120×5320 | 126.141s | 1.952s | 128.133s | 802.7MiB |

Final-library single warm 10m/90° SensorModel calls: moon0.207ms / cave0.073ms.
The eight-heading sparse utility kernel over7845 native disk targets took
2.00ms / 5.95ms respectively; every target/heading matched eight actual sensor
observations, including the ±π FOV boundary. These are kernel measurements at
one pose, not full graph timings or 8-environment training throughput. Default
radius50 geometry cache:6,276,472bytes,116,937 prefix nodes,4,776 distinct beams.

Independent closed-square beam oracles cover all native R stances, internal and
external viewpoints, nonstandable visible targets, first-hit corner groups,
finite FOV/mount yaw, fixed-pose self-closure and producer roundtrip. The original
7×7 failure is retained. Complete native40m cave measurements round-trip through
the actual persistent producer with B/M/observed parity and task exhaustion.
Final160/240/300m ideal native cave probes have measuredR==truthR,
knownarea==AE1240.60/2347.68/3818.68m² and zero residual interfaces. TaskAnalyzer
phases are0.080/0.179/0.232s; whole proofs5.10/10.83/16.80s. The previous v4
160m case retained43 interfaces; its evidence is preserved in the work report.

TaskAnalyzer first tests actual UNKNOWN interfaces adjacent to known nonblocking
cells, including interfaces outside task and UNKNOWN R sources. A demand hit
must follow a visible first interface within2×range, so a conservative square
support mask can reject distant negative demands. Every remaining demand still
needs its own exact forward beam witness; a visible interface is never blanket
visibility proof. This replaces119.94s of300m negative-demand analysis with0.232s
without changing any legal stance or completion condition. Sparse interface
filtering is skipped when it would enumerate no fewer cells than the demands.
Withholding one actual wall hit in the160m cave preserves nativeR and correctly
leaves one frontier (known1240.56m² < AE1240.60m²), but analysis still takes5.43s.
This mixed-state negative-query cost remains a measured limitation; closed-world
fast completion does not establish per-decision training throughput.

Current synthetic measured-open graph100/300/1000m builds:
0.185/0.714/4.771s, peak76.2/160.2/958.5MiB. Node/edge/frontier counts remain
1,999/1,998/2,000;5,999/5,998/6,000;19,999/19,998/20,000. All mandatory interfaces
are retained; these graph timings include the new directional utility.
Humble/Orin, DDS, rosbag, navigation/controller closed loop and training remain
`NOT_RUN`. Arrival-region precision versus distinct sensing-cell execution is a
separate Task7 responsibility, not proved by this observation/reference closure.

## Observed task graph and joint pose actions (Task 4)

`TaskAnalyzer(task, sensor).update(snapshot)` returns a frozen `TaskReport`.
Known area counts effective classified centers, including BLOCKED centers once;
M never supplies the observed predicate or an optical blocker. Potential motion
uses M != BLOCKED after removing native reachable R. Unknown task demands seed
potential components; shared native B rays add cross-component observation
witnesses, including nonstandable demands. Potential source queries only consider
components with an actual R entry after unifying exterior component labels;
disconnected non-entry components cannot contribute an executable interface.
This retains every piece of a unified exterior component with any R entry.
Direct R visibility is independently
checked for every nearby pending demand, even when its potential movement
component has no entry from R. A sparse native first-source search preserves this
relation without enumerating all Python source/target pairs. A shared once-per-map
optical support prefilter rejects impossible direct observations through fully
closed B barriers, with exact forward beam tests for all survivors. The raster includes all measured
evidence plus task/range margin, and its outer boundary joins a single external
UNKNOWN component. An interface cannot recruit an unrelated branch by searching
through R. `frontier_cells[N,2]` and aligned `witnesses[N,2]` use **snapshot global
integer x,y cell indices**. Frontiers identify first pending center measurements,
including legitimate task-external transit interfaces. The first UNKNOWN hit is
selected on a proven successful forward demand beam; every simultaneous contact
is a measurement under the shared model, with no detached center-ray recheck. Adjacent
interfaces beyond sensor range are not presented as observable witnesses. Missing classified support
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
JSON, observation model version and complete provided bounds for regeneration. Arbitrary `from_heights` terrain instead supplies its terrain
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
| 100 m | 1,999 / 1,998 | 2,000 | 0.175 s | 0.214 MiB | 69.7 MiB |
| 300 m | 5,999 / 5,998 | 6,000 | 0.665 s | 0.641 MiB | 147.8 MiB |
| 1000 m | 19,999 / 19,998 | 20,000 | 4.405 s | 2.137 MiB | 879.0 MiB |

These historical center-ray measurements were rerun after the Task 4 direct-witness / center-hit review
fixes and supersede the initial 3.705 s 1 km timing. They predate the shared finite beam-prefix refinement. The 1 km input tiles alone
occupy 275 MiB; analysis still has raster-sized scratch.
These measurements demonstrate perimeter graph growth and retained witnesses,
not 2 Hz / 30x throughput at 1 km, terrain-generation performance, training, live
navigation, Humble, Orin, DDS or vehicle readiness. Those runtime layers remain
`NOT_RUN` for Task 4. Regressions also explore all 60 branches of an adversarial
large arrival-region fixture after off-center actual arrivals and revision rebuilds;
all graph degrees remain <=19. The separate normal 0.30 m arrival test proves
that unchanged actual poses do not repeatedly expose already-satisfied portals.

## Sparse Actor and asymmetric discrete SAC

The training-only `model`, `batch` and `sac` modules require PyTorch. `config`
remains Torch-free, including `ModelConfig` and `LearningConfig`; importing
`DecisionCore` in an observed ROS worker does not load Torch or privileged code.
The default model has 6 residual sparse attention layers, width 128 and 8 heads.
Each measured graph is read once by its current-node query. Ordered polygon-edge
features preserve boundary adjacency; platform context and the candidate position,
world heading, directional utility and visit bit enter one shared action head.
There is no recurrent state, dense node-square attention or graph truncation.

```python
import torch
from lunar_drl_exploration.config import ModelConfig, LearningConfig
from lunar_drl_exploration.model import Actor
from lunar_drl_exploration.sac import SACLearner

actor = Actor(ModelConfig()).eval()  # CPU collector, batch of 1..8 observations
outputs = actor(observations)       # list[PolicyOutput], in input order
# Each output has 1-D logits, probs, log_probs in frozen action order.
# For deterministic inference, stack per-output argmax and transfer once.
choices = torch.stack([out.probs.argmax() for out in outputs]).cpu().tolist()

learner = SACLearner(ModelConfig(), LearningConfig(), device="cuda")
metrics = learner.update(transitions, scenes)  # exactly 64 Transition records
publication = learner.actor_state()
actor.load_state_dict(publication["state_dict"])
version = publication["version"]
checkpoint = learner.state_dict()
learner.load_state_dict(checkpoint)
```

`scenes` is a mapping `scene_id -> PrivilegedScene`. `PrivilegedState.observed`
is a row-major **packed uint8, little-bit-order** effective observation mask with
`ceil(prod(scene.reference_shape)/8)` bytes, aligned to `packed_reference`.
It is not a byte-per-cell mask. Each Q call pools bits indexed by the scene's CSR
`reference_indices` into current observed fractions; empty owner slices have
fraction zero. Relative static node coordinates, log reference-owner counts and
these fractions encode the privileged graph. Each Critic independently encodes
both the measured observation and privileged graph and returns a tuple of 1-D Q
vectors aligned with the Actor's exact valid action lists. Actor/Q1/Q2/target1/
target2 own independent parameter storage. No scene-ID embedding or persistent
GPU scene cache is used; ephemeral packing retains no scene beyond its call.

`LearningConfig` defaults: effective batch 64, microbatch 16, learning rate 1e-5,
gamma 1, Polyak 0.005, initial alpha 5e-5, maximum alpha 1e-4, target entropy
`0.01 * log(number_of_valid_actions)`. Actual microbatch sample counts weight
all means. Every complete update performs one step per Critic, one Actor step,
one temperature step and one target soft update. Exact expectations enumerate
all valid actions. Only `terminated` suppresses bootstrapping; truncated records
must contain the real successor before reset. True terminal successors may have
empty action lists and are skipped entirely. The trainable log-alpha is clamped
after its step. Metrics cross to CPU once per complete update. Training uses deterministic
per-layer activation recomputation; inference and detached Q/target passes skip
it. Each Critic backpropagates and releases its activations before the other
Critic runs; optimizer steps still occur once, after all microbatches.

`actor_state()` returns owned CPU tensor copies with keys `schema` (task_graph_v1),
`model_config` (primitive dict), `version` (completed update count), `state_dict`.
Repeated publication without an update has the same version. The scheduler owns
the 16-update publication cadence; publication itself does not advance it.
`state_dict()` returns an owned Torch-serializable record with schema
`sparse_graph_sac_v1`, primitive model/learning config dicts, `updates`, trainable
`log_alpha`, all five model states and all four optimizer states. Load rejects old
schemas and changed model or experience-objective settings: effective batch 64,
gamma, target entropy factor and maximum alpha remain checked. Continuation may
change microbatch size, learning rate and Polyak coefficient. Loading preserves
all saved Adam moments and step counters, then applies the currently configured
learning rate to all four optimizers; subsequent target updates use the current
Polyak value. `initial_alpha` only initializes fresh training: restore always
loads the saved actual log-alpha and temperature optimizer history, even if the
constructor's initial alpha differs. Models and optimizer tensors restore onto
the constructed learner's device through their standard load APIs. Scheduler/replay/
scene ownership, collector versions, RNG,
curriculum and update credits remain external Task 6 checkpoint responsibilities.
No PlatformConfig mappingproxy is embedded in these records.

Task 5 numerical benchmark, 2026-09-15: Torch 2.8.0+cu128, NVIDIA GeForce RTX
5070 Ti Laptop GPU. Synthetic sparse chains match the listed node/edge sizes;
these are numerical model benchmarks, not generated-terrain navigation results.
Every observation has 20 positions x 8 headings = 160 valid joint actions.
CPU inference uses 2 threads, CUDA learner 4 threads, sequential benchmarks after
one warmup. Times include CPU packing/transfers; medians of 5 inference runs and
3 complete updates, explicit CUDA synchronization only around benchmark timing.

| Measured and truth graph (each) | CPU batch 1 | CPU batch 8 | CUDA batch 64 / micro 16 | CUDA peak allocated |
| --- | --- | --- | --- | --- |
| 128 nodes, 127 undirected edges | 3.085 ms | 16.696 ms | 0.2582 s | 0.150 GiB |
| 1,999 nodes, 1,998 undirected edges | 23.424 ms | 312.383 ms | 3.3376 s | 0.773 GiB |
| 5,999 nodes, 5,998 undirected edges | 91.386 ms | 1,213.055 ms | 10.6520 s | 2.097 GiB |

The initial implementation retained all layer activations: 1,999-node updates
peaked at 8.507 GiB, and 5,999 nodes failed inside the first Critic's truth encoder
at 11.10 GiB allocated (48 MiB allocation failed). Deterministic per-layer
recomputation and separate Critic backward passes resolved that measured OOM.
The table reports the final implementation; numerical tests compare its outputs,
gradients and complete updates against direct execution.

Packing adds reverse directions and self loops. The larger benchmark retains all
graph nodes and both dynamic Critic inputs; its memory/time does not establish
30x collection throughput. Native closed-loop exploration, full 3D, Humble,
Orin, DDS and vehicle model deployment remain `NOT_RUN` for this task.

## Bounded replay, scheduling and continuation (Task 6)

`TrainingConfig()` contains the operational defaults: 8 environments, target RTF
30, integration 0.05 simulation seconds, observations 2 Hz, 0.05 m navigation
arrival tolerance, batch64/micro16, warmup1024, update ratio0.25, maximum pending
credit32, Actor publication every16 completed updates, and worker/collector/learner
numeric threads1/2/4 with zero DataLoader subprocesses. Curriculum decision budgets
are512/2048/8192 and extents40–80/80–150/100–300 m. Sensor starts at10 m/90°.
The configuration itself does not start environments or change native parameters;
Task7/8 must apply the shared arrival value to navigation and the public controller.
`config_record(config)` and `training_config_from_record(record)` provide explicit
JSON/spawn records, including the canonical platform capability. They support
mappingproxy without `dataclasses.asdict`, Torch, Scene or native terrain imports.

Default replay residency is8 GiB, saved replay payload at most2 GiB, total output
at most20 GiB including the old checkpoint and atomic temporary, metrics8 MiB and
save interval1800 monotonic seconds. Output is
`training-output/drl-exploration-redesign/`, relative to the project root. These
are selected operating budgets, not claims about currently available hardware.
The default system available-memory reserve is2 GiB; an optional explicit
`owned_pss_limit_bytes` may also be configured. No hardware or map admission gate
is added. The runtime must measure actual RTF rather than skipping physical steps.

`ReplayBuffer(max_bytes).add(transition, scenes)` accepts complete `Transition`
records and a mapping of needed IDs to immutable `PrivilegedScene`. The registry
retains only the static numeric graph, packed reference and frozen descriptor,
never `TerrainGrid`, high-resolution terrain, ROS objects or worker maps. Its ID
is `CoverageReference.reference_id`, binding terrain/task/start/sensor semantics;
`generator_descriptor['terrain_id']` preserves original terrain identity. Different
references over one terrain coexist and evict independently. Existing IDs reuse
the registry's canonical scene without a whole-scene comparison on each admission.

`bytes_used` charges unique reachable allocations (including shared array backing)
plus conservative unique ledger-row overhead. Every retained transition and static
scene separately charges its ownership-key tuple and newly allocated ID integers;
entry tuples, scene-ID sets, index/refcount integers and actual retained dictionary
tables are charged independently of shared payloads. `bytes_used` stays O(1), and
admission/release walks only the affected roots; no whole-replay scan is added.
When only one new entry remains, an already checked fresh ownership table can
replace oversized retained table capacity. Empty baseline containers are excluded.
`buffer_bytes` reports unique
numeric bytes-backed storage alone. It is a replay ownership estimate, distinct
from live process PSS and allocator residency. `scene_refcounts` counts retained
transitions referring to each scene, once per transition even if both states use
it. Oldest entries evict until the byte budget fits; the last reference releases
the scene. A single record and its scene larger than the budget fail *before*
eviction, so the caller still owns the unacknowledged finished message.
`sample(64, numpy_generator)` samples with replacement in O(batch) dictionary
lookups, without copying the complete replay index. `scenes` is a read-only view
for the synchronous learner call; do not mutate replay during that call.

`dumps_transport(value)` / `loads_transport(bytes)` are the explicit trusted-local
IPC boundary for observations, transitions, static scenes, primitive containers
and NumPy RNG records. The wire contains tagged primitive records and immutable
bytes, not native-object/NumPy/dataclass pickle reducers. Decoding runs contract
validation and retains immutable array views, shared backing and repeated object
references within the payload. Actual spawn/Pipe and checkpoint tests verify this;
ordinary NumPy pickle flags alone are insufficient. Mappingproxy is supported.
Worker `MapSnapshot`/tiles stay local and are deliberately unsupported transport
payloads. Send `config_record` for configuration and send only numeric static
scene records when a learner first needs them. Collector Torch RNG tensors can
use Torch's normal process transport; they are a separate collector-state field.

`replay.snapshot(max_bytes)` returns a bounded byte payload containing complete
retained transitions and every referenced scene. A conservative residency subset
is selected first, followed by an actual serialized-length check. It may retain
less than the byte ceiling, and drops oversized-for-snapshot records rather than
splitting them. `snapshot(max_bytes, require_sample=True)` instead raises
`ResourceLimitError` if bounded selection cannot retain one complete record. The
error reports the configured byte budget in its reason and observed sample count0
versus required minimum1. `TrainingState.capture` automatically requires a sample
whenever `schedule.can_update` is true. It fails before publishing a checkpoint;
Task8 must report the snapshot budget failure and preserve the previous good
resume, without clearing credits or bypassing backpressure. A complete sample
suffices for64 draws with replacement;64 distinct saved records are not required.
Fresh or pre-update empty states that have no whole earned update remain allowed.
`ReplayBuffer.from_snapshot(payload, max_bytes)` reconstructs
immutable contracts and refcounts. It never recreates hidden terrain. Reducing the
runtime replay budget still must fit each restored transition plus its scene;
an individual oversized record gives an explicit error.

The learner owns admission counters and update credit. The exact rational scheduler
has no warmup debt and never clamps earned credit:

```python
schedule = UpdateSchedule(warmup=1024, ratio=.25, max_credit=32)
if schedule.can_dispatch():       # includes the proposed action and all reservations
    schedule.reserve_dispatch()  # BEFORE dispatch, one reservation per future result
    # Task8 dispatches one action to an available worker slot.

# Task8 keeps each complete result until learner acknowledgement.
replay.add(finished_transition, received_scenes)
schedule.collected(reserved=True) # AFTER successful admission, exactly once
# Now acknowledge the result. A retry must not re-admit an acknowledged message.

# Infrastructure failure with no constructable successor only:
schedule.cancel_dispatch()       # releases the reservation, creates no experience

if schedule.can_update:
    with boundary.update():      # boundary = UpdateBoundary()
        metrics = learner.update(replay.sample(64, replay_rng), replay.scenes)
        schedule.updated()       # AFTER the whole successful optimizer/target update
```

`can_dispatch(inflight_count=None, count=1)` also accepts an external count including
all outstanding reservations; it rejects counts smaller than its own reservations.
`reserve_dispatch(count)` supports batches, and `cancel_dispatch(count)` is only for
unconstructable work. `collected()` without `reserved=True` is available for direct
synchronous admission tests and applies the same backpressure. Credit is a Fraction;
convert to float when writing metrics. Already reserved completions always fit the
credit cap, including while new dispatch is paused. Save/restore preserves collected
and update counters and fractional credit; unfinished reservations reset to zero
because restored environments start new episodes. Publication at completed update
multiples of16 is Task8's responsibility, using the existing `actor_state()` API.

`TrainingState.capture` requires keyword arguments `learner`, `config`, `replay`,
`schedule`, `boundary`, `replay_rng`, `curriculum_rng`, `curriculum`,
`collector_state` and `counters`. Learner and scheduler completed-update counts
must agree and replay admission cannot be ahead of that learner-owned collection
counter. Capture/save/load also reject earned-update credit paired with an empty
saved replay. Restore validates that combination and reconstructs replay/scenes
before changing learner weights, optimizers or RNG. A16-byte `LDRLRP2` envelope stores the sample count; save/load inspect only
that fixed header. Restore decodes the payload once and checks actual count,
contracts and scene references before learner mutation. The header is written into
the same serialization stream, and restore uses a memoryview to skip it: neither
step concatenates/slices a second full replay byte copy. No decoded-replay cache is
retained. The earlier, unreleased headerless Task6 snapshot format is rejected with
an explicit replay-envelope schema error. Curriculum/counter
payloads are decoded before learner mutation as well. `curriculum` is a primitive mapping: Task8 must include its stage, episode
progress and scene-seed counter. `collector_state` must include its CPU Actor
sampling `policy_rng` Torch tensor and `actor_version`, plus next-episode/scene-seed
counters owned by the collector. `counters` carries episode totals, accumulated
simulation/wall time and other runtime totals. All supplied state is copied or
serialized at capture; no in-progress environment state is included.

The snapshot includes all five networks, four optimizers, actual log-alpha, update
count, schedule, limited replay/scenes, Python/global NumPy/CPU Torch/CUDA RNG,
independent replay/curriculum generator states, collector RNG/version, curriculum,
counters and explicit config/schema semantics. CUDA RNG is restored for continuation
on CUDA with matching device count; GPU-to-CPU restore retains CPU state and does
not claim the same cross-device random stream. Exact portable model/Adam restoration
is supported by the learner. Collector RNG is returned for the collector to install.

```python
manager = CheckpointManager(config.output_dir, config.save_interval_s,
    total_max_bytes=config.total_output_max_bytes,
    snapshot_max_bytes=config.snapshot_max_bytes)
manager.write_run(config, owned_pids=owned_pids, code_revision=current_sha,
                  extra={"seed": config.seed, "completed_updates": learner.updates})
# Main thread only: signal.signal(signal.SIGINT, manager.request_sigint)
# Resource stop: manager.request_save(str(resource_error), stop=True)
# At the completed-update / collector-acknowledgement barrier:
if manager.save_requested:
    manager.save(TrainingState.capture(...))  # explicit keyword arguments above

state = manager.load(config)                  # trusted local resume.pt only
restored = state.restore(learner, config)       # learner constructed with requested config
replay, schedule = restored.replay, restored.schedule
replay_rng, curriculum_rng = restored.replay_rng, restored.curriculum_rng
# Install restored.collector_state RNG/version and restored.curriculum/counters.
# Start NEW environment episodes; never join unfinished motion to new observations.
```

Task8's barrier pauses dispatch and synchronizes collector policy RNG, versions,
scene-seed/curriculum counters and the acknowledgement watermark with learner
admission. Finished messages stay queued until admission and ACK; capture need not
wait for episode completion. The runtime serializes replay mutation, updates and
output writes under this single owner. A signal handler only sets save/stop flags;
normal SIGINT or observed resource shutdown waits for the full update boundary.
If *any exception interrupts a partial optimizer step*, `UpdateBoundary` is poisoned
and capture is forbidden. Preserve the previous good resume and report failure;
do not save changed weights alongside an old update count or silently reset the
guard. Restart from the prior complete checkpoint unless an actual full rollback
is implemented. Task6 does not implement the asynchronous collector or runtime loop.

`CheckpointManager.save` writes, fsyncs, atomically replaces one `resume.pt`, then
fsyncs its directory. Every physical temporary write checks simultaneous old+temp
output bytes and observed initial disk free space. A failed write/rename removes
only that attempt's temporary and preserves the old checkpoint. A hard process
kill can leave an orphan temporary; it remains counted on the next attempt and
is not silently deleted. The manager uses monotonic interval time, and failed
saves do not advance it. `stop_requested` remains set after a successful save so
Task8 can exit. No historical checkpoints or unevaluated `best.pt` are written.
An optional externally evaluated best Actor, if later added, counts in the same
output budget. There is no default video, bag or map image output.

Full-training `load(config)` compares model, effective batch, discount, entropy
objective/cap, reward, observation model/origin/effective-center visibility,
sensor, platform, physical integration/observation/arrival settings, schedule and
curriculum semantics. It rejects old schemas and names incompatible fields.
Microbatch size, learning rate, Polyak, initial-alpha initialization, operational
paths/save interval/RTF/threads/output budgets can change; saved Adam history and
actual temperature remain authoritative. A changed credit cap must still fit saved
pending credit. Actor-only loading uses the existing Actor shape/schema contract;
it does **not** call the full-replay semantic gate, so frozen evaluations may change
range/FOV and recompute real graph utility/context. Forward tests at5/15 m and
60/120/180° prove interface acceptance, not policy generalization.

`MetricsWriter(path, max_bytes, total_max_bytes=...)` maintains one bounded JSONL
file containing complete recent finite JSON records. It rejects oversized/NaN
records before changing the file and repairs incomplete trailing writes on open.
It compacts in place, so metrics are recoverable best-effort telemetry rather than
a crash-atomic checkpoint. Writes account for the shared output budget.
`ResourceMonitor(owned_pids, system_reserve_bytes=...,
owned_pss_limit_bytes=...).observe(output_dir)` reads **current**
`/proc/<pid>/smaps_rollup` PSS once per explicit owned PID and `/proc/meminfo`
MemAvailable, plus current disk/output bytes. `check(snapshot)` raises
`ResourceLimitError` with `reason`, actual `observed` and `limit`; it never sums RSS
or substitutes a lifetime peak. Update the explicit PID set when workers restart;
unreadable/missing PID observations are errors, not zero memory. `write_run` records
current CPU affinity/count, GPU total/free when nvidia-smi is available, RAM/disk,
code revision and chosen config. Missing GPU introspection is recorded, not an
admission prerequisite. Task8 supplies bounded throughput, credit, batch, Actor lag,
coverage/decision progress, RTF and execution-reason metrics and terminal display.
