# Planner Core Incremental Optimization Design

## Status

Approach A, incremental repair of the existing planner core, was approved in
chat on 2026-08-24. The parallel, worktree-isolated execution sequence is
recorded in
`docs/superpowers/plans/2026-08-24-planner-core-incremental-optimization.md`.
Neither document changes production source by itself.

## Decision

Improve the existing pure planner in place. Preserve its ROS interfaces,
platform capability contracts, certified motion primitives, request-scoped
SE(2) wheel frame, exact certified terminal connector, terrain checks and
fail-closed safety behavior.

The implementation will repair correctness before optimizing runtime:

1. establish the confirmed one-, two- and three-second request-time contract;
2. make global occupancy safety and local footprint safety agree;
3. restore correct ARA* ordering and termination semantics;
4. change the wheeled interior graph to primitive-based Hybrid A* states;
5. replace dense and repeated work with sparse containers, immutable
   revision-keyed caches and staged edge certification;
6. integrate progress reporting and rolling replanning without changing the
   `PlanMotion.action` wire contract.

This design rejects both a wholesale Nav2/OMPL replacement and a timeout-only
patch. A wholesale replacement would have to reproduce the project's elevation,
footprint, clearance, underbody, slope, roughness and dynamics certification.
A timeout-only patch leaves the known search-graph and safety-contract defects
unchanged.

## Scope and Existing Baseline

The active lunar-surface request is hierarchical:

```text
coherent input snapshot
  -> global occupancy projection and global route
  -> locally reachable rolling goal selection
  -> wheeled local search
  -> complete path certification and reference composition
  -> ROS result and reference publication
```

The shared ARA* implementation is also used by global, legged and hopper
planning. Changes to the shared engine therefore require all-platform
regression, even though the measured 750 m failure is wheeled.

The current wheel test binary has 55 tests: 54 pass and the unmodified 750 m
rolling test fails deterministically at segment 21. The failing one-second
local search expands about 9,151 states, evaluates about 24,089 edges and
performs about 30,117,568 footprint/terrain cell checks. Extending only that
call to ten seconds does not solve it: the graph reaches its 131,072-state cap
and performs about 190,917,836 cell checks. Moving only the diagnostic target
one metre away from the global bottleneck allows the unmodified remaining
pipeline to complete all 161 rolling segments. These observations define the
regression baseline; the diagnostic target modification is not a proposed fix.

## Requirements

### End-to-End Time Contract

One request clock covers snapshot construction, map projection, global search,
local-goal selection, local search, certification, composition and output.
Global and local stages do not receive independent additive time allowances.

The exact boundary semantics are:

```text
elapsed < 1 s       target met
1 s <= elapsed < 2 s  slow, still searching or successful
2 s <= elapsed < 3 s  performance SLA failed, continue searching
elapsed >= 3 s      hard timeout, stop the request
```

Crossing one or two seconds must not clear OPEN, CLOSED, INCONS, parents,
incumbents, map projections or edge-certification state. Crossing three
seconds must stop all remaining global/local/certification work promptly.

A fully certified reference completed before three seconds is a planning
success. If it completes at or after two seconds but before three seconds, the
ROS result remains `NEW_REFERENCE_AVAILABLE` and uses
`reason_code=PLAN_FOUND_LATE`; elapsed diagnostics record the SLA failure. If
no result has completed when the clock reaches three seconds, the request
returns the existing typed `RESOURCE_EXHAUSTED`/`TIMEOUT` result with no new
reference.

The hard time includes final certification and output construction. A path
whose search reaches the goal before three seconds but whose certified result
is not complete before three seconds is not committed as success.

### Termination Conditions

Before three seconds, the search may terminate only when:

- a complete certified reference is ready;
- OPEN is exhausted and the finite discretized graph proves `NO_PATH`;
- the active request is canceled or replaced;
- the input or numerical structure is invalid; or
- an actual system resource failure such as allocation failure occurs.

The one- and two-second performance milestones are not search termination
conditions.

### Safety and Compatibility

The optimization must not relax:

- footprint sweep and obstacle intersection;
- minimum clearance;
- wheel support and elevation sampling;
- maximum slope and roughness;
- local obstacle relief and underbody clearance;
- curvature, velocity, acceleration, braking or motion-mode constraints;
- cancellation and request-replacement behavior.

The public ROS topics, `PlanMotion.action`, map/odometry/TF contracts and
platform capability YAML remain compatible. Existing result fields and ROS
diagnostics carry the new latency classification; no message-schema change is
required.

## Request Timing Architecture

Introduce one immutable request timing policy owned by the top-level planning
call:

```text
target_milestone = started + 1 s
sla_milestone    = started + 2 s
hard_deadline    = started + 3 s
```

Only `hard_deadline` is passed as the stop deadline to global, local and shared
controlled work. The one- and two-second milestones drive throttled progress
and diagnostics. The top-level planner must no longer substitute hard-coded
one-second total, 150 ms global or 50 ms output deadlines for caller policy.

The first implementation keeps the search continuously alive inside the
request's planning worker. It does not introduce a cross-request or
cross-timeslice persistent search service. This is sufficient to preserve
OPEN/CLOSED state through the one- and two-second milestones, avoids session
serialization overhead, and retains existing stop-token cancellation. A
persistent incremental session is deferred until a separate requirement asks
for map-update repair or cooperative scheduling across requests.

Progress sampling must be throttled by elapsed time rather than emitted per
expanded state. The ROS server publishes existing `PlanMotion` feedback fields
for phase, elapsed time, expanded states and best cost. Result diagnostics
record at least:

- total request elapsed time;
- snapshot/projection, global, local-goal, local-search, certification and
  output elapsed times;
- target/SLA latency class;
- expanded/generated/reopened states;
- edge validation evaluations and cache hits;
- footprint/terrain sweep cell checks;
- whether global, local-terrain and goal-distance caches were hit.

## Global Safety Projection

### Root Cause

The current global projection computes Euclidean distance between raster cell
centres and compares it directly with the vehicle circumscribed radius plus
minimum clearance. For a one-metre raster, the centre of an adjacent free cell
is one metre from an occupied-cell centre but only 0.5 m from the occupied
cell's boundary. The project wheel's circumscribed radius plus required
clearance is about 0.9187 m, so the global mask can accept a centreline that the
local polygon certification must reject.

### Chosen Representation

Global hard feasibility must use distance to occupied cell area, not distance
to occupied cell centre. The implementation will construct a deterministic
Minkowski inflation stencil from:

```text
inflation = circumscribed_footprint_radius + minimum_clearance
```

For every relative raster offset, the stencil uses the exact minimum Euclidean
distance between the candidate cell centre and the occupied cell's closed
axis-aligned square. An offset is blocked when that distance is smaller than
the required inflation. Unknown and malformed occupancy remain individually
infeasible, but only valid occupied cells are inflation and clearance sources.

The global clearance value exposed to route costs must use the same
occupied-cell-area distance convention. Route preview and simplification
continue to be checked against the same inflated hard-feasibility mask, so
simplification cannot reintroduce a globally unsafe segment.

### Local Goal Portals

A rolling horizon is a progress distance, not a requirement to use one exact
interpolated centreline point. Local-goal selection will produce an ordered,
bounded set of portal candidates near the desired horizon:

1. candidates farther along the valid global corridor rank first;
2. candidates must lie inside the current local map and global inflated
   corridor;
3. greater global and local clearance breaks equal-progress ties;
4. candidates include longitudinal backoff positions and safe lateral corridor
   alternatives;
5. duplicate map cells and candidates outside the active local map are removed;
6. at most 32 candidates are retained by deterministic rank.

The rolling local search consumes the candidates as one internal goal set, not
as 32 independent searches. Its relaxed two-dimensional distance field is a
single multi-source field seeded by every candidate cell. `IsGoal()` and the
certified scaled-primitive terminal connector may satisfy any candidate, while
the final trajectory records which portal was selected. This shares OPEN and
CLOSED work across alternatives and prevents one bad centreline point from
consuming the request before fallback begins. Final-mission planning near the
true destination uses only the exact requested final goal, not the rolling
portal set.

Candidate failure does not relax safety and does not immediately fail the
whole request. `NO_PATH` is returned only after the bounded multi-goal graph is
exhausted.

## Shared ARA* Repair

### Anchor Ordering

OPEN ordering must use the ARA* anchor key as its primary key:

```text
f_epsilon(state) = g(state) + epsilon * h_anchor(state)
```

Path cost, stable state identity and insertion sequence provide deterministic
ties. Optional route or clearance guidance may be a later tie-break but cannot
precede the anchor key. Incumbent termination compares the incumbent with the
minimum anchor key over OPEN, not the entry with minimum guidance.

### Heuristic Contract

`h_anchor` must be finite, non-negative and a lower bound under the planner's
normalized edge-cost units. For the wheeled graph it is the maximum of valid
lower bounds derived from:

- planar distance to the goal region;
- a relaxed point-robot two-dimensional goal-distance field; and
- minimum unavoidable goal-yaw motion cost when a yaw constraint exists.

The two-dimensional field is a relaxation of the footprint/motion graph and
must use matching corner-blocking semantics. A cell that is unreachable in a
field used only for secondary guidance is never converted to zero; it is
pruned when disconnection is conclusive or excluded from that secondary
guidance while the finite anchor remains authoritative.

### First Certified Solution

The response objective is the first fully certified executable reference, not
an epsilon-one optimality proof. The first valid incumbent may finish the
request. Later anytime refinement is not allowed to delay first-reference
publication past the target merely to improve cost. Background refinement and
reference replacement are non-goals for this change.

## Primitive-Based Hybrid A* Wheel Graph

### Relationship to the Existing Request-Scoped Frame

The exact request start still defines an immutable local SE(2) frame, so every
finite, in-bounds and physically valid continuous start maps to relative
`(0, 0, 0)`. World-space map lookup and safety certification remain unchanged.

This design supersedes the canonical-interior-state requirement in
`2026-08-24-request-scoped-se2-lattice-design.md`. It retains that document's
request frame and exact certified terminal connector, but no longer replaces
an interior primitive endpoint with a canonical raster pose. It also
supersedes fixed 64/128 yaw-bin counts as a permanent contract.

### State and Edge Semantics

Each search node stores:

- the exact continuous world pose produced by its incoming certified motion
  primitive;
- its request-frame quantized `(x, y, yaw)` key;
- motion mode and narrow/wide resolution class;
- path cost, parent and incoming primitive identity.

Quantization indexes CLOSED and duplicate detection; it never changes the
physical endpoint. Every interior edge is still one supplied platform motion
primitive applied from the source node's continuous pose. The existing scaled
primitive terminal connector remains the only way to shorten a primitive into
the exact position/yaw goal tolerance.

Because continuous representatives inside one quantized bin can have
different future collision outcomes, duplicate handling cannot assume that
the lowest path cost alone dominates every representative. Each key retains a
deterministic label set of at most four entries. A label is dominated only when
another label has no greater path cost, matches motion mode/resolution class,
and differs by no more than one quarter of the active XY key resolution and
one quarter of one yaw-bin width. Non-dominated labels remain separate state
IDs. If inserting a fifth non-dominated label, all five are ranked by anchor
key, path cost, descending certified pose clearance, distance to the key-bin
centre, yaw residual and stable creation sequence; the best four survive.
This is a deterministic resolution-complete approximation, not a claim of
continuous-space completeness. Capacity accounting uses actual allocated
labels, not a preallocated fixed maximum.

Yaw discretization is derived from reachable interior primitive increments.
The builder tests candidate bin counts 16, 32, 64, 128 and 256 in ascending
order and selects the first for which every non-zero primitive yaw change is
an integer number of bins within `1e-9 rad`. The project wheel's `pi/16` turn
and spin increments therefore select 32 relative headings. Capabilities that
are not commensurate with any candidate use 256 bins rather than silently
changing their primitive geometry. Narrow terrain may refine position bins but
does not allocate otherwise unreachable yaw headings. Continuous physical
endpoints remain unchanged across wide/narrow keys.

### Expected Effect

Straight, reverse, spin and stop/switch successors currently accepted remain
available. Standard forward/reverse arcs that were lost after canonical
snapping become valid interior successors. The output remains a sampled
piecewise trajectory of certified lines, circular arcs, spins and mode
switches; no unconstrained spline or lateral motion is introduced.

## Runtime and Memory Optimization

Correctness changes precede performance work so that benchmarks optimize the
intended graph rather than the defective graph.

### Sparse Search Storage

- Allocate nodes, parent records and cached successor lists only for discovered
  labels.
- Replace `std::set` OPEN with a binary heap and lazy stale-entry removal.
- Represent INCONS and touched states with compact vectors plus membership
  flags.
- Rekey only OPEN/INCONS entries when epsilon changes; never scan a nominal
  capacity that was not discovered.
- Remove the arbitrary 131,072-state preallocation/cap as a normal planning
  limit. Real allocation failure remains `RESOURCE_EXHAUSTED`.

### Revision-Keyed Immutable Caches

The ROS input store's global/local sequence identities define cache lifetime.
Cache keys include every parameter that changes semantics:

- global snapshot/projection: global sequence, occupancy threshold, platform
  inflation;
- local snapshot/terrain: local sequence and local occupancy threshold;
- two-dimensional goal field: local terrain identity and goal cell;
- certified edges: local terrain identity, capability identity, continuous
  source label and primitive identity.

A sequence or capability change invalidates affected entries before they can
be used. Cache reuse never crosses incompatible map/capability semantics.

### Projection Work

- Replace local priority-queue clearance propagation with a linear-time exact
  Euclidean distance transform using the same cell-area convention required by
  safety.
- Replace per-cell roughness heap allocations with fixed-size stack storage and
  accumulated normal-equation terms.
- Preserve fail-closed non-finite elevation and occupancy handling.

### Edge Certification Pipeline

Certification proceeds in increasing cost:

1. bounds, mode and primitive-shape checks;
2. precomputed clearance field and yaw-indexed footprint-mask broad phase;
3. exact polygon/cell, wheel support, elevation, slope, roughness, relief,
   underbody and dynamics evaluation for competitive edges;
4. map/capability identity confirmation for every edge in the returned path.

Broad-phase acceptance is never final acceptance. Every returned edge must
have a complete exact certification record.

## Rolling and ROS Behavior

The rolling server retains one global route while its input identity remains
valid. It does not rerun local planning merely because the continuously
projected target moved by floating-point epsilon. Replanning is triggered by:

- the active local reference reaching its handoff region;
- a semantically new local-map sequence after the minimum replan interval;
- route deviation beyond the configured bound;
- global-map or transform invalidation; or
- invalidation of the active certified reference.

All triggers use the same end-to-end timing policy for the resulting replan.
The first lunar-surface request includes cold global plus first local planning.
Later rolling cycles may reuse the global projection/route and are measured as
separate local replan requests. Vehicle travel time and the duration of the
whole mission are not part of the one-second planning target.

If a two-to-three-second result is fully certified and the request/map identity
is still active, it may be published as `PLAN_FOUND_LATE`. The latency SLA
failure remains visible in diagnostics. A three-second timeout never publishes
an incomplete or stale new reference; execution policy may retain an already
active independently valid reference or command its existing safe hold
behavior.

## Implementation Stages

Each stage is independently testable and must keep the preceding stages green.

1. **Baseline and timing instrumentation**
   - freeze deterministic counters and current failure evidence;
   - add fake-clock tests for exact one-, two- and three-second boundaries;
   - add per-phase and cache-hit diagnostics without changing search behavior.
2. **Request time contract**
   - remove one-second/150 ms termination;
   - apply one shared three-second deadline;
   - implement late-success and hard-timeout result behavior.
3. **Global cell-area inflation and portal selection**
   - fix hard-feasibility geometry;
   - introduce deterministic portal candidates and fallback;
   - make the original 750 m fixture progress through the bottleneck without a
     test-only target shift.
4. **Shared ARA* correctness**
   - restore anchor ordering, termination and unreachable-guidance behavior;
   - run global, wheel, legged and hopper regression.
5. **Primitive-based Hybrid A***
   - retain continuous primitive endpoints and labeled quantized keys;
   - add consecutive-arc, S-turn, reverse-arc, narrow-corridor and arbitrary
     continuous-start coverage.
6. **Sparse containers and caches**
   - replace dense initialization and tree OPEN;
   - cache immutable projections/fields by sequence;
   - add staged edge certification.
7. **Rolling ROS integration and target validation**
   - apply stable replan triggers and progress feedback;
   - validate cold initial planning and warm rolling replans in RViz;
   - record Jazzy host and target Orin evidence separately.

## Acceptance Criteria

### Functional

1. All currently passing wheel tests remain green.
2. The unmodified 750 m test passes ten consecutive runs.
3. At least two standard project arc primitives can be consecutive interior
   edges; the test cannot be satisfied by a direct terminal connector.
4. Arbitrary translated/yaw starts, S-turns, narrow corridors, reverse arcs and
   stop/switch mode transitions have deterministic success fixtures.
5. Global portal failure tries deterministic safe alternatives before
   concluding `NO_PATH`.
6. Shared ARA* tests plus wheel, legged, hopper and global suites pass.
7. Identical input/map/capability identities produce deterministic references
   and diagnostics except wall-clock fields.

### Timing

1. On each declared reference platform, deterministic acceptance scenarios
   never use independent additive global/local deadlines.
2. Reference-platform p95 end-to-end first-certified-reference latency is less
   than one second.
3. Any deterministic performance acceptance sample reaching two seconds is a
   recorded SLA failure even if it later returns a valid late reference.
4. Fake-clock tests prove that one and two seconds do not stop or reset search.
5. Fake-clock tests prove that three seconds stops global, local,
   certification and result commitment.
6. The former segment-21 scenario does not hit a fixed state capacity, and its
   footprint/terrain cell checks fall below 5,000,000 on the fixed fixture.
7. Existing simple wheel scenarios regress by no more than ten percent in p95
   end-to-end time on the same build and hardware.

### Safety and Runtime

1. Returned paths retain complete footprint, clearance, terrain and dynamics
   certification.
2. A map/capability identity change invalidates incompatible cached
   projections, fields and edges.
3. A canceled or replaced request cannot publish a late reference.
4. A three-second timeout publishes no new reference.
5. ROS 2 Jazzy builds with tests enabled and disabled; test compilation may be
   omitted only in the explicitly requested deployment build, never from
   verification evidence.
6. RViz evidence shows valid frame IDs, a non-empty certified path and the
   expected latency classification.
7. Host validation and Jetson Orin readiness are reported separately.

## Documentation Deliverables

Implementation updates must keep these documents consistent:

- this design document and its implementation plan;
- the request-scoped SE(2) design, with a supersession note for canonical
  interior endpoints and fixed yaw-bin counts;
- planner package documentation describing latency classes, the three-second
  hard deadline, Hybrid A* state semantics, cache identities and metrics;
- RViz demo documentation with commands and expected diagnostics.

## Non-Goals

- Replacing the planner with Nav2, OMPL or another external planning stack.
- Changing the controller or smoothing paths into unconstrained splines.
- Relaxing safety constraints to meet timing targets.
- Changing map, odometry or TF data to align the vehicle with a raster.
- Changing the `PlanMotion.action` schema.
- Proving continuous-space optimality or completeness.
- Delaying the first certified reference for epsilon-one optimization.
- Persisting search state across independent requests or arbitrary map
  revisions in this change.
- Treating total vehicle mission duration as planning-request latency.

## Risks and Mitigations

- **Hybrid-state aliasing:** non-dominated continuous labels within one key are
  retained and covered by obstacle-sensitive regression tests.
- **State growth:** sparse allocation, reachable yaw bins, corridor restriction
  and deterministic label dominance replace an arbitrary preallocated cap.
- **Global over-conservatism:** exact cell-area distance avoids both centre
  under-inflation and an unnecessarily large half-diagonal blanket margin.
- **Cache staleness:** every cache key includes source sequence and semantic
  parameters; returned paths recheck identities before publication.
- **Shared-engine regression:** ARA* changes are isolated behind shared unit
  tests and then validated on all three platforms and global search.
- **Timing test flakiness:** threshold semantics use fake clocks; wall-clock
  performance tests run separately on declared hardware with repeated samples.
- **Late-result race:** active request and map identities are checked again
  before committing any two-to-three-second reference.
- **Design overlap:** this document explicitly supersedes only the canonical
  interior-state and fixed yaw-bin portions of the earlier request-scoped SE(2)
  design; its exact-start frame and certified terminal connector remain in
  force.
