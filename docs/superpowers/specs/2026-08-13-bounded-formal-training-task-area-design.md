# Bounded Formal Training Task Area Design

## Decision

Formal PPO episodes use a deterministic pseudo-random square task area around
the cached, platform-qualified start cell. Its side length is sampled uniformly
from `100.0, 104.0, ..., 500.0 m`. The task edge is aligned to the `4.0 m`
global grid, so an episode contains between `25 x 25` and `125 x 125` global
cells and the corresponding `500 x 500` to `2500 x 2500` cells on the `0.2 m`
detail grid.

Sampling uses `deterministic-uniform-square/v1`, keyed by the shared episode
seed. Matching platform-local lanes therefore receive the same side length,
while checkpoint restore and replay reproduce it exactly. The window is
centered on the qualified start where possible; near a physical map edge it is
translated intact rather than truncated or resized.

This scoped task replaces the former whole-scene exploration qualification for
training. A whole-scene closed-loop gate is no longer a prerequisite for
starting training.

## Map and authority boundaries

- The cached `1024 m x 1024 m` physical scene remains unchanged and available
  to the planner. A task edge is not converted into an obstacle.
- Candidate generation, the observed-only Oracle, sensor reveal, reward and the
  coverage denominator use the same scoped mission ROI.
- Platform coverability is intersected with the exact `0.2 m` detail task
  window. Obstacles, slope, clearance, support, connectivity and observability
  remain the authority for which cells count toward coverage.
- Sensor range remains `30.0 m`, formal ground local search margin remains
  `2.0 m`, and the policy/PPO architecture and hyperparameters do not change.
- WHEELED, LEGGED and HOPPER use the same task-area contract. Platform-specific
  qualified starts and coverability remain independent.

## Identity and lifecycle

`task_area` is a required frozen training-config mapping with minimum side
`100.0 m`, maximum side `500.0 m`, and sampling algorithm
`deterministic-uniform-square/v1`. It is included in the frozen configuration
hash, scenario schedule identity, run manifest and checkpoint compatibility. Existing
calibration artifacts and checkpoints cannot resume this changed task
distribution; the new formal run starts at global step zero.

The physical cache uses a bounded, stratified 128-scene inventory rather than
requiring all 1734 source scenarios. The frozen split allocation is 98 train,
12 validation, 12 test and all 6 holdout scenarios. The cache is formally
eligible only when every split has at least one exact common WHEELED, LEGGED
and HOPPER scene. This retains strict cache/source/capability identities and
paired evaluation while removing whole-scene materialization as a launch
prerequisite. Calibration and training construct their environments from the
same bounded cache and scoped config.

## Validation and launch

Required evidence before launch is limited to:

1. config RED/GREEN for the exact range and sampling algorithm;
2. task-area geometry, deterministic sampling, bounds and source-immutability
   RED/GREEN for all three platforms;
3. focused formal-builder/config/CLI suites plus UTF-8 and diff checks;
4. one real worker boundary per platform using the bounded cache;
5. fresh calibration for the new config, then a step-zero formal training
   launch with process, GPU, manifest and first-metric health checks.

No additional whole-scene closed-loop gate is required.
