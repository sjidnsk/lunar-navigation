# Ground Candidate Exhaustion Upgrade Scan

## Goal

Avoid declaring `ZERO_EXPECTED_GAIN` merely because the normal three-per-
frontier sampler produced no positive candidate, while preserving the
observed-only policy contract and a bounded runtime cost.

## Scope

This change integrates the scan only as the ground candidate builder's
zero-positive fallback. Normal three-per-frontier refreshes are unchanged;
HOPPER behavior, PPO interfaces and checkpoint schemas are unchanged. The
scan result uses the explicit `GROUND_EXHAUSTION` pipeline kind, so a terminal
zero-gain result is distinguishable from normal frontier sampling.

## Observed-only inputs

The scan may use only the mission authorization mask, current observed mask,
observed terrain/obstacle layers, platform reachability, exact safe target
poses, sensor model and current pose.  It must not use the hidden physical
coverability denominator or any reward-only truth mask to choose a candidate.

## Candidate library

At normal candidate exhaustion, form residual demand from authorized, not-yet
observed coarse cells.  Split it into canonical four-neighbour components.
For every component, enumerate every currently globally reachable,
endpoint-feasible observation pose within the sensor-radius coarse stencil of
that component.  Evaluate the same batched exact gain function used by normal
candidates.  Positive poses are ranked deterministically and retained as at
most three candidates per residual component.

The result is complete only relative to this finite canonical pose library;
it makes no claim about arbitrary continuous poses.  A later integration may
terminate only after this library has been fully scanned and contains no
positive pose.

## Data and diagnostics

`ExhaustionUpgradeDiagnostics` records the residual component count, raw
reachable pose count, endpoint-feasible pose count, exact-gain evaluated pose
count, positive pose count and elapsed time.  The isolated API returns these
diagnostics plus canonical positive pose-cell witnesses and exact gain pairs.
The ground fallback converts them to the existing `PhysicalCandidateUniverse`
only when ordinary candidates contain no positive gain; no PPO tensor, action
or checkpoint field is added.

The radius prefilter is an exact Chebyshev-stencil existence test implemented
with a summed-area table.  It is O(map cells), rather than repeatedly
translating the full map for every offset in the sensor-radius stencil.

## Performance gates before integration

Use the real update-307 300 m terminal snapshots for WHEELED and LEGGED.  The
measurement reports wall time, candidate counts, peak temporary array bytes
and exact-gain batch count.  Integration is rejected unless the scan has one
batched exact-gain call, bounded temporary storage, deterministic output, and
does not materially increase ordinary (non-exhausted) candidate-refresh cost;
ordinary refresh must not invoke it at all.

## Required tests

1. A residual component with a positive reachable observation pose returns a
   candidate even when the normal frontier input is empty.
2. Disconnected or endpoint-infeasible poses are diagnosed but never emitted.
3. A no-gain residual returns no candidate with completed diagnostics.
4. Equivalent inputs produce bit-identical candidate IDs, ordering and
   diagnostics across repeated calls.
5. The production normal candidate builder does not invoke the upgrade scan.
