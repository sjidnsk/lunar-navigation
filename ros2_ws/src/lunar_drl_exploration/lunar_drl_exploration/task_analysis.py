"""Observed task opportunity analysis; B occludes, M constrains movement.

Potential components are formed *after* removing native reachable R. The
storage exterior is a single unbounded UNKNOWN component. Component membership
is not a claim that an unknown viewpoint is executable.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import binary_dilation, label, maximum_filter
from .contracts import TaskReport
from .coverage import CoverageHistory
from .geometry import polygon_mask, world_to_cell
from .maps import FREE, BLOCKED, UNKNOWN
from .sensor import (
    direct_witnesses,
    first_pending_cells,
    optical_candidate_mask,
    source_visibility,
    validate_sensor,
)

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


@dataclass
class MeasuredWorkspace:
    bounds: tuple
    origin: tuple
    known: np.ndarray
    intrinsic: np.ndarray
    navigation: np.ndarray
    reachable: np.ndarray
    task_mask: np.ndarray
    available: bool


def measured_workspace(snapshot, task, sensor, coverage=None):
    """Rasterize all measured evidence and task/range margin, never hidden truth."""
    resolution = snapshot.resolution_m
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("positive map resolution required")
    corners = np.floor(
        (task.polygon.astype(float) - snapshot.origin[:2]) / resolution
    ).astype(np.int64)
    anchor = world_to_cell(
        snapshot.pose.x, snapshot.pose.y, snapshot.origin, resolution
    )
    low = np.minimum(corners.min(axis=0), anchor)
    high = np.maximum(corners.max(axis=0) + 1, np.array(anchor) + 1)
    for (tx, ty), tile in snapshot.tiles.items():
        evidence = (
            (tile.states != UNKNOWN)
            | (tile.intrinsic_states != UNKNOWN)
            | (tile.observed != UNKNOWN)
        )
        evidence = evidence.reshape(256, 256)
        ys = np.flatnonzero(evidence.any(axis=1))
        xs = np.flatnonzero(evidence.any(axis=0))
        if len(xs):
            low = np.minimum(low, [tx * 256 + xs[0], ty * 256 + ys[0]])
            high = np.maximum(high, [tx * 256 + xs[-1] + 1, ty * 256 + ys[-1] + 1])
    if coverage is not None and coverage.bounds is not None:
        low=np.minimum(low,coverage.bounds[:2]);high=np.maximum(high,coverage.bounds[2:])
    margin = math.ceil(sensor.range_m / resolution) + 1
    low -= margin
    high += margin
    bounds = (*map(int, low), *map(int, high))
    raster = snapshot.raster(bounds)
    origin = (
        snapshot.origin[0] + low[0] * resolution,
        snapshot.origin[1] + low[1] * resolution,
    )
    native_components, _ = label(raster.navigation_states == FREE, CROSS)
    connections = np.asarray(snapshot.start_connections).reshape(-1, 2) - low
    valid = (
        (connections[:, 0] >= 0)
        & (connections[:, 0] < native_components.shape[1])
        & (connections[:, 1] >= 0)
        & (connections[:, 1] < native_components.shape[0])
    )
    connections = connections[valid]
    seeds = (
        np.unique(native_components[connections[:, 1], connections[:, 0]])
        if len(connections)
        else []
    )
    seeds = np.asarray(seeds)
    seeds = seeds[seeds != 0]
    available = snapshot.start_connection_status == "READY" and len(seeds) > 0
    reachable = (
        np.isin(native_components, seeds)
        if available
        else np.zeros(native_components.shape, bool)
    )
    return MeasuredWorkspace(
        bounds,
        origin,
        raster.states != UNKNOWN if coverage is None else coverage.mask(bounds),
        raster.intrinsic_states,
        raster.navigation_states,
        reachable,
        polygon_mask(reachable.shape, origin, resolution, task.polygon),
        available,
    )


def _xy(mask):
    return np.argwhere(mask)[:, ::-1].copy()


class TaskAnalyzer:
    def __init__(self, task, sensor):
        validate_sensor(sensor)
        self.task, self.sensor = task, sensor
        self._epoch = None
        self._known_area = 0.0
        self.workspace = None
        self.coverage = CoverageHistory(task)

    def update(self, snapshot):
        self.coverage.consume(snapshot)
        w = measured_workspace(snapshot, self.task, self.sensor, self.coverage)
        self.workspace = w
        area = self.coverage.known_area_m2
        previous = self._known_area if self._epoch == self.coverage.identity else 0.0
        self._epoch, self._known_area = self.coverage.identity, area
        empty = np.empty((0, 2), np.int64)
        if not w.available:
            return TaskReport(
                area,
                max(0.0, area - previous),
                empty,
                empty,
                False,
                snapshot.revision,
                False,
                "INPUT_UNAVAILABLE",
            )
        pending = w.task_mask & ~w.known
        if not pending.any():
            return TaskReport(
                area, max(0.0, area - previous), empty, empty, True, snapshot.revision
            )
        potential = (w.navigation != BLOCKED) & ~w.reachable
        components, count = label(potential, CROSS)
        # Every raster-side potential boundary touches the same infinite exterior.
        exterior = np.unique(
            np.concatenate(
                (components[0], components[-1], components[:, 0], components[:, -1])
            )
        )
        exterior = exterior[exterior != 0]
        remap = np.arange(count + 1, dtype=np.int32)
        if len(exterior):
            remap[exterior] = exterior[0]
        components = remap[components]
        relevant = np.zeros(count + 1, bool)
        relevant[np.unique(components[pending])] = True
        relevant[0] = False
        radius = self.sensor.range_m / snapshot.resolution_m
        d = math.ceil(radius)
        # Most demands seed their movement component. Extra potential stances
        # need a cross-component ray; direct R visibility is a separate relation.
        # A potential movement component can expose an R entry only when it
        # actually touches R. Disconnected non-entry components cannot change
        # frontier witnesses or exhaustion; direct R observation is independent.
        entry = potential & binary_dilation(w.reachable, structure=CROSS)
        has_entry = np.zeros(count + 1, bool)
        has_entry[np.unique(components[entry])] = True
        has_entry[0] = False
        other = potential & has_entry[components] & ~relevant[components]
        near_other = maximum_filter(other, size=2 * d + 1, mode="constant", cval=0)
        near_r = maximum_filter(w.reachable, size=2 * d + 1, mode="constant", cval=0)
        # Once-per-analysis necessary optical support rejects closed-barrier
        # negative queries; UNKNOWN still transmits. Exact forward witnesses
        # remain mandatory for every surviving demand.
        near_r &= optical_candidate_mask(w.intrinsic, w.reachable)
        targets = _xy(pending & near_r)
        reachable_bytes = np.asarray(w.reachable, dtype=np.uint8)
        # Any beam reaching a pending demand first emits an UNKNOWN interface
        # beside a known transmitting cell, or starts at an UNKNOWN R source.
        # Include interfaces OUTSIDE the task. A visible demand and that first
        # interface are each within radius of the same source, hence <=2*radius
        # apart. This square-dilation support is necessary only; ordinary exact
        # demand queries below still determine all positives and their witnesses.
        interfaces = (
            ~w.known
            & binary_dilation(w.known & (w.intrinsic != BLOCKED), structure=CROSS)
        ) | (w.reachable & ~w.known)
        interface_cells = _xy(interfaces & near_r)
        if len(interface_cells) < len(targets):
            interface_sources = direct_witnesses(
                w.intrinsic, reachable_bytes, interface_cells, radius
            )
            visible_interfaces = interface_cells[interface_sources[:, 0] >= 0]
            if not len(visible_interfaces):
                targets = empty
            else:
                interfaces.fill(False)
                interfaces[visible_interfaces[:, 1], visible_interfaces[:, 0]] = True
                support = maximum_filter(
                    interfaces, size=4 * d + 1, mode="constant", cval=0
                )
                targets = _xy(pending & near_r & support)
        sources = direct_witnesses(w.intrinsic, reachable_bytes, targets, radius)
        direct = {
            tuple(map(int, target)): tuple(map(int, source))
            for target, source in zip(targets, sources)
            if source[0] >= 0
        }
        special = pending & near_other
        h, width = potential.shape
        for tx, ty in _xy(special):
            x0, x1 = max(0, tx - d), min(width, tx + d + 1)
            y0, y1 = max(0, ty - d), min(h, ty + d + 1)
            local_components = components[y0:y1, x0:x1]
            wanted = (
                potential[y0:y1, x0:x1]
                & has_entry[local_components]
                & ~relevant[local_components]
            )
            stances = _xy(wanted) + [x0, y0]
            if not len(stances):
                continue
            # Evaluate FORWARD from every potential stance. Prefix-cell hits
            # are directed, even though the underlying lattice is symmetric.
            seen = stances[source_visibility(w.intrinsic, stances, (tx, ty), radius)]
            if not len(seen):
                continue
            relevant[components[seen[:, 1], seen[:, 0]]] = True
            relevant[0] = False
        # Only pending center measurements at real interfaces contribute utility.
        frontier_mask = ~w.known & binary_dilation(w.known, structure=CROSS)
        frontier_mask &= relevant[components] & has_entry[components] & near_r
        # Optical K boundaries may lie beyond a measured non-R footprint band.
        # Preserve the movement component's R entry, but bind its visible first
        # pending interface to an exact native R optical witness, not adjacency.
        witnesses = {}
        known_bytes = np.asarray(w.known, dtype=np.uint8)
        for target, source in direct.items():
            first = first_pending_cells(
                w.intrinsic, known_bytes, source, [target], radius
            )[0]
            if first >= 0:
                witnesses[(int(first % width), int(first // width))] = source
        movement_targets = _xy(frontier_mask)
        movement_sources = direct_witnesses(
            w.intrinsic, reachable_bytes, movement_targets, radius
        )
        by_source = {}
        for target, source in zip(movement_targets, movement_sources):
            if source[0] >= 0:
                by_source.setdefault(tuple(map(int, source)), []).append(target)
        for source, targets in by_source.items():
            firsts = first_pending_cells(
                w.intrinsic, known_bytes, source, targets, radius
            )
            for first in firsts[firsts >= 0]:
                witnesses.setdefault((int(first % width), int(first // width)), source)
        # A height-only input can have measured FREE movement with unclassified
        # centers. Those are pending observations from R, not exhausted coverage.
        for x, y in _xy(pending & w.reachable):
            witnesses.setdefault((int(x), int(y)), (int(x), int(y)))
        if witnesses:
            fronts = np.asarray(sorted(witnesses), np.int64)
            sources = np.asarray([witnesses[tuple(p)] for p in fronts], np.int64)
            shift = np.array(w.bounds[:2])
            fronts += shift
            sources += shift
        else:
            fronts, sources = empty, empty
        # Opportunity may be real even if there is insufficient measured support
        # to expose its next interface. This is unavailable, never false exhausted.
        opportunity = bool(np.any(relevant[components] & entry)) or bool(direct)
        available = bool(len(fronts)) or not opportunity
        return TaskReport(
            area,
            max(0.0, area - previous),
            fronts,
            sources,
            not opportunity and not len(fronts),
            snapshot.revision,
            available,
            "READY" if available else "INPUT_UNAVAILABLE",
        )
