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
from .geometry import polygon_mask, world_to_cell
from .maps import FREE, BLOCKED, UNKNOWN
from .sensor import first_pending_cells, target_visibility, validate_sensor

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


def measured_workspace(snapshot, task, sensor):
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
        raster.states != UNKNOWN,
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

    def update(self, snapshot):
        w = measured_workspace(snapshot, self.task, self.sensor)
        self.workspace = w
        area = float(np.count_nonzero(w.known & w.task_mask)) * snapshot.resolution_m**2
        previous = self._known_area if self._epoch == snapshot.epoch else 0.0
        self._epoch, self._known_area = snapshot.epoch, area
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
        direct = {}  # pending measurement -> actual R observation witness
        # Most demands directly seed their movement component. Only stances in
        # other components, and blocked-M/R demands, need additional ray tests.
        other = potential & ~relevant[components]
        near_other = maximum_filter(other, size=2 * d + 1, mode="constant", cval=0)
        near_r = maximum_filter(w.reachable, size=2 * d + 1, mode="constant", cval=0)
        special = pending & (near_other | ((components == 0) & near_r))
        h, width = potential.shape
        for tx, ty in _xy(special):
            x0, x1 = max(0, tx - d), min(width, tx + d + 1)
            y0, y1 = max(0, ty - d), min(h, ty + d + 1)
            wanted = potential[y0:y1, x0:x1] & ~relevant[components[y0:y1, x0:x1]]
            if components[ty, tx] == 0:
                wanted |= w.reachable[y0:y1, x0:x1]
            stances = _xy(wanted) + [x0, y0]
            if not len(stances):
                continue
            # Center rays are reciprocal when both endpoints are exempt from
            # intermediate blocking; measured B alone determines occlusion.
            seen = stances[target_visibility(w.intrinsic, (tx, ty), stances, radius)]
            if not len(seen):
                continue
            relevant[components[seen[:, 1], seen[:, 0]]] = True
            relevant[0] = False
            rr = seen[w.reachable[seen[:, 1], seen[:, 0]]]
            if len(rr):
                direct[(int(tx), int(ty))] = tuple(map(int, rr[0]))
        # Only pending center measurements at real interfaces contribute utility.
        frontier_mask = ~w.known & binary_dilation(w.known, structure=CROSS)
        frontier_mask &= relevant[components]
        # Bind every R/potential interface to its native R-side cell, preserving
        # external transit even when its pending center lies outside the task.
        witnesses = {}
        known_bytes = np.asarray(w.known, dtype=np.uint8)
        for target, source in direct.items():
            first = first_pending_cells(w.intrinsic, known_bytes, source, [target])[0]
            if first >= 0:
                witnesses[(int(first % width), int(first // width))] = source
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ys, xs = np.nonzero(
                w.reachable & np.roll(frontier_mask, (-dy, -dx), axis=(0, 1))
            )
            for x, y in zip(xs, ys):
                target = (int(x + dx), int(y + dy))
                witnesses.setdefault(target, (int(x), int(y)))
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
        opportunity = bool(
            np.any(relevant[components] & binary_dilation(w.reachable, structure=CROSS))
        ) or bool(direct)
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
