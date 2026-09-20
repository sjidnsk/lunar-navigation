"""Current task observation opportunities; B occludes and measured M defines R.

Only currently reachable, measured stances support observations. Unknown transit
never creates future stances or an obligation to investigate outside the task.
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
            return TaskReport(area, max(0.0, area-previous), empty, empty, False,
                              snapshot.revision, False, "INPUT_UNAVAILABLE")

        # This is a range-bounded single-observation query from measured R.
        # UNKNOWN can be revealed by that observation, but is never a stance.
        radius = self.sensor.range_m / snapshot.resolution_m
        d = math.ceil(radius)
        pending = w.task_mask & ~w.known
        # Any beam entering task-unknown space first hits its boundary beside
        # measured transmitting space or the task exterior. Query those current
        # interfaces, not every cell deep in an unobserved room. A source whose
        # center is unclassified is itself a pending observation.
        boundary = binary_dilation(
            (w.known & (w.intrinsic != BLOCKED)) | ~w.task_mask, structure=CROSS)
        pending &= boundary | w.reachable
        near = maximum_filter(w.reachable, size=2*d+1, mode="constant", cval=0)
        pending &= near & optical_candidate_mask(w.intrinsic, w.reachable)
        targets = _xy(pending)
        sources = direct_witnesses(w.intrinsic, w.reachable, targets, radius)
        by_source = {}
        for target, source in zip(targets, sources):
            if source[0] >= 0:
                by_source.setdefault(tuple(map(int, source)), []).append(target)

        # Keep only the first unmeasured interface of each supported beam. An
        # outside interface is relevant only through a current direct task view.
        witnesses = {}
        known = np.asarray(w.known, dtype=np.uint8)
        width = w.known.shape[1]
        for source, demands in by_source.items():
            firsts = first_pending_cells(w.intrinsic, known, source, demands, radius)
            for first in firsts[firsts >= 0]:
                witnesses.setdefault((int(first % width), int(first // width)), source)
        if witnesses:
            fronts = np.asarray(sorted(witnesses), np.int64)
            sources = np.asarray([witnesses[tuple(p)] for p in fronts], np.int64)
            shift = np.array(w.bounds[:2])
            fronts += shift
            sources += shift
        else:
            fronts, sources = empty, empty

        # A far outside start has no task observations yet. Keep the policy
        # active on its measured graph; the ordinary episode budget still applies.
        started = area > 0.0
        return TaskReport(area, max(0.0, area-previous), fronts, sources,
                          not witnesses and started, snapshot.revision,
                          reason_code="TASK_NOT_OBSERVED" if not started and not witnesses else "READY")
