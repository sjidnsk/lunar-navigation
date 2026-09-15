"""Effective center measurements. This module never imports privileged builders."""

from dataclasses import dataclass
import math
import numpy as np
import lunar_drl_terrain_native as native
from .contracts import freeze_array
from .geometry import world_to_cell


def validate_sensor(sensor):
    if sensor.offset_x_m != 0 or sensor.offset_y_m != 0:
        raise ValueError(
            "virtual sensor translation is not supported by reference mapping"
        )
    if (
        not math.isfinite(sensor.range_m)
        or sensor.range_m <= 0
        or not math.isfinite(sensor.fov_deg)
        or not 0 < sensor.fov_deg <= 360
        or not math.isfinite(sensor.offset_yaw_rad)
    ):
        raise ValueError("invalid sensor range/FOV/mounting yaw")


@dataclass(frozen=True)
class VisibleMeasurements:
    shape: tuple
    indices: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    heights: np.ndarray
    stats: np.ndarray

    def __post_init__(self):
        for name in ("indices", "rows", "cols", "heights", "stats"):
            object.__setattr__(self, name, freeze_array(getattr(self, name)))

    @property
    def mask(self):
        mask = np.zeros(self.shape, bool)
        mask[self.rows, self.cols] = True
        return mask


def visible_cells(intrinsic, origin, resolution_m, pose, sensor):
    """Visible row/column indices from measured B; scratch is range bounded.

    Deployment may call this helper on a measured local grid. Neither Scene nor
    CoverageReference is imported. Quantize only origin; preserve physical yaw.
    """
    validate_sensor(sensor)
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("positive resolution required")
    x, y = world_to_cell(pose.x, pose.y, origin, resolution_m)
    h, w = intrinsic.shape
    if not 0 <= x < w or not 0 <= y < h:
        raise ValueError("sensor outside provided grid")
    radius = math.ceil(sensor.range_m / resolution_m)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = np.ascontiguousarray(intrinsic[y0:y1, x0:x1], dtype=np.uint8)
    mask = np.empty(patch.shape, np.uint8)
    native.observe(
        patch,
        x - x0,
        y - y0,
        sensor.range_m / resolution_m,
        pose.yaw + sensor.offset_yaw_rad,
        math.radians(sensor.fov_deg),
        mask,
    )
    rows, cols = np.nonzero(mask)
    return rows + y0, cols + x0


class SensorModel:
    @staticmethod
    def observe(terrain, pose, sensor):
        rows, cols = visible_cells(
            terrain.intrinsic, terrain.origin, terrain.resolution_m, pose, sensor
        )
        valid = np.isfinite(terrain.heights[rows, cols])
        rows = rows[valid]
        cols = cols[valid]
        return VisibleMeasurements(
            terrain.shape,
            rows * terrain.shape[1] + cols,
            rows,
            cols,
            terrain.heights[rows, cols],
            terrain.stats[rows, cols],
        )


def target_visibility(intrinsic, source_xy, targets_xy, range_cells):
    """Full-heading geometric relation, using the same native rays as observe.

    Coordinates are integer raster cells. Callers apply physical heading/FOV to
    this single relation, avoiding eight duplicate raycasts.
    """
    targets = np.ascontiguousarray(targets_xy, dtype=np.int64).reshape(-1, 2)
    result = np.empty((len(targets), 1), np.uint8)
    native.visible_targets(
        np.ascontiguousarray(intrinsic, dtype=np.uint8),
        int(source_xy[0]),
        int(source_xy[1]),
        float(range_cells),
        targets,
        result,
    )
    return result[:, 0].astype(bool)


def first_pending_cells(intrinsic, known, source_xy, targets_xy):
    """First pending center with its own visible native ray; -1 if none.

    Supercover traversal contacts whose center rays are blocked are skipped.
    A visible pending demand remains the fallback, so its opportunity is retained.
    """
    targets = np.ascontiguousarray(targets_xy, dtype=np.int64).reshape(-1, 2)
    out = np.empty((len(targets), 1), np.int64)
    native.first_pending(
        np.ascontiguousarray(intrinsic, dtype=np.uint8),
        np.ascontiguousarray(known, dtype=np.uint8),
        int(source_xy[0]),
        int(source_xy[1]),
        targets,
        out,
    )
    return out[:, 0]


def direct_witnesses(intrinsic, reachable, targets_xy, range_cells):
    """First actual reachable center-ray source per target, or [-1,-1].

    Movement components do not enter this optical relation. Sparse native
    first-source search avoids enumerating every source/target Python pair.
    """
    targets = np.ascontiguousarray(targets_xy, dtype=np.int64).reshape(-1, 2)
    result = np.empty((len(targets), 2), np.int64)
    native.visible_witnesses(
        np.ascontiguousarray(intrinsic, dtype=np.uint8),
        np.ascontiguousarray(reachable, dtype=np.uint8),
        targets,
        float(range_cells),
        result,
    )
    return result
