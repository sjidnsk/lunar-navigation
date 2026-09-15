"""Deterministic bounded truth geometry and native terrain derivation.

Offline ownership only. No privileged raster is built on import. Geometry uses
analytic crater/rock disks and single-layer room/passage unions. Segment-distance
helper adapted from this repository's observability-coverage scene.py (read-only reference inspected 2026-09-15);
no old native binding, classifier, launch-surface override or learner is reused.
"""

import hashlib
import json
import math
import numpy as np
import lunar_drl_terrain_native as native
from .contracts import Pose, TaskSpec
from .geometry import cell_center, world_to_cell


def _segment_distance(x, y, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(x - ax - t * dx, y - ay - t * dy)


class Scene:
    GENERATOR_VERSION = 3

    def __init__(self, seed, family, extent_m, resolution_m=0.2, platform=None):
        if family not in ("moon", "cave"):
            raise ValueError("family must be moon or cave")
        if not math.isfinite(extent_m) or not 20 <= extent_m <= 1000:
            raise ValueError("extent_m must be in [20,1000]")
        if not math.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if platform is None:
            from .config import load_platform_config

            platform = load_platform_config()
        self.seed = int(seed)
        self.family = family
        self.extent_m = float(extent_m)
        self.resolution_m = float(resolution_m)
        self.platform = platform
        e = self.extent_m
        margin = 12.0
        self.origin = (-e / 2 - margin, -e / 2 - margin)
        n = math.ceil((e + 2 * margin) / resolution_m)
        self.shape = (n, n)
        self.bounds = (
            *self.origin,
            self.origin[0] + n * resolution_m,
            self.origin[1] + n * resolution_m,
        )
        identity = dict(
            version=self.GENERATOR_VERSION,
            seed=self.seed,
            family=family,
            extent_m=e,
            resolution_m=resolution_m,
            capability=dict(platform.capability),
        )
        self.scene_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.task = TaskSpec(
            self.scene_id,
            "map",
            [[-e / 2, -e / 2], [e / 2, -e / 2], [e / 2, e / 2], [-e / 2, e / 2]],
        )
        self.task_polygon = self.task.polygon
        rng = np.random.default_rng(self.seed)
        self._phase = float(rng.uniform(-math.pi, math.pi))
        self._rocks = []
        self._craters = []
        self._rooms = []
        self._passages = []
        self._nominal_start = (-0.32 * e, -0.2 * e)
        self._yaw = float(rng.uniform(-math.pi, math.pi))
        if family == "moon":
            for _ in range(max(4, int(e / 20))):
                self._craters.append(
                    (
                        *rng.uniform(-0.5 * e, 0.5 * e, 2),
                        rng.uniform(1.0, max(2.0, e * 0.05)),
                        rng.uniform(0.2, 1.0),
                    )
                )
            for _ in range(max(15, int(e / 8))):
                self._rocks.append(
                    (
                        *rng.uniform(-0.5 * e - 8, 0.5 * e + 8, 2),
                        rng.uniform(0.3, 1.5),
                        rng.uniform(0.5, 2.0),
                    )
                )
        else:
            # A main loop includes legitimate terrain outside the task; narrow
            # spurs and a sealed chamber remain part of the same unchanged scene.
            points = [
                self._nominal_start,
                (-e / 2 - 5, -0.2 * e),
                (-e / 2 - 5, 0.2 * e),
                (-0.25 * e, 0.2 * e),
                (0, 0.2 * e),
                (0, -0.2 * e),
            ]
            radius = max(2.5, 0.035 * e)
            self._rooms = [(x, y, radius) for x, y in points]
            edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
            self._passages = [(*points[a], *points[b], 1.4) for a, b in edges]
            for _ in range(max(4, int(e / 40))):
                x, y = rng.uniform([-0.38 * e, -0.38 * e], [0.05 * e, 0.38 * e])
                parent = points[int(rng.integers(len(points)))]
                rr = float(rng.uniform(2.0, max(3.0, 0.03 * e)))
                self._rooms.append((x, y, rr))
                self._passages.append(
                    (*parent, x, y, float(rng.choice([0.45, 1.4, 2.0])))
                )
            self.sealed_room = (0.32 * e, 0.22 * e, max(2.0, 0.035 * e))
            self._rooms.append(self.sealed_room)

    def height_tile(self, x0, y0, width, height):
        if (
            min(x0, y0) < 0
            or min(width, height) <= 0
            or x0 + width > self.shape[1]
            or y0 + height > self.shape[0]
        ):
            raise ValueError("tile outside provided terrain bounds")
        x = (
            self.origin[0]
            + (np.arange(x0, x0 + width, dtype=np.float64) + 0.5) * self.resolution_m
        )
        y = (
            self.origin[1]
            + (np.arange(y0, y0 + height, dtype=np.float64) + 0.5) * self.resolution_m
        )
        x = x[None, :]
        y = y[:, None]
        if self.family == "moon":
            z = 0.03 * np.sin(x * 0.13 + self._phase) + 0.02 * np.cos(
                y * 0.11 - self._phase
            )
            for cx, cy, r, depth in self._craters:
                if (
                    cx + 2 * r < x.min()
                    or cx - 2 * r > x.max()
                    or cy + 2 * r < y.min()
                    or cy - 2 * r > y.max()
                ):
                    continue
                q = np.hypot(x - cx, y - cy) / r
                z += np.where(
                    q < 2,
                    depth
                    * (-np.exp(-q * q * 3) + 0.25 * np.exp(-(((q - 1) / 0.2) ** 2))),
                    0.0,
                )
            for cx, cy, r, rise in self._rocks:
                if (
                    cx + r < x.min()
                    or cx - r > x.max()
                    or cy + r < y.min()
                    or cy - r > y.max()
                ):
                    continue
                z += np.where((x - cx) ** 2 + (y - cy) ** 2 <= r * r, rise, 0.0)
        else:
            inside = np.zeros((height, width), bool)
            for cx, cy, r in self._rooms:
                if (
                    cx + r < x.min()
                    or cx - r > x.max()
                    or cy + r < y.min()
                    or cy - r > y.max()
                ):
                    continue
                inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r * r
            for ax, ay, bx, by, r in self._passages:
                if (
                    max(ax, bx) + r < x.min()
                    or min(ax, bx) - r > x.max()
                    or max(ay, by) + r < y.min()
                    or min(ay, by) - r > y.max()
                ):
                    continue
                inside |= _segment_distance(x, y, ax, ay, bx, by) <= r
            z = np.where(inside, 0.0, 2.0)
        return np.asarray(z, np.float32)

    def initial_pose(self, terrain):
        """Select native M FREE support near the generated main floor, no override."""
        if terrain.terrain_id != self.scene_id:
            raise ValueError("terrain belongs to different scene")
        x, y = terrain.world_to_cell(*self._nominal_start)
        radius = math.ceil(max(3.0, 0.04 * self.extent_m) / self.resolution_m)
        y0, y1 = max(0, y - radius), min(self.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(self.shape[1], x + radius + 1)
        valid = terrain.navigation[y0:y1, x0:x1] == 1
        if self.family == "cave":
            valid &= terrain.heights[y0:y1, x0:x1] == 0
        rows, cols = np.nonzero(valid)
        if not len(rows):
            raise ValueError("generated main terrain has no legal native start")
        i = np.argmin((rows + y0 - y) ** 2 + (cols + x0 - x) ** 2)
        return Pose(
            *terrain.cell_center(int(cols[i] + x0), int(rows[i] + y0)), self._yaw
        )


class TerrainGrid:
    """Compact intrinsic B, physical navigation M and effective float32 evidence."""

    def __init__(
        self, heights, resolution_m, origin, platform, stats=None, terrain_id=None
    ):
        self.heights = np.ascontiguousarray(heights, dtype=np.float32)
        if self.heights.ndim != 2 or not self.heights.size:
            raise ValueError("nonempty height grid required")
        self.shape = self.heights.shape
        self.resolution_m = float(resolution_m)
        self.origin = tuple(origin)
        supplied = stats is not None
        self.stats = (
            np.ascontiguousarray(stats, dtype=np.float32).copy()
            if supplied
            else np.empty((*self.shape, 4), np.float32)
        )
        if self.stats.shape != (*self.shape, 4):
            raise ValueError("four effective stats required")
        if supplied:
            known = np.isfinite(self.heights)
            if (
                not np.all(np.isfinite(self.stats[known]))
                or np.any(self.stats[known, :3] < 0)
                or not np.all(np.isin(self.stats[known, 3], [0, 1]))
            ):
                raise ValueError("invalid effective measurements")
        self.intrinsic = np.empty(self.shape, np.uint8)
        self.navigation = np.empty(self.shape, np.uint8)
        native.derive(
            self.heights,
            self.resolution_m,
            *self.origin,
            dict(platform.capability),
            self.stats,
            self.intrinsic,
            self.navigation,
            int(supplied)
        )
        self.terrain_id = (
            terrain_id
            or hashlib.sha256(
                self.heights.tobytes()
                + self.stats.tobytes()
                + json.dumps(
                    (self.resolution_m, self.origin, dict(platform.capability)),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
        for array in (self.heights, self.stats, self.intrinsic, self.navigation):
            array.setflags(write=False)

    @classmethod
    def from_heights(cls, heights, resolution_m, origin, platform, *, stats=None):
        return cls(heights, resolution_m, origin, platform, stats)

    @classmethod
    def from_scene(cls, scene):
        heights = np.empty(scene.shape, np.float32)
        for y in range(0, scene.shape[0], 256):
            for x in range(0, scene.shape[1], 256):
                h = min(256, scene.shape[0] - y)
                w = min(256, scene.shape[1] - x)
                heights[y : y + h, x : x + w] = scene.height_tile(x, y, w, h)
        return cls(
            heights,
            scene.resolution_m,
            scene.origin,
            scene.platform,
            terrain_id=scene.scene_id,
        )

    def world_to_cell(self, x, y):
        return world_to_cell(x, y, self.origin, self.resolution_m)

    def cell_center(self, x, y):
        return cell_center(x, y, self.origin, self.resolution_m)

    def reachable(self, start):
        x, y = self.world_to_cell(start.x, start.y)
        result = np.empty(self.shape, np.uint8)
        native.reachable(self.navigation, x, y, result)
        return result
