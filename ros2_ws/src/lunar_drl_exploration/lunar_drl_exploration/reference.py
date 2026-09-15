"""Fixed offline coverable set: reachable native M stances, all attainable yaws."""

from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from scipy.ndimage import maximum_filter
import lunar_drl_terrain_native as native
from .contracts import freeze_array
from .geometry import polygon_mask
from .sensor import validate_sensor


@dataclass(frozen=True)
class CoverageReference:
    shape: tuple
    packed_mask: np.ndarray
    cell_area_m2: float
    area_m2: float
    reference_id: str
    reachable_bits: np.ndarray
    bitorder: str = "little"

    def __post_init__(self):
        object.__setattr__(
            self, "packed_mask", freeze_array(self.packed_mask, dtype=np.uint8)
        )
        object.__setattr__(
            self, "reachable_bits", freeze_array(self.reachable_bits, dtype=np.uint8)
        )

    @classmethod
    def build(cls, terrain, start, task, sensor):
        validate_sensor(sensor)
        reachable = terrain.reachable(start)
        task_mask = polygon_mask(
            terrain.shape, terrain.origin, terrain.resolution_m, task.polygon
        )
        # Cheap bounded-memory necessary distance test: square dilation is a
        # superset of the disk. Exact distance and ray tests remain native.
        d = math.ceil(sensor.range_m / terrain.resolution_m)
        candidate = maximum_filter(reachable, size=2 * d + 1, mode="constant", cval=0)
        candidate &= task_mask
        candidate &= np.isfinite(terrain.heights)
        visible = np.empty(terrain.shape, np.uint8)
        native.visible_union(
            terrain.intrinsic,
            reachable,
            candidate,
            sensor.range_m / terrain.resolution_m,
            visible,
        )
        packed = np.packbits(visible.ravel(), bitorder="little")
        digest = hashlib.sha256()
        digest.update(terrain.terrain_id.encode())
        digest.update(task.polygon.tobytes())
        digest.update(
            json.dumps((start.x, start.y, sensor.__dict__), sort_keys=True).encode()
        )
        return cls(
            terrain.shape,
            packed,
            terrain.resolution_m**2,
            int(np.count_nonzero(visible)) * terrain.resolution_m**2,
            digest.hexdigest(),
            np.packbits(reachable.ravel(), bitorder="little"),
        )

    def pack(self, mask):
        if np.asarray(mask).shape != self.shape:
            raise ValueError("mask shape mismatch")
        return np.packbits(np.asarray(mask, bool).ravel(), bitorder=self.bitorder)

    def unpack(self, bits):
        bits = np.asarray(bits)
        if bits.dtype != np.uint8 or bits.shape != self.packed_mask.shape:
            raise ValueError("packed mask shape/dtype mismatch")
        return (
            np.unpackbits(bits, count=math.prod(self.shape), bitorder=self.bitorder)
            .reshape(self.shape)
            .astype(bool)
        )

    def mask(self):
        return self.unpack(self.packed_mask)

    def covered_area(self, observed_bits):
        bits = np.asarray(observed_bits)
        if bits.dtype != np.uint8 or bits.shape != self.packed_mask.shape:
            raise ValueError("packed observed mask shape/dtype mismatch")
        return int(np.unpackbits(bits & self.packed_mask).sum()) * self.cell_area_m2

    def coverage_ratio(self, observed_bits):
        return self.covered_area(observed_bits) / self.area_m2 if self.area_m2 else 0.0

    def linear_index(self, x, y):
        if not 0 <= x < self.shape[1] or not 0 <= y < self.shape[0]:
            raise ValueError("cell outside reference")
        return y * self.shape[1] + x
