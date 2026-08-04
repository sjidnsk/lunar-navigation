"""Deterministic procedural rocks and craters for a source-bound polar window."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from shapely import from_wkb, to_wkb
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .raster import GLOBAL_GEOMETRY, GridGeometry


GENERATOR_VERSION = "lunar-polar-hazards/v1"


@dataclass(frozen=True)
class HazardScene:
    """Generated hazards separated into terrain truth and physical obstacles."""

    seed: str
    rock_footprints_wkb: tuple[bytes, ...]
    crater_polygons_wkb: tuple[bytes, ...]
    no_go_polygons_wkb: tuple[bytes, ...]
    crater_elevation_delta_m: np.ndarray
    physical_obstacle_ratio: np.ndarray

    @property
    def rock_footprints(self) -> tuple[object, ...]:
        return tuple(from_wkb(value) for value in self.rock_footprints_wkb)


def scene_seed(window_sha256: str, scenario_seed: int, *, generator_version: str = GENERATOR_VERSION) -> str:
    """Return the required identity binding for a generated scene."""
    return sha256(f"{window_sha256}:{scenario_seed}:{generator_version}".encode("utf-8")).hexdigest()


def _stream(seed: str, name: str) -> np.random.Generator:
    value = int.from_bytes(sha256(f"{seed}:{name}".encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(value)


def physical_obstacle_ratio(footprints: tuple[object, ...], geometry: GridGeometry = GLOBAL_GEOMETRY) -> np.ndarray:
    """Rasterize the union of rock footprints by exact polygon-cell area."""
    result = np.zeros((geometry.cells, geometry.cells), dtype=np.float32)
    if not footprints:
        return result
    union = unary_union(footprints)
    cell_area = geometry.resolution_m * geometry.resolution_m
    for row in range(geometry.cells):
        y0 = row * geometry.resolution_m
        for column in range(geometry.cells):
            x0 = column * geometry.resolution_m
            cell = box(x0, y0, x0 + geometry.resolution_m, y0 + geometry.resolution_m)
            if union.intersects(cell):
                result[row, column] = np.float32(min(1.0, union.intersection(cell).area / cell_area))
    return result


def generate_hazard_scene(window_sha256: str, scenario_seed: int, *, geometry: GridGeometry = GLOBAL_GEOMETRY, rock_count: int = 32, crater_count: int = 8, no_go_count: int | None = None) -> HazardScene:
    """Generate independent rock, crater-height and no-go polygon streams.

    Only rock footprints are physical obstacles.  Crater geometry contributes
    elevation truth only; slope and clearance are owned by injected C++
    capability projection rather than recomputed in Python.
    """
    if rock_count < 0 or crater_count < 0 or (no_go_count is not None and no_go_count < 0):
        raise ValueError("hazard counts must be non-negative")
    if no_go_count is None:
        no_go_count = crater_count
    seed = scene_seed(window_sha256, scenario_seed)
    rock_rng, crater_rng, no_go_rng = _stream(seed, "rocks"), _stream(seed, "crater-heights"), _stream(seed, "no-go-polygons")
    rocks = []
    for _ in range(rock_count):
        radius = float(rock_rng.uniform(0.15, 1.2))
        x, y = rock_rng.uniform(radius, geometry.size_m - radius, size=2)
        rocks.append(Point(float(x), float(y)).buffer(radius, quad_segs=12))
    craters = []
    delta = np.zeros((geometry.cells, geometry.cells), dtype=np.float32)
    yy, xx = np.meshgrid(
        (np.arange(geometry.cells, dtype=np.float32) + 0.5) * geometry.resolution_m,
        (np.arange(geometry.cells, dtype=np.float32) + 0.5) * geometry.resolution_m,
        indexing="ij",
    )
    for _ in range(crater_count):
        maximum_radius = min(16.0, geometry.size_m / 2.0 - 1e-6)
        if maximum_radius <= 0.15:
            raise ValueError("geometry is too small for procedural craters")
        radius = float(crater_rng.uniform(min(2.0, maximum_radius), maximum_radius))
        x, y = crater_rng.uniform(radius, geometry.size_m - radius, size=2)
        depth = float(crater_rng.uniform(0.05, 1.0))
        distance = np.hypot(xx - x, yy - y)
        delta -= (depth * np.clip(1.0 - distance / radius, 0.0, 1.0)).astype(np.float32)
        craters.append(Point(float(x), float(y)).buffer(radius, quad_segs=12))
    no_go_polygons = []
    for _ in range(no_go_count):
        radius = float(no_go_rng.uniform(0.25, min(8.0, geometry.size_m / 2.0 - 1e-6)))
        x, y = no_go_rng.uniform(radius, geometry.size_m - radius, size=2)
        no_go_polygons.append(Point(float(x), float(y)).buffer(radius, quad_segs=8))
    ratio = physical_obstacle_ratio(tuple(rocks), geometry)
    return HazardScene(
        seed=seed,
        rock_footprints_wkb=tuple(to_wkb(item) for item in rocks),
        crater_polygons_wkb=tuple(to_wkb(item) for item in craters),
        no_go_polygons_wkb=tuple(to_wkb(item) for item in no_go_polygons),
        crater_elevation_delta_m=delta,
        physical_obstacle_ratio=ratio,
    )


__all__ = ["GENERATOR_VERSION", "HazardScene", "generate_hazard_scene", "physical_obstacle_ratio", "scene_seed"]
