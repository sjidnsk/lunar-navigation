"""Deterministic procedural rocks and craters for a source-bound polar window."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from shapely import from_wkb, to_wkb
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .raster import GLOBAL_GEOMETRY, GridGeometry, MapCanvas


GENERATOR_VERSION = "lunar-polar-hazards/v1"
FORMAL_GENERATOR_VERSION = "lunar-polar-multires-hazards/v3"
FORMAL_ROCK_COUNT = 260
FORMAL_CRATER_COUNT = 32
FORMAL_NO_GO_COUNT = 8


def formal_hazard_distribution() -> dict[str, object]:
    """Return a fresh canonical document of the frozen v2 distribution."""
    return {
        "rocks": {
            "count": FORMAL_ROCK_COUNT,
            "radius_m": [0.15, 1.2],
            "height_to_radius": [0.6, 1.5],
        },
        "craters": {
            "count": FORMAL_CRATER_COUNT,
            "radius_m": [2.0, 16.0],
            "depth_m": [0.05, 1.0],
        },
        "no_go_polygons": {
            "count": FORMAL_NO_GO_COUNT,
            "circumradius_m": [2.0, 8.0],
            "vertices": 6,
        },
    }


@dataclass(frozen=True)
class RockCircle:
    """Resolution-independent circular rock footprint in map metres."""

    x_m: float
    y_m: float
    radius_m: float
    height_m: float


@dataclass(frozen=True)
class CraterBowl:
    """Resolution-independent conical crater depression."""

    x_m: float
    y_m: float
    radius_m: float
    depth_m: float


@dataclass(frozen=True)
class NoGoPolygon:
    """Resolution-independent convex no-go polygon."""

    vertices_m: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class VectorHazardScene:
    """One formal hazard definition shared by every raster resolution."""

    seed: str
    canvas: MapCanvas
    rocks: tuple[RockCircle, ...]
    craters: tuple[CraterBowl, ...]
    no_go_polygons: tuple[NoGoPolygon, ...]
    generator_version: str = FORMAL_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if self.generator_version != FORMAL_GENERATOR_VERSION:
            raise ValueError("formal vector scene generator version is unsupported")
        if len(self.seed) != 64:
            raise ValueError("formal vector scene seed must be a SHA-256")

    @property
    def vector_sha256(self) -> str:
        """Content digest of the overlay, distinct from a catalogue scene ID."""
        payload = {
            "canvas_identity": self.canvas.identity,
            "seed": self.seed,
            "generator_version": self.generator_version,
            "rocks": [vars(value) for value in self.rocks],
            "craters": [vars(value) for value in self.craters],
            "no_go_polygons": [value.vertices_m for value in self.no_go_polygons],
        }
        return sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class CanvasRatioLayer:
    """A finite ratio raster bound to its absolute map canvas."""

    canvas: MapCanvas
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        cells = self.canvas.geometry.cells
        if values.shape != (cells, cells):
            raise ValueError(f"canvas ratio layer must have shape [{cells},{cells}]")
        if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
            raise ValueError("canvas ratio layer must be finite and in [0,1]")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class HazardScene:
    """Generated hazards separated into terrain truth and physical obstacles."""

    seed: str
    canvas: MapCanvas
    rock_footprints_wkb: tuple[bytes, ...]
    crater_polygons_wkb: tuple[bytes, ...]
    no_go_polygons_wkb: tuple[bytes, ...]
    crater_elevation_delta_m: np.ndarray
    physical_obstacle_layer: CanvasRatioLayer

    def __post_init__(self) -> None:
        if not isinstance(self.physical_obstacle_layer, CanvasRatioLayer):
            raise ValueError("hazard physical obstacle input must be a canvas-bound ratio layer")
        if self.physical_obstacle_layer.canvas.identity != self.canvas.identity:
            raise ValueError("hazard obstacle layer canvas identity does not match scene")

    @property
    def rock_footprints(self) -> tuple[object, ...]:
        return tuple(from_wkb(value) for value in self.rock_footprints_wkb)


def scene_seed(window_sha256: str, scenario_seed: int, *, generator_version: str = GENERATOR_VERSION) -> str:
    """Return the required identity binding for a generated scene."""
    if len(window_sha256) != 64 or any(character not in "0123456789abcdef" for character in window_sha256):
        raise ValueError("window_sha256 must be exactly 64 lowercase hexadecimal characters")
    return sha256(f"{window_sha256}:{scenario_seed}:{generator_version}".encode("utf-8")).hexdigest()


def _stream(seed: str, name: str) -> np.random.Generator:
    value = int.from_bytes(sha256(f"{seed}:{name}".encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(value)


def generate_vector_hazard_scene(
    window_sha256: str,
    scenario_seed: int,
    *,
    canvas: MapCanvas,
    rock_count: int = FORMAL_ROCK_COUNT,
    crater_count: int = FORMAL_CRATER_COUNT,
    no_go_count: int = FORMAL_NO_GO_COUNT,
) -> VectorHazardScene:
    """Generate the formal v2 vector scene using three independent RNG streams."""
    if canvas.window_sha256 != window_sha256:
        raise ValueError("formal vector scene canvas identity mismatch")
    if min(rock_count, crater_count, no_go_count) < 0:
        raise ValueError("hazard counts must be non-negative")
    seed = scene_seed(
        window_sha256,
        scenario_seed,
        generator_version=FORMAL_GENERATOR_VERSION,
    )
    left, bottom, right, top = canvas.bounds_m
    size_m = min(right - left, top - bottom)

    def bounded_radius(
        rng: np.random.Generator, minimum: float, maximum: float
    ) -> float:
        upper = min(maximum, size_m / 2.0 - 1e-6)
        if upper <= 0.0:
            raise ValueError("canvas is too small for formal hazards")
        lower = min(minimum, upper)
        return float(rng.uniform(lower, upper)) if lower < upper else upper

    rock_rng = _stream(seed, "rocks")
    rocks: list[RockCircle] = []
    for _ in range(rock_count):
        radius = bounded_radius(rock_rng, 0.15, 1.2)
        height = radius * float(rock_rng.uniform(0.6, 1.5))
        rocks.append(
            RockCircle(
                x_m=float(rock_rng.uniform(left + radius, right - radius)),
                y_m=float(rock_rng.uniform(bottom + radius, top - radius)),
                radius_m=radius,
                height_m=height,
            )
        )

    crater_rng = _stream(seed, "craters")
    craters: list[CraterBowl] = []
    for _ in range(crater_count):
        radius = bounded_radius(crater_rng, 2.0, 16.0)
        craters.append(
            CraterBowl(
                x_m=float(crater_rng.uniform(left + radius, right - radius)),
                y_m=float(crater_rng.uniform(bottom + radius, top - radius)),
                radius_m=radius,
                depth_m=float(crater_rng.uniform(0.05, 1.0)),
            )
        )

    no_go_rng = _stream(seed, "no-go-polygons")
    no_go_polygons: list[NoGoPolygon] = []
    for _ in range(no_go_count):
        radius = bounded_radius(no_go_rng, 2.0, 8.0)
        center_x = float(no_go_rng.uniform(left + radius, right - radius))
        center_y = float(no_go_rng.uniform(bottom + radius, top - radius))
        rotation = float(no_go_rng.uniform(-math.pi, math.pi))
        aspect = float(no_go_rng.uniform(0.55, 1.0))
        vertices = tuple(
            (
                center_x + radius * math.cos(rotation + index * math.pi / 3.0),
                center_y
                + radius * aspect * math.sin(rotation + index * math.pi / 3.0),
            )
            for index in range(6)
        )
        no_go_polygons.append(NoGoPolygon(vertices))

    return VectorHazardScene(
        seed=seed,
        canvas=canvas,
        rocks=tuple(rocks),
        craters=tuple(craters),
        no_go_polygons=tuple(no_go_polygons),
    )


def physical_obstacle_ratio(footprints: tuple[object, ...], geometry: GridGeometry = GLOBAL_GEOMETRY, *, canvas: MapCanvas | None = None) -> np.ndarray:
    """Rasterize the union of rock footprints by exact polygon-cell area."""
    result = np.zeros((geometry.cells, geometry.cells), dtype=np.float32)
    if not footprints:
        return result
    union = unary_union(footprints)
    cell_area = geometry.resolution_m * geometry.resolution_m
    for row in range(geometry.cells):
        y0 = (canvas.bounds_m[3] - (row + 1) * geometry.resolution_m) if canvas else row * geometry.resolution_m
        for column in range(geometry.cells):
            x0 = (canvas.bounds_m[0] + column * geometry.resolution_m) if canvas else column * geometry.resolution_m
            cell = box(x0, y0, x0 + geometry.resolution_m, y0 + geometry.resolution_m)
            if union.intersects(cell):
                result[row, column] = np.float32(min(1.0, union.intersection(cell).area / cell_area))
    return result


def generate_hazard_scene(window_sha256: str, scenario_seed: int, *, canvas: MapCanvas, geometry: GridGeometry = GLOBAL_GEOMETRY, rock_count: int = 32, crater_count: int = 8, no_go_count: int | None = None) -> HazardScene:
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
    if canvas.window_sha256 != window_sha256 or canvas.geometry != geometry:
        raise ValueError("hazard canvas identity/geometry mismatch")
    left, bottom, _, top = canvas.bounds_m
    rock_rng, crater_rng, no_go_rng = _stream(seed, "rocks"), _stream(seed, "crater-heights"), _stream(seed, "no-go-polygons")
    rocks = []
    for _ in range(rock_count):
        radius = float(rock_rng.uniform(0.15, 1.2))
        x, y = rock_rng.uniform(radius, geometry.size_m - radius, size=2)
        rocks.append(Point(left + float(x), bottom + float(y)).buffer(radius, quad_segs=12))
    craters = []
    delta = np.zeros((geometry.cells, geometry.cells), dtype=np.float32)
    yy, xx = np.meshgrid(
        canvas.bounds_m[3] - (np.arange(geometry.cells, dtype=np.float32) + 0.5) * geometry.resolution_m,
        canvas.bounds_m[0] + (np.arange(geometry.cells, dtype=np.float32) + 0.5) * geometry.resolution_m,
        indexing="ij",
    )
    for _ in range(crater_count):
        maximum_radius = min(16.0, geometry.size_m / 2.0 - 1e-6)
        if maximum_radius <= 0.15:
            raise ValueError("geometry is too small for procedural craters")
        radius = float(crater_rng.uniform(min(2.0, maximum_radius), maximum_radius))
        x, y = crater_rng.uniform(radius, geometry.size_m - radius, size=2)
        depth = float(crater_rng.uniform(0.05, 1.0))
        distance = np.hypot(xx - (left + x), yy - (bottom + y))
        delta -= (depth * np.clip(1.0 - distance / radius, 0.0, 1.0)).astype(np.float32)
        craters.append(Point(left + float(x), bottom + float(y)).buffer(radius, quad_segs=12))
    no_go_polygons = []
    for _ in range(no_go_count):
        radius = float(no_go_rng.uniform(0.25, min(8.0, geometry.size_m / 2.0 - 1e-6)))
        x, y = no_go_rng.uniform(radius, geometry.size_m - radius, size=2)
        no_go_polygons.append(Point(left + float(x), bottom + float(y)).buffer(radius, quad_segs=8))
    ratio = physical_obstacle_ratio(tuple(rocks), geometry, canvas=canvas)
    return HazardScene(
        seed=seed,
        canvas=canvas,
        rock_footprints_wkb=tuple(to_wkb(item) for item in rocks),
        crater_polygons_wkb=tuple(to_wkb(item) for item in craters),
        no_go_polygons_wkb=tuple(to_wkb(item) for item in no_go_polygons),
        crater_elevation_delta_m=delta,
        physical_obstacle_layer=CanvasRatioLayer(canvas, ratio),
    )


__all__ = [
    "CanvasRatioLayer",
    "CraterBowl",
    "FORMAL_GENERATOR_VERSION",
    "FORMAL_CRATER_COUNT",
    "FORMAL_NO_GO_COUNT",
    "FORMAL_ROCK_COUNT",
    "GENERATOR_VERSION",
    "HazardScene",
    "NoGoPolygon",
    "RockCircle",
    "VectorHazardScene",
    "generate_hazard_scene",
    "generate_vector_hazard_scene",
    "formal_hazard_distribution",
    "physical_obstacle_ratio",
    "scene_seed",
]
