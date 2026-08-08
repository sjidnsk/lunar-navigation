"""Deterministic multi-resolution projections of one formal vector scene."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from shapely import area as geometry_area
from shapely import box as geometry_box
from shapely import intersection
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .hazards import FORMAL_GENERATOR_VERSION, VectorHazardScene
from .raster import LOCAL_TILE_GEOMETRY, GridGeometry, MapCanvas


LOCAL_DETAIL_PROVENANCE = "synthetic_subgrid_on_locked_dem"
GENERATOR_SHA256 = sha256(FORMAL_GENERATOR_VERSION.encode("utf-8")).hexdigest()


def _readonly_grid(name: str, value: np.ndarray, cells: int, dtype: object) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if array.shape != (cells, cells):
        raise ValueError(f"{name} must have shape [{cells},{cells}]")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ProjectedHazards:
    vector_sha256: str
    canvas: MapCanvas
    crater_elevation_delta_m: np.ndarray
    physical_obstacle_ratio: np.ndarray
    physical_obstacle_height_m: np.ndarray
    forbidden_ratio: np.ndarray
    local_detail_provenance: str = LOCAL_DETAIL_PROVENANCE

    def __post_init__(self) -> None:
        cells = self.canvas.geometry.cells
        object.__setattr__(
            self,
            "crater_elevation_delta_m",
            _readonly_grid(
                "crater_elevation_delta_m",
                self.crater_elevation_delta_m,
                cells,
                np.float32,
            ),
        )
        for name in ("physical_obstacle_ratio", "forbidden_ratio"):
            values = _readonly_grid(name, getattr(self, name), cells, np.float32)
            if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
                raise ValueError(f"{name} must be finite and in [0,1]")
            object.__setattr__(self, name, values)
        obstacle_height = _readonly_grid(
            "physical_obstacle_height_m",
            self.physical_obstacle_height_m,
            cells,
            np.float32,
        )
        if not np.isfinite(obstacle_height).all() or (obstacle_height < 0.0).any():
            raise ValueError("physical_obstacle_height_m must be finite and non-negative")
        object.__setattr__(self, "physical_obstacle_height_m", obstacle_height)
        if self.local_detail_provenance != LOCAL_DETAIL_PROVENANCE:
            raise ValueError("local detail provenance is unsupported")


@dataclass(frozen=True)
class ProjectedScene(ProjectedHazards):
    elevation_m: np.ndarray | None = None
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        cells = self.canvas.geometry.cells
        if self.elevation_m is None or self.valid_mask is None:
            raise ValueError("projected scene requires elevation and validity truth")
        elevation = _readonly_grid("elevation_m", self.elevation_m, cells, np.float32)
        valid = _readonly_grid("valid_mask", self.valid_mask, cells, np.bool_)
        if not np.isfinite(elevation[valid]).all():
            raise ValueError("projected scene NoData cannot be valid")
        object.__setattr__(self, "elevation_m", elevation)
        object.__setattr__(self, "valid_mask", valid)


def _aligned_axis(offset_m: float, resolution_m: float) -> int:
    index = round(offset_m / resolution_m)
    if not math.isclose(
        offset_m,
        index * resolution_m,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("projection grid must be aligned to the scene origin")
    return index


def _cell_coordinates(
    scene_canvas: MapCanvas, target_canvas: MapCanvas
) -> tuple[np.ndarray, np.ndarray]:
    if target_canvas.window_sha256 != scene_canvas.window_sha256:
        raise ValueError("projection target window identity differs from vector scene")
    left, bottom, right, top = target_canvas.bounds_m
    scene_left, scene_bottom, scene_right, scene_top = scene_canvas.bounds_m
    if (
        left < scene_left
        or bottom < scene_bottom
        or right > scene_right
        or top > scene_top
    ):
        raise ValueError("projection target lies outside vector scene")
    resolution = target_canvas.geometry.resolution_m
    start_column = _aligned_axis(left - scene_left, resolution)
    start_row = _aligned_axis(scene_top - top, resolution)
    _aligned_axis(right - scene_left, resolution)
    _aligned_axis(scene_top - bottom, resolution)
    columns = start_column + np.arange(target_canvas.geometry.cells, dtype=np.int64)
    rows = start_row + np.arange(target_canvas.geometry.cells, dtype=np.int64)
    x = scene_left + (columns.astype(np.float64) + 0.5) * resolution
    y = scene_top - (rows.astype(np.float64) + 0.5) * resolution
    return x, y


def _exact_area_ratio(
    geometries: tuple[object, ...],
    x: np.ndarray,
    y: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    result = np.zeros((y.size, x.size), dtype=np.float64)
    if not geometries:
        return result.astype(np.float32)
    union = unary_union(geometries)
    half = resolution_m / 2.0
    components = tuple(union.geoms) if hasattr(union, "geoms") else (union,)
    for component in components:
        minimum_x, minimum_y, maximum_x, maximum_y = component.bounds
        columns = np.flatnonzero(
            (x + half > minimum_x) & (x - half < maximum_x)
        )
        rows = np.flatnonzero(
            (y + half > minimum_y) & (y - half < maximum_y)
        )
        if rows.size == 0 or columns.size == 0:
            continue
        cells = geometry_box(
            x[columns][None, :] - half,
            y[rows][:, None] - half,
            x[columns][None, :] + half,
            y[rows][:, None] + half,
        )
        values = np.asarray(
            geometry_area(intersection(cells, component)), dtype=np.float64
        )
        result[np.ix_(rows, columns)] += values / (
            resolution_m * resolution_m
        )
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def project_vector_scene(
    scene: VectorHazardScene,
    target_canvas: MapCanvas,
) -> ProjectedHazards:
    """Project one immutable vector scene without drawing new random values."""
    x, y = _cell_coordinates(scene.canvas, target_canvas)
    xx, yy = np.meshgrid(x, y)
    delta = np.zeros(xx.shape, dtype=np.float64)
    target_left, target_bottom, target_right, target_top = target_canvas.bounds_m
    for crater in scene.craters:
        if (
            crater.x_m + crater.radius_m <= target_left
            or crater.x_m - crater.radius_m >= target_right
            or crater.y_m + crater.radius_m <= target_bottom
            or crater.y_m - crater.radius_m >= target_top
        ):
            continue
        distance = np.hypot(xx - crater.x_m, yy - crater.y_m)
        delta -= crater.depth_m * np.clip(
            1.0 - distance / crater.radius_m,
            0.0,
            1.0,
        )
    relevant_rocks = tuple(
        rock
        for rock in scene.rocks
        if not (
            rock.x_m + rock.radius_m <= target_left
            or rock.x_m - rock.radius_m >= target_right
            or rock.y_m + rock.radius_m <= target_bottom
            or rock.y_m - rock.radius_m >= target_top
        )
    )
    rocks = tuple(
        Point(rock.x_m, rock.y_m).buffer(rock.radius_m, quad_segs=24)
        for rock in relevant_rocks
    )
    obstacle_height = np.zeros(xx.shape, dtype=np.float32)
    half = target_canvas.geometry.resolution_m / 2.0
    for rock in relevant_rocks:
        dx = np.maximum(np.abs(x - rock.x_m) - half, 0.0)
        dy = np.maximum(np.abs(y - rock.y_m) - half, 0.0)
        intersects = (
            dy[:, None] * dy[:, None] + dx[None, :] * dx[None, :]
            <= rock.radius_m * rock.radius_m
        )
        obstacle_height[intersects] = np.maximum(
            obstacle_height[intersects], np.float32(rock.height_m)
        )
    no_go = tuple(
        polygon
        for value in scene.no_go_polygons
        for polygon in (Polygon(value.vertices_m),)
        if not (
            polygon.bounds[2] <= target_left
            or polygon.bounds[0] >= target_right
            or polygon.bounds[3] <= target_bottom
            or polygon.bounds[1] >= target_top
        )
    )
    resolution = target_canvas.geometry.resolution_m
    return ProjectedHazards(
        vector_sha256=scene.vector_sha256,
        canvas=target_canvas,
        crater_elevation_delta_m=delta.astype(np.float32),
        physical_obstacle_ratio=_exact_area_ratio(rocks, x, y, resolution),
        physical_obstacle_height_m=obstacle_height,
        forbidden_ratio=_exact_area_ratio(no_go, x, y, resolution),
    )


@dataclass(frozen=True)
class MultiResolutionScene:
    """Locked coarse DEM plus a shared synthetic vector overlay."""

    base_canvas: MapCanvas
    base_elevation_m: np.ndarray
    base_valid_mask: np.ndarray
    hazards: VectorHazardScene
    scenario_id: str | None = None

    def __post_init__(self) -> None:
        cells = self.base_canvas.geometry.cells
        elevation = _readonly_grid(
            "base_elevation_m", self.base_elevation_m, cells, np.float32
        )
        valid = _readonly_grid("base_valid_mask", self.base_valid_mask, cells, np.bool_)
        if not np.isfinite(elevation[valid]).all():
            raise ValueError("base DEM NoData cannot be valid")
        if self.hazards.canvas.identity != self.base_canvas.identity:
            raise ValueError("base DEM and vector hazards must share one canvas")
        if self.scenario_id is not None and (
            len(self.scenario_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.scenario_id
            )
        ):
            raise ValueError("catalogue scenario ID must be a lowercase SHA-256")
        object.__setattr__(self, "base_elevation_m", elevation)
        object.__setattr__(self, "base_valid_mask", valid)

    @property
    def scene_id(self) -> str:
        return self.scenario_id or self.hazards.vector_sha256

    def _sample_base(
        self, target_canvas: MapCanvas
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = _cell_coordinates(self.base_canvas, target_canvas)
        left, _, _, top = self.base_canvas.bounds_m
        resolution = self.base_canvas.geometry.resolution_m
        column = np.clip(
            (x - left) / resolution - 0.5,
            0.0,
            self.base_canvas.geometry.cells - 1.0,
        )
        row = np.clip(
            (top - y) / resolution - 0.5,
            0.0,
            self.base_canvas.geometry.cells - 1.0,
        )
        column0 = np.floor(column).astype(np.intp)
        row0 = np.floor(row).astype(np.intp)
        column1 = np.minimum(column0 + 1, self.base_canvas.geometry.cells - 1)
        row1 = np.minimum(row0 + 1, self.base_canvas.geometry.cells - 1)
        wx = (column - column0).astype(np.float64)[None, :]
        wy = (row - row0).astype(np.float64)[:, None]
        source = np.where(np.isfinite(self.base_elevation_m), self.base_elevation_m, 0.0)
        north = (
            source[row0[:, None], column0[None, :]] * (1.0 - wx)
            + source[row0[:, None], column1[None, :]] * wx
        )
        south = (
            source[row1[:, None], column0[None, :]] * (1.0 - wx)
            + source[row1[:, None], column1[None, :]] * wx
        )
        elevation = north * (1.0 - wy) + south * wy
        valid = (
            self.base_valid_mask[row0[:, None], column0[None, :]]
            & self.base_valid_mask[row0[:, None], column1[None, :]]
            & self.base_valid_mask[row1[:, None], column0[None, :]]
            & self.base_valid_mask[row1[:, None], column1[None, :]]
        )
        return elevation.astype(np.float32), valid

    def project(self, target_canvas: MapCanvas) -> ProjectedScene:
        hazards = project_vector_scene(self.hazards, target_canvas)
        base, valid = self._sample_base(target_canvas)
        elevation = base + hazards.crater_elevation_delta_m
        elevation[~valid] = np.nan
        return ProjectedScene(
            vector_sha256=hazards.vector_sha256,
            canvas=target_canvas,
            crater_elevation_delta_m=hazards.crater_elevation_delta_m,
            physical_obstacle_ratio=hazards.physical_obstacle_ratio,
            physical_obstacle_height_m=hazards.physical_obstacle_height_m,
            forbidden_ratio=hazards.forbidden_ratio,
            elevation_m=elevation,
            valid_mask=valid,
        )


class SceneTileProvider:
    """Bounded content-addressed LRU of fixed-coordinate detail tiles."""

    def __init__(
        self,
        scene: MultiResolutionScene,
        *,
        tile_geometry: GridGeometry = LOCAL_TILE_GEOMETRY,
        capacity: int = 8,
    ) -> None:
        if capacity < 1:
            raise ValueError("tile cache capacity must be positive")
        tiles_per_axis = scene.base_canvas.geometry.size_m / tile_geometry.size_m
        if not math.isclose(tiles_per_axis, round(tiles_per_axis)):
            raise ValueError("tile size must divide the scene canvas")
        _aligned_axis(tile_geometry.size_m, tile_geometry.resolution_m)
        self.scene = scene
        self.tile_geometry = tile_geometry
        self.capacity = capacity
        self.tiles_per_axis = int(round(tiles_per_axis))
        self._cache: OrderedDict[tuple[str, int, int, str], ProjectedScene] = (
            OrderedDict()
        )

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def detail_cells_per_axis(self) -> int:
        return self.tiles_per_axis * self.tile_geometry.cells

    def world_to_detail(self, x_m: float, y_m: float) -> tuple[int, int]:
        left, bottom, right, top = self.scene.base_canvas.bounds_m
        if not (left <= x_m < right and bottom < y_m <= top):
            raise ValueError("detail world point lies outside the scene")
        resolution = self.tile_geometry.resolution_m
        return (
            int(math.floor((top - y_m) / resolution)),
            int(math.floor((x_m - left) / resolution)),
        )

    def tile(self, tile_row: int, tile_column: int) -> ProjectedScene:
        if not (
            0 <= tile_row < self.tiles_per_axis
            and 0 <= tile_column < self.tiles_per_axis
        ):
            raise ValueError("tile index lies outside the scene")
        key = (self.scene.scene_id, tile_row, tile_column, GENERATOR_SHA256)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        left, _, _, top = self.scene.base_canvas.bounds_m
        size = self.tile_geometry.size_m
        tile_left = left + tile_column * size
        tile_top = top - tile_row * size
        canvas = MapCanvas(
            self.scene.base_canvas.window_sha256,
            (tile_left, tile_top - size, tile_left + size, tile_top),
            self.tile_geometry,
        )
        projected = self.scene.project(canvas)
        self._cache[key] = projected
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return projected

    def read_window(
        self, start_row: int, start_column: int, *, cells: int
    ) -> ProjectedScene:
        """Compose an aligned square detail window from fixed cached tiles."""
        if (
            type(start_row) is not int
            or type(start_column) is not int
            or type(cells) is not int
            or cells < 1
            or start_row < 0
            or start_column < 0
            or start_row + cells > self.detail_cells_per_axis
            or start_column + cells > self.detail_cells_per_axis
        ):
            raise ValueError("detail window lies outside the scene")
        fields = (
            "crater_elevation_delta_m",
            "physical_obstacle_ratio",
            "physical_obstacle_height_m",
            "forbidden_ratio",
            "elevation_m",
        )
        arrays = {
            name: np.empty((cells, cells), dtype=np.float32) for name in fields
        }
        valid = np.empty((cells, cells), dtype=np.bool_)
        tile_cells = self.tile_geometry.cells
        first_tile_row = start_row // tile_cells
        last_tile_row = (start_row + cells - 1) // tile_cells
        first_tile_column = start_column // tile_cells
        last_tile_column = (start_column + cells - 1) // tile_cells
        for tile_row in range(first_tile_row, last_tile_row + 1):
            for tile_column in range(
                first_tile_column, last_tile_column + 1
            ):
                tile = self.tile(tile_row, tile_column)
                tile_start_row = tile_row * tile_cells
                tile_start_column = tile_column * tile_cells
                global_row0 = max(start_row, tile_start_row)
                global_row1 = min(start_row + cells, tile_start_row + tile_cells)
                global_column0 = max(start_column, tile_start_column)
                global_column1 = min(
                    start_column + cells, tile_start_column + tile_cells
                )
                destination = (
                    slice(global_row0 - start_row, global_row1 - start_row),
                    slice(
                        global_column0 - start_column,
                        global_column1 - start_column,
                    ),
                )
                source = (
                    slice(
                        global_row0 - tile_start_row,
                        global_row1 - tile_start_row,
                    ),
                    slice(
                        global_column0 - tile_start_column,
                        global_column1 - tile_start_column,
                    ),
                )
                for name in fields:
                    arrays[name][destination] = getattr(tile, name)[source]
                valid[destination] = tile.valid_mask[source]
        resolution = self.tile_geometry.resolution_m
        geometry = GridGeometry(cells * resolution, resolution, cells)
        left, _, _, top = self.scene.base_canvas.bounds_m
        window_left = left + start_column * resolution
        window_top = top - start_row * resolution
        canvas = MapCanvas(
            self.scene.base_canvas.window_sha256,
            (
                window_left,
                window_top - geometry.size_m,
                window_left + geometry.size_m,
                window_top,
            ),
            geometry,
        )
        return ProjectedScene(
            vector_sha256=self.scene.hazards.vector_sha256,
            canvas=canvas,
            valid_mask=valid,
            **arrays,
        )


__all__ = [
    "GENERATOR_SHA256",
    "LOCAL_DETAIL_PROVENANCE",
    "MultiResolutionScene",
    "ProjectedHazards",
    "ProjectedScene",
    "SceneTileProvider",
    "project_vector_scene",
]
