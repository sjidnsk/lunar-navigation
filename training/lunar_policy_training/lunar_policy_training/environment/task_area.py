"""Deterministic bounded task views over immutable formal scene evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from typing import Any

import numpy as np

from ..config import TaskAreaConfig
from ..polar_data.raster import GLOBAL_GEOMETRY
from .coverability import mask_sha256, pack_detail_mask, unpack_detail_mask
from .formal_start_qualification import build_formal_mission_roi


DETAIL_PER_GLOBAL = 20
_PLATFORMS = frozenset(("WHEELED", "LEGGED", "HOPPER"))


@dataclass(frozen=True, slots=True)
class ScopedTaskArea:
    span_cells: int
    size_m: float
    coarse_bounds_half_open: tuple[int, int, int, int]
    detail_bounds_half_open: tuple[int, int, int, int]
    coverable_detail_cell_count: int
    coverable_detail_mask_sha256: str


def sample_task_area_span_cells(
    config: TaskAreaConfig, episode_seed: str
) -> int:
    """Sample one unbiased 4 m-aligned square side from a replayable seed."""
    if not isinstance(config, TaskAreaConfig):
        raise TypeError("task area sampling requires TaskAreaConfig")
    if (
        not isinstance(episode_seed, str)
        or len(episode_seed) != 64
        or any(character not in "0123456789abcdef" for character in episode_seed)
    ):
        raise ValueError("task area episode seed must be a lowercase SHA-256")
    resolution_m = float(GLOBAL_GEOMETRY.resolution_m)
    minimum = round(config.minimum_size_m / resolution_m)
    maximum = round(config.maximum_size_m / resolution_m)
    if (
        minimum * resolution_m != config.minimum_size_m
        or maximum * resolution_m != config.maximum_size_m
        or not 0 < minimum <= maximum <= GLOBAL_GEOMETRY.cells
    ):
        raise ValueError("task area range is not aligned to the global grid")
    count = maximum - minimum + 1
    modulus = 1 << 256
    acceptance_limit = modulus - modulus % count
    counter = 0
    while True:
        payload = (
            f"{config.sampling_algorithm}\0{episode_seed}\0{counter}"
        ).encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        if value < acceptance_limit:
            return minimum + value % count
        counter += 1


def task_area_bounds(
    start_cell: tuple[int, int], span_cells: int
) -> tuple[int, int, int, int]:
    """Center an intact square where possible and translate it at map edges."""
    if (
        not isinstance(start_cell, tuple)
        or len(start_cell) != 2
        or any(type(value) is not int for value in start_cell)
    ):
        raise TypeError("task area start cell must be an integer pair")
    if type(span_cells) is not int or not 0 < span_cells <= GLOBAL_GEOMETRY.cells:
        raise ValueError("task area span is invalid")

    def axis_bounds(center: int) -> tuple[int, int]:
        if not 0 <= center < GLOBAL_GEOMETRY.cells:
            raise ValueError("task area start leaves the global grid")
        lower = center - span_cells // 2
        lower = min(max(lower, 0), GLOBAL_GEOMETRY.cells - span_cells)
        return lower, lower + span_cells

    row0, row1 = axis_bounds(start_cell[0])
    column0, column1 = axis_bounds(start_cell[1])
    return row0, row1, column0, column1


def scope_formal_task_area(
    loaded: Any,
    *,
    platform_type: str,
    start_cell: tuple[int, int],
    config: TaskAreaConfig,
    episode_seed: str,
) -> tuple[Any, ScopedTaskArea]:
    """Return a detached task view without changing cached physical evidence."""
    if platform_type not in _PLATFORMS:
        raise ValueError("task area platform is invalid")
    span_cells = sample_task_area_span_cells(config, episode_seed)
    row0, row1, column0, column1 = task_area_bounds(start_cell, span_cells)
    original_roi = build_formal_mission_roi(loaded.arrays)
    coarse_scope = np.zeros_like(original_roi, dtype=np.bool_)
    coarse_scope[row0:row1, column0:column1] = True
    scoped_roi = np.ascontiguousarray(original_roi & coarse_scope)
    if not bool(scoped_roi[start_cell]):
        raise ValueError("task area excludes its qualified start")

    entry = deepcopy(dict(loaded.entry))
    coverability = entry["platform_coverability"][platform_type]
    shape_raw = coverability["coverable_detail_shape"]
    if not isinstance(shape_raw, list) or len(shape_raw) != 2:
        raise ValueError("task area coverability shape is invalid")
    detail_shape = int(shape_raw[0]), int(shape_raw[1])
    expected = GLOBAL_GEOMETRY.cells * DETAIL_PER_GLOBAL
    if detail_shape != (expected, expected):
        raise ValueError("task area detail geometry is incompatible")
    prefix = platform_type.lower()
    original_coverable = unpack_detail_mask(
        np.ascontiguousarray(
            loaded.arrays[f"{prefix}_coverable_detail_bits"], dtype=np.uint8
        ),
        detail_shape,
    )
    detail_row0 = row0 * DETAIL_PER_GLOBAL
    detail_row1 = row1 * DETAIL_PER_GLOBAL
    detail_column0 = column0 * DETAIL_PER_GLOBAL
    detail_column1 = column1 * DETAIL_PER_GLOBAL
    detail_scope = np.zeros(detail_shape, dtype=np.bool_)
    detail_scope[
        detail_row0:detail_row1, detail_column0:detail_column1
    ] = True
    scoped_coverable = np.ascontiguousarray(original_coverable & detail_scope)
    scoped_count = int(scoped_coverable.sum(dtype=np.int64))
    if scoped_count <= 0:
        raise ValueError("task area contains no coverable detail cells")
    scoped_sha256 = mask_sha256(scoped_coverable)

    arrays = dict(loaded.arrays)
    arrays["scoped_mission_roi"] = scoped_roi
    arrays[f"{prefix}_coverable_detail_bits"] = pack_detail_mask(
        scoped_coverable
    )
    scoped_ratio = scoped_coverable.reshape(
        GLOBAL_GEOMETRY.cells,
        DETAIL_PER_GLOBAL,
        GLOBAL_GEOMETRY.cells,
        DETAIL_PER_GLOBAL,
    ).mean(axis=(1, 3), dtype=np.float32)
    arrays[f"{prefix}_coverable_ratio"] = np.ascontiguousarray(
        scoped_ratio, dtype=np.float32
    )
    coverability["coverable_detail_cell_count"] = scoped_count
    coverability["coverable_detail_mask_sha256"] = scoped_sha256
    coverability["mission_target_detail_cell_count"] = scoped_count
    coverability["mission_target_detail_mask_sha256"] = scoped_sha256
    coverability["mission_coverable_fraction"] = 1.0
    task = ScopedTaskArea(
        span_cells=span_cells,
        size_m=span_cells * float(GLOBAL_GEOMETRY.resolution_m),
        coarse_bounds_half_open=(row0, row1, column0, column1),
        detail_bounds_half_open=(
            detail_row0,
            detail_row1,
            detail_column0,
            detail_column1,
        ),
        coverable_detail_cell_count=scoped_count,
        coverable_detail_mask_sha256=scoped_sha256,
    )
    entry["task_area"] = {
        "sampling_algorithm": config.sampling_algorithm,
        "episode_seed": episode_seed,
        "span_cells": task.span_cells,
        "size_m": task.size_m,
        "coarse_bounds_half_open": list(task.coarse_bounds_half_open),
        "detail_bounds_half_open": list(task.detail_bounds_half_open),
        "coverable_detail_cell_count": task.coverable_detail_cell_count,
        "coverable_detail_mask_sha256": task.coverable_detail_mask_sha256,
    }
    return replace(loaded, arrays=arrays, entry=entry), task


__all__ = [
    "ScopedTaskArea",
    "sample_task_area_span_cells",
    "scope_formal_task_area",
    "task_area_bounds",
]
