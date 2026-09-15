"""Measured geometry shared by deployment and offline tools; no truth imports."""

import math
import numpy as np


def world_to_cell(x, y, origin, resolution_m):
    """Floor on this raster's lattice (including negative/fractional origins)."""
    return (
        math.floor((x - origin[0]) / resolution_m),
        math.floor((y - origin[1]) / resolution_m),
    )


def cell_center(x, y, origin, resolution_m):
    return (origin[0] + (x + 0.5) * resolution_m, origin[1] + (y + 0.5) * resolution_m)


def polygon_mask(shape, origin, resolution_m, polygon):
    """Cell centers in a possibly concave polygon, boundary included.

    Call with TaskSpec.polygon so all users share the canonical float32 corners.
    Process rows in bounded chunks rather than allocating full XY rasters.
    """
    polygon = np.asarray(polygon)
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or len(polygon) < 3
        or not np.all(np.isfinite(polygon))
    ):
        raise ValueError("finite polygon with at least three corners required")
    h, w = shape
    result = np.zeros(shape, bool)
    xs = origin[0] + (np.arange(w, dtype=np.float64) + 0.5) * resolution_m
    for first in range(0, h, 128):
        ys = (
            origin[1]
            + (np.arange(first, min(h, first + 128), dtype=np.float64) + 0.5)
            * resolution_m
        )
        x = xs[None, :]
        y = ys[:, None]
        inside = np.zeros((len(ys), w), bool)
        boundary = inside.copy()
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            ax, ay = map(float, a)
            bx, by = map(float, b)
            dx, dy = bx - ax, by - ay
            cross = (x - ax) * dy - (y - ay) * dx
            boundary |= (
                (np.abs(cross) <= 1e-10 * max(1.0, math.hypot(dx, dy)))
                & (x >= min(ax, bx) - 1e-12)
                & (x <= max(ax, bx) + 1e-12)
                & (y >= min(ay, by) - 1e-12)
                & (y <= max(ay, by) + 1e-12)
            )
            if dy:
                inside ^= ((ay > y) != (by > y)) & (x < (bx - ax) * (y - ay) / dy + ax)
        result[first : first + len(ys)] = inside | boundary
    return result
