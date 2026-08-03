from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import hypot

import numpy as np


@dataclass(frozen=True)
class PlanningResult:
    reachable: bool
    path: tuple[tuple[int, int], ...]
    total_cost: float
    expanded_nodes: int


def _neighbors(x: int, y: int, width: int, height: int, allow_diagonal: bool) -> tuple[tuple[int, int], ...]:
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    result: list[tuple[int, int]] = []
    for dx, dy in offsets:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            result.append((nx, ny))
    return tuple(result)


def astar_path(
    cost: np.ndarray,
    passable_mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    resolution: float,
    allow_diagonal: bool = True,
) -> PlanningResult:
    """在非负栅格代价图层上运行 A*。"""

    cost_array = np.asarray(cost, dtype=float)
    passable = passable_mask.astype(bool, copy=False)
    height, width = cost_array.shape
    sx, sy = start
    gx, gy = goal

    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if np.nanmin(cost_array) < 0.0:
        raise ValueError("A* requires nonnegative costs")
    if not (0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height):
        raise ValueError("start and goal must be inside the grid")
    if not passable[sy, sx] or not passable[gy, gx]:
        return PlanningResult(False, tuple(), float("inf"), 0)

    frontier: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    expanded = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        expanded += 1
        if current == goal:
            break

        cx, cy = current
        for nx, ny in _neighbors(cx, cy, width, height, allow_diagonal):
            if not passable[ny, nx]:
                continue
            step_length = hypot(nx - cx, ny - cy) * resolution
            step_cost = step_length * max(float(cost_array[ny, nx]), 0.0)
            tentative = g_score[current] + step_cost
            neighbor = (nx, ny)
            if tentative < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative
                heuristic = hypot(gx - nx, gy - ny) * resolution
                heapq.heappush(frontier, (tentative + heuristic, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return PlanningResult(False, tuple(), float("inf"), expanded)

    path: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = came_from[cursor]
    path.reverse()
    return PlanningResult(True, tuple(path), float(g_score[goal]), expanded)
