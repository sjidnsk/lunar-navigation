from __future__ import annotations

import numpy as np

from lunar_policy_training.environment import platform_reachability
from lunar_policy_training.polar_data.raster import GridGeometry, MapCanvas


def test_ground_point_goal_feasibility_matches_cell_box_tolerance() -> None:
    canvas = MapCanvas(
        "e" * 64,
        (0.0, 0.0, 0.8, 0.8),
        GridGeometry(size_m=0.8, resolution_m=0.2, cells=4),
    )
    hard = np.zeros((canvas.geometry.cells, canvas.geometry.cells), dtype=np.bool_)
    hard[1, 2] = True
    target = np.asarray(
        [[0.36, 0.5, 0.0], [0.1, 0.1, 0.0], [2.0, 2.0, 0.0]],
        dtype=np.float64,
    )

    covered, feasible = platform_reachability.ground_point_goal_feasibility(
        hard,
        canvas=canvas,
        target_positions_m=target,
        tolerance_m=0.05,
    )

    assert covered.tolist() == [True, True, False]
    assert feasible.tolist() == [True, False, False]
