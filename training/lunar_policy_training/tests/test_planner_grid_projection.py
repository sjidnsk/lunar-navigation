from __future__ import annotations

import numpy as np

from lunar_policy_training.environment import formal_builder
from lunar_policy_training.polar_data.raster import GridGeometry, MapCanvas


def test_planner_grid_projection_reuses_fixed_geometry_indices() -> None:
    source = MapCanvas(
        "a" * 64,
        (0.0, 0.0, 12.0, 12.0),
        GridGeometry(size_m=12.0, resolution_m=4.0, cells=3),
    )
    target = MapCanvas(
        "b" * 64,
        (0.0, 0.0, 12.0, 12.0),
        GridGeometry(size_m=12.0, resolution_m=2.0, cells=6),
    )
    values = np.arange(9, dtype=np.float32).reshape(3, 3)

    projection = formal_builder._planner_grid_projection(source, target)
    first = formal_builder._project_task_grid_to_planner_level(
        values,
        source=source,
        target=target,
        fill_value=np.float32(0.0),
        projection=projection,
    )
    second = formal_builder._project_task_grid_to_planner_level(
        values + np.float32(10.0),
        source=source,
        target=target,
        fill_value=np.float32(0.0),
        projection=projection,
    )

    np.testing.assert_array_equal(
        first, np.repeat(np.repeat(values, 2, axis=0), 2, axis=1)
    )
    np.testing.assert_array_equal(second, first + np.float32(10.0))
    assert not projection.source_rows.flags.writeable
    assert not projection.source_columns.flags.writeable
