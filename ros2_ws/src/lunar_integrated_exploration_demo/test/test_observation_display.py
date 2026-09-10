"""Observed-cell overlays must preserve asymmetric source coordinates and NaNs."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lunar_incremental_controller_demo'))
pytest.importorskip('rclpy')
from builtin_interfaces.msg import Time
from lunar_integrated_exploration_demo.vehicle import observation_grid_map
from lunar_integrated_exploration_demo.visualizer import observed_world_points


def test_observed_overlay_preserves_world_cells_across_moving_windows():
    for ox, oy in [(-.4, .6), (34.2, -18.4)]:
        source = {'origin_x': ox, 'origin_y': oy, 'resolution': .2,
                  'values': np.array([[1., np.nan, 3.], [np.nan, 5., np.nan]])}
        message = observation_grid_map(source, Time(sec=12))
        actual = observed_world_points(message)
        expected = [[ox+.1, oy+.1, 1.], [ox+.5, oy+.1, 3.], [ox+.3, oy+.3, 5.]]
        np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_observed_overlay_does_not_fill_empty_or_occluded_cells():
    message = observation_grid_map({'origin_x': 0., 'origin_y': 0., 'resolution': .2,
                                    'values': np.full((3, 4), np.nan)}, Time(sec=12))
    assert observed_world_points(message).shape == (0, 3)
