"""Geometric terrain and synthetic sensor contracts, without a ROS runtime."""

import importlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture(scope='module')
def terrain_type():
    module_path = PACKAGE_ROOT / 'lunar_integrated_exploration_demo' / 'terrain.py'
    assert module_path.is_file(), 'the terrain implementation has not been added'
    return importlib.import_module('lunar_integrated_exploration_demo.terrain').Terrain


@pytest.fixture(scope='module')
def terrain(terrain_type):
    return terrain_type()


def sample(observation, x, y):
    ix = int(np.argmin(np.abs(observation['x'] - x)))
    iy = int(np.argmin(np.abs(observation['y'] - y)))
    return observation['values'][iy, ix]


def single_obstacle_scene(terrain_type, **geometry):
    obstacle_type = importlib.import_module(terrain_type.__module__).Obstacle

    class SingleObstacleTerrain(terrain_type):
        def _make_obstacles(self):
            return (obstacle_type(**geometry),)

    return SingleObstacleTerrain()


def test_seed_reproduces_geometry_and_elevation(terrain_type):
    first, second = terrain_type(seed=47), terrain_type(seed=47)
    third = terrain_type(seed=48)
    assert first.obstacles == second.obstacles
    assert first.obstacles != third.obstacles
    x, y = np.meshgrid(np.linspace(-120, 120, 91), np.linspace(-120, 120, 87))
    np.testing.assert_array_equal(first.elevation(x, y), second.elevation(x, y))
    assert not np.array_equal(first.elevation(x, y), third.elevation(x, y))


def test_vector_elevation_matches_scalar_queries_and_broadcasts(terrain):
    x = np.array([-7.0, 0.0, 10.0, 65.0])
    y = np.array([2.0, 0.0, 5.0, 41.0])
    expected = [terrain.elevation(float(a), float(b)) for a, b in zip(x, y)]
    np.testing.assert_allclose(terrain.elevation(x, y), expected, atol=1e-7)
    assert terrain.elevation(x[:, None], y[None, :]).shape == (4, 4)
    assert isinstance(terrain.elevation(0.0, 0.0), float)


def test_start_has_flat_support_and_room_for_any_heading(terrain):
    assert terrain.elevation(0.0, 0.0) == 0.0
    x, y = np.meshgrid(np.linspace(-3, 3, 13), np.linspace(-3, 3, 13))
    np.testing.assert_array_equal(terrain.elevation(x, y), np.zeros_like(x))
    for yaw in np.linspace(-math.pi, math.pi, 25):
        assert not terrain.collides(0.0, 0.0, float(yaw))


def test_scene_has_near_and_far_rocks_ridges_and_open_wide_corridors(terrain):
    rocks = [o for o in terrain.obstacles if o.kind == 'rock']
    assert len(rocks) > 40
    assert sum(10 < math.hypot(o.x, o.y) < 40 for o in rocks) >= 12
    assert sum(math.hypot(o.x, o.y) > 80 for o in rocks) > 20
    assert sum(o.kind == 'ridge' for o in terrain.obstacles) > 8
    for value in np.linspace(-140, 140, 81):
        for offset in (-1.6, 0.0, 1.6):
            assert not terrain.collides(float(value), offset, 0.0)
            assert not terrain.collides(offset, float(value), math.pi / 2)
    heights = terrain.elevation(np.zeros(101), np.linspace(-145, 145, 101))
    assert np.ptp(heights) > 0.4


def test_observation_is_local_aligned_and_nan_outside_visible_cells(terrain):
    observation = terrain.observe(0.17, -0.31, 0.0)
    assert observation['values'].shape == (140, 140)
    assert observation['values'].dtype == np.float32
    assert observation['mask'].dtype == np.bool_
    np.testing.assert_array_equal(np.isfinite(observation['values']), observation['mask'])
    for name in ('origin_x', 'origin_y'):
        assert observation[name] / 0.2 == pytest.approx(round(observation[name] / 0.2))
    np.testing.assert_allclose(np.diff(observation['x']), 0.2)
    np.testing.assert_allclose(np.diff(observation['y']), 0.2)
    assert observation['x'][0] == pytest.approx(observation['origin_x'] + 0.1)
    assert observation['y'][0] == pytest.approx(observation['origin_y'] + 0.1)
    assert observation['length_m'] == pytest.approx(28.0)
    assert 0 < observation['mask'].sum() < observation['mask'].size / 3
    x, y = np.meshgrid(observation['x'], observation['y'])
    assert not observation['mask'][np.hypot(x - 0.17, y + 0.31) > 12.0].any()
    assert np.isnan(sample(observation, -5.0, 0.0))
    assert np.isfinite(sample(observation, -0.7, 0.0))


def test_sensor_follows_yaw_and_near_field_is_explicit(terrain):
    ahead = terrain.observe(0.0, 0.0, 0.0)
    behind = terrain.observe(0.0, 0.0, math.pi)
    assert np.isfinite(sample(ahead, 6.0, 0.0))
    assert np.isnan(sample(ahead, -6.0, 0.0))
    assert np.isfinite(sample(behind, -6.0, 0.0))
    assert np.isnan(sample(behind, 6.0, 0.0))
    no_near_field = terrain.observe(0.0, 0.0, 0.0, near_field_radius_m=0.0)
    assert np.isnan(sample(no_near_field, -0.7, 0.0))


def test_tall_rock_occludes_ground_behind_it_but_not_beside_it(terrain):
    # The fixed near scene contains a 3 x 2.4 m rock centred at (10, 5).
    observation = terrain.observe(4.0, 5.0, 0.0)
    assert np.isfinite(sample(observation, 7.9, 5.1))
    assert np.isnan(sample(observation, 12.1, 5.1))
    assert np.isfinite(sample(observation, 12.1, 9.1))


@pytest.mark.parametrize('pose,cell,minimum_height', [
    ((9.0, 9.0, math.atan2(-4.0, 1.0)), (9.1, 6.3), 1.5),
    ((6.08127752226607, -7.5, 0.6236889801656514), (7.5, -6.7), 1.5),
])
def test_real_collision_obstacles_return_visible_first_surface(terrain, pose, cell,
                                                              minimum_height):
    assert not terrain.collides(*pose)
    observation = terrain.observe(*pose)
    assert sample(observation, *cell) > minimum_height


def test_non_aligned_wide_wall_returns_its_intersecting_cell_only(terrain_type):
    wall = single_obstacle_scene(terrain_type, x=10.0, y=5.0, length=10.0,
                                 width=2.5, yaw=0.0, height=1.3)
    assert not wall.collides(10.0, 9.0, -math.pi / 2)
    observation = wall.observe(10.0, 9.0, -math.pi / 2, fov_deg=30.0)
    # The wall's north face is y = 6.25, inside the [6.2, 6.4] cell.
    # Its centre is ground in the continuous scene; that cell nevertheless
    # contains the first observed solid surface and must carry its evidence.
    assert wall.elevation(10.1, 6.3) < 1.0
    assert sample(observation, 10.1, 6.3) > 1.5
    assert np.isnan(sample(observation, 10.1, 6.1))  # hidden solid interior
    assert np.isnan(sample(observation, 10.1, 3.3))  # hidden ground behind wall
    assert np.isfinite(sample(observation, 10.1, 8.1))


def test_rotated_narrow_solid_has_first_surface_from_all_sampled_sides(terrain_type):
    thin = single_obstacle_scene(terrain_type, x=6.13, y=1.07, length=3.0,
                                 width=0.24, yaw=0.37, height=1.4)
    for angle in np.arange(24) * (2 * math.pi / 24):
        x, y = 6.13 + 4 * math.cos(angle), 1.07 + 4 * math.sin(angle)
        yaw = math.atan2(1.07 - y, 6.13 - x)
        assert not thin.collides(x, y, yaw)
        observation = thin.observe(x, y, yaw)
        xx, yy = np.meshgrid(observation['x'], observation['y'])
        # A literal box enclosing this fixture, where smooth ground is < 1 m.
        near_solid = (xx >= 4.4) & (xx <= 7.9) & (yy >= 0.3) & (yy <= 1.9)
        assert np.any(observation['values'][near_solid] > 1.0), float(angle)


def test_grazing_ray_does_not_reveal_ground_beyond_rock_corner(terrain):
    observation = terrain.observe(4.0, 5.0, 0.0)
    # This exact ray enters rock #0 at 4.65654 m and exits at 4.66725 m;
    # a centre-angle bin without inflation used to miss the narrow interval.
    assert np.isnan(sample(observation, 14.9, 2.1))
    assert sample(observation, 8.5, 5.1) > 1.0


def test_boundary_cell_representative_is_fixed_across_windows_and_viewpoints(terrain_type):
    wall = single_obstacle_scene(terrain_type, x=10.0, y=5.0, length=10.0,
                                 width=2.5, yaw=0.0, height=1.3)
    surface, ground = [], []
    for cx, cy, window in ((10.0, 9.0, 28.0), (9.4, 8.3, 24.2),
                           (11.3, 9.1, 27.8), (10.0, 8.5, 30.6)):
        observation = wall.observe(cx, cy, -math.pi / 2, window_m=window)
        surface.append(sample(observation, 10.1, 6.3))
        ground.append(sample(observation, 10.1, 8.1))
    assert np.all(np.array(surface) > 1.5)
    for values in (surface, ground):
        values = np.asarray(values, dtype=np.float32)
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values.view(np.uint32),
                                      np.full(values.shape, values[0]).view(np.uint32))


@pytest.mark.parametrize('x,y', [(0.19, 0.19), (-0.01, -0.01), (0.0, 0.0),
                               (0.1, 0.1), (0.39, 0.29), (-0.21, -0.31)])
def test_single_cell_window_contains_start_and_returns_local_support(terrain, x, y):
    observation = terrain.observe(x, y, 0.0, window_m=0.2)
    assert observation['values'].shape == (1, 1)
    assert observation['origin_x'] <= x < observation['origin_x'] + 0.2
    assert observation['origin_y'] <= y < observation['origin_y'] + 0.2
    assert observation['mask'][0, 0]
    assert observation['values'][0, 0] == 0.0


def test_moving_windows_keep_ridge_boundary_cell_height(terrain):
    # Live regression: this global cell changed by 1.51 m when the rolling
    # window moved, although every sample was finite and the terrain static.
    world_coordinates, heights = [], []
    for robot_y in (76.3, 76.5, 76.7, 77.1, 77.3,
                    77.7, 78.1, 78.3, 78.5, 78.7):
        observation = terrain.observe(41.0, robot_y, -math.pi / 2)
        ix = int(np.argmin(np.abs(observation['x'] - 40.1)))
        iy = int(np.argmin(np.abs(observation['y'] - 72.1)))
        assert observation['mask'][iy, ix]
        world_coordinates.append((observation['x'][ix], observation['y'][iy]))
        heights.append(observation['values'][iy, ix])
    expected_height = np.float32(1.5736843347549438)
    np.testing.assert_array_equal(np.array(heights, dtype=np.float32).view(np.uint32),
                                  np.full(10, expected_height).view(np.uint32))
    coordinates = np.array(world_coordinates, dtype=np.float64).view(np.uint64)
    np.testing.assert_array_equal(coordinates, np.tile(coordinates[0], (10, 1)))


def test_boundary_rounding_preserves_surface_without_extending_rock(terrain):
    # The ridge is centred at y = 71 and has half-width 1.1. A single ULP
    # around its mathematical face must not switch between rock and ground.
    face = 72.1
    y = np.array([np.nextafter(face, -np.inf), face,
                  np.nextafter(face, np.inf)])
    height = terrain.elevation(40.1, y).astype(np.float32)
    np.testing.assert_array_equal(height.view(np.uint32),
                                  np.full(3, np.float32(1.5736843347549438)).view(np.uint32))
    # A point genuinely outside remains ground; no physical obstacle padding.
    assert terrain.elevation(40.1, face + 1e-6) < 0.1
    assert terrain.elevation(40.1, face - 1e-6) > 1.5


@pytest.mark.parametrize('base_x,base_y,yaw', [
    (41.0, 77.0, -math.pi / 2), (-40.0, 1.0, 0.0),
    (40.0, -1.0, 0.0), (-40.0, -1.0, math.pi), (40.0, 1.0, math.pi),
])
def test_overlapping_windows_share_exact_coordinates_and_observed_values(
        terrain, base_x, base_y, yaw):
    rng = np.random.default_rng(910)
    reference = terrain.observe(base_x, base_y, yaw)
    checked_observed_cells = 0
    for window_m in rng.choice([24.2, 27.8, 28.0, 30.6], size=20):
        moved = terrain.observe(base_x + rng.uniform(-0.6, 0.6),
                                base_y + rng.uniform(-0.6, 0.6),
                                yaw + rng.uniform(-0.2, 0.2), window_m=float(window_m))
        # Match by spatial cell keys, not by float coordinates or array index.
        _, ref_x, moved_x = np.intersect1d(
            np.floor(reference['x'] / 0.2).astype(np.int64),
            np.floor(moved['x'] / 0.2).astype(np.int64), return_indices=True)
        _, ref_y, moved_y = np.intersect1d(
            np.floor(reference['y'] / 0.2).astype(np.int64),
            np.floor(moved['y'] / 0.2).astype(np.int64), return_indices=True)
        np.testing.assert_array_equal(reference['x'][ref_x].view(np.uint64),
                                      moved['x'][moved_x].view(np.uint64))
        np.testing.assert_array_equal(reference['y'][ref_y].view(np.uint64),
                                      moved['y'][moved_y].view(np.uint64))
        first = reference['values'][np.ix_(ref_y, ref_x)]
        second = moved['values'][np.ix_(moved_y, moved_x)]
        both_observed = np.isfinite(first) & np.isfinite(second)
        checked_observed_cells += int(both_observed.sum())
        np.testing.assert_array_equal(first[both_observed].view(np.uint32),
                                      second[both_observed].view(np.uint32))
        np.testing.assert_array_equal(np.isfinite(moved['values']), moved['mask'])
    assert checked_observed_cells > 1000


def test_collision_uses_rotated_full_footprint_and_includes_ridges(terrain):
    assert terrain.collides(10.0, 5.0, 0.0)
    # Centre is outside the rock, but the long nose reaches its x = 8.5 face.
    assert terrain.collides(8.0, 5.0, 0.0)
    assert not terrain.collides(8.0, 5.0, math.pi / 2)
    assert not terrain.collides(7.8, 5.0, 0.0)
    ridge = next(o for o in terrain.obstacles if o.kind == 'ridge')
    assert terrain.collides(ridge.x, ridge.y, ridge.yaw)


def test_world_edge_is_unknown_and_cannot_be_crossed(terrain):
    assert math.isnan(terrain.elevation(151.0, 0.0))
    assert terrain.collides(149.6, 0.0, 0.0)
    assert not terrain.collides(149.0, 0.0, 0.0)
    edge = terrain.observe(149.0, 0.0, 0.0)
    assert np.isnan(edge['values'][:, edge['x'] > 150.0]).all()
    assert terrain.collides(math.nan, 0.0, 0.0)


def test_display_preview_does_not_expand_sensor_knowledge(terrain):
    before = terrain.observe(0.0, 0.0, 0.0)
    preview = terrain.preview(resolution=2.0)
    assert preview['values'].shape == (150, 150)
    assert np.isfinite(preview['values']).all()
    assert np.ptp(preview['values']) > 1.0
    after = terrain.observe(0.0, 0.0, 0.0)
    np.testing.assert_array_equal(before['values'], after['values'])


@pytest.mark.parametrize('keyword,value', [
    ('resolution', 0.0), ('range_m', -1.0), ('fov_deg', 361.0),
    ('window_m', math.nan), ('near_field_radius_m', 13.0),
])
def test_invalid_sensor_geometry_is_rejected(terrain, keyword, value):
    with pytest.raises(ValueError):
        terrain.observe(0.0, 0.0, 0.0, **{keyword: value})
