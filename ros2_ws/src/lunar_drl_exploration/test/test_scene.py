import numpy as np
import pytest
from lunar_drl_exploration.scene import Scene, TerrainGrid
from lunar_drl_exploration.config import load_platform_config


@pytest.mark.parametrize("family", ["moon", "cave"])
def test_scene_seed_tiles_id_and_legitimate_start(family):
    p = load_platform_config()
    a = Scene(23, family, 40, 0.5, p)
    b = Scene(23, family, 40, 0.5, p)
    assert a.scene_id == b.scene_id
    assert a.scene_id != Scene(24, family, 40, 0.5, p).scene_id
    assert a.scene_id != Scene(23, family, 40, 0.4, p).scene_id
    t = TerrainGrid.from_scene(a)
    pose = a.initial_pose(t)
    x, y = t.world_to_cell(pose.x, pose.y)
    assert t.navigation[y, x] == 1
    tile = a.height_tile(0, 0, 7, 9)
    assert np.array_equal(tile, t.heights[:9, :7])
    assert np.array_equal(a.height_tile(0, 0, 7, 9), b.height_tile(0, 0, 7, 9))
    assert a.bounds[0] < a.task_polygon[:, 0].min()
    assert a.bounds[2] > a.task_polygon[:, 0].max()
    assert t.heights.dtype == np.float32 and t.intrinsic.dtype == np.uint8
    assert t.navigation.dtype == np.uint8
    assert not np.all(t.intrinsic[0] == 2) if family == "moon" else True


def test_scene_requires_supported_family_and_bounded_tile():
    with pytest.raises(ValueError):
        Scene(1, "invalid", 40, 0.2, load_platform_config())
    a = Scene(1, "moon", 40, 0.5, load_platform_config())
    with pytest.raises(ValueError):
        a.height_tile(-1, 0, 10, 10)


def test_cave_keeps_disconnected_free_chamber_and_external_connection():
    from lunar_drl_exploration.geometry import polygon_mask

    scene = Scene(23, "cave", 40, 0.2, load_platform_config())
    terrain = TerrainGrid.from_scene(scene)
    reachable = terrain.reachable(scene.initial_pose(terrain))
    x, y = terrain.world_to_cell(*scene.sealed_room[:2])
    assert terrain.intrinsic[y, x] == 1 and terrain.navigation[y, x] == 1
    assert reachable[y, x] == 0
    task = polygon_mask(terrain.shape, terrain.origin, 0.2, scene.task.polygon)
    assert np.any((reachable == 1) & ~task)


@pytest.mark.parametrize("family", ["moon", "cave"])
def test_seed_randomizes_polygon_and_main_component_start_without_changing_yaw(family):
    import math
    from lunar_drl_exploration.contracts import Pose

    polygons, starts = [], []
    for seed in (23, 24, 25):
        scene = Scene(seed, family, 40, 0.2, load_platform_config())
        terrain = TerrainGrid.from_scene(scene)
        pose = scene.initial_pose(terrain)
        clone = Scene(seed, family, 40, 0.2, load_platform_config())
        assert np.array_equal(scene.task_polygon, clone.task_polygon)
        assert pose == scene.initial_pose(terrain) == clone.initial_pose(terrain)
        original_rng = np.random.default_rng(seed)
        original_rng.uniform(-math.pi, math.pi)  # terrain phase, then unchanged yaw
        assert pose.yaw == float(original_rng.uniform(-math.pi, math.pi))
        x, y = terrain.world_to_cell(pose.x, pose.y)
        assert terrain.navigation[y, x] == 1
        main = terrain.reachable(pose)
        if family == "moon":
            remaining = terrain.navigation == 1
            component_sizes = []
            while np.any(remaining):
                ay, ax = np.argwhere(remaining)[0]
                component = (
                    terrain.reachable(Pose(*terrain.cell_center(int(ax), int(ay)), 0))
                    != 0
                )
                component_sizes.append(np.count_nonzero(component))
                remaining[component] = False
            assert np.count_nonzero(main) == max(component_sizes)
        if family == "cave":
            ax, ay = terrain.world_to_cell(*scene._nominal_start)
            assert terrain.navigation[ay, ax] == 1 and main[ay, ax] == 1
            sx, sy = terrain.world_to_cell(*scene.sealed_room[:2])
            assert main[sy, sx] == 0
            assert terrain.heights[y, x] == 0
        assert np.all(scene.task_polygon[:, 0] > scene.bounds[0])
        assert np.all(scene.task_polygon[:, 0] < scene.bounds[2])
        assert np.all(scene.task_polygon[:, 1] > scene.bounds[1])
        assert np.all(scene.task_polygon[:, 1] < scene.bounds[3])
        polygons.append(scene.task_polygon.tobytes())
        starts.append((pose.x, pose.y))
    assert len(set(polygons)) == 3
    assert len(set(starts)) == 3


@pytest.mark.parametrize("seed", range(6))
def test_largest_main_component_matches_native_no_corner_cut_partition(seed):
    import lunar_drl_terrain_native as native
    from lunar_drl_exploration.scene import _largest_native_free_component

    m = np.asarray(np.random.default_rng(seed).random((8, 9)) > 0.5, np.uint8)
    remaining = m == 1
    expected = np.zeros(m.shape, bool)
    while np.any(remaining):
        y, x = np.argwhere(remaining)[0]
        component = np.empty_like(m)
        native.reachable(m, int(x), int(y), component)
        if np.count_nonzero(component) > np.count_nonzero(expected):
            expected = component != 0
        remaining[component != 0] = False
    assert np.array_equal(_largest_native_free_component(m), expected)
    m.fill(0)
    m[2:5, 2:5] = 1
    m[1, 1] = 1
    assert not _largest_native_free_component(m)[1, 1]
    assert np.count_nonzero(_largest_native_free_component(m)) == 9
