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
