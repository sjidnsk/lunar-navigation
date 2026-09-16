import numpy as np
from lunar_obj_tcp_sim.prepare import prepare_map
from lunar_obj_tcp_sim.geometry import TerrainMap


def test_wall_front_visible_ground_behind_unknown_and_stable_cells(tmp_path):
    obj = tmp_path / 'wall.obj'
    obj.write_text('g ground\nv -10 -10 0\nv 10 -10 0\nv 10 10 0\nv -10 10 0\nf 1 2 3\nf 1 3 4\ng rock\nv 3 -3 0\nv 3 3 0\nv 3 3 3\nv 3 -3 3\nf 5 6 7\nf 5 7 8\n')
    prepare_map(obj, tmp_path / 'map', center=(0, 0), size=20,
                resolution=.2, halo=0, scale=1, axes='x,y,z')
    terrain = TerrainMap(tmp_path / 'map')
    observation = terrain.observe((0, 0, 0), (0, 0, 0, 1), observation_window_m=20)
    def cell(x, y):
        return observation.elevation[int((y-observation.origin_xy[1])/.2), int((x-observation.origin_xy[0])/.2)]
    assert np.isfinite(cell(2, .1))
    assert np.isnan(cell(5, .1))
    assert np.nanmax(observation.elevation[:, 65]) > 1
    shifted = terrain.observe((.07, .04, 0), (0, 0, 0, 1), observation_window_m=20)
    assert shifted.origin_xy == observation.origin_xy


def test_terrain_ridge_occludes_nearfield_and_yaw_changes_visibility(tmp_path):
    obj = tmp_path / 'ridge.obj'
    obj.write_text('g terrain\nv -5 -5 0\nv -5 5 0\nv 1 -5 3\nv 1 5 3\nv 5 -5 0\nv 5 5 0\nf 1 3 2\nf 2 3 4\nf 3 5 4\nf 4 5 6\n')
    prepare_map(obj, tmp_path/'map', center=(0, 0), size=10, resolution=.2, halo=0, scale=1, axes='x,y,z')
    terrain = TerrainMap(tmp_path/'map')
    assert terrain.visible((0, 0, 3), [(2, 0, 2.25), (-1, 0, 2)]).tolist() == [False, True]
    forward = terrain.observe((0, 0, 1.5), (0, 0, 0, 1), near_field_radius_m=0, observation_window_m=10)
    backward = terrain.observe((0, 0, 1.5), (0, 0, 1, 0), near_field_radius_m=0, observation_window_m=10)
    assert np.any(np.isfinite(backward.elevation) & ~np.isfinite(forward.elevation))
    close = terrain.observe((0, 0, 1.5), (0, 0, 1, 0), near_field_radius_m=4, observation_window_m=10)
    # Behind the ridge stays unknown even within the explicit omnidirectional near field.
    assert np.isnan(close.elevation[25, 35])
    chunks = list(terrain.iter_known_chunks(13))
    assert sum(np.isfinite(c.elevation).sum() for c in chunks) == terrain.metadata['valid_cells']


def test_visibility_matches_independent_scalar_intersections():
    rng = np.random.default_rng(931)
    triangles = rng.uniform(-4, 4, (35, 3, 3))
    targets = rng.uniform(-6, 6, (80, 3))
    origin = np.array((.2, -.1, 1.7))
    terrain = object.__new__(TerrainMap)
    terrain._local_triangles = lambda lower, upper: triangles
    expected = []
    for target in targets:
        blocked = False
        for a, b, c in triangles:
            # Solve target segment = barycentric triangle directly, independent of Moller implementation.
            matrix = np.column_stack((target-origin, -(b-a), -(c-a)))
            if abs(np.linalg.det(matrix)) < 1e-10:
                continue
            t, u, v = np.linalg.solve(matrix, a-origin)
            blocked |= 1e-7 < t < 1-1e-5 and u >= -1e-8 and v >= -1e-8 and u+v <= 1+1e-8
        expected.append(not blocked)
    assert terrain.visible(origin, targets).tolist() == expected
