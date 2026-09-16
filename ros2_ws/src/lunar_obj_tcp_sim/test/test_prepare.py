import numpy as np
from lunar_obj_tcp_sim.prepare import prepare_map
from lunar_obj_tcp_sim.geometry import TerrainMap


def test_interpolation_crossing_roi_negative_indices_filter(tmp_path):
    obj = tmp_path / 'input.obj'
    obj.write_text('g Landscape\nv -10 -10 -20\nv 10 -10 0\nv 0 10 10\nf -3 -2 -1\ng SkySphere\nv -1 -1 99\nv 1 -1 99\nv 0 1 99\nf 4 5 6\n')
    prepare_map(obj, tmp_path / 'map', center=(0, 0), size=2, resolution=.2,
                halo=0, scale=1, axes='x,y,z')
    terrain = TerrainMap(tmp_path / 'map')
    assert abs(terrain.height_at(.1, .1) - .2) < 1e-4
    assert terrain.metadata['triangle_count'] == 1
    assert terrain.metadata['alignment_status'] == 'NOT_EXTERNALLY_ALIGNED'
    assert terrain.metadata['groups']['SkySphere']['kept'] is False


def test_transform_and_missing_surface_and_no_overwrite(tmp_path):
    import pytest
    obj = tmp_path / 'input.obj'
    obj.write_text('g terrain\nv 0 200 0\nv 100 200 0\nv 0 200 -100\nf 1 2 3\n')
    prepare_map(obj, tmp_path/'map', center=(10.5, 20.5), size=1, resolution=.2,
                halo=0, scale=.01, axes='x,-z,y', offset=(10, 20, 30))
    terrain = TerrainMap(tmp_path/'map')
    assert terrain.height_at(10.1, 20.1) == 32
    assert np.isnan(terrain.height_at(10.9, 20.9))
    with pytest.raises(ValueError, match='empty'):
        prepare_map(obj, tmp_path/'map', center=(0, 0))


def test_roi_sat_rejects_bbox_only_overlap_and_invalid_axis():
    import pytest
    from lunar_obj_tcp_sim.prepare import _intersects, axis_transform
    triangles = np.array([[[0, 2, 0], [2, 0, 0], [2, 2, 0]],
                          [[-10, -10, 0], [10, -10, 0], [0, 10, 0]]], dtype=float)
    assert _intersects(triangles, [0, 0, .5, .5]).tolist() == [False, True]
    with pytest.raises(ValueError):
        axis_transform('x,x,z', 1, (0, 0, 0))


def test_excluded_object_cannot_be_reenabled_by_child_group(tmp_path):
    obj = tmp_path/'objects.obj'
    obj.write_text('o SkySphere\ng mesh\nv -2 -2 50\nv 2 -2 50\nv 0 2 50\nf 1 2 3\no Terrain\ng floor\nv -2 -2 0\nv 2 -2 0\nv 0 2 0\nf 4 5 6\n')
    prepare_map(obj, tmp_path/'map', center=(0, 0), size=1, resolution=.2, halo=0, scale=1, axes='x,y,z')
    assert TerrainMap(tmp_path/'map').height_at(.1, .1) == 0


def test_initial_ungrouped_faces_obey_include_and_exclude(tmp_path):
    obj = tmp_path/'ungrouped.obj'
    obj.write_text('v -2 -2 50\nv 2 -2 50\nv 0 2 50\nf 1 2 3\ng terrain\nv -2 -2 0\nv 2 -2 0\nv 0 2 0\nf 4 5 6\n')
    for name, filters in [('include', dict(includes=('terrain',))),
                          ('exclude', dict(excludes=('default',)))]:
        prepare_map(obj, tmp_path/name, center=(0, 0), size=1, resolution=.2,
                    halo=0, scale=1, axes='x,y,z', **filters)
        terrain = TerrainMap(tmp_path/name)
        assert terrain.height_at(.1, .1) == 0
        assert terrain.metadata['groups']['default']['kept'] is False
        assert terrain.metadata['triangle_count'] == 1


def test_nontriangle_faces_fail_with_actionable_message(tmp_path):
    import pytest
    obj = tmp_path/'concave.obj'
    polygon = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    obj.write_text('g terrain\n'+''.join(f'v {x} {y} 0\n' for x, y in polygon)+'f 1 2 3 4 5 6 7 8\n')
    with pytest.raises(ValueError, match='triangulate mesh'):
        prepare_map(obj, tmp_path/'map', center=(1.5, 1.5), size=3,
                    resolution=.2, halo=0, scale=1, axes='x,y,z')
    assert not (tmp_path/'map'/'metadata.json').exists()
