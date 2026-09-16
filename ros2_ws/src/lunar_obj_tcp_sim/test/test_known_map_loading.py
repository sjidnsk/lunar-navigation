import numpy as np
import pytest
from lunar_obj_tcp_sim.geometry import TerrainMap
from lunar_obj_tcp_sim.prepare import prepare_map
from lunar_obj_tcp_sim.crop_map import crop_map


@pytest.mark.parametrize('focus', [(33.2, -9.7), (10., -20.), (70., 30.)])
def test_center_first_chunks_cover_every_cell_once(focus):
    terrain = object.__new__(TerrainMap)
    terrain.origin_xy=np.array([10.,-20.]); terrain.resolution=.2
    terrain.elevation=np.arange(103*97,dtype=np.float32).reshape(103,97)
    terrain.elevation[2,4]=np.nan
    chunks=list(terrain.iter_known_chunks(32,focus))
    visits=np.zeros_like(terrain.elevation,dtype=int)
    rebuilt=np.full_like(terrain.elevation,np.nan)
    for chunk in chunks:
        x,y=np.rint((chunk.origin_xy-terrain.origin_xy)/terrain.resolution).astype(int)
        h,w=chunk.elevation.shape
        visits[y:y+h,x:x+w]+=1
        rebuilt[y:y+h,x:x+w]=chunk.elevation
    assert np.all(visits==1)
    np.testing.assert_equal(rebuilt,terrain.elevation)
    if 10 < focus[0] < 29 and -20 < focus[1] < 0:
        first=chunks[0];upper=np.array(first.origin_xy)+np.array(first.elevation.shape[::-1])*.2
        assert np.all(np.array(first.origin_xy)<=focus) and np.all(upper>focus)


def test_first_chunk_centers_robot_even_on_old_tile_boundary():
    terrain=object.__new__(TerrainMap)
    terrain.origin_xy=np.array([0.,0.]);terrain.resolution=.2
    terrain.elevation=np.ones((1024,1024),dtype=np.float32)
    first=next(terrain.iter_known_chunks(512,(102.4,102.4)))
    assert first.origin_xy == (51.2,51.2)
    assert first.elevation.shape == (512,512)


def test_crop_preserves_samples_heights_and_visibility(tmp_path):
    obj=tmp_path/'flat.obj'
    obj.write_text('v -10 -10 0\nv 10 -10 0\nv 10 10 0\nv -10 10 0\nf 1 2 3\nf 1 3 4\n')
    prepare_map(obj,tmp_path/'source',center=(0,0),size=20,resolution=.2,halo=0,scale=1,axes='x,y,z')
    metadata=crop_map(tmp_path/'source',tmp_path/'crop',size=8,halo=2)
    old,new=TerrainMap(tmp_path/'source'),TerrainMap(tmp_path/'crop')
    np.testing.assert_equal(new.elevation,old.elevation[20:80,20:80])
    np.testing.assert_equal(new.sample_xy,old.sample_xy[20:80,20:80])
    assert metadata['task_bounds_xy']==[-4,-4,4,4]
    assert new.visible((0,0,1.5),[(3,0,0)]).tolist()==[True]
    with pytest.raises(ValueError,match='empty'):
        crop_map(tmp_path/'source',tmp_path/'crop',size=8,halo=2)
