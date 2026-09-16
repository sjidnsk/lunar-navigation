import pytest
from lunar_obj_tcp_sim.observation_state import exploration_bounds, EvidenceCoverage


def test_task_shrinks_without_changing_map_metadata():
    metadata = dict(task_bounds_xy=[714.,-1610.,1714.,-610.],
                    resolution=.2, origin_xy=[702.,-1622.],shape=[5120,5120])
    bounds = exploration_bounds(metadata['task_bounds_xy'],100.)
    assert bounds == [1164.,-1160.,1264.,-1060.]
    coverage = EvidenceCoverage(dict(metadata,task_bounds_xy=bounds))
    assert coverage.total_cells == 250000
    assert metadata['task_bounds_xy'] == [714.,-1610.,1714.,-610.]
    assert metadata['shape'] == [5120,5120]


def test_small_maps_keep_their_bounds_and_size_is_configurable():
    assert exploration_bounds([-4,-4,4,4],100)==[-4,-4,4,4]
    assert exploration_bounds([-500,-500,500,500],200)==[-100,-100,100,100]
    with pytest.raises(ValueError):
        exploration_bounds([-4,-4,4,4],0)
