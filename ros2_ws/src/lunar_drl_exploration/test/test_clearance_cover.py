import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
from lunar_drl_exploration.metric_graph import MetricGrid


def test_open_room_starts_in_interior_and_retains_existing_nodes():
    mask = np.ones((15, 21), bool)
    graph = MetricGrid(mask, .2)
    nodes = graph.cover(radius=.7)
    assert nodes[0].tolist() == [7, 7]
    assert np.array_equal(graph.cover(nodes, radius=.7), nodes)
    assert graph.cover([[0, 0]], radius=.7)[0].tolist() == [0, 0]
    targets = np.argwhere(mask)[:, ::-1].copy()
    assert graph.distances(nodes, np.zeros(len(nodes)), np.inf, targets).max() <= .7 + 1e-9


@pytest.mark.parametrize('seed', range(12))
def test_cropped_priority_matches_full_navigation_field(seed):
    from lunar_drl_exploration.metric_graph import clearance_order
    rng = np.random.default_rng(seed)
    navigation = np.zeros((45, 67), bool)
    navigation[8:30, 11:45] = rng.random((22, 34)) > .15
    reachable = navigation.copy()
    reachable[20:] = False
    full = distance_transform_edt(np.pad(navigation, 1))[1:-1, 1:-1]
    cells = np.argwhere(reachable)[:, ::-1].copy()
    expected = cells[np.lexsort((cells[:, 0], cells[:, 1], -full[cells[:, 1], cells[:, 0]]))]
    assert np.array_equal(clearance_order(reachable, navigation), expected)
    assert np.array_equal(clearance_order(np.zeros_like(reachable), navigation), np.empty((0, 2), np.int64))


def test_outside_navigation_boundary_and_translated_arrays():
    from lunar_drl_exploration.metric_graph import clearance_order
    mask = np.ones((9, 13), bool)
    order = clearance_order(mask, mask)
    shifted = np.pad(mask, ((4, 2), (7, 3)))
    assert np.array_equal(clearance_order(shifted, shifted) - [7, 4], order)
    assert order[0].tolist() == [4, 4]


def test_free_coverage_no_corner_cut_and_no_spacing_gate():
    mask = np.eye(3, dtype=bool)
    graph = MetricGrid(mask, .2)
    nodes = graph.cover(radius=2.)
    assert len(nodes) == 3
    assert len(graph.connections(nodes, 8.)[0]) == 0
    narrow = MetricGrid(np.ones((1, 30), bool), .2)
    targets = np.array([[i, 0] for i in range(30)])
    nodes = narrow.cover(radius=2.)
    assert narrow.distances(nodes, np.zeros(len(nodes)), np.inf, targets).max() <= 2. + 1e-9
