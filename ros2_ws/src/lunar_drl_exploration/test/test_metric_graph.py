import numpy as np
from lunar_drl_exploration.metric_graph import MetricGrid, greedy_spanner

def test_cover_metric_and_retention():
    m=np.ones((15,20),bool);m[:12,10]=False
    g=MetricGrid(m,.2); cells=g.cover([[2,2]], radius=.7)
    assert [2,2] in cells.tolist()
    targets=np.argwhere(m)[:,::-1].copy()
    d=g.distances(cells,np.zeros(len(cells)),np.inf,targets)
    assert d.max()<=.7+1e-9
    assert np.array_equal(cells,g.cover(cells,radius=.7))

def test_barrier_and_corner_cut():
    g=MetricGrid(np.eye(2,dtype=bool),1.)
    assert np.isinf(g.distances([[0,0]],[0.],8,[[1,1]])[0])
    m=np.ones((10,10),bool);m[:9,5]=False
    g=MetricGrid(m,1.)
    assert len(g.connections([[4,0],[6,0]],8)[0])==0

def test_spanner_bound():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path
    g=MetricGrid(np.ones((25,25),bool),.4)
    nodes=g.cover(radius=2);e,l=g.connections(nodes,8)
    se,sl=greedy_spanner(len(nodes),e,l,1.2)
    def distances(e,l):
        a=coo_matrix((l,(e[:,0],e[:,1])),shape=(len(nodes),)*2).tocsr()
        return shortest_path(a+a.T,directed=False)
    assert np.all(distances(se,sl)<=1.2*distances(e,l)+1e-8)
