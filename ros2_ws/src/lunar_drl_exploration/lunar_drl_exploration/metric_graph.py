"""Native bounded path metric on the exported wheel FREE-cell graph.

Eight neighbors, with both side cells required FREE for diagonal connections.
Distances are path lengths, never Euclidean visibility shortcuts.
"""
import numpy as np
import lunar_drl_graph_native as native


def _cells(cells):
    return np.ascontiguousarray(cells, dtype=np.int64).reshape(-1, 2)


def clearance_order(reachable, navigation_free):
    """Reachable cells ordered by measured navigation clearance, then y/x.

    Only the FREE bounding box needs an EDT: everything outside is non-FREE.
    One padded non-FREE ring also represents the edge of the provided map.
    Distances are to non-FREE cell centers, not physical body-to-wall gaps.
    """
    from scipy.ndimage import distance_transform_edt
    cells = np.argwhere(reachable)[:, ::-1].copy()
    if not len(cells):
        return _cells(cells)
    ys = np.flatnonzero(navigation_free.any(axis=1))
    xs = np.flatnonzero(navigation_free.any(axis=0))
    x0, x1 = xs[0], xs[-1] + 1
    y0, y1 = ys[0], ys[-1] + 1
    field = distance_transform_edt(np.pad(navigation_free[y0:y1, x0:x1], 1))[1:-1, 1:-1]
    values = field[cells[:, 1] - y0, cells[:, 0] - x0]
    return _cells(cells[np.lexsort((cells[:, 0], cells[:, 1], -values))])


class MetricGrid:
    def __init__(self, mask, resolution, navigation_free=None):
        self.mask = np.ascontiguousarray(mask, dtype=bool)
        self.navigation_free = self.mask if navigation_free is None else navigation_free
        self.resolution = float(resolution)
        if self.mask.ndim != 2 or not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('finite positive resolution and 2D mask required')

    def cover(self, retained_cells=(), radius=2.):
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('positive finite cover radius required')
        order = clearance_order(self.mask, self.navigation_free)
        return _cells(native.cover(self.mask, self.resolution, _cells(retained_cells), radius, order))

    def distances(self, starts, start_costs, bound, target_cells):
        starts, targets = _cells(starts), _cells(target_cells)
        costs = np.ascontiguousarray(start_costs, dtype=np.float64)
        if costs.shape != (len(starts),) or np.any(~np.isfinite(costs)) or np.any(costs < 0) or np.isnan(bound) or bound < 0:
            raise ValueError('nonnegative costs and bound required')
        out = np.empty(len(targets), np.float64)
        native.distances(self.mask, self.resolution, starts, costs, bound, targets, out)
        return out

    def connections(self, nodes, limit=8.):
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError('positive finite connection limit required')
        values = np.asarray(native.connections(self.mask, self.resolution, _cells(nodes), limit), float).reshape(-1, 3)
        return values[:, :2].astype(np.int64), values[:, 2].copy()


def greedy_spanner(node_count, edges, lengths, stretch=1.2):
    """Deterministic greedy t-spanner; node ordering breaks equal-length ties."""
    if not np.isfinite(stretch) or stretch < 1:
        raise ValueError('stretch must be finite and >=1')
    edges=_cells(edges)
    lengths=np.ascontiguousarray(lengths,dtype=np.float64)
    if lengths.shape != (len(edges),) or np.any(~np.isfinite(lengths)) or np.any(lengths < 0):
        raise ValueError('finite nonnegative edge lengths required')
    if np.any(edges < 0) or np.any(edges >= node_count):
        raise ValueError('edge endpoint outside graph')
    kept=np.asarray(native.spanner(node_count,edges,lengths,stretch),dtype=np.int64)
    return edges[kept],lengths[kept]
