"""Native bounded path metric on the exported wheel FREE-cell graph.

Eight neighbors, with both side cells required FREE for diagonal connections.
Distances are path lengths, never Euclidean visibility shortcuts.
"""
import numpy as np
import lunar_drl_graph_native as native


def _cells(cells):
    return np.ascontiguousarray(cells, dtype=np.int64).reshape(-1, 2)


class MetricGrid:
    def __init__(self, mask, resolution):
        self.mask = np.ascontiguousarray(mask, dtype=bool)
        self.resolution = float(resolution)
        if self.mask.ndim != 2 or not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('finite positive resolution and 2D mask required')

    def cover(self, retained_cells=(), radius=2.):
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('positive finite cover radius required')
        return _cells(native.cover(self.mask, self.resolution, _cells(retained_cells), radius))

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
