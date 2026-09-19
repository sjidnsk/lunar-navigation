"""Training-owned truth skeleton and frozen candidate descriptors.

This module is never needed by the online decision path. Terrain and temporary
ray results stay with the environment; replay owns only scalar gains and sparse
read-only query supports. Truth does not add or remove measured actions.
"""
import numpy as np
from scipy.spatial import cKDTree
from .config import GraphConfig
from .contracts import PrivilegedScene, PrivilegedState, PrivilegedActionContext, Pose
from .geometry import polygon_mask
from .metric_graph import MetricGrid, greedy_spanner
from .sensor import visible_cells


def build_truth(terrain, reference, config=None):
    """Build the scene-owned static metric skeleton once per environment reset."""
    config = config or GraphConfig()
    reachable = reference.unpack(reference.reachable_bits)
    grid = MetricGrid(reachable, terrain.resolution_m)
    cells = grid.cover(radius=config.coverage_radius_m)
    points = (cells + 0.5) * terrain.resolution_m + np.asarray(terrain.origin[:2])
    candidates, distances = grid.connections(cells, limit=config.connection_limit_m)
    edges, lengths = greedy_spanner(len(cells), candidates, distances, config.stretch)
    indices = np.flatnonzero(reference.unpack(reference.packed_mask).ravel())
    if len(points):
        owners = np.empty(len(indices), np.int64)
        tree = cKDTree(points)
        for first in range(0, len(indices), 65536):
            chunk = indices[first:first + 65536]
            xy = np.column_stack((chunk % reference.shape[1], chunk // reference.shape[1]))
            xy = (xy + 0.5) * terrain.resolution_m + np.asarray(terrain.origin[:2])
            owners[first:first + len(chunk)] = tree.query(xy)[1]
        order = np.argsort(owners, kind='stable')
        indices = indices[order]
        offsets = np.r_[0, np.cumsum(np.bincount(owners, minlength=len(points)))]
    elif len(indices):
        raise ValueError('reference cells require reachable truth skeleton')
    else:
        offsets = np.array([0], np.int64)
    descriptor = getattr(terrain, 'generator_descriptor', None)
    if descriptor is None:
        descriptor = dict(resolution_m=terrain.resolution_m, origin=tuple(terrain.origin),
                          shape=tuple(terrain.shape))
    descriptor = dict(descriptor, terrain_id=terrain.terrain_id,
                      graph_schema='metric_cover_spanner_v1')
    return PrivilegedScene(reference.reference_id, points, edges, lengths, offsets,
        indices, reference.packed_mask, tuple(reference.shape), descriptor)


class PrivilegedBuilder:
    """One environment's truth geometry; state-dependent descriptors never cache K."""
    def __init__(self, terrain, scene, task, sensor, config=None):
        self.terrain, self.scene, self.sensor = terrain, scene, sensor
        self.config = config or GraphConfig()
        self.grid = MetricGrid(terrain.navigation == 1, terrain.resolution_m)
        self.cells = np.floor((scene.positions.astype(float) - terrain.origin[:2]) /
                              terrain.resolution_m).astype(np.int64)
        self.target = polygon_mask(terrain.shape, terrain.origin, terrain.resolution_m,
                                   task.polygon)
        self.target &= np.isfinite(terrain.heights) & (terrain.intrinsic != 0)

    def build(self, observation, observed):
        """Freeze all current actions against their exact pre-action observed bits."""
        positions, inverse = np.unique(observation.goals[:, :2], axis=0, return_inverse=True)
        offsets = [0]
        supports, support_distances = [], []
        terrain = self.terrain
        resolution = terrain.resolution_m
        origin = np.asarray(terrain.origin[:2])
        for xy in positions:
            cell = np.floor((xy - origin) / resolution).astype(np.int64)
            x, y = map(int, cell)
            if not (0 <= y < terrain.shape[0] and 0 <= x < terrain.shape[1]
                    and terrain.navigation[y, x] == 1):
                raise ValueError('candidate query lacks native truth FREE start connection')
            # Formal BuildStartConnections returns the containing grid cell in
            # the evidence-FREE case. This is also the plant's truth precondition.
            entry = float(np.linalg.norm((cell + .5) * resolution + origin - xy))
            distance = self.grid.distances(cell.reshape(1, 2), np.array([entry]),
                self.config.coverage_radius_m + entry, self.cells)
            selected = np.flatnonzero(np.isfinite(distance))
            if not len(selected):
                raise ValueError('truth cover provides no legal candidate query support')
            supports.extend(selected.tolist())
            support_distances.extend(distance[selected].tolist())
            offsets.append(len(supports))
        observed = np.asarray(observed, dtype=np.uint8)
        gains = np.empty(len(observation.goals), np.float32)
        for i, goal in enumerate(observation.goals):
            rows, cols = visible_cells(terrain.intrinsic, terrain.origin, resolution,
                                        Pose(*goal), self.sensor)
            indices = rows * terrain.shape[1] + cols
            known = ((observed[indices // 8] >> (indices % 8)) & 1).astype(bool)
            gains[i] = np.count_nonzero(self.target[rows, cols] & ~known) * resolution**2
        actions = PrivilegedActionContext(positions, inverse, observation.goals[:, 2], np.asarray(offsets),
            np.asarray(supports, np.int64), np.asarray(support_distances, np.float32), gains)
        return PrivilegedState(self.scene.scene_id, observed, actions)
