"""Sparse physical graphs from native FREE rectangles and observation witnesses.

Row-run rectangles exactly cover R, including corridors narrower than a coarse
sampling lattice. Each touching rectangle pair has a physical portal. Within a
convex FREE rectangle, a Euclidean minimum spanning tree connects its center,
portals and mandatory witnesses; obstacle route cycles remain in the quotient.
There is no node crop or sensor-range limit on known-interior edges.
"""

import math
import heapq
import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from .config import GraphConfig, load_platform_config
from .contracts import DecisionObservation, PrivilegedScene
from .geometry import world_to_cell
from .history import HEADINGS
from .sensor import directional_visibility
from .task_analysis import measured_workspace


def _rectangles(mask):
    labels = np.full(mask.shape, -1, np.int32)
    rectangles, active = [], {}
    for y, row in enumerate(mask):
        changes = np.flatnonzero(np.diff(np.r_[False, row, False]))
        next_active = {}
        for x0, x1 in changes.reshape(-1, 2):
            key = (int(x0), int(x1))
            if key in active:
                index = active[key]
                rectangles[index][3] = y + 1
            else:
                index = len(rectangles)
                rectangles.append([int(x0), y, int(x1), y + 1])
            labels[y, x0:x1] = index
            next_active[key] = index
        active = next_active
    return rectangles, labels


def _tree(points):
    """Sparse Euclidean MST; collinear native portals use a sorted chain."""
    n = len(points)
    if n < 2:
        return []
    points = np.asarray(points, float)
    try:
        if n < 3:
            raise QhullError("two points")
        triangles = Delaunay(points).simplices
        pairs = np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
        )
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        lengths = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        sparse = coo_matrix((lengths, (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
        tree = minimum_spanning_tree(sparse + sparse.T).tocoo()
        return list(zip(tree.row.tolist(), tree.col.tolist()))
    except QhullError:
        order = np.lexsort((points[:, 1], points[:, 0]))
        return list(zip(order[:-1].tolist(), order[1:].tolist()))


def physical_graph(reachable, mandatory=()):
    """Return integer-cell nodes and undirected physically certified edges."""
    rectangles, labels = _rectangles(reachable)
    cells, lookup, members, edges = [], {}, [set() for _ in rectangles], set()

    def add(x, y):
        key = (int(x), int(y))
        if key not in lookup:
            lookup[key] = len(cells)
            cells.append(key)
        index = lookup[key]
        members[int(labels[y, x])].add(index)
        return index

    for x0, y0, x1, y1 in rectangles:
        add((x0 + x1 - 1) // 2, (y0 + y1 - 1) // 2)
    for x, y in np.asarray(mandatory, dtype=int).reshape(-1, 2):
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] and labels[y, x] >= 0:
            add(x, y)
    for i, (x0, y0, x1, y1) in enumerate(rectangles):
        # Only bottom/right faces are necessary. One portal per overlapping
        # rectangle pair preserves every physical branch and obstacle loop.
        for side in ("right", "bottom"):
            if side == "right":
                if x1 == labels.shape[1]:
                    continue
                adjacent = labels[y0:y1, x1]
            else:
                if y1 == labels.shape[0]:
                    continue
                adjacent = labels[y1, x0:x1]
            for neighbor in np.unique(adjacent):
                if neighbor < 0 or neighbor == i:
                    continue
                offsets = np.flatnonzero(adjacent == neighbor)
                offset = int(offsets[len(offsets) // 2])
                if side == "right":
                    a, b = add(x1 - 1, y0 + offset), add(x1, y0 + offset)
                else:
                    a, b = add(x0 + offset, y1 - 1), add(x0 + offset, y1)
                edges.add(tuple(sorted((a, b))))
    points = np.asarray(cells, np.int64).reshape(-1, 2)
    for group in members:
        ids = sorted(group)
        for a, b in _tree(points[ids]):
            edges.add(tuple(sorted((ids[a], ids[b]))))
    return points, edges


def _relay_order(cells):
    """Deterministic separated ordering of a connected local FREE region.

    Interleaving opposite halves avoids sub-tolerance chains of neighboring
    ports. Evaluate axis and angular orders, using the largest minimum progress.
    No co-located relay IDs or discarded physical branches are introduced.
    """
    n = len(cells)
    if not n:
        return []
    if n == 1:
        return [0]
    points = cells.astype(float)
    center = points.mean(axis=0)
    orders = [
        np.lexsort((points[:, 1], points[:, 0])),
        np.lexsort((points[:, 0], points[:, 1])),
        np.argsort(
            np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]),
            kind="stable",
        ),
    ]
    candidates = []
    split = (n + 1) // 2
    for order in orders:
        for reverse in (False, True):
            low, high = order[:split], order[split:]
            if reverse:
                high = high[::-1]
            for offset in range(min(8, max(1, len(high)))):
                high_shift = np.roll(high, offset)
                sequence = np.empty(n, int)
                sequence[::2] = low
                sequence[1::2] = high_shift
                progress = np.linalg.norm(
                    points[sequence] - points[np.roll(sequence, -1)], axis=1
                )
                candidates.append((float(progress.min()), sequence.copy()))
    order = max(candidates, key=lambda item: item[0])[1].tolist()
    return order


def _route_distances(adjacency, lengths, starts, points, actual, targets):
    distances = {n: float(np.linalg.norm(points[n] - actual)) for n in starts}
    queue = [(v, n) for n, v in distances.items()]
    heapq.heapify(queue)
    done = set()
    pending = set(targets)
    while queue and pending:
        distance, node = heapq.heappop(queue)
        if node in done:
            continue
        done.add(node)
        pending.discard(node)
        for neighbor in adjacency[node]:
            candidate = distance + lengths[tuple(sorted((node, neighbor)))]
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def _stable_ids(cells):
    # Signed world lattice coordinates packed losslessly into a 64-bit identity.
    cells = np.asarray(cells, np.int64)
    if np.any(cells < -(2**31)) or np.any(cells >= 2**31):
        raise ValueError("world cell index outside stable int32 lattice")
    return (cells[:, 0] << 32) | (cells[:, 1] & 0xFFFFFFFF)


def _inside(points, polygon):
    points = np.asarray(points, float)
    result = np.zeros(len(points), bool)
    boundary = result.copy()
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        dx, dy = b.astype(float) - a
        cross = (points[:, 0] - a[0]) * dy - (points[:, 1] - a[1]) * dx
        boundary |= (
            (np.abs(cross) <= 1e-10 * max(1.0, math.hypot(dx, dy)))
            & np.all(points >= np.minimum(a, b) - 1e-10, axis=1)
            & np.all(points <= np.maximum(a, b) + 1e-10, axis=1)
        )
        if dy:
            result ^= ((a[1] > points[:, 1]) != (b[1] > points[:, 1])) & (
                points[:, 0] < dx * (points[:, 1] - a[1]) / dy + a[0]
            )
    return result | boundary


class GraphBuilder:
    def __init__(self, config=None):
        self.config = config or GraphConfig()
        self.platform = self.config.platform or load_platform_config()
        self.workspace = None
        self._relay_epoch = None

    def build(self, snapshot, report, history, task, sensor, velocity):
        if report.revision != snapshot.revision:
            raise ValueError("task report does not match snapshot revision")
        w = self.workspace or measured_workspace(snapshot, task, sensor)
        # DecisionCore supplies this exact snapshot workspace once; don't retain
        # a previous snapshot by accident on direct GraphBuilder use.
        self.workspace = None
        shift = np.array(w.bounds[:2])
        starts = np.asarray(snapshot.start_connections).reshape(-1, 2) - shift
        mandatory = np.concatenate((report.witnesses - shift, starts))
        cells, edges = physical_graph(w.reachable, mandatory)
        points = (cells + shift + 0.5) * snapshot.resolution_m + snapshot.origin[:2]
        actual = np.array([snapshot.pose.x, snapshot.pose.y], np.float64)
        current = len(cells)
        points = np.vstack((points, actual))
        ids = np.r_[_stable_ids(cells + shift), np.int64(-(2**63))]
        if ids[:-1].size and np.any(ids[:-1] == ids[-1]):
            raise ValueError("world lattice collides with reserved actual anchor ID")
        lookup = {tuple(p): i for i, p in enumerate(cells)}
        start_nodes = {lookup[tuple(p)] for p in starts if tuple(p) in lookup}
        lengths = {
            pair: float(np.linalg.norm(points[pair[0]] - points[pair[1]]))
            for pair in edges
        }
        adjacency = [[] for _ in cells]
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        tolerance = snapshot.goal_position_tolerance_m
        if not np.isfinite(tolerance):
            tolerance = self.config.history_tolerance_m
        arrived = np.linalg.norm(points[:-1] - actual, axis=1) <= tolerance + 1e-9
        # Native connection segments followed by certified graph edges supply
        # physical route lengths, including bends around obstacles.
        distances = {n: float(np.linalg.norm(points[n] - actual)) for n in start_nodes}
        queue = [(distance, n) for n, distance in distances.items()]
        heapq.heapify(queue)
        visited = set()
        boundary = set()
        while queue:
            distance, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            if not arrived[node]:
                boundary.add(node)
                continue
            for neighbor in adjacency[node]:
                candidate = distance + lengths[tuple(sorted((node, neighbor)))]
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        if not boundary:
            boundary = start_nodes
        if len(cells) and (len(boundary) > 19 or self._relay_epoch == snapshot.epoch):
            self._relay_epoch = snapshot.epoch
            # Deterministic real-position relays replace an overfull arrival
            # neighborhood. The local cycle is a compression of a native-FREE graph
            # walk, not an executable navigation planner. Keep every original
            # edge (and therefore every physical obstacle-route alternative).
            # Crucially, transient actual-start insertions do not change the
            # relay order when rebuilding after an executed goal.
            stable_cells, stable_edges = physical_graph(
                w.reachable, report.witnesses - shift
            )
            # World-aligned local cells make decomposition independent of a
            # requested goal or transient anchor insertion. Include cut-edge
            # endpoints so moving to a relay can expose the next local region.
            zone_width = max(1, int(math.ceil(4 * tolerance / snapshot.resolution_m)))
            stable_world = stable_cells + shift
            nearest_stable = int(
                np.argmin(
                    np.linalg.norm(
                        (stable_world + 0.5) * snapshot.resolution_m
                        + snapshot.origin[:2]
                        - actual,
                        axis=1,
                    )
                )
            )
            zone = stable_world[nearest_stable] // zone_width
            interior = set(
                np.flatnonzero(
                    np.all(stable_world // zone_width == zone, axis=1)
                ).tolist()
            )
            # Include the measured arrival component when the actual anchor
            # lies near a zone boundary; never return yaw-only because a fixed
            # bin ended before an available physical-progress endpoint.
            visited_cells = {tuple(cells[i]) for i in visited}
            interior |= {
                i for i, p in enumerate(stable_cells) if tuple(p) in visited_cells
            }
            allowed = set(interior)
            for a, b in stable_edges:
                if a in interior:
                    allowed.add(b)
                if b in interior:
                    allowed.add(a)
            stable_adjacency = [[] for _ in stable_cells]
            for a, b in stable_edges:
                if a in allowed and b in allowed:
                    stable_adjacency[a].append(b)
                    stable_adjacency[b].append(a)
            local = set()
            todo = [nearest_stable]
            while todo:
                node = todo.pop()
                if node in local:
                    continue
                local.add(node)
                todo.extend(stable_adjacency[node])
            local = sorted(local)
            tour = _relay_order(stable_cells[local])
            tour_nodes = [lookup[tuple(stable_cells[local[i]])] for i in tour]
            if tour_nodes:
                distances_to_actual = np.linalg.norm(
                    points[tour_nodes] - actual, axis=1
                )
                nearest = int(np.argmin(distances_to_actual))
                ordinary_boundary = boundary
                boundary = set()
                # Two neighboring relay routes advance beyond the arrival
                # region. Relay order maximizes separation of neighboring ports.
                threshold = tolerance
                for direction in (-1, 1):
                    for step in range(1, len(tour_nodes) + 1):
                        index = (nearest + direction * step) % len(tour_nodes)
                        if distances_to_actual[index] > threshold + 1e-9:
                            boundary.add(tour_nodes[index])
                            break
                if len(boundary | ordinary_boundary) <= 19:
                    boundary |= ordinary_boundary
                # Obtain the length of actual native/graph paths to the selected
                # relays; no Euclidean shortcut across a blocker is asserted.
                distances = _route_distances(
                    adjacency, lengths, start_nodes, points, actual, boundary
                )
        for node in boundary:
            pair = (node, current)
            edges.add(pair)
            lengths[pair] = distances[node]
        edge_array = np.asarray(sorted(edges), np.int64).reshape(-1, 2)
        feature = np.zeros((len(points), 19), np.float32)
        feature[:, :2] = (points - actual) / 10.0
        feature[:, 2] = _inside(points, task.polygon)
        feature[:, 11:] = history.bits(points)
        fronts = report.frontier_cells - shift
        if len(fronts):
            tree = cKDTree(fronts)
            source_cells = np.floor(
                (points - np.array(w.origin)) / snapshot.resolution_m
            ).astype(np.int64)
            radius = sensor.range_m / snapshot.resolution_m
            for i, source in enumerate(source_cells):
                candidates = tree.query_ball_point(source, radius + 1e-10)
                if not candidates:
                    continue
                targets = fronts[candidates]
                visible = directional_visibility(
                    w.intrinsic,
                    source,
                    targets,
                    radius,
                    HEADINGS + sensor.offset_yaw_rad,
                    math.radians(sensor.fov_deg),
                )
                feature[i, 3:11] = visible.sum(axis=0) / 100.0
        action_positions = [current] + sorted(boundary)
        action_nodes = np.repeat(action_positions, 8)
        yaws = np.tile(HEADINGS, len(action_positions))
        goals = np.column_stack((points[action_nodes], yaws))
        return DecisionObservation(
            ids,
            points,
            feature,
            edge_array,
            np.asarray([lengths[tuple(p)] for p in edge_array]),
            current,
            (task.polygon.astype(float) - actual) / 10.0,
            self.platform.actor_context(
                snapshot.pose,
                linear_speed_mps=float(velocity[0]),
                angular_speed_radps=float(velocity[1]),
                sensor_range_m=sensor.range_m,
                sensor_fov_rad=math.radians(sensor.fov_deg),
            ),
            action_nodes,
            yaws,
            goals,
            snapshot.epoch,
            snapshot.revision,
        )

    def build_truth(self, terrain, reference):
        # Explicit training-only entry point, no Scene/reference module import.
        reachable = reference.unpack(reference.reachable_bits)
        cells, edge_set = physical_graph(reachable)
        points = (cells + 0.5) * terrain.resolution_m + terrain.origin[:2]
        edges = np.asarray(sorted(edge_set), np.int64).reshape(-1, 2)
        indices = np.flatnonzero(reference.unpack(reference.packed_mask).ravel())
        if len(points):
            # Bounded chunks avoid an O(reference-cells) float64 XY temporary.
            owners = np.empty(len(indices), np.int64)
            tree = cKDTree(points)
            for first in range(0, len(indices), 65536):
                batch = indices[first : first + 65536]
                xy = np.column_stack(
                    (batch % reference.shape[1], batch // reference.shape[1])
                )
                xy = (xy + 0.5) * terrain.resolution_m + terrain.origin[:2]
                owners[first : first + len(batch)] = tree.query(xy)[1]
            order = np.argsort(owners, kind="stable")
            indices = indices[order]
            offsets = np.r_[0, np.cumsum(np.bincount(owners, minlength=len(points)))]
        elif len(indices):
            raise ValueError("reference cells require reachable static graph")
        else:
            offsets = np.array([0], np.int64)
        descriptor = getattr(terrain, "generator_descriptor", None)
        if descriptor is None:
            descriptor = {
                "terrain_id": terrain.terrain_id,
                "resolution_m": terrain.resolution_m,
                "origin": tuple(terrain.origin),
                "shape": tuple(terrain.shape),
            }
        descriptor = dict(descriptor, terrain_id=terrain.terrain_id)
        return PrivilegedScene(
            reference.reference_id,
            points,
            edges,
            np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1),
            offsets,
            indices,
            reference.packed_mask,
            tuple(reference.shape),
            descriptor,
        )
