"""Deterministic metric cover, exact observation supplements and greedy spanner."""
import math
import numpy as np
from scipy.spatial import cKDTree
from .config import GraphConfig, load_platform_config
from .contracts import DecisionObservation
from .history import HEADINGS
from .sensor import direct_witnesses, directional_visibility
from .task_analysis import measured_workspace
from .metric_graph import MetricGrid, greedy_spanner

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
        self._base_points = np.empty((0, 2))
        self._epoch = None
        self._lattice_origin = None
        self.unrepresented_interfaces = np.empty((0, 2), np.int64)

    def build(self, snapshot, report, history, task, sensor, velocity):
        if report.revision != snapshot.revision:
            raise ValueError("task report does not match snapshot revision")
        w = self.workspace or measured_workspace(snapshot, task, sensor)
        self.workspace = None
        identity = (snapshot.epoch, snapshot.resolution_m)
        lattice_origin = np.asarray(snapshot.origin[:2], dtype=float)
        same_lattice = False
        if self._lattice_origin is not None and self._epoch == identity:
            shift_cells = (lattice_origin-self._lattice_origin)/snapshot.resolution_m
            same_lattice = bool(np.all(np.abs(shift_cells-np.rint(shift_cells)) <= 1e-7))
        if not same_lattice:
            self._base_points = np.empty((0, 2))
            self._epoch = identity
            self._lattice_origin = lattice_origin.copy()
        shift = np.array(w.bounds[:2])
        origin = np.asarray(w.origin)
        resolution = snapshot.resolution_m
        metric = MetricGrid(w.reachable, resolution, navigation_free=w.navigation == 1)
        retained = np.rint((self._base_points-origin)/resolution-.5).astype(np.int64)
        base = metric.cover(retained, self.config.coverage_radius_m)
        self._base_points = (base+.5)*resolution+origin
        actual=np.array([snapshot.pose.x,snapshot.pose.y],np.float64)
        tolerance=snapshot.goal_position_tolerance_m
        if not np.isfinite(tolerance):tolerance=self.config.history_tolerance_m
        # Keep complete base coverage internally, but contract arrived bases
        # before optical selection. Only executable stances can cover F here.
        active_base = base[np.linalg.norm(self._base_points-actual,axis=1)>tolerance+1e-9]
        # Stable world position order, independent of map-array shifts.
        fronts = report.frontier_cells-shift
        selected = [tuple(p) for p in active_base]
        selected_set = set(selected)
        tree = cKDTree(fronts) if len(fronts) else None
        radius = sensor.range_m/resolution
        visibility_cache = {}
        def visible(source):
            key = tuple(source)
            if key in visibility_cache:return visibility_cache[key]
            if tree is None:return set()
            candidates = tree.query_ball_point(source,radius+1e-10)
            if not candidates:return set()
            seen = directional_visibility(w.intrinsic, source, fronts[candidates],
                radius, HEADINGS+sensor.offset_yaw_rad, math.radians(sensor.fov_deg))
            result = set(np.asarray(candidates)[seen.any(axis=1)].tolist())
            visibility_cache[key] = result
            return result
        uncovered = set(range(len(fronts)))
        actual_cell = np.floor((actual-origin)/resolution).astype(np.int64)
        uncovered.difference_update(visible(actual_cell))
        for source in active_base:
            uncovered.difference_update(visible(source))
        # Observation supplements belong to executable graph geometry. The
        # analyzer's first witness can be contracted by arrival tolerance even
        # when another measured stance can represent the same interface.
        witnesses = np.empty((0, 2), np.int64)
        if uncovered:
            yy, xx = np.ogrid[:w.reachable.shape[0], :w.reachable.shape[1]]
            distance2 = ((origin[0]+(xx+.5)*resolution-actual[0])**2
                         + (origin[1]+(yy+.5)*resolution-actual[1])**2)
            executable = w.reachable & (distance2 > (tolerance+1e-9)**2)
            found = direct_witnesses(w.intrinsic, executable, fronts[sorted(uncovered)], radius)
            witnesses = np.unique(found[found[:, 0] >= 0], axis=0)
        options=[]
        for source in witnesses:
            x,y=source
            if (tuple(source) not in selected_set and 0<=y<w.reachable.shape[0]
                    and 0<=x<w.reachable.shape[1] and w.reachable[y,x]
                    and np.linalg.norm((source+.5)*resolution+origin-actual)>tolerance+1e-9):
                options.append((tuple(source),visible(source)))
        while uncovered:
            if not options:break
            best=max(range(len(options)),key=lambda i:(len(options[i][1]&uncovered),
                -options[i][0][1],-options[i][0][0]))
            source,seen=options.pop(best)
            if not seen&uncovered:break
            selected.append(source);uncovered.difference_update(seen)
        self.unrepresented_interfaces = report.frontier_cells[sorted(uncovered)].copy()
        cells=np.asarray(sorted(selected,key=lambda p:(p[1],p[0])),np.int64).reshape(-1,2)
        points=(cells+.5)*resolution+origin
        starts=np.asarray(snapshot.start_connections).reshape(-1,2)-shift
        costs=np.linalg.norm((starts+.5)*resolution+origin-actual,axis=1)
        distances=metric.distances(starts,costs,self.config.connection_limit_m,cells)
        # Actual anchor distances use native start segments. Arrived nodes
        # were contracted before optical selection and cannot prune exits.
        edges,lengths=metric.connections(cells,self.config.connection_limit_m)
        current=len(cells)
        neighbors=np.flatnonzero(np.isfinite(distances))
        edges=np.vstack((edges,np.column_stack((neighbors,np.full(len(neighbors),current)))))
        lengths=np.r_[lengths,distances[neighbors]]
        edge_array,edge_lengths=greedy_spanner(current+1,edges,lengths,self.config.stretch)
        lengths={tuple(e):float(l) for e,l in zip(edge_array,edge_lengths)}
        boundary={int(a) for a,b in edge_array if b==current}
        points=np.vstack((points,actual))
        # IDs encode world lattice coordinates, not workspace-local indices.
        world_cells=np.floor(points[:-1]/resolution+1e-8).astype(np.int64)
        ids=np.r_[_stable_ids(world_cells),np.int64(-(2**63))]
        if np.any(ids[:-1] == ids[-1]):
            raise ValueError("world lattice collides with actual anchor identity")
        feature = np.zeros((len(points), 19), np.float32)
        feature[:, :2] = (points - actual) / 10.0
        feature[:, 2] = _inside(points, task.polygon)
        feature[:, 11:] = history.action_bits(points, sensor)
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
        # A fallback native cell at the actual XY is already represented by
        # the exact floating-point anchor; roundoff must not duplicate 8 actions.
        action_positions = [current] + sorted(boundary)
        action_nodes = np.repeat(action_positions, 8)
        yaws = np.tile(HEADINGS, len(action_positions))
        # A turn is an observation action only when a pending task interface is
        # visible there. Transit is allowed even with zero immediate utility.
        # Use the navigation owner's arrival tolerances, not visitation bans.
        yaw_tolerance = snapshot.goal_yaw_tolerance_rad
        if not np.isfinite(yaw_tolerance):
            yaw_tolerance = 0.0
        yaw_error = np.abs(np.arctan2(np.sin(yaws - snapshot.pose.yaw),
                                      np.cos(yaws - snapshot.pose.yaw)))
        headings = np.tile(np.arange(8), len(action_positions))
        moving = action_nodes != current
        valid = moving | ((yaw_error > yaw_tolerance + 1e-9) &
                          (feature[current, 3 + headings] > 0))
        action_nodes, yaws = action_nodes[valid], yaws[valid]
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
        """Compatibility facade; privileged construction stays training-only."""
        from .privileged import build_truth
        return build_truth(terrain, reference, config=self.config)
