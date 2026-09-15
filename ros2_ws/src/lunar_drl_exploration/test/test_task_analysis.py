"""Task opportunities depend on measured B/M and observation witnesses."""

from dataclasses import replace
import numpy as np
from lunar_drl_exploration.contracts import Pose, SensorSpec, TaskSpec
from lunar_drl_exploration.maps import PolicyMapSnapshot, PolicyMapTile
from lunar_drl_exploration.task_analysis import TaskAnalyzer


def snapshot(m, b=None, k=None, pose=None, resolution=1.0, origin=(0.0, 0.0, 0.0)):
    m = np.asarray(m, np.uint8)
    b = (
        np.where(m == 1, 1, 0).astype(np.uint8)
        if b is None
        else np.asarray(b, np.uint8)
    )
    k = b if k is None else np.asarray(k, np.uint8)
    tiles = {}
    h, w = m.shape
    for y in range(0, h, 256):
        for x in range(0, w, 256):
            arrays = []
            for a in (m, b, k):
                out = np.zeros((256, 256), np.uint8)
                out[: min(256, h - y), : min(256, w - x)] = a[y : y + 256, x : x + 256]
                arrays.append(out)
            tiles[x // 256, y // 256] = PolicyMapTile(
                *arrays, np.zeros(65536), np.full(65536, np.nan)
            )
    if pose is None:
        y, x = np.argwhere(m == 1)[0]
        pose = Pose(
            origin[0] + (x + 0.5) * resolution, origin[1] + (y + 0.5) * resolution, 0.0
        )
    cell = np.floor((np.array([pose.x, pose.y]) - origin[:2]) / resolution).astype(int)
    return PolicyMapSnapshot(
        "test",
        1,
        resolution,
        origin,
        tiles,
        pose,
        cell.reshape(1, 2),
        (0, 0, w, h),
        "native-test",
        goal_position_tolerance_m=0.1,
        goal_yaw_tolerance_rad=0.05,
    )


def task(x0, y0, x1, y1):
    return TaskSpec("task", "map", [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def test_open_world_unknown_storage_edge_and_known_area_counts_classes():
    m = np.zeros((12, 12), np.uint8)
    m[3:9, 3:9] = 1
    s = snapshot(m)
    a = TaskAnalyzer(task(0, 0, 12, 12), SensorSpec(range_m=3))
    r = a.update(s)
    assert r.available and not r.exhausted and len(r.witnesses)
    assert r.known_area_m2 == 36 and r.new_area_m2 == 36
    assert a.update(s).new_area_m2 == 0
    edge = snapshot(np.ones((8, 8), np.uint8))
    assert (
        not TaskAnalyzer(task(0, 0, 15, 8), SensorSpec(range_m=3))
        .update(edge)
        .exhausted
    )


def test_sealed_unknown_chamber_is_exhausted_without_relabeling_island():
    m = np.ones((15, 15), np.uint8)
    b = m.copy()
    m[4:11, 4:11] = 2
    b[4:11, 4:11] = 2
    m[6:9, 6:9] = 0
    b[6:9, 6:9] = 0
    r = TaskAnalyzer(task(6, 6, 9, 9), SensorSpec(range_m=4)).update(snapshot(m, b))
    assert r.available and r.exhausted


def test_visible_unstandable_gap_but_long_impassable_corridor_exhausts():
    m = np.full((9, 25), 2, np.uint8)
    b = np.full_like(m, 2)
    m[3:6, 1:4] = 1
    b[3:6, 1:4] = 1
    b[4, 4:23] = 0  # M blocked is not an occlusion wall.
    s = snapshot(m, b, pose=Pose(3.5, 4.5, 0))
    near = TaskAnalyzer(task(5, 4, 6, 5), SensorSpec(range_m=4)).update(s)
    far = TaskAnalyzer(task(20, 4, 21, 5), SensorSpec(range_m=4)).update(s)
    assert not near.exhausted and np.any(np.all(near.frontier_cells == [4, 4], axis=1))
    assert far.exhausted


def test_no_native_start_support_is_unavailable_not_exhausted():
    s = snapshot(np.ones((4, 4), np.uint8))
    s = replace(
        s,
        start_connections=np.empty((0, 2), int),
        start_connection_status="INPUT_UNAVAILABLE",
    )
    r = TaskAnalyzer(task(0, 0, 4, 4), SensorSpec()).update(s)
    assert not r.available and not r.exhausted and r.reason_code == "INPUT_UNAVAILABLE"


def test_outside_only_transit_retains_witness_but_closed_irrelevant_spur_does_not():
    m = np.full((17, 23), 2, np.uint8)
    b = m.copy()
    m[7:10, 2:10] = 1
    b[7:10, 2:10] = 1
    # The upper pending branch reaches an unknown task room; the lower one is a sealed dead end.
    m[3:7, 7:9] = 0
    b[3:7, 7:9] = 0
    m[3:5, 7:20] = 0
    b[3:5, 7:20] = 0
    m[10:15, 3:5] = 0
    b[10:15, 3:5] = 0
    r = TaskAnalyzer(task(17, 3, 20, 5), SensorSpec(range_m=2)).update(snapshot(m, b))
    assert not r.exhausted
    assert np.any(r.frontier_cells[:, 1] == 6)
    assert not np.any(r.frontier_cells[:, 1] == 10)
    assert np.all(r.witnesses[:, 0] < 17)


def test_missing_height_only_classification_remains_pending_even_on_free_navigation():
    m = np.ones((9, 9), np.uint8)
    k = m.copy()
    k[4, 4] = 0
    s = snapshot(m, m, k, pose=Pose(4.5, 4.5, 0))
    r = TaskAnalyzer(task(4, 4, 5, 5), SensorSpec(range_m=2)).update(s)
    assert not r.exhausted and r.known_area_m2 == 0
    assert [4, 4] in r.frontier_cells.tolist()


def test_cross_component_visibility_through_free_B_blocked_M_aperture():
    m = np.full((13, 20), 2, np.uint8)
    b = m.copy()
    m[5:8, 1:5] = 1
    b[5:8, 1:5] = 1
    m[5:8, 5:9] = 0
    b[5:8, 5:9] = 0
    # Impassable center strip separates potential components, but it does not
    # occlude a task demand on the other side.
    b[6, 9:12] = 0
    m[5:8, 12:16] = 0
    b[5:8, 12:16] = 0
    s = snapshot(m, b, pose=Pose(4.5, 6.5, 0))
    r = TaskAnalyzer(task(12, 6, 13, 7), SensorSpec(range_m=5)).update(s)
    assert not r.exhausted and [5, 6] in r.frontier_cells.tolist()


def test_acceleration_matches_exhaustive_tiny_observation_witness_enumeration():
    from scipy.ndimage import label, binary_dilation
    from lunar_drl_exploration.task_analysis import CROSS
    from lunar_drl_exploration.sensor import visible_cells

    for seed in range(25):
        rng = np.random.default_rng(seed)
        m = np.full((8, 9), 2, np.uint8)
        b = m.copy()
        m[1:7, 1:8] = rng.choice([0, 1, 2], (6, 7), p=[0.3, 0.45, 0.25])
        b[1:7, 1:8] = np.where(m[1:7, 1:8] == 1, 1, rng.choice([0, 2], (6, 7)))
        m[3, 3] = 1
        b[3, 3] = 1
        s = snapshot(m, b, pose=Pose(3.5, 3.5, 0))
        sensor = SensorSpec(range_m=2)
        analyzer = TaskAnalyzer(task(4, 2, 7, 6), sensor)
        report = analyzer.update(s)
        w = analyzer.workspace
        potential = (w.navigation != 2) & ~w.reachable
        labels, count = label(potential, CROSS)
        exterior = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        exterior = exterior[exterior != 0]
        for component in exterior:
            labels[labels == component] = exterior[0]
        demand = w.task_mask & ~w.known
        relevant = set()
        direct = False
        for y, x in np.argwhere(potential | w.reachable):
            rows, cols = visible_cells(
                w.intrinsic,
                w.origin,
                1.0,
                Pose(w.origin[0] + x + 0.5, w.origin[1] + y + 0.5, 0),
                SensorSpec(range_m=2, fov_deg=360),
            )
            if np.any(demand[rows, cols]):
                if w.reachable[y, x]:
                    direct = True
                else:
                    relevant.add(int(labels[y, x]))
        entry = binary_dilation(w.reachable, structure=CROSS) & potential
        expected = direct or bool(relevant.intersection(set(labels[entry])))
        assert report.exhausted == (not expected), seed
        if len(report.frontier_cells):
            _assert_real_center_witnesses(s, report, sensor)


def test_direct_visible_deep_demand_reports_first_unknown_interface_center():
    m = np.full((11, 17), 2, np.uint8)
    b = np.zeros_like(m)
    m[3:8, 1:4] = 1
    b[3:8, 1:4] = 1
    s = snapshot(m, b, pose=Pose(3.5, 5.5, 0))
    a = TaskAnalyzer(task(6, 5, 7, 6), SensorSpec(range_m=5))
    report = a.update(s)
    assert not report.exhausted
    w = a.workspace
    from scipy.ndimage import binary_dilation
    from lunar_drl_exploration.task_analysis import CROSS

    interface = ~w.known & binary_dilation(w.known, structure=CROSS)
    local = report.frontier_cells - np.array(w.bounds[:2])
    assert np.all(interface[local[:, 1], local[:, 0]])
    assert [6, 5] not in report.frontier_cells.tolist()


def _assert_real_center_witnesses(snapshot_value, report, sensor):
    from lunar_drl_exploration.sensor import target_visibility, visible_cells
    from lunar_drl_exploration.task_analysis import measured_workspace

    assert len(report.frontier_cells) == len(report.witnesses) > 0
    extent_task = task(-10, -10, 30, 30)
    workspace = measured_workspace(snapshot_value, extent_task, sensor)
    shift = np.array(workspace.bounds[:2])
    for frontier, witness in zip(
        report.frontier_cells - shift, report.witnesses - shift
    ):
        assert workspace.reachable[witness[1], witness[0]]
        assert not workspace.known[frontier[1], frontier[0]]
        assert target_visibility(
            workspace.intrinsic,
            witness,
            [frontier],
            sensor.range_m / snapshot_value.resolution_m,
        )[0]
        yaw = np.arctan2(*(frontier - witness)[::-1]) - sensor.offset_yaw_rad
        pose = Pose(
            workspace.origin[0] + (witness[0] + 0.5) * snapshot_value.resolution_m,
            workspace.origin[1] + (witness[1] + 0.5) * snapshot_value.resolution_m,
            yaw,
        )
        rows, cols = visible_cells(
            workspace.intrinsic,
            workspace.origin,
            snapshot_value.resolution_m,
            pose,
            sensor,
        )
        assert tuple(frontier) in set(zip(cols.tolist(), rows.tolist()))


def test_direct_R_visibility_is_independent_of_potential_movement_component():
    m = np.full((9, 15), 2, np.uint8)
    b = m.copy()
    m[3:6, 1:4] = 1
    b[3:6, 1:4] = 1
    b[4, 4:7] = 1
    m[4, 7] = b[4, 7] = 0
    sensor = SensorSpec(range_m=5)
    s = snapshot(m, b, pose=Pose(3.5, 4.5, 0))
    report = TaskAnalyzer(task(7, 4, 8, 5), sensor).update(s)
    assert report.available and not report.exhausted
    assert [7, 4] in report.frontier_cells.tolist()
    _assert_real_center_witnesses(s, report, sensor)


def test_off_axis_pending_contact_is_an_actual_beam_hit_without_losing_demand():
    from lunar_drl_exploration.sensor import target_visibility
    from lunar_drl_exploration.decision import DecisionCore

    b = np.ones((9, 9), np.uint8)
    b[5, 4] = 2
    b[5, 3] = b[6, 0] = 0
    m = np.full_like(b, 2)
    m[4, 4] = 1
    sensor = SensorSpec(range_m=5)
    s = snapshot(m, b, pose=Pose(4.5, 4.5, 0))
    assert target_visibility(b, (4, 4), [[0, 6]], 5)[0]
    # Under prefix emission the previously hidden contact is now a real hit;
    # retain the old sole demand and require every chosen interface to be hit.
    assert target_visibility(b, (4, 4), [[3, 5]], 5)[0]
    observation, report = DecisionCore(task(0, 6, 1, 7), sensor).observe(s)
    assert report.available and not report.exhausted
    _assert_real_center_witnesses(s, report, sensor)
    assert len(report.frontier_cells) > 0
    assert observation.features[observation.current_index, 3:11].max() > 0
    # Measuring the now-valid first interface must expose the original demand,
    # not permanently filter it out as a workaround for the old P1 finding.
    b[5, 3] = 1
    follow = snapshot(m, b, pose=s.pose)
    _, after = DecisionCore(task(0, 6, 1, 7), sensor).observe(follow)
    assert after.available and not after.exhausted
    assert [0, 6] in after.frontier_cells.tolist()
    _assert_real_center_witnesses(follow, after, sensor)


def test_subcell_sensor_range_does_not_create_unobservable_adjacent_frontier():
    m = np.zeros((5, 5), np.uint8)
    m[2, 2] = 1
    report = TaskAnalyzer(task(0, 0, 5, 5), SensorSpec(range_m=0.5)).update(snapshot(m))
    assert not report.available and not report.exhausted
    assert not len(report.frontier_cells)


def test_sparse_direct_witness_search_matches_exhaustive_sensor_sources():
    from lunar_drl_exploration.sensor import (
        direct_witnesses,
        target_visibility,
        visible_cells,
    )

    rng = np.random.default_rng(941)
    for radius in (0.5, 2.5, 5.0):
        b = rng.choice([0, 1, 2], (9, 11), p=[0.3, 0.45, 0.25]).astype(np.uint8)
        reachable = (b == 1) & (rng.random(b.shape) < 0.4)
        targets = np.argwhere(np.ones_like(b, bool))[:, ::-1].copy()
        sources = direct_witnesses(b, reachable, targets, radius)
        expected = np.zeros_like(reachable)
        for y, x in np.argwhere(reachable):
            rows, cols = visible_cells(
                b,
                (0.0, 0.0),
                1.0,
                Pose(x + 0.5, y + 0.5, 0),
                SensorSpec(range_m=radius, fov_deg=360),
            )
            expected[rows, cols] = True
        assert np.array_equal(
            sources[:, 0] >= 0, expected[targets[:, 1], targets[:, 0]]
        )
        for target, source in zip(targets, sources):
            if source[0] >= 0:
                assert reachable[source[1], source[0]]
                assert target_visibility(b, source, [target], radius)[0]


def test_direct_pending_interface_certificate_keeps_external_and_unknown_R_sources():
    # All movement beyond R is blocked, but B is transparent. The first unknown
    # interface lies outside the dense task demand region; it cannot be cropped.
    for known_source in (False, True):
        m = np.full((17, 23), 2, np.uint8)
        b = np.zeros_like(m)
        m[7:10, 1:4] = b[7:10, 1:4] = 1
        k = b if known_source else np.zeros_like(b)
        sensor = SensorSpec(range_m=20)
        s = snapshot(m, b, k, pose=Pose(3.5, 8.5, 0))
        report = TaskAnalyzer(task(10, 4, 20, 14), sensor).update(s)
        assert report.available and not report.exhausted
        assert len(report.frontier_cells)
        assert np.all(report.frontier_cells[:, 0] < 10)
        _assert_real_center_witnesses(s, report, sensor)


def _outside_task_known_band(*, no_entry=False, opaque=False):
    # Native footprint support ends at x=5, while center measurements extend
    # through x=8. The task is beyond range, reached via the unknown corridor.
    m=np.full((17,35),2,np.uint8); b=m.copy()
    m[6:11,2:6]=1
    m[7:10,6:33]=0
    b[6:11,2:9]=1
    b[7:10,9:33]=0
    if no_entry: m[7:10,6]=2  # Transparent, impassable separation.
    if opaque: m[7:10,8]=b[7:10,8]=2
    return snapshot(m,b,pose=Pose(5.5,8.5,0))


def test_outside_task_transit_witness_crosses_known_nonreachable_footprint_band():
    s=_outside_task_known_band(); sensor=SensorSpec(range_m=5)
    report=TaskAnalyzer(task(29,7,32,10),sensor).update(s)
    assert report.available and not report.exhausted
    assert [9,8] in report.frontier_cells.tolist()
    assert np.all(report.frontier_cells[:,0]<29)
    assert np.all(report.witnesses[:,0]<=5)
    _assert_real_center_witnesses(s,report,sensor)
    # The observation witness must survive graph construction as an actual goal.
    from lunar_drl_exploration.decision import DecisionCore
    observation,again=DecisionCore(task(29,7,32,10),sensor).observe(s)
    assert again.available and len(observation.goals)>0
    assert any(observation.features[:,3:11].max(axis=1)>0)


def test_optical_movement_frontier_requires_entry_and_unblocked_sight():
    for mode in ('no_entry','opaque'):
        s=_outside_task_known_band(**{mode:True})
        report=TaskAnalyzer(task(29,7,32,10),SensorSpec(range_m=5)).update(s)
        assert report.available and report.exhausted,mode
        assert len(report.frontier_cells)==0,mode
    s=_outside_task_known_band()
    report=TaskAnalyzer(task(2,6,5,10),SensorSpec(range_m=5)).update(s)
    assert report.available and report.exhausted and not len(report.frontier_cells)
