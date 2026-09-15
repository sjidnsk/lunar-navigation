"""Sparse physical topology, exact executable anchors and observation-only input."""

from dataclasses import replace
import numpy as np
from lunar_drl_exploration.config import GraphConfig
from lunar_drl_exploration.contracts import Pose, SensorSpec
from lunar_drl_exploration.decision import DecisionCore
from lunar_drl_exploration.graph import GraphBuilder
from lunar_drl_exploration.history import DirectionHistory
from test_task_analysis import snapshot, task


def test_actual_fractional_anchor_all_headings_frozen_and_zero_gain_transit():
    origin = (100000.123, -200000.456, 0.0)
    actual = Pose(origin[0] + 3.5123456789, origin[1] + 4.5987654321, 0.234)
    m = np.ones((9, 9), np.uint8)
    s = snapshot(m, pose=actual, origin=origin)
    t = task(origin[0], origin[1], origin[0] + 9, origin[1] + 9)
    obs, report = DecisionCore(t, SensorSpec(range_m=3)).observe(s, (0.0, 0.0))
    assert obs.features.shape[1] == 19 and obs.context.shape == (8,)
    assert len(obs.goals) <= 160
    anchors = obs.goals[obs.action_nodes == obs.current_index]
    assert len(anchors) == 8 and anchors.dtype == np.float64
    assert np.all(anchors[:, :2] == [actual.x, actual.y])
    assert not obs.goals.flags.writeable
    assert report.exhausted and np.all(obs.features[:, 3:11] == 0)
    assert len(np.unique(obs.action_nodes)) > 1


def test_native_one_cell_corridor_and_obstacle_loop_survive():
    m = np.zeros((23, 31), np.uint8)
    m[2:21, 2:29] = 1
    m[7:16, 11:20] = 2
    m[11, 20:29] = 0
    m[11, 20:28] = 1
    s = snapshot(m, pose=Pose(4.5, 10.5, 0))
    o, _ = DecisionCore(task(0, 0, 31, 23), SensorSpec(range_m=3)).observe(
        s, (0.0, 0.0)
    )
    assert len(o.edges) >= len(o.node_ids)  # physical obstacle creates a cycle
    degree = np.bincount(o.edges.ravel(), minlength=len(o.node_ids))
    assert degree.max() <= 19
    # A native M FREE corridor only 1m wide, with no coarse 2m lattice point.
    corridor = np.zeros((12, 21), np.uint8)
    corridor[1:5, 1:6] = 1
    corridor[3, 6:16] = 1
    corridor[1:5, 16:20] = 1
    co, _ = DecisionCore(task(0, 0, 21, 12), SensorSpec(range_m=2)).observe(
        snapshot(corridor), (0, 0)
    )
    adj = [[] for _ in co.node_ids]
    for a, b in co.edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = {co.current_index}
    todo = list(seen)
    while todo:
        for n in adj[todo.pop()]:
            if n not in seen:
                seen.add(n)
                todo.append(n)
    assert len(seen) == len(co.node_ids)
    assert co.positions[:, 0].max() > 16


def test_history_records_executed_world_pose_not_requested_index():
    h = DirectionHistory(tolerance_m=0.1)
    h.record(Pose(3.51, 4.52, np.pi / 2), np.array([[1, 2], [2, 2]]), SensorSpec())
    bits = h.bits(np.array([[3.5, 4.5], [3.8, 4.5]]))
    assert bits[0, 2] == 1 and bits[1].sum() == 0
    assert h.bits(np.array([[3.51, 4.52]]))[0, 2] == 1


def test_directional_utilities_match_sensor_hits_and_disappear_after_measurement():
    from lunar_drl_exploration.sensor import visible_cells

    m = np.zeros((19, 23), np.uint8)
    m[3:15, 3:18] = 1
    b = np.where(m == 1, 1, 0).astype(np.uint8)
    m[6:12, 9:12] = 2
    b[6:12, 9:12] = 2
    s = snapshot(m, b, pose=Pose(6.537, 8.519, 0.713))
    sensor = SensorSpec(range_m=4, fov_deg=90, offset_yaw_rad=0.13)
    core = DecisionCore(task(0, 0, 23, 19), sensor)
    obs, report = core.observe(s, (0.1, 0.2))
    w = core.analyzer.workspace
    front = set(map(tuple, report.frontier_cells - np.array(w.bounds[:2])))
    for index in range(len(obs.positions)):
        xy = obs.goals[obs.action_nodes == index, :2]
        xy = xy[0] if len(xy) else obs.positions[index].astype(float)
        for heading in range(8):
            rows, cols = visible_cells(
                w.intrinsic,
                w.origin,
                s.resolution_m,
                Pose(*xy, heading * np.pi / 4),
                sensor,
            )
            expected = (
                sum((int(x), int(y)) in front for y, x in zip(rows, cols)) / 100.0
            )
            assert np.isclose(obs.features[index, 3 + heading], expected)
    filled = snapshot(np.ones((19, 23), np.uint8), pose=s.pose)
    after, done = core.observe(replace(filled, revision=2), (0, 0))
    assert done.exhausted and np.all(after.features[:, 3:11] == 0)
    assert len(after.action_nodes) > 8


def test_sole_outside_viewpoint_is_a_mandatory_graph_node():
    m = np.full((11, 19), 2, np.uint8)
    b = m.copy()
    m[5, 2:9] = 1
    b[5, 2:9] = 1
    b[5, 9:14] = 0  # task is nonstandable, visible only from external reachable stance
    s = snapshot(m, b, pose=Pose(2.5, 5.5, 0))
    core = DecisionCore(task(12, 5, 13, 6), SensorSpec(range_m=4))
    obs, report = core.observe(s, (0, 0))
    assert not report.exhausted
    assert [8, 5] in report.witnesses.tolist()
    assert np.any(np.all(obs.positions == [8.5, 5.5], axis=1))
    assert np.all(obs.features[:, 2] == 0)


def test_observed_actor_path_is_truth_independent_in_fresh_interpreter():
    import os, subprocess, sys

    source = """
import sys
class RejectTruth:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('torch','lunar_drl_exploration.scene','lunar_drl_exploration.reference'):
            raise AssertionError('observed path attempted privileged import: '+fullname)
sys.meta_path.insert(0,RejectTruth())
from lunar_drl_exploration.decision import DecisionCore
from lunar_drl_exploration.contracts import SensorSpec
from test_task_analysis import snapshot,task
import numpy as np
s=snapshot(np.ones((8,8),np.uint8))
core=DecisionCore(task(0,0,10,10),SensorSpec(range_m=2))
a,_=core.observe(s,(0,0))
# Arbitrary hidden world replacement cannot be passed into the actor API.
hidden_truth=np.random.default_rng(42).random((100,100))
b,_=core.observe(s,(0,0))
for name in ('features','positions','edges','goals','context','polygon'):
    assert np.array_equal(getattr(a,name),getattr(b,name)),name
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(__import__("pathlib").Path(__file__).parent)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_arrival_region_skips_portals_that_cannot_make_physical_progress():
    # FREE center domain, not a claim about fitting the body through 1m walls.
    m = np.zeros((20, 70), np.uint8)
    m[5:15, 1:25] = 1
    m[9:11, 25:45] = 1
    m[5:15, 45:69] = 1
    sensor = SensorSpec(range_m=1)
    core = DecisionCore(task(0, 0, 14, 4), sensor)
    s = snapshot(m, resolution=0.2, pose=Pose(4.87, 1.91, 0))
    s = replace(s, goal_position_tolerance_m=0.3)
    first, _ = core.observe(s, (0, 0))
    for obs in (first, core.observe(s, (0, 0))[0]):
        moving = obs.goals[obs.action_nodes != obs.current_index, :2]
        assert len(moving) > 0
        assert np.all(np.linalg.norm(moving - [s.pose.x, s.pose.y], axis=1) > 0.3)
    goal = first.goals[first.action_nodes != first.current_index][0]
    pose = Pose(*goal)
    moved = replace(
        s,
        pose=pose,
        start_connections=np.floor(goal[:2] / 0.2).astype(int).reshape(1, 2),
    )
    second, _ = core.observe(moved, (0, 0))
    moving = second.goals[second.action_nodes != second.current_index, :2]
    assert np.all(np.linalg.norm(moving - goal[:2], axis=1) > 0.3)


def test_privileged_graph_owns_reference_and_row_major_mapping():
    from types import SimpleNamespace
    from lunar_drl_exploration.reference import CoverageReference

    m = np.ones((8, 11), np.uint8)
    m[2:6, 4:7] = 2
    packed = np.packbits(np.ones(m.shape, bool).ravel(), bitorder="little")
    reference = CoverageReference(
        m.shape,
        packed,
        0.04,
        m.size * 0.04,
        "ref",
        np.packbits((m == 1).ravel(), bitorder="little"),
    )
    terrain = SimpleNamespace(
        terrain_id="synthetic",
        resolution_m=0.2,
        origin=(1.2, -3.7, 0.0),
        shape=m.shape,
        generator_descriptor={
            "version": 4,
            "seed": 7,
            "family": "test",
            "nested": {"values": [1, 2]},
        },
    )
    scene = GraphBuilder(GraphConfig()).build_truth(terrain, reference)
    assert np.array_equal(np.sort(scene.reference_indices), np.arange(m.size))
    assert scene.reference_offsets[-1] == m.size
    assert len(scene.reference_offsets) == len(scene.positions) + 1
    assert scene.generator_descriptor["seed"] == 7
    terrain.generator_descriptor["nested"]["values"][0] = 99
    assert scene.generator_descriptor["nested"]["values"] == (1, 2)
    packed[:] = 0
    assert np.all(reference.unpack(scene.packed_reference))
    assert (
        not scene.positions.flags.writeable
        and not scene.reference_indices.flags.writeable
    )


def test_more_than_nineteen_local_ports_remain_action_reachable_after_rebuild():
    # Stress a native-exported large arrival region. Sixty physical branches
    # cannot be silently cropped to the first nineteen neighbors.
    m = np.full((80, 80), 2, np.uint8)
    m[32:48, 10:70] = 1
    for x in range(11, 70, 2):
        m[5:32, x] = 1
        m[48:75, x] = 1
    s = replace(
        snapshot(m, m, m, pose=Pose(8.03, 8.07, 0), resolution=0.2),
        goal_position_tolerance_m=5.0,
    )
    core = DecisionCore(task(0, 0, 16, 16), SensorSpec(range_m=1))
    todo = [s.pose]
    seen = set()
    largest_degree = 0
    while todo:
        pose = todo.pop()
        key = (round(pose.x, 7), round(pose.y, 7))
        if key in seen:
            continue
        seen.add(key)
        assert len(seen) < 500
        state = replace(
            s,
            pose=pose,
            revision=len(seen) + 1,
            start_connections=np.floor(np.array([pose.x, pose.y]) / 0.2)
            .astype(int)
            .reshape(1, 2),
        )
        obs, _ = core.observe(state, (0, 0))
        largest_degree = max(
            largest_degree,
            int(np.bincount(obs.edges.ravel(), minlength=len(obs.node_ids)).max()),
        )
        assert len(obs.goals) <= 160
        for n in np.unique(obs.action_nodes):
            if n == obs.current_index:
                continue
            goal = obs.goals[obs.action_nodes == n][0]
            assert np.linalg.norm(goal[:2] - [pose.x, pose.y]) > 5.0
            # Simulate actual non-central arrival, then rebuild a new revision.
            todo.append(Pose(goal[0] + 0.03, goal[1] + 0.03, goal[2]))
    assert len({round(x, 5) for x, y in seen if y < 6.4}) == 30
    assert len({round(x, 5) for x, y in seen if y > 9.6}) == 30
    assert largest_degree <= 19


def test_first_executed_observation_survives_first_snapshot_and_epoch_clears_history():
    s = replace(
        snapshot(np.ones((9, 9), np.uint8), pose=Pose(3.5, 4.5, 0)),
        goal_position_tolerance_m=0.3,
    )
    core = DecisionCore(task(0, 0, 9, 9), SensorSpec())
    core.record_observation(Pose(3.71, 4.5, np.pi / 2), np.array([[3, 4]]))
    first, _ = core.observe(s, (0, 0))
    assert first.features[first.current_index, 13] == 1
    second, _ = core.observe(replace(s, epoch="new"), (0, 0))
    assert second.features[:, 11:].sum() == 0


def test_scene_generator_descriptor_survives_terrain_to_privileged_handoff():
    import json
    from lunar_drl_exploration.scene import Scene, TerrainGrid
    from lunar_drl_exploration.reference import CoverageReference

    scene = Scene(92, "moon", 20, resolution_m=1.0)
    terrain = TerrainGrid.from_scene(scene)
    start = scene.initial_pose(terrain)
    reference = CoverageReference.build(terrain, start, scene.task, SensorSpec())
    privileged = GraphBuilder().build_truth(terrain, reference)
    descriptor = privileged.generator_descriptor
    assert descriptor["seed"] == 92 and descriptor["generator_version"] == 4
    assert descriptor["family"] == "moon" and descriptor["extent_m"] == 20
    assert json.loads(descriptor["capability_json"])["maximum_forward_speed_mps"] == 0.2


def test_known_empty_interior_growth_is_perimeter_not_area_and_has_no_node_crop():
    results = []
    for n in (50, 100, 200):
        s = snapshot(
            np.ones((n, n), np.uint8),
            resolution=0.2,
            pose=Pose(n * 0.1 + 0.013, n * 0.1 + 0.017, 0),
        )
        obs, report = DecisionCore(
            task(-1, -1, n * 0.2 + 1, n * 0.2 + 1), SensorSpec(range_m=2)
        ).observe(s, (0, 0))
        results.append(len(obs.node_ids))
        assert len(report.frontier_cells) == 4 * n
        assert len(obs.node_ids) >= 4 * n - 4  # mandatory witnesses were not cropped
        assert len(obs.node_ids) < 5 * n
        assert len(obs.edges) < 6 * n
    assert results[2] < 2.1 * results[1]


def test_core_unavailable_input_still_freezes_actual_anchor_without_navigation_support():
    s = snapshot(np.ones((8, 8), np.uint8), pose=Pose(3.51987654321, 4.517654321, 0))
    s = replace(
        s,
        start_connections=np.empty((0, 2), int),
        start_connection_status="INPUT_UNAVAILABLE",
    )
    obs, report = DecisionCore(task(0, 0, 8, 8), SensorSpec()).observe(s, (0, 0))
    assert not report.available and not report.exhausted
    assert len(obs.node_ids) == 1 and len(obs.goals) == 8
    assert np.all(obs.goals[:, :2] == [s.pose.x, s.pose.y])
