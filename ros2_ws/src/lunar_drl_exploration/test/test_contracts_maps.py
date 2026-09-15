"""Behavioral contracts for the read-only native policy-map cache."""

from types import SimpleNamespace

import numpy as np
import pytest

from lunar_drl_exploration.config import load_platform_config
from lunar_drl_exploration.contracts import DecisionObservation, Pose
from lunar_drl_exploration.maps import BLOCKED, FREE, UNKNOWN, PolicyMapStore


def _response(*, epoch="epoch-a", revision=1, full_snapshot=True,
              state=UNKNOWN, intrinsic=UNKNOWN, observed=UNKNOWN):
    """A hand-built wire-shaped policy-map response for one native tile."""
    count = 256 * 256
    states = np.full(count, state, dtype=np.uint8)
    intrinsic_states = np.full(count, intrinsic, dtype=np.uint8)
    effective = np.full(count, observed, dtype=np.uint8)
    return SimpleNamespace(
        ready=True,
        reason_code="READY",
        epoch=epoch,
        fine_revision=revision,
        base_revision=revision - 1,
        raw_elevation_revision=revision,
        full_snapshot=full_snapshot,
        frame_id="map",
        resolution_m=0.1,
        origin=(0.0, 0.0, 0.0),
        profile_hash="wheel-profile",
        pose=Pose(0.0, 0.0, 0.0),
        start_connections=np.empty((0, 2), dtype=np.int64),
        local_bounds=(0, 0, 256, 256),
        tiles=[SimpleNamespace(
            tile_x=0, tile_y=0, states=states,
            intrinsic_states=intrinsic_states, observed=effective,
            costs=np.full(count, 0.5, dtype=np.float32),
            elevation_m=np.full(count, 1.25, dtype=np.float32),
        )],
    )


def test_delta_replaces_only_changed_tile_without_mutating_prior_snapshot():
    # Removing structural sharing or returning writable tile buffers would
    # silently corrupt replay observations from an earlier revision.
    store = PolicyMapStore()
    before = store.apply(_response())
    delta = _response(revision=2, full_snapshot=False,
                      state=BLOCKED, intrinsic=FREE, observed=FREE)
    after = store.apply(delta)

    assert before.cell_at(1, 1).state == UNKNOWN
    assert after.cell_at(1, 1).state == FREE
    assert after.cell_at(1, 1).navigation_state == BLOCKED
    assert not before.tiles[(0, 0)].states.flags.writeable
    with pytest.raises(ValueError):
        before.tiles[(0, 0)].states[0] = FREE


def test_epoch_replacement_discards_old_tiles_and_missed_delta_requires_full():
    # Accepting an out-of-sequence delta would make action masks refer to a
    # fabricated mixture of map revisions.
    store = PolicyMapStore()
    store.apply(_response(revision=4, state=FREE, intrinsic=FREE, observed=FREE))
    with pytest.raises(ValueError, match="full snapshot"):
        store.apply(_response(revision=6, full_snapshot=False,
                              state=FREE, intrinsic=FREE, observed=FREE))

    replaced = store.apply(_response(epoch="epoch-b", revision=1,
                                     state=UNKNOWN, intrinsic=UNKNOWN,
                                     observed=UNKNOWN))
    assert replaced.epoch == "epoch-b"
    assert replaced.revision == 1
    assert replaced.cell_at(1, 1).state == UNKNOWN


def test_effective_observed_classification_never_infers_measurement_from_navigation_state():
    # Treating an inflated M-blocked cell as observed would inflate discovery
    # reward and leak a footprint-derived label into actor input.
    snapshot = PolicyMapStore().apply(
        _response(state=BLOCKED, intrinsic=UNKNOWN, observed=UNKNOWN))
    cell = snapshot.cell_at(0, 0)
    assert cell.state == UNKNOWN
    assert cell.intrinsic_state == UNKNOWN
    assert cell.navigation_state == BLOCKED
    assert not np.isnan(cell.elevation_m)


def test_decision_observation_freezes_action_indices_and_world_goals_at_construction():
    # A replay writer mutating its staging arrays must not change the action
    # identity or world goal attached to an already queued observation.
    node_ids = np.array([7, 9], dtype=np.int64)
    goals = np.array([[3.0, 4.0, 0.0], [5.0, 6.0, 1.57]], dtype=np.float32)
    observation = DecisionObservation(
        node_ids=node_ids,
        positions=np.zeros((2, 2), dtype=np.float32),
        features=np.zeros((2, 19), dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        edge_lengths=np.array([1.0], dtype=np.float32),
        current_index=0,
        polygon=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        context=np.zeros(8, dtype=np.float32),
        action_nodes=np.array([1, 0], dtype=np.int64), action_yaws=np.array([0.0, 1.57], dtype=np.float32), goals=goals,
        epoch="epoch-a",
        revision=2,
    )
    node_ids[1] = 99
    goals[1, 0] = 99.0

    assert observation.node_ids.tolist() == [7, 9]
    assert observation.goals[1].tolist() == pytest.approx([5.0, 6.0, 1.57])
    assert observation.goals[observation.action_nodes[0]].tolist() == pytest.approx([5.0, 6.0, 1.57])
    with pytest.raises(ValueError):
        observation.action_nodes[0] = 0
    with pytest.raises(ValueError):
        observation.goals[1, 0] = 0.0


def test_action_nodes_address_graph_nodes_not_goal_rows():
    # Comparing action node indices to goal rows rejects a valid one-action
    # decision on a 200-node graph and destroys action/world-goal identity.
    common = dict(node_ids=np.arange(200), positions=np.zeros((200, 2)),
                  features=np.zeros((200, 19)), edges=np.empty((0, 2)),
                  edge_lengths=np.empty(0), current_index=0,
                  polygon=np.zeros((3, 2)), context=np.zeros(8),
                  action_yaws=np.array([0.0]), goals=np.array([[3., 4., .5]]),
                  epoch="epoch", revision=1)
    valid = DecisionObservation(action_nodes=np.array([199]), **common)
    assert valid.action_nodes.tolist() == [199]
    assert valid.goals.shape == (1, 3)
    with pytest.raises(ValueError, match="graph nodes"):
        DecisionObservation(action_nodes=np.array([200]), **common)


def test_store_consumes_generated_get_policy_map_wire_geometry_and_anchor():
    # The actual ROS wire has Point/Pose and split start-connection arrays;
    # tuple placeholders hide broken client decoding.
    from lunar_planning_msgs.msg import PolicyMapTile
    from lunar_planning_msgs.srv import GetPolicyMap

    response = GetPolicyMap.Response()
    response.ready = True
    response.reason_code = "READY"
    response.epoch = "wire-epoch"
    response.fine_revision = 1
    response.full_snapshot = True
    response.frame_id = "map"
    response.resolution_m = 0.2
    response.origin.x, response.origin.y, response.origin.z = 1.0, 2.0, 3.0
    response.anchor_pose.position.x, response.anchor_pose.position.y = 4.0, 5.0
    response.anchor_pose.orientation.w = np.cos(.25)
    response.anchor_pose.orientation.z = np.sin(.25)
    response.start_connection_x, response.start_connection_y = [7], [9]
    response.start_connection_status = 1
    response.local_bounds = [-256, -256, 256, 256]
    response.profile_hash = "profile"
    tile = PolicyMapTile()
    tile.tile_x, tile.tile_y = 0, 0
    tile.states = [BLOCKED] * (256 * 256)
    tile.intrinsic_states = [FREE] * (256 * 256)
    tile.observed = [FREE] * (256 * 256)
    tile.costs = [.5] * (256 * 256)
    tile.elevation_m = [1.] * (256 * 256)
    response.tiles = [tile]

    snapshot = PolicyMapStore().apply(response)
    assert snapshot.origin == pytest.approx((1.0, 2.0, 3.0))
    assert snapshot.pose == Pose(4.0, 5.0, pytest.approx(.5))
    assert snapshot.start_connections.tolist() == [[7, 9]]


def test_platform_config_reads_canonical_wheel_limits_and_actor_context_scaling():
    # Duplicating platform speed/spin limits in training config would drift
    # action context away from the actual wheel navigation profile.
    config = load_platform_config()
    context = config.actor_context(Pose(0.0, 0.0, np.pi / 2),
                                   linear_speed_mps=0.1,
                                   angular_speed_radps=0.5,
                                   sensor_range_m=10.0,
                                   sensor_fov_rad=np.pi / 2)
    assert config.maximum_forward_speed_mps == pytest.approx(0.2)
    assert config.maximum_spin_rate_radps == pytest.approx(1.0)
    assert context.tolist() == pytest.approx([0.0, 1.0, 0.5, 0.5,
                                               1.0, 0.5, config.footprint_radius_m,
                                               1.0])
