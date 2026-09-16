import math
import numpy as np
from lunar_obj_tcp_sim.coordinates import orientation, wire_orientation, world_position, PoseRates
from lunar_obj_tcp_sim.geometry import rotation_matrix


def test_upright_body_mount_and_sensor():
    q = orientation((0, math.sqrt(.5), -math.sqrt(.5), 0))
    np.testing.assert_allclose(rotation_matrix(q), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation_matrix(q) @ [0, 0, 1.5], [0, 0, 1.5])
    assert world_position((1, 2, 3)) == (1, -2, 3)


def test_rotations_match_independent_basis_matrix_and_roundtrip():
    world = np.diag([1, -1, 1])
    body = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    for q in np.random.default_rng(41).normal(size=(100, 4)):
        actual = rotation_matrix(orientation(q))
        np.testing.assert_allclose(actual, world @ rotation_matrix(q) @ body.T, atol=1e-12)
        np.testing.assert_allclose(rotation_matrix(wire_orientation(orientation(q))), rotation_matrix(q), atol=1e-12)


def test_pose_rates_forward_left_right_and_yaw_wrap():
    rates = PoseRates()
    def q(yaw):
        return (0, 0, math.sin(yaw/2), math.cos(yaw/2))
    assert rates.update((0, 0, 0), q(0), 1.) is None
    linear, angular = rates.update((.01, 0, 0), q(0), 1.1)
    np.testing.assert_allclose(linear, [.1, 0, 0], atol=1e-10)
    np.testing.assert_allclose(angular, [0, 0, 0], atol=1e-10)
    _, angular = rates.update((.01, 0, 0), q(.02), 1.2)
    np.testing.assert_allclose(angular, [0, 0, .2], atol=1e-10)
    _, angular = rates.update((.01, 0, 0), q(0), 1.3)
    np.testing.assert_allclose(angular, [0, 0, -.2], atol=1e-10)
    rates = PoseRates()
    rates.update((0, 0, 0), q(math.pi-.01), 1.)
    _, angular = rates.update((0, 0, 0), q(-math.pi+.01), 1.1)
    np.testing.assert_allclose(angular, [0, 0, .2], atol=1e-10)
    assert rates.update((0, 0, 0), q(0), 2.) is None


def test_confirmed_actor_forward_and_up_match_ros_axes():
    q = (-.12305014580488205, -.6443171501159668, .7122557759284973, .24981138110160828)
    raw = rotation_matrix(q)
    ros = rotation_matrix(orientation(q))
    world = np.diag([1, -1, 1])
    np.testing.assert_allclose(ros[:, 0], world @ (-raw[:, 0]))
    np.testing.assert_allclose(ros[:, 2], world @ (-raw[:, 1]))
    assert (ros @ [0, 0, 1.5])[2] > 1.45


def test_linear_rates_resist_bunched_tcp_samples_and_reset_on_gap():
    rates = PoseRates()
    q = (0, 0, 0, 1)
    rates.update((0, 0, 0), q, 1.)
    for i in range(1, 5):
        rates.update((i*.01, 0, 0), q, 1+i*.05)
    linear, _ = rates.update((.05, 0, 0), q, 1.2006)
    assert 0 < linear[0] < .4
    assert rates.update((1., 0, 0), q, 2.) is None
    linear, _ = rates.update((1., 0, 0), q, 2.05)
    np.testing.assert_allclose(linear, [0, 0, 0])
