import math
import numpy as np
import pytest
from lunar_drl_exploration.contracts import Pose
from lunar_drl_exploration.config import load_platform_config
from lunar_drl_exploration.scene import TerrainGrid
from lunar_drl_exploration.plant import KinematicPlant, GeometryFailure


def terrain():
    return TerrainGrid.from_heights(np.zeros((80,80), np.float32), .2, (-8.,-8.), load_platform_config())


def test_forward_reverse_curves_spin_and_whole_planar_motion():
    p = KinematicPlant(terrain(), Pose(0.1, 0.1, 0.0))
    for _ in range(40): p.advance(.2, .4, .05)
    assert 0 < p.pose.y < 1
    assert p.distance_m == pytest.approx(.2 * 2 - .035, abs=.01)
    assert p.turn_rad >= abs(p.pose.yaw) - 1e-12
    start = p.pose
    for _ in range(80): p.advance(-20, -20, .05)
    assert p.linear_mps == -.2
    assert abs(p.angular_radps) <= 1
    before = p.distance_m
    for _ in range(100): p.advance(0, 1, .05)
    stopped = p.pose
    for _ in range(10): p.advance(0, 1, .05)
    assert p.pose.x == stopped.x and p.pose.y == stopped.y
    assert p.distance_m >= before
    assert p.turn_rad > 5
    assert p.pose != start


def test_collision_is_explicit_and_no_motion_is_fabricated():
    h = np.zeros((80,80), np.float32)
    h[:,46:] = 2
    t = TerrainGrid.from_heights(h, .2, (-8.,-8.), load_platform_config())
    p = KinematicPlant(t, Pose(.1,.1,0.))
    for _ in range(200):
        try: p.advance(.2, 0., .05)
        except GeometryFailure: break
    else: pytest.fail('collision was missed')
    assert p.pose.x < 1.2
    assert p.linear_mps == 0
    assert p.distance_m == pytest.approx(p.pose.x - .1)


def test_motion_obeys_acceleration_and_dt():
    p = KinematicPlant(terrain(), Pose(.1,.1,0.))
    p.advance(.2,1.,.05)
    assert p.linear_mps == pytest.approx(.025)
    assert p.angular_radps == pytest.approx(.025)
    with pytest.raises(ValueError): p.advance(.2,1.,.5)
