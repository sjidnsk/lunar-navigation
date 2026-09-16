import math
import pytest
from lunar_obj_tcp_sim.turn_test import summarize


def test_summary_separates_no_motion_from_signed_turn_and_does_not_calibrate_gain():
    samples = []
    for phase, sign in [('stationary',0), ('positive_turn',1), ('negative_turn',-1)]:
        for n in range(10):
            angle = -sign*n*.01
            samples.append(dict(phase=phase,t=n*.1,
                                orientation_xyzw=(0,0,math.sin(angle/2),math.cos(angle/2)),
                                angular_velocity=(0,sign*5.,0),wheel_speeds=(0,)*4))
    result = summarize(samples)
    assert result['stationary']['yaw_delta_deg'] == 0
    assert result['stationary']['yaw_rad_per_raw_y'] is None
    assert result['positive_turn']['yaw_delta_deg'] == pytest.approx(math.degrees(.09))
    assert result['negative_turn']['yaw_delta_deg'] == pytest.approx(-math.degrees(.09))
    assert result['positive_turn']['yaw_rad_per_raw_y'] == pytest.approx(.02)
