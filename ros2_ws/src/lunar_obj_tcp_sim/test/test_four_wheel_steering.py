import math
import numpy as np
import pytest
@pytest.mark.parametrize('v,w', [(.2,0),(-.2,0),(0,.5),(0,-.5),(.2,.3),(-.2,.3),(.05,.5)])
def test_each_wheel_velocity_matches_rigid_body_motion(v,w):
    from lunar_obj_tcp_sim.kinematics import FourWheelSteering
    model = FourWheelSteering(.319,.67,.8175)
    result = model.solve(v,w)
    for (x,y), speed, angle in zip(model.positions, result.wheel_speeds, result.turn_angles):
        assert [speed*.1595*math.cos(angle),speed*.1595*math.sin(angle)] == pytest.approx([v-w*y,w*x])
        assert abs(angle)<=math.pi/2


def test_saturation_preserves_curvature_and_stop_holds_steering():
    from lunar_obj_tcp_sim.kinematics import FourWheelSteering
    model=FourWheelSteering(.319,.67,.8175,max_wheel_speed_radps=.5)
    result=model.solve(.2,.4)
    assert result.saturated and max(map(abs,result.wheel_speeds))==pytest.approx(.5)
    for (x,y),speed,angle in zip(model.positions,result.wheel_speeds,result.turn_angles):
        assert speed*.1595*np.array([math.cos(angle),math.sin(angle)]) == pytest.approx(result.scale*np.array([.2-.4*y,.4*x]))
    stopped=model.solve(0,0)
    assert stopped.wheel_speeds==(0.,)*4
    assert stopped.turn_angles==result.turn_angles


def test_separate_wire_orders_signs_and_offsets():
    from lunar_obj_tcp_sim.kinematics import FourWheelSteering
    standard=FourWheelSteering(.319,.67,.8175).solve(.2,.4)
    model=FourWheelSteering(.319,.67,.8175,wheel_order=(3,2,1,0),steering_order=(1,0,3,2),wheel_signs=(-1,1,-1,1),steering_signs=(1,-1,1,-1),steering_zero_rad=(.1,.2,.3,.4))
    result=model.solve(.2,.4)
    assert result.wheel_speeds==pytest.approx([-standard.wheel_speeds[3],standard.wheel_speeds[2],-standard.wheel_speeds[1],standard.wheel_speeds[0]])
    assert result.turn_angles==pytest.approx([standard.turn_angles[1]+.1,-standard.turn_angles[0]+.2,standard.turn_angles[3]+.3,-standard.turn_angles[2]+.4])
    assert model.aligned(result,result.turn_angles)
    assert not model.aligned(result,(0.,)*4)
    assert not model.aligned(result,None)


def test_unachievable_steering_is_reported_not_silently_clamped():
    from lunar_obj_tcp_sim.kinematics import FourWheelSteering
    with pytest.raises(ValueError,match='steering'):
        FourWheelSteering(.319,.67,.8175,max_steering_angle_rad=.3).solve(0,.5)
