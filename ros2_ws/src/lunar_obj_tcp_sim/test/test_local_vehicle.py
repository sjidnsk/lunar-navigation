import socket
import time

import pytest

from lunar_obj_tcp_sim.local_vehicle import LocalVehicle
from lunar_obj_tcp_sim.protocol import encode_command, FeedbackParser


def test_local_vehicle_uses_geometry_follows_height_and_stops_without_commands():
    vehicle = LocalVehicle((0., 0., 2.), radius=.2, track=.8, height_at=lambda x,y:2+x,
                           command_timeout=.15)
    parser = FeedbackParser()
    feedback = []
    try:
        with socket.create_connection(('127.0.0.1', vehicle.port), timeout=2) as connection:
            connection.settimeout(2)
            packet = encode_command((1., 1., 1., 1.))
            connection.sendall(packet[:9]); connection.sendall(packet[9:])
            end = time.monotonic()+.7
            while time.monotonic()<end:
                feedback += parser.feed(connection.recv(4096))
            assert vehicle.commands[0] == pytest.approx((.2, 0))
            assert any(item.linear_velocity[0] > 0 for item in feedback)
            assert feedback[-1].linear_velocity[0] == pytest.approx(0)
            assert feedback[-1].position_m[2] == pytest.approx(2+feedback[-1].position_m[0])
            assert not vehicle.errors
    finally:
        vehicle.close()


def test_local_vehicle_rejects_bad_packet():
    vehicle = LocalVehicle((0.,0.,0.))
    try:
        with pytest.raises(ValueError, match='invalid'):
            vehicle.decode(bytes(57))
    finally:
        vehicle.close()


@pytest.mark.parametrize('v,w', [(0.,.2),(0.,-.2),(.15,.15)])
def test_four_wheel_tcp_loop_turns_after_steering_feedback(v,w):
    from lunar_obj_tcp_sim.kinematics import FourWheelSteering
    from lunar_obj_tcp_sim.transport import TcpVehicleTransport
    model=FourWheelSteering(.319,.67,.8175)
    vehicle=LocalVehicle((0.,0.,0.))
    transport=TcpVehicleTransport('127.0.0.1',vehicle.port,send_rate_hz=50.)
    aligned=False
    try:
        transport.start()
        deadline=time.monotonic()+3.
        while time.monotonic()<deadline:
            result=model.solve(v,w)
            feedback=transport.latest_feedback()
            ready=model.aligned(result,feedback.feedback.turn_angles if feedback else None)
            aligned |= ready
            transport.set_wheel_command(result.wheel_speeds if ready else (0.,)*4,result.turn_angles)
            if abs(vehicle.yaw)>.07:
                break
            time.sleep(.02)
        assert aligned and vehicle.yaw*w>0 and abs(vehicle.yaw)>.07
        if v==0:
            assert abs(vehicle.position[0])+abs(vehicle.position[1])<1e-6
        else:
            assert vehicle.position[0]>.01
        assert not vehicle.errors
    finally:
        transport.close()
        vehicle.close()
