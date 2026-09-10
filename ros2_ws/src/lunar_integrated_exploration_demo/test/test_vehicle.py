"""Real ROS-message and node boundary checks for the clock-producing vehicle."""
import math
from pathlib import Path
import sys
import time

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE.parent / 'lunar_incremental_controller_demo'))

pytest.importorskip('rclpy')
import rclpy
from builtin_interfaces.msg import Time
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rosgraph_msgs.msg import Clock


def vehicle_module():
    from lunar_integrated_exploration_demo import vehicle
    return vehicle


def test_grid_map_preserves_asymmetric_observation_coordinates_and_unknown():
    vehicle = vehicle_module()
    observation = {'origin_x': -0.4, 'origin_y': 0.6, 'resolution': 0.2,
                   'values': np.array([[1., np.nan, 3.], [4., 5., 6.]], dtype=np.float32)}
    msg = vehicle.observation_grid_map(observation, Time(sec=7, nanosec=123))
    assert msg.header.frame_id == 'odom'
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (7, 123)
    assert msg.info.length_x == pytest.approx(0.6)
    assert msg.info.length_y == pytest.approx(0.4)
    assert msg.info.pose.position.x == pytest.approx(-0.1)
    assert msg.info.pose.position.y == pytest.approx(0.8)
    assert [(dim.label, dim.size, dim.stride) for dim in msg.data[0].layout.dim] == [
        ('column_index', 2, 6), ('row_index', 3, 3)]
    assert list(msg.data[0].data[:4]) == [6., 5., 4., 3.]
    assert math.isnan(msg.data[0].data[4])
    assert msg.data[0].data[5] == 1.0
    assert msg.outer_start_index == msg.inner_start_index == 0


@pytest.mark.parametrize('change', [{'origin_x': 0.03},
                                    {'values': np.array([[math.inf]])},
                                    {'values': np.array([])}])
def test_grid_map_rejects_misaligned_origin_or_invalid_elevation(change):
    vehicle = vehicle_module()
    observation = {'origin_x': 0.0, 'origin_y': 0.0, 'resolution': 0.2,
                   'values': np.array([[1.0]])}
    observation.update(change)
    with pytest.raises(ValueError):
        vehicle.observation_grid_map(observation, Time(sec=1))


@pytest.fixture
def ros_context():
    context = Context()
    rclpy.init(args=[], context=context, domain_id=187)
    yield context
    context.shutdown()


def test_dynamic_scale_parameter_rejects_invalid_changes_and_updates_wall_period(ros_context):
    vehicle = vehicle_module()
    node = vehicle.VehicleNode(context=ros_context, parameter_overrides=[
        Parameter('time_scale', value=30.0), Parameter('use_sim_time', value=True)])
    try:
        for value in [0.0, 61.0, math.inf, math.nan, True, '30']:
            result = node.set_parameters_atomically([Parameter('time_scale', value=value)])
            assert not result.successful
            assert node.sim.clock.time_scale == 30.0
        result = node.set_parameters_atomically([Parameter('time_scale', value=1)])
        assert result.successful, result.reason
        assert node.sim.clock.time_scale == 1.0
        assert node.clock_timer.timer_period_ns == 20_000_000
        result = node.set_parameters_atomically([Parameter('time_scale', value=60.0)])
        assert result.successful, result.reason
        assert node.sim.clock.time_scale == 60.0
        assert node.clock_timer.timer_period_ns == 333_333
    finally:
        node.destroy_node()


def test_use_sim_time_clock_producer_emits_coherent_clock_pose_and_observation(ros_context):
    vehicle = vehicle_module()
    node = vehicle.VehicleNode(context=ros_context, parameter_overrides=[
        Parameter('time_scale', value=1.0), Parameter('use_sim_time', value=True)])
    observer = Node('vehicle_clock_test_observer', context=ros_context)
    stamps = set()
    poses, maps = [], []
    observer.create_subscription(Clock, '/clock', lambda msg: stamps.add(
        (msg.clock.sec, msg.clock.nanosec)), qos_profile_sensor_data)
    observer.create_subscription(Odometry, vehicle.PREFIX + '/odometry', poses.append,
                                 qos_profile_sensor_data)
    observer.create_subscription(GridMap, vehicle.PREFIX + '/grid_map', maps.append,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    executor = SingleThreadedExecutor(context=ros_context)
    executor.add_node(node)
    executor.add_node(observer)
    try:
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline and (len(poses) < 5 or not maps):
            executor.spin_once(timeout_sec=0.02)
        # Drain enough callbacks to receive the paired /clock sample.
        for _ in range(6):
            executor.spin_once(timeout_sec=0.02)
        assert len(poses) >= 5 and maps
        assert all(p.header.stamp.sec >= 1 for p in poses)
        assert (poses[0].header.stamp.sec, poses[0].header.stamp.nanosec) in stamps
        assert maps[-1].header.stamp.sec >= 1
        assert node.sim.plant.x == node.sim.plant.y == 0.0
        assert not node.get_publishers_info_by_topic(vehicle.PREFIX + '/cmd_vel')
    finally:
        executor.remove_node(observer)
        executor.remove_node(node)
        observer.destroy_node()
        node.destroy_node()
        executor.shutdown()


def test_publish_failure_keeps_unpublished_samples_and_does_not_count_a_publication(
        ros_context, monkeypatch):
    vehicle = vehicle_module()
    node = vehicle.VehicleNode(context=ros_context)
    try:
        pending = node.sim.pending_observation.snapshot()
        assert pending['sample_count'] == 1

        def failed_transport(_message):
            raise RuntimeError('transport refused the batch')

        monkeypatch.setattr(node.map_pub, 'publish', failed_transport)
        with pytest.raises(RuntimeError, match='transport refused'):
            node.publish_map()
        assert node.map_schedule.publications == 0
        assert node.sim.pending_observation.sample_count == 1
        np.testing.assert_array_equal(node.sim.pending_observation.snapshot()['values'],
                                      pending['values'])
    finally:
        node.destroy_node()
