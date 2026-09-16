"""Use real ROS publishers/subscribers and a real converted mesh."""
import time
import numpy as np
import pytest
rclpy = pytest.importorskip('rclpy')
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import MarkerArray


@pytest.mark.parametrize('initial_phase', ['WAITING_FOR_STATE', 'BOOTSTRAP_FAILED'])
def test_sensor_does_not_reobserve_or_restamp_an_old_pose(tmp_path, initial_phase):
    from lunar_obj_tcp_sim.sensor_node import ObjVirtualSensorNode
    from lunar_obj_tcp_sim.prepare import prepare_map
    obj = tmp_path/'flat.obj'
    obj.write_text('g ground\nv -16 -16 0\nv 16 -16 0\nv 16 16 0\nv -16 16 0\nf 1 2 3\nf 1 3 4\n')
    prepare_map(obj,tmp_path/'map',center=(0,0),size=8,resolution=.2,halo=12,scale=1,axes='x,y,z')
    rclpy.init()
    node=ObjVirtualSensorNode(parameter_overrides=[
        Parameter('map_directory',value=str(tmp_path/'map')),
        Parameter('auto_start',value=False), Parameter('initial_scan',value=False),
        Parameter('mode',value='explore')])
    node.phase = initial_phase
    peer=Node('sensor_probe')
    publisher=peer.create_publisher(Odometry,'/lunar_sim/odometry',10)
    received=[]; observed=[]
    peer.create_subscription(MarkerArray,"/lunar_sim/observed_cells",observed.append,
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    peer.create_subscription(GridMap,'/lunar_sim/grid_map',received.append,
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    executor=SingleThreadedExecutor(); executor.add_node(peer); executor.add_node(node)
    try:
        limit=time.monotonic()+3
        while publisher.get_subscription_count()==0 and time.monotonic()<limit:
            executor.spin_once(timeout_sec=.02)
        msg=Odometry(); msg.header.frame_id='odom'; msg.child_frame_id='base_link'
        msg.header.stamp=peer.get_clock().now().to_msg(); msg.pose.pose.orientation.w=1.
        publisher.publish(msg)
        limit=time.monotonic()+3
        while not received and time.monotonic()<limit:
            executor.spin_once(timeout_sec=.02)
        assert received, 'visible surface was not published'
        assert received[0].header.stamp == msg.header.stamp
        assert any(np.isfinite(received[0].data[0].data))
        assert any(np.isnan(received[0].data[0].data)), 'FOV unknown was filled'
        limit=time.monotonic()+.8
        while time.monotonic()<limit:
            executor.spin_once(timeout_sec=.02)
        assert len(received)==1, 'old feedback was restamped or reobserved'
        assert node.phase != 'OBSERVATION_ERROR'
        assert observed and observed[0].markers[0].points
        if initial_phase == 'BOOTSTRAP_FAILED':
            assert node.phase == 'BOOTSTRAP_FAILED'
            assert not node.task_started
    finally:
        executor.shutdown(); node.destroy_node(); peer.destroy_node(); rclpy.shutdown()


def test_sensor_does_not_start_exploration_with_only_its_own_subscriber():
    from types import SimpleNamespace
    from geometry_msgs.msg import Twist
    from lunar_obj_tcp_sim.scan_sequence import ScanSequence
    from unittest.mock import Mock
    from lunar_obj_tcp_sim.sensor_node import ObjVirtualSensorNode
    fake=SimpleNamespace(mode='explore',settings={'auto_start':True}, task_started=False,
        map_seen=True,samples=1,client=SimpleNamespace(server_is_ready=lambda:True),
        scan=ScanSequence(None,enabled=False), latest=None, command=Twist(),
        command_received=time.monotonic(), task_pub=Mock(),
        exploration_received=False,phase='OBSERVING')
    fake.task_pub.get_subscription_count.return_value=1
    ObjVirtualSensorNode.advance_task(fake)
    fake.task_pub.publish.assert_not_called()
    assert not fake.task_started
