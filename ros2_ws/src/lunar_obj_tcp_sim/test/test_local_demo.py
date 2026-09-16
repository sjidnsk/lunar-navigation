"""Opt-in test of the user-facing shell entry and RViz's goal topic."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('LUNAR_RUN_TCP_E2E') != '1',
                               reason='requires built ROS overlay')


def test_local_script_rviz_goal_and_ordered_stop(tmp_path):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from action_msgs.msg import GoalStatusArray, GoalStatus
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from lunar_planning_msgs.action import NavigateToPose
    from lunar_planning_msgs.msg import PathReference
    from lunar_obj_tcp_sim.prepare import prepare_map

    repo = Path(__file__).resolve().parents[4]
    obj = tmp_path/'terrain.obj'
    obj.write_text('v -20 -20 0\nv 20 -20 0\nv 20 20 0\nv -20 20 0\nf 1 2 3\nf 1 3 4\n')
    prepare_map(obj, tmp_path/'map', center=(0,0),size=8,resolution=.2,halo=12,scale=1,axes='x,y,z')
    command = ['bash',str(repo/'scripts/simulation/run_local_rviz_demo.sh'),
               '--mode','nav','--map',str(tmp_path/'map'),
               '--domain-id',os.environ.get('ROS_DOMAIN_ID','176')]
    if os.environ.get('LUNAR_LOCAL_RVIZ') != '1':
        command.append('--no-rviz')
    log_path = tmp_path/'local-demo.log'
    log = log_path.open('w')
    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rclpy.init()
    node = Node('local_demo_goal_probe')
    executor = SingleThreadedExecutor(); executor.add_node(node)
    observation = {}; statuses = []; references = []; odometry = []
    qos = QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String,'/lunar_sim/observation_status',lambda m:observation.update(json.loads(m.data)),qos)
    node.create_subscription(GoalStatusArray,'/lunar_sim/navigate_to_pose/_action/status',statuses.append,qos)
    node.create_subscription(PathReference,'/lunar_sim/path_reference',references.append,qos)
    node.create_subscription(Odometry,'/lunar_sim/odometry',odometry.append,10)
    goal_pub = node.create_publisher(PoseStamped,'/lunar_sim/goal_pose',10)
    def wait(predicate, seconds=30):
        end = time.monotonic()+seconds
        while time.monotonic()<end:
            executor.spin_once(timeout_sec=.02)
            assert process.poll() is None, log_path.read_text()
            if predicate():return
        raise AssertionError(log_path.read_text())
    try:
        wait(lambda:observation.get('phase')=='KNOWN_MAP_READY' and goal_pub.get_subscription_count())
        goal = PoseStamped();goal.header.frame_id='map';goal.header.stamp=node.get_clock().now().to_msg()
        goal.pose.position.x=1.;goal.pose.orientation.w=1.
        goal_pub.publish(goal)
        wait(lambda:any(item.status==GoalStatus.STATUS_SUCCEEDED for m in statuses for item in m.status_list))
        done = next(item for m in statuses for item in m.status_list if item.status==GoalStatus.STATUS_SUCCEEDED)
        client = node.create_client(NavigateToPose.Impl.GetResultService,'/lunar_sim/navigate_to_pose/_action/get_result')
        wait(client.service_is_ready)
        request = NavigateToPose.Impl.GetResultService.Request();request.goal_id=done.goal_info.goal_id
        future = client.call_async(request);wait(future.done)
        assert future.result().result.reason_code=='GOAL_REACHED'
        assert any(r.state==r.ACTIVE and r.path.poses for r in references)
        assert abs(odometry[-1].pose.pose.position.x-1.)<.25
        assert abs(odometry[-1].twist.twist.linear.x)<.01
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=12)
        finally:
            log.close();executor.shutdown();node.destroy_node();rclpy.shutdown()
    content = log_path.read_text()
    assert process.returncode==0, content
    assert 'ordered ROS stop confirmed' in content, content
    assert 'Traceback' not in content and 'process has died' not in content, content
