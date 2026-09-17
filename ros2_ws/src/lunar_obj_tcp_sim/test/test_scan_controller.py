from dataclasses import replace
import json
import pytest
import rclpy
from std_msgs.msg import String
from lunar_obj_tcp_sim.scan_controller import ScanControllerNode,scan_policy
from lunar_pure_wheeled_controller.execution import PathExecutor
from lunar_pure_wheeled_controller.tracking import TrackingPolicy,TrackingState


def test_formal_executor_scan_rotation_respects_reduced_speed_and_acceleration():
    normal=TrackingPolicy()
    limited=scan_policy(normal,.1,.1)
    executor=PathExecutor(limited)
    executor.set_path(((0,0,0),(0,0,1.)))
    yaw=w=0.
    for _ in range(60):
        result=executor.update(TrackingState(0,0,yaw,angular_radps=w),.05)
        command=result.command.angular_z_radps
        assert abs(command)<=.1+1e-9
        assert abs(command-w)<=.1*.05+1e-9
        w=command;yaw+=w*.05
    assert yaw>.1
    assert normal.max_angular_radps>.1
    assert limited.goal_position_tolerance_m==normal.goal_position_tolerance_m


def test_scan_wrapper_restores_original_formal_policy_only_after_scan_completion():
    rclpy.init(args=['--ros-args','-p','scan_profile_enabled:=true',
                     '-p','command_topic:=/scan_profile_test/cmd_vel'])
    node=ScanControllerNode()
    try:
        original=node._normal_policy
        assert node._executor.policy.max_angular_radps==.1
        assert node._executor.policy.max_angular_accel_radps2==.1
        node.on_scan_status(String(data=json.dumps({'phase':'BOOTSTRAP_FAILED'})))
        assert node._executor.policy.max_angular_radps==.1
        node.on_scan_status(String(data=json.dumps({'phase':'EXPLORATION_TASK_STARTED'})))
        assert node._executor.policy is original and node._policy is original
    finally:
        node.destroy_node();rclpy.shutdown()
