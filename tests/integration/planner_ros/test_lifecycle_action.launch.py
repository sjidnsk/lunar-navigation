from pathlib import Path
import unittest

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import launch_testing
import pytest


@pytest.mark.launch_test
def generate_test_description():
    executable = (
        Path(get_package_prefix("lunar_planner_ros"))
        / "lib"
        / "lunar_planner_ros"
        / "lunar_planner_ros_plan_motion_server_test"
    )
    lifecycle_action_tests = ExecuteProcess(
        cmd=[
            str(executable),
            "--gtest_color=no",
            "--gtest_filter="
            "PlanMotionServerTest.UsesFiveDistinctMutuallyExclusiveCallbackGroups:"
            "PlanMotionServerTest.ConfigureFailsWhenAnyRequiredTimeLimitIsMissing:"
            "PlanMotionServerTest.ConfiguresAndActivatesWithExplicitSnapshotPolicy:"
            "PlanMotionServerTest.RejectsGoalsWhileInactive:"
            "PlanMotionServerTest.ReturnsNoRouteAsSucceededActionWithEmptyReference:"
            "PlanMotionServerTest.InternalResultInvariantUsesRecoverableErrorPath",
        ],
        output="screen",
    )
    return (
        LaunchDescription(
            [
                lifecycle_action_tests,
                launch_testing.util.KeepAliveProc(),
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"lifecycle_action_tests": lifecycle_action_tests},
    )


class TestLifecycleActionProcess(unittest.TestCase):
    def test_process_completes(self, proc_info, lifecycle_action_tests):
        proc_info.assertWaitForShutdown(lifecycle_action_tests, timeout=30.0)


@launch_testing.post_shutdown_test()
class TestLifecycleActionExit(unittest.TestCase):
    def test_exit_code(self, proc_info, lifecycle_action_tests):
        launch_testing.asserts.assertExitCodes(
            proc_info, process=lifecycle_action_tests
        )
