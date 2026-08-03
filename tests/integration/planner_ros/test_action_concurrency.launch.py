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
    concurrency_tests = ExecuteProcess(
        cmd=[
            str(executable),
            "--gtest_color=no",
            "--gtest_filter="
            "PlanMotionServerTest.RejectsSecondGoalAndSerializesExplicitReplacement:"
            "PlanMotionServerTest.CooperativelyCancelsActiveGoal:"
            "PlanMotionServerTest.RejectsOldRevisionAndPausingCancelsUncommittedWork:"
            "PlanMotionServerTest.LocksNewGoalsAfterActivatingHopperReference",
        ],
        output="screen",
    )
    return (
        LaunchDescription(
            [
                concurrency_tests,
                launch_testing.util.KeepAliveProc(),
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"concurrency_tests": concurrency_tests},
    )


class TestActionConcurrencyProcess(unittest.TestCase):
    def test_process_completes(self, proc_info, concurrency_tests):
        proc_info.assertWaitForShutdown(concurrency_tests, timeout=30.0)


@launch_testing.post_shutdown_test()
class TestActionConcurrencyExit(unittest.TestCase):
    def test_exit_code(self, proc_info, concurrency_tests):
        launch_testing.asserts.assertExitCodes(
            proc_info, process=concurrency_tests
        )
