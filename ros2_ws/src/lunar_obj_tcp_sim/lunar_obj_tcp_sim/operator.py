"""Supervise the launch process and coordinate a bounded ROS shutdown."""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time


def _twist_is_stopped(twist, tolerance=1e-3):
    return all(
        abs(value) <= tolerance
        for value in (
            twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.x, twist.angular.y, twist.angular.z,
        )
    )


def _cancel_response_has_no_active_goals(response, error_none):
    return response.return_code == error_none and not response.goals_canceling


def _receipt_is_fresh(received_at, started_at, now, max_age_s=1.0):
    return received_at >= started_at and 0.0 <= now - received_at <= max_age_s


def finish_process_group(process, stop_client, timeout_s, signal_group=os.killpg):
    """Request a ROS stop before interrupting only the owned process group."""
    deadline = time.monotonic() + timeout_s
    ros_timeout = max(0.0, timeout_s - 1.0)
    confirmed = False
    try:
        confirmed = stop_client.request_and_confirm_stop(ros_timeout)
    except Exception as error:  # shutdown must still reap the owned group
        print(f"WARNING: ordered ROS stop failed: {error}", file=sys.stderr)
    if not confirmed:
        print("WARNING: stop was not confirmed before shutdown timeout", file=sys.stderr)
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        try:
            signal_group(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                signal_group(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return confirmed


class _UnavailableStopClient:
    def request_and_confirm_stop(self, _timeout_s):
        return False


class RosStopClient:
    """Transient ROS client which observes stop evidence but publishes no Twist."""

    def __init__(self, prefix, mode):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        from action_msgs.msg import GoalStatus, GoalStatusArray
        from action_msgs.srv import CancelGoal
        from geometry_msgs.msg import Twist
        from lunar_pure_exploration_msgs.msg import (
            PureExplorationStatus, PureExplorationTask,
        )
        from nav_msgs.msg import Odometry

        self.rclpy = rclpy
        self.CancelGoal = CancelGoal
        self.GoalStatus = GoalStatus
        self.PureExplorationTask = PureExplorationTask
        self.context = rclpy.context.Context()
        rclpy.init(
            args=[], context=self.context,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self.node = rclpy.create_node("obj_tcp_shutdown_client", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.prefix = prefix.rstrip("/")
        self.mode = mode
        self.started_at = 0.0
        self.odom_stopped = False
        self.command_stopped = False
        self.odom_received_at = 0.0
        self.command_received_at = 0.0
        self.exploration_terminal = mode == "nav"
        self.goals_terminal = False
        self.task_publisher = self.node.create_publisher(
            PureExplorationTask, self.prefix + "/exploration/task", 10
        )
        self.cancel_client = self.node.create_client(
            CancelGoal, self.prefix + "/navigate_to_pose/_action/cancel_goal"
        )
        self.node.create_subscription(
            Odometry, self.prefix + "/odometry", self._on_odometry, 10
        )
        self.node.create_subscription(
            Twist, self.prefix + "/cmd_vel", self._on_command, 10
        )
        self.node.create_subscription(
            PureExplorationStatus,
            self.prefix + "/exploration/status",
            self._on_exploration,
            10,
        )
        self.node.create_subscription(
            GoalStatusArray,
            self.prefix + "/navigate_to_pose/_action/status",
            self._on_goal_status,
            10,
        )

    def _on_odometry(self, message):
        from rclpy.time import Time

        self.odom_received_at = time.monotonic()
        clock = self.node.get_clock()
        stamp = Time.from_msg(message.header.stamp, clock_type=clock.clock_type)
        age_ns = (clock.now() - stamp).nanoseconds
        header_fresh = -100_000_000 <= age_ns <= 1_000_000_000
        self.odom_stopped = header_fresh and _twist_is_stopped(message.twist.twist)

    def _on_command(self, message):
        self.command_received_at = time.monotonic()
        self.command_stopped = _twist_is_stopped(message)

    def _on_exploration(self, message):
        terminal_for_task = (
            message.task_id == "obj-tcp-exploration"
            and message.state in {message.IDLE, message.COMPLETED, message.ERROR}
        )
        canceled_to_idle = (
            not message.task_id
            and message.state == message.IDLE
            and message.reason_code == "CANCELED"
        )
        self.exploration_terminal = terminal_for_task or canceled_to_idle

    def _on_goal_status(self, message):
        terminal = {
            self.GoalStatus.STATUS_SUCCEEDED,
            self.GoalStatus.STATUS_CANCELED,
            self.GoalStatus.STATUS_ABORTED,
        }
        self.goals_terminal = all(item.status in terminal for item in message.status_list)

    def request_and_confirm_stop(self, timeout_s):
        self.started_at = time.monotonic()
        self.odom_stopped = False
        self.command_stopped = False
        self.odom_received_at = 0.0
        self.command_received_at = 0.0
        task = self.PureExplorationTask()
        task.header.stamp = self.node.get_clock().now().to_msg()
        task.task_id = "obj-tcp-exploration"
        task.command = task.CANCEL
        cancel_future = None
        cancel_response_handled = False
        published_cancel = False
        next_publish = self.started_at
        deadline = self.started_at + timeout_s
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
            now = time.monotonic()
            if self.task_publisher.get_subscription_count() > 0 and now >= next_publish:
                self.task_publisher.publish(task)
                published_cancel = True
                next_publish = now + 0.25
            if published_cancel and cancel_future is None and self.cancel_client.service_is_ready():
                cancel_future = self.cancel_client.call_async(self.CancelGoal.Request())
            if (
                cancel_future is not None
                and cancel_future.done()
                and not cancel_response_handled
            ):
                response = cancel_future.result()
                cancel_response_handled = True
                if response is not None and _cancel_response_has_no_active_goals(
                    response, self.CancelGoal.Response.ERROR_NONE
                ):
                    self.goals_terminal = True
            if (
                self.exploration_terminal
                and self.goals_terminal
                and self.odom_stopped
                and self.command_stopped
                and _receipt_is_fresh(self.odom_received_at, self.started_at, now)
                and _receipt_is_fresh(self.command_received_at, self.started_at, now)
            ):
                print("ordered ROS stop confirmed: tasks terminal and vehicle feedback stopped")
                return True
        return False

    def close(self):
        self.executor.remove_node(self.node)
        self.executor.shutdown(timeout_sec=0.5)
        self.node.destroy_node()
        self.rclpy.shutdown(context=self.context)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="/lunar_sim")
    parser.add_argument("--mode", choices=("explore", "nav"), default="explore")
    parser.add_argument("--shutdown-timeout", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a launch command is required after --")
    child = subprocess.Popen(command, start_new_session=True)
    stopping = threading.Event()

    def request_stop(_signum, _frame):
        stopping.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while child.poll() is None and not stopping.wait(0.1):
        pass
    original_returncode = child.poll()
    client = None
    try:
        try:
            client = RosStopClient(args.prefix, args.mode)
        except Exception as error:
            print(f"WARNING: shutdown client unavailable: {error}", file=sys.stderr)
        finish_process_group(
            child, client or _UnavailableStopClient(), args.shutdown_timeout
        )
    finally:
        if client is not None:
            client.close()
    return original_returncode if original_returncode is not None else child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
