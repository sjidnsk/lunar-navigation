"""Forward explicit mission goals through PlanMotion to the WHEELED controller."""

from __future__ import annotations

from uuid import uuid4

from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from lunar_navigation_msgs.msg import ExplorationTask
from lunar_planning_msgs.action import PlanMotion
from lunar_planning_msgs.msg import GoalRegion, MotionReference

from .coordinator import ActiveMission, ExecutionGoal, PlanResult, decide_plan_result, make_plan_request
from .coordinator_node_support import make_ros_plan_goal


class TaskExecutionCoordinatorNode(Node):
    """The only component that owns PlanMotion Action client requests."""

    def __init__(self) -> None:
        super().__init__("luna_task_execution_coordinator")
        self.declare_parameter("exploration_task_topic", "/mission/exploration_task")
        self.declare_parameter("execution_goal_topic", "/mission/execution_goal")
        self.declare_parameter("plan_motion_action", "/plan_motion")
        self.declare_parameter("reference_topic", "/execution/wheeled_reference")
        self._mission: ActiveMission | None = None
        self._client = ActionClient(
            self, PlanMotion, str(self.get_parameter("plan_motion_action").value)
        )
        self._references = self.create_publisher(
            MotionReference,
            str(self.get_parameter("reference_topic").value),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(
            ExplorationTask,
            str(self.get_parameter("exploration_task_topic").value),
            self._on_task,
            10,
        )
        self.create_subscription(
            GoalRegion,
            str(self.get_parameter("execution_goal_topic").value),
            self._on_execution_goal,
            10,
        )

    def _on_task(self, task: ExplorationTask) -> None:
        if task.desired_state == ExplorationTask.ACTIVE and task.mission_id:
            self._mission = ActiveMission(task.mission_id, task.revision)
        else:
            self._mission = None
            self._publish_cancel()

    def _on_execution_goal(self, region: GoalRegion) -> None:
        if self._mission is None:
            self.get_logger().warning("execution goal rejected: no active mission")
            self._publish_cancel()
            return
        if not self._client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warning("execution goal rejected: PlanMotion unavailable")
            self._publish_cancel()
            return
        request = make_plan_request(
            self._mission, ExecutionGoal(region), request_id=f"wheel/{uuid4()}"
        )
        future = self._client.send_goal_async(make_ros_plan_goal(request))
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future: object) -> None:
        try:
            handle = future.result()
        except Exception as error:  # rclpy future exception boundary
            self.get_logger().error(f"PlanMotion request failed: {error}")
            self._publish_cancel()
            return
        if not handle.accepted:
            self._publish_cancel()
            return
        handle.get_result_async().add_done_callback(self._on_plan_result)

    def _on_plan_result(self, future: object) -> None:
        try:
            result = future.result().result
            decision = decide_plan_result(
                PlanResult(
                    has_reference=result.has_reference,
                    execution_directive=result.execution_directive,
                    reference=result.reference,
                )
            )
        except Exception as error:  # Action/result conversion boundary
            self.get_logger().error(f"PlanMotion result failed: {error}")
            self._publish_cancel()
            return
        if decision.reference is None:
            self._publish_cancel()
        else:
            self._references.publish(decision.reference)

    def _publish_cancel(self) -> None:
        self._references.publish(MotionReference())


def main() -> None:
    import rclpy

    rclpy.init()
    node = TaskExecutionCoordinatorNode()
    try:
        rclpy.spin(node)
    finally:
        node._publish_cancel()
        node.destroy_node()
        rclpy.shutdown()
