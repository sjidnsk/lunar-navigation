"""interface-v1 旧策略的 ROS 2 生命周期闭环节点。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

from grid_map_msgs.msg import GridMap
from lunar_navigation_msgs.msg import (
    ExplorationTask,
    LocalizationStatus,
    MotionExecutionFeedback,
)
from lunar_planning_msgs.action import PlanMotion
from lunar_planning_msgs.msg import MotionReference
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2
from lunar_policy_training.environment.visibility import (
    NativeVisibilityEstimator,
    SensorGeometry,
)

from .action_selection import DeterministicPolicy
from .coordinator import (
    ClosedLoopCoordinator,
    CoordinatorError,
    CoordinatorState,
    PlannerResult,
)
from .cpp_projection import CppV3Projector
from .grid_map_runtime import DecodedGridMap, decode_grid_map
from .inference import OnnxPolicyRuntime
from .ros_runtime import feedback_from_ros, map_pose_from_odom, planner_goal_to_ros
from .snapshot_assembler import MissionDefinition, SnapshotAssembler
from lunar_external_adapter.profile import load_interface_profile


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _yaw(quaternion) -> float:
    values = (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("quaternion must be finite")
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("quaternion must be normalized")
    sin_yaw = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cos_yaw = 1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2)
    return math.atan2(sin_yaw, cos_yaw)


class InterfaceV1PolicyNode(LifecycleNode):
    """旧模型每次只授权一个目标，执行完成后等待新状态再滚动决策。"""

    def __init__(self) -> None:
        super().__init__("lunar_interface_v1_policy")
        self.declare_parameter("model_dir", "/opt/lunar_navigation/models/interface_v1")
        self.declare_parameter(
            "platform_profile_file", "/etc/lunar_navigation/platform_profile.yaml"
        )
        self.declare_parameter(
            "interface_profile_file", "/etc/lunar_navigation/interface_profile.yaml"
        )
        self.declare_parameter("repository_root", "/opt/lunar_navigation/src/lunar-navigation")
        self.declare_parameter("decision_period_s", 0.1)
        self._lock = threading.RLock()
        self._enabled = False
        self._subscriptions: list[Any] = []
        self._status_publisher = None
        self._reference_publisher = None
        self._timer = None
        self._action_client = None
        self._assembler = None
        self._coordinator = None
        self._platform_type = ""
        self._global_map: DecodedGridMap | None = None
        self._local_map: DecodedGridMap | None = None
        self._odometry: Odometry | None = None
        self._localization: LocalizationStatus | None = None
        self._mission: ExplorationTask | None = None
        self._map_from_odom = None
        self._last_decision_boundary: tuple[str, str, int] | None = None
        self._active_goal_handle = None
        self._last_elapsed_s = 0.0

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        del state
        try:
            model_dir = Path(self.get_parameter("model_dir").value).resolve(strict=True)
            repository_root = Path(
                self.get_parameter("repository_root").value
            ).resolve(strict=True)
            profile_file = Path(
                self.get_parameter("platform_profile_file").value
            ).resolve(strict=True)
            interface_profile = Path(
                self.get_parameter("interface_profile_file").value
            ).resolve(strict=True)
            load_interface_profile(interface_profile)
            runtime = OnnxPolicyRuntime(model_dir)
            projector = CppV3Projector(repository_root, profile_file)
            record = runtime.manifest.capability_profiles[projector.platform_type]
            if (
                record.sha256 != projector.profile_sha256
                or record.capability_version != projector.capability.capability_version
            ):
                raise ValueError("model package and selected platform profile differ")
            visibility = NativeVisibilityEstimator(
                SensorGeometry(30.0, 2.0 * math.pi), resolution_m=4.0
            )
            self._assembler = SnapshotAssembler(
                CandidateBuilderV2(visibility), projector
            )
            self._coordinator = ClosedLoopCoordinator(DeterministicPolicy(runtime))
            self._platform_type = projector.platform_type
            qos = QoSProfile(depth=10)
            mission_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._subscriptions = [
                self.create_subscription(GridMap, "/environment/map_global", self._on_global_map, qos),
                self.create_subscription(GridMap, "/environment/map_local", self._on_local_map, qos),
                self.create_subscription(Odometry, "/localization/odometry", self._on_odometry, qos),
                self.create_subscription(LocalizationStatus, "/localization/status", self._on_localization, qos),
                self.create_subscription(ExplorationTask, "/mission/exploration_task", self._on_mission, mission_qos),
                self.create_subscription(MotionExecutionFeedback, "/execution/motion_feedback", self._on_feedback, qos),
                self.create_subscription(TFMessage, "/tf", self._on_tf, qos),
            ]
            self._status_publisher = self.create_lifecycle_publisher(
                String, "/lunar/interface_v1/status", qos
            )
            # 外部执行控制器消费该参考。volatile 可避免控制器重启后重放旧命令。
            reference_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._reference_publisher = self.create_lifecycle_publisher(
                MotionReference, "/lunar/motion_reference", reference_qos
            )
            self._action_client = ActionClient(self, PlanMotion, "/plan_motion")
            period = float(self.get_parameter("decision_period_s").value)
            if not math.isfinite(period) or not 0.02 <= period <= 5.0:
                raise ValueError("decision_period_s must be in [0.02,5.0]")
            self._timer = self.create_timer(period, self._attempt_decision)
            self._publish_status("CONFIGURED")
            return TransitionCallbackReturn.SUCCESS
        except Exception as error:
            self.get_logger().error(f"interface-v1 configure failed: {error}")
            self._destroy_runtime_entities()
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        result = super().on_activate(state)
        if result == TransitionCallbackReturn.SUCCESS:
            self._enabled = True
            self._publish_status("ACTIVE")
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._enabled = False
        self._halt_current("DEACTIVATED")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        del state
        self._enabled = False
        self._destroy_runtime_entities()
        return TransitionCallbackReturn.SUCCESS

    def _destroy_runtime_entities(self) -> None:
        if self._timer is not None:
            self.destroy_timer(self._timer)
        if self._action_client is not None:
            self._action_client.destroy()
        for subscription in self._subscriptions:
            self.destroy_subscription(subscription)
        if self._status_publisher is not None:
            self.destroy_publisher(self._status_publisher)
        if self._reference_publisher is not None:
            self.destroy_publisher(self._reference_publisher)
        self._subscriptions.clear()
        self._timer = None
        self._action_client = None
        self._status_publisher = None
        self._reference_publisher = None

    def _reject(self, channel: str, error: Exception) -> None:
        self.get_logger().warning(
            f"channel={channel} reason_code=INPUT_REJECTED detail={error}"
        )

    def _on_global_map(self, message: GridMap) -> None:
        try:
            decoded = decode_grid_map(message, expected_frame="map")
            with self._lock:
                self._global_map = decoded
        except Exception as error:
            self._reject("map_global", error)

    def _on_local_map(self, message: GridMap) -> None:
        try:
            decoded = decode_grid_map(message, expected_frame="odom")
            with self._lock:
                self._local_map = decoded
        except Exception as error:
            self._reject("map_local", error)

    def _on_odometry(self, message: Odometry) -> None:
        try:
            if message.header.frame_id != "odom" or _stamp_ns(message.header.stamp) <= 0:
                raise ValueError("odometry frame/stamp is invalid")
            _yaw(message.pose.pose.orientation)
            with self._lock:
                self._odometry = message
        except Exception as error:
            self._reject("odometry", error)

    def _on_localization(self, message: LocalizationStatus) -> None:
        with self._lock:
            self._localization = message

    def _on_mission(self, message: ExplorationTask) -> None:
        try:
            if not message.mission_id or message.revision <= 0:
                raise ValueError("mission identity is invalid")
            with self._lock:
                self._mission = message
                if message.desired_state != ExplorationTask.ACTIVE:
                    self._halt_current("MISSION_NOT_ACTIVE")
        except Exception as error:
            self._reject("exploration_task", error)

    def _on_tf(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if transform.header.frame_id == "map" and transform.child_frame_id == "odom":
                try:
                    if _stamp_ns(transform.header.stamp) <= 0:
                        raise ValueError("map->odom stamp is invalid")
                    _yaw(transform.transform.rotation)
                    with self._lock:
                        self._map_from_odom = transform
                except Exception as error:
                    self._reject("tf_map_odom", error)

    def _on_feedback(self, message: MotionExecutionFeedback) -> None:
        try:
            feedback = feedback_from_ros(message)
            with self._lock:
                if self._coordinator is not None and self._coordinator.accept_feedback(feedback):
                    self._publish_status("DECISION_BOUNDARY")
        except Exception as error:
            self._reject("motion_feedback", error)

    def _ready_inputs(self):
        if not self._enabled or any(
            value is None
            for value in (
                self._global_map,
                self._local_map,
                self._odometry,
                self._localization,
                self._mission,
                self._map_from_odom,
            )
        ):
            return None
        if self._localization.status not in (
            LocalizationStatus.VALID,
            LocalizationStatus.DEGRADED,
        ):
            return None
        if self._mission.desired_state != ExplorationTask.ACTIVE:
            return None
        if self._coordinator.state not in (
            CoordinatorState.WAITING_INPUTS,
            CoordinatorState.HOLD_ERROR,
        ):
            return None
        boundary = (
            self._global_map.content_id,
            self._local_map.content_id,
            _stamp_ns(self._odometry.header.stamp),
        )
        if boundary == self._last_decision_boundary:
            return None
        return boundary

    def _attempt_decision(self) -> None:
        with self._lock:
            boundary = self._ready_inputs()
            if boundary is None:
                return
            started = time.perf_counter()
            try:
                odometry = self._odometry
                transform = self._map_from_odom.transform
                pose = odometry.pose.pose
                pose_map = map_pose_from_odom(
                    odom_xyz=(pose.position.x, pose.position.y, pose.position.z),
                    odom_yaw_rad=_yaw(pose.orientation),
                    map_from_odom_xyz=(
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    ),
                    map_from_odom_yaw_rad=_yaw(transform.rotation),
                )
                stamp_ns = _stamp_ns(odometry.header.stamp)
                robot_digest = hashlib.sha256(
                    repr((stamp_ns, pose_map.x_m, pose_map.y_m, pose_map.yaw_rad)).encode("utf-8")
                ).hexdigest()
                mission = MissionDefinition(
                    self._mission.mission_id,
                    int(self._mission.revision),
                    (
                        self._mission.roi_min_x_m,
                        self._mission.roi_min_y_m,
                        self._mission.roi_max_x_m,
                        self._mission.roi_max_y_m,
                    ),
                )
                snapshot = self._assembler.build(
                    global_map=self._global_map,
                    local_map=self._local_map,
                    pose_map=pose_map,
                    robot_state_id=robot_digest,
                    state_time_ns=stamp_ns,
                    mission=mission,
                    platform_type=self._platform_type,
                )
                goal = self._coordinator.start_decision(
                    snapshot, mission_id=mission.mission_id
                )
                self._last_decision_boundary = boundary
                if goal is None:
                    self._last_elapsed_s = time.perf_counter() - started
                    self._publish_status("NO_CANDIDATE")
                    return
                self._last_elapsed_s = time.perf_counter() - started
                self._send_goal(goal, stamp_ns)
            except CoordinatorError as error:
                self._last_elapsed_s = time.perf_counter() - started
                self._reject("decision", error)
            except Exception as error:
                self._last_elapsed_s = time.perf_counter() - started
                self._reject("snapshot", error)

    def _send_goal(self, goal, stamp_ns: int) -> None:
        if not self._action_client.server_is_ready():
            self._coordinator.accept_planner_result(
                PlannerResult(goal.request_id, False, "", "PLAN_MOTION_UNAVAILABLE")
            )
            self._publish_status("PLAN_MOTION_UNAVAILABLE")
            return
        future = self._action_client.send_goal_async(
            planner_goal_to_ros(goal, stamp_ns=stamp_ns)
        )
        future.add_done_callback(
            lambda completed, request_id=goal.request_id: self._on_goal_response(
                completed, request_id
            )
        )

    def _on_goal_response(self, future, request_id: str) -> None:
        try:
            handle = future.result()
            if not self._enabled:
                if handle.accepted:
                    handle.cancel_goal_async()
                return
            if not handle.accepted:
                self._accept_planner_result(
                    PlannerResult(request_id, False, "", "GOAL_REJECTED")
                )
                return
            self._active_goal_handle = handle
            result_future = handle.get_result_async()
            result_future.add_done_callback(
                lambda completed, expected=request_id: self._on_action_result(
                    completed, expected
                )
            )
        except Exception as error:
            self._reject("plan_motion_goal", error)
            self._accept_planner_result(
                PlannerResult(request_id, False, "", "ACTION_TRANSPORT_ERROR")
            )

    def _on_action_result(self, future, request_id: str) -> None:
        try:
            result = future.result().result
            with self._lock:
                self._active_goal_handle = None
                if not self._enabled:
                    # deactivate/reset 后到达的 Action 结果属于旧上下文，禁止下发。
                    return
                planner_result = PlannerResult(
                    request_id,
                    bool(result.has_reference),
                    result.reference.plan_id if result.has_reference else "",
                    result.reason_code,
                )
                if self._accept_planner_result(planner_result) and result.has_reference:
                    # 协调器先验证 request_id 和状态，再把同一参考交给执行器。
                    try:
                        self._reference_publisher.publish(result.reference)
                    except Exception as error:
                        self._reject("motion_reference", error)
                        self._halt_current("REFERENCE_PUBLISH_FAILED")
        except Exception as error:
            self._reject("plan_motion_result", error)
            self._accept_planner_result(
                PlannerResult(request_id, False, "", "ACTION_TRANSPORT_ERROR")
            )

    def _halt_current(self, reason: str) -> None:
        with self._lock:
            if self._active_goal_handle is not None:
                try:
                    self._active_goal_handle.cancel_goal_async()
                except Exception as error:
                    self._reject("plan_motion_cancel", error)
            self._active_goal_handle = None
            if self._coordinator is not None:
                self._coordinator.reset(reason)
            if (
                self._global_map is not None
                and self._local_map is not None
                and self._odometry is not None
            ):
                self._last_decision_boundary = (
                    self._global_map.content_id,
                    self._local_map.content_id,
                    _stamp_ns(self._odometry.header.stamp),
                )
            self._publish_status(reason)

    def _accept_planner_result(self, result: PlannerResult) -> bool:
        with self._lock:
            try:
                self._coordinator.accept_planner_result(result)
                self._publish_status(
                    "REFERENCE_AVAILABLE" if result.has_reference else result.reason_code
                )
                return True
            except CoordinatorError as error:
                self._reject("plan_motion_result", error)
                return False

    def _publish_status(self, event: str) -> None:
        if self._status_publisher is None:
            return
        state = self._coordinator.state.name if self._coordinator is not None else "UNCONFIGURED"
        message = String()
        message.data = json.dumps(
            {
                "schema_version": "lunar-interface-v1-status/v1",
                "event": event,
                "platform_type": self._platform_type,
                "state": state,
                "reason_code": (
                    self._coordinator.last_reason if self._coordinator is not None else event
                ),
                "last_decision_elapsed_s": self._last_elapsed_s,
            },
            sort_keys=True,
        )
        try:
            self._status_publisher.publish(message)
        except Exception:
            # 未激活的 lifecycle publisher 不应改变主状态机。
            pass


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = InterfaceV1PolicyNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


__all__ = ["InterfaceV1PolicyNode", "main"]
