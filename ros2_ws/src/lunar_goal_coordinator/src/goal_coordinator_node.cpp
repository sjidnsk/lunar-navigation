#include "lunar_goal_coordinator/goal_coordinator_node.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <lunar_planner_ros/grid_map_adapter.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace lunar::goal_coordinator {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using GoalHandle = rclcpp_action::ClientGoalHandle<Action>;
using CallbackReturn = GoalCoordinatorNode::CallbackReturn;

std::optional<std::int64_t> Nanoseconds(
    const builtin_interfaces::msg::Time& stamp) noexcept {
  if (stamp.sec < 0 || stamp.nanosec >= 1'000'000'000U ||
      (stamp.sec == 0 && stamp.nanosec == 0U)) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
      static_cast<std::int64_t>(stamp.nanosec);
}

builtin_interfaces::msg::Time Stamp(const std::int64_t nanoseconds) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
  stamp.nanosec = static_cast<std::uint32_t>(nanoseconds % 1'000'000'000LL);
  return stamp;
}

double Yaw(const geometry_msgs::msg::Quaternion& q) noexcept {
  return std::atan2(
      2.0 * (q.w * q.z + q.x * q.y),
      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

std::string StateName(const CoordinatorState state) {
  switch (state) {
    case CoordinatorState::kIdle:
      return "IDLE";
    case CoordinatorState::kWaitingForMap:
      return "WAITING_FOR_MAP";
    case CoordinatorState::kReady:
      return "READY";
    case CoordinatorState::kPlanning:
      return "PLANNING";
    case CoordinatorState::kExecuting:
      return "EXECUTING";
    case CoordinatorState::kWaitingForSnapshot:
      return "WAITING_FOR_SNAPSHOT";
    case CoordinatorState::kCanceling:
      return "CANCELING";
    case CoordinatorState::kSucceeded:
      return "SUCCEEDED";
    case CoordinatorState::kHold:
      return "HOLD";
  }
  return "UNKNOWN";
}

std::string RandomUuid() {
  std::array<std::uint8_t, 16U> bytes{};
  std::random_device random;
  for (auto& byte : bytes) {
    byte = static_cast<std::uint8_t>(random());
  }
  bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0fU) | 0x40U);
  bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3fU) | 0x80U);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    if (index == 4U || index == 6U || index == 8U || index == 10U) {
      output << '-';
    }
    output << std::setw(2) << static_cast<unsigned>(bytes[index]);
  }
  return output.str();
}

diagnostic_msgs::msg::KeyValue KeyValue(
    std::string key, std::string value) {
  diagnostic_msgs::msg::KeyValue output;
  output.key = std::move(key);
  output.value = std::move(value);
  return output;
}

class RosPlanMotionTransport final : public PlanMotionTransport {
 public:
  RosPlanMotionTransport(
      GoalCoordinatorNode& node, const std::string& action_name,
      const rclcpp::CallbackGroup::SharedPtr& callback_group)
      : client_(rclcpp_action::create_client<Action>(
            &node, action_name, callback_group)) {}

  [[nodiscard]] bool Available() const override {
    return client_->action_server_is_ready();
  }

  void Send(const Action::Goal& goal, ResultCallback callback) override {
    auto complete_once = std::make_shared<std::atomic_bool>(false);
    rclcpp_action::Client<Action>::SendGoalOptions options;
    options.goal_response_callback =
        [this, callback, complete_once](const GoalHandle::SharedPtr handle) {
          {
            std::scoped_lock lock{mutex_};
            current_goal_ = handle;
          }
          if (!handle && !complete_once->exchange(true)) {
            PlanCompletion completion;
            completion.action_succeeded = false;
            completion.reason_code = "PLAN_MOTION_GOAL_REJECTED";
            callback(std::move(completion));
          }
        };
    options.result_callback =
        [this, callback, complete_once](const GoalHandle::WrappedResult& result) {
          {
            std::scoped_lock lock{mutex_};
            current_goal_.reset();
          }
          if (complete_once->exchange(true)) {
            return;
          }
          PlanCompletion completion;
          completion.action_succeeded =
              result.code == rclcpp_action::ResultCode::SUCCEEDED &&
              static_cast<bool>(result.result);
          completion.reason_code = completion.action_succeeded
              ? "PLAN_MOTION_SUCCEEDED"
              : "PLAN_MOTION_ACTION_FAILED";
          if (result.result) {
            completion.result = *result.result;
          }
          callback(std::move(completion));
        };
    static_cast<void>(client_->async_send_goal(goal, options));
  }

  void Cancel(CancelCallback callback) override {
    GoalHandle::SharedPtr handle;
    {
      std::scoped_lock lock{mutex_};
      handle = current_goal_;
    }
    if (!handle) {
      callback(true);
      return;
    }
    static_cast<void>(client_->async_cancel_goal(handle));
    callback(true);
  }

 private:
  rclcpp_action::Client<Action>::SharedPtr client_;
  mutable std::mutex mutex_;
  GoalHandle::SharedPtr current_goal_;
};

}  // namespace

class GoalCoordinatorNode::Impl final {
 public:
  Impl(
      GoalCoordinatorNode& node, GoalCoordinatorDependencies dependencies)
      : node_(node), dependencies_(std::move(dependencies)) {
    DeclareParameters();
    parameter_callback_ = node_.add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>&) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = !configured_;
          result.reason = configured_
              ? "COORDINATOR_PARAMETERS_FROZEN_AFTER_CONFIGURE"
              : "ACCEPTED";
          return result;
        });
  }

  CallbackReturn Configure() {
    std::scoped_lock lock{mutex_};
    try {
      ValidateParameters();
      transition_group_ = node_.create_callback_group(
          rclcpp::CallbackGroupType::MutuallyExclusive);
      action_group_ = node_.create_callback_group(
          rclcpp::CallbackGroupType::MutuallyExclusive);
      CreateRosEntities();
      if (!dependencies_.planner) {
        dependencies_.planner = std::make_shared<RosPlanMotionTransport>(
            node_, node_.get_parameter("plan_motion_action").as_string(),
            action_group_);
      }
      if (!dependencies_.target_uuid) {
        dependencies_.target_uuid = [] { return RandomUuid(); };
      }
      configured_ = true;
      active_ = false;
      return CallbackReturn::SUCCESS;
    } catch (const std::exception& error) {
      RCLCPP_ERROR(node_.get_logger(), "configure failed: %s", error.what());
      CleanupEntities();
      configured_ = false;
      return CallbackReturn::FAILURE;
    }
  }

  CallbackReturn Activate() {
    std::scoped_lock lock{mutex_};
    if (!configured_ || !dependencies_.planner) {
      return CallbackReturn::FAILURE;
    }
    mission_publisher_->on_activate();
    reference_publisher_->on_activate();
    status_publisher_->on_activate();
    active_ = true;
    PublishStatus();
    MaybeDispatch();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn Deactivate() {
    std::scoped_lock lock{mutex_};
    if (active_) {
      machine_.ForceHold("COORDINATOR_DEACTIVATED");
      RequestHoldIfNeeded();
    }
    active_ = false;
    if (mission_publisher_ && mission_publisher_->is_activated()) {
      mission_publisher_->on_deactivate();
    }
    if (reference_publisher_ && reference_publisher_->is_activated()) {
      reference_publisher_->on_deactivate();
    }
    if (status_publisher_ && status_publisher_->is_activated()) {
      status_publisher_->on_deactivate();
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn Cleanup() {
    std::scoped_lock lock{mutex_};
    if (active_) {
      (void)Deactivate();
    }
    CleanupEntities();
    dependencies_.planner.reset();
    global_map_.reset();
    local_map_stamp_ns_.reset();
    odometry_.reset();
    transform_stamp_ns_.reset();
    pending_command_.reset();
    pending_plan_goal_.reset();
    machine_ = GoalStateMachine{};
    configured_ = false;
    return CallbackReturn::SUCCESS;
  }

  void ReceiveGoal(const geometry_msgs::msg::PoseStamped& message) {
    std::scoped_lock lock{mutex_};
    if (!active_) {
      return;
    }
    const auto stamp = Nanoseconds(message.header.stamp);
    GoalPose goal{
        .frame_id = message.header.frame_id,
        .stamp_ns = stamp.value_or(0),
        .x_m = message.pose.position.x,
        .y_m = message.pose.position.y,
        .z_m = message.pose.position.z,
        .qx = message.pose.orientation.x,
        .qy = message.pose.orientation.y,
        .qz = message.pose.orientation.z,
        .qw = message.pose.orientation.w,
    };
    static_cast<void>(
        machine_.SubmitGoal(goal, dependencies_.target_uuid()));
    PublishStatus();
    MaybeDispatch();
  }

  void ReceiveGlobalMap(const grid_map_msgs::msg::GridMap& message) {
    std::scoped_lock lock{mutex_};
    if (!active_) {
      return;
    }
    auto adapted = map_adapter_.Adapt(message, "map");
    if (!adapted.ok()) {
      machine_.ForceHold(
          adapted.error ? adapted.error->reason_code
                        : "GLOBAL_MAP_INVALID");
      RequestHoldIfNeeded();
      PublishStatus();
      return;
    }
    global_map_ = std::make_shared<lunar::planning::GridMap>(
        std::move(*adapted.map));
    TryUpdateSnapshot();
  }

  void ReceiveLocalMap(const grid_map_msgs::msg::GridMap& message) {
    std::scoped_lock lock{mutex_};
    if (!active_) {
      return;
    }
    if (message.header.frame_id != "odom") {
      machine_.ForceHold("LOCAL_MAP_WRONG_FRAME");
      RequestHoldIfNeeded();
      PublishStatus();
      return;
    }
    local_map_stamp_ns_ = Nanoseconds(message.header.stamp);
    TryUpdateSnapshot();
  }

  void ReceiveOdometry(const nav_msgs::msg::Odometry& message) {
    std::scoped_lock lock{mutex_};
    if (!active_) {
      return;
    }
    const auto stamp = Nanoseconds(message.header.stamp);
    const auto& position = message.pose.pose.position;
    const auto& orientation = message.pose.pose.orientation;
    if (!stamp || message.header.frame_id != "odom" ||
        message.child_frame_id != "base_footprint" ||
        !std::isfinite(position.x) || !std::isfinite(position.y) ||
        !std::isfinite(orientation.x) || !std::isfinite(orientation.y) ||
        !std::isfinite(orientation.z) || !std::isfinite(orientation.w)) {
      machine_.ForceHold("WHEELED_ODOMETRY_INVALID");
      RequestHoldIfNeeded();
      PublishStatus();
      return;
    }
    odometry_ = RobotState{
        .stamp_ns = *stamp,
        .x_m = position.x,
        .y_m = position.y,
        .yaw_rad = Yaw(orientation),
    };
    TryUpdateSnapshot();
  }

  void ReceiveTf(const tf2_msgs::msg::TFMessage& message) {
    std::scoped_lock lock{mutex_};
    if (!active_) {
      return;
    }
    for (const auto& transform : message.transforms) {
      if (transform.header.frame_id == "map" &&
          transform.child_frame_id == "odom") {
        transform_stamp_ns_ = Nanoseconds(transform.header.stamp);
      }
    }
    TryUpdateSnapshot();
  }

  void ReceiveFeedback(
      const lunar_navigation_msgs::msg::MotionExecutionFeedback& message) {
    std::scoped_lock lock{mutex_};
    if (!active_ || message.platform_type != message.WHEELED) {
      return;
    }
    const auto stamp = Nanoseconds(message.header.stamp);
    std::optional<ExecutionState> state;
    if (message.state == message.ACCEPTED) {
      state = ExecutionState::kAccepted;
    } else if (message.state == message.EXECUTING) {
      state = ExecutionState::kExecuting;
    } else if (message.state == message.SEGMENT_COMPLETE) {
      state = ExecutionState::kSegmentComplete;
    } else if (message.state == message.FAILED) {
      state = ExecutionState::kFailed;
    } else if (message.state == message.CANCELED) {
      state = ExecutionState::kCanceled;
    }
    if (!stamp || !state) {
      machine_.ForceHold("EXECUTION_FEEDBACK_INVALID");
    } else {
      machine_.HandleExecutionEvent(ExecutionEvent{
          .stamp_ns = *stamp,
          .plan_id = message.plan_id,
          .state = *state,
          .reason_code = message.reason_code,
      });
    }
    RequestHoldIfNeeded();
    PublishStatus();
    MaybeDispatch();
  }

  void ReceiveBridgeStatus(
      const diagnostic_msgs::msg::DiagnosticArray& message) {
    std::scoped_lock lock{mutex_};
    for (const auto& status : message.status) {
      if (status.name != "lunar_unreal_tcp_bridge") {
        continue;
      }
      std::string session_id;
      std::string session_state;
      for (const auto& value : status.values) {
        if (value.key == "session_id") {
          session_id = value.value;
        } else if (value.key == "session_state") {
          session_state = value.value;
        }
      }
      if (session_id.empty() || session_state == "DISCONNECTED" ||
          session_state == "HANDSHAKING") {
        pending_command_.reset();
        pending_plan_goal_.reset();
        machine_.SetSession({});
        global_map_.reset();
        local_map_stamp_ns_.reset();
        odometry_.reset();
        transform_stamp_ns_.reset();
      } else {
        if (session_id != machine_.session_id()) {
          pending_command_.reset();
          pending_plan_goal_.reset();
        }
        machine_.SetSession(std::move(session_id));
      }
      RequestHoldIfNeeded();
      PublishStatus();
      return;
    }
  }

  bool Cancel() {
    std::scoped_lock lock{mutex_};
    if (!active_ || !machine_.BeginCancel()) {
      return false;
    }
    pending_command_.reset();
    pending_plan_goal_.reset();
    const bool has_reference = !machine_.active_plan_id().empty();
    RequestHoldIfNeeded();
    dependencies_.planner->Cancel(
        [this, has_reference](const bool canceled) {
          std::scoped_lock callback_lock{mutex_};
          if (!has_reference) {
            machine_.CompleteCancel(canceled);
          } else if (!canceled) {
            machine_.ForceHold("PLAN_MOTION_CANCEL_FAILED");
          }
          PublishStatus();
        });
    PublishStatus();
    return true;
  }

  void Dispatch() {
    std::scoped_lock lock{mutex_};
    MaybeDispatch();
  }

  CoordinatorState state() const noexcept {
    std::scoped_lock lock{mutex_};
    return machine_.state();
  }

  std::string reason() const {
    std::scoped_lock lock{mutex_};
    return machine_.reason_code();
  }

  std::optional<lunar_navigation_msgs::msg::ExplorationTask> last_mission()
      const {
    std::scoped_lock lock{mutex_};
    return last_mission_;
  }

  std::optional<lunar_planning_msgs::msg::MotionReference> last_reference()
      const {
    std::scoped_lock lock{mutex_};
    return last_reference_;
  }

  std::optional<Action::Goal> last_plan_goal() const {
    std::scoped_lock lock{mutex_};
    return last_plan_goal_;
  }

 private:
  void DeclareParameters() {
    node_.declare_parameter<std::string>("map_frame", "map");
    node_.declare_parameter<std::string>("odom_frame", "odom");
    node_.declare_parameter<std::string>("base_frame", "base_footprint");
    node_.declare_parameter<std::string>("plan_motion_action", "/plan_motion");
    node_.declare_parameter<double>(
        "position_tolerance_m", kDefaultPositionToleranceM);
    node_.declare_parameter<double>(
        "yaw_tolerance_rad", kDefaultYawToleranceRad);
    node_.declare_parameter<double>("maximum_snapshot_skew_s", 0.2);
  }

  void ValidateParameters() const {
    constexpr double tolerance = 1.0e-12;
    if (node_.get_parameter("map_frame").as_string() != "map" ||
        node_.get_parameter("odom_frame").as_string() != "odom" ||
        node_.get_parameter("base_frame").as_string() != "base_footprint" ||
        node_.get_parameter("plan_motion_action").as_string() !=
            "/plan_motion" ||
        std::abs(node_.get_parameter("position_tolerance_m").as_double() -
                 kDefaultPositionToleranceM) > tolerance ||
        std::abs(node_.get_parameter("yaw_tolerance_rad").as_double() -
                 kDefaultYawToleranceRad) > tolerance ||
        std::abs(node_.get_parameter("maximum_snapshot_skew_s").as_double() -
                 0.2) > tolerance) {
      throw std::invalid_argument("FROZEN_COORDINATOR_PARAMETER_MISMATCH");
    }
  }

  void CreateRosEntities() {
    const rclcpp::QoS transient_qos =
        rclcpp::QoS{1}.reliable().transient_local();
    mission_publisher_ = node_.create_publisher<
        lunar_navigation_msgs::msg::ExplorationTask>(
        "/mission/exploration_task", transient_qos);
    reference_publisher_ = node_.create_publisher<
        lunar_planning_msgs::msg::MotionReference>(
        "/lunar/motion_reference", rclcpp::QoS{1}.reliable().durability_volatile());
    status_publisher_ = node_.create_publisher<
        diagnostic_msgs::msg::DiagnosticArray>(
        "/lunar/path_planning/status", rclcpp::QoS{10}.reliable());

    rclcpp::SubscriptionOptions options;
    options.callback_group = transition_group_;
    goal_subscription_ = node_.create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", rclcpp::QoS{1}.reliable().durability_volatile(),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
          ReceiveGoal(*message);
        },
        options);
    global_map_subscription_ = node_.create_subscription<grid_map_msgs::msg::GridMap>(
        "/environment/map_global", transient_qos,
        [this](const grid_map_msgs::msg::GridMap::SharedPtr message) {
          ReceiveGlobalMap(*message);
        },
        options);
    local_map_subscription_ = node_.create_subscription<grid_map_msgs::msg::GridMap>(
        "/environment/map_local", transient_qos,
        [this](const grid_map_msgs::msg::GridMap::SharedPtr message) {
          ReceiveLocalMap(*message);
        },
        options);
    odometry_subscription_ = node_.create_subscription<nav_msgs::msg::Odometry>(
        "/lunar/unreal/wheeled_odometry", rclcpp::SensorDataQoS{},
        [this](const nav_msgs::msg::Odometry::SharedPtr message) {
          ReceiveOdometry(*message);
        },
        options);
    tf_subscription_ = node_.create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::QoS{100}.best_effort(),
        [this](const tf2_msgs::msg::TFMessage::SharedPtr message) {
          ReceiveTf(*message);
        },
        options);
    feedback_subscription_ = node_.create_subscription<
        lunar_navigation_msgs::msg::MotionExecutionFeedback>(
        "/execution/motion_feedback", rclcpp::QoS{10}.reliable(),
        [this](const lunar_navigation_msgs::msg::MotionExecutionFeedback::SharedPtr
                   message) { ReceiveFeedback(*message); },
        options);
    bridge_status_subscription_ = node_.create_subscription<
        diagnostic_msgs::msg::DiagnosticArray>(
        "/lunar/unreal/status", rclcpp::QoS{10}.reliable(),
        [this](const diagnostic_msgs::msg::DiagnosticArray::SharedPtr message) {
          ReceiveBridgeStatus(*message);
        },
        options);

    cancel_service_ = node_.create_service<std_srvs::srv::Trigger>(
        "/lunar/path_planning/cancel",
        [this](
            const std_srvs::srv::Trigger::Request::SharedPtr,
            std_srvs::srv::Trigger::Response::SharedPtr response) {
          response->success = Cancel();
          response->message = response->success
              ? "CANCEL_REQUESTED_WAITING_FOR_HOLD_OR_CANCELED"
              : "NO_ACTIVE_GOAL";
        },
        rmw_qos_profile_services_default, transition_group_);
    hold_client_ = node_.create_client<std_srvs::srv::Trigger>(
        "/lunar/unreal/hold", rmw_qos_profile_services_default,
        transition_group_);
    dispatch_timer_ = node_.create_wall_timer(
        std::chrono::milliseconds{100}, [this] { Dispatch(); },
        transition_group_);
  }

  void CleanupEntities() {
    dispatch_timer_.reset();
    hold_client_.reset();
    cancel_service_.reset();
    bridge_status_subscription_.reset();
    feedback_subscription_.reset();
    tf_subscription_.reset();
    odometry_subscription_.reset();
    local_map_subscription_.reset();
    global_map_subscription_.reset();
    goal_subscription_.reset();
    status_publisher_.reset();
    reference_publisher_.reset();
    mission_publisher_.reset();
    action_group_.reset();
    transition_group_.reset();
  }

  void TryUpdateSnapshot() {
    if (!global_map_ || !local_map_stamp_ns_ || !odometry_ ||
        !transform_stamp_ns_) {
      PublishStatus();
      return;
    }
    const std::int64_t map_stamp_ns =
        global_map_->stamp.nanoseconds_since_epoch;
    constexpr std::int64_t maximum_skew_ns = 200'000'000LL;
    if (*local_map_stamp_ns_ != map_stamp_ns ||
        std::abs(odometry_->stamp_ns - map_stamp_ns) > maximum_skew_ns ||
        std::abs(*transform_stamp_ns_ - map_stamp_ns) > maximum_skew_ns) {
      PublishStatus();
      return;
    }
    static_cast<void>(machine_.UpdateSnapshot(PlanningSnapshot{
        .global_map = global_map_,
        .local_map_stamp_ns = *local_map_stamp_ns_,
        .transform_stamp_ns = *transform_stamp_ns_,
        .robot = *odometry_,
    }));
    RequestHoldIfNeeded();
    PublishStatus();
    MaybeDispatch();
  }

  void MaybeDispatch() {
    if (!active_) {
      return;
    }
    RequestHoldIfNeeded();
    if (pending_command_ && pending_plan_goal_) {
      if (!dependencies_.planner->Available()) {
        return;
      }
      const PlanningCommand command = std::move(*pending_command_);
      Action::Goal goal = std::move(*pending_plan_goal_);
      pending_command_.reset();
      pending_plan_goal_.reset();
      SendPlanningGoal(command, goal);
      return;
    }
    if (machine_.state() != CoordinatorState::kReady ||
        !dependencies_.planner->Available()) {
      return;
    }
    const auto command = machine_.TakePlanningCommand();
    if (!command) {
      return;
    }
    if (command->publish_mission) {
      lunar_navigation_msgs::msg::ExplorationTask mission;
      mission.header.frame_id = "map";
      mission.header.stamp = Stamp(command->stamp_ns);
      mission.mission_id = command->mission_id;
      mission.revision = command->mission_revision;
      mission.desired_state = mission.ACTIVE;
      mission.roi_min_x_m = command->roi_min_x_m;
      mission.roi_min_y_m = command->roi_min_y_m;
      mission.roi_max_x_m = command->roi_max_x_m;
      mission.roi_max_y_m = command->roi_max_y_m;
      mission.science_regions.clear();
      mission_publisher_->publish(mission);
      last_mission_ = mission;
    }

    Action::Goal goal;
    goal.request_id = command->request_id;
    goal.mission_id = command->mission_id;
    goal.mission_revision = command->mission_revision;
    goal.replace_active_request = false;
    goal.goal.header.frame_id = "map";
    goal.goal.header.stamp = Stamp(command->stamp_ns);
    goal.goal.goal_id = command->mission_id;
    goal.goal.goal_type = goal.goal.POINT;
    goal.goal.point.x = command->goal_x_m;
    goal.goal.point.y = command->goal_y_m;
    goal.goal.point.z = command->goal_z_m;
    goal.goal.position_tolerance_m = command->position_tolerance_m;
    goal.goal.has_yaw_constraint = true;
    goal.goal.yaw_rad = command->goal_yaw_rad;
    goal.goal.yaw_tolerance_rad = command->yaw_tolerance_rad;
    last_plan_goal_ = goal;
    if (command->publish_mission) {
      pending_command_ = *command;
      pending_plan_goal_ = goal;
      PublishStatus();
      return;
    }
    SendPlanningGoal(*command, goal);
  }

  void SendPlanningGoal(
      const PlanningCommand& command, const Action::Goal& goal) {
    dependencies_.planner->Send(
        goal,
        [this, command](PlanCompletion completion) {
          std::scoped_lock callback_lock{mutex_};
          PlannerReply reply{
              .request_id = command.request_id,
              .mission_id = command.mission_id,
              .mission_revision = completion.result.mission_revision,
              .planning_outcome = completion.result.planning_outcome,
              .execution_directive = completion.result.execution_directive,
              .reason_code = completion.action_succeeded
                  ? completion.result.reason_code
                  : completion.reason_code,
              .has_reference = completion.action_succeeded &&
                  completion.result.has_reference,
              .reference_platform_type =
                  completion.result.reference.platform_type,
              .plan_id = completion.result.reference.plan_id,
          };
          if (!completion.action_succeeded) {
            reply.mission_revision = command.mission_revision;
            reply.planning_outcome = 4U;
          }
          if (machine_.HandlePlannerReply(reply)) {
            reference_publisher_->publish(completion.result.reference);
            last_reference_ = completion.result.reference;
          }
          RequestHoldIfNeeded();
          PublishStatus();
        });
    PublishStatus();
  }

  void RequestHoldIfNeeded() {
    if (!machine_.ConsumeHoldRequest()) {
      return;
    }
    if (dependencies_.request_hold) {
      dependencies_.request_hold();
      return;
    }
    if (hold_client_ && hold_client_->service_is_ready()) {
      static_cast<void>(hold_client_->async_send_request(
          std::make_shared<std_srvs::srv::Trigger::Request>()));
    }
  }

  void PublishStatus() {
    if (!active_ || !status_publisher_ || !status_publisher_->is_activated()) {
      return;
    }
    diagnostic_msgs::msg::DiagnosticArray message;
    message.header.stamp = node_.now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "lunar_goal_coordinator";
    status.hardware_id = "unreal-wheeled-path-planning";
    status.level = machine_.state() == CoordinatorState::kHold
        ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
        : (machine_.state() == CoordinatorState::kWaitingForMap
               ? diagnostic_msgs::msg::DiagnosticStatus::WARN
               : diagnostic_msgs::msg::DiagnosticStatus::OK);
    status.message = machine_.reason_code();
    status.values = {
        KeyValue("state", StateName(machine_.state())),
        KeyValue("reason_code", machine_.reason_code()),
        KeyValue("session_id", machine_.session_id()),
        KeyValue("mission_id", machine_.mission_id()),
    };
    message.status.push_back(std::move(status));
    status_publisher_->publish(message);
  }

  GoalCoordinatorNode& node_;
  GoalCoordinatorDependencies dependencies_;
  mutable std::recursive_mutex mutex_;
  GoalStateMachine machine_;
  lunar::planning::ros::GridMapAdapter map_adapter_;
  std::shared_ptr<const lunar::planning::GridMap> global_map_;
  std::optional<std::int64_t> local_map_stamp_ns_;
  std::optional<RobotState> odometry_;
  std::optional<std::int64_t> transform_stamp_ns_;
  bool configured_{};
  bool active_{};

  rclcpp::CallbackGroup::SharedPtr transition_group_;
  rclcpp::CallbackGroup::SharedPtr action_group_;
  rclcpp_lifecycle::LifecyclePublisher<
      lunar_navigation_msgs::msg::ExplorationTask>::SharedPtr
      mission_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<
      lunar_planning_msgs::msg::MotionReference>::SharedPtr
      reference_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<
      diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_publisher_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      goal_subscription_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr
      global_map_subscription_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr
      local_map_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscription_;
  rclcpp::Subscription<
      lunar_navigation_msgs::msg::MotionExecutionFeedback>::SharedPtr
      feedback_subscription_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      bridge_status_subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cancel_service_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr hold_client_;
  rclcpp::TimerBase::SharedPtr dispatch_timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      parameter_callback_;

  std::optional<lunar_navigation_msgs::msg::ExplorationTask> last_mission_;
  std::optional<lunar_planning_msgs::msg::MotionReference> last_reference_;
  std::optional<Action::Goal> last_plan_goal_;
  std::optional<PlanningCommand> pending_command_;
  std::optional<Action::Goal> pending_plan_goal_;
};

GoalCoordinatorNode::GoalCoordinatorNode(
    const rclcpp::NodeOptions& options,
    GoalCoordinatorDependencies dependencies)
    : rclcpp_lifecycle::LifecycleNode("lunar_goal_coordinator", options),
      impl_(std::make_unique<Impl>(*this, std::move(dependencies))) {}

GoalCoordinatorNode::~GoalCoordinatorNode() = default;

GoalCoordinatorNode::CallbackReturn GoalCoordinatorNode::on_configure(
    const rclcpp_lifecycle::State&) {
  return impl_->Configure();
}

GoalCoordinatorNode::CallbackReturn GoalCoordinatorNode::on_activate(
    const rclcpp_lifecycle::State&) {
  return impl_->Activate();
}

GoalCoordinatorNode::CallbackReturn GoalCoordinatorNode::on_deactivate(
    const rclcpp_lifecycle::State&) {
  return impl_->Deactivate();
}

GoalCoordinatorNode::CallbackReturn GoalCoordinatorNode::on_cleanup(
    const rclcpp_lifecycle::State&) {
  return impl_->Cleanup();
}

GoalCoordinatorNode::CallbackReturn GoalCoordinatorNode::on_shutdown(
    const rclcpp_lifecycle::State&) {
  return impl_->Cleanup();
}

GoalCoordinatorNode::CallbackReturn GoalCoordinatorNode::on_error(
    const rclcpp_lifecycle::State&) {
  return impl_->Deactivate();
}

void GoalCoordinatorNode::ReceiveGoalForTesting(
    const geometry_msgs::msg::PoseStamped& message) {
  impl_->ReceiveGoal(message);
}

void GoalCoordinatorNode::ReceiveGlobalMapForTesting(
    const grid_map_msgs::msg::GridMap& message) {
  impl_->ReceiveGlobalMap(message);
}

void GoalCoordinatorNode::ReceiveLocalMapForTesting(
    const grid_map_msgs::msg::GridMap& message) {
  impl_->ReceiveLocalMap(message);
}

void GoalCoordinatorNode::ReceiveOdometryForTesting(
    const nav_msgs::msg::Odometry& message) {
  impl_->ReceiveOdometry(message);
}

void GoalCoordinatorNode::ReceiveTfForTesting(
    const tf2_msgs::msg::TFMessage& message) {
  impl_->ReceiveTf(message);
}

void GoalCoordinatorNode::ReceiveExecutionFeedbackForTesting(
    const lunar_navigation_msgs::msg::MotionExecutionFeedback& message) {
  impl_->ReceiveFeedback(message);
}

void GoalCoordinatorNode::ReceiveBridgeStatusForTesting(
    const diagnostic_msgs::msg::DiagnosticArray& message) {
  impl_->ReceiveBridgeStatus(message);
}

bool GoalCoordinatorNode::CancelForTesting() { return impl_->Cancel(); }

void GoalCoordinatorNode::DispatchForTesting() { impl_->Dispatch(); }

CoordinatorState GoalCoordinatorNode::state_for_testing() const noexcept {
  return impl_->state();
}

std::string GoalCoordinatorNode::reason_for_testing() const {
  return impl_->reason();
}

std::optional<lunar_navigation_msgs::msg::ExplorationTask>
GoalCoordinatorNode::last_mission_for_testing() const {
  return impl_->last_mission();
}

std::optional<lunar_planning_msgs::msg::MotionReference>
GoalCoordinatorNode::last_reference_for_testing() const {
  return impl_->last_reference();
}

std::optional<Action::Goal>
GoalCoordinatorNode::last_plan_goal_for_testing() const {
  return impl_->last_plan_goal();
}

}  // namespace lunar::goal_coordinator
