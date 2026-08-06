#include "lunar_planner_ros/plan_motion_server.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <lunar_navigation_msgs/msg/exploration_task.hpp>
#include <lunar_navigation_msgs/msg/hopper_propellant_state.hpp>
#include <lunar_navigation_msgs/msg/localization_status.hpp>
#include <lunar_navigation_msgs/msg/motion_execution_feedback.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/create_subscription.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp_action/create_server.hpp>
#include <rclcpp_action/server.hpp>
#include <rclcpp_action/server_goal_handle.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_planner_core/planner.hpp"
#include "lunar_planner_ros/execution_feedback_tracker.hpp"
#include "lunar_planner_ros/message_conversion.hpp"
#include "lunar_planner_ros/reference_guard.hpp"
#include "lunar_planner_ros/route_marker_publisher.hpp"
#include "lunar_planner_ros/snapshot_builder.hpp"
#include "lunar_planner_ros/snapshot_store.hpp"

namespace lunar::planning::ros {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using namespace std::chrono_literals;

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
constexpr std::array<std::string_view, 15U> kRetiredSearchParameters{
    "global_search.maximum_expanded_states",
    "global_search.maximum_reopened_states",
    "global_search.maximum_generated_candidates",
    "global_search.maximum_open_states",
    "global_search.maximum_memory_bytes",
    "global_search.resources.maximum_expanded_states",
    "global_search.resources.maximum_reopened_states",
    "global_search.resources.maximum_generated_candidates",
    "global_search.resources.maximum_open_states",
    "global_search.resources.maximum_memory_bytes",
    "hopper.maximum_landing_regions",
    "hopper.maximum_graph_nodes",
    "hopper.maximum_graph_out_degree",
    "hopper.maximum_nominal_aim_points_per_region",
    "hopper.maximum_certification_attempts",
};

[[nodiscard]] std::optional<lunar::planning::TimePoint> TimePointFromStamp(
    const builtin_interfaces::msg::Time& stamp) noexcept {
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond) {
    return std::nullopt;
  }
  return lunar::planning::TimePoint{
      static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
          static_cast<std::int64_t>(stamp.nanosec),
  };
}

[[nodiscard]] std::optional<rclcpp::Time> RosTimeFromStamp(
    const builtin_interfaces::msg::Time& stamp) noexcept {
  const auto time = TimePointFromStamp(stamp);
  if (!time || time->nanoseconds_since_epoch <= 0) {
    return std::nullopt;
  }
  try {
    return rclcpp::Time{stamp, RCL_ROS_TIME};
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

[[nodiscard]] bool ValidMissionMessage(
    const lunar_navigation_msgs::msg::ExplorationTask& mission) noexcept {
  return !mission.mission_id.empty() && mission.revision > 0U &&
      mission.header.frame_id == "map" &&
      RosTimeFromStamp(mission.header.stamp).has_value() &&
      (mission.desired_state == mission.ACTIVE ||
       mission.desired_state == mission.PAUSED ||
       mission.desired_state == mission.CANCELED);
}

[[nodiscard]] std::optional<std::chrono::nanoseconds> PositiveDuration(
    const double seconds) noexcept {
  if (!std::isfinite(seconds) || seconds <= 0.0 ||
      seconds > static_cast<double>(
          std::numeric_limits<std::int64_t>::max()) /
          static_cast<double>(kNanosecondsPerSecond)) {
    return std::nullopt;
  }
  const double nanoseconds =
      seconds * static_cast<double>(kNanosecondsPerSecond);
  if (!std::isfinite(nanoseconds) || nanoseconds < 1.0) {
    return std::nullopt;
  }
  return std::chrono::nanoseconds{
      static_cast<std::int64_t>(std::llround(nanoseconds))};
}

[[nodiscard]] bool SnapshotIsStale(const SnapshotErrorCode code) noexcept {
  switch (code) {
    case SnapshotErrorCode::kMissingGlobalMap:
    case SnapshotErrorCode::kMissingLocalMap:
    case SnapshotErrorCode::kMissingOdometry:
    case SnapshotErrorCode::kMissingLocalizationStatus:
    case SnapshotErrorCode::kStaleGlobalMap:
    case SnapshotErrorCode::kStaleLocalMap:
    case SnapshotErrorCode::kStaleOdometry:
    case SnapshotErrorCode::kStaleLocalizationStatus:
    case SnapshotErrorCode::kStaleHopperPropellant:
    case SnapshotErrorCode::kInputSkew:
    case SnapshotErrorCode::kInvalidLocalization:
    case SnapshotErrorCode::kCovarianceLimit:
    case SnapshotErrorCode::kStaleTf:
      return true;
    case SnapshotErrorCode::kConfigurationInvalid:
    case SnapshotErrorCode::kInvalidGlobalMap:
    case SnapshotErrorCode::kInvalidLocalMap:
    case SnapshotErrorCode::kInvalidOdometry:
    case SnapshotErrorCode::kInvalidHopperPropellant:
    case SnapshotErrorCode::kInvalidGoal:
      return false;
  }
  return false;
}

[[nodiscard]] std::shared_ptr<Action::Result> CanceledResult(
    const std::uint64_t mission_revision,
    std::string reason_code) {
  auto result = std::make_shared<Action::Result>();
  result->planning_outcome = Action::Result::CANCELED;
  result->execution_directive = Action::Result::HOLD_POSITION;
  result->reason_code = std::move(reason_code);
  result->mission_revision = mission_revision;
  result->has_reference = false;
  result->reference = lunar_planning_msgs::msg::MotionReference{};
  result->diagnostics.planner_name = "cpp_v3";
  return result;
}

[[nodiscard]] std::shared_ptr<Action::Result> InvariantFailureResult(
    const std::uint64_t mission_revision,
    std::string reason_code) {
  auto result = std::make_shared<Action::Result>();
  result->planning_outcome = Action::Result::NUMERICAL_FAILURE;
  result->execution_directive = Action::Result::NO_SAFE_REFERENCE;
  result->reason_code = std::move(reason_code);
  result->mission_revision = mission_revision;
  result->has_reference = false;
  result->reference = lunar_planning_msgs::msg::MotionReference{};
  result->diagnostics.planner_name = "cpp_v3";
  return result;
}

[[nodiscard]] std::string DiagnosticDouble(const double value) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<double>::max_digits10)
         << value;
  return stream.str();
}

[[nodiscard]] std::string JoinWarningCodes(
    const std::vector<std::string>& warning_codes) {
  std::string joined;
  for (std::size_t index = 0U; index < warning_codes.size(); ++index) {
    if (index > 0U) {
      joined.push_back(',');
    }
    joined.append(warning_codes[index]);
  }
  return joined;
}

struct ExecutionDiagnosticInfo final {
  std::string plan_id;
  std::string segment_id;
  std::string state{"IDLE"};
};

[[nodiscard]] std::string GroundExecutionStateName(
    const lunar::planning::GroundExecutionState state) {
  switch (state) {
    case lunar::planning::GroundExecutionState::kIdle:
      return "IDLE";
    case lunar::planning::GroundExecutionState::kExecuting:
      return "EXECUTING";
    case lunar::planning::GroundExecutionState::kHolding:
      return "HOLDING";
    case lunar::planning::GroundExecutionState::kFault:
      return "FAULT";
  }
  return "UNKNOWN";
}

[[nodiscard]] std::string HopperExecutionStateName(
    const lunar::planning::HopperExecutionState state) {
  switch (state) {
    case lunar::planning::HopperExecutionState::kGroundHold:
      return "GROUND_HOLD";
    case lunar::planning::HopperExecutionState::kJumpReady:
      return "JUMP_READY";
    case lunar::planning::HopperExecutionState::kJumpCommitted:
      return "JUMP_COMMITTED";
    case lunar::planning::HopperExecutionState::kInFlight:
      return "IN_FLIGHT";
    case lunar::planning::HopperExecutionState::kLandedHold:
      return "LANDED_HOLD";
    case lunar::planning::HopperExecutionState::kEmergencyDelegated:
      return "EMERGENCY_DELEGATED";
  }
  return "UNKNOWN";
}

[[nodiscard]] ExecutionDiagnosticInfo ExecutionDiagnostic(
    const std::optional<lunar::planning::ExecutionContext>& context,
    const lunar::planning::MotionReference* reference) {
  if (reference != nullptr) {
    ExecutionDiagnosticInfo info{
        .plan_id = reference->plan_id,
        .segment_id = reference->plan_id,
        .state = "AWAITING_FEEDBACK",
    };
    if (const auto* hops =
            std::get_if<lunar::planning::HopReference>(&reference->data);
        hops != nullptr && !hops->segments.empty()) {
      info.segment_id = hops->segments.front().segment_id;
    }
    return info;
  }
  if (!context.has_value()) {
    return {};
  }
  return std::visit(
      [](const auto& execution) {
        using Context = std::decay_t<decltype(execution)>;
        std::string state;
        if constexpr (std::is_same_v<
                          Context,
                          lunar::planning::GroundExecutionContext>) {
          state = GroundExecutionStateName(execution.state);
        } else {
          state = HopperExecutionStateName(execution.state);
        }
        return ExecutionDiagnosticInfo{
            .plan_id = execution.active_plan_id.value_or(""),
            .segment_id = execution.active_segment_id.value_or(""),
            .state = std::move(state),
        };
      },
      *context);
}

[[nodiscard]] std::string GoalIdentity(
    const lunar::planning::GoalRegion& goal) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<double>::max_digits10)
         << goal.goal_id << '|';
  std::visit(
      [&stream](const auto& target) {
        using Target = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Target, lunar::planning::PointGoal>) {
          stream << "point|" << target.position_m.x << '|'
                 << target.position_m.y << '|' << target.position_m.z << '|'
                 << target.tolerance_m;
        } else {
          stream << "region|" << target.normal_tolerance_m << '|'
                 << target.boundary_m.size();
          for (const auto& point : target.boundary_m) {
            stream << '|' << point.x << '|' << point.y << '|' << point.z;
          }
        }
      },
      goal.target);
  stream << "|yaw|";
  if (goal.yaw_rad.has_value()) {
    stream << *goal.yaw_rad;
  } else {
    stream << "none";
  }
  stream << '|' << goal.yaw_tolerance_rad;
  return stream.str();
}

[[nodiscard]] bool FeedbackStateAllowsRolling(
    const lunar::planning::PlatformType platform,
    const std::optional<lunar::planning::ExecutionContext>& context) noexcept {
  if (!context.has_value()) {
    return false;
  }
  if (platform == lunar::planning::PlatformType::kHopper) {
    return std::holds_alternative<lunar::planning::HopperExecutionContext>(
        *context);
  }
  const auto* ground =
      std::get_if<lunar::planning::GroundExecutionContext>(&*context);
  return ground != nullptr &&
      (ground->state == lunar::planning::GroundExecutionState::kExecuting ||
       ground->state == lunar::planning::GroundExecutionState::kHolding);
}

}  // namespace

struct PlanMotionServer::Impl final {
  struct PendingGoal final {
    std::string request_id;
    std::string mission_id;
    std::string frame_id;
    rclcpp::Time stamp;
    lunar::planning::GoalRegion goal;
    std::uint64_t mission_revision{};
    bool replace_active_request{};
  };

  struct CachedRoute final {
    std::shared_ptr<const lunar::planning::RouteContinuation> continuation;
    std::string route_id;
    std::string mission_id;
    std::uint64_t mission_revision{};
    std::string platform_id;
    std::string capability_version;
    std::uint64_t global_map_generation{};
    std::string goal_identity;
  };

  PlanMotionServer& node;
  PlanMotionServerDependencies dependencies;

  rclcpp::CallbackGroup::SharedPtr map_group;
  rclcpp::CallbackGroup::SharedPtr localization_group;
  rclcpp::CallbackGroup::SharedPtr propellant_group;
  rclcpp::CallbackGroup::SharedPtr tf_group;
  rclcpp::CallbackGroup::SharedPtr mission_group;
  rclcpp::CallbackGroup::SharedPtr action_group;

  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr global_map_sub;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr local_map_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub;
  rclcpp::Subscription<
      lunar_navigation_msgs::msg::LocalizationStatus>::SharedPtr
      localization_status_sub;
  rclcpp::Subscription<
      lunar_navigation_msgs::msg::HopperPropellantState>::SharedPtr
      hopper_propellant_state_sub;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub;
  rclcpp::Subscription<
      lunar_navigation_msgs::msg::ExplorationTask>::SharedPtr mission_sub;
  rclcpp::Subscription<
      lunar_navigation_msgs::msg::MotionExecutionFeedback>::SharedPtr
      execution_feedback_sub;
  rclcpp_action::Server<Action>::SharedPtr action_server;
  rclcpp_lifecycle::LifecyclePublisher<
      diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher;
  rclcpp_lifecycle::LifecyclePublisher<
      visualization_msgs::msg::MarkerArray>::SharedPtr route_marker_publisher;

  mutable std::mutex state_mutex;
  mutable std::mutex diagnostic_mutex;
  bool configured{};
  bool active{};
  bool faulted{};
  std::shared_ptr<SnapshotStore> snapshot_store;
  std::unique_ptr<SnapshotBuilder> snapshot_builder;
  std::optional<LoadedCapabilities> capabilities;
  std::unique_ptr<ReferenceGuard> reference_guard;
  std::optional<lunar_navigation_msgs::msg::ExplorationTask> mission;
  std::optional<PendingGoal> pending_goal;
  std::optional<CachedRoute> cached_route;
  std::chrono::nanoseconds execution_feedback_max_age{1s};
  ExecutionFeedbackTracker execution_feedback_tracker;
  RouteMarkerPublisher route_markers;
  std::mutex route_marker_mutex;
  std::shared_ptr<GoalHandle> active_goal;
  std::unique_ptr<std::jthread> worker;
  bool worker_running{};
  std::uint64_t worker_generation{};
  std::string worker_stop_reason{"REQUEST_CANCELED"};
  std::string last_diagnostic_reason;
  std::string diagnostic_hardware_id{"unconfigured"};

  explicit Impl(
      PlanMotionServer& owner,
      PlanMotionServerDependencies injected_dependencies)
      : node(owner), dependencies(std::move(injected_dependencies)) {
    if (!dependencies.planner) {
      auto planner = std::make_shared<lunar::planning::Planner>();
      dependencies.planner =
          [planner = std::move(planner)](
              const lunar::planning::PlannerInput& input) {
            return planner->Plan(input);
          };
    }

    map_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    localization_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    propellant_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    tf_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    mission_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    action_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    diagnostics_publisher =
        node.create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            "/diagnostics", rclcpp::QoS{10}.reliable());
    route_marker_publisher =
        node.create_publisher<visualization_msgs::msg::MarkerArray>(
            "/planning/certified_route_markers",
            rclcpp::QoS{1}.reliable().transient_local());
    action_server = rclcpp_action::create_server<Action>(
        node.get_node_base_interface(),
        node.get_node_clock_interface(),
        node.get_node_logging_interface(),
        node.get_node_waitables_interface(),
        "/plan_motion",
        [this](
            const rclcpp_action::GoalUUID& uuid,
            const std::shared_ptr<const Action::Goal> goal) {
          return HandleGoal(uuid, goal);
        },
        [this](const std::shared_ptr<GoalHandle> goal_handle) {
          return HandleCancel(goal_handle);
        },
        [this](const std::shared_ptr<GoalHandle> goal_handle) {
          HandleAccepted(goal_handle);
        },
        rcl_action_server_get_default_options(), action_group);

    DeclareParameters();
  }

  ~Impl() {
    StopAndJoinWorker("NODE_SHUTDOWN");
  }

  void DeclareParameters() {
    for (const char* name : {
             "global_map_max_age", "local_map_max_age",
             "odometry_max_age", "localization_status_max_age",
             "propellant_state_max_age", "tf_max_age",
             "max_pairwise_skew"}) {
      node.declare_parameter(name, rclcpp::ParameterType::PARAMETER_DOUBLE);
    }
    node.declare_parameter<double>("degraded_pose_covariance_limit", 0.5);
    node.declare_parameter<double>("degraded_twist_covariance_limit", 0.5);
    node.declare_parameter<double>("execution_feedback_max_age", 1.0);
    node.declare_parameter<std::int64_t>("maximum_transform_samples", 256);
    node.declare_parameter<double>("base_resolution_m", 0.2);
    node.declare_parameter<std::int64_t>("maximum_global_level", 4);
    node.declare_parameter<std::int64_t>("maximum_global_cells", 1'048'576);
    node.declare_parameter<std::int64_t>("maximum_global_axis_cells", 4'096);
    node.declare_parameter(
        "capability_package", rclcpp::ParameterType::PARAMETER_STRING);
    node.declare_parameter(
        "platform_capability_file", rclcpp::ParameterType::PARAMETER_STRING);
    node.declare_parameter(
        "observation_capability_file", rclcpp::ParameterType::PARAMETER_STRING);
  }

  [[nodiscard]] std::chrono::nanoseconds RequiredDuration(
      const std::string& name) const {
    const auto value = PositiveDuration(node.get_parameter(name).as_double());
    if (!value) {
      throw std::invalid_argument{name + " must be a positive finite duration"};
    }
    return *value;
  }

  [[nodiscard]] SnapshotPolicy ReadPolicy() const {
    const double pose_limit =
        node.get_parameter("degraded_pose_covariance_limit").as_double();
    const double twist_limit =
        node.get_parameter("degraded_twist_covariance_limit").as_double();
    SnapshotPolicy policy{
        .global_map_max_age = RequiredDuration("global_map_max_age"),
        .local_map_max_age = RequiredDuration("local_map_max_age"),
        .odometry_max_age = RequiredDuration("odometry_max_age"),
        .localization_status_max_age =
            RequiredDuration("localization_status_max_age"),
        .propellant_state_max_age =
            RequiredDuration("propellant_state_max_age"),
        .tf_max_age = RequiredDuration("tf_max_age"),
        .max_pairwise_skew = RequiredDuration("max_pairwise_skew"),
        .degraded_pose_covariance_limit = pose_limit,
        .degraded_twist_covariance_limit = twist_limit,
    };
    if (!ValidateSnapshotPolicy(policy)) {
      throw std::invalid_argument{"snapshot policy is invalid"};
    }
    return policy;
  }

  [[nodiscard]] std::size_t RequiredSizeParameter(
      const std::string& name) const {
    const std::int64_t value = node.get_parameter(name).as_int();
    if (value <= 0 ||
        static_cast<std::uint64_t>(value) >
            std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument{name + " must be a positive size"};
    }
    return static_cast<std::size_t>(value);
  }

  [[nodiscard]] lunar::planning::PlannerConfig ReadPlannerConfig() const {
    lunar::planning::PlannerConfig config;
    const double base_resolution_m =
        node.get_parameter("base_resolution_m").as_double();
    if (!std::isfinite(base_resolution_m) || base_resolution_m <= 0.0) {
      throw std::invalid_argument{
          "base_resolution_m must be positive and finite"};
    }
    const std::size_t maximum_level =
        RequiredSizeParameter("maximum_global_level");
    if (maximum_level != 4U) {
      throw std::invalid_argument{
          "maximum_global_level must equal the supported level 4"};
    }
    config.global_map.base_resolution_m = base_resolution_m;
    config.global_map.maximum_level = maximum_level;
    config.global_map.maximum_cells =
        RequiredSizeParameter("maximum_global_cells");
    config.global_map.maximum_axis_cells =
        RequiredSizeParameter("maximum_global_axis_cells");
    return config;
  }

  void RejectRetiredSearchParameters() const {
    const auto& overrides = node.get_node_parameters_interface()
                                ->get_parameter_overrides();
    for (const std::string_view name : kRetiredSearchParameters) {
      if (overrides.contains(std::string{name})) {
        throw std::invalid_argument{
            "retired planner search parameter: " + std::string{name}};
      }
    }
  }

  [[nodiscard]] LoadedCapabilities LoadCapabilities() const {
    if (dependencies.preloaded_capabilities) {
      return *dependencies.preloaded_capabilities;
    }
    const std::string package =
        node.get_parameter("capability_package").as_string();
    const std::string platform_file =
        node.get_parameter("platform_capability_file").as_string();
    const std::string observation_file =
        node.get_parameter("observation_capability_file").as_string();
    const CapabilityLoadResult loaded = CapabilityLoader{}.LoadFromPackageShare(
        package, platform_file, observation_file);
    if (!loaded.ok()) {
      throw std::runtime_error{
          loaded.error ? loaded.error->reason_code :
                         std::string{"CAPABILITY_LOAD_FAILED"}};
    }
    return *loaded.capabilities;
  }

  [[nodiscard]] ReferenceGuardLimits GuardLimits(
      const lunar::planning::PlatformCapability& capability) const {
    if (const auto* hopper =
            std::get_if<lunar::planning::HopperCapability>(&capability)) {
      return ReferenceGuardLimits{
          .minimum_settle_guard = hopper->minimum_settle_guard,
          .maximum_landing_speed_mps = hopper->maximum_landing_speed_mps,
          .maximum_angular_speed_radps = hopper->maximum_angular_speed_radps,
      };
    }
    return ReferenceGuardLimits{
        .minimum_settle_guard = 1ns,
        .maximum_landing_speed_mps = 1.0,
        .maximum_angular_speed_radps = 1.0,
    };
  }

  CallbackReturn Configure() {
    try {
      StopAndJoinWorker("NODE_RECONFIGURED");
      ResetSubscriptions();
      RejectRetiredSearchParameters();
      const SnapshotPolicy policy = ReadPolicy();
      const auto feedback_max_age =
          RequiredDuration("execution_feedback_max_age");
      lunar::planning::PlannerConfig planner_config = ReadPlannerConfig();
      LoadedCapabilities loaded = LoadCapabilities();
      const auto transform_samples =
          node.get_parameter("maximum_transform_samples").as_int();
      if (transform_samples <= 0) {
        throw std::invalid_argument{
            "maximum_transform_samples must be positive"};
      }
      auto new_store = std::make_shared<SnapshotStore>(
          static_cast<std::size_t>(transform_samples));
      auto new_builder = std::make_unique<SnapshotBuilder>(
          new_store, policy, loaded.platform, std::move(planner_config),
          loaded.base_frame_id);
      auto new_guard = std::make_unique<ReferenceGuard>(
          GuardLimits(loaded.platform));

      {
        std::scoped_lock lock{state_mutex};
        snapshot_store = std::move(new_store);
        snapshot_builder = std::move(new_builder);
        capabilities = std::move(loaded);
        reference_guard = std::move(new_guard);
        mission.reset();
        pending_goal.reset();
        cached_route.reset();
        execution_feedback_max_age = feedback_max_age;
        configured = true;
        active = false;
        faulted = false;
      }
      execution_feedback_tracker.Clear();
      {
        std::scoped_lock marker_lock{route_marker_mutex};
        (void)route_markers.DeleteOwned(node.now());
      }
      {
        std::scoped_lock lock{diagnostic_mutex};
        diagnostic_hardware_id = capabilities->platform_id;
      }
      CreateSubscriptions();
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::OK,
          "PLANNER_CONFIGURED");
      return CallbackReturn::SUCCESS;
    } catch (const std::exception& error) {
      RCLCPP_ERROR(node.get_logger(), "planner configure failed: %s", error.what());
      {
        std::scoped_lock lock{state_mutex};
        configured = false;
        active = false;
        faulted = false;
        snapshot_builder.reset();
        snapshot_store.reset();
        capabilities.reset();
        reference_guard.reset();
        mission.reset();
        pending_goal.reset();
        cached_route.reset();
      }
      execution_feedback_tracker.Clear();
      ResetSubscriptions();
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::ERROR,
          "PLANNER_CONFIGURE_FAILED");
      return CallbackReturn::FAILURE;
    }
  }

  CallbackReturn Activate() {
    {
      std::scoped_lock lock{state_mutex};
      if (!configured || faulted || !snapshot_builder || !reference_guard) {
        return CallbackReturn::FAILURE;
      }
      active = true;
    }
    diagnostics_publisher->on_activate();
    route_marker_publisher->on_activate();
    PublishDiagnostic(
        diagnostic_msgs::msg::DiagnosticStatus::OK,
        "PLANNER_ACTIVE");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn Deactivate(const std::string& reason) {
    {
      std::scoped_lock lock{state_mutex};
      active = false;
      pending_goal.reset();
      cached_route.reset();
    }
    StopAndJoinWorker(reason);
    {
      std::scoped_lock lock{state_mutex};
      if (reference_guard) {
        reference_guard->Reset();
      }
    }
    execution_feedback_tracker.Clear();
    PublishOwnedMarkerDeletes();
    PublishDiagnostic(
        diagnostic_msgs::msg::DiagnosticStatus::WARN,
        reason);
    if (diagnostics_publisher->is_activated()) {
      diagnostics_publisher->on_deactivate();
    }
    if (route_marker_publisher->is_activated()) {
      route_marker_publisher->on_deactivate();
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn Cleanup() {
    Deactivate("PLANNER_CLEANUP");
    ResetSubscriptions();
    std::scoped_lock lock{state_mutex};
    configured = false;
    faulted = false;
    snapshot_builder.reset();
    snapshot_store.reset();
    capabilities.reset();
    reference_guard.reset();
    mission.reset();
    cached_route.reset();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn Error() {
    {
      std::scoped_lock lock{state_mutex};
      active = false;
      configured = false;
      faulted = true;
      pending_goal.reset();
      cached_route.reset();
    }
    RequestWorkerStop("PLANNER_INTERNAL_INVARIANT");
    ResetSubscriptions();
    {
      std::scoped_lock lock{state_mutex};
      snapshot_builder.reset();
      snapshot_store.reset();
      capabilities.reset();
      if (reference_guard) {
        reference_guard->Reset();
      }
      reference_guard.reset();
      mission.reset();
    }
    execution_feedback_tracker.Clear();
    PublishOwnedMarkerDeletes();
    if (diagnostics_publisher->is_activated()) {
      diagnostics_publisher->on_deactivate();
    }
    if (route_marker_publisher->is_activated()) {
      route_marker_publisher->on_deactivate();
    }
    return CallbackReturn::SUCCESS;
  }

  void CreateSubscriptions() {
    rclcpp::SubscriptionOptions map_options;
    map_options.callback_group = map_group;
    const auto map_qos = rclcpp::QoS{1}.reliable().transient_local();
    global_map_sub = node.create_subscription<grid_map_msgs::msg::GridMap>(
        "/environment/map_global", map_qos,
        [this](const grid_map_msgs::msg::GridMap::SharedPtr message) {
          const auto store = Store();
          if (store) {
            store->UpdateGlobalMap(*message);
          }
        },
        map_options);
    local_map_sub = node.create_subscription<grid_map_msgs::msg::GridMap>(
        "/environment/map_local", map_qos,
        [this](const grid_map_msgs::msg::GridMap::SharedPtr message) {
          const auto store = Store();
          if (store) {
            store->UpdateLocalMap(*message);
          }
        },
        map_options);

    rclcpp::SubscriptionOptions localization_options;
    localization_options.callback_group = localization_group;
    odometry_sub = node.create_subscription<nav_msgs::msg::Odometry>(
        "/localization/odometry", rclcpp::SensorDataQoS{},
        [this](const nav_msgs::msg::Odometry::SharedPtr message) {
          OnOdometry(*message);
        },
        localization_options);
    localization_status_sub =
        node.create_subscription<
            lunar_navigation_msgs::msg::LocalizationStatus>(
            "/localization/status", rclcpp::QoS{10}.reliable(),
            [this](
                const lunar_navigation_msgs::msg::LocalizationStatus::SharedPtr
                    message) {
              OnLocalizationStatus(*message);
            },
            localization_options);

    rclcpp::SubscriptionOptions propellant_options;
    propellant_options.callback_group = propellant_group;
    hopper_propellant_state_sub = node.create_subscription<
        lunar_navigation_msgs::msg::HopperPropellantState>(
        "/platform/hopper_propellant_state", rclcpp::QoS{10}.reliable(),
        [this](const lunar_navigation_msgs::msg::HopperPropellantState::
                   SharedPtr message) {
          const auto store = Store();
          if (store) {
            store->UpdateHopperPropellantState(*message);
          }
        },
        propellant_options);

    rclcpp::SubscriptionOptions tf_options;
    tf_options.callback_group = tf_group;
    tf_sub = node.create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::QoS{100}.best_effort(),
        [this](const tf2_msgs::msg::TFMessage::SharedPtr message) {
          const auto store = Store();
          if (store) {
            store->UpdateTransforms(*message);
          }
        },
        tf_options);

    rclcpp::SubscriptionOptions mission_options;
    mission_options.callback_group = mission_group;
    mission_sub =
        node.create_subscription<lunar_navigation_msgs::msg::ExplorationTask>(
            "/mission/exploration_task",
            rclcpp::QoS{1}.reliable().transient_local(),
            [this](
                const lunar_navigation_msgs::msg::ExplorationTask::SharedPtr
                    message) {
              OnMission(*message);
            },
            mission_options);

    execution_feedback_sub = node.create_subscription<
        lunar_navigation_msgs::msg::MotionExecutionFeedback>(
        "/execution/motion_feedback", rclcpp::QoS{10}.reliable(),
        [this](const lunar_navigation_msgs::msg::MotionExecutionFeedback::
                   SharedPtr message) {
          const FeedbackAcceptResult accepted =
              execution_feedback_tracker.Accept(*message, node.now());
          if (!accepted.ok()) {
            PublishDiagnostic(
                diagnostic_msgs::msg::DiagnosticStatus::WARN,
                accepted.reason_code);
          }
        },
        mission_options);
  }

  void ResetSubscriptions() {
    global_map_sub.reset();
    local_map_sub.reset();
    odometry_sub.reset();
    localization_status_sub.reset();
    hopper_propellant_state_sub.reset();
    tf_sub.reset();
    mission_sub.reset();
    execution_feedback_sub.reset();
  }

  [[nodiscard]] std::shared_ptr<SnapshotStore> Store() const {
    std::scoped_lock lock{state_mutex};
    return snapshot_store;
  }

  void OnOdometry(const nav_msgs::msg::Odometry& message) {
    const auto store = Store();
    if (!store) {
      return;
    }
    store->UpdateOdometry(message);
    std::optional<ReferenceGuardDecision> decision;
    {
      std::scoped_lock lock{state_mutex};
      if (active && reference_guard &&
          reference_guard->state() != ReferenceGuardState::kGroundHold) {
        decision = reference_guard->MayReplace(node.now(), message);
      }
    }
    if (decision &&
        decision->state == ReferenceGuardState::kUnresolved) {
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::ERROR,
          decision->reason_code);
    }
  }

  void OnLocalizationStatus(
      const lunar_navigation_msgs::msg::LocalizationStatus& message) {
    const auto store = Store();
    if (!store) {
      return;
    }
    store->UpdateLocalizationStatus(message);
    if (message.status == message.RELOCALIZING) {
      store->ClearTransforms();
      RequestStopIfReplaceable("LOCALIZATION_RELOCALIZING");
    } else if (message.status == message.UNKNOWN ||
               message.status == message.INVALID) {
      RequestStopIfReplaceable("LOCALIZATION_NOT_PLANNABLE");
    }
  }

  void OnMission(const lunar_navigation_msgs::msg::ExplorationTask& message) {
    if (!ValidMissionMessage(message)) {
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::ERROR,
          "MISSION_MESSAGE_INVALID");
      return;
    }
    bool stop = false;
    bool invalidate_route = false;
    {
      std::scoped_lock lock{state_mutex};
      if (mission && mission->mission_id == message.mission_id &&
          message.revision < mission->revision) {
        return;
      }
      mission = message;
      stop = message.desired_state == message.PAUSED ||
          message.desired_state == message.CANCELED;
      if (cached_route &&
          (cached_route->mission_id != message.mission_id ||
           cached_route->mission_revision != message.revision || stop)) {
        cached_route.reset();
        invalidate_route = true;
      }
    }
    if (invalidate_route) {
      execution_feedback_tracker.Clear();
      PublishOwnedMarkerDeletes();
    }
    if (stop) {
      RequestStopIfReplaceable(
          message.desired_state == message.PAUSED
              ? "MISSION_PAUSED" : "MISSION_CANCELED");
    }
  }

  void RequestStopIfReplaceable(const std::string& reason) {
    ReferenceGuardDecision decision;
    bool has_guard = false;
    {
      std::scoped_lock lock{state_mutex};
      if (reference_guard) {
        const auto view = snapshot_store
            ? snapshot_store->Capture() : SnapshotStoreView{};
        decision = reference_guard->MayReplace(node.now(), view.odometry);
        has_guard = true;
      }
    }
    if (!has_guard || decision.may_replace) {
      RequestWorkerStop(reason);
    } else {
      PublishDiagnostic(
          decision.state == ReferenceGuardState::kUnresolved
              ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
              : diagnostic_msgs::msg::DiagnosticStatus::WARN,
          decision.reason_code);
    }
  }

  rclcpp_action::GoalResponse HandleGoal(
      const rclcpp_action::GoalUUID&,
      const std::shared_ptr<const Action::Goal> goal) {
    std::scoped_lock lock{state_mutex};

    // Frozen validation order: lifecycle, request/mission, reference guard,
    // worker ownership, then the explicit replacement flag.
    if (!active || faulted || !configured) {
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::WARN,
          "PLANNER_NOT_ACTIVE");
      return rclcpp_action::GoalResponse::REJECT;
    }

    if (!goal || goal->request_id.empty() || goal->mission_id.empty() ||
        goal->mission_revision == 0U ||
        (goal->goal.header.frame_id != "map" &&
         goal->goal.header.frame_id != "odom")) {
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::WARN,
          "REQUEST_INVALID");
      return rclcpp_action::GoalResponse::REJECT;
    }
    const auto stamp = RosTimeFromStamp(goal->goal.header.stamp);
    const GoalMessageConversion converted = ConvertGoalMessage(goal->goal);
    if (!stamp || !converted.ok() || !mission ||
        mission->mission_id != goal->mission_id ||
        mission->revision != goal->mission_revision ||
        mission->desired_state != mission->ACTIVE) {
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::WARN,
          converted.reason_code.empty()
              ? "MISSION_REVISION_INVALID" : converted.reason_code);
      return rclcpp_action::GoalResponse::REJECT;
    }

    if (!reference_guard || !snapshot_store) {
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::ERROR,
          "PLANNER_INTERNAL_INVARIANT");
      return rclcpp_action::GoalResponse::REJECT;
    }
    const auto decision = reference_guard->MayReplace(
        node.now(), snapshot_store->Capture().odometry);
    if (!decision.may_replace) {
      PublishDiagnostic(
          decision.state == ReferenceGuardState::kUnresolved
              ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
              : diagnostic_msgs::msg::DiagnosticStatus::WARN,
          decision.reason_code);
      return rclcpp_action::GoalResponse::REJECT;
    }

    if (pending_goal) {
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::WARN,
          "PLANNER_GOAL_PENDING");
      return rclcpp_action::GoalResponse::REJECT;
    }
    if (worker_running && !goal->replace_active_request) {
      PublishDiagnostic(
          diagnostic_msgs::msg::DiagnosticStatus::WARN,
          "ACTIVE_REQUEST_EXISTS");
      return rclcpp_action::GoalResponse::REJECT;
    }

    pending_goal = PendingGoal{
        .request_id = goal->request_id,
        .mission_id = goal->mission_id,
        .frame_id = goal->goal.header.frame_id,
        .stamp = *stamp,
        .goal = *converted.goal,
        .mission_revision = goal->mission_revision,
        .replace_active_request = goal->replace_active_request,
    };
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse HandleCancel(
      const std::shared_ptr<GoalHandle> goal_handle) {
    std::scoped_lock lock{state_mutex};
    if (!active || !worker_running || !active_goal ||
        active_goal.get() != goal_handle.get() || !reference_guard ||
        !snapshot_store) {
      return rclcpp_action::CancelResponse::REJECT;
    }
    const auto decision = reference_guard->MayReplace(
        node.now(), snapshot_store->Capture().odometry);
    if (!decision.may_replace) {
      PublishDiagnostic(
          decision.state == ReferenceGuardState::kUnresolved
              ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
              : diagnostic_msgs::msg::DiagnosticStatus::WARN,
          decision.reason_code);
      return rclcpp_action::CancelResponse::REJECT;
    }
    worker_stop_reason = "REQUEST_CANCELED";
    if (worker) {
      worker->request_stop();
    }
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void HandleAccepted(const std::shared_ptr<GoalHandle> goal_handle) {
    std::optional<PendingGoal> accepted;
    {
      std::scoped_lock lock{state_mutex};
      if (pending_goal && goal_handle &&
          pending_goal->request_id == goal_handle->get_goal()->request_id) {
        accepted = std::move(pending_goal);
        pending_goal.reset();
      }
    }
    if (!accepted || !goal_handle) {
      if (goal_handle) {
        goal_handle->abort(InvariantFailureResult(0U, "GOAL_RESERVATION_LOST"));
      }
      return;
    }

    StopAndJoinWorker(
        accepted->replace_active_request
            ? "REQUEST_REPLACED" : "PREVIOUS_WORKER_COMPLETE");

    std::uint64_t generation = 0U;
    {
      std::scoped_lock lock{state_mutex};
      if (!active || faulted || !configured) {
        goal_handle->abort(CanceledResult(
            accepted->mission_revision, "PLANNER_NOT_ACTIVE"));
        return;
      }
      active_goal = goal_handle;
      worker_running = true;
      worker_stop_reason = "REQUEST_CANCELED";
      generation = ++worker_generation;
      worker = std::make_unique<std::jthread>(
          [this, goal_handle, request = std::move(*accepted), generation](
              const std::stop_token token) mutable {
            ExecuteGoal(goal_handle, std::move(request), generation, token);
          });
    }
  }

  void ExecuteGoal(
      const std::shared_ptr<GoalHandle>& goal_handle,
      PendingGoal request,
      const std::uint64_t generation,
      const std::stop_token stop_token) {
    const auto started = std::chrono::steady_clock::now();
    PublishFeedback(goal_handle, Action::Feedback::VALIDATING_INPUT, started);
    if (stop_token.stop_requested()) {
      CompleteCanceled(goal_handle, request.mission_revision, generation);
      return;
    }

    SnapshotBuilder* builder = nullptr;
    lunar::planning::PlatformType configured_platform{};
    std::string platform_id;
    std::string capability_version;
    std::shared_ptr<const lunar::planning::RouteContinuation>
        continuation_candidate;
    {
      std::scoped_lock lock{state_mutex};
      builder = snapshot_builder.get();
      if (capabilities) {
        configured_platform =
            lunar::planning::CapabilityPlatform(capabilities->platform);
        platform_id = capabilities->platform_id;
        capability_version = capabilities->capability_version;
      }
      if (cached_route) {
        continuation_candidate = cached_route->continuation;
      }
    }
    if (!builder) {
      CompleteInvariantFailure(
          goal_handle, request.mission_revision, generation,
          "SNAPSHOT_BUILDER_MISSING");
      return;
    }
    const auto previous_execution =
        execution_feedback_tracker.context(node.now());

    PublishFeedback(goal_handle, Action::Feedback::BUILDING_SNAPSHOT, started);
    SnapshotBuildResult snapshot = builder->Freeze(
        GoalRequest{
            .request_id = request.request_id,
            .mission_id = request.mission_id,
            .mission_revision = request.mission_revision,
            .platform_id = std::move(platform_id),
            .capability_version = std::move(capability_version),
            .frame_id = request.frame_id,
            .stamp = request.stamp,
            .goal = std::move(request.goal),
            .previous_execution = previous_execution,
            .continuation = continuation_candidate,
            .stop_token = stop_token,
        },
        node.now());
    if (stop_token.stop_requested()) {
      CompleteCanceled(goal_handle, request.mission_revision, generation);
      return;
    }
    if (!snapshot.ok()) {
      ClearRollingRoute();
      CompleteSnapshotFailure(
          goal_handle, request.mission_revision, generation,
          snapshot.error, started);
      return;
    }

    bool cache_matches = false;
    std::string route_id;
    {
      std::scoped_lock lock{state_mutex};
      if (cached_route && cached_route->continuation != nullptr) {
        cache_matches =
            cached_route->mission_id == snapshot.input->mission_id &&
            cached_route->mission_revision ==
                snapshot.input->mission_revision &&
            cached_route->platform_id == snapshot.input->platform_id &&
            cached_route->capability_version ==
                snapshot.input->capability_version &&
            cached_route->global_map_generation ==
                snapshot.input->global_map_generation &&
            cached_route->goal_identity ==
                GoalIdentity(snapshot.input->goal_map);
        if (!cache_matches) {
          cached_route.reset();
        } else {
          route_id = cached_route->route_id;
        }
      }
      snapshot.input->continuation =
          cache_matches ? snapshot.input->continuation : nullptr;
    }
    if (!cache_matches && continuation_candidate != nullptr) {
      PublishOwnedMarkerDeletes();
    }
    if (cache_matches && !previous_execution.has_value()) {
      CompleteExecutionFeedbackFailure(
          goal_handle, request.mission_revision, generation,
          "EXECUTION_FEEDBACK_MISSING_OR_STALE", started);
      return;
    }
    if (cache_matches &&
        !FeedbackStateAllowsRolling(configured_platform, previous_execution)) {
      CompleteExecutionFeedbackFailure(
          goal_handle, request.mission_revision, generation,
          "EXECUTION_FEEDBACK_STATE_NOT_READY", started);
      return;
    }

    PublishFeedback(goal_handle, Action::Feedback::SEARCHING, started);
    lunar::planning::PlannerOutput output = dependencies.planner(*snapshot.input);
    if (stop_token.stop_requested() ||
        output.outcome == lunar::planning::PlanningOutcome::kCanceled) {
      CompleteCanceled(goal_handle, request.mission_revision, generation);
      return;
    }
    if (output.reference &&
        output.reference->platform_type != configured_platform) {
      CompleteInvariantFailure(
          goal_handle, request.mission_revision, generation,
          "REFERENCE_PLATFORM_CAPABILITY_MISMATCH");
      return;
    }

    PublishFeedback(goal_handle, Action::Feedback::CERTIFYING, started);
    const ActionResultConversion converted = ConvertPlannerOutput(
        output,
        PlannerResultContext{
            .global_map_stamp = snapshot.input->world.global_map.stamp,
            .local_map_stamp = snapshot.input->world.local_map.stamp,
            .state_stamp = snapshot.input->state_time,
            .mission_revision = request.mission_revision,
            .preview_frame = "map",
            .execution_frame = "odom",
        });
    if (!converted.ok()) {
      CompleteInvariantFailure(
          goal_handle, request.mission_revision, generation,
          converted.reason_code);
      return;
    }

    if (converted.result->has_reference &&
        converted.result->execution_directive ==
            Action::Result::ACTIVATE_NEW_REFERENCE &&
        converted.result->reference.platform_type ==
            lunar_planning_msgs::msg::MotionReference::HOPPER) {
      bool committed = false;
      {
        std::scoped_lock lock{state_mutex};
        committed = reference_guard &&
            reference_guard->Commit(converted.result->reference);
      }
      if (!committed) {
        CompleteInvariantFailure(
            goal_handle, request.mission_revision, generation,
            "HOP_REFERENCE_COMMIT_FAILED");
        return;
      }
    }

    const bool successful_reference =
        converted.result->has_reference && output.continuation != nullptr &&
        (output.outcome ==
             lunar::planning::PlanningOutcome::kNewReferenceAvailable ||
         output.outcome ==
             lunar::planning::PlanningOutcome::kSafeFrontierReferenceAvailable);
    if (successful_reference) {
      if (route_id.empty()) {
        route_id = "route/" + request.request_id;
      }
      {
        std::scoped_lock lock{state_mutex};
        cached_route = CachedRoute{
            .continuation = output.continuation,
            .route_id = route_id,
            .mission_id = snapshot.input->mission_id,
            .mission_revision = snapshot.input->mission_revision,
            .platform_id = snapshot.input->platform_id,
            .capability_version = snapshot.input->capability_version,
            .global_map_generation = snapshot.input->global_map_generation,
            .goal_identity = GoalIdentity(snapshot.input->goal_map),
        };
      }
    } else {
      ClearRollingRoute();
    }
    if (route_id.empty()) {
      route_id = "route/" + request.request_id;
    }

    if (converted.result->has_reference && output.reference.has_value() &&
        converted.result->execution_directive ==
            Action::Result::ACTIVATE_NEW_REFERENCE) {
      SetExpectedExecution(*output.reference);
    }
    if (!output.certified_hops.empty() && output.reference.has_value()) {
      PublishCertifiedRouteMarkers(
          output, *snapshot.input, route_id);
    } else {
      PublishOwnedMarkerDeletes();
    }

    auto result = std::make_shared<Action::Result>(std::move(*converted.result));
    PublishDiagnostic(
        diagnostic_msgs::msg::DiagnosticStatus::OK,
        result->reason_code,
        &output.diagnostics,
        output.reference.has_value() ? &*output.reference : nullptr);
    try {
      goal_handle->succeed(result);
    } catch (const std::exception& error) {
      RCLCPP_ERROR(
          node.get_logger(), "failed to succeed action goal: %s", error.what());
    }
    FinishWorker(generation, goal_handle);
  }

  void SetExpectedExecution(
      const lunar::planning::MotionReference& reference) {
    std::string base_frame_id;
    std::chrono::nanoseconds maximum_age;
    {
      std::scoped_lock lock{state_mutex};
      if (!capabilities) {
        execution_feedback_tracker.Clear();
        return;
      }
      base_frame_id = capabilities->base_frame_id;
      maximum_age = execution_feedback_max_age;
    }
    std::string segment_id = reference.plan_id;
    if (reference.platform_type == lunar::planning::PlatformType::kHopper) {
      const auto* hops = std::get_if<lunar::planning::HopReference>(
          &reference.data);
      if (hops == nullptr || hops->segments.size() != 1U ||
          hops->segments.front().segment_id.empty()) {
        execution_feedback_tracker.Clear();
        return;
      }
      segment_id = hops->segments.front().segment_id;
    }
    execution_feedback_tracker.SetExpected(ExpectedExecution{
        .platform_type = reference.platform_type,
        .base_frame_id = std::move(base_frame_id),
        .plan_id = reference.plan_id,
        .segment_id = std::move(segment_id),
        .maximum_age = maximum_age,
    });
  }

  void PublishCertifiedRouteMarkers(
      const lunar::planning::PlannerOutput& output,
      const lunar::planning::PlannerInput& input,
      const std::string& route_id) {
    if (!route_marker_publisher || !route_marker_publisher->is_activated() ||
        !output.reference.has_value()) {
      return;
    }
    std::string authorized_segment;
    if (const auto* hops = std::get_if<lunar::planning::HopReference>(
            &output.reference->data);
        hops != nullptr && !hops->segments.empty()) {
      authorized_segment = hops->segments.front().segment_id;
    }
    lunar::planning::Vec3 gravity{};
    if (const auto* hopper =
            std::get_if<lunar::planning::HopperCapability>(&input.capability)) {
      gravity = hopper->gravity_mps2;
    }
    std::scoped_lock marker_lock{route_marker_mutex};
    auto markers = route_markers.Replace(
        output.certified_hops,
        RouteMarkerContext{
            .stamp = node.now(),
            .route_id = route_id,
            .reference_plan_id = output.reference->plan_id,
            .authorized_segment_id = std::move(authorized_segment),
            .map_from_odom = input.world.map_from_odom,
            .gravity_mps2 = gravity,
        });
    if (!markers.markers.empty()) {
      route_marker_publisher->publish(markers);
    }
  }

  void PublishOwnedMarkerDeletes() {
    std::scoped_lock marker_lock{route_marker_mutex};
    auto markers = route_markers.DeleteOwned(node.now());
    if (!markers.markers.empty() && route_marker_publisher &&
        route_marker_publisher->is_activated()) {
      route_marker_publisher->publish(markers);
    }
  }

  void ClearRollingRoute() {
    {
      std::scoped_lock lock{state_mutex};
      cached_route.reset();
    }
    PublishOwnedMarkerDeletes();
  }

  void CompleteExecutionFeedbackFailure(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::uint64_t mission_revision,
      const std::uint64_t generation,
      std::string reason_code,
      const std::chrono::steady_clock::time_point started) {
    auto result = std::make_shared<Action::Result>();
    result->planning_outcome = Action::Result::STALE_INPUT;
    result->execution_directive = Action::Result::HOLD_POSITION;
    result->reason_code = std::move(reason_code);
    result->mission_revision = mission_revision;
    result->has_reference = false;
    result->reference = lunar_planning_msgs::msg::MotionReference{};
    PopulateLatestStamps(*result);
    result->diagnostics.planner_name = "execution_feedback_tracker";
    result->diagnostics.elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    PublishDiagnostic(
        diagnostic_msgs::msg::DiagnosticStatus::WARN,
        result->reason_code);
    try {
      goal_handle->succeed(result);
    } catch (const std::exception& error) {
      RCLCPP_ERROR(
          node.get_logger(), "failed to return feedback hold result: %s",
          error.what());
    }
    FinishWorker(generation, goal_handle);
  }

  void CompleteSnapshotFailure(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::uint64_t mission_revision,
      const std::uint64_t generation,
      const std::optional<SnapshotError>& error,
      const std::chrono::steady_clock::time_point started) {
    auto result = std::make_shared<Action::Result>();
    const SnapshotErrorCode code = error
        ? error->code : SnapshotErrorCode::kConfigurationInvalid;
    result->planning_outcome = SnapshotIsStale(code)
        ? Action::Result::STALE_INPUT : Action::Result::INVALID_REQUEST;
    result->execution_directive = Action::Result::HOLD_POSITION;
    result->reason_code = error
        ? error->reason_code : "SNAPSHOT_BUILD_FAILED";
    result->mission_revision = mission_revision;
    result->has_reference = false;
    result->reference = lunar_planning_msgs::msg::MotionReference{};
    PopulateLatestStamps(*result);
    result->diagnostics.planner_name = "snapshot_builder";
    result->diagnostics.elapsed_s =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    PublishDiagnostic(
        diagnostic_msgs::msg::DiagnosticStatus::WARN,
        result->reason_code);
    try {
      goal_handle->succeed(result);
    } catch (const std::exception& exception) {
      RCLCPP_ERROR(
          node.get_logger(), "failed to return snapshot result: %s",
          exception.what());
    }
    FinishWorker(generation, goal_handle);
  }

  void CompleteCanceled(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::uint64_t mission_revision,
      const std::uint64_t generation) {
    std::string reason;
    {
      std::scoped_lock lock{state_mutex};
      reason = worker_stop_reason;
    }
    auto result = CanceledResult(
        mission_revision, reason.empty() ? "REQUEST_CANCELED" : reason);
    PopulateLatestStamps(*result);
    PublishDiagnostic(
        diagnostic_msgs::msg::DiagnosticStatus::WARN,
        result->reason_code);
    try {
      if (result->reason_code == "REQUEST_CANCELED") {
        for (std::size_t attempt = 0U;
             attempt < 20U && !goal_handle->is_canceling(); ++attempt) {
          std::this_thread::sleep_for(1ms);
        }
      }
      if (goal_handle->is_canceling()) {
        goal_handle->canceled(result);
      } else {
        goal_handle->abort(result);
      }
    } catch (const std::exception& error) {
      RCLCPP_ERROR(
          node.get_logger(), "failed to cancel action goal: %s", error.what());
    }
    FinishWorker(generation, goal_handle);
  }

  void CompleteInvariantFailure(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::uint64_t mission_revision,
      const std::uint64_t generation,
      const std::string& detail) {
    const std::string reason = detail.empty()
        ? "PLANNER_INTERNAL_INVARIANT" : detail;
    PublishDiagnostic(
        diagnostic_msgs::msg::DiagnosticStatus::ERROR,
        reason);
    try {
      auto result = InvariantFailureResult(mission_revision, reason);
      PopulateLatestStamps(*result);
      goal_handle->abort(result);
    } catch (const std::exception& error) {
      RCLCPP_ERROR(
          node.get_logger(), "failed to abort invariant goal: %s", error.what());
    }
    FinishWorker(generation, goal_handle);
    try {
      {
        std::scoped_lock lock{state_mutex};
        faulted = true;
      }
      const auto& inactive_state = node.deactivate();
      if (inactive_state.id() ==
          lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
        node.cleanup();
      }
    } catch (const std::exception& error) {
      RCLCPP_ERROR(
          node.get_logger(), "failed to enter lifecycle error path: %s",
          error.what());
    }
  }

  void PublishFeedback(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::uint8_t phase,
      const std::chrono::steady_clock::time_point started) {
    if (!goal_handle || !goal_handle->is_active()) {
      return;
    }
    auto feedback = std::make_shared<Action::Feedback>();
    feedback->phase = phase;
    feedback->elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    feedback->expanded_states = 0U;
    feedback->has_best_cost = false;
    feedback->best_cost = 0.0;
    try {
      goal_handle->publish_feedback(feedback);
    } catch (const std::exception& error) {
      RCLCPP_WARN(
          node.get_logger(), "failed to publish action feedback: %s",
          error.what());
    }
  }

  void PopulateLatestStamps(Action::Result& result) const {
    const auto store = Store();
    if (!store) {
      return;
    }
    const SnapshotStoreView view = store->Capture();
    if (view.global_map) {
      result.global_map_stamp = view.global_map->header.stamp;
    }
    if (view.local_map) {
      result.local_map_stamp = view.local_map->header.stamp;
    }
    if (view.odometry) {
      result.state_stamp = view.odometry->header.stamp;
    }
  }

  void FinishWorker(
      const std::uint64_t generation,
      const std::shared_ptr<GoalHandle>& goal_handle) {
    std::scoped_lock lock{state_mutex};
    if (generation == worker_generation) {
      worker_running = false;
      if (active_goal.get() == goal_handle.get()) {
        active_goal.reset();
      }
    }
  }

  void RequestWorkerStop(const std::string& reason) {
    std::scoped_lock lock{state_mutex};
    if (worker_running && worker) {
      worker_stop_reason = reason;
      worker->request_stop();
    }
  }

  void StopAndJoinWorker(const std::string& reason) {
    std::unique_ptr<std::jthread> joinable;
    {
      std::scoped_lock lock{state_mutex};
      if (!worker) {
        worker_running = false;
        active_goal.reset();
        return;
      }
      if (worker_running) {
        worker_stop_reason = reason;
      }
      worker->request_stop();
      if (worker->get_id() == std::this_thread::get_id()) {
        return;
      }
      joinable = std::move(worker);
    }
    if (joinable && joinable->joinable()) {
      joinable->join();
    }
    std::scoped_lock lock{state_mutex};
    worker_running = false;
    active_goal.reset();
  }

  void PublishDiagnostic(
      const std::uint8_t level,
      const std::string& reason_code,
      const lunar::planning::PlannerDiagnostics* diagnostics = nullptr,
      const lunar::planning::MotionReference* active_reference = nullptr) {
    std::scoped_lock lock{diagnostic_mutex};
    last_diagnostic_reason = reason_code;
    if (!diagnostics_publisher || !diagnostics_publisher->is_activated()) {
      return;
    }
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = node.now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.level = level;
    status.name = "lunar_planner_ros";
    status.hardware_id = diagnostic_hardware_id;
    status.message = reason_code;
    diagnostic_msgs::msg::KeyValue reason_value;
    reason_value.key = "reason_code";
    reason_value.value = reason_code;
    status.values.push_back(std::move(reason_value));
    const auto append = [&status](std::string key, std::string value) {
      diagnostic_msgs::msg::KeyValue entry;
      entry.key = std::move(key);
      entry.value = std::move(value);
      status.values.push_back(std::move(entry));
    };
    if (diagnostics != nullptr && diagnostics->hierarchical) {
      const auto& hierarchical = *diagnostics->hierarchical;
      append(
          "hierarchical_global_level",
          std::to_string(hierarchical.global_level));
      append(
          "hierarchical_global_resolution_m",
          DiagnosticDouble(hierarchical.global_resolution_m));
      append(
          "hierarchical_global_cells",
          std::to_string(hierarchical.global_cells));
      append(
          "hierarchical_global_elapsed_s",
          DiagnosticDouble(std::chrono::duration<double>(
              hierarchical.global_elapsed).count()));
      append(
          "hierarchical_local_elapsed_s",
          DiagnosticDouble(std::chrono::duration<double>(
              hierarchical.local_elapsed).count()));
      append(
          "hierarchical_global_expanded_states",
          std::to_string(hierarchical.global_expanded_states));
      append(
          "hierarchical_local_expanded_states",
          std::to_string(hierarchical.local_expanded_states));
      append(
          "hierarchical_global_open_peak",
          std::to_string(hierarchical.global_open_peak));
      append(
          "hierarchical_estimated_work_memory_bytes",
          std::to_string(hierarchical.estimated_work_memory_bytes));
      append(
          "hierarchical_raw_route_points",
          std::to_string(hierarchical.raw_route_points));
      append(
          "hierarchical_simplified_route_points",
          std::to_string(hierarchical.simplified_route_points));
      append(
          "hierarchical_local_frontier_distance_m",
          DiagnosticDouble(hierarchical.local_frontier_distance_m));
      append(
          "hierarchical_local_attempts",
          std::to_string(hierarchical.local_attempts));
      append(
          "hierarchical_corridor_width_m",
          DiagnosticDouble(hierarchical.corridor_width_m));
      append(
          "hierarchical_hopper_graph_nodes",
          std::to_string(hierarchical.hopper_graph_nodes));
      append(
          "hierarchical_hopper_graph_edges",
          std::to_string(hierarchical.hopper_graph_edges));
      append(
          "hierarchical_hopper_route_hops",
          std::to_string(hierarchical.hopper_route_hops));
      append(
          "hierarchical_hopper_certification_attempts",
          std::to_string(hierarchical.hopper_certification_attempts));
      append(
          "global_search_elapsed_s",
          DiagnosticDouble(std::chrono::duration<double>(
              hierarchical.global_elapsed).count()));
      append(
          "local_planning_elapsed_s",
          DiagnosticDouble(std::chrono::duration<double>(
              hierarchical.local_elapsed).count()));
      append(
          "landing_field_elapsed_s",
          DiagnosticDouble(std::chrono::duration<double>(
              hierarchical.landing_field_elapsed).count()));
      append(
          "spatial_index_elapsed_s",
          DiagnosticDouble(std::chrono::duration<double>(
              hierarchical.spatial_index_elapsed).count()));
      append(
          "ballistic_solve_elapsed_s",
          DiagnosticDouble(std::chrono::duration<double>(
              hierarchical.ballistic_solve_elapsed).count()));
      append(
          "flight_tube_certification_elapsed_s",
          DiagnosticDouble(std::chrono::duration<double>(
              hierarchical.flight_tube_certification_elapsed).count()));
      append(
          "global_expanded_nodes",
          std::to_string(hierarchical.global_expanded_states));
      append("open_peak", std::to_string(hierarchical.global_open_peak));
      append(
          "safe_landing_nodes",
          std::to_string(hierarchical.safe_landing_nodes));
      append(
          "candidate_edges_evaluated",
          std::to_string(hierarchical.candidate_edges_evaluated));
      append(
          "coarse_edges_rejected",
          std::to_string(hierarchical.coarse_edges_rejected));
      append(
          "full_edges_certified",
          std::to_string(hierarchical.full_edges_certified));
      append(
          "full_edges_invalidated",
          std::to_string(hierarchical.full_edges_invalidated));
      append(
          "edge_certificate_cache_hits",
          std::to_string(hierarchical.edge_certificate_cache_hits));
      append("route_reused", hierarchical.route_reused ? "true" : "false");
      append("route_cursor", std::to_string(hierarchical.route_cursor));
      append(
          "rolling_request_count",
          std::to_string(hierarchical.rolling_request_count));
    }
    if (diagnostics != nullptr && diagnostics->local_trajectory) {
      const auto& local = *diagnostics->local_trajectory;
      append("trajectory_mode", std::string{ToString(local.trajectory_mode)});
      append(
          "start_anchor_error_m",
          DiagnosticDouble(local.start_anchor_error_m));
      append("endpoint_error_m", DiagnosticDouble(local.endpoint_error_m));
      append(
          "maximum_curvature_per_m",
          DiagnosticDouble(local.maximum_curvature_per_m));
      append(
          "collision_validation",
          std::string{ToString(local.collision_validation)});
      append(
          "smoothing_elapsed_s",
          DiagnosticDouble(local.smoothing_elapsed_s));
      if (!diagnostics->hierarchical.has_value()) {
        append(
            "landing_field_elapsed_s",
            DiagnosticDouble(local.landing_field_elapsed_s));
      }
    }
    if (diagnostics != nullptr) {
      append("warning_codes", JoinWarningCodes(diagnostics->warning_codes));
    }
    const ExecutionDiagnosticInfo execution = ExecutionDiagnostic(
        execution_feedback_tracker.context(), active_reference);
    append("active_plan_id", execution.plan_id);
    append("active_segment_id", execution.segment_id);
    append("execution_state", execution.state);
    array.status.push_back(std::move(status));
    diagnostics_publisher->publish(array);
  }

  [[nodiscard]] std::array<rclcpp::CallbackGroup::SharedPtr, 6U>
  CallbackGroups() const {
    return {map_group, localization_group, propellant_group, tf_group,
            mission_group, action_group};
  }
};

PlanMotionServer::PlanMotionServer(
    const rclcpp::NodeOptions& options,
    PlanMotionServerDependencies dependencies)
    : rclcpp_lifecycle::LifecycleNode("lunar_planner", options),
      impl_(std::make_unique<Impl>(*this, std::move(dependencies))) {}

PlanMotionServer::~PlanMotionServer() = default;

PlanMotionServer::CallbackReturn PlanMotionServer::on_configure(
    const rclcpp_lifecycle::State&) {
  return impl_->Configure();
}

PlanMotionServer::CallbackReturn PlanMotionServer::on_activate(
    const rclcpp_lifecycle::State&) {
  return impl_->Activate();
}

PlanMotionServer::CallbackReturn PlanMotionServer::on_deactivate(
    const rclcpp_lifecycle::State&) {
  return impl_->Deactivate("PLANNER_DEACTIVATED");
}

PlanMotionServer::CallbackReturn PlanMotionServer::on_cleanup(
    const rclcpp_lifecycle::State&) {
  return impl_->Cleanup();
}

PlanMotionServer::CallbackReturn PlanMotionServer::on_shutdown(
    const rclcpp_lifecycle::State&) {
  return impl_->Deactivate("PLANNER_SHUTDOWN");
}

PlanMotionServer::CallbackReturn PlanMotionServer::on_error(
    const rclcpp_lifecycle::State&) {
  return impl_->Error();
}

std::size_t PlanMotionServer::callback_group_count_for_testing() const noexcept {
  return impl_->CallbackGroups().size();
}

bool PlanMotionServer::callback_groups_mutually_exclusive_for_testing() const {
  const auto groups = impl_->CallbackGroups();
  for (std::size_t first = 0U; first < groups.size(); ++first) {
    if (!groups[first] || groups[first]->type() !=
            rclcpp::CallbackGroupType::MutuallyExclusive) {
      return false;
    }
    for (std::size_t second = first + 1U; second < groups.size(); ++second) {
      if (groups[first] == groups[second]) {
        return false;
      }
    }
  }
  return true;
}

bool PlanMotionServer::worker_active_for_testing() const {
  std::scoped_lock lock{impl_->state_mutex};
  return impl_->worker_running;
}

std::optional<std::uint64_t>
PlanMotionServer::mission_revision_for_testing() const {
  std::scoped_lock lock{impl_->state_mutex};
  if (!impl_->mission) {
    return std::nullopt;
  }
  return impl_->mission->revision;
}

std::string PlanMotionServer::last_diagnostic_reason_for_testing() const {
  std::scoped_lock lock{impl_->diagnostic_mutex};
  return impl_->last_diagnostic_reason;
}

}  // namespace lunar::planning::ros
