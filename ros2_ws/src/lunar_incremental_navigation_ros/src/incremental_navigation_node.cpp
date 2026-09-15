#include "lunar_incremental_navigation_ros/incremental_navigation_node.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <action_msgs/srv/cancel_goal.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_planning_msgs/action/navigate_to_pose.hpp>
#include <lunar_planning_msgs/msg/path_reference.hpp>
#include <lunar_planning_msgs/msg/tracking_status.hpp>
#include <lunar_planning_msgs/srv/get_policy_map.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/subscription_options.hpp>
#include <rclcpp_action/create_server.hpp>
#include <rclcpp_action/server.hpp>
#include <rclcpp_action/server_goal_handle.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_incremental_navigation_core/global_route_planner.hpp"
#include "lunar_incremental_navigation_core/legged_local_planner.hpp"
#include "lunar_incremental_navigation_core/local_target_selector.hpp"
#include "lunar_incremental_navigation_core/request_local_start_patch.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"
#include "lunar_incremental_navigation_ros/elevation_pipeline.hpp"
#include "lunar_incremental_navigation_ros/input_store.hpp"
#include "lunar_incremental_navigation_ros/incremental_map_publisher.hpp"
#include "lunar_incremental_navigation_ros/map_adapters.hpp"
#include "lunar_incremental_navigation_ros/message_conversion.hpp"
#include "lunar_incremental_navigation_ros/platform_config.hpp"
#include "lunar_incremental_navigation_ros/policy_map_exporter.hpp"
#include "lunar_incremental_navigation_ros/request_diagnostics.hpp"
#include "lunar_incremental_navigation_ros/state_adapter.hpp"
#include "lunar_incremental_navigation_ros/traversability_qos.hpp"
#if defined(LUNAR_BUILD_DEMO)
#include "lunar_incremental_navigation_ros/planning_debug_publisher.hpp"
#endif

namespace lunar::incremental_navigation_ros {
namespace {

using Action = lunar_planning_msgs::action::NavigateToPose;
using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using namespace std::chrono_literals;
namespace core = lunar::incremental_navigation;

constexpr std::string_view kPackageName{"lunar_incremental_navigation_ros"};

struct RuntimeParameters final {
  std::string map_frame, odom_frame, base_frame;
  std::string platform_name;
  core::PlatformType platform_type;
  core::PlatformCapability capability;
  core::TraversabilityProfile profile;
  std::string local_map_topic;
  std::string local_map_qos_reliability;
  std::string local_map_qos_durability;
  std::chrono::milliseconds planning_sla;
  std::chrono::milliseconds planning_hard_timeout;
  std::chrono::milliseconds global_subdeadline;
  double coarse_resolution_m;
  double local_window_size_m;
  std::string odometry_topic;
  std::string tf_topic;
  std::string action_name;
  std::string path_reference_topic;
  std::string local_path_topic;
  std::string global_route_topic;
  std::string diagnostics_topic;
  std::string exploration_map_topic;
#if defined(LUNAR_BUILD_DEMO)
  std::optional<PlanningDebugPublisherConfig> debug_visualization;
#endif
};

[[nodiscard]] core::PlatformType ParsePlatform(const std::string_view value) {
  if (value == "wheel") {
    return core::PlatformType::kWheeled;
  }
  if (value == "legged") {
    return core::PlatformType::kLegged;
  }
  throw std::runtime_error("PLANNER_ERROR: unknown platform_type");
}

[[nodiscard]] std::filesystem::path ResolvePlatformConfig(
    const std::string_view platform, const std::string_view configured_path) {
  const std::filesystem::path share =
      ament_index_cpp::get_package_share_directory(std::string{kPackageName});
  if (configured_path.empty()) {
    return share / "config" / (std::string{platform} + ".yaml");
  }
  const std::filesystem::path path{configured_path};
  return path.is_absolute() ? path : share / "config" / path;
}

[[nodiscard]] RuntimeParameters ReadRuntimeParameters(rclcpp::Node& node) {
  const std::string platform_name =
      node.declare_parameter<std::string>("platform_type", "wheel");
  const std::string configured_path =
      node.declare_parameter<std::string>("platform_config", "");
  const core::PlatformType platform_type = ParsePlatform(platform_name);
  auto loaded = LoadPlatformConfig(
      ResolvePlatformConfig(platform_name, configured_path), platform_name);
  if (!loaded.capability) {
    throw std::runtime_error("PLANNER_ERROR: " + loaded.error_detail);
  }
  if (!loaded.start_blind_zone_margin_m) {
    throw std::runtime_error("PLANNER_ERROR: " + loaded.error_detail);
  }
  const std::int64_t planning_sla_ms =
      node.declare_parameter<std::int64_t>("planning_sla_ms", 2000);
  const std::int64_t planning_hard_timeout_ms =
      node.declare_parameter<std::int64_t>("planning_hard_timeout_ms", 3000);
  const std::int64_t global_subdeadline_ms =
      node.declare_parameter<std::int64_t>("global_subdeadline_ms", 500);
  const double coarse_resolution_m =
      node.declare_parameter<double>("coarse_resolution_m", 1.0);
  const double local_window_size_m =
      node.declare_parameter<double>("local_window_size_m", 64.0);
  if (planning_sla_ms <= 0 ||
      planning_hard_timeout_ms <= planning_sla_ms ||
      global_subdeadline_ms <= 0 ||
      global_subdeadline_ms >= planning_hard_timeout_ms ||
      !std::isfinite(coarse_resolution_m) || coarse_resolution_m <= 0.0 ||
      !std::isfinite(local_window_size_m) || local_window_size_m <= 0.0) {
    throw std::runtime_error(
        "PLANNER_ERROR: require 0 < planning_sla_ms < "
        "planning_hard_timeout_ms and 0 < global_subdeadline_ms < "
        "planning_hard_timeout_ms and coarse_resolution_m > 0 and "
        "local_window_size_m > 0");
  }
  RuntimeParameters parameters{
      .map_frame = node.declare_parameter<std::string>("map_frame", "map"),
      .odom_frame = node.declare_parameter<std::string>("odom_frame", "odom"),
      .base_frame = node.declare_parameter<std::string>("base_frame", "base_link"),
      .platform_name = platform_name,
      .platform_type = platform_type,
      .capability = std::move(*loaded.capability),
      .local_map_topic = node.declare_parameter<std::string>(
          "local_map_topic", "/Car/T3/mapping/grid_map"),
      .local_map_qos_reliability = node.declare_parameter<std::string>(
          "local_map_qos_reliability", "reliable"),
      .local_map_qos_durability = node.declare_parameter<std::string>(
          "local_map_qos_durability", "transient_local"),
      .planning_sla = std::chrono::milliseconds{planning_sla_ms},
      .planning_hard_timeout =
          std::chrono::milliseconds{planning_hard_timeout_ms},
      .global_subdeadline =
          std::chrono::milliseconds{global_subdeadline_ms},
      .coarse_resolution_m = coarse_resolution_m,
      .local_window_size_m = local_window_size_m,
      .odometry_topic = node.declare_parameter<std::string>(
          "odometry_topic", "/Car/T3/localization/odometry"),
      .tf_topic = node.declare_parameter<std::string>("tf_topic", "/tf"),
      .action_name = node.declare_parameter<std::string>(
          "action_name", "/Car/T4/navigation/navigate_to_pose"),
      .path_reference_topic = node.declare_parameter<std::string>(
          "path_reference_topic", "/Car/T4/planning/path_reference"),
      .local_path_topic = node.declare_parameter<std::string>(
          "local_path_topic", "/Car/T4/planning/local_path"),
      .global_route_topic = node.declare_parameter<std::string>(
          "global_route_topic", "/Car/T4/planning/global_route"),
      .diagnostics_topic = node.declare_parameter<std::string>(
          "diagnostics_topic", "/Car/T4/planning/diagnostics"),
      .exploration_map_topic = node.declare_parameter<std::string>(
          "exploration_map_topic", "/Car/T4/mapping/exploration_map"),
  };
#if defined(LUNAR_BUILD_DEMO)
  const bool debug_enabled =
      node.declare_parameter<bool>("enable_debug_visualization", false);
  const std::string debug_topic_prefix = node.declare_parameter<std::string>(
      "debug_topic_prefix", "/planning_demo/debug");
  const double debug_fine_window_m =
      node.declare_parameter<double>("debug_fine_window_m", 40.0);
  const double debug_cost_display_max =
      node.declare_parameter<double>("debug_cost_display_max", 2.0);
  if (debug_enabled) {
    parameters.debug_visualization = PlanningDebugPublisherConfig{
        .enabled = true,
        .topic_prefix = debug_topic_prefix,
        .fine_window_m = debug_fine_window_m,
        .cost_display_max = debug_cost_display_max,
    };
  }
#endif
  parameters.profile = PlatformProfileFor(
      parameters.capability, *loaded.start_blind_zone_margin_m);
  return parameters;
}

[[nodiscard]] std::string PlatformName(const core::PlatformType platform) {
  return platform == core::PlatformType::kWheeled ? "WHEELED" : "LEGGED";
}

[[nodiscard]] core::StateInput MissingState() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  return {.base_link_pose = {.position_m = {.x = nan, .y = nan},
                             .yaw_rad = nan}};
}

[[nodiscard]] std::string SessionHex(const core::SessionId& id) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string value;
  value.reserve(id.bytes.size() * 2U);
  for (const std::uint8_t byte : id.bytes) {
    value.push_back(kHex[byte >> 4U]);
    value.push_back(kHex[byte & 0x0fU]);
  }
  return value;
}

[[nodiscard]] std::string GuidanceName(const core::GuidanceStatus status) {
  switch (status) {
    case core::GuidanceStatus::kAvailable:
      return "AVAILABLE";
    case core::GuidanceStatus::kUnavailable:
      return "UNAVAILABLE";
    case core::GuidanceStatus::kNoRoute:
      return "NO_ROUTE";
    case core::GuidanceStatus::kTimeout:
      return "TIMEOUT";
  }
  return "UNAVAILABLE";
}

[[nodiscard]] double PathLength(const core::PathReference& reference) {
  double length{};
  for (std::size_t i = 1U; i < reference.path.poses.size(); ++i) {
    const auto& a = reference.path.poses[i - 1U].position_m;
    const auto& b = reference.path.poses[i].position_m;
    length += std::hypot(b.x - a.x, b.y - a.y);
  }
  return length;
}

[[nodiscard]] double DistanceToPath(const core::Point2 point,
                                    const core::GeometricPath& path) {
  if (path.poses.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  if (path.poses.size() == 1U) {
    return std::hypot(point.x - path.poses.front().position_m.x,
                      point.y - path.poses.front().position_m.y);
  }
  double nearest = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0U; index + 1U < path.poses.size(); ++index) {
    const auto& begin = path.poses[index].position_m;
    const auto& end = path.poses[index + 1U].position_m;
    const double dx = end.x - begin.x;
    const double dy = end.y - begin.y;
    const double length_squared = dx * dx + dy * dy;
    const double fraction = length_squared > 0.0
                                ? std::clamp(((point.x - begin.x) * dx +
                                              (point.y - begin.y) * dy) /
                                                 length_squared,
                                             0.0, 1.0)
                                : 0.0;
    nearest = std::min(
        nearest,
        std::hypot(point.x - std::lerp(begin.x, end.x, fraction),
                   point.y - std::lerp(begin.y, end.y, fraction)));
  }
  return nearest;
}

[[nodiscard]] core::CycleTrigger TriggerForState(
    const core::StateInput& state, const core::FineTraversabilitySnapshot& fine,
    const core::PathReference& active) {
  const double resolution = fine.geometry().resolution_m();
  const double segment_end_distance =
      std::max(0.5 * resolution, fine.hard_inflation_radius_m());
  const double maximum_deviation =
      std::max(resolution, 2.0 * fine.hard_inflation_radius_m());
  if (DistanceToPath(state.base_link_pose.position_m, active.path) >
      maximum_deviation) {
    return core::CycleTrigger::kDeviation;
  }
  const auto& end = active.path.poses.back().position_m;
  if (std::hypot(state.base_link_pose.position_m.x - end.x,
                 state.base_link_pose.position_m.y - end.y) <=
      segment_end_distance) {
    return core::CycleTrigger::kSegmentEnd;
  }
  return core::CycleTrigger::kContinue;
}

struct CycleMetrics final {
  bool global_route_reused{};
  core::SearchStatistics global_statistics;
  core::SearchStatistics local_statistics;
  std::size_t start_patch_assumed_cells{};
  double start_patch_radius_m{};
  bool start_patch_used{};
  double global_elapsed_ms{};
  double local_elapsed_ms{};
  double start_patch_elapsed_ms{};
  double postprocess_elapsed_ms{};
};

template <typename Function>
[[nodiscard]] double MeasureMilliseconds(Function&& function) {
  const auto begin = std::chrono::steady_clock::now();
  std::invoke(std::forward<Function>(function));
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - begin)
      .count();
}

}  // namespace

ActionTerminalState TerminalStateFor(
    const core::SessionOutcome outcome) noexcept {
  if (outcome == core::SessionOutcome::kGoalReached) {
    return ActionTerminalState::kSucceeded;
  }
  if (outcome == core::SessionOutcome::kCanceled) {
    return ActionTerminalState::kCanceled;
  }
  return ActionTerminalState::kAborted;
}

CancelHandlingDecision ClassifyCancelHandling(const bool active,
                                              const bool canceling) noexcept {
  if (!active) {
    return CancelHandlingDecision::kIgnore;
  }
  return canceling ? CancelHandlingDecision::kFinalize
                   : CancelHandlingDecision::kDefer;
}

rclcpp::QoS PathReferenceQos() {
  return rclcpp::QoS{rclcpp::KeepLast{1}}.reliable().transient_local();
}

rclcpp::QoS StateInputQos() {
  return rclcpp::QoS{rclcpp::KeepLast{10}}
      .best_effort()
      .durability_volatile();
}

core::TraversabilityProfile PlatformProfileFor(
    const core::PlatformCapability& capability,
    const double start_blind_zone_margin_m) {
  if (!std::isfinite(start_blind_zone_margin_m) ||
      start_blind_zone_margin_m < 0.0) {
    throw std::invalid_argument("invalid start blind-zone margin");
  }
  core::TraversabilityProfile profile;
  profile.slope_weight = 1.0;
  profile.relief_weight = 1.0;
  profile.clearance_weight = 0.0;
  profile.start_blind_zone_margin_m = start_blind_zone_margin_m;
  if (const auto* wheel = std::get_if<core::WheeledCapability>(&capability)) {
    profile.planar_envelope_xy_m = wheel->footprint_xy_m;
    profile.preferred_clearance_m = wheel->minimum_clearance_m;
    profile.maximum_slope_rad = wheel->maximum_slope_rad;
    profile.goal_position_tolerance_m = 0.30;
    profile.goal_yaw_tolerance_rad = 0.2617993877991494;
  } else if (const auto* legged =
                 std::get_if<core::LeggedCapability>(&capability)) {
    const double half_x = legged->body_extent_m.x / 2.0;
    const double half_y = legged->body_extent_m.y / 2.0;
    profile.planar_envelope_xy_m = {
        {-half_x, -half_y}, {half_x, -half_y},
        {half_x, half_y}, {-half_x, half_y}};
    profile.maximum_slope_rad = legged->maximum_slope_rad;
    profile.preferred_clearance_m = legged->minimum_body_clearance_m;
    profile.goal_position_tolerance_m = 0.20;
    profile.goal_yaw_tolerance_rad = 0.17453292519943295;
  }
  return profile;
}

std::unique_ptr<rclcpp::Executor> MakeIncrementalNavigationExecutor() {
  return std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
      rclcpp::ExecutorOptions{}, 4U);
}

core::PlanningSessionPorts RealSessionPorts(
    const core::PlatformCapability& capability) {
  auto global = std::make_shared<core::GlobalRoutePlanner>();
  auto selector = std::make_shared<core::LocalTargetSelector>();
  auto patch = std::make_shared<core::RequestLocalStartPatchBuilder>();
  auto wheel = std::get_if<core::WheeledCapability>(&capability)
                   ? std::make_shared<core::WheelLocalPlanner>(
                         *std::get_if<core::WheeledCapability>(&capability))
                   : nullptr;
  auto legged = std::get_if<core::LeggedCapability>(&capability)
                    ? std::make_shared<core::LeggedLocalPlanner>(
                          *std::get_if<core::LeggedCapability>(&capability))
                    : nullptr;
  return core::PlanningSessionPorts{
      .plan_global =
          [global](const core::GlobalGuidanceSnapshot& snapshot,
                   const core::Point2 start, const core::Point2 goal,
                   const core::SearchDeadline deadline,
                   const core::StopToken& stop) {
            return global->Plan(snapshot, start, goal, deadline, stop);
          },
      .select_target =
          [selector](const core::FineTraversabilitySnapshot& fine,
                     const core::SparseGridGeometry& local_window,
                     const core::Point2 start, const core::FinalGoal& goal,
                     const std::optional<core::GlobalRoute>& guidance) {
            return selector->SelectRolling(fine, local_window, start, goal,
                                    guidance);
          },
      .build_start_patch =
          [patch](std::shared_ptr<const core::FineTraversabilitySnapshot> fine,
                  const core::SparseGridGeometry& local_window,
                  const core::Pose2& p0,
                  const core::PlatformCapability& platform,
                  const core::TraversabilityProfile& profile) {
            return patch->Build(std::move(fine), local_window, p0, platform,
                                profile);
          },
      .plan_wheel =
          [wheel](const core::RequestLocalPlanningView& view,
                  const core::Pose2& start, const core::LocalTarget& target,
                  const core::SearchDeadline deadline,
                  const core::StopToken& stop) {
            return wheel ? wheel->Plan(view, start, target, deadline, stop)
                         : core::LocalPlanResult{};
          },
      .plan_legged =
          [legged](const core::RequestLocalPlanningView& view,
                   const core::Pose2& start, const core::LocalTarget& target,
                   const core::SearchDeadline deadline,
                   const core::StopToken& stop) {
            return legged ? legged->Plan(view, start, target, deadline, stop)
                          : core::LocalPlanResult{};
          },
  };
}

struct IncrementalNavigationNode::Impl final {
  enum class EventKind { kGoal, kCancel };
  struct ActionEvent final {
    EventKind kind;
    std::shared_ptr<GoalHandle> handle;
  };
  struct PendingTerminal final {
    std::shared_ptr<GoalHandle> handle;
    core::SessionTerminal terminal;
    std::function<void(const core::SessionTerminal&)> on_committed;
    std::chrono::steady_clock::time_point retry_not_before{};
    std::size_t failed_attempts{};
  };

  Impl(IncrementalNavigationNode& owner, IncrementalNavigationNodeDependencies dependencies)
      : node(owner),
        parameters(ReadRuntimeParameters(owner)),
        input_store(parameters.map_frame, parameters.odom_frame),
        pipeline(parameters.capability, parameters.profile,
                 parameters.coarse_resolution_m),
        exploration_map_publisher(node, parameters.exploration_map_topic),
        test_snapshots(std::move(dependencies.snapshots)),
        latest_state(std::move(dependencies.state)),
        state_source(std::move(dependencies.state_source)),
        snapshot_source(std::move(dependencies.snapshot_source)),
        event_sink(std::move(dependencies.event_sink)),
        before_goal_processing(
            std::move(dependencies.before_goal_processing)),
        before_terminal_commit(
            std::move(dependencies.before_terminal_commit)) {
    state_changed = latest_state.has_value();
    SessionPortsFactory factory = std::move(dependencies.ports_factory);
    core::PlanningSessionPorts ports =
        factory ? factory(parameters.capability)
                : RealSessionPorts(parameters.capability);
    tracking_feedback_enabled = node.declare_parameter<bool>("enable_tracking_feedback", false);
    coordinator = std::make_unique<core::PlanningSessionCoordinator>(
        parameters.capability, parameters.profile,
        core::PlanningSessionCoordinatorConfig{
            .goal_position_tolerance_m =
                parameters.profile.goal_position_tolerance_m,
            .goal_yaw_tolerance_rad =
                parameters.profile.goal_yaw_tolerance_rad,
            .local_window_size_m = parameters.local_window_size_m,
            .global_subdeadline = parameters.global_subdeadline,
            .require_execution_confirmation = tracking_feedback_enabled},
        Instrument(std::move(ports)));

    action_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    map_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    state_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    fine_derivation_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    guidance_derivation_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    planning_group = node.create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    auto local_map_qos = MakeTraversabilityInputQos(
        parameters.local_map_qos_reliability,
        parameters.local_map_qos_durability);
    if (!local_map_qos) {
      throw std::runtime_error(
          "PLANNER_ERROR: local map QoS must use reliability "
          "{reliable,best_effort} and durability "
          "{transient_local,volatile}");
    }
    const auto local_depth = node.declare_parameter<int>("local_map_qos_depth", 1);
    if (local_depth <= 0) throw std::invalid_argument("local_map_qos_depth must be positive");
    local_map_qos->keep_last(local_depth);
    if (parameters.map_frame.empty() || parameters.odom_frame.empty() || parameters.base_frame.empty() || parameters.map_frame == parameters.odom_frame || parameters.map_frame == parameters.base_frame || parameters.odom_frame == parameters.base_frame) {
      throw std::invalid_argument("map_frame, odom_frame, base_frame must be distinct nonempty names");
    }
    const auto state_qos = StateInputQos();

    path_publisher =
        node.create_publisher<lunar_planning_msgs::msg::PathReference>(
            parameters.path_reference_topic, PathReferenceQos());
    local_path_publisher = node.create_publisher<nav_msgs::msg::Path>(
        parameters.local_path_topic, PathReferenceQos());
    global_route_publisher = node.create_publisher<nav_msgs::msg::Path>(
        parameters.global_route_topic, PathReferenceQos());
    diagnostics_publisher =
        node.create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            parameters.diagnostics_topic,
            rclcpp::QoS{rclcpp::KeepLast{10}}.reliable());
    policy_map_exporter = std::make_unique<PolicyMapExporter>(
        parameters.capability, parameters.profile);
    policy_map_epoch = parameters.platform_name + ":" + parameters.map_frame + ":" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto policy_map_service_name = node.declare_parameter<std::string>(
        "policy_map_service", "/Car/T4/mapping/get_policy_map");
    policy_map_service = node.create_service<lunar_planning_msgs::srv::GetPolicyMap>(
        policy_map_service_name,
        [this](const std::shared_ptr<lunar_planning_msgs::srv::GetPolicyMap::Request> request,
               std::shared_ptr<lunar_planning_msgs::srv::GetPolicyMap::Response> response) {
          std::optional<core::StateInput> state;
          {
            std::scoped_lock lock{event_mutex};
            state = latest_state;
          }
          const auto policy_snapshot = pipeline.CapturePolicyMapSnapshot();
          const auto& bundle = policy_snapshot.bundle;
          std::optional<core::Pose2> anchor;
          if (state && std::isfinite(state->base_link_pose.position_m.x) &&
              std::isfinite(state->base_link_pose.position_m.y) &&
              std::isfinite(state->base_link_pose.yaw_rad)) {
            anchor = state->base_link_pose;
          }
          *response = policy_map_exporter->Export(
              {.fine = bundle.fine, .anchor = anchor,
               .local_window_size_m = parameters.local_window_size_m,
               .epoch = policy_map_epoch,
               .processed_stamp_ns = policy_snapshot.processed_map_stamp_ns},
              request->since_revision, request->minimum_map_stamp_ns);
        });

    const auto make_map_options = [this](const std::string& topic) {
      rclcpp::SubscriptionOptions options;
      options.callback_group = map_group;
      options.event_callbacks.message_lost_callback =
          [this](rclcpp::QOSMessageLostInfo& information) {
            if (information.total_count_change > 0) {
              middleware_lost_count.fetch_add(
                  static_cast<std::uint64_t>(information.total_count_change));
            }
          };
      options.event_callbacks.incompatible_qos_callback =
          [this, topic](rclcpp::QOSRequestedIncompatibleQoSInfo& information) {
            if (information.total_count_change > 0) {
              incompatible_qos_count.fetch_add(
                  static_cast<std::uint64_t>(information.total_count_change));
            }
            RCLCPP_ERROR(node.get_logger(),
                         "incompatible QoS on map input '%s' (policy=%d)",
                         topic.c_str(),
                         static_cast<int>(information.last_policy_kind));
            const auto update =
                LatestMapUpdate().value_or(core::ElevationUpdateResult{});
            PublishMapDiagnostics(update, 0.0, 0.0, 0.0);
          };
      return options;
    };
    const auto local_map_options =
        make_map_options(parameters.local_map_topic);
    rclcpp::SubscriptionOptions state_options;
    state_options.callback_group = state_group;
    local_subscription =
        node.create_subscription<grid_map_msgs::msg::GridMap>(
            parameters.local_map_topic, *local_map_qos,
            [this](grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
              OnLocalMap(std::move(message));
            },
            local_map_options);
    const auto tracking_topic = node.declare_parameter<std::string>(
        "tracking_status_topic", "/Car/T4/control/tracking_status");
    if (tracking_feedback_enabled) {
      tracking_subscription = node.create_subscription<lunar_planning_msgs::msg::TrackingStatus>(
          tracking_topic, rclcpp::QoS(10),
          [this](lunar_planning_msgs::msg::TrackingStatus::ConstSharedPtr message) {
            std::scoped_lock lock{event_mutex};
            pending_tracking = *message;
            if (event_sink) event_sink("tracking:RECEIVED");
          }, state_options);
    }
    odometry_subscription = node.create_subscription<nav_msgs::msg::Odometry>(
        parameters.odometry_topic, state_qos,
        [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
          OnOdometry(std::move(message));
        },
        state_options);
    tf_subscription = node.create_subscription<tf2_msgs::msg::TFMessage>(
        parameters.tf_topic, state_qos,
        [this](tf2_msgs::msg::TFMessage::ConstSharedPtr message) {
          input_store.UpdateTf(*message);
          RefreshState();
          MaybeApplyLatestLocalMap();
        },
        state_options);
#if defined(LUNAR_BUILD_DEMO)
    if (parameters.debug_visualization) {
      debug_publisher = std::make_unique<PlanningDebugPublisher>(
          node, *parameters.debug_visualization);
    }
#endif

    action_server = rclcpp_action::create_server<Action>(
        node.get_node_base_interface(), node.get_node_clock_interface(),
        node.get_node_logging_interface(), node.get_node_waitables_interface(),
        parameters.action_name,
        [](const rclcpp_action::GoalUUID&,
           const std::shared_ptr<const Action::Goal>) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](const std::shared_ptr<GoalHandle> handle) {
          static_cast<void>(coordinator->RequestStopCurrentCycle());
          bool internal_preemption = false;
          {
            std::scoped_lock lock{event_mutex};
            internal_preemption =
                preempt_requested_for.has_value() &&
                *preempt_requested_for == handle->get_goal_id();
          }
          if (!internal_preemption) {
            Enqueue({EventKind::kCancel, handle});
          }
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<GoalHandle> handle) {
          static_cast<void>(coordinator->RequestStopCurrentCycle());
          Enqueue({EventKind::kGoal, handle});
        },
        rcl_action_server_get_default_options(), action_group);
    cancel_client = node.create_client<action_msgs::srv::CancelGoal>(
        parameters.action_name + "/_action/cancel_goal");

    planning_timer = node.create_wall_timer(
        10ms, [this] { PlanningTick(); }, planning_group);
    fine_derivation_timer = node.create_wall_timer(
        10ms, [this] { RunFineDerivationWorker(); }, fine_derivation_group);
    guidance_derivation_timer = node.create_wall_timer(
        10ms, [this] { RunGuidanceDerivationWorker(); },
        guidance_derivation_group);
  }

  [[nodiscard]] core::PlanningSessionPorts Instrument(
      core::PlanningSessionPorts ports) {
    core::PlanningSessionPorts wrapped;
    if (ports.plan_global) {
      wrapped.plan_global =
          [this, function = std::move(ports.plan_global)](
              const core::GlobalGuidanceSnapshot& snapshot,
              const core::Point2 start, const core::Point2 goal,
              const core::SearchDeadline deadline,
              const core::StopToken& stop) {
            core::GlobalRouteResult result;
            cycle_metrics.global_elapsed_ms = MeasureMilliseconds(
                [&] { result = function(snapshot, start, goal, deadline, stop); });
            cycle_metrics.global_route_reused = result.reused_cache;
            cycle_metrics.global_statistics = result.statistics;
            return result;
          };
    }
    wrapped.select_target = std::move(ports.select_target);
    if (ports.build_start_patch) {
      wrapped.build_start_patch =
          [this, function = std::move(ports.build_start_patch)](
              std::shared_ptr<const core::FineTraversabilitySnapshot> fine,
              const core::SparseGridGeometry& local_window,
              const core::Pose2& p0,
              const core::PlatformCapability& capability,
              const core::TraversabilityProfile& profile) {
            core::StartPatchResult result;
            cycle_metrics.start_patch_elapsed_ms = MeasureMilliseconds([&] {
              result = function(std::move(fine), local_window, p0, capability,
                                profile);
            });
            cycle_metrics.start_patch_assumed_cells = result.assumed_cells;
            cycle_metrics.start_patch_radius_m =
                result.view ? result.view->start_patch_radius_m() : 0.0;
#if defined(LUNAR_BUILD_DEMO)
            if (debug_publisher && result.view) {
              debug_publisher->PublishStartPatch(
                  *result.view,
                  result.view->base()->hard_inflation_radius_m());
            }
#endif
            return result;
          };
    }
    const auto wrap_local = [this](auto function) {
      return [this, function = std::move(function)](
                 const core::RequestLocalPlanningView& view,
                 const core::Pose2& start, const core::LocalTarget& target,
                 const core::SearchDeadline deadline,
                 const core::StopToken& stop) {
        core::LocalPlanResult result;
        cycle_metrics.local_elapsed_ms = MeasureMilliseconds(
            [&] { result = function(view, start, target, deadline, stop); });
#if defined(LUNAR_BUILD_DEMO)
        if (debug_publisher) {
          debug_publisher->PublishLocalGoals(target, result, view.geometry().frame_id());
        }
#endif
        cycle_metrics.local_statistics = result.statistics;
        cycle_metrics.postprocess_elapsed_ms =
            std::chrono::duration<double, std::milli>(
                result.postprocess_elapsed)
                .count();
        const auto& path = result.path.empty() ? result.raw_path : result.path;
        cycle_metrics.start_patch_used = std::any_of(
            path.begin(), path.end(), [](const core::PathPoint& point) {
              return point.phase == core::StartPhase::kStartPrefix;
            });
        return result;
      };
    };
    if (ports.plan_wheel) {
      wrapped.plan_wheel = wrap_local(std::move(ports.plan_wheel));
    }
    if (ports.plan_legged) {
      wrapped.plan_legged = wrap_local(std::move(ports.plan_legged));
    }
    return wrapped;
  }

  void Enqueue(ActionEvent event) {
    std::scoped_lock lock{event_mutex};
    events.push_back(std::move(event));
  }

  void OnLocalMap(grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
    received_map_count.fetch_add(1U);
    input_store.UpdateLocal(std::move(message));
    MaybeApplyLatestLocalMap();
  }

  void MaybeApplyLatestLocalMap() {
    std::scoped_lock lock{local_map_apply_mutex};
    const InputSnapshot input = input_store.Capture();
    if (!input.local_map ||
        input.local_sequence == completed_local_map_sequence) {
      return;
    }
    if (has_local_map_attempt &&
        input.local_sequence == last_local_map_attempt_sequence &&
        input.tf_sequence == last_local_map_attempt_tf_sequence) {
      return;
    }
    has_local_map_attempt = true;
    last_local_map_attempt_sequence = input.local_sequence;
    last_local_map_attempt_tf_sequence = input.tf_sequence;

    const bool has_usable_map_from_odom =
        input.map_from_odom &&
        AdaptDirectMapFromOdom(*input.map_from_odom, parameters.map_frame, parameters.odom_frame).value.has_value();
    if (!has_usable_map_from_odom) {
      RCLCPP_DEBUG(node.get_logger(),
                   "local map deferred until direct map -> odom TF arrives");
      const auto update =
          LatestMapUpdate().value_or(core::ElevationUpdateResult{});
      PublishMapDiagnostics(update, 0.0, 0.0, 0.0);
      return;
    }
    const auto adapted = AdaptLocalElevation(input);
    const std::int64_t map_stamp_ns =
        static_cast<std::int64_t>(input.local_map->header.stamp.sec) * 1000000000LL +
        static_cast<std::int64_t>(input.local_map->header.stamp.nanosec);
    const auto begin = std::chrono::steady_clock::now();
    const core::ElevationUpdateResult update = pipeline.ApplyLocal(adapted, map_stamp_ns);
    const double raw_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - begin)
                              .count();
    completed_local_map_sequence = input.local_sequence;
    RememberMapUpdate(update);
    PublishMapDiagnostics(update, raw_ms, 0.0, 0.0);
  }

  void RememberMapUpdate(const core::ElevationUpdateResult& update) {
    std::scoped_lock lock{map_diagnostics_mutex};
    latest_map_update = update;
  }

  [[nodiscard]] std::optional<core::ElevationUpdateResult> LatestMapUpdate()
      const {
    std::scoped_lock lock{map_diagnostics_mutex};
    return latest_map_update;
  }

  void RunFineDerivationWorker() {
    if (pipeline.PendingFineDirtyTileCount() == 0U) {
      return;
    }
    bool fine_updated = false;
    double fine_ms{};
    try {
      const auto fine_begin = std::chrono::steady_clock::now();
      fine_updated = pipeline.RunFineDerivation();
      fine_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - fine_begin)
                    .count();
    } catch (const std::exception& error) {
      RCLCPP_WARN(node.get_logger(), "fine derivation deferred: %s",
                  error.what());
      return;
    }
    if (fine_updated) {
      std::scoped_lock lock{event_mutex};
      fine_changed = true;
    }
    if (const auto update = LatestMapUpdate()) {
      PublishMapDiagnostics(*update, 0.0, fine_ms, 0.0);
    }
  }

  void RunGuidanceDerivationWorker() {
    const std::uint64_t before = pipeline.GuidanceDerivationCount();
    double guidance_ms{};
    try {
      const auto guidance_begin = std::chrono::steady_clock::now();
      if (pipeline.RunGuidanceDerivation()) {
        static_cast<void>(
            exploration_map_publisher.Publish(pipeline.CaptureBundle()));
      }
      guidance_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - guidance_begin)
                        .count();
    } catch (const std::exception& error) {
      RCLCPP_WARN(node.get_logger(), "global guidance derivation deferred: %s",
                  error.what());
      return;
    }
    if (pipeline.GuidanceDerivationCount() != before) {
      if (const auto update = LatestMapUpdate()) {
        PublishMapDiagnostics(*update, 0.0, 0.0, guidance_ms);
      }
    }
  }

  void PublishMapDiagnostics(const core::ElevationUpdateResult& update,
                             const double raw_ms, const double fine_ms,
                             const double guidance_ms) {
    const core::ElevationMapCounters local = pipeline.LocalCounters();
    const core::SnapshotBundle bundle = CaptureBundle();
    diagnostics_publisher->publish(MakeMapDiagnostics(MapDiagnosticsRecord{
        .platform = PlatformName(parameters.platform_type),
        .raw_elevation_revision = update.raw_elevation_revision,
        .fine_traversability_revision =
            bundle.fine ? bundle.fine->fine_traversability_revision() : 0U,
        .global_guidance_revision =
            bundle.guidance ? bundle.guidance->global_guidance_revision() : 0U,
        .received_map_count = received_map_count.load(),
        .applied_map_count = local.applied_updates,
        .duplicate_map_count = local.duplicate_updates,
        .rejected_map_count = local.rejected_updates,
        .middleware_lost_count = middleware_lost_count.load(),
        .incompatible_qos_count = incompatible_qos_count.load(),
        .updated_cells = update.updated_cells,
        .dirty_tiles = update.dirty_tiles.size(),
        .derivation_lag = pipeline.PendingFineDirtyTileCount(),
        .allocated_tiles = local.allocated_tiles,
        .raw_update_ms = raw_ms,
        .fine_derivation_ms = fine_ms,
        .guidance_derivation_ms = guidance_ms,
    }));
  }

  void OnOdometry(nav_msgs::msg::Odometry::ConstSharedPtr message) {
    input_store.UpdateOdometry(std::move(message));
    RefreshState();
  }

  void RefreshState() {
    const InputSnapshot input = input_store.Capture();
    if (!input.odometry || !input.map_from_odom) {
      return;
    }
    const auto transform = AdaptDirectMapFromOdom(*input.map_from_odom, parameters.map_frame, parameters.odom_frame);
    if (!transform.value) {
      return;
    }
    const auto state = AdaptStateInput(*transform.value, *input.odometry, parameters.map_frame, parameters.odom_frame, parameters.base_frame);
    if (!state.value) {
      return;
    }
    std::scoped_lock lock{event_mutex};
    latest_state = *state.value;
    state_changed = true;
  }

  [[nodiscard]] core::SnapshotBundle CaptureBundle() const {
    return test_snapshots ? *test_snapshots : pipeline.CaptureBundle();
  }

  void PlanningTick() {
    if (state_source) {
      if (auto injected = state_source()) {
        std::scoped_lock lock{event_mutex};
        latest_state = *injected;
        state_changed = true;
      }
    }
    if (snapshot_source) {
      if (auto injected = snapshot_source()) {
        test_snapshots = std::move(*injected);
        std::scoped_lock lock{event_mutex};
        fine_changed = true;
      }
    }
    std::deque<ActionEvent> local_events;
    std::optional<lunar_planning_msgs::msg::TrackingStatus> tracking;
    bool process_fine{};
    bool process_state{};
    std::optional<core::StateInput> state;
    {
      std::scoped_lock lock{event_mutex};
      local_events.swap(events);
      tracking = std::exchange(pending_tracking, std::nullopt);
      process_fine = std::exchange(fine_changed, false);
      process_state = std::exchange(state_changed, false);
      state = latest_state;
    }
    for (auto& event : local_events) {
      if (event.kind == EventKind::kGoal) {
        ProcessGoal(std::move(event.handle), state.value_or(MissingState()));
      } else {
        ProcessCancel(event.handle);
      }
    }
    if (pending_terminal) {
      if (std::chrono::steady_clock::now() >=
          pending_terminal->retry_not_before) {
        static_cast<void>(TryCommitPendingTerminal());
      }
      if (pending_terminal) {
        return;
      }
    }
    if (replan_pending) {
      replan_pending = false;
      if (active_goal && active_goal->is_active()) {
        RunCycle(state.value_or(MissingState()),
                 core::CycleTrigger::kContinue);
        return;
      }
    }
    const core::SnapshotBundle bundle = CaptureBundle();
    if (state && bundle.fine) {
      exploration_map_publisher.PublishLocalFine(*bundle.fine, state->base_link_pose.position_m);
    }
#if defined(LUNAR_BUILD_DEMO)
    if (debug_publisher && state && bundle.fine) {
      debug_publisher->PublishSnapshots(bundle, state->base_link_pose.position_m);
    }
#endif
    if (process_fine && bundle.fine) {
      if (auto invalidated = coordinator->OnFineSnapshot(bundle.fine)) {
        PublishPath(*invalidated);
      }
      if (coordinator->state() == core::CoordinatorState::kReplanning) {
        RunCycle(state.value_or(MissingState()), core::CycleTrigger::kContinue);
        return;
      }
    }
    // Apply pending fine evidence before accepting execution completion: a
    // collision invalidates the reference, so its queued feedback is stale.
    // Only the stopped executor of the current reference may request replanning.
    // Publishing invalidation clears active_reference, naturally deduplicating it.
    if (tracking && state && active_goal && active_goal->is_active() &&
        active_reference &&
        (tracking->state == lunar_planning_msgs::msg::TrackingStatus::FAILED ||
         tracking->state == lunar_planning_msgs::msg::TrackingStatus::COMPLETED) &&
        tracking->session_id.uuid == active_goal->get_goal_id() &&
        tracking->segment_revision == active_reference->segment_revision &&
        std::isfinite(tracking->linear_speed_mps) &&
        std::isfinite(tracking->angular_speed_radps) &&
        std::abs(tracking->linear_speed_mps) <= 0.01 &&
        std::abs(tracking->angular_speed_radps) <= 0.03) {
      RunCycle(*state, tracking->state == lunar_planning_msgs::msg::TrackingStatus::FAILED
          ? core::CycleTrigger::kDeviation : core::CycleTrigger::kSegmentEnd);
      replan_pending = coordinator->state() == core::CoordinatorState::kReplanning;
      return;
    }
    if (process_state && active_goal && active_goal->is_active()) {
      core::CycleTrigger trigger = core::CycleTrigger::kContinue;
      if (state && active_reference && bundle.fine) {
        trigger = TriggerForState(*state, *bundle.fine, *active_reference);
        if (tracking_feedback_enabled && trigger == core::CycleTrigger::kSegmentEnd) {
          trigger = core::CycleTrigger::kContinue;
        }
      }
      RunCycle(state.value_or(MissingState()), trigger);
      if (trigger == core::CycleTrigger::kDeviation && active_goal &&
          active_goal->is_active() &&
          coordinator->state() == core::CoordinatorState::kReplanning) {
        replan_pending = true;
      }
    }
  }

  void ProcessGoal(std::shared_ptr<GoalHandle> handle,
                   const core::StateInput& state) {
    if (!handle) {
      return;
    }
    if (before_goal_processing) {
      before_goal_processing();
    }
    if (!handle->is_active()) {
      return;
    }
    if (handle->is_canceling()) {
      CommitStandaloneCanceled(handle);
      return;
    }
    if (pending_terminal) {
      Enqueue({EventKind::kGoal, std::move(handle)});
      return;
    }
    if (active_goal && active_goal->is_active()) {
      if (!active_goal->is_canceling()) {
        bool request_preemption = false;
        {
          std::scoped_lock lock{event_mutex};
          if (!preempt_requested_for) {
            preempt_requested_for = active_goal->get_goal_id();
            request_preemption = true;
          }
        }
        if (request_preemption) {
          auto request = std::make_shared<action_msgs::srv::CancelGoal::Request>();
          request->goal_info.goal_id.uuid = active_goal->get_goal_id();
          request->goal_info.stamp.sec = 0;
          request->goal_info.stamp.nanosec = 0U;
          cancel_client->async_send_request(request);
        }
        Enqueue({EventKind::kGoal, std::move(handle)});
        return;
      }
      if (!PublishTerminal(
              active_goal,
              coordinator->Cancel(core::CancelReason::kPreempted))) {
        Enqueue({EventKind::kGoal, std::move(handle)});
        return;
      }
      {
        std::scoped_lock lock{event_mutex};
        preempt_requested_for.reset();
      }
    }
    const core::SessionId id{handle->get_goal_id()};
    replan_pending = false;
    const GoalConversionResult converted = ConvertGoal(*handle->get_goal());
    if (!converted.ok()) {
      core::SessionTerminal terminal{
          .session_id = id,
          .result = {.outcome = core::SessionOutcome::kInvalidGoal,
                     .reason_code = "INVALID_GOAL"}};
      static_cast<void>(PublishTerminal(handle, terminal));
      return;
    }
    const core::StartSessionResult started = coordinator->Start(id, *converted.goal);
    if (!started.accepted || started.terminal) {
      static_cast<void>(PublishTerminal(
          handle, started.terminal.value_or(core::SessionTerminal{
                      .session_id = id,
                      .result = {
                          .outcome = core::SessionOutcome::kInternalError,
                          .reason_code = "INTERNAL_ERROR"}})));
      return;
    }
    active_goal = std::move(handle);
    RunCycle(state, core::CycleTrigger::kContinue);
  }

  void ProcessCancel(const std::shared_ptr<GoalHandle>& handle) {
    if (!handle) {
      return;
    }
    const bool belongs_to_active =
        active_goal && active_goal.get() == handle.get();
    const auto decision = ClassifyCancelHandling(
        handle->is_active(), handle->is_canceling());
    if (!belongs_to_active) {
      if (decision == CancelHandlingDecision::kFinalize) {
        CommitStandaloneCanceled(handle);
      } else if (decision == CancelHandlingDecision::kDefer) {
        Enqueue({EventKind::kCancel, handle});
      }
      return;
    }
    if (decision == CancelHandlingDecision::kDefer) {
      Enqueue({EventKind::kCancel, handle});
      return;
    }
    if (decision == CancelHandlingDecision::kFinalize) {
      if (pending_terminal &&
          pending_terminal->handle.get() == handle.get()) {
        pending_terminal->terminal.result.outcome =
            core::SessionOutcome::kCanceled;
        pending_terminal->terminal.result.reason_code = "CANCELED";
        pending_terminal->retry_not_before = {};
        static_cast<void>(TryCommitPendingTerminal());
        return;
      }
      static_cast<void>(PublishTerminal(
          active_goal, coordinator->Cancel(core::CancelReason::kCanceled)));
    }
  }

  void CommitStandaloneCanceled(const std::shared_ptr<GoalHandle>& handle) {
    if (!handle || !handle->is_active() || !handle->is_canceling()) {
      return;
    }
    auto result = std::make_shared<Action::Result>();
    result->outcome = Action::Result::CANCELED;
    result->reason_code = "CANCELED";
    try {
      handle->canceled(result);
      if (event_sink) {
        event_sink("terminal:CANCELED");
      }
    } catch (const rclcpp::exceptions::RCLError& error) {
      RCLCPP_WARN(node.get_logger(),
                  "standalone canceled terminal lost race: %s", error.what());
    } catch (const std::exception& error) {
      RCLCPP_ERROR(node.get_logger(),
                   "standalone canceled terminal failed: %s", error.what());
    }
  }

  void RunCycle(const core::StateInput& state,
                const core::CycleTrigger trigger) {
    if (!active_goal || !active_goal->is_active()) {
      return;
    }
    cycle_metrics = {};
    const auto started = std::chrono::steady_clock::now();
    const auto hard_deadline = started + parameters.planning_hard_timeout;
    const core::SnapshotBundle bundle = CaptureBundle();
    core::CycleOutput output = coordinator->PlanCycle(
        state, bundle, trigger, hard_deadline);
    const auto total_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
#if defined(LUNAR_BUILD_DEMO)
    if (debug_publisher) {
      debug_publisher->PublishSnapshots(bundle, state.base_link_pose.position_m);
    }
#endif
    if (output.global_route) {
      global_route_publisher->publish(ConvertGlobalRoute(output.global_route, parameters.map_frame));
    } else if (!output.terminal && output.feedback.reason_code == "PLAN_FOUND") {
      global_route_publisher->publish(ConvertGlobalRoute(std::nullopt, parameters.map_frame));
    }
    if (output.path_reference) {
      PublishPath(*output.path_reference);
    }
    if (active_goal && active_goal->is_active()) {
      active_goal->publish_feedback(
          std::make_shared<Action::Feedback>(ConvertFeedback(output.feedback)));
    }
    if (output.terminal) {
      if (output.terminal->invalidated) {
        PublishPath(*output.terminal->invalidated);
      }
      static_cast<void>(PublishTerminal(
          active_goal, *output.terminal, true,
          [this, output, bundle, total_elapsed](
              const core::SessionTerminal& committed) mutable {
            output.terminal = committed;
            PublishCycleDiagnostics(output, bundle, total_elapsed);
          }));
      return;
    }
    PublishCycleDiagnostics(output, bundle, total_elapsed);
  }

  void PublishPath(const core::PathReference& reference) {
    const auto converted = ConvertPathReference(reference, parameters.map_frame);
    path_publisher->publish(converted);
    local_path_publisher->publish(converted.path);
    last_reference = reference;
    if (reference.state == core::PathState::kActive) {
      active_reference = reference;
    } else if (active_reference &&
               active_reference->session_id == reference.session_id &&
               active_reference->segment_revision ==
                   reference.segment_revision) {
      active_reference.reset();
    }
    if (event_sink) {
      event_sink(reference.state == core::PathState::kActive
                         ? "path:ACTIVE"
                         : "path:INVALIDATED");
    }
  }

  void PublishCycleDiagnostics(const core::CycleOutput& output,
                               const core::SnapshotBundle& bundle,
                               const std::chrono::nanoseconds total_elapsed) {
    const core::PathReference* path = nullptr;
    if (output.terminal && output.terminal->invalidated) {
      path = &*output.terminal->invalidated;
    } else if (output.terminal && last_reference &&
               last_reference->session_id == output.terminal->session_id &&
               last_reference->state == core::PathState::kInvalidated &&
               last_reference->segment_revision ==
                   output.terminal->result.last_segment_revision) {
      path = &*last_reference;
    } else if (output.path_reference) {
      path = &*output.path_reference;
    }
    const std::string reason = output.terminal
                                   ? output.terminal->result.reason_code
                                   : output.feedback.reason_code;
    diagnostics_publisher->publish(MakePlanningCycleDiagnostics(
        PlanningCycleDiagnosticsRecord{
            .session_id = output.terminal
                ? SessionHex(output.terminal->session_id)
                : (active_goal
                       ? SessionHex(
                             core::SessionId{active_goal->get_goal_id()})
                       : std::string{}),
            .planning_cycle = output.feedback.planning_cycle,
            .segment_revision = output.terminal
                ? output.terminal->result.last_segment_revision
                : (path ? path->segment_revision
                        : output.feedback.active_segment_revision),
            .cycle_result = reason,
            .reason_code = reason,
            .path_state = path
                ? (path->state == core::PathState::kActive ? "ACTIVE"
                                                           : "INVALIDATED")
                : "NONE",
            .reaches_final_goal = path && path->reaches_final_goal,
            .fine_traversability_revision = path
                ? path->traversability_revision
                : (bundle.fine ? bundle.fine->fine_traversability_revision()
                               : 0U),
            .global_guidance_revision = bundle.guidance
                ? bundle.guidance->global_guidance_revision()
                : 0U,
            .global_guidance_status = GuidanceName(output.guidance_status),
            .global_route_reused = cycle_metrics.global_route_reused,
            .global_elapsed_ms = cycle_metrics.global_elapsed_ms,
            .global_expanded_states =
                cycle_metrics.global_statistics.expanded_states,
            .global_open_peak = cycle_metrics.global_statistics.open_peak,
            .local_elapsed_ms = cycle_metrics.local_elapsed_ms,
            .local_expanded_states =
                cycle_metrics.local_statistics.expanded_states,
            .local_open_peak = cycle_metrics.local_statistics.open_peak,
            .start_patch_used = cycle_metrics.start_patch_used,
            .start_patch_radius_m = cycle_metrics.start_patch_radius_m,
            .start_patch_assumed_cells =
                cycle_metrics.start_patch_assumed_cells,
            .start_patch_elapsed_ms = cycle_metrics.start_patch_elapsed_ms,
            .postprocess_elapsed_ms = cycle_metrics.postprocess_elapsed_ms,
            .total_elapsed_ms =
                std::chrono::duration<double, std::milli>(total_elapsed)
                    .count(),
            .latency_class = ClassifyPlanningCycleLatency(
                total_elapsed, parameters.planning_sla,
                parameters.planning_hard_timeout),
            .path_points = path ? path->path.poses.size() : 0U,
            .path_length_m = path ? PathLength(*path) : 0.0,
        }));
  }

  [[nodiscard]] bool TryCommitPendingTerminal() {
    if (!pending_terminal) {
      return true;
    }
    auto handle = pending_terminal->handle;
    if (!handle || !handle->is_active()) {
      if (active_goal.get() == handle.get()) {
        active_goal.reset();
        replan_pending = false;
      }
      pending_terminal.reset();
      return true;
    }
    core::SessionTerminal effective = pending_terminal->terminal;
    ActionTerminalState transition = TerminalStateFor(effective.result.outcome);
    if (handle->is_canceling()) {
      if (effective.result.outcome != core::SessionOutcome::kCanceled) {
        effective.result.reason_code = "CANCELED";
      }
      effective.result.outcome = core::SessionOutcome::kCanceled;
      transition = ActionTerminalState::kCanceled;
    }

    bool terminal_committed = false;
    try {
      if (before_terminal_commit) {
        before_terminal_commit();
      }
      if (handle->is_canceling()) {
        if (effective.result.outcome != core::SessionOutcome::kCanceled) {
          effective.result.reason_code = "CANCELED";
        }
        effective.result.outcome = core::SessionOutcome::kCanceled;
        transition = ActionTerminalState::kCanceled;
      }
      auto result =
          std::make_shared<Action::Result>(ConvertResult(effective.result));
      switch (transition) {
        case ActionTerminalState::kSucceeded:
          handle->succeed(result);
          terminal_committed = true;
          break;
        case ActionTerminalState::kCanceled:
          if (!handle->is_canceling()) {
            break;
          }
          handle->canceled(result);
          terminal_committed = true;
          break;
        case ActionTerminalState::kAborted:
          handle->abort(result);
          terminal_committed = true;
          break;
      }
    } catch (const rclcpp::exceptions::RCLError& error) {
      RCLCPP_WARN(node.get_logger(), "action terminal commit lost race: %s",
                  error.what());
    } catch (const std::exception& error) {
      RCLCPP_WARN(node.get_logger(), "action terminal commit deferred: %s",
                  error.what());
    }

    if (!terminal_committed && handle->is_active() &&
        handle->is_canceling()) {
      if (effective.result.outcome != core::SessionOutcome::kCanceled) {
        effective.result.reason_code = "CANCELED";
      }
      effective.result.outcome = core::SessionOutcome::kCanceled;
      try {
        handle->canceled(
            std::make_shared<Action::Result>(ConvertResult(effective.result)));
        terminal_committed = true;
      } catch (const rclcpp::exceptions::RCLError& error) {
        RCLCPP_WARN(node.get_logger(),
                    "canceled terminal commit deferred: %s", error.what());
      } catch (const std::exception& error) {
        RCLCPP_WARN(node.get_logger(),
                    "canceled terminal commit deferred: %s", error.what());
      }
    }

    if (!terminal_committed) {
      if (!handle->is_active()) {
        if (active_goal.get() == handle.get()) {
          active_goal.reset();
          replan_pending = false;
        }
        pending_terminal.reset();
        return true;
      }
      pending_terminal->terminal = std::move(effective);
      ++pending_terminal->failed_attempts;
      pending_terminal->retry_not_before = std::chrono::steady_clock::now() +
          (pending_terminal->failed_attempts == 1U ? 0ms : 50ms);
      return false;
    }

    const std::string terminal_reason = effective.result.reason_code;
    auto on_committed = std::move(pending_terminal->on_committed);
    if (event_sink) {
      event_sink("terminal:" + terminal_reason);
    }
    if (active_goal.get() == handle.get()) {
      active_goal.reset();
      replan_pending = false;
    }
    pending_terminal.reset();
    if (on_committed) {
      try {
        on_committed(effective);
      } catch (const std::exception& error) {
        RCLCPP_WARN(node.get_logger(),
                    "terminal diagnostics publication failed: %s",
                    error.what());
      }
    }
    return true;
  }

  [[nodiscard]] bool PublishTerminal(
      const std::shared_ptr<GoalHandle>& handle,
      const core::SessionTerminal& terminal,
      const bool invalidation_already_published = false,
      std::function<void(const core::SessionTerminal&)> on_committed = {}) {
    if (!handle || !handle->is_active()) {
      return true;
    }
    if (pending_terminal) {
      if (pending_terminal->handle.get() != handle.get()) {
        return false;
      }
      if (terminal.result.outcome == core::SessionOutcome::kCanceled) {
        pending_terminal->terminal.result.outcome =
            core::SessionOutcome::kCanceled;
        pending_terminal->terminal.result.reason_code =
            terminal.result.reason_code;
      }
      if (on_committed) {
        pending_terminal->on_committed = std::move(on_committed);
      }
      return TryCommitPendingTerminal();
    }
    if (terminal.invalidated && !invalidation_already_published) {
      PublishPath(*terminal.invalidated);
    }
    global_route_publisher->publish(ConvertGlobalRoute(std::nullopt, parameters.map_frame));
    pending_terminal = PendingTerminal{
        .handle = handle,
        .terminal = terminal,
        .on_committed = std::move(on_committed)};
    return TryCommitPendingTerminal();
  }

  ~Impl() {
    // Goal-handle destruction auto-cancels unfinished goals. After context
    // shutdown, the server can no longer publish their results. Drop its owner
    // first so the handles' weak terminal callbacks cannot enter a dead server.
    action_server.reset();
  }

  IncrementalNavigationNode& node;
  RuntimeParameters parameters;
  InputStore input_store;
  ElevationPipeline pipeline;
  IncrementalMapPublisher exploration_map_publisher;
  std::optional<core::SnapshotBundle> test_snapshots;
  std::unique_ptr<core::PlanningSessionCoordinator> coordinator;
  std::unique_ptr<PolicyMapExporter> policy_map_exporter;
  std::string policy_map_epoch;
  CycleMetrics cycle_metrics;

  rclcpp::CallbackGroup::SharedPtr action_group;
  rclcpp::CallbackGroup::SharedPtr map_group;
  rclcpp::CallbackGroup::SharedPtr state_group;
  rclcpp::CallbackGroup::SharedPtr fine_derivation_group;
  rclcpp::CallbackGroup::SharedPtr guidance_derivation_group;
  rclcpp::CallbackGroup::SharedPtr planning_group;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr
      local_subscription;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscription;
  rclcpp::Publisher<lunar_planning_msgs::msg::PathReference>::SharedPtr
      path_publisher;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_publisher;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_route_publisher;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_publisher;
  rclcpp::Service<lunar_planning_msgs::srv::GetPolicyMap>::SharedPtr
      policy_map_service;
#if defined(LUNAR_BUILD_DEMO)
  std::unique_ptr<PlanningDebugPublisher> debug_publisher;
#endif
  rclcpp_action::Server<Action>::SharedPtr action_server;
  rclcpp::Client<action_msgs::srv::CancelGoal>::SharedPtr cancel_client;
  rclcpp::TimerBase::SharedPtr planning_timer;
  rclcpp::TimerBase::SharedPtr fine_derivation_timer;
  rclcpp::TimerBase::SharedPtr guidance_derivation_timer;

  rclcpp::Subscription<lunar_planning_msgs::msg::TrackingStatus>::SharedPtr tracking_subscription;
  std::optional<lunar_planning_msgs::msg::TrackingStatus> pending_tracking;
  std::mutex event_mutex;
  std::mutex local_map_apply_mutex;
  mutable std::mutex map_diagnostics_mutex;
  std::optional<core::ElevationUpdateResult> latest_map_update;
  std::uint64_t completed_local_map_sequence{};
  std::uint64_t last_local_map_attempt_sequence{};
  std::uint64_t last_local_map_attempt_tf_sequence{};
  bool has_local_map_attempt{};
  std::deque<ActionEvent> events;
  std::optional<rclcpp_action::GoalUUID> preempt_requested_for;
  std::optional<core::StateInput> latest_state;
  std::function<std::optional<core::StateInput>()> state_source;
  std::function<std::optional<core::SnapshotBundle>()> snapshot_source;
  bool state_changed{};
  bool fine_changed{};
  bool replan_pending{};
  bool tracking_feedback_enabled{};
  std::shared_ptr<GoalHandle> active_goal;
  std::optional<core::PathReference> active_reference;
  std::optional<core::PathReference> last_reference;
  std::optional<PendingTerminal> pending_terminal;
  std::function<void(const std::string&)> event_sink;
  std::function<void()> before_goal_processing;
  std::function<void()> before_terminal_commit;
  std::atomic<std::uint64_t> received_map_count{};
  std::atomic<std::uint64_t> middleware_lost_count{};
  std::atomic<std::uint64_t> incompatible_qos_count{};
};

IncrementalNavigationNode::IncrementalNavigationNode(
    const rclcpp::NodeOptions& options,
    IncrementalNavigationNodeDependencies dependencies)
    : rclcpp::Node("incremental_navigation", options),
      impl_(std::make_unique<Impl>(*this, std::move(dependencies))) {
  startup_parameters_ = add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& values) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    for (const auto& value : values) {
      if (value.get_name() == "use_sim_time") continue;
      if (has_parameter(value.get_name()) && get_parameter(value.get_name()).get_parameter_value() != value.get_parameter_value()) {
        result.successful = false;
        result.reason = value.get_name() + ": startup-only parameter; edit configuration and restart node";
        break;
      }
    }
    return result;
  });
}

IncrementalNavigationNode::~IncrementalNavigationNode() = default;

}  // namespace lunar::incremental_navigation_ros
