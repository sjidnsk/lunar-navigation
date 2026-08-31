#include "lunar_pure_planner_ros/pure_plan_motion_server.hpp"

#include "accepted_goal_finalizer.hpp"
#include "action_execution_state.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <lunar_planning_msgs/msg/timed_path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/subscription_options.hpp>
#include <rclcpp_action/create_server.hpp>
#include <rclcpp_action/server.hpp>
#include <rclcpp_action/server_goal_handle.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_pure_planner_core/planner.hpp"
#include "grid_v1/grid_v1_planner.hpp"
#include "hierarchical/frame_transform.hpp"
#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/reference_composer.hpp"
#include "hierarchical/surface_portal_set.hpp"
#include "hierarchical/surface_rolling_session.hpp"
#include "lunar_pure_planner_ros/input_store.hpp"
#include "lunar_pure_planner_ros/incremental_traversability.hpp"
#include "lunar_pure_planner_ros/map_adapters.hpp"
#include "lunar_pure_planner_ros/message_conversion.hpp"
#include "lunar_pure_planner_ros/platform_config.hpp"
#include "lunar_pure_planner_ros/request_diagnostics.hpp"
#include "lunar_pure_planner_ros/state_adapter.hpp"
#include "lunar_pure_planner_ros/traversability_input.hpp"
#include "lunar_pure_planner_ros/trusted_bridge.hpp"

namespace lunar::pure_planner_ros {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using namespace std::chrono_literals;

constexpr std::string_view kPackageName{"lunar_pure_planner_ros"};
constexpr double kLeggedRollingHorizonM = 4.0;
constexpr double kLeggedReplanStrideM = 1.5;
constexpr double kLeggedGoalToleranceEpsilonM = 1.0e-9;

struct RollingSurfaceParameters final {
  bool enabled{};
  double horizon_m{8.0};
  std::int64_t poll_period_ms{100};
  std::int64_t min_replan_interval_ms{500};
  double max_deviation_m{2.0};
};

struct RuntimeParameters final {
  std::string platform_name;
  lunar::pure_planning::PlatformType platform_type;
  lunar::pure_planning::PlatformCapability capability;
  lunar::pure_planning::AnytimePlannerConfig planner_config;
  std::string global_map_topic;
  std::string local_map_topic;
  std::string odometry_topic;
  std::string tf_topic;
  std::string action_name;
  std::string diagnostics_topic;
  std::string wheeled_reference_topic;
  std::string wheeled_path_topic;
  std::string wheeled_global_path_topic;
  std::string wheeled_timed_path_topic;
  std::string legged_path_topic;
  std::string legged_global_path_topic;
  RollingSurfaceParameters rolling_surface;
};

[[nodiscard]] lunar::pure_planning::PlatformType ParsePlatform(
    const std::string_view platform) {
  if (platform == "wheel") {
    return lunar::pure_planning::PlatformType::kWheeled;
  }
  if (platform == "legged") {
    return lunar::pure_planning::PlatformType::kLegged;
  }
  if (platform == "hopper") {
    return lunar::pure_planning::PlatformType::kHopper;
  }
  throw std::runtime_error{"PLANNER_ERROR: unknown platform_type"};
}

[[nodiscard]] lunar::pure_planning::WheelPlannerMode ParseWheelPlannerMode(
    const std::string_view mode) {
  if (mode == "legacy_certified") {
    return lunar::pure_planning::WheelPlannerMode::kLegacyCertified;
  }
  if (mode == "grid_traversability_v1") {
    return lunar::pure_planning::WheelPlannerMode::kGridTraversabilityV1;
  }
  throw std::runtime_error{"PLANNER_ERROR: unknown wheel_planner_mode"};
}

[[nodiscard]] lunar::pure_planning::LeggedGlobalMode ParseLeggedGlobalMode(
    const std::string_view mode) {
  if (mode == "legacy_occupancy") {
    return lunar::pure_planning::LeggedGlobalMode::kLegacyOccupancy;
  }
  if (mode == "grid_traversability_v1") {
    return lunar::pure_planning::LeggedGlobalMode::kGridTraversabilityV1;
  }
  throw std::runtime_error{"PLANNER_ERROR: unknown legged_global_mode"};
}

[[nodiscard]] std::filesystem::path ResolvePlatformConfig(
    const std::string_view platform, const std::string_view configured_path) {
  const std::filesystem::path share =
      ament_index_cpp::get_package_share_directory(std::string{kPackageName});
  if (configured_path.empty()) {
    return share / "config" / (std::string{platform} + ".yaml");
  }
  const std::filesystem::path explicit_path{configured_path};
  return explicit_path.is_absolute() ? explicit_path
                                     : share / "config" / explicit_path;
}

[[nodiscard]] bool AbsoluteTopic(const std::string& value) noexcept {
  return value.size() > 1U && value.front() == '/';
}

[[nodiscard]] RuntimeParameters ReadRuntimeParameters(rclcpp::Node& node) {
  const std::string platform =
      node.declare_parameter<std::string>("platform_type", "wheel");
  const std::string platform_config =
      node.declare_parameter<std::string>("platform_config", "");
  const auto platform_type = ParsePlatform(platform);
  const std::filesystem::path config_path =
      ResolvePlatformConfig(platform, platform_config);
  auto loaded = LoadPlatformConfig(config_path, platform);
  if (!loaded.capability.has_value()) {
    throw std::runtime_error{"PLANNER_ERROR: platform_config rejected: " +
                             config_path.string()};
  }

  const std::int64_t global_threshold = node.declare_parameter<std::int64_t>(
      "global_occupancy_threshold_percent", 50);
  const double local_threshold =
      node.declare_parameter<double>("local_occupancy_threshold", 0.5);
  const auto wheel_planner_mode = ParseWheelPlannerMode(
      node.declare_parameter<std::string>("wheel_planner_mode",
                                          "legacy_certified"));
  const auto legged_global_mode = ParseLeggedGlobalMode(
      node.declare_parameter<std::string>("legged_global_mode",
                                          "grid_traversability_v1"));
  if (global_threshold < 0 || global_threshold > 100 ||
      !std::isfinite(local_threshold) || local_threshold < 0.0 ||
      local_threshold > 1.0) {
    throw std::runtime_error{"PLANNER_ERROR: occupancy threshold invalid"};
  }

  const RollingSurfaceParameters rolling_surface{
      .enabled = node.declare_parameter<bool>("rolling_surface_enabled", false),
      .horizon_m = node.declare_parameter<double>("rolling_horizon_m", 8.0),
      .poll_period_ms =
          node.declare_parameter<std::int64_t>("rolling_poll_period_ms", 100),
      .min_replan_interval_ms = node.declare_parameter<std::int64_t>(
          "rolling_min_replan_interval_ms", 500),
      .max_deviation_m =
          node.declare_parameter<double>("rolling_max_deviation_m", 2.0),
  };
  if (!std::isfinite(rolling_surface.horizon_m) ||
      rolling_surface.horizon_m <= 0.0 ||
      !std::isfinite(rolling_surface.max_deviation_m) ||
      rolling_surface.max_deviation_m <= 0.0 ||
      rolling_surface.poll_period_ms < 10 ||
      rolling_surface.poll_period_ms > 10000 ||
      rolling_surface.min_replan_interval_ms < 10 ||
      rolling_surface.min_replan_interval_ms > 10000) {
    throw std::runtime_error{"PLANNER_ERROR: rolling parameter invalid"};
  }

  RuntimeParameters parameters{
      .platform_name = platform,
      .platform_type = platform_type,
      .capability = std::move(*loaded.capability),
      .planner_config = {
          .global_occupancy_threshold =
              static_cast<std::int32_t>(global_threshold),
          .local_occupancy_threshold = local_threshold,
          .wheel_planner_mode = wheel_planner_mode,
          .legged_global_mode = legged_global_mode,
      },
      .global_map_topic = node.declare_parameter<std::string>(
          "global_map_topic", "/Car/T3/mapping/global_overview"),
      .local_map_topic = node.declare_parameter<std::string>(
          "local_map_topic", "/Car/T3/mapping/grid_map"),
      .odometry_topic = node.declare_parameter<std::string>(
          "odometry_topic", "/Car/T3/localization/odometry"),
      .tf_topic = node.declare_parameter<std::string>("tf_topic", "/tf"),
      .action_name = node.declare_parameter<std::string>(
          "action_name", "/Car/T4/plan_motion"),
      .diagnostics_topic = node.declare_parameter<std::string>(
          "diagnostics_topic", "/Car/T4/planning/diagnostics"),
      .wheeled_reference_topic = node.declare_parameter<std::string>(
          "wheeled_reference_topic", "/Car/T4/planning/wheeled_reference"),
      .wheeled_path_topic = node.declare_parameter<std::string>(
          "wheeled_path_topic", "/Car/T4/planning/wheeled_path"),
      .wheeled_global_path_topic = node.declare_parameter<std::string>(
          "wheeled_global_path_topic",
          "/Car/T4/planning/wheeled_global_path"),
      .wheeled_timed_path_topic = node.declare_parameter<std::string>(
          "wheeled_timed_path_topic", "/Car/T4/planning/wheeled_path_timing"),
      .legged_path_topic = node.declare_parameter<std::string>(
          "legged_path_topic", "/Car/T4/planning/legged_path"),
      .legged_global_path_topic = node.declare_parameter<std::string>(
          "legged_global_path_topic", "/Car/T4/planning/legged_global_path"),
      .rolling_surface = rolling_surface,
  };
  if (parameters.planner_config.wheel_planner_mode ==
          lunar::pure_planning::WheelPlannerMode::kGridTraversabilityV1 &&
      parameters.platform_type !=
          lunar::pure_planning::PlatformType::kWheeled) {
    throw std::runtime_error{
        "PLANNER_ERROR: grid_traversability_v1 requires wheel"};
  }
  for (const std::string* interface_name : {
           &parameters.global_map_topic, &parameters.local_map_topic,
           &parameters.odometry_topic, &parameters.tf_topic,
           &parameters.action_name, &parameters.diagnostics_topic,
           &parameters.wheeled_reference_topic, &parameters.wheeled_path_topic,
           &parameters.wheeled_global_path_topic,
           &parameters.wheeled_timed_path_topic,
           &parameters.legged_path_topic,
           &parameters.legged_global_path_topic}) {
    if (!AbsoluteTopic(*interface_name)) {
      throw std::runtime_error{"PLANNER_ERROR: interface name must be absolute"};
    }
  }
  return parameters;
}

[[nodiscard]] lunar::pure_planning::EnvironmentMode EnvironmentMode(
    const std::uint8_t value) noexcept {
  if (value == Action::Goal::LAVA_TUBE) {
    return lunar::pure_planning::EnvironmentMode::kLavaTube;
  }
  return value == Action::Goal::LUNAR_SURFACE
             ? lunar::pure_planning::EnvironmentMode::kLunarSurface
             : static_cast<lunar::pure_planning::EnvironmentMode>(value);
}

[[nodiscard]] const lunar::pure_planning::Pose3* PlatformPose(
    const lunar::pure_planning::PlatformState& state) noexcept {
  return std::visit(
      [](const auto& value) -> const lunar::pure_planning::Pose3* {
        using State = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<State,
                                     lunar::pure_planning::LeggedState>) {
          return &value.body_pose;
        } else {
          return &value.pose;
        }
      },
      state);
}

[[nodiscard]] lunar::pure_planning::PlanningResult Failure(
    const lunar::pure_planning::PlanningStatus status,
    std::string reason_code) {
  return {.status = status, .reason_code = std::move(reason_code)};
}

[[nodiscard]] lunar::pure_planning::PlanningResult LocalFailure(
    const lunar::pure_planning::LocalPlanStatus status,
    std::string reason_code) {
  using lunar::pure_planning::LocalPlanStatus;
  using lunar::pure_planning::PlanningStatus;
  switch (status) {
    case LocalPlanStatus::kNoPath:
      return Failure(PlanningStatus::kNoPath, "NO_PATH");
    case LocalPlanStatus::kTimedOut:
      return Failure(PlanningStatus::kTimedOut, "TIMEOUT");
    case LocalPlanStatus::kCanceled:
      return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED");
    case LocalPlanStatus::kInvalidInput:
      return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
    case LocalPlanStatus::kPlannerError:
      return Failure(PlanningStatus::kPlannerError,
                     reason_code.empty() ? "PLANNER_ERROR"
                                         : std::move(reason_code));
    case LocalPlanStatus::kSolved:
      break;
  }
  return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
}

[[nodiscard]] lunar::pure_planning::PlanningResult FailureForReason(
    const std::string_view reason_code) {
  using lunar::pure_planning::PlanningStatus;
  if (reason_code == "REQUEST_CANCELED") {
    return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (reason_code == "TIMEOUT") {
    return Failure(PlanningStatus::kTimedOut, "TIMEOUT");
  }
  if (reason_code == "NO_PATH") {
    return Failure(PlanningStatus::kNoPath, "NO_PATH");
  }
  if (reason_code == "PLANNER_ERROR") {
    return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
  }
  return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
}

[[nodiscard]] std::uint8_t FeedbackPhase(
    const lunar::pure_planning::PlannerPhase phase) noexcept {
  using lunar::pure_planning::PlannerPhase;
  switch (phase) {
    case PlannerPhase::kSnapshotProjection:
      return Action::Feedback::BUILDING_SNAPSHOT;
    case PlannerPhase::kGlobal:
    case PlannerPhase::kLocalGoal:
    case PlannerPhase::kLocalSearch:
      return Action::Feedback::SEARCHING;
    case PlannerPhase::kCertification:
    case PlannerPhase::kOutput:
      return Action::Feedback::CERTIFYING;
  }
  return Action::Feedback::VALIDATING_INPUT;
}

[[nodiscard]] bool SameGoal(const std::shared_ptr<GoalHandle>& left,
                            const std::shared_ptr<GoalHandle>& right) noexcept {
  return left && right && left.get() == right.get();
}

void SetDiagnosticValue(diagnostic_msgs::msg::DiagnosticArray& diagnostics,
                        const std::string_view key, std::string value) {
  if (diagnostics.status.size() != 1U) {
    return;
  }
  for (auto& field : diagnostics.status.front().values) {
    if (field.key == key) {
      field.value = std::move(value);
      return;
    }
  }
}

[[nodiscard]] std::vector<lunar::pure_planning::Pose3> ResultPathPoses(
    const lunar::pure_planning::PlanningResult& result) {
  if (!result.reference.has_value()) {
    return {};
  }
  if (!result.reference->preview.poses_map.empty()) {
    return result.reference->preview.poses_map;
  }
  const auto* trajectory =
      std::get_if<lunar::pure_planning::TrajectoryReference>(
          &result.reference->data);
  if (trajectory == nullptr) {
    return {};
  }
  std::vector<lunar::pure_planning::Pose3> poses;
  poses.reserve(trajectory->points.size());
  for (const auto& point : trajectory->points) {
    poses.push_back(point.pose);
  }
  return poses;
}

[[nodiscard]] builtin_interfaces::msg::Duration RosDuration(
    const std::chrono::nanoseconds duration) noexcept {
  constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
  constexpr std::int64_t kMaxSeconds = std::numeric_limits<std::int32_t>::max();
  const std::int64_t nanoseconds = std::max<std::int64_t>(duration.count(), 0LL);
  builtin_interfaces::msg::Duration converted;
  converted.sec = static_cast<std::int32_t>(
      std::min(nanoseconds / kNanosecondsPerSecond, kMaxSeconds));
  converted.nanosec = static_cast<std::uint32_t>(nanoseconds % kNanosecondsPerSecond);
  return converted;
}

void SetFinalizedTiming(
    const std::chrono::nanoseconds elapsed,
    lunar::pure_planning::PlanningResult& result,
    Action::Result& action_result) {
  result.timing.total_elapsed = elapsed;
  action_result.diagnostics.elapsed_s =
      std::chrono::duration<double>(elapsed).count();
  const auto latency_class =
      lunar::pure_planning::ClassifyRequestLatency(elapsed);
  action_result.diagnostics.warning_codes.clear();
  if (latency_class !=
      lunar::pure_planning::RequestLatencyClass::kTargetMet) {
    action_result.diagnostics.warning_codes.emplace_back("TARGET_MISSED");
  }
  if (latency_class == lunar::pure_planning::RequestLatencyClass::kSlaMissed ||
      latency_class ==
          lunar::pure_planning::RequestLatencyClass::kHardTimeout) {
    action_result.diagnostics.warning_codes.emplace_back(
        "PLANNING_SLA_MISSED");
  }
}

void SetFinalizedTiming(
    const std::chrono::nanoseconds elapsed,
    lunar::pure_planning::PlanningResult& result,
    Action::Result& action_result,
    diagnostic_msgs::msg::DiagnosticArray& diagnostics) {
  SetFinalizedTiming(elapsed, result, action_result);
  const auto latency_class =
      lunar::pure_planning::ClassifyRequestLatency(elapsed);
  std::ostringstream milliseconds;
  milliseconds << std::setprecision(15)
               << std::chrono::duration<double, std::milli>(elapsed).count();
  SetDiagnosticValue(diagnostics, "total_elapsed_ms", milliseconds.str());
  SetDiagnosticValue(
      diagnostics, "latency_class",
      std::string{lunar::pure_planning::RequestLatencyClassName(latency_class)});
}

void PopulateGridV1Diagnostics(
    lunar::pure_planning::PlanningResult& result,
    const InputSnapshot& input,
    const std::optional<TraversabilityInputSnapshot>& traversability,
    const bool publish_check_enabled,
    const std::optional<std::uint64_t> publish_check_revision = std::nullopt) {
  auto& diagnostics = result.grid_v1;
  diagnostics.active = true;
  diagnostics.global_input_sequence = input.global_sequence;
  diagnostics.local_input_sequence = input.local_sequence;
  diagnostics.odometry_input_sequence = input.odometry_sequence;
  if (!traversability.has_value() || !traversability->snapshot ||
      !traversability->snapshot->valid()) {
    return;
  }
  const auto& map = *traversability->snapshot;
  const auto metrics = map.metrics();
  diagnostics.traversability_revision = map.revision();
  diagnostics.publish_check_revision = publish_check_enabled
      ? publish_check_revision.value_or(map.revision())
      : 0U;
  diagnostics.profile_hash = map.profile_hash();
  diagnostics.canonical_resolution_m = map.resolution_m();
  diagnostics.map_origin_m = map.origin_m();
  diagnostics.allocated_tiles = metrics.allocated_tiles;
  diagnostics.estimated_map_bytes = metrics.estimated_bytes;
  diagnostics.updated_cells = metrics.updated_cells;
  diagnostics.dirty_tiles = metrics.dirty_tiles;
  diagnostics.halo_recomputed_cells = metrics.halo_recomputed_cells;
  diagnostics.free_cells = metrics.free_cells;
  diagnostics.blocked_cells = metrics.blocked_cells;
  diagnostics.unknown_cells = metrics.unknown_cells;
  diagnostics.prior_conflicts = metrics.prior_conflicts;
}

struct AppliedTrustedBridge final {
  lunar::pure_planning::Pose3 start;
  lunar::pure_planning::Pose3 endpoint;
};

[[nodiscard]] std::optional<AppliedTrustedBridge> ApplyTrustedBridge(
    lunar::pure_planning::PlanningRequest& request,
    const InputSnapshot& snapshot) {
  if (!snapshot.local_map || !snapshot.odometry ||
      snapshot.local_map->header.frame_id != snapshot.odometry->header.frame_id ||
      request.environment_mode != lunar::pure_planning::EnvironmentMode::kLavaTube) {
    return std::nullopt;
  }
  auto* state = std::get_if<lunar::pure_planning::WheeledState>(
      &request.current_state);
  if (state == nullptr) {
    return std::nullopt;
  }
  IncrementalTraversability classifier{MakeTraversabilityProfile(
      request.capability, request.config.local_occupancy_threshold)};
  const auto traversability = classifier.Update(*snapshot.local_map);
  if (!traversability.has_value()) {
    return std::nullopt;
  }
  const auto bridge = FindTrustedBridge(traversability->map,
                                        state->pose.position_m);
  if (!bridge.has_value() || bridge->frame_id != snapshot.odometry->header.frame_id) {
    return std::nullopt;
  }
  const lunar::pure_planning::Pose3 start = state->pose;
  state->pose.position_m = bridge->endpoint_m;
  state->pose.position_m.z = start.position_m.z;
  return AppliedTrustedBridge{.start = start, .endpoint = state->pose};
}

[[nodiscard]] bool PrependTrustedBridge(
    lunar::pure_planning::PlanningResult& result,
    const AppliedTrustedBridge& bridge) {
  if (!result.reference.has_value()) {
    return false;
  }
  auto* trajectory = std::get_if<lunar::pure_planning::TrajectoryReference>(
      &result.reference->data);
  if (trajectory == nullptr) {
    return false;
  }
  constexpr auto kBridgeDuration = 1s;
  for (auto& point : trajectory->points) {
    point.time_from_start += kBridgeDuration;
  }
  trajectory->points.insert(trajectory->points.begin(), {
      {.time_from_start = 0ns, .pose = bridge.start},
      {.time_from_start = kBridgeDuration, .pose = bridge.endpoint},
  });
  result.reference->preview.poses_map.insert(
      result.reference->preview.poses_map.begin(),
      {bridge.start, bridge.endpoint});
  result.reason_code = "TRUSTED_BRIDGE_USED";
  return true;
}

}  // namespace

PlannerFn RealPlannerFn() {
  auto planner = std::make_shared<lunar::pure_planning::Planner>();
  return [planner = std::move(planner)](
             const lunar::pure_planning::PlanningRequest& request) {
    return planner->Plan(request);
  };
}

LocalPlannerFn RealLocalPlannerFn() {
  auto planner = std::make_shared<lunar::pure_planning::Planner>();
  return [planner = std::move(planner)](
             const lunar::pure_planning::PlanningRequest& request,
             const lunar::pure_planning::LocalGoalSet& goals,
             lunar::pure_planning::SearchControl control) {
    return planner->PlanLocal(request, goals, std::move(control));
  };
}

std::unique_ptr<rclcpp::Executor> MakePurePlannerExecutor() {
  return std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
      rclcpp::ExecutorOptions{}, 2U);
}

struct PurePlanMotionServer::Impl final {
  Impl(PurePlanMotionServer& owner, PlannerFn planner,
       LocalPlannerFn local_planner)
      : node(owner), parameters(ReadRuntimeParameters(owner)),
        planner(std::move(planner)), local_planner(std::move(local_planner)) {
    if (!this->planner) {
      this->planner = RealPlannerFn();
    }
    if (!this->local_planner) {
      this->local_planner = RealLocalPlannerFn();
    }
    const bool use_wheel_grid_v1 =
        parameters.platform_type ==
            lunar::pure_planning::PlatformType::kWheeled &&
        parameters.planner_config.wheel_planner_mode ==
            lunar::pure_planning::WheelPlannerMode::kGridTraversabilityV1;
    const bool use_legged_grid_v1 =
        parameters.platform_type ==
            lunar::pure_planning::PlatformType::kLegged &&
        parameters.planner_config.legged_global_mode ==
            lunar::pure_planning::LeggedGlobalMode::kGridTraversabilityV1;
    if (use_wheel_grid_v1) {
      const auto* capability =
          std::get_if<lunar::pure_planning::WheeledCapability>(
              &parameters.capability);
      if (capability == nullptr || capability->footprint_xy_m.empty()) {
        throw std::runtime_error{
            "PLANNER_ERROR: wheel traversability profile invalid"};
      }
      double footprint_radius_m = 0.0;
      for (const auto& vertex : capability->footprint_xy_m) {
        footprint_radius_m =
            std::max(footprint_radius_m, std::hypot(vertex.x, vertex.y));
      }
      traversability_input = std::make_unique<TraversabilityInput>(
          lunar::pure_planning::TraversabilityProfile{
              .global_occupancy_threshold =
                  parameters.planner_config.global_occupancy_threshold,
              .local_occupancy_threshold =
                  parameters.planner_config.local_occupancy_threshold,
              .maximum_slope_rad = capability->maximum_slope_rad,
              .inflation_radius_m =
                  footprint_radius_m + capability->minimum_clearance_m,
          });
    } else if (use_legged_grid_v1) {
      const auto* capability =
          std::get_if<lunar::pure_planning::LeggedCapability>(
              &parameters.capability);
      if (capability == nullptr) {
        throw std::runtime_error{
            "PLANNER_ERROR: legged traversability profile invalid"};
      }
      traversability_input = std::make_unique<TraversabilityInput>(
          lunar::pure_planning::TraversabilityProfile{
              .global_occupancy_threshold =
                  parameters.planner_config.global_occupancy_threshold,
              .local_occupancy_threshold =
                  parameters.planner_config.local_occupancy_threshold,
              .maximum_slope_rad = capability->maximum_slope_rad,
              .inflation_radius_m =
                  std::hypot(capability->body_extent_m.x,
                             capability->body_extent_m.y) /
                      2.0 +
                  capability->minimum_body_clearance_m,
          });
    }
    node.declare_parameter<bool>("trusted_bridge_once", false);
    action_group =
        node.create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    input_group =
        node.create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions input_options;
    input_options.callback_group = input_group;
    const auto input_qos = rclcpp::QoS{rclcpp::KeepLast{10}}.best_effort();
    global_subscription = node.create_subscription<nav_msgs::msg::OccupancyGrid>(
        parameters.global_map_topic, input_qos,
        [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
          if (traversability_input) {
            traversability_input->UpdateGlobal(message);
          }
          input_store.UpdateGlobal(std::move(message));
        }, input_options);
    local_subscription = node.create_subscription<grid_map_msgs::msg::GridMap>(
        parameters.local_map_topic, input_qos,
        [this](grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
          if (traversability_input) {
            traversability_input->UpdateLocal(message);
          }
          input_store.UpdateLocal(std::move(message));
        }, input_options);
    odometry_subscription = node.create_subscription<nav_msgs::msg::Odometry>(
        parameters.odometry_topic, input_qos,
        [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
          input_store.UpdateOdometry(std::move(message));
        }, input_options);
    tf_subscription = node.create_subscription<tf2_msgs::msg::TFMessage>(
        parameters.tf_topic, input_qos,
        [this](tf2_msgs::msg::TFMessage::ConstSharedPtr message) {
          if (traversability_input) {
            traversability_input->UpdateTf(*message);
          }
          input_store.UpdateTf(*message);
        }, input_options);
    diagnostics_publisher =
        node.create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            parameters.diagnostics_topic, rclcpp::QoS{10}.reliable());
    wheeled_reference_publisher =
        node.create_publisher<lunar_planning_msgs::msg::MotionReference>(
            parameters.wheeled_reference_topic, rclcpp::QoS{1}.reliable());
    wheeled_path_publisher = node.create_publisher<nav_msgs::msg::Path>(
        parameters.wheeled_path_topic, rclcpp::QoS{1}.reliable());
    wheeled_global_path_publisher = node.create_publisher<nav_msgs::msg::Path>(
        parameters.wheeled_global_path_topic, rclcpp::QoS{1}.reliable());
    timed_path_publisher = node.create_publisher<lunar_planning_msgs::msg::TimedPath>(
        parameters.wheeled_timed_path_topic, rclcpp::QoS{1}.reliable());
    if (parameters.rolling_surface.enabled &&
        parameters.platform_type ==
            lunar::pure_planning::PlatformType::kLegged) {
      legged_path_publisher = node.create_publisher<nav_msgs::msg::Path>(
          parameters.legged_path_topic, rclcpp::QoS{1}.reliable());
      legged_global_path_publisher = node.create_publisher<nav_msgs::msg::Path>(
          parameters.legged_global_path_topic, rclcpp::QoS{1}.reliable());
    }
    action_server = rclcpp_action::create_server<Action>(
        node.get_node_base_interface(), node.get_node_clock_interface(),
        node.get_node_logging_interface(), node.get_node_waitables_interface(),
        parameters.action_name,
        [this](const rclcpp_action::GoalUUID&,
               const std::shared_ptr<const Action::Goal> goal) {
          return HandleGoal(goal);
        },
        [this](const std::shared_ptr<GoalHandle> goal) {
          return HandleCancel(goal);
        },
        [this](const std::shared_ptr<GoalHandle> goal) { HandleAccepted(goal); },
        rcl_action_server_get_default_options(), action_group);
  }

  ~Impl() {
    teardown_requested.store(true, std::memory_order_release);
    cancel_transition_waiter.Notify();
    StopAndJoinWorker();
  }

  struct FeedbackState final {
    std::optional<std::uint8_t> phase;
    std::chrono::steady_clock::time_point last_publish{};
    std::chrono::steady_clock::time_point cycle_started{};
    std::uint64_t expanded_states{};
    std::optional<double> best_cost;
  };

  void BeginFeedbackCycle(const std::shared_ptr<GoalHandle>& goal_handle,
                          const std::stop_token stop_token,
                          FeedbackState& state,
                          const std::chrono::steady_clock::time_point started) {
    state = FeedbackState{.cycle_started = started};
    PublishFeedback(goal_handle, stop_token, state,
                    Action::Feedback::VALIDATING_INPUT);
  }

  void PublishFeedback(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::stop_token stop_token, FeedbackState& state,
      const std::uint8_t phase, const std::uint64_t expanded_states = 0U,
      const std::optional<double> best_cost = std::nullopt) {
    if (!goal_handle || stop_token.stop_requested()) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    const bool phase_changed = !state.phase.has_value() || *state.phase != phase;
    if (!phase_changed && state.last_publish != std::chrono::steady_clock::time_point{} &&
        now - state.last_publish < 100ms) {
      return;
    }
    {
      std::scoped_lock lock{state_mutex};
      if (!SameGoal(active_goal, goal_handle)) {
        return;
      }
    }
    state.phase = phase;
    state.last_publish = now;
    state.expanded_states = std::max(state.expanded_states, expanded_states);
    if (best_cost.has_value() && std::isfinite(*best_cost)) {
      state.best_cost = best_cost;
    }
    auto feedback = std::make_shared<Action::Feedback>();
    feedback->phase = phase;
    feedback->elapsed_s = std::chrono::duration<double>(
                              now - state.cycle_started)
                              .count();
    feedback->expanded_states = state.expanded_states;
    feedback->has_best_cost = state.best_cost.has_value();
    feedback->best_cost = state.best_cost.value_or(0.0);
    try {
      goal_handle->publish_feedback(feedback);
    } catch (...) {
      SafeLogError("PLANNER_ERROR: failed to publish planner feedback");
    }
  }

  void PublishCoreProgress(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::stop_token stop_token, FeedbackState& state,
      const lunar::pure_planning::PlannerProgress& progress) {
    PublishFeedback(goal_handle, stop_token, state,
                    FeedbackPhase(progress.phase));
  }

  [[nodiscard]] static bool SameRollingPlanningIdentity(
      const InputSnapshot& planned, const InputSnapshot& latest,
      const bool legged_rolling) noexcept {
    return planned.global_sequence == latest.global_sequence &&
           planned.tf_sequence == latest.tf_sequence &&
           (legged_rolling ||
            planned.local_sequence == latest.local_sequence);
  }

  void PublishCycleDiagnostics(
      const std::shared_ptr<const Action::Goal>& request_goal,
      const lunar::pure_planning::PlanningResult& result,
      const bool stale_input = false) {
    auto diagnostics = MakeRequestDiagnostics(
        request_goal ? std::string_view{request_goal->request_id}
                     : std::string_view{},
        parameters.platform_type,
        request_goal ? EnvironmentMode(request_goal->environment_mode)
                     : lunar::pure_planning::EnvironmentMode::kLunarSurface,
        result);
    if (stale_input) {
      if (diagnostics.status.size() == 1U) {
        diagnostics.status.front().level =
            diagnostic_msgs::msg::DiagnosticStatus::WARN;
      }
      SetDiagnosticValue(diagnostics, "planning_outcome",
                         std::to_string(Action::Result::STALE_INPUT));
      SetDiagnosticValue(diagnostics, "reason_code", "STALE_INPUT");
    }
    if (!ContextIsValid()) {
      return;
    }
    try {
      diagnostics.header.stamp = node.now();
      diagnostics_publisher->publish(diagnostics);
      LogDiagnostics(diagnostics);
    } catch (...) {
      SafeLogError("PLANNER_ERROR: failed to publish rolling diagnostics");
    }
  }

  [[nodiscard]] rclcpp_action::GoalResponse HandleGoal(
      const std::shared_ptr<const Action::Goal> goal) {
    std::scoped_lock lock{state_mutex};
    return execution_state.MayAccept(goal != nullptr,
                                     goal && goal->replace_active_request)
               ? rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE
               : rclcpp_action::GoalResponse::REJECT;
  }

  [[nodiscard]] rclcpp_action::CancelResponse HandleCancel(
      const std::shared_ptr<GoalHandle> goal) {
    std::unique_lock lock{state_mutex};
    if (!SameGoal(active_goal, goal) ||
        !execution_state.AcceptCancel(goal.get(), worker != nullptr)) {
      return rclcpp_action::CancelResponse::REJECT;
    }
    worker->request_stop();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  struct WorkerStateGuard final {
    Impl& impl;
    std::uint64_t generation;
    std::shared_ptr<GoalHandle> goal;

    ~WorkerStateGuard() {
      try {
        impl.FinishWorker(generation, goal);
      } catch (...) {
        impl.SafeLogError("PLANNER_ERROR: failed to clear active request");
      }
    }
  };

  void HandleAccepted(const std::shared_ptr<GoalHandle> goal) {
    if (!goal) {
      return;
    }
    {
      std::scoped_lock lock{state_mutex};
      if (execution_state.RequestReplacement()) {
        if (worker) {
          worker->request_stop();
        }
      }
    }
    StopAndJoinWorker();
    std::optional<std::uint64_t> generation;
    {
      std::scoped_lock lock{state_mutex};
      generation = execution_state.TryStart(goal.get());
      if (generation.has_value()) {
        active_goal = goal;
      }
    }
    if (!generation.has_value()) {
      FinishUnownedAcceptedGoal(goal);
      return;
    }
    const bool trusted_bridge_once = ConsumeTrustedBridgeOnce();
    try {
      auto started_worker = std::make_unique<std::jthread>(
          [this, goal, generation = *generation, trusted_bridge_once](
              const std::stop_token stop_token) {
            ExecuteGoal(goal, generation, trusted_bridge_once, stop_token);
          });
      std::scoped_lock lock{state_mutex};
      worker = std::move(started_worker);
    } catch (...) {
      const auto started = std::chrono::steady_clock::now();
      WorkerStateGuard cleanup{*this, *generation, goal};
      try {
        ExecuteKnownResult(
            goal, *generation, started, input_store.Capture(),
            Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                    "PLANNER_ERROR"));
      } catch (...) {
        SafeLogError("PLANNER_ERROR: worker creation fallback failed");
      }
    }
  }

  void ExecuteGoal(const std::shared_ptr<GoalHandle>& goal_handle,
                   const std::uint64_t generation, const bool trusted_bridge_once,
                   const std::stop_token stop_token) noexcept {
    WorkerStateGuard cleanup{*this, generation, goal_handle};
    const auto started = std::chrono::steady_clock::now();
    try {
      ExecuteGoalBody(goal_handle, generation, trusted_bridge_once, stop_token,
                      started);
    } catch (...) {
      SafeLogError("PLANNER_ERROR: unexpected request execution failure");
      try {
        ExecuteKnownResult(
            goal_handle, generation, started, InputSnapshot{},
            Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                    "PLANNER_ERROR"));
      } catch (...) {
        SafeLogError("PLANNER_ERROR: emergency finalization failed");
      }
    }
  }

  void ExecuteGoalBody(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::uint64_t generation,
      const bool trusted_bridge_once,
      const std::stop_token stop_token,
      const std::chrono::steady_clock::time_point started) {
    FeedbackState feedback_state;
    if (UseRollingSurface(goal_handle->get_goal())) {
      InputSnapshot final_snapshot;
      bool result_diagnostic_published = false;
      auto result = ExecuteRollingSurface(
          goal_handle, goal_handle->get_goal(), stop_token, &feedback_state,
          &final_snapshot, &result_diagnostic_published);
      try {
        ExecuteKnownResult(goal_handle, generation, started, final_snapshot,
                           std::move(result), result_diagnostic_published);
      } catch (...) {
        SafeLogError("PLANNER_ERROR: rolling finalization failed");
      }
      return;
    }
    BeginFeedbackCycle(goal_handle, stop_token, feedback_state, started);
    const auto timing_policy =
        lunar::pure_planning::MakeRequestTimingPolicy(started);
    const InputSnapshot snapshot = input_store.Capture();
    const auto request_goal = goal_handle->get_goal();
    const bool use_wheel_grid_v1 =
        parameters.platform_type ==
            lunar::pure_planning::PlatformType::kWheeled &&
        parameters.planner_config.wheel_planner_mode ==
            lunar::pure_planning::WheelPlannerMode::kGridTraversabilityV1;
    const bool use_legged_grid_v1 =
        request_goal &&
        EnvironmentMode(request_goal->environment_mode) ==
            lunar::pure_planning::EnvironmentMode::kLunarSurface &&
        parameters.platform_type ==
            lunar::pure_planning::PlatformType::kLegged &&
        parameters.planner_config.legged_global_mode ==
            lunar::pure_planning::LeggedGlobalMode::kGridTraversabilityV1;
    const bool uses_traversability_snapshot =
        use_wheel_grid_v1 || use_legged_grid_v1;
    std::optional<TraversabilityInputSnapshot> traversability;
    lunar::pure_planning::PlanningResult result;
    try {
      if (uses_traversability_snapshot && traversability_input) {
        traversability = traversability_input->Capture();
      }
      if (stop_token.stop_requested()) {
        result = Failure(lunar::pure_planning::PlanningStatus::kCanceled,
                         "REQUEST_CANCELED");
      } else if (!request_goal || !snapshot.map_from_odom.has_value() ||
                 !snapshot.odometry) {
        result = Failure(lunar::pure_planning::PlanningStatus::kInvalidInput,
                         "INVALID_INPUT");
      } else {
        const auto map_from_odom =
            AdaptDirectMapFromOdom(*snapshot.map_from_odom);
        const auto converted_goal = map_from_odom.value.has_value()
            ? ConvertGoal(*request_goal, *map_from_odom.value)
            : GoalConversionResult{};
        const bool require_surface_global_map = !use_wheel_grid_v1;
        auto world = AdaptSnapshot(EnvironmentMode(request_goal->environment_mode),
                                   snapshot, require_surface_global_map);
        const auto state = AdaptOdometry(*snapshot.odometry,
                                         parameters.platform_type);
        if (!map_from_odom.value.has_value() || !converted_goal.ok() ||
            !world.value.has_value() || !state.value.has_value()) {
          result = Failure(lunar::pure_planning::PlanningStatus::kInvalidInput,
                           "INVALID_INPUT");
        } else if (uses_traversability_snapshot &&
                   (!traversability.has_value() ||
                    !traversability->snapshot ||
                    !traversability->snapshot->valid() ||
                    !traversability->reason_code.empty())) {
          result = Failure(
              lunar::pure_planning::PlanningStatus::kInvalidInput,
              traversability.has_value() &&
                      !traversability->reason_code.empty()
                  ? traversability->reason_code
                  : "INVALID_INPUT");
        } else {
          if (uses_traversability_snapshot) {
            world.value->traversability_snapshot = traversability->snapshot;
          }
          lunar::pure_planning::PlanningRequest request{
              .request_id = request_goal->request_id,
              .environment_mode =
                  EnvironmentMode(request_goal->environment_mode),
              .current_state = std::move(*state.value),
              .goal_map = std::move(*converted_goal.goal),
              .world = std::move(*world.value),
              .capability = parameters.capability,
              .config = parameters.planner_config,
              .control = {
                  .deadline = timing_policy.hard_deadline,
                  .stop_token = stop_token,
                  .now = [] { return std::chrono::steady_clock::now(); },
              },
              .request_started_at = started,
              .progress = [this, goal_handle, stop_token, &feedback_state](
                              const lunar::pure_planning::PlannerProgress&
                                  progress) {
                PublishCoreProgress(goal_handle, stop_token, feedback_state,
                                    progress);
              },
          };
          const auto bridge =
              trusted_bridge_once && !uses_traversability_snapshot
                  ? ApplyTrustedBridge(request, snapshot)
                  : std::optional<AppliedTrustedBridge>{};
          result = planner(request);
          PublishFeedback(goal_handle, stop_token, feedback_state,
                          Action::Feedback::CERTIFYING,
                          result.expanded_states, result.best_cost);
          if (bridge.has_value() &&
              result.status == lunar::pure_planning::PlanningStatus::kSuccess &&
              !PrependTrustedBridge(result, *bridge)) {
            result = Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                             "PLANNER_ERROR");
          }
        }
      }
    } catch (...) {
      result = Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                       "PLANNER_ERROR");
    }

    std::optional<std::uint64_t> publish_check_revision;
    if (use_wheel_grid_v1 &&
        result.status == lunar::pure_planning::PlanningStatus::kSuccess &&
        result.reference.has_value()) {
      const auto latest = traversability_input
                              ? traversability_input->Capture()
                              : TraversabilityInputSnapshot{};
      const auto poses = ResultPathPoses(result);
      std::size_t checked_cells{};
      if (!latest.reason_code.empty() || !latest.snapshot ||
          !latest.snapshot->valid()) {
        const auto timing = result.timing;
        result = Failure(
            lunar::pure_planning::PlanningStatus::kInvalidInput,
            latest.reason_code.empty() ? "INVALID_INPUT"
                                       : latest.reason_code);
        result.timing = timing;
      } else if (poses.empty() ||
                 !lunar::pure_planning::grid_v1::PathIsFree(
                     *latest.snapshot, poses, &checked_cells)) {
        const auto timing = result.timing;
        result = Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                         "STALE_PATH_INVALIDATED");
        result.timing = timing;
        publish_check_revision = latest.snapshot->revision();
      } else {
        publish_check_revision = latest.snapshot->revision();
        result.grid_v1.final_supercover_cells = checked_cells;
      }
    }

    if (uses_traversability_snapshot) {
      PopulateGridV1Diagnostics(result, snapshot, traversability,
                                use_wheel_grid_v1,
                                publish_check_revision);
    }

    if (result.status == lunar::pure_planning::PlanningStatus::kSuccess &&
        !result.reference.has_value()) {
      const auto timing = result.timing;
      result = Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                       "PLANNER_ERROR");
      result.timing = timing;
    }
    if (stop_token.stop_requested()) {
      const auto timing = result.timing;
      result = Failure(lunar::pure_planning::PlanningStatus::kCanceled,
                       "REQUEST_CANCELED");
      result.timing = timing;
    }

    try {
      ExecuteKnownResult(goal_handle, generation, started, snapshot,
                         std::move(result));
    } catch (...) {
      SafeLogError("PLANNER_ERROR: finalization failed; returning planner error");
      try {
        ExecuteKnownResult(
            goal_handle, generation, started, snapshot,
            Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                    "PLANNER_ERROR"));
      } catch (...) {
        SafeLogError("PLANNER_ERROR: fallback finalization failed");
      }
    }
  }

  [[nodiscard]] bool UseRollingSurface(
      const std::shared_ptr<const Action::Goal>& goal) const noexcept {
    if (!goal || !parameters.rolling_surface.enabled ||
        EnvironmentMode(goal->environment_mode) !=
            lunar::pure_planning::EnvironmentMode::kLunarSurface) {
      return false;
    }
    const bool wheel_legacy =
        parameters.platform_type ==
            lunar::pure_planning::PlatformType::kWheeled &&
        parameters.planner_config.wheel_planner_mode ==
            lunar::pure_planning::WheelPlannerMode::kLegacyCertified;
    const bool legged_grid_v1 =
        parameters.platform_type ==
            lunar::pure_planning::PlatformType::kLegged &&
        parameters.planner_config.legged_global_mode ==
            lunar::pure_planning::LeggedGlobalMode::kGridTraversabilityV1;
    return wheel_legacy || legged_grid_v1;
  }

  [[nodiscard]] lunar::pure_planning::PlanningResult ExecuteRollingSurface(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::shared_ptr<const Action::Goal>& request_goal,
      const std::stop_token stop_token, FeedbackState* const feedback_state,
      InputSnapshot* const final_snapshot,
      bool* const result_diagnostic_published) {
    using lunar::pure_planning::EnvironmentMode;
    using lunar::pure_planning::LocalPlanStatus;
    using lunar::pure_planning::PlanningRequest;
    using lunar::pure_planning::PlanningResult;
    using lunar::pure_planning::PlanningStatus;
    using lunar::pure_planning::Pose3;
    using lunar::pure_planning::SearchControl;
    using RollingDecision =
        lunar::pure_planning::hierarchical::SurfaceRollingDecision;

    const bool legged_rolling =
        parameters.platform_type == lunar::pure_planning::PlatformType::kLegged;

    if (result_diagnostic_published != nullptr) {
      *result_diagnostic_published = false;
    }
    std::optional<lunar::pure_planning::GlobalRoute> route;
    std::optional<lunar::pure_planning::hierarchical::SurfaceRollingSession>
        session;
    std::optional<lunar::pure_planning::GoalRegion> active_goal;
    std::optional<PlanningResult> last_segment;
    std::uint64_t route_global_sequence{};
    std::uint64_t route_tf_sequence{};
    std::uint64_t seen_local_sequence{};
    double minimum_route_progress_m{};
    std::optional<double> legged_replan_anchor_progress_m;
    bool force_replan = true;
    auto last_replan = std::chrono::steady_clock::now() -
        std::chrono::milliseconds{
            parameters.rolling_surface.min_replan_interval_ms};

    while (!stop_token.stop_requested()) {
      InputSnapshot snapshot = input_store.Capture();
      if (final_snapshot != nullptr) {
        *final_snapshot = snapshot;
      }
      if (!request_goal || !snapshot.map_from_odom.has_value() ||
          !snapshot.odometry || !snapshot.local_map || !snapshot.global_map) {
        return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
      }
      const bool route_identity_changed =
          !route.has_value() ||
          route_global_sequence != snapshot.global_sequence ||
          route_tf_sequence != snapshot.tf_sequence;

      const auto transform = AdaptDirectMapFromOdom(*snapshot.map_from_odom);
      const auto converted_goal = transform.value.has_value()
          ? ConvertGoal(*request_goal, *transform.value)
          : GoalConversionResult{};
      auto world = AdaptSnapshot(EnvironmentMode::kLunarSurface, snapshot);
      const auto state = AdaptOdometry(*snapshot.odometry,
                                       parameters.platform_type);
      if (!transform.value.has_value() || !converted_goal.ok() ||
          !world.value.has_value() || !state.value.has_value()) {
        return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
      }
      std::optional<TraversabilityInputSnapshot> traversability;
      if (legged_rolling) {
        traversability = traversability_input
            ? std::optional<TraversabilityInputSnapshot>{
                  traversability_input->Capture()}
            : std::nullopt;
        if (!traversability.has_value() || !traversability->snapshot ||
            !traversability->snapshot->valid() ||
            !traversability->reason_code.empty()) {
          return Failure(
              PlanningStatus::kInvalidInput,
              traversability.has_value() &&
                      !traversability->reason_code.empty()
                  ? traversability->reason_code
                  : "INVALID_INPUT");
        }
        world.value->traversability_snapshot = traversability->snapshot;
      }
      const auto* pose_odom = PlatformPose(*state.value);
      const auto pose_map = pose_odom == nullptr
          ? std::optional<Pose3>{}
          : lunar::pure_planning::hierarchical::TransformPose(
                *pose_odom, *transform.value,
                lunar::pure_planning::hierarchical::TransformDirection::
                    kChildToParent);
      if (pose_odom == nullptr || !pose_map.has_value()) {
        return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
      }
      const auto* final_point =
          std::get_if<lunar::pure_planning::PointGoal>(
              &converted_goal.goal->target);
      if (last_segment.has_value() && final_point != nullptr &&
          std::hypot(pose_map->position_m.x - final_point->position_m.x,
                     pose_map->position_m.y - final_point->position_m.y) <=
              final_point->tolerance_m) {
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        return std::move(*last_segment);
      }

      std::optional<lunar::pure_planning::RequestTimingPolicy> cycle_timing;
      lunar::pure_planning::PlannerCallTiming cycle_call_timing;
      if (route_identity_changed) {
        const auto cycle_started = std::chrono::steady_clock::now();
        cycle_timing =
            lunar::pure_planning::MakeRequestTimingPolicy(cycle_started);
        if (feedback_state != nullptr) {
          BeginFeedbackCycle(goal_handle, stop_token, *feedback_state,
                             cycle_started);
          PublishFeedback(goal_handle, stop_token, *feedback_state,
                          Action::Feedback::BUILDING_SNAPSHOT);
          PublishFeedback(goal_handle, stop_token, *feedback_state,
                          Action::Feedback::SEARCHING);
        }
        PlanningRequest global_request{
            .request_id = request_goal->request_id,
            .environment_mode = EnvironmentMode::kLunarSurface,
            .current_state = *state.value,
            .goal_map = *converted_goal.goal,
            .world = *world.value,
            .capability = parameters.capability,
            .config = parameters.planner_config,
            .control = {.deadline = cycle_timing->hard_deadline,
                        .stop_token = stop_token,
                        .now = [] { return std::chrono::steady_clock::now(); }},
            .request_started_at = cycle_started,
        };
        global_request.config.search.stop_after_first_solution = true;
        const auto global_started = std::chrono::steady_clock::now();
        auto global = lunar::pure_planning::hierarchical::PlanSurfaceGlobal(
            global_request, global_request.control);
        const auto global_finished = std::chrono::steady_clock::now();
        cycle_call_timing.global_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                global_finished - global_started);
        cycle_call_timing.global_call_count = 1U;
        if (stop_token.stop_requested() ||
            global.reason_code == "REQUEST_CANCELED") {
          auto canceled =
              Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED");
          canceled.timing = cycle_call_timing;
          canceled.timing.total_elapsed =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  global_finished - cycle_timing->started_at);
          PublishCycleDiagnostics(request_goal, canceled);
          if (result_diagnostic_published != nullptr) {
            *result_diagnostic_published = true;
          }
          return canceled;
        }
        if (global_finished >= cycle_timing->hard_deadline) {
          auto timeout = Failure(PlanningStatus::kTimedOut, "TIMEOUT");
          timeout.timing = cycle_call_timing;
          timeout.timing.total_elapsed =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  global_finished - cycle_timing->started_at);
          PublishCycleDiagnostics(request_goal, timeout);
          if (result_diagnostic_published != nullptr) {
            *result_diagnostic_published = true;
          }
          return timeout;
        }
        if (!global.route.has_value()) {
          PlanningResult failed = FailureForReason(
              global.reason_code.empty() ? "PLANNER_ERROR"
                                         : global.reason_code);
          failed.timing = cycle_call_timing;
          failed.timing.total_elapsed =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  global_finished - cycle_timing->started_at);
          PublishCycleDiagnostics(request_goal, failed);
          if (result_diagnostic_published != nullptr) {
            *result_diagnostic_published = true;
          }
          return failed;
        }
        route = std::move(global.route);
        route_global_sequence = snapshot.global_sequence;
        route_tf_sequence = snapshot.tf_sequence;
        minimum_route_progress_m = 0.0;
        legged_replan_anchor_progress_m.reset();
        active_goal.reset();
        session.emplace(
            *route, *converted_goal.goal,
            lunar::pure_planning::hierarchical::SurfaceRollingConfig{
                .horizon_m = legged_rolling
                    ? kLeggedRollingHorizonM
                    : parameters.rolling_surface.horizon_m,
                .max_deviation_m =
                    parameters.rolling_surface.max_deviation_m});
        force_replan = true;
      }

      if (!session.has_value() || !route.has_value()) {
        return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
      }
      const RollingDecision decision =
          session->Decide(*pose_map, minimum_route_progress_m);
      if (decision.kind == RollingDecision::Kind::kInvalidRoute) {
        return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
      }
      if (decision.kind == RollingDecision::Kind::kFinalGoalReached) {
        if (last_segment.has_value()) {
          if (result_diagnostic_published != nullptr) {
            *result_diagnostic_published = true;
          }
          return std::move(*last_segment);
        }
        return Failure(PlanningStatus::kNoPath, "NO_PATH");
      }
      if (legged_rolling) {
        minimum_route_progress_m = std::max(
            minimum_route_progress_m, decision.projected_route_progress_m);
      }

      const auto* active = active_goal.has_value()
          ? std::get_if<lunar::pure_planning::PointGoal>(&active_goal->target)
          : nullptr;
      const bool target_reached =
          active != nullptr &&
          std::hypot(pose_map->position_m.x - active->position_m.x,
                     pose_map->position_m.y - active->position_m.y) <=
              active->tolerance_m +
                  (legged_rolling ? kLeggedGoalToleranceEpsilonM : 0.0);
      const bool local_changed =
          !legged_rolling &&
          snapshot.local_sequence != seen_local_sequence &&
          std::chrono::steady_clock::now() - last_replan >=
              std::chrono::milliseconds{
                  parameters.rolling_surface.min_replan_interval_ms};
      const bool deviated =
          !legged_rolling &&
          decision.lateral_deviation_m >
              parameters.rolling_surface.max_deviation_m;
      const bool legged_stride_reached =
          legged_rolling && legged_replan_anchor_progress_m.has_value() &&
          decision.projected_route_progress_m +
                  kLeggedGoalToleranceEpsilonM >=
              *legged_replan_anchor_progress_m + kLeggedReplanStrideM;
      if (!force_replan && last_segment.has_value() && !target_reached &&
          !local_changed && !deviated && !legged_stride_reached) {
        std::this_thread::sleep_for(std::chrono::milliseconds{
            parameters.rolling_surface.poll_period_ms});
        continue;
      }

      if (!cycle_timing.has_value()) {
        const auto cycle_started = std::chrono::steady_clock::now();
        cycle_timing =
            lunar::pure_planning::MakeRequestTimingPolicy(cycle_started);
        if (feedback_state != nullptr) {
          BeginFeedbackCycle(goal_handle, stop_token, *feedback_state,
                             cycle_started);
          PublishFeedback(goal_handle, stop_token, *feedback_state,
                          Action::Feedback::BUILDING_SNAPSHOT);
        }
      }
      PlanningRequest local_request{
          .request_id = request_goal->request_id,
          .environment_mode = EnvironmentMode::kLunarSurface,
          .current_state = *state.value,
          .goal_map = *converted_goal.goal,
          .world = *world.value,
          .capability = parameters.capability,
          .config = parameters.planner_config,
          .control = {.deadline = cycle_timing->hard_deadline,
                      .stop_token = stop_token,
                      .now = [] { return std::chrono::steady_clock::now(); }},
          .request_started_at = cycle_timing->started_at,
      };
      local_request.config.search.stop_after_first_solution = true;

      const auto local_goal_started = std::chrono::steady_clock::now();
      const auto portals =
          lunar::pure_planning::hierarchical::BuildSurfacePortalSet(
              local_request, *route, decision, 32U, local_request.control);
      lunar::pure_planning::hierarchical::LocalGoalSetResult local_goals;
      if (portals.ok()) {
        local_goals = lunar::pure_planning::hierarchical::
            ConvertSurfacePortalsToLocalGoals(
                portals, decision, *pose_odom, *world.value->global_map,
                world.value->local_map);
      } else {
        local_goals.reason_code = portals.reason_code;
      }
      const auto local_goal_finished = std::chrono::steady_clock::now();
      cycle_call_timing.local_goal_elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              local_goal_finished - local_goal_started);
      if (!local_goals.ok()) {
        PlanningResult failed = FailureForReason(local_goals.reason_code);
        failed.timing = cycle_call_timing;
        failed.timing.total_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                local_goal_finished - cycle_timing->started_at);
        if (stop_token.stop_requested()) {
          failed = Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED");
          failed.timing = cycle_call_timing;
          failed.timing.total_elapsed =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  local_goal_finished - cycle_timing->started_at);
        } else if (local_goal_finished >= cycle_timing->hard_deadline) {
          failed = Failure(PlanningStatus::kTimedOut, "TIMEOUT");
          failed.timing = cycle_call_timing;
          failed.timing.total_elapsed =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  local_goal_finished - cycle_timing->started_at);
        }
        PublishCycleDiagnostics(request_goal, failed);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        return failed;
      }

      if (feedback_state != nullptr) {
        PublishFeedback(goal_handle, stop_token, *feedback_state,
                        Action::Feedback::SEARCHING);
      }
      const auto local_started = std::chrono::steady_clock::now();
      lunar::pure_planning::LocalStageResult local = local_planner(
          local_request, *local_goals.goals, local_request.control);
      const auto local_finished = std::chrono::steady_clock::now();
      cycle_call_timing.local_search_elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(local_finished -
                                                               local_started);
      cycle_call_timing.local_elapsed = cycle_call_timing.local_search_elapsed;
      cycle_call_timing.local_call_count = 1U;
      if (feedback_state != nullptr) {
        PublishFeedback(goal_handle, stop_token, *feedback_state,
                        Action::Feedback::CERTIFYING,
                        local.expanded_states, local.best_cost);
      }

      PlanningResult segment;
      if (local.status != LocalPlanStatus::kSolved) {
        segment = LocalFailure(local.status, std::move(local.reason_code));
      } else if (!local.data.has_value() ||
                 !local.selected_goal_index.has_value() ||
                 *local.selected_goal_index >=
                     local_goals.goals->goals_odom.size()) {
        segment = Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
      } else {
        const auto certification_started = std::chrono::steady_clock::now();
        auto composed =
            lunar::pure_planning::hierarchical::ComposeSurfaceReference(
                local_request, *route, std::move(*local.data),
                local_request.control);
        const auto certification_finished = std::chrono::steady_clock::now();
        cycle_call_timing.certification_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                certification_finished - certification_started);
        if (!composed.ok()) {
          segment = FailureForReason(composed.reason_code);
        } else {
          segment = PlanningResult{
              .status = PlanningStatus::kSuccess,
              .reason_code = "PLAN_FOUND",
              .reference = std::move(composed.reference),
              .global_route_preview = {.poses_map = route->poses_map},
              .expanded_states = local.expanded_states,
              .selected_goal_index = local.selected_goal_index,
              .best_cost = local.best_cost,
          };
        }
      }
      segment.expanded_states = local.expanded_states;
      segment.selected_goal_index = local.selected_goal_index;
      segment.best_cost = local.best_cost;
      segment.legged_local = local.legged_local;
      if (legged_rolling) {
        PopulateGridV1Diagnostics(segment, snapshot, traversability, false);
      }
      auto cycle_finalized = std::chrono::steady_clock::now();
      segment.timing = cycle_call_timing;
      segment.timing.total_elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              cycle_finalized - cycle_timing->started_at);
      if (stop_token.stop_requested() ||
          segment.status == PlanningStatus::kCanceled) {
        auto canceled = Failure(PlanningStatus::kCanceled,
                                "REQUEST_CANCELED");
        canceled.timing = segment.timing;
        PublishCycleDiagnostics(request_goal, canceled);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        return canceled;
      }
      if (cycle_finalized >= cycle_timing->hard_deadline) {
        auto timeout = Failure(PlanningStatus::kTimedOut, "TIMEOUT");
        timeout.timing = segment.timing;
        PublishCycleDiagnostics(request_goal, timeout);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        return timeout;
      }
      if (segment.status != PlanningStatus::kSuccess ||
          !segment.reference.has_value()) {
        PublishCycleDiagnostics(request_goal, segment);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        return segment;
      }

      const InputSnapshot latest = input_store.Capture();
      if (final_snapshot != nullptr) {
        *final_snapshot = latest;
      }
      if (stop_token.stop_requested()) {
        auto canceled = Failure(PlanningStatus::kCanceled,
                                "REQUEST_CANCELED");
        canceled.timing = segment.timing;
        PublishCycleDiagnostics(request_goal, canceled);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        return canceled;
      }
      cycle_finalized = std::chrono::steady_clock::now();
      if (cycle_finalized >= cycle_timing->hard_deadline) {
        auto timeout = Failure(PlanningStatus::kTimedOut, "TIMEOUT");
        timeout.timing = segment.timing;
        timeout.timing.total_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                cycle_finalized - cycle_timing->started_at);
        PublishCycleDiagnostics(request_goal, timeout);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        return timeout;
      }
      if (!SameRollingPlanningIdentity(snapshot, latest, legged_rolling)) {
        PublishCycleDiagnostics(request_goal, segment, true);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        force_replan = true;
        continue;
      }

      auto converted = ConvertResult(segment, request_goal->mission_revision);
      cycle_finalized = std::chrono::steady_clock::now();
      segment.timing.total_elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              cycle_finalized - cycle_timing->started_at);
      if (stop_token.stop_requested()) {
        auto canceled = Failure(PlanningStatus::kCanceled,
                                "REQUEST_CANCELED");
        canceled.timing = segment.timing;
        PublishCycleDiagnostics(request_goal, canceled);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        return canceled;
      }
      if (cycle_finalized >= cycle_timing->hard_deadline) {
        auto timeout = Failure(PlanningStatus::kTimedOut, "TIMEOUT");
        timeout.timing = segment.timing;
        PublishCycleDiagnostics(request_goal, timeout);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        return timeout;
      }
      const InputSnapshot publish_snapshot = input_store.Capture();
      if (final_snapshot != nullptr) {
        *final_snapshot = publish_snapshot;
      }
      if (!SameRollingPlanningIdentity(snapshot, publish_snapshot,
                                       legged_rolling)) {
        PublishCycleDiagnostics(request_goal, segment, true);
        if (result_diagnostic_published != nullptr) {
          *result_diagnostic_published = true;
        }
        force_replan = true;
        continue;
      }
      if (cycle_finalized >= cycle_timing->sla_milestone) {
        segment.reason_code = "PLAN_FOUND_LATE";
        converted.reason_code = "PLAN_FOUND_LATE";
      }
      SetFinalizedTiming(segment.timing.total_elapsed, segment, converted);
      if (ContextIsValid()) {
        if (legged_rolling) {
          PublishLeggedPath(segment, converted);
        } else {
          PublishWheeledPath(segment, converted,
                             segment.timing.total_elapsed);
        }
      }
      PublishCycleDiagnostics(request_goal, segment);
      if (result_diagnostic_published != nullptr) {
        *result_diagnostic_published = true;
      }

      const std::size_t selected_index = *local.selected_goal_index;
      active_goal = local_goals.goals->goals_odom[selected_index];
      if (!decision.targets_final_goal) {
        const auto* selected = std::get_if<lunar::pure_planning::PointGoal>(
            &active_goal->target);
        if (selected != nullptr) {
          for (const auto& portal : portals.candidates) {
            const auto* candidate =
                std::get_if<lunar::pure_planning::PointGoal>(
                    &portal.goal_odom.target);
            if (candidate != nullptr &&
                candidate->position_m.x == selected->position_m.x &&
                candidate->position_m.y == selected->position_m.y) {
              if (!legged_rolling) {
                minimum_route_progress_m =
                    std::max(minimum_route_progress_m,
                             portal.route_progress_m);
              }
              break;
            }
          }
        }
      }
      if (legged_rolling) {
        legged_replan_anchor_progress_m =
            decision.projected_route_progress_m;
      }
      last_segment = std::move(segment);
      seen_local_sequence = snapshot.local_sequence;
      last_replan = std::chrono::steady_clock::now();
      force_replan = false;
      std::this_thread::sleep_for(std::chrono::milliseconds{
          parameters.rolling_surface.poll_period_ms});
    }
    return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED");
  }

  struct OutputBundle final {
    lunar::pure_planning::PlanningResult result;
    std::shared_ptr<Action::Result> action_result;
    diagnostic_msgs::msg::DiagnosticArray diagnostics;
  };

  [[nodiscard]] OutputBundle MakeOutputs(
      lunar::pure_planning::PlanningResult result,
      const std::shared_ptr<const Action::Goal>& request_goal,
      const InputSnapshot& snapshot) {
    const std::string_view request_id =
        request_goal ? std::string_view{request_goal->request_id}
                     : std::string_view{};
    OutputBundle output{std::move(result), {},
                        diagnostic_msgs::msg::DiagnosticArray{}};
    output.action_result = std::make_shared<Action::Result>(ConvertResult(
        output.result, request_goal ? request_goal->mission_revision : 0U));
    output.diagnostics = MakeRequestDiagnostics(
        request_id, parameters.platform_type,
        request_goal ? EnvironmentMode(request_goal->environment_mode)
                     : lunar::pure_planning::EnvironmentMode::kLunarSurface,
        output.result);
    if (snapshot.global_map) {
      output.action_result->global_map_stamp = snapshot.global_map->header.stamp;
    }
    if (snapshot.local_map) {
      output.action_result->local_map_stamp = snapshot.local_map->header.stamp;
    }
    if (snapshot.odometry) {
      output.action_result->state_stamp = snapshot.odometry->header.stamp;
    }
    return output;
  }

  void FinishUnownedAcceptedGoal(
      const std::shared_ptr<GoalHandle>& goal_handle) noexcept {
    const auto started = std::chrono::steady_clock::now();
    detail::FinalizeAcceptedGoal(
        [&] {
          std::uint64_t mission_revision = 0U;
          try {
            const auto request_goal = goal_handle->get_goal();
            if (request_goal) {
              mission_revision = request_goal->mission_revision;
            }
          } catch (...) {
            SafeLogError(
                "PLANNER_ERROR: failed to read rejected replacement goal");
          }
          const auto elapsed =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - started);
          return detail::MakeMinimalPlannerErrorResult<Action>(mission_revision,
                                                                elapsed);
        },
        [&](const std::shared_ptr<Action::Result>& result) {
          goal_handle->abort(result);
        },
        [&] {
          const auto request_goal = goal_handle->get_goal();
          auto result = Failure(
              lunar::pure_planning::PlanningStatus::kPlannerError,
              "PLANNER_ERROR");
          result.timing.total_elapsed =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - started);
          return MakeRequestDiagnostics(
              request_goal ? std::string_view{request_goal->request_id}
                           : std::string_view{},
              parameters.platform_type,
              request_goal ? EnvironmentMode(request_goal->environment_mode)
                           : lunar::pure_planning::EnvironmentMode::kLunarSurface,
              result);
        },
        [&](diagnostic_msgs::msg::DiagnosticArray& diagnostics) {
          if (!ContextIsValid()) {
            SafeLogError(
                "PLANNER_ERROR: ROS context invalid; rejected replacement "
                "diagnostics not published");
            return;
          }
          diagnostics.header.stamp = node.now();
          diagnostics_publisher->publish(diagnostics);
        },
        [&](const diagnostic_msgs::msg::DiagnosticArray& diagnostics) {
          if (ContextIsValid()) {
            LogDiagnostics(diagnostics);
          }
        },
        [&](const detail::AcceptedGoalFinalizationFailure failure) {
          switch (failure) {
            case detail::AcceptedGoalFinalizationFailure::kResultConstruction:
              SafeLogFatal(
                  "PLANNER_ERROR: failed to construct minimal replacement "
                  "result");
              return;
            case detail::AcceptedGoalFinalizationFailure::kTerminalAttempt:
              SafeLogFatal(
                  "PLANNER_ERROR: failed to terminate rejected accepted "
                  "replacement");
              return;
            case detail::AcceptedGoalFinalizationFailure::
                kDiagnosticConstruction:
              SafeLogError(
                  "PLANNER_ERROR: failed to construct rejected replacement "
                  "diagnostics");
              return;
            case detail::AcceptedGoalFinalizationFailure::
                kDiagnosticPublication:
              SafeLogError(
                  "PLANNER_ERROR: failed to publish rejected replacement "
                  "diagnostics");
              return;
            case detail::AcceptedGoalFinalizationFailure::
                kDiagnosticFormatting:
              SafeLogError(
                  "PLANNER_ERROR: failed to format rejected replacement "
                  "diagnostics");
              return;
          }
        });
  }

  void ExecuteKnownResult(
      const std::shared_ptr<GoalHandle>& goal_handle,
      const std::uint64_t generation,
      const std::chrono::steady_clock::time_point started,
      const InputSnapshot& snapshot,
      lunar::pure_planning::PlanningResult result,
      const bool normal_output_already_published = false) {
    const auto request_goal = goal_handle->get_goal();
    auto normal = MakeOutputs(std::move(result), request_goal, snapshot);
    auto canceled_result = Failure(
        lunar::pure_planning::PlanningStatus::kCanceled, "REQUEST_CANCELED");
    canceled_result.timing = normal.result.timing;
    auto canceled = MakeOutputs(std::move(canceled_result), request_goal,
                                snapshot);
    auto timeout_result = Failure(
        lunar::pure_planning::PlanningStatus::kTimedOut, "TIMEOUT");
    timeout_result.timing = normal.result.timing;
    auto timeout = MakeOutputs(std::move(timeout_result), request_goal,
                               snapshot);
    const bool rolling_surface = UseRollingSurface(request_goal);
    const auto timing_policy =
        lunar::pure_planning::MakeRequestTimingPolicy(started);

    if (ContextIsValid()) {
      try {
        const auto stamp = node.now();
        normal.diagnostics.header.stamp = stamp;
        canceled.diagnostics.header.stamp = stamp;
        timeout.diagnostics.header.stamp = stamp;
      } catch (...) {
        SafeLogError("PLANNER_ERROR: failed to stamp request diagnostics");
      }
    }
    bool client_cancel_was_accepted = false;
    bool replacement_was_accepted = false;
    OutputBundle* committed = &normal;
    {
      std::scoped_lock lock{state_mutex};
      if (!SameGoal(active_goal, goal_handle)) {
        return;
      }
      const auto claim = execution_state.ClaimTerminal(generation,
                                                       goal_handle.get());
      if (!claim.claimed) {
        return;
      }
      client_cancel_was_accepted = claim.client_cancel_accepted;
      replacement_was_accepted = claim.replacement_accepted;
      if (client_cancel_was_accepted || replacement_was_accepted) {
        committed = &canceled;
      }
      const auto finalized = std::chrono::steady_clock::now();
      if (committed == &normal && !rolling_surface) {
        if (finalized >= timing_policy.hard_deadline) {
          committed = &timeout;
        } else if (
            finalized >= timing_policy.sla_milestone &&
            normal.result.status ==
                lunar::pure_planning::PlanningStatus::kSuccess &&
            normal.result.reference.has_value()) {
          normal.result.reason_code = "PLAN_FOUND_LATE";
          normal.action_result->reason_code = "PLAN_FOUND_LATE";
          SetDiagnosticValue(normal.diagnostics, "reason_code",
                             "PLAN_FOUND_LATE");
        }
      }
      const auto elapsed = rolling_surface
                               ? committed->result.timing.total_elapsed
                               : std::chrono::duration_cast<
                                     std::chrono::nanoseconds>(finalized -
                                                               started);
      SetFinalizedTiming(elapsed, committed->result,
                         *committed->action_result, committed->diagnostics);
    }
    const bool suppress_duplicate_reference =
        normal_output_already_published && committed == &normal &&
        committed->result.status ==
            lunar::pure_planning::PlanningStatus::kSuccess;
    if (ContextIsValid()) {
      if (!suppress_duplicate_reference) {
        try {
          PublishWheeledPath(committed->result, *committed->action_result,
                             committed->result.timing.total_elapsed);
        } catch (...) {
          SafeLogError("PLANNER_ERROR: failed to publish wheeled path");
        }
      }
      if (!normal_output_already_published) {
        try {
          diagnostics_publisher->publish(committed->diagnostics);
        } catch (...) {
          SafeLogError("PLANNER_ERROR: failed to publish request diagnostics");
        }
        try {
          LogDiagnostics(committed->diagnostics);
        } catch (...) {
          SafeLogError("PLANNER_ERROR: failed to log request diagnostics");
        }
      }
    } else if (!ContextIsValid()) {
      SafeLogError(
          "PLANNER_ERROR: ROS context invalid; diagnostics not published");
    }
    if (client_cancel_was_accepted) {
      const auto transition = WaitForRclCanceling(goal_handle);
      if (transition != detail::CancelTransitionWaiter::Result::kCanceling) {
        MarkTerminalFailed(generation, goal_handle);
        if (transition ==
            detail::CancelTransitionWaiter::Result::kPredicateError) {
          SafeLogError(
              "PLANNER_ERROR: failed to inspect RCL cancel transition");
        } else {
          SafeLogError(
              "PLANNER_ERROR: cancel transition stopped during teardown");
        }
        return;
      }
    }
    try {
      if (committed->result.status ==
          lunar::pure_planning::PlanningStatus::kSuccess) {
        goal_handle->succeed(committed->action_result);
      } else if (committed->result.status ==
                     lunar::pure_planning::PlanningStatus::kCanceled &&
                 client_cancel_was_accepted) {
        goal_handle->canceled(committed->action_result);
      } else {
        goal_handle->abort(committed->action_result);
      }
      MarkTerminalDelivered(generation, goal_handle);
    } catch (...) {
      MarkTerminalFailed(generation, goal_handle);
      SafeLogFatal("PLANNER_ERROR: failed to set Action terminal state");
    }
  }

  [[nodiscard]] bool ConsumeTrustedBridgeOnce() {
    try {
      bool enabled{};
      if (!node.get_parameter("trusted_bridge_once", enabled) || !enabled) {
        return false;
      }
      (void)node.set_parameter(rclcpp::Parameter{"trusted_bridge_once", false});
      return true;
    } catch (...) {
      return false;
    }
  }

  [[nodiscard]] nav_msgs::msg::Path EmptyMapPath() const {
    nav_msgs::msg::Path path;
    path.header.frame_id = "map";
    path.header.stamp = node.now();
    return path;
  }

  void PublishLeggedPath(
      const lunar::pure_planning::PlanningResult& planning_result,
      const Action::Result& result) {
    nav_msgs::msg::Path local_path = EmptyMapPath();
    nav_msgs::msg::Path global_path = EmptyMapPath();
    if (result.has_reference &&
        result.reference.platform_type == result.reference.LEGGED) {
      local_path = ConvertTrajectoryPath(result.reference);
      if (!planning_result.global_route_preview.poses_map.empty()) {
        global_path = ConvertGlobalPath(planning_result.global_route_preview,
                                        result.reference.header);
      } else {
        global_path = result.reference.path_preview;
      }
    }
    if (legged_path_publisher) {
      legged_path_publisher->publish(local_path);
    }
    if (legged_global_path_publisher) {
      legged_global_path_publisher->publish(global_path);
    }
  }

  void PublishWheeledPath(
      const lunar::pure_planning::PlanningResult& planning_result,
      const Action::Result& result,
      const std::chrono::nanoseconds planning_time) {
    lunar_planning_msgs::msg::MotionReference reference;
    nav_msgs::msg::Path path = EmptyMapPath();
    if (result.has_reference &&
        result.reference.platform_type == result.reference.WHEELED) {
      reference = result.reference;
      path = reference.path_preview;
    }
    wheeled_reference_publisher->publish(reference);
    wheeled_path_publisher->publish(path);

    nav_msgs::msg::Path global_path = EmptyMapPath();
    if (result.has_reference &&
        result.reference.platform_type == result.reference.WHEELED &&
        !planning_result.global_route_preview.poses_map.empty()) {
      global_path = ConvertGlobalPath(planning_result.global_route_preview,
                                      result.reference.header);
    }
    wheeled_global_path_publisher->publish(global_path);

    lunar_planning_msgs::msg::TimedPath timed_path;
    timed_path.path = std::move(path);
    timed_path.planning_time = RosDuration(planning_time);
    timed_path_publisher->publish(timed_path);
  }

  [[nodiscard]] bool ContextIsValid() const noexcept {
    try {
      return rclcpp::ok(node.get_node_base_interface()->get_context());
    } catch (...) {
      return false;
    }
  }

  [[nodiscard]] detail::CancelTransitionWaiter::Result WaitForRclCanceling(
      const std::shared_ptr<GoalHandle>& goal_handle) noexcept {
    return cancel_transition_waiter.Wait(
        [&] { return goal_handle->is_canceling(); },
        [&] { return ContextIsValid(); },
        [&] {
          return teardown_requested.load(std::memory_order_acquire);
        });
  }

  void SafeLogError(const char* message) const noexcept {
    try {
      RCLCPP_ERROR(node.get_logger(), "%s", message);
    } catch (...) {
    }
  }

  void SafeLogFatal(const char* message) const noexcept {
    try {
      RCLCPP_FATAL(node.get_logger(), "%s", message);
    } catch (...) {
    }
  }

  void MarkTerminalDelivered(
      const std::uint64_t generation,
      const std::shared_ptr<GoalHandle>& goal) {
    std::scoped_lock lock{state_mutex};
    if (SameGoal(active_goal, goal)) {
      execution_state.MarkTerminalDelivered(generation, goal.get());
    }
  }

  void MarkTerminalFailed(
      const std::uint64_t generation,
      const std::shared_ptr<GoalHandle>& goal) noexcept {
    try {
      std::scoped_lock lock{state_mutex};
      if (SameGoal(active_goal, goal)) {
        execution_state.MarkTerminalFailed(generation, goal.get());
      }
    } catch (...) {
    }
  }

  void LogDiagnostics(
      const diagnostic_msgs::msg::DiagnosticArray& diagnostics) const {
    if (diagnostics.status.size() != 1U) {
      return;
    }
    std::string fields;
    for (const auto& value : diagnostics.status.front().values) {
      if (!fields.empty()) {
        fields.push_back(' ');
      }
      fields.append(value.key).push_back('=');
      fields.append(value.value);
    }
    RCLCPP_INFO(node.get_logger(), "%s", fields.c_str());
  }

  void FinishWorker(const std::uint64_t generation,
                    const std::shared_ptr<GoalHandle>& goal) {
    std::scoped_lock lock{state_mutex};
    if (SameGoal(active_goal, goal)) {
      execution_state.Finish(generation, goal.get());
    }
    if (execution_state.phase() ==
            detail::ActionExecutionState::Phase::kIdle &&
        SameGoal(active_goal, goal)) {
      active_goal.reset();
    }
  }

  void StopAndJoinWorker() {
    std::unique_ptr<std::jthread> joinable;
    {
      std::scoped_lock lock{state_mutex};
      if (!worker) {
        return;
      }
      worker->request_stop();
      if (worker->get_id() == std::this_thread::get_id()) {
        return;
      }
      joinable = std::move(worker);
    }
    if (joinable->joinable()) {
      joinable->join();
    }
  }

  PurePlanMotionServer& node;
  RuntimeParameters parameters;
  PlannerFn planner;
  LocalPlannerFn local_planner;
  InputStore input_store;
  std::unique_ptr<TraversabilityInput> traversability_input;
  rclcpp::CallbackGroup::SharedPtr action_group;
  rclcpp::CallbackGroup::SharedPtr input_group;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
      global_subscription;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr
      local_subscription;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscription;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_publisher;
  rclcpp::Publisher<lunar_planning_msgs::msg::MotionReference>::SharedPtr
      wheeled_reference_publisher;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr wheeled_path_publisher;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr
      wheeled_global_path_publisher;
  rclcpp::Publisher<lunar_planning_msgs::msg::TimedPath>::SharedPtr
      timed_path_publisher;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr legged_path_publisher;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr
      legged_global_path_publisher;
  rclcpp_action::Server<Action>::SharedPtr action_server;
  std::mutex state_mutex;
  std::unique_ptr<std::jthread> worker;
  std::shared_ptr<GoalHandle> active_goal;
  detail::ActionExecutionState execution_state;
  detail::CancelTransitionWaiter cancel_transition_waiter;
  std::atomic<bool> teardown_requested{false};
};

PurePlanMotionServer::PurePlanMotionServer(const rclcpp::NodeOptions& options,
                                           PlannerFn planner,
                                           LocalPlannerFn local_planner)
    : rclcpp::Node("pure_planner", options),
      impl_(std::make_unique<Impl>(*this, std::move(planner),
                                   std::move(local_planner))) {}

PurePlanMotionServer::~PurePlanMotionServer() = default;

}  // namespace lunar::pure_planner_ros
