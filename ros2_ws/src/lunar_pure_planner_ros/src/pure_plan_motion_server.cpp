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
#include <utility>
#include <variant>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/subscription_options.hpp>
#include <rclcpp_action/create_server.hpp>
#include <rclcpp_action/server.hpp>
#include <rclcpp_action/server_goal_handle.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_pure_planner_core/planner.hpp"
#include "hierarchical/frame_transform.hpp"
#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/surface_rolling_session.hpp"
#include "lunar_pure_planner_ros/input_store.hpp"
#include "lunar_pure_planner_ros/incremental_traversability.hpp"
#include "lunar_pure_planner_ros/map_adapters.hpp"
#include "lunar_pure_planner_ros/message_conversion.hpp"
#include "lunar_pure_planner_ros/platform_config.hpp"
#include "lunar_pure_planner_ros/request_diagnostics.hpp"
#include "lunar_pure_planner_ros/state_adapter.hpp"
#include "lunar_pure_planner_ros/trusted_bridge.hpp"

namespace lunar::pure_planner_ros {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using namespace std::chrono_literals;

constexpr std::string_view kPackageName{"lunar_pure_planner_ros"};

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
      .rolling_surface = rolling_surface,
  };
  for (const std::string* interface_name : {
           &parameters.global_map_topic, &parameters.local_map_topic,
           &parameters.odometry_topic, &parameters.tf_topic,
           &parameters.action_name, &parameters.diagnostics_topic,
           &parameters.wheeled_reference_topic}) {
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

[[nodiscard]] lunar::pure_planning::PlanningResult Failure(
    const lunar::pure_planning::PlanningStatus status,
    std::string reason_code) {
  return {.status = status, .reason_code = std::move(reason_code)};
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

void SetFinalizedTiming(
    const std::chrono::nanoseconds elapsed,
    lunar::pure_planning::PlanningResult& result,
    Action::Result& action_result,
    diagnostic_msgs::msg::DiagnosticArray& diagnostics) {
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
  std::ostringstream milliseconds;
  milliseconds << std::setprecision(15)
               << std::chrono::duration<double, std::milli>(elapsed).count();
  SetDiagnosticValue(diagnostics, "total_elapsed_ms", milliseconds.str());
  SetDiagnosticValue(
      diagnostics, "latency_class",
      std::string{lunar::pure_planning::RequestLatencyClassName(latency_class)});
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

std::unique_ptr<rclcpp::Executor> MakePurePlannerExecutor() {
  return std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
      rclcpp::ExecutorOptions{}, 2U);
}

struct PurePlanMotionServer::Impl final {
  Impl(PurePlanMotionServer& owner, PlannerFn planner)
      : node(owner), parameters(ReadRuntimeParameters(owner)),
        planner(std::move(planner)) {
    if (!this->planner) {
      this->planner = RealPlannerFn();
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
          input_store.UpdateGlobal(std::move(message));
        }, input_options);
    local_subscription = node.create_subscription<grid_map_msgs::msg::GridMap>(
        parameters.local_map_topic, input_qos,
        [this](grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
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
          input_store.UpdateTf(*message);
        }, input_options);
    diagnostics_publisher =
        node.create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            parameters.diagnostics_topic, rclcpp::QoS{10}.reliable());
    wheeled_reference_publisher = node.create_publisher<
        lunar_planning_msgs::msg::MotionReference>(
        parameters.wheeled_reference_topic, rclcpp::QoS{1}.reliable());
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
    if (UseRollingSurface(goal_handle->get_goal())) {
      InputSnapshot final_snapshot;
      auto result = ExecuteRollingSurfaceWheel(
          goal_handle->get_goal(), stop_token, &final_snapshot);
      try {
        ExecuteKnownResult(goal_handle, generation, started, final_snapshot,
                           std::move(result));
      } catch (...) {
        SafeLogError("PLANNER_ERROR: rolling finalization failed");
      }
      return;
    }
    const auto timing_policy =
        lunar::pure_planning::MakeRequestTimingPolicy(started);
    const InputSnapshot snapshot = input_store.Capture();
    const auto request_goal = goal_handle->get_goal();
    lunar::pure_planning::PlanningResult result;
    try {
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
        const auto world = AdaptSnapshot(
            EnvironmentMode(request_goal->environment_mode), snapshot);
        const auto state = AdaptOdometry(*snapshot.odometry,
                                         parameters.platform_type);
        if (!map_from_odom.value.has_value() || !converted_goal.ok() ||
            !world.value.has_value() || !state.value.has_value()) {
          result = Failure(lunar::pure_planning::PlanningStatus::kInvalidInput,
                           "INVALID_INPUT");
        } else {
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
          };
          const auto bridge = trusted_bridge_once
                                  ? ApplyTrustedBridge(request, snapshot)
                                  : std::optional<AppliedTrustedBridge>{};
          result = planner(request);
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
    return goal && parameters.rolling_surface.enabled &&
           parameters.platform_type == lunar::pure_planning::PlatformType::kWheeled &&
           EnvironmentMode(goal->environment_mode) ==
               lunar::pure_planning::EnvironmentMode::kLunarSurface;
  }

  [[nodiscard]] lunar::pure_planning::PlanningResult ExecuteRollingSurfaceWheel(
      const std::shared_ptr<const Action::Goal>& request_goal,
      const std::stop_token stop_token,
      InputSnapshot* const final_snapshot) {
    using lunar::pure_planning::EnvironmentMode;
    using lunar::pure_planning::PlanningRequest;
    using lunar::pure_planning::PlanningResult;
    using lunar::pure_planning::PlanningStatus;
    using lunar::pure_planning::Pose3;
    using lunar::pure_planning::SearchControl;
    using lunar::pure_planning::WheeledState;
    const auto global_started = std::chrono::steady_clock::now();
    const auto global_timing =
        lunar::pure_planning::MakeRequestTimingPolicy(global_started);
    InputSnapshot snapshot = input_store.Capture();
    if (final_snapshot != nullptr) {
      *final_snapshot = snapshot;
    }
    if (!request_goal || !snapshot.map_from_odom.has_value() ||
        !snapshot.odometry || !snapshot.local_map || !snapshot.global_map) {
      return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
    }
    const auto initial_transform =
        AdaptDirectMapFromOdom(*snapshot.map_from_odom);
    const auto initial_goal = initial_transform.value.has_value()
        ? ConvertGoal(*request_goal, *initial_transform.value)
        : GoalConversionResult{};
    const auto initial_world = AdaptSnapshot(EnvironmentMode::kLunarSurface,
                                             snapshot);
    const auto initial_state = AdaptOdometry(
        *snapshot.odometry, lunar::pure_planning::PlatformType::kWheeled);
    if (!initial_transform.value.has_value() || !initial_goal.ok() ||
        !initial_world.value.has_value() || !initial_state.value.has_value()) {
      return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
    }
    PlanningRequest global_request{
        .request_id = request_goal->request_id,
        .environment_mode = EnvironmentMode::kLunarSurface,
        .current_state = *initial_state.value,
        .goal_map = *initial_goal.goal,
        .world = *initial_world.value,
        .capability = parameters.capability,
        .config = parameters.planner_config,
        .control = {.deadline = global_timing.hard_deadline,
                    .stop_token = stop_token,
                    .now = [] { return std::chrono::steady_clock::now(); }},
        .request_started_at = global_started,
    };
    global_request.config.search.stop_after_first_solution = true;
    const auto global = lunar::pure_planning::hierarchical::PlanSurfaceGlobal(
        global_request, global_request.control);
    if (!global.route.has_value()) {
      return Failure(global.reason_code == "NO_PATH" ? PlanningStatus::kNoPath
                     : global.reason_code == "TIMEOUT" ? PlanningStatus::kTimedOut
                     : global.reason_code == "REQUEST_CANCELED" ? PlanningStatus::kCanceled
                     : PlanningStatus::kInvalidInput,
                     global.reason_code.empty() ? "PLANNER_ERROR" : global.reason_code);
    }
    lunar::pure_planning::hierarchical::SurfaceRollingSession session{
        *global.route, global_request.goal_map,
        {.horizon_m = parameters.rolling_surface.horizon_m,
         .max_deviation_m = parameters.rolling_surface.max_deviation_m}};
    std::optional<lunar::pure_planning::GoalRegion> active_goal;
    std::optional<PlanningResult> last_segment;
    std::uint64_t seen_local_sequence{};
    auto last_replan = global_started -
        std::chrono::milliseconds{parameters.rolling_surface.min_replan_interval_ms};
    while (!stop_token.stop_requested()) {
      const auto cycle_started = std::chrono::steady_clock::now();
      snapshot = input_store.Capture();
      if (final_snapshot != nullptr) {
        *final_snapshot = snapshot;
      }
      if (!snapshot.map_from_odom.has_value() || !snapshot.odometry ||
          !snapshot.local_map) {
        return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
      }
      const auto transform = AdaptDirectMapFromOdom(*snapshot.map_from_odom);
      const auto world = AdaptSnapshot(EnvironmentMode::kLavaTube, snapshot);
      const auto state = AdaptOdometry(
          *snapshot.odometry, lunar::pure_planning::PlatformType::kWheeled);
      if (!transform.value.has_value() || !world.value.has_value() ||
          !state.value.has_value()) {
        return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
      }
      const auto* wheel_state = std::get_if<WheeledState>(&*state.value);
      const auto pose_map = wheel_state == nullptr
          ? std::optional<Pose3>{}
          : lunar::pure_planning::hierarchical::TransformPose(
                wheel_state->pose, *transform.value,
                lunar::pure_planning::hierarchical::TransformDirection::kChildToParent);
      if (!pose_map.has_value()) {
        return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
      }
      const auto decision = session.Decide(*pose_map);
      if (decision.kind == lunar::pure_planning::hierarchical::SurfaceRollingDecision::Kind::kInvalidRoute) {
        return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
      }
      if (decision.kind == lunar::pure_planning::hierarchical::SurfaceRollingDecision::Kind::kFinalGoalReached) {
        return last_segment.has_value()
            ? std::move(*last_segment)
            : Failure(PlanningStatus::kNoPath, "NO_PATH");
      }
      const auto* candidate = decision.goal.has_value()
          ? std::get_if<lunar::pure_planning::PointGoal>(&decision.goal->target)
          : nullptr;
      const auto* active = active_goal.has_value()
          ? std::get_if<lunar::pure_planning::PointGoal>(&active_goal->target)
          : nullptr;
      const bool target_reached = active != nullptr && pose_map.has_value() &&
          std::hypot(pose_map->position_m.x - active->position_m.x,
                     pose_map->position_m.y - active->position_m.y) <=
              active->tolerance_m;
      const bool local_changed = snapshot.local_sequence != seen_local_sequence &&
          std::chrono::steady_clock::now() - last_replan >=
              std::chrono::milliseconds{parameters.rolling_surface.min_replan_interval_ms};
      const bool deviated = decision.lateral_deviation_m >
          parameters.rolling_surface.max_deviation_m;
      const bool target_changed = candidate != nullptr && active != nullptr &&
          std::hypot(candidate->position_m.x - active->position_m.x,
                     candidate->position_m.y - active->position_m.y) > 1.0e-6;
      if (!last_segment.has_value() || target_reached || local_changed || deviated ||
          target_changed) {
        PlanningRequest local_request = global_request;
        local_request.environment_mode = EnvironmentMode::kLavaTube;
        local_request.current_state = *state.value;
        local_request.goal_map = *decision.goal;
        local_request.world = *world.value;
        const auto cycle_timing =
            lunar::pure_planning::MakeRequestTimingPolicy(cycle_started);
        local_request.control = {
            .deadline = cycle_timing.hard_deadline,
            .stop_token = stop_token,
            .now = [] { return std::chrono::steady_clock::now(); },
        };
        local_request.request_started_at = cycle_started;
        PlanningResult segment = planner(local_request);
        if (segment.status != PlanningStatus::kSuccess ||
            !segment.reference.has_value()) {
          return segment;
        }
        const auto converted = ConvertResult(segment, request_goal->mission_revision);
        if (ContextIsValid()) {
          PublishWheeledReference(converted);
        }
        active_goal = *decision.goal;
        last_segment = std::move(segment);
        seen_local_sequence = snapshot.local_sequence;
        last_replan = std::chrono::steady_clock::now();
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds{parameters.rolling_surface.poll_period_ms});
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
      lunar::pure_planning::PlanningResult result) {
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
    if (ContextIsValid()) {
      try {
        PublishWheeledReference(*committed->action_result);
      } catch (...) {
        SafeLogError("PLANNER_ERROR: failed to publish wheeled reference");
      }
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
    } else {
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

  void PublishWheeledReference(const Action::Result& result) {
    lunar_planning_msgs::msg::MotionReference reference;
    if (result.has_reference &&
        result.reference.platform_type == result.reference.WHEELED) {
      reference = result.reference;
    }
    wheeled_reference_publisher->publish(reference);
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
  InputStore input_store;
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
  rclcpp_action::Server<Action>::SharedPtr action_server;
  std::mutex state_mutex;
  std::unique_ptr<std::jthread> worker;
  std::shared_ptr<GoalHandle> active_goal;
  detail::ActionExecutionState execution_state;
  detail::CancelTransitionWaiter cancel_transition_waiter;
  std::atomic<bool> teardown_requested{false};
};

PurePlanMotionServer::PurePlanMotionServer(const rclcpp::NodeOptions& options,
                                           PlannerFn planner)
    : rclcpp::Node("pure_planner", options),
      impl_(std::make_unique<Impl>(*this, std::move(planner))) {}

PurePlanMotionServer::~PurePlanMotionServer() = default;

}  // namespace lunar::pure_planner_ros
