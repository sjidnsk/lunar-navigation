#include "lunar_pure_exploration_ros/exploration_node.hpp"
#include "lunar_pure_planner_core/global_goal_feasibility.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <numbers>
#include <ranges>
#include <set>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <lunar_pure_exploration_core/exploration_state_machine.hpp>
#include <lunar_pure_exploration_core/occupancy_grid.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_task.hpp>
#include <rclcpp/qos.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_pure_exploration_ros/marker_builder.hpp"
#include "lunar_pure_exploration_ros/planner_client.hpp"
#include "lunar_pure_exploration_ros/planner_timing_accumulator.hpp"
#include "lunar_pure_exploration_ros/platform_config_loader.hpp"
#include "lunar_pure_exploration_ros/pose_resolver.hpp"

namespace lunar::pure_exploration_ros {

ExecutionMonitor::ExecutionMonitor(ExecutionMonitorParameters parameters)
    : position_tolerance_m_(parameters.position_tolerance_m),
      yaw_tolerance_rad_(parameters.yaw_tolerance_rad),
      progress_({.maximum_executable_path_points = parameters.maximum_executable_path_points,
                 .timeout = parameters.stuck_window,
                 .minimum_progress_m = parameters.minimum_progress_m}) {
  if (!std::isfinite(position_tolerance_m_) || position_tolerance_m_ <= 0.0 ||
      !std::isfinite(yaw_tolerance_rad_) || yaw_tolerance_rad_ <= 0.0) {
    throw std::invalid_argument{"execution monitor tolerances must be positive"};
  }
}

bool ExecutionMonitor::ReachedFinalGoal(
    const lunar::pure_exploration::Pose2 position,
    const lunar::pure_exploration::Pose2 target) const {
  if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
      !std::isfinite(position.yaw) || !std::isfinite(target.x) ||
      !std::isfinite(target.y) || !std::isfinite(target.yaw)) {
    return false;
  }
  return std::hypot(position.x - target.x, position.y - target.y) <=
             position_tolerance_m_ &&
         std::abs(std::remainder(position.yaw - target.yaw,
                                 2.0 * std::numbers::pi)) <= yaw_tolerance_rad_;
}

bool ExecutionMonitor::ReachedSegmentEndpoint(
    const lunar::pure_exploration::Vec2 position,
    const lunar::pure_exploration::Vec2 endpoint) const {
  return std::isfinite(position.x) && std::isfinite(position.y) &&
         std::isfinite(endpoint.x) && std::isfinite(endpoint.y) &&
         std::hypot(position.x - endpoint.x, position.y - endpoint.y) <=
             position_tolerance_m_;
}

void ExecutionMonitor::ResetProgress(
    const std::chrono::steady_clock::time_point now,
    const std::span<const lunar::pure_exploration::Vec2> polyline,
    const lunar::pure_exploration::Vec2 position) {
  progress_.Reset(now, polyline, position);
}

bool ExecutionMonitor::UpdateProgress(
    const std::chrono::steady_clock::time_point now,
    const lunar::pure_exploration::Vec2 position, const bool paused) {
  return progress_.Update(now, position, paused);
}

namespace {

using lunar::pure_exploration::CalculateCoverage;
using lunar::pure_exploration::BoundaryGuidance;
using lunar::pure_exploration::CandidateGain;
using lunar::pure_exploration::CandidateGenerator;
using lunar::pure_exploration::CandidateRanker;
using lunar::pure_exploration::CellState;
using lunar::pure_exploration::CoverageStats;
using lunar::pure_exploration::ExplorationState;
using lunar::pure_exploration::FrontierDetector;
using lunar::pure_exploration::FrontierParameters;
using lunar::pure_exploration::GridGeometry;
using lunar::pure_exploration::InformationGainEvaluator;
using lunar::pure_exploration::MakeActiveGoal;
using lunar::pure_exploration::MakeBoundaryApproachActiveGoal;
using lunar::pure_exploration::NavigationPhase;
using lunar::pure_exploration::OccupancyGridView;
using lunar::pure_exploration::PlannedCandidate;
using lunar::pure_exploration::Polygon2;
using lunar::pure_exploration::Pose2;
using lunar::pure_exploration::TaskRaster;
using lunar::pure_exploration::Vec2;
using lunar::pure_exploration::ActiveGoal;
using lunar::pure_planning::EvaluateGlobalGoalFeasibility;
using lunar::pure_planning::GlobalGoalFeasibilityRequest;
using Task = lunar_pure_exploration_msgs::msg::PureExplorationTask;
using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;
using MotionReference = lunar_planning_msgs::msg::MotionReference;
using DiagnosticArray = diagnostic_msgs::msg::DiagnosticArray;
using DiagnosticStatus = diagnostic_msgs::msg::DiagnosticStatus;
using KeyValue = diagnostic_msgs::msg::KeyValue;

static_assert(std::is_nothrow_move_constructible_v<MotionReference>);

class SerializedWorkQueue final {
 public:
  using Work = std::function<void()>;

  SerializedWorkQueue()
      : worker_([this](const std::stop_token stop_token) {
          Run(stop_token);
        }) {}

  ~SerializedWorkQueue() noexcept { Stop(); }

  SerializedWorkQueue(const SerializedWorkQueue&) = delete;
  SerializedWorkQueue& operator=(const SerializedWorkQueue&) = delete;

  bool Submit(Work work) {
    std::scoped_lock lock{mutex_};
    if (stopping_) {
      return false;
    }
    pending_ = std::move(work);
    condition_.notify_one();
    return true;
  }

  void Stop() noexcept {
    {
      std::scoped_lock lock{mutex_};
      if (stopping_) {
        return;
      }
      stopping_ = true;
      pending_.reset();
    }
    worker_.request_stop();
    condition_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

 private:
  void Run(const std::stop_token stop_token) noexcept {
    for (;;) {
      Work work;
      {
        std::unique_lock lock{mutex_};
        condition_.wait(lock, [this, stop_token] {
          return stopping_ || stop_token.stop_requested() || pending_;
        });
        if (stopping_ || stop_token.stop_requested()) {
          return;
        }
        work = std::move(*pending_);
        pending_.reset();
      }
      try {
        work();
      } catch (...) {
        // Work items own their fail-closed transition. Never terminate the
        // process at the serialized worker boundary.
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<Work> pending_;
  bool stopping_{false};
  std::jthread worker_;
};

bool Finite(const Pose2 pose) {
  return std::isfinite(pose.x) && std::isfinite(pose.y) &&
         std::isfinite(pose.yaw);
}

bool FiniteGeometry(const GridGeometry& geometry) {
  return std::isfinite(geometry.resolution) && geometry.resolution > 0.0 &&
         std::isfinite(geometry.origin_x) &&
         std::isfinite(geometry.origin_y) &&
         std::isfinite(geometry.origin_yaw);
}

double QuaternionYaw(const geometry_msgs::msg::Quaternion& quaternion) {
  const double x = quaternion.x;
  const double y = quaternion.y;
  const double z = quaternion.z;
  const double w = quaternion.w;
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
      !std::isfinite(w)) {
    throw std::invalid_argument{"global map quaternion must be finite"};
  }
  const double scale =
      std::max({std::abs(x), std::abs(y), std::abs(z), std::abs(w)});
  if (scale == 0.0) {
    throw std::invalid_argument{"global map quaternion must be nonzero"};
  }
  const double scaled_norm =
      std::hypot(std::hypot(x / scale, y / scale),
                 std::hypot(z / scale, w / scale));
  if (!std::isfinite(scaled_norm) || scaled_norm == 0.0) {
    throw std::invalid_argument{"global map quaternion cannot normalize"};
  }
  const double nx = x / scale / scaled_norm;
  const double ny = y / scale / scaled_norm;
  const double nz = z / scale / scaled_norm;
  const double nw = w / scale / scaled_norm;
  const double yaw = std::atan2(2.0 * (nw * nz + nx * ny),
                                1.0 - 2.0 * (ny * ny + nz * nz));
  if (!std::isfinite(yaw)) {
    throw std::invalid_argument{"global map yaw must be finite"};
  }
  return yaw == 0.0 ? 0.0 : yaw;
}

FrozenGlobalMapContent FreezeGlobalMap(
    const nav_msgs::msg::OccupancyGrid& map,
    const std::int8_t occupied_threshold) {
  if (map.header.frame_id != "map") {
    throw std::invalid_argument{"global map frame must be map"};
  }
  const auto& position = map.info.origin.position;
  if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
      !std::isfinite(position.z)) {
    throw std::invalid_argument{"global map origin must be finite"};
  }
  FrozenGlobalMapContent result{
      .geometry = {.width = map.info.width,
                   .height = map.info.height,
                   .resolution = static_cast<double>(map.info.resolution),
                   .origin_x = position.x,
                   .origin_y = position.y,
                   .origin_yaw = QuaternionYaw(map.info.origin.orientation)},
      .data = map.data};
  static_cast<void>(OccupancyGridView(result.geometry, result.data,
                                      occupied_threshold));
  return result;
}

bool GlobalGoalCellFeasible(const FrozenGlobalMapContent& map,
                            const Pose2 pose,
                            const double inflation_m,
                            const std::int8_t occupied_threshold) {
  if (map.geometry.origin_yaw != 0.0) {
    return false;
  }
  GlobalGoalFeasibilityRequest request;
  request.global_map.frame_id = "map";
  request.global_map.width = map.geometry.width;
  request.global_map.height = map.geometry.height;
  request.global_map.resolution_m = map.geometry.resolution;
  request.global_map.origin_m = {.x = map.geometry.origin_x,
                                 .y = map.geometry.origin_y,
                                 .z = 0.0};
  request.global_map.layers.emplace(
      "occupancy", lunar::pure_planning::GridLayer{.values = map.data});
  request.goal_position_m = {.x = pose.x, .y = pose.y};
  request.obstacle_threshold_percent = occupied_threshold;
  request.inflation_m = inflation_m;
  return EvaluateGlobalGoalFeasibility(std::move(request)).feasible;
}

bool SameGeometry(const GridGeometry& left, const GridGeometry& right) {
  return left.width == right.width && left.height == right.height &&
         left.resolution == right.resolution &&
         left.origin_x == right.origin_x &&
         left.origin_y == right.origin_y &&
         left.origin_yaw == right.origin_yaw;
}

void ValidateBatchContent(
    const FrozenGlobalMapContent& map,
    const std::vector<lunar::pure_exploration::FrontierCluster>& frontiers,
    const std::vector<lunar::pure_exploration::CandidateView>& candidates) {
  static_cast<void>(OccupancyGridView(map.geometry, map.data, 50));
  std::set<std::pair<std::vector<std::int64_t>,
                     lunar::pure_exploration::CandidateKey>>
      identities;
  for (const auto& candidate : candidates) {
    if (candidate.frontier_index >= frontiers.size()) {
      throw std::invalid_argument{"candidate frontier index is out of range"};
    }
    const auto& frontier = frontiers[candidate.frontier_index];
    if (frontier.canonical_key.empty() ||
        !candidate.frontier_canonical_key ||
        candidate.frontier_canonical_key->empty()) {
      throw std::invalid_argument{"candidate frontier key is empty"};
    }
    if (candidate.frontier_id != frontier.id) {
      throw std::invalid_argument{"candidate frontier id mismatch"};
    }
    if (*candidate.frontier_canonical_key != frontier.canonical_key) {
      throw std::invalid_argument{"candidate frontier key mismatch"};
    }
    if (!Finite(candidate.pose) ||
        !std::isfinite(candidate.frontier_distance_m) ||
        candidate.frontier_distance_m < 0.0) {
      throw std::invalid_argument{"candidate structure is invalid"};
    }
    if (!identities
             .insert({frontier.canonical_key, candidate.key})
             .second) {
      throw std::invalid_argument{"candidate full identity is duplicated"};
    }
  }
}

bool FiniteRanked(const lunar::pure_exploration::RankedCandidate& row) {
  return std::isfinite(row.information_gain_m2) &&
         std::isfinite(row.euclidean_distance_m) &&
         std::isfinite(row.path_length_m) &&
         std::isfinite(row.heading_change_rad) &&
         std::isfinite(row.revisit_penalty) &&
         std::isfinite(row.rank_value);
}

std::string ResourceReason(const std::length_error& error) {
  const std::string_view message{error.what()};
  if (message.find("boundary guidance map") != std::string_view::npos ||
      message.find("boundary guidance task raster") !=
          std::string_view::npos) {
    return "RESOURCE_GUIDANCE_GRID_LIMIT";
  }
  if (message.find("boundary guidance work") != std::string_view::npos ||
      message.find("boundary guidance route") != std::string_view::npos ||
      message.find("boundary guidance visibility") !=
          std::string_view::npos) {
    return "RESOURCE_GUIDANCE_WORK_LIMIT";
  }
  if (message.find("boundary guidance approach candidate") !=
      std::string_view::npos) {
    return "RESOURCE_GUIDANCE_CANDIDATE_LIMIT";
  }
  if (message.find("task raster") != std::string_view::npos) {
    return "RESOURCE_TASK_RASTER_LIMIT";
  }
  if (message.find("position probe") != std::string_view::npos) {
    return "RESOURCE_CANDIDATE_POSITION_PROBE_LIMIT";
  }
  if (message.find("candidate view") != std::string_view::npos ||
      message.find("candidate output") != std::string_view::npos) {
    return "RESOURCE_CANDIDATE_VIEW_LIMIT";
  }
  if (message.find("collision") != std::string_view::npos ||
      message.find("footprint") != std::string_view::npos) {
    return "RESOURCE_COLLISION_WORK_LIMIT";
  }
  if (message.find("visibility") != std::string_view::npos) {
    return "RESOURCE_VISIBILITY_WORK_LIMIT";
  }
  if (message.find("failure entry") != std::string_view::npos) {
    return "RESOURCE_FAILURE_ENTRY_LIMIT";
  }
  if (message.find("failure total") != std::string_view::npos) {
    return "RESOURCE_FAILURE_TOTAL_PATCH_LIMIT";
  }
  if (message.find("failure patch") != std::string_view::npos) {
    return "RESOURCE_FAILURE_PATCH_LIMIT";
  }
  return "RESOURCE_LIMIT_EXCEEDED";
}

std::string GuidanceResourceReason(const std::length_error& error) {
  const std::string_view message{error.what()};
  if (message.find("boundary guidance map") != std::string_view::npos ||
      message.find("boundary guidance task raster") !=
          std::string_view::npos) {
    return "RESOURCE_GUIDANCE_GRID_LIMIT";
  }
  if (message.find("boundary guidance approach candidate") !=
      std::string_view::npos) {
    return "RESOURCE_GUIDANCE_CANDIDATE_LIMIT";
  }
  // BoundaryGuidance owns the shared SafePoseValidator calls.  Its
  // footprint/collision limit messages intentionally retain their generic
  // wording, but exhausting that work is guidance work at this boundary.
  return "RESOURCE_GUIDANCE_WORK_LIMIT";
}

std::uint32_t ToU32(const std::size_t value) {
  return static_cast<std::uint32_t>(
      std::min(value,
               static_cast<std::size_t>(
                   std::numeric_limits<std::uint32_t>::max())));
}

std::uint8_t StatusState(const ExplorationState state) {
  switch (state) {
    case ExplorationState::kIdle:
      return Status::IDLE;
    case ExplorationState::kWaitingForInput:
      return Status::WAITING_FOR_INPUT;
    case ExplorationState::kSelectingFrontier:
      return Status::SELECTING_FRONTIER;
    case ExplorationState::kPlanning:
      return Status::PLANNING;
    case ExplorationState::kExecuting:
      return Status::EXECUTING;
    case ExplorationState::kReplanning:
      return Status::REPLANNING;
    case ExplorationState::kPaused:
      return Status::PAUSED;
    case ExplorationState::kCompleted:
      return Status::COMPLETED;
    case ExplorationState::kError:
      return Status::ERROR;
  }
  throw std::invalid_argument{"unknown exploration state"};
}

Polygon2 TaskBoundary(const Task& task) {
  if (task.task_id.empty() || task.header.frame_id != "map" ||
      task.boundary.points.size() < 3U) {
    throw std::invalid_argument{"invalid START task structure"};
  }
  Polygon2 boundary;
  boundary.vertices.reserve(task.boundary.points.size());
  for (const auto& point : task.boundary.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
      throw std::invalid_argument{"task boundary must be finite"};
    }
    boundary.vertices.push_back(
        {static_cast<double>(point.x), static_cast<double>(point.y)});
  }
  return boundary;
}

struct ExecutableReference {
  std::vector<Vec2> polyline;
  std::optional<Vec2> endpoint;
};

ExecutableReference ValidateExecutableReference(
    const lunar_planning_msgs::msg::MotionReference& reference,
    const std::string_view expected_platform_type,
    const std::size_t maximum_points) {
  std::uint8_t expected_type{};
  if (expected_platform_type == "WHEELED") {
    expected_type = reference.WHEELED;
  } else if (expected_platform_type == "LEGGED") {
    expected_type = reference.LEGGED;
  } else if (expected_platform_type == "HOPPER") {
    expected_type = reference.HOPPER;
  } else {
    throw std::invalid_argument{"unsupported executable platform type"};
  }
  if (reference.plan_id.empty() || reference.platform_type != expected_type) {
    throw std::invalid_argument{"motion reference platform mismatch"};
  }

  ExecutableReference result;
  if (expected_type == reference.HOPPER) {
    if (!reference.trajectory.points.empty() || reference.hops.empty()) {
      throw std::invalid_argument{"invalid hopper executable payload"};
    }
    if (reference.hops.size() > maximum_points) {
      throw std::length_error{"executable path point limit exceeded"};
    }
    result.polyline.reserve(reference.hops.size());
    for (const auto& hop : reference.hops) {
      const auto& landing = hop.nominal_landing_point;
      if (!std::isfinite(landing.x) || !std::isfinite(landing.y)) {
        throw std::invalid_argument{"hopper landing XY must be finite"};
      }
      result.polyline.push_back({landing.x, landing.y});
    }
  } else {
    const auto& points = reference.trajectory.points;
    if (!reference.hops.empty() || points.empty()) {
      throw std::invalid_argument{"invalid trajectory executable payload"};
    }
    if (points.size() > maximum_points) {
      throw std::length_error{"executable path point limit exceeded"};
    }
    result.polyline.reserve(points.size());
    for (const auto& point : points) {
      if (point.transforms.size() != 1U) {
        throw std::invalid_argument{"trajectory point is malformed"};
      }
      const auto& translation = point.transforms.front().translation;
      if (!std::isfinite(translation.x) || !std::isfinite(translation.y)) {
        throw std::invalid_argument{"trajectory XY must be finite"};
      }
      result.polyline.push_back({translation.x, translation.y});
    }
  }
  result.endpoint = result.polyline.back();
  return result;
}

std::size_t PositiveSizeParameter(rclcpp::Node& node,
                                  const std::string& name) {
  const std::int64_t value = node.declare_parameter<std::int64_t>(name, 0);
  if (value <= 0 ||
      static_cast<std::uint64_t>(value) >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument{name + " must be a required positive integer"};
  }
  return static_cast<std::size_t>(value);
}

std::string AbsoluteTopicParameter(rclcpp::Node& node,
                                   const std::string& name,
                                   const std::string& default_value) {
  std::string value = node.declare_parameter<std::string>(name, default_value);
  if (value.empty() || value.front() != '/') {
    throw std::invalid_argument{name + " must be absolute"};
  }
  return value;
}

ExplorationNodeParameters LoadParameters(rclcpp::Node& node) {
  const std::string platform_selector =
      node.declare_parameter<std::string>("platform_selector", "wheel");
  const std::string platform_config =
      node.declare_parameter<std::string>("platform_config", "");
  if (platform_config.empty()) {
    throw std::invalid_argument{"platform_config is required"};
  }
  auto platform =
      LoadPlatformConfig(std::filesystem::path{platform_config},
                         platform_selector)
          .geometry;

  const std::int64_t threshold =
      node.declare_parameter<std::int64_t>("global_occupied_threshold", 50);
  const std::int64_t maximum_task_raster_cells =
      node.declare_parameter<std::int64_t>("maximum_task_raster_cells",
                                           1048576);
  const double sensor_range_m =
      node.declare_parameter<double>("sensor_range_m", 10.0);
  const double sensor_fov_deg =
      node.declare_parameter<double>("sensor_fov_deg", 90.0);
  const std::vector<double> yaw_offsets_deg =
      node.declare_parameter<std::vector<double>>(
          "yaw_offsets_deg", {-45.0, -22.5, 0.0, 22.5, 45.0});
  const double information_weight = node.declare_parameter<double>(
      "score_weights.information_gain", 0.60);
  const double path_weight = node.declare_parameter<double>(
      "score_weights.global_path_length", 0.30);
  const double heading_weight = node.declare_parameter<double>(
      "score_weights.heading_change", 0.05);
  const double revisit_weight =
      node.declare_parameter<double>("score_weights.revisit", 0.05);
  const std::int64_t maximum_replans = node.declare_parameter<std::int64_t>(
      "maximum_replans_per_candidate", 2);
  const std::int64_t maximum_candidate_retryable_retries =
      node.declare_parameter<std::int64_t>(
          "maximum_candidate_retryable_retries", 1);
  const double yaw_tolerance_deg =
      node.declare_parameter<double>("goal_yaw_tolerance_deg", 11.25);
  const double planner_goal_response_timeout_s =
      node.declare_parameter<double>("planner_goal_response_timeout_s", 1.0);
  const double planner_timeout_s =
      node.declare_parameter<double>("planner_result_timeout_s", 3.5);
  const bool stop_before_planning =
      node.declare_parameter<bool>("stop_before_planning", true);
  const double stationary_linear_speed_mps =
      node.declare_parameter<double>("stationary_linear_speed_mps", 0.01);
  const double stationary_angular_speed_radps =
      node.declare_parameter<double>("stationary_angular_speed_radps", 0.02);
  const std::int64_t stationary_confirmation_samples =
      node.declare_parameter<std::int64_t>(
          "stationary_confirmation_samples", 3);
  const double stop_wait_diagnostic_s =
      node.declare_parameter<double>("stop_wait_diagnostic_s", 5.0);

  if (threshold < 0 || threshold > 100 || maximum_task_raster_cells <= 0 ||
      yaw_offsets_deg.size() != 5U || maximum_replans < 0 ||
      maximum_replans > std::numeric_limits<std::uint8_t>::max() ||
      maximum_candidate_retryable_retries < 0 ||
      maximum_candidate_retryable_retries >
          std::numeric_limits<std::uint8_t>::max() ||
      !std::isfinite(sensor_range_m) || sensor_range_m <= 0.0 ||
      !std::isfinite(sensor_fov_deg) || sensor_fov_deg <= 0.0 ||
      !std::isfinite(yaw_tolerance_deg) || yaw_tolerance_deg <= 0.0 ||
      !std::isfinite(planner_goal_response_timeout_s) ||
      planner_goal_response_timeout_s <= 0.0 ||
      !std::isfinite(planner_timeout_s) || planner_timeout_s <= 0.0 ||
      !std::isfinite(stationary_linear_speed_mps) ||
      stationary_linear_speed_mps < 0.0 ||
      !std::isfinite(stationary_angular_speed_radps) ||
      stationary_angular_speed_radps < 0.0 ||
      stationary_confirmation_samples <= 0 ||
      static_cast<std::uint64_t>(stationary_confirmation_samples) >
          std::numeric_limits<std::uint32_t>::max() ||
      !std::isfinite(stop_wait_diagnostic_s) ||
      stop_wait_diagnostic_s <= 0.0) {
    throw std::invalid_argument{"invalid exploration parameters"};
  }
  lunar::pure_exploration::CandidateParameters candidate_parameters;
  for (std::size_t index = 0U; index < yaw_offsets_deg.size(); ++index) {
    if (!std::isfinite(yaw_offsets_deg[index])) {
      throw std::invalid_argument{"yaw offsets must be finite"};
    }
    candidate_parameters.yaw_offsets_rad[index] =
        yaw_offsets_deg[index] * std::numbers::pi / 180.0;
  }

  return ExplorationNodeParameters{
      .platform = std::move(platform),
      .candidate_parameters = candidate_parameters,
      .candidate_limits =
          {PositiveSizeParameter(node, "maximum_position_probes"),
           PositiveSizeParameter(node, "maximum_candidate_views"),
           PositiveSizeParameter(node, "maximum_collision_work_units")},
      .task_raster_limits =
          {static_cast<std::size_t>(maximum_task_raster_cells)},
      .sensor_model =
          {sensor_range_m, sensor_fov_deg * std::numbers::pi / 180.0},
      .information_gain_limits =
          {PositiveSizeParameter(node, "maximum_visibility_work_units")},
      .score_weights = {information_weight, path_weight, heading_weight,
                        revisit_weight},
      .failure_memory_limits =
          {PositiveSizeParameter(node, "maximum_failure_entries"),
           PositiveSizeParameter(node,
                                 "maximum_failure_patch_cells_per_entry"),
           PositiveSizeParameter(node,
                                 "maximum_failure_total_patch_cells")},
      .global_occupied_threshold = static_cast<std::int8_t>(threshold),
      .maximum_path_preview_poses =
          PositiveSizeParameter(node, "maximum_path_preview_poses"),
      .maximum_executable_path_points =
          PositiveSizeParameter(node, "maximum_executable_path_points"),
      .maximum_replans = static_cast<std::uint8_t>(maximum_replans),
      .maximum_candidate_retryable_retries =
          static_cast<std::uint8_t>(maximum_candidate_retryable_retries),
      .goal_yaw_tolerance_rad =
          yaw_tolerance_deg * std::numbers::pi / 180.0,
      .stop_before_planning = stop_before_planning,
      .stationary_gate =
          {.maximum_linear_speed_mps = stationary_linear_speed_mps,
           .maximum_angular_speed_radps = stationary_angular_speed_radps,
           .confirmation_samples = static_cast<std::uint32_t>(
               stationary_confirmation_samples),
           .diagnostic_period = std::chrono::duration_cast<
               std::chrono::steady_clock::duration>(
               std::chrono::duration<double>{stop_wait_diagnostic_s})},
      .planner_goal_response_timeout = std::chrono::duration_cast<
          std::chrono::steady_clock::duration>(
          std::chrono::duration<double>{planner_goal_response_timeout_s}),
      .planner_result_timeout = std::chrono::duration_cast<
          std::chrono::steady_clock::duration>(
          std::chrono::duration<double>{planner_timeout_s}),
      .global_map_topic = AbsoluteTopicParameter(
          node, "global_map_topic", "/Car/T3/mapping/global_overview"),
      .odometry_topic = AbsoluteTopicParameter(
          node, "odometry_topic", "/Car/T3/localization/odometry"),
      .tf_topic = AbsoluteTopicParameter(node, "tf_topic", "/tf"),
      .task_topic = AbsoluteTopicParameter(
          node, "task_topic", "/Car/T4/exploration/task"),
      .planner_action = AbsoluteTopicParameter(
          node, "planner_action", "/Car/T4/plan_motion"),
      .planner_diagnostics_topic = AbsoluteTopicParameter(
          node, "planner_diagnostics_topic",
          "/Car/T4/planning/diagnostics"),
      .motion_reference_topic = AbsoluteTopicParameter(
          node, "motion_reference_topic",
          "/Car/T4/execution/motion_reference"),
      .execution_cancel_topic = AbsoluteTopicParameter(
          node, "execution_cancel_topic", "/Car/T4/execution/cancel"),
      .status_topic = "/Car/T4/exploration/status",
      .current_goal_topic = "/Car/T4/exploration/current_goal",
      .frontiers_topic = "/Car/T4/exploration/frontiers",
      .diagnostics_topic = "/Car/T4/exploration/diagnostics"};
}

}  // namespace

bool FrozenGlobalMapContent::EqualsGeometryAndData(
    const nav_msgs::msg::OccupancyGrid& latest) const {
  const FrozenGlobalMapContent candidate = FreezeGlobalMap(latest, 50);
  return SameGeometry(geometry, candidate.geometry) && data == candidate.data;
}

FrozenCandidateBatch::FrozenCandidateBatch(
    FrozenGlobalMapContent global_map_content,
    std::vector<lunar::pure_exploration::FrontierCluster> frontiers,
    std::vector<lunar::pure_exploration::CandidateView> candidates)
    : global_map_content_((ValidateBatchContent(global_map_content, frontiers,
                                                candidates),
                           std::move(global_map_content))),
      frontiers_(std::move(frontiers)),
      candidates_(std::move(candidates)),
      latest_request_by_candidate_(candidates_.size()) {}

std::span<const lunar::pure_exploration::FrontierCluster>
FrozenCandidateBatch::frontiers() const {
  return frontiers_;
}

std::span<const lunar::pure_exploration::CandidateView>
FrozenCandidateBatch::candidates() const {
  return candidates_;
}

void FrozenCandidateBatch::RegisterRequest(std::string request_id,
                                           const std::size_t candidate_index) {
  if (request_id.empty()) {
    throw std::invalid_argument{"request id must be nonempty"};
  }
  if (candidate_index >= candidates_.size()) {
    throw std::out_of_range{"request candidate index is out of range"};
  }
  if (request_to_candidate_index_.contains(request_id)) {
    throw std::invalid_argument{"request id must be unique"};
  }
  const auto [inserted, was_inserted] =
      request_to_candidate_index_.emplace(std::move(request_id),
                                          candidate_index);
  if (!was_inserted) {
    throw std::logic_error{"request registration lost uniqueness"};
  }
  try {
    latest_request_by_candidate_[candidate_index] = inserted->first;
  } catch (...) {
    request_to_candidate_index_.erase(inserted);
    throw;
  }
}

std::optional<std::size_t> FrozenCandidateBatch::CandidateIndexForRequest(
    const std::string_view request_id) const {
  const auto found = request_to_candidate_index_.find(std::string{request_id});
  if (found == request_to_candidate_index_.end()) {
    return std::nullopt;
  }
  return found->second;
}

std::optional<std::string>
FrozenCandidateBatch::LatestRequestIdForCandidate(
    const std::size_t candidate_index) const {
  if (candidate_index >= latest_request_by_candidate_.size()) {
    return std::nullopt;
  }
  return latest_request_by_candidate_[candidate_index];
}

bool FrozenCandidateBatch::GlobalMapContentEquals(
    const nav_msgs::msg::OccupancyGrid& latest) const {
  return global_map_content_.EqualsGeometryAndData(latest);
}

FrozenPlanningCycle::FrozenPlanningCycle(
    FrozenCandidateBatchPtr batch, const Pose2 frozen_robot_pose,
    const double frozen_resolution_m,
    std::vector<CandidateGain> complete_gains,
    std::vector<lunar::pure_exploration::RankedCandidate> coarse_order)
    : batch_(std::move(batch)),
      frozen_robot_pose_(frozen_robot_pose),
      frozen_resolution_m_(frozen_resolution_m),
      complete_gains_(std::move(complete_gains)),
      coarse_order_(std::move(coarse_order)) {
  if (!batch_ || !Finite(frozen_robot_pose_) ||
      !std::isfinite(frozen_resolution_m_) || frozen_resolution_m_ <= 0.0) {
    throw std::invalid_argument{"invalid frozen planning cycle context"};
  }
  const std::size_t candidate_count = batch_->candidates().size();
  if (complete_gains_.size() != candidate_count) {
    throw std::invalid_argument{"complete gains must cover all candidates"};
  }
  std::vector<double> gain_by_index(candidate_count, 0.0);
  std::vector<bool> gain_seen(candidate_count, false);
  std::size_t positive_count = 0U;
  for (const auto& gain : complete_gains_) {
    if (gain.candidate_index >= candidate_count ||
        gain_seen[gain.candidate_index] ||
        !std::isfinite(gain.information_gain_m2) ||
        gain.information_gain_m2 < 0.0) {
      throw std::invalid_argument{"invalid complete gain authority"};
    }
    gain_seen[gain.candidate_index] = true;
    gain_by_index[gain.candidate_index] = gain.information_gain_m2;
    if (gain.information_gain_m2 > 0.0) {
      ++positive_count;
    }
  }
  if (std::ranges::find(gain_seen, false) != gain_seen.end() ||
      coarse_order_.size() != positive_count) {
    throw std::invalid_argument{"coarse order must cover positive gains"};
  }
  std::vector<bool> order_seen(candidate_count, false);
  for (const auto& row : coarse_order_) {
    if (row.candidate_index >= candidate_count ||
        order_seen[row.candidate_index] || !FiniteRanked(row) ||
        gain_by_index[row.candidate_index] <= 0.0 ||
        row.information_gain_m2 != gain_by_index[row.candidate_index]) {
      throw std::invalid_argument{"invalid coarse order authority"};
    }
    order_seen[row.candidate_index] = true;
  }
  for (std::size_t index = 0U; index < candidate_count; ++index) {
    if ((gain_by_index[index] > 0.0) != order_seen[index]) {
      throw std::invalid_argument{"coarse order omitted a positive gain"};
    }
  }
}

const FrozenCandidateBatchPtr& FrozenPlanningCycle::batch() const {
  return batch_;
}

Pose2 FrozenPlanningCycle::frozen_robot_pose() const {
  return frozen_robot_pose_;
}

double FrozenPlanningCycle::frozen_resolution_m() const {
  return frozen_resolution_m_;
}

std::span<const CandidateGain> FrozenPlanningCycle::complete_gains() const {
  return complete_gains_;
}

std::span<const lunar::pure_exploration::RankedCandidate>
FrozenPlanningCycle::coarse_order() const {
  return coarse_order_;
}

std::size_t FrozenPlanningCycle::coarse_cursor() const {
  return coarse_cursor_;
}

std::vector<std::size_t> FrozenPlanningCycle::TakeNextCandidateIndices() {
  const std::size_t end = std::min(coarse_cursor_ + 16U,
                                   coarse_order_.size());
  std::vector<std::size_t> result;
  result.reserve(end - coarse_cursor_);
  for (; coarse_cursor_ < end; ++coarse_cursor_) {
    result.push_back(coarse_order_[coarse_cursor_].candidate_index);
  }
  return result;
}

void FrozenPlanningCycle::AddReachable(FrozenReachableCandidate result) {
  if (result.metrics.candidate_index >= batch_->candidates().size()) {
    throw std::out_of_range{"reachable candidate index is out of range"};
  }
  if (!std::isfinite(result.metrics.path_length_m) ||
      result.metrics.path_length_m < 0.0) {
    throw std::invalid_argument{"reachable path length is invalid"};
  }
  if (std::ranges::any_of(reachable_, [&](const auto& row) {
        return row.metrics.candidate_index == result.metrics.candidate_index;
      })) {
    throw std::invalid_argument{"reachable candidate is duplicated"};
  }
  reachable_.push_back(std::move(result));
}

std::span<const FrozenReachableCandidate> FrozenPlanningCycle::reachable()
    const {
  return reachable_;
}

FrozenApproachCycle::FrozenApproachCycle(
    FrozenGlobalMapContent global_map_content,
    lunar::pure_exploration::BoundaryGuidanceResult guidance,
    const Pose2 frozen_robot_pose, const double frozen_resolution_m,
    std::vector<std::size_t> coarse_order)
    : global_map_content_((static_cast<void>(OccupancyGridView(
                               global_map_content.geometry,
                               global_map_content.data, 50)),
                           std::move(global_map_content))),
      guidance_(std::move(guidance)),
      frozen_robot_pose_(frozen_robot_pose),
      frozen_resolution_m_(frozen_resolution_m),
      coarse_order_(std::move(coarse_order)),
      latest_request_by_candidate_(guidance_.candidates.size()) {
  if (!Finite(frozen_robot_pose_) || !std::isfinite(frozen_resolution_m_) ||
      frozen_resolution_m_ <= 0.0 ||
      guidance_.phase !=
          lunar::pure_exploration::NavigationPhase::kApproachTask ||
      coarse_order_.size() != guidance_.candidates.size()) {
    throw std::invalid_argument{"invalid frozen approach cycle context"};
  }
  std::vector<bool> seen(guidance_.candidates.size(), false);
  for (const std::size_t index : coarse_order_) {
    if (index >= seen.size() || seen[index]) {
      throw std::invalid_argument{"invalid frozen approach coarse order"};
    }
    seen[index] = true;
  }
}

const lunar::pure_exploration::BoundaryGuidanceResult&
FrozenApproachCycle::guidance() const {
  return guidance_;
}

Pose2 FrozenApproachCycle::frozen_robot_pose() const {
  return frozen_robot_pose_;
}

double FrozenApproachCycle::frozen_resolution_m() const {
  return frozen_resolution_m_;
}

std::span<const std::size_t> FrozenApproachCycle::coarse_order() const {
  return coarse_order_;
}

std::size_t FrozenApproachCycle::coarse_cursor() const {
  return coarse_cursor_;
}

std::vector<std::size_t> FrozenApproachCycle::TakeNextCandidateIndices() {
  const std::size_t end =
      std::min(coarse_cursor_ + 16U, coarse_order_.size());
  std::vector<std::size_t> result;
  result.reserve(end - coarse_cursor_);
  for (; coarse_cursor_ < end; ++coarse_cursor_) {
    result.push_back(coarse_order_[coarse_cursor_]);
  }
  return result;
}

void FrozenApproachCycle::RegisterRequest(
    std::string request_id, const std::size_t candidate_index) {
  if (request_id.empty()) {
    throw std::invalid_argument{"request id must be nonempty"};
  }
  if (candidate_index >= guidance_.candidates.size()) {
    throw std::out_of_range{"request candidate index is out of range"};
  }
  const auto [inserted, was_inserted] =
      request_to_candidate_index_.emplace(std::move(request_id),
                                          candidate_index);
  if (!was_inserted) {
    throw std::invalid_argument{"request id must be unique"};
  }
  try {
    latest_request_by_candidate_[candidate_index] = inserted->first;
  } catch (...) {
    request_to_candidate_index_.erase(inserted);
    throw;
  }
}

std::optional<std::size_t> FrozenApproachCycle::CandidateIndexForRequest(
    const std::string_view request_id) const {
  const auto found = request_to_candidate_index_.find(std::string{request_id});
  return found == request_to_candidate_index_.end()
             ? std::nullopt
             : std::optional<std::size_t>{found->second};
}

std::optional<std::string> FrozenApproachCycle::LatestRequestIdForCandidate(
    const std::size_t candidate_index) const {
  if (candidate_index >= latest_request_by_candidate_.size()) {
    return std::nullopt;
  }
  return latest_request_by_candidate_[candidate_index];
}

void FrozenApproachCycle::AddReachable(FrozenReachableCandidate result) {
  if (result.metrics.candidate_index >= guidance_.candidates.size()) {
    throw std::out_of_range{"reachable candidate index is out of range"};
  }
  if (!std::isfinite(result.metrics.path_length_m) ||
      result.metrics.path_length_m < 0.0) {
    throw std::invalid_argument{"reachable path length is invalid"};
  }
  if (std::ranges::any_of(reachable_, [&](const auto& row) {
        return row.metrics.candidate_index == result.metrics.candidate_index;
      })) {
    throw std::invalid_argument{"reachable candidate is duplicated"};
  }
  reachable_.push_back(std::move(result));
}

std::span<const FrozenReachableCandidate> FrozenApproachCycle::reachable()
    const {
  return reachable_;
}

bool FrozenApproachCycle::GlobalMapContentEquals(
    const nav_msgs::msg::OccupancyGrid& latest) const {
  return global_map_content_.EqualsGeometryAndData(latest);
}

namespace {

ExplorationNodeParameters NormalizeApproachLimits(
    ExplorationNodeParameters parameters) {
  auto& limits = parameters.boundary_guidance_limits;
  if (limits.maximum_guidance_grid_cells == 0U) {
    limits.maximum_guidance_grid_cells =
        parameters.task_raster_limits.maximum_raster_cell_count;
  }
  if (limits.maximum_guidance_work_units == 0U) {
    if (limits.maximum_guidance_grid_cells >
        std::numeric_limits<std::size_t>::max() / 8U) {
      throw std::overflow_error{
          "boundary guidance default work limit overflow"};
    }
    limits.maximum_guidance_work_units =
        8U * limits.maximum_guidance_grid_cells;
  }
  if (limits.maximum_approach_candidates == 0U) {
    limits.maximum_approach_candidates =
        parameters.candidate_limits.maximum_candidate_views;
  }
  return parameters;
}

}  // namespace

struct ExplorationNode::Runtime final {
  enum class PendingControl { kNone, kPause, kCancel, kReplacementStart };
  enum class PendingStationaryAction : std::uint8_t {
    kNone,
    kBuildFreshBatch,
    kReplanActiveGoal,
  };

  struct PendingStart {
    std::string task_id;
    Polygon2 boundary;
  };

  struct StopWaitRecord final {
    SpeedObservation entry_speed{};
    std::optional<SpeedObservation> confirmed_speed;
    std::uint32_t confirmation_samples{};
    std::chrono::steady_clock::duration elapsed{};
  };

  enum class PlannerTimingBucket : std::uint8_t {
    kCandidate,
    kRolling,
    kStuck,
  };
  struct PlannerTimingProvenance {
    std::uint64_t task_generation;
    PlannerTimingBucket bucket;
  };
  struct PlannerTimingSummary {
    std::uint64_t call_count{0U};
    std::uint64_t record_count{0U};
    std::optional<PlannerTiming> last;
    double global_elapsed_ms{0.0};
    std::uint64_t global_call_count{0U};
    double local_elapsed_ms{0.0};
    std::uint64_t local_call_count{0U};
    double total_elapsed_ms{0.0};
  };
  struct ConsumedPlannerTiming {
    std::uint64_t task_generation;
    PlannerTimingBucket bucket;
    PlannerTiming timing;
  };
  struct SnapshotToGoalStart {
    std::weak_ptr<FrozenPlanningCycle> cycle;
    std::weak_ptr<FrozenApproachCycle> approach_cycle;
    std::uint64_t epoch;
    std::uint64_t build_generation;
    std::chrono::steady_clock::time_point start;
  };
  struct LocalTiming {
    std::uint64_t call_count{0U};
    double last_elapsed_ms{0.0};
    double accumulated_elapsed_ms{0.0};
  };
  struct BlockedApproachSnapshot final {
    FrozenGlobalMapContent map;
    std::uint64_t pose_generation;
    std::uint64_t task_generation;
  };
  struct ApproachObservabilitySnapshot final {
    ApproachMarkers markers;
    std::optional<double> remaining_guidance_m;
    std::optional<std::uint32_t> guidance_unknown_cell_count;
    std::optional<std::string> wait_reason;
  };

  Runtime(rclcpp::Node& node, ExplorationNodeParameters input)
      : parameters(NormalizeApproachLimits(std::move(input))),
        stationary_gate(parameters.stationary_gate),
        candidate_generator(parameters.platform,
                            parameters.candidate_parameters,
                            parameters.candidate_limits),
        boundary_guidance(parameters.platform, parameters.sensor_model,
                          parameters.goal_yaw_tolerance_rad,
                          parameters.boundary_guidance_limits),
        information_gain(parameters.sensor_model,
                         parameters.information_gain_limits),
        ranker(parameters.score_weights,
               candidate_generator.platform_length_m()),
        failure_memory(candidate_generator.platform_length_m(),
                       parameters.failure_memory_limits),
        state_machine(parameters.maximum_replans),
        timing(parameters.candidate_limits.maximum_candidate_views),
        clock(node.get_clock()),
        reference_publisher(
            node.create_publisher<lunar_planning_msgs::msg::MotionReference>(
                parameters.motion_reference_topic,
                rclcpp::QoS{1}.reliable())),
        cancel_publisher(node.create_publisher<std_msgs::msg::String>(
            parameters.execution_cancel_topic,
            rclcpp::QoS{10}.reliable())),
        status_publisher(node.create_publisher<Status>(
            parameters.status_topic,
            rclcpp::QoS{10}.reliable().transient_local())),
        current_goal_publisher(
            node.create_publisher<geometry_msgs::msg::PoseStamped>(
                parameters.current_goal_topic,
                rclcpp::QoS{1}.reliable().transient_local())),
        frontiers_publisher(
            node.create_publisher<visualization_msgs::msg::MarkerArray>(
                parameters.frontiers_topic,
                rclcpp::QoS{1}.reliable().transient_local())),
        diagnostics_publisher(node.create_publisher<DiagnosticArray>(
            parameters.diagnostics_topic,
            rclcpp::QoS{1}.reliable().transient_local())),
        planner_client(std::make_unique<PlannerClient>(
            node, parameters.planner_action,
            PlannerClientParameters{
                .maximum_path_preview_poses =
                    parameters.maximum_path_preview_poses,
                .maximum_executable_path_points =
                    parameters.maximum_executable_path_points,
                .goal_response_timeout =
                    parameters.planner_goal_response_timeout,
                .result_timeout = parameters.planner_result_timeout},
            parameters.steady_now
                ? parameters.steady_now
                : PlannerClient::SteadyNow{
                      [] { return std::chrono::steady_clock::now(); }})),
        work_queue(std::make_shared<SerializedWorkQueue>()),
        map_validation_queue(std::make_shared<SerializedWorkQueue>()) {}

  std::mutex mutex;
  bool teardown{false};
  ExplorationNodeParameters parameters;
  StationaryPlanningGate stationary_gate;
  SpeedObservation latest_speed{};
  PendingStationaryAction pending_stationary_action{
      PendingStationaryAction::kNone};
  std::optional<lunar::pure_exploration::GoalReleaseReason>
      pending_fresh_batch_release_reason;
  std::optional<StopWaitRecord> current_stop_wait;
  std::optional<StopWaitRecord> last_stop_wait;
  std::uint64_t planning_epoch{0U};
  CandidateGenerator candidate_generator;
  BoundaryGuidance boundary_guidance;
  InformationGainEvaluator information_gain;
  CandidateRanker ranker;
  lunar::pure_exploration::FailureMemory failure_memory;
  lunar::pure_exploration::ExplorationStateMachine state_machine;
  PlannerTimingAccumulator timing;
  PoseResolver pose_resolver;
  std::optional<FrozenGlobalMapContent> latest_map;
  std::optional<nav_msgs::msg::OccupancyGrid> latest_map_message;
  std::optional<Polygon2> task_boundary;
  std::optional<TaskRaster> active_raster;
  CoverageStats coverage{};
  std::size_t frontier_count{0U};
  std::size_t candidate_count{0U};
  NavigationPhase navigation_phase{NavigationPhase::kExploreTask};
  bool fully_inside_task{false};
  std::optional<std::string> status_reason_override;
  std::optional<BlockedApproachSnapshot> blocked_approach_snapshot;
  std::optional<ApproachObservabilitySnapshot> approach_observability;
  FrozenPlanningCyclePtr active_cycle;
  FrozenApproachCyclePtr active_approach_cycle;
  std::vector<std::size_t> current_batch;
  std::size_t current_batch_cursor{0U};
  std::optional<std::size_t> current_candidate;
  std::uint8_t current_candidate_retryable_retries{0U};
  bool planner_in_flight{false};
  bool build_in_flight{false};
  bool final_rank_in_flight{false};
  std::uint64_t epoch{0U};
  std::uint64_t build_generation{0U};
  std::uint64_t map_generation{0U};
  std::uint64_t pose_generation{0U};
  std::uint64_t task_generation{0U};
  std::uint64_t request_sequence{0U};
  double active_elapsed_s{0.0};
  std::optional<std::chrono::steady_clock::time_point> active_elapsed_start;
  MarkerBuilder marker_builder;
  std::deque<std::string> timing_provenance_order;
  std::unordered_map<std::string, PlannerTimingProvenance> timing_provenance;
  std::deque<std::string> consumed_timing_order;
  std::unordered_map<std::string, ConsumedPlannerTiming> consumed_timing;
  std::array<PlannerTimingSummary, 3U> planner_timing{};
  LocalTiming frontier_detection_timing;
  LocalTiming information_gain_timing;
  LocalTiming coarse_rank_timing;
  LocalTiming final_rank_timing;
  LocalTiming snapshot_to_goal_timing;
  std::optional<SnapshotToGoalStart> snapshot_to_goal_start;
  std::optional<std::string> execution_replan_request_id;
  std::optional<PlannerTimingBucket> execution_replan_bucket;
  PendingControl pending_control{PendingControl::kNone};
  std::optional<PendingStart> pending_start;
  bool pending_execution_cancel_sent{false};
  std::optional<std::string> deferred_execution_cancel;
  std::optional<std::chrono::steady_clock::time_point>
      candidate_retry_deadline;
  bool execution_replan_waiting_terminal{false};
  bool execution_replan_retry_pending{false};
  bool pending_map_rebuild{false};
  bool pending_stuck_rebuild{false};
  std::vector<Vec2> completed_goal_positions;
  std::optional<lunar_planning_msgs::msg::MotionReference> active_reference;
  std::vector<Vec2> active_executable_polyline;
  std::optional<Vec2> active_executable_endpoint;
  std::optional<ExecutionMonitor> execution_monitor;
  rclcpp::Clock::SharedPtr clock;
  rclcpp::Publisher<lunar_planning_msgs::msg::MotionReference>::SharedPtr
      reference_publisher;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cancel_publisher;
  rclcpp::Publisher<Status>::SharedPtr status_publisher;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      current_goal_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      frontiers_publisher;
  rclcpp::Publisher<DiagnosticArray>::SharedPtr diagnostics_publisher;
  std::unique_ptr<PlannerClient> planner_client;
  std::shared_ptr<SerializedWorkQueue> work_queue;
  std::shared_ptr<SerializedWorkQueue> map_validation_queue;
};

namespace {

template <typename RuntimeT>
void FreezeActiveElapsedLocked(RuntimeT& runtime);

template <typename RuntimeT>
std::optional<std::string> ExecutionCancelLocked(RuntimeT& runtime);

template <typename RuntimeT>
std::chrono::steady_clock::time_point SteadyNowLocked(RuntimeT& runtime);

template <typename RuntimeT>
bool PlanningIsAdmittedLocked(const RuntimeT& runtime) {
  return runtime.pending_stationary_action ==
         RuntimeT::PendingStationaryAction::kNone;
}

template <typename RuntimeT>
std::string StatusReasonLocked(const RuntimeT& runtime) {
  if (runtime.pending_stationary_action !=
      RuntimeT::PendingStationaryAction::kNone) {
    return "WAITING_FOR_STOP";
  }
  return runtime.status_reason_override
             ? *runtime.status_reason_override
             : runtime.state_machine.reason_code();
}

template <typename RuntimeT>
std::optional<std::string> RequestStationaryActionLocked(
    RuntimeT& runtime,
    const typename RuntimeT::PendingStationaryAction action) {
  if (!runtime.parameters.stop_before_planning) {
    runtime.stationary_gate.Cancel();
    runtime.pending_stationary_action =
        RuntimeT::PendingStationaryAction::kNone;
    runtime.current_stop_wait.reset();
    return std::nullopt;
  }
  if (runtime.pending_stationary_action == action) {
    return std::nullopt;
  }
  runtime.pending_stationary_action = action;
  runtime.stationary_gate.Begin(SteadyNowLocked(runtime),
                                runtime.latest_speed);
  runtime.current_stop_wait.emplace(typename RuntimeT::StopWaitRecord{
      .entry_speed = runtime.latest_speed,
      .confirmed_speed = std::nullopt,
      .confirmation_samples = 0U,
      .elapsed = std::chrono::steady_clock::duration::zero(),
  });
  return ExecutionCancelLocked(runtime);
}

template <typename RuntimeT>
typename RuntimeT::PendingStationaryAction
ConsumeConfirmedStationaryActionLocked(RuntimeT& runtime) {
  const auto action = runtime.pending_stationary_action;
  runtime.pending_stationary_action =
      RuntimeT::PendingStationaryAction::kNone;
  return action;
}

template <typename RuntimeT>
void CancelPendingStationaryActionLocked(RuntimeT& runtime) {
  runtime.stationary_gate.Cancel();
  runtime.pending_stationary_action =
      RuntimeT::PendingStationaryAction::kNone;
  runtime.pending_fresh_batch_release_reason.reset();
  runtime.current_stop_wait.reset();
}

template <typename RuntimeT>
void ResetStopWaitTaskLifetimeLocked(RuntimeT& runtime) {
  runtime.current_stop_wait.reset();
  runtime.last_stop_wait.reset();
  runtime.planning_epoch = 0U;
}

template <typename RuntimeT>
void ClearExecutionPayloadsLocked(RuntimeT& runtime) {
  runtime.active_reference.reset();
  runtime.active_executable_polyline.clear();
  runtime.active_executable_endpoint.reset();
  runtime.execution_monitor.reset();
}

template <typename RuntimeT>
void ClearApproachSelectionLocked(RuntimeT& runtime) {
  if (!runtime.approach_observability) {
    return;
  }
  runtime.approach_observability->markers.selected_identity.reset();
  runtime.approach_observability->markers.selected_guidance.clear();
  runtime.approach_observability->remaining_guidance_m.reset();
  runtime.approach_observability->guidance_unknown_cell_count.reset();
}

template <typename RuntimeT>
void ClearSelectionCycleLocked(RuntimeT& runtime) {
  ClearApproachSelectionLocked(runtime);
  runtime.active_cycle.reset();
  runtime.active_approach_cycle.reset();
  runtime.active_raster.reset();
  runtime.current_batch.clear();
  runtime.current_batch_cursor = 0U;
  runtime.current_candidate.reset();
  runtime.current_candidate_retryable_retries = 0U;
  runtime.candidate_retry_deadline.reset();
  runtime.final_rank_in_flight = false;
  runtime.snapshot_to_goal_start.reset();
  runtime.blocked_approach_snapshot.reset();
  runtime.status_reason_override.reset();
  runtime.navigation_phase = NavigationPhase::kExploreTask;
  runtime.fully_inside_task = false;
}

enum class PlannerTerminalRebuild {
  kNone,
  kAwaitingStop,
  kBuildNow,
};

template <typename RuntimeT>
PlannerTerminalRebuild ResolvePendingRebuildAfterPlannerTerminalLocked(
    RuntimeT& runtime) {
  if (!runtime.pending_map_rebuild && !runtime.pending_stuck_rebuild) {
    return PlannerTerminalRebuild::kNone;
  }
  runtime.pending_map_rebuild = false;
  runtime.pending_stuck_rebuild = false;
  if (runtime.pending_stationary_action !=
      RuntimeT::PendingStationaryAction::kNone) {
    return PlannerTerminalRebuild::kAwaitingStop;
  }
  ClearExecutionPayloadsLocked(runtime);
  ClearSelectionCycleLocked(runtime);
  ++runtime.epoch;
  ++runtime.build_generation;
  return PlannerTerminalRebuild::kBuildNow;
}

struct MapInvalidationDisposition final {
  bool cancel_planner{false};
  bool build_now{false};
  std::optional<std::string> execution_cancel;
};

template <typename RuntimeT>
MapInvalidationDisposition InvalidateActiveGoalForMapLocked(
    RuntimeT& runtime) {
  MapInvalidationDisposition result;
  runtime.execution_replan_retry_pending = false;
  if (runtime.parameters.stop_before_planning) {
    if (runtime.state_machine.state() == ExplorationState::kExecuting) {
      static_cast<void>(runtime.state_machine.BeginReplanning(
          lunar::pure_exploration::ReplanCause::kRollingSegment));
    }
    runtime.pending_fresh_batch_release_reason =
        lunar::pure_exploration::GoalReleaseReason::kCandidateInvalid;
    result.execution_cancel = RequestStationaryActionLocked(
        runtime, RuntimeT::PendingStationaryAction::kBuildFreshBatch);
    if (runtime.planner_in_flight) {
      runtime.pending_map_rebuild = true;
      result.cancel_planner = true;
    }
    return result;
  }

  result.execution_cancel = ExecutionCancelLocked(runtime);
  runtime.state_machine.ReleaseGoal(
      lunar::pure_exploration::GoalReleaseReason::kCandidateInvalid);
  ClearExecutionPayloadsLocked(runtime);
  if (runtime.planner_in_flight) {
    runtime.pending_map_rebuild = true;
    result.cancel_planner = true;
  } else {
    ClearSelectionCycleLocked(runtime);
    ++runtime.epoch;
    result.build_now = true;
  }
  return result;
}

template <typename RuntimeT>
void FailLocked(RuntimeT& runtime, std::string reason) {
  if (const auto execution_cancel = ExecutionCancelLocked(runtime)) {
    runtime.deferred_execution_cancel = *execution_cancel;
  }
  FreezeActiveElapsedLocked(runtime);
  runtime.state_machine.Fail(std::move(reason));
  ClearSelectionCycleLocked(runtime);
  runtime.approach_observability.reset();
  runtime.execution_replan_request_id.reset();
  runtime.execution_replan_bucket.reset();
  runtime.pending_control = RuntimeT::PendingControl::kNone;
  runtime.pending_start.reset();
  runtime.candidate_retry_deadline.reset();
  runtime.execution_replan_waiting_terminal = false;
  runtime.execution_replan_retry_pending = false;
  runtime.pending_map_rebuild = false;
  runtime.pending_stuck_rebuild = false;
  CancelPendingStationaryActionLocked(runtime);
  ResetStopWaitTaskLifetimeLocked(runtime);
  ClearExecutionPayloadsLocked(runtime);
  runtime.build_in_flight = false;
  ++runtime.epoch;
  ++runtime.build_generation;
}

template <typename RuntimeT>
std::chrono::steady_clock::time_point SteadyNowLocked(RuntimeT& runtime) {
  return runtime.parameters.steady_now ? runtime.parameters.steady_now()
                                       : std::chrono::steady_clock::now();
}

template <typename RuntimeT>
void ResetActiveElapsedLocked(RuntimeT& runtime) {
  runtime.active_elapsed_s = 0.0;
  runtime.active_elapsed_start = SteadyNowLocked(runtime);
}

template <typename RuntimeT>
void FreezeActiveElapsedLocked(RuntimeT& runtime) {
  if (!runtime.active_elapsed_start) {
    return;
  }
  const auto now = SteadyNowLocked(runtime);
  if (now >= *runtime.active_elapsed_start) {
    const double seconds =
        std::chrono::duration<double>(now - *runtime.active_elapsed_start)
            .count();
    if (std::isfinite(seconds) && seconds >= 0.0 &&
        std::isfinite(runtime.active_elapsed_s + seconds)) {
      runtime.active_elapsed_s += seconds;
    }
  }
  // A backwards injected steady clock never subtracts or overflows elapsed
  // time.  It fail-closes to the already committed value.
  runtime.active_elapsed_start.reset();
}

template <typename RuntimeT>
double ActiveElapsedLocked(RuntimeT& runtime) {
  const auto state = runtime.state_machine.state();
  const bool frozen = state == ExplorationState::kPaused ||
                      state == ExplorationState::kCompleted ||
                      state == ExplorationState::kError ||
                      runtime.pending_control == RuntimeT::PendingControl::kPause ||
                      runtime.state_machine.task_id().empty();
  if (frozen) {
    FreezeActiveElapsedLocked(runtime);
    return runtime.active_elapsed_s;
  }
  if (!runtime.active_elapsed_start) {
    runtime.active_elapsed_start = SteadyNowLocked(runtime);
    return runtime.active_elapsed_s;
  }
  const auto now = SteadyNowLocked(runtime);
  if (now < *runtime.active_elapsed_start) {
    return runtime.active_elapsed_s;
  }
  const double seconds =
      std::chrono::duration<double>(now - *runtime.active_elapsed_start).count();
  return std::isfinite(seconds) && seconds >= 0.0 &&
                 std::isfinite(runtime.active_elapsed_s + seconds)
             ? runtime.active_elapsed_s + seconds
             : runtime.active_elapsed_s;
}

template <typename RuntimeT>
void RegisterPlannerTimingRequestLocked(
    RuntimeT& runtime, const std::string& request_id,
    const typename RuntimeT::PlannerTimingBucket bucket) {
  auto& summary = runtime.planner_timing[static_cast<std::size_t>(bucket)];
  if (summary.call_count == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"planner timing call count overflow"};
  }
  ++summary.call_count;
  const std::size_t capacity = runtime.parameters.candidate_limits.maximum_candidate_views;
  while (runtime.timing_provenance_order.size() >= capacity) {
    runtime.timing_provenance.erase(runtime.timing_provenance_order.front());
    runtime.timing_provenance_order.pop_front();
  }
  const auto [_, inserted] = runtime.timing_provenance.emplace(
      request_id, typename RuntimeT::PlannerTimingProvenance{
                      .task_generation = runtime.task_generation,
                      .bucket = bucket});
  if (!inserted) {
    throw std::logic_error{"planner timing request id was reused"};
  }
  runtime.timing_provenance_order.push_back(request_id);
}

template <typename RuntimeT>
void ConsumePlannerTimingLocked(RuntimeT& runtime,
                                const std::string& request_id) {
  const auto provenance = runtime.timing_provenance.find(request_id);
  const auto timing = runtime.timing.Find(request_id);
  if (provenance == runtime.timing_provenance.end() || !timing ||
      provenance->second.task_generation != runtime.task_generation) {
    return;
  }
  auto& summary = runtime.planner_timing[static_cast<std::size_t>(
      provenance->second.bucket)];
  if (!std::isfinite(summary.global_elapsed_ms + timing->global_elapsed_ms) ||
      !std::isfinite(summary.local_elapsed_ms + timing->local_elapsed_ms) ||
      !std::isfinite(summary.total_elapsed_ms + timing->total_elapsed_ms) ||
      std::numeric_limits<std::uint64_t>::max() - summary.global_call_count <
          timing->global_call_count ||
      std::numeric_limits<std::uint64_t>::max() - summary.local_call_count <
          timing->local_call_count) {
    return;
  }
  ++summary.record_count;
  summary.last = *timing;
  summary.global_elapsed_ms += timing->global_elapsed_ms;
  summary.global_call_count += timing->global_call_count;
  summary.local_elapsed_ms += timing->local_elapsed_ms;
  summary.local_call_count += timing->local_call_count;
  summary.total_elapsed_ms += timing->total_elapsed_ms;
  while (runtime.consumed_timing_order.size() >=
         runtime.parameters.candidate_limits.maximum_candidate_views) {
    runtime.consumed_timing.erase(runtime.consumed_timing_order.front());
    runtime.consumed_timing_order.pop_front();
  }
  runtime.consumed_timing.emplace(
      request_id, typename RuntimeT::ConsumedPlannerTiming{
                      .task_generation = provenance->second.task_generation,
                      .bucket = provenance->second.bucket,
                      .timing = *timing});
  runtime.consumed_timing_order.push_back(request_id);
  runtime.timing_provenance.erase(provenance);
  runtime.timing_provenance_order.erase(
      std::ranges::find(runtime.timing_provenance_order, request_id));
}

template <typename TimingT>
void RecordLocalTiming(TimingT& target, const double elapsed_ms) {
  if (!std::isfinite(elapsed_ms) || elapsed_ms < 0.0 ||
      target.call_count == std::numeric_limits<std::uint64_t>::max() ||
      !std::isfinite(target.accumulated_elapsed_ms + elapsed_ms)) {
    return;
  }
  ++target.call_count;
  target.last_elapsed_ms = elapsed_ms;
  target.accumulated_elapsed_ms += elapsed_ms;
}

template <typename TargetTimingT, typename SourceTimingT>
void MergeLocalTiming(TargetTimingT& target, const SourceTimingT& source) {
  if (source.call_count == 0U ||
      std::numeric_limits<std::uint64_t>::max() - target.call_count <
          source.call_count ||
      !std::isfinite(target.accumulated_elapsed_ms +
                     source.accumulated_elapsed_ms)) {
    return;
  }
  target.call_count += source.call_count;
  target.last_elapsed_ms = source.last_elapsed_ms;
  target.accumulated_elapsed_ms += source.accumulated_elapsed_ms;
}

template <typename RuntimeT>
double SteadyElapsedMs(RuntimeT& runtime,
                       const std::chrono::steady_clock::time_point start) {
  const auto end = SteadyNowLocked(runtime);
  if (end < start) {
    return -1.0;
  }
  const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
  return std::isfinite(elapsed) ? elapsed : -1.0;
}

template <typename RuntimeT>
Status MakeStatusLocked(RuntimeT& runtime) {
  Status status;
  status.header.stamp = runtime.clock->now();
  status.header.frame_id = "map";
  status.task_id = runtime.state_machine.task_id();
  status.state = StatusState(runtime.state_machine.state());
  status.reason_code = StatusReasonLocked(runtime);
  status.polygon_area_m2 = runtime.coverage.polygon_area_m2;
  status.task_raster_area_m2 = runtime.coverage.task_raster_area_m2;
  status.known_free_area_m2 = runtime.coverage.known_free_area_m2;
  status.known_occupied_area_m2 = runtime.coverage.known_occupied_area_m2;
  status.unknown_area_m2 = runtime.coverage.unknown_area_m2;
  status.outside_map_area_m2 = runtime.coverage.outside_map_area_m2;
  status.coverage_ratio = runtime.coverage.coverage_ratio;
  status.frontier_cluster_count = ToU32(runtime.frontier_count);
  status.candidate_count = ToU32(runtime.candidate_count);
  status.reachable_candidate_count =
      runtime.active_cycle ? ToU32(runtime.active_cycle->reachable().size())
      : runtime.active_approach_cycle
          ? ToU32(runtime.active_approach_cycle->reachable().size())
          : 0U;
  status.failed_candidate_count = runtime.state_machine.task_id().empty()
                                      ? 0U
                                      : ToU32(runtime.failure_memory.size());
  status.completed_goal_count = ToU32(runtime.completed_goal_positions.size());
  if (runtime.state_machine.active_goal()) {
    const auto& goal = *runtime.state_machine.active_goal();
    status.replan_count = goal.replan_count();
    status.current_goal.position.x = goal.target().x;
    status.current_goal.position.y = goal.target().y;
    status.current_goal.orientation.z = std::sin(goal.target().yaw / 2.0);
    status.current_goal.orientation.w = std::cos(goal.target().yaw / 2.0);
  }
  if (runtime.active_reference) {
    status.current_plan_id = runtime.active_reference->plan_id;
  }
  status.active_elapsed_s = ActiveElapsedLocked(runtime);
  return status;
}

template <typename RuntimeT>
struct CurrentPlannerRequest final {
  std::string request_id;
  typename RuntimeT::PlannerTimingBucket bucket;
};

template <typename RuntimeT>
std::optional<CurrentPlannerRequest<RuntimeT>> CurrentPlannerRequestLocked(
    const RuntimeT& runtime) {
  if (runtime.state_machine.state() == ExplorationState::kReplanning &&
      runtime.execution_replan_request_id && runtime.execution_replan_bucket) {
    return CurrentPlannerRequest<RuntimeT>{
        .request_id = *runtime.execution_replan_request_id,
        .bucket = *runtime.execution_replan_bucket};
  }
  if (runtime.state_machine.active_goal()) {
    return CurrentPlannerRequest<RuntimeT>{
        .request_id = runtime.state_machine.active_goal()->request_id(),
        .bucket = RuntimeT::PlannerTimingBucket::kCandidate};
  }
  if (runtime.active_cycle && runtime.current_candidate) {
    const auto request_id = runtime.active_cycle->batch()
                                ->LatestRequestIdForCandidate(
                                    *runtime.current_candidate);
    if (request_id) {
      return CurrentPlannerRequest<RuntimeT>{
          .request_id = *request_id,
          .bucket = RuntimeT::PlannerTimingBucket::kCandidate};
      }
  }
  if (runtime.active_approach_cycle && runtime.current_candidate) {
    const auto request_id =
        runtime.active_approach_cycle->LatestRequestIdForCandidate(
            *runtime.current_candidate);
    if (request_id) {
      return CurrentPlannerRequest<RuntimeT>{
          .request_id = *request_id,
          .bucket = RuntimeT::PlannerTimingBucket::kCandidate};
    }
  }
  return std::nullopt;
}

template <typename RuntimeT>
void PublishStatus(const std::shared_ptr<RuntimeT>& runtime) {
  Status status;
  rclcpp::Publisher<Status>::SharedPtr publisher;
  std::optional<geometry_msgs::msg::PoseStamped> current_goal;
  visualization_msgs::msg::MarkerArray markers;
  DiagnosticArray diagnostics;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      current_goal_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      frontiers_publisher;
  rclcpp::Publisher<DiagnosticArray>::SharedPtr diagnostics_publisher;
  std::optional<std::string> deferred_execution_cancel;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cancel_publisher;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    status = MakeStatusLocked(*runtime);
    publisher = runtime->status_publisher;
    if (runtime->state_machine.active_goal()) {
      const auto& goal = *runtime->state_machine.active_goal();
      geometry_msgs::msg::PoseStamped message;
      message.header = status.header;
      message.pose = status.current_goal;
      current_goal = std::move(message);
    }
    std::span<const lunar::pure_exploration::FrontierCluster> frontiers;
    std::span<const lunar::pure_exploration::CandidateView> candidates;
    std::optional<MarkerSelection> selected;
    const ApproachMarkers* approach = nullptr;
    std::optional<double> remaining_guidance_m;
    std::optional<std::uint32_t> guidance_unknown_cell_count;
    if (runtime->active_cycle) {
      frontiers = runtime->active_cycle->batch()->frontiers();
      candidates = runtime->active_cycle->batch()->candidates();
      if (runtime->state_machine.active_goal() &&
          runtime->state_machine.active_goal()->kind() ==
              lunar::pure_exploration::GoalKind::kTaskFrontier) {
        const auto& goal = *runtime->state_machine.active_goal();
        selected.emplace(MarkerSelection{.candidate_key = goal.candidate_key(),
                                         .frontier_canonical_key =
                                             std::vector<std::int64_t>(
                                                 goal.frontier_canonical_key().begin(),
                                                 goal.frontier_canonical_key().end()),
                                         .target = goal.target()});
      }
    }
    if (runtime->navigation_phase == NavigationPhase::kApproachTask &&
        runtime->approach_observability) {
      approach = &runtime->approach_observability->markers;
      const auto remaining =
          runtime->approach_observability->remaining_guidance_m;
      if (remaining && std::isfinite(*remaining) && *remaining >= 0.0) {
        remaining_guidance_m = remaining;
      }
      guidance_unknown_cell_count =
          runtime->approach_observability->guidance_unknown_cell_count;
    }
    markers =
        runtime->marker_builder.Build(frontiers, candidates, selected, approach);
    for (auto& marker : markers.markers) {
      marker.header = status.header;
    }
    DiagnosticStatus diagnostic;
    diagnostic.name = "pure_exploration";
    diagnostic.hardware_id = "pure_exploration";
    const auto add = [&diagnostic](std::string key, std::string value) {
      diagnostic.values.push_back(
          KeyValue{}.set__key(std::move(key)).set__value(std::move(value)));
    };
    add("task_id", status.task_id);
    add("state", std::to_string(status.state));
    add("reason_code", status.reason_code);
    add("active_candidate_id", runtime->state_machine.active_goal()
                                   ? std::to_string(runtime->state_machine.active_goal()->candidate_id())
                                   : "unavailable");
    const auto current_request = CurrentPlannerRequestLocked(*runtime);
    const auto current_timing = [&]() -> std::optional<PlannerTiming> {
      if (!current_request) {
        return std::nullopt;
      }
      const auto found = runtime->consumed_timing.find(
          current_request->request_id);
      if (found == runtime->consumed_timing.end() ||
          found->second.task_generation != runtime->task_generation ||
          found->second.bucket != current_request->bucket) {
        return std::nullopt;
      }
      return found->second.timing;
    }();
    add("request_id", current_request ? current_request->request_id
                                       : "unavailable");
    add("coverage_ratio", std::to_string(status.coverage_ratio));
    add("frontier_cluster_count", std::to_string(status.frontier_cluster_count));
    add("candidate_count", std::to_string(status.candidate_count));
    add("navigation_phase",
        runtime->navigation_phase == NavigationPhase::kApproachTask
            ? "APPROACH_TASK"
            : "EXPLORE_TASK");
    add("fully_inside_task", runtime->fully_inside_task ? "true" : "false");
    add("approach_intent_count",
        std::to_string(approach ? approach->intent_points.size() : 0U));
    add("approach_candidate_count",
        std::to_string(approach ? approach->candidates.size() : 0U));
    add("remaining_guidance_m",
        remaining_guidance_m ? std::to_string(*remaining_guidance_m)
                             : "unavailable");
    add("guidance_unknown_cell_count",
        guidance_unknown_cell_count
            ? std::to_string(*guidance_unknown_cell_count)
            : "unavailable");
    add("approach_wait_reason",
        runtime->navigation_phase == NavigationPhase::kApproachTask &&
                runtime->approach_observability &&
                runtime->approach_observability->wait_reason
            ? *runtime->approach_observability->wait_reason
            : "unavailable");
    const auto add_local = [&add](const char* name, const auto& timing) {
      const std::string base{name};
      add(base + "_call_count", std::to_string(timing.call_count));
      add(base + "_last_elapsed_ms", std::to_string(timing.last_elapsed_ms));
      add(base + "_accumulated_elapsed_ms",
          std::to_string(timing.accumulated_elapsed_ms));
    };
    add_local("frontier_detection", runtime->frontier_detection_timing);
    add_local("information_gain", runtime->information_gain_timing);
    add_local("coarse_rank", runtime->coarse_rank_timing);
    add_local("final_rank", runtime->final_rank_timing);
    add_local("snapshot_to_goal", runtime->snapshot_to_goal_timing);
    const auto add_planner = [&add](const char* prefix,
                                    const auto& summary) {
      const std::string base{prefix};
      add(base + "call_count", std::to_string(summary.call_count));
      add(base + "record_count", std::to_string(summary.record_count));
      const auto value = [&summary](const double number) {
        return summary.last ? std::to_string(number) : std::string{"unavailable"};
      };
      const auto count = [&summary](const std::uint64_t number) {
        return summary.last ? std::to_string(number) : std::string{"unavailable"};
      };
      add(base + "global_elapsed_ms", value(summary.global_elapsed_ms));
      add(base + "global_call_count", count(summary.global_call_count));
      add(base + "local_elapsed_ms", value(summary.local_elapsed_ms));
      add(base + "local_call_count", count(summary.local_call_count));
      add(base + "total_elapsed_ms", value(summary.total_elapsed_ms));
    };
    add_planner("candidate_", runtime->planner_timing[0]);
    add_planner("rolling_", runtime->planner_timing[1]);
    add_planner("stuck_", runtime->planner_timing[2]);
    // The unprefixed tuple is a single exact request record.  Bucket totals
    // above are historical aggregates and must never stand in for it.
    add("global_elapsed_ms", current_timing
                                 ? std::to_string(current_timing->global_elapsed_ms)
                                 : "unavailable");
    add("global_call_count", current_timing
                                ? std::to_string(current_timing->global_call_count)
                                : "unavailable");
    add("local_elapsed_ms", current_timing
                                ? std::to_string(current_timing->local_elapsed_ms)
                                : "unavailable");
    add("local_call_count", current_timing
                               ? std::to_string(current_timing->local_call_count)
                               : "unavailable");
    add("total_elapsed_ms", current_timing
                                ? std::to_string(current_timing->total_elapsed_ms)
                                : "unavailable");
    const auto* stop_wait = runtime->current_stop_wait
                                ? &*runtime->current_stop_wait
                                : runtime->last_stop_wait
                                      ? &*runtime->last_stop_wait
                                      : nullptr;
    const auto finite_value = [](const double value) {
      return std::isfinite(value) ? std::to_string(value)
                                  : std::string{"unavailable"};
    };
    add("stop_wait_elapsed_ms",
        stop_wait
            ? finite_value(std::chrono::duration<double, std::milli>(
                               stop_wait->elapsed)
                               .count())
            : "unavailable");
    add("stop_entry_linear_mps",
        stop_wait ? finite_value(stop_wait->entry_speed.linear_speed_mps)
                  : "unavailable");
    add("stop_entry_angular_radps",
        stop_wait ? finite_value(stop_wait->entry_speed.angular_speed_radps)
                  : "unavailable");
    add("stop_confirmed_linear_mps",
        stop_wait && stop_wait->confirmed_speed
            ? finite_value(stop_wait->confirmed_speed->linear_speed_mps)
            : "unavailable");
    add("stop_confirmed_angular_radps",
        stop_wait && stop_wait->confirmed_speed
            ? finite_value(stop_wait->confirmed_speed->angular_speed_radps)
            : "unavailable");
    add("stop_confirmation_samples",
        stop_wait ? std::to_string(stop_wait->confirmation_samples)
                  : "unavailable");
    add("planning_epoch", std::to_string(runtime->planning_epoch));
    diagnostics.status.push_back(std::move(diagnostic));
    diagnostics.header = status.header;
    current_goal_publisher = runtime->current_goal_publisher;
    frontiers_publisher = runtime->frontiers_publisher;
    diagnostics_publisher = runtime->diagnostics_publisher;
    deferred_execution_cancel = std::move(runtime->deferred_execution_cancel);
    cancel_publisher = runtime->cancel_publisher;
  }
  if (deferred_execution_cancel) {
    std_msgs::msg::String message;
    message.data = std::move(*deferred_execution_cancel);
    cancel_publisher->publish(std::move(message));
  }
  publisher->publish(std::move(status));
  if (current_goal) {
    current_goal_publisher->publish(std::move(*current_goal));
  }
  frontiers_publisher->publish(std::move(markers));
  diagnostics_publisher->publish(std::move(diagnostics));
}

template <typename RuntimeT>
bool WaitingForInputsLocked(RuntimeT& runtime) {
  return !runtime.latest_map || !runtime.latest_map_message ||
         !runtime.task_boundary ||
         !runtime.pose_resolver.LatestPoseInMap().has_value();
}

template <typename RuntimeT>
void Pump(const std::shared_ptr<RuntimeT>& runtime);

template <typename RuntimeT>
void PumpApproach(const std::shared_ptr<RuntimeT>& runtime);

template <typename RuntimeT>
void PumpCurrentCycle(const std::shared_ptr<RuntimeT>& runtime) {
  bool approach_cycle = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    approach_cycle = static_cast<bool>(runtime->active_approach_cycle);
  }
  if (approach_cycle) {
    PumpApproach(runtime);
  } else {
    Pump(runtime);
  }
}

template <typename RuntimeT>
void StartExecutionReplan(const std::shared_ptr<RuntimeT>& runtime,
                          bool stationary_confirmed = false);

template <typename RuntimeT>
std::optional<std::string> ExecutionCancelLocked(RuntimeT& runtime);

struct FrozenBuildSnapshot final {
  std::string task_id;
  std::uint64_t epoch;
  std::uint64_t generation;
  FrozenGlobalMapContent map;
  Polygon2 boundary;
  Pose2 robot_pose;
  std::uint64_t pose_generation;
  std::uint64_t task_generation;
  lunar::pure_exploration::FailureMemory failure_memory;
  std::chrono::steady_clock::time_point snapshot_to_goal_start;
};

struct FrozenBuildProduct final {
  struct LocalTiming final {
    std::uint64_t call_count{0U};
    double last_elapsed_ms{0.0};
    double accumulated_elapsed_ms{0.0};
  };
  std::optional<TaskRaster> raster;
  CoverageStats coverage{};
  lunar::pure_exploration::FailureMemory failure_memory;
  FrozenPlanningCyclePtr cycle;
  FrozenApproachCyclePtr approach_cycle;
  std::optional<ApproachMarkers> approach_markers;
  std::size_t frontier_count{0U};
  std::size_t candidate_count{0U};
  LocalTiming frontier_detection_timing;
  LocalTiming information_gain_timing;
  LocalTiming coarse_rank_timing;
  bool has_reachable_free_start{true};
  NavigationPhase navigation_phase{NavigationPhase::kExploreTask};
  bool fully_inside_task{false};
  lunar::pure_exploration::ApproachWaitReason approach_wait_reason{
      lunar::pure_exploration::ApproachWaitReason::kNone};
  std::string error;
};

ApproachMarkers FreezeApproachMarkers(
    const lunar::pure_exploration::BoundaryGuidanceResult& guidance,
    const TaskRaster& raster) {
  ApproachMarkers result;
  result.intent_points.reserve(guidance.intents.size());
  for (const auto& intent : guidance.intents) {
    result.intent_points.push_back(raster.CellCenter(intent.cell));
  }
  result.candidates.reserve(guidance.candidates.size());
  for (const auto& candidate : guidance.candidates) {
    result.candidates.push_back(ApproachMarkerCandidate{
        .identity = candidate.identity, .pose = candidate.pose});
  }
  return result;
}

template <typename RuntimeT>
bool CandidateRemainsValidOnLatestMap(
    RuntimeT& runtime, const FrozenGlobalMapContent& map,
    const Polygon2& boundary, const Pose2& robot_pose,
    const lunar::pure_exploration::CandidateView& candidate);

template <typename RuntimeT>
bool HasFrozenCycleLocked(const RuntimeT& runtime) {
  return static_cast<bool>(runtime.active_cycle) ||
         static_cast<bool>(runtime.active_approach_cycle);
}

template <typename RuntimeT>
bool BlockedApproachSnapshotMatchesLocked(const RuntimeT& runtime) {
  return runtime.blocked_approach_snapshot && runtime.latest_map &&
         SameGeometry(runtime.blocked_approach_snapshot->map.geometry,
                      runtime.latest_map->geometry) &&
         runtime.blocked_approach_snapshot->map.data ==
             runtime.latest_map->data &&
         runtime.blocked_approach_snapshot->pose_generation ==
             runtime.pose_generation &&
         runtime.blocked_approach_snapshot->task_generation ==
             runtime.task_generation;
}

template <typename RuntimeT>
void BlockApproachSnapshotLocked(RuntimeT& runtime,
                                 std::string reason_code) {
  if (!runtime.latest_map) {
    throw std::logic_error{"cannot block approach without a map"};
  }
  runtime.blocked_approach_snapshot.emplace(
      typename RuntimeT::BlockedApproachSnapshot{
          .map = *runtime.latest_map,
          .pose_generation = runtime.pose_generation,
          .task_generation = runtime.task_generation});
  runtime.current_batch.clear();
  runtime.current_batch_cursor = 0U;
  runtime.current_candidate.reset();
  runtime.current_candidate_retryable_retries = 0U;
  runtime.candidate_retry_deadline.reset();
  runtime.final_rank_in_flight = false;
  runtime.snapshot_to_goal_start.reset();
  runtime.navigation_phase = NavigationPhase::kApproachTask;
  runtime.fully_inside_task = false;
  if (runtime.approach_observability) {
    ClearApproachSelectionLocked(runtime);
    runtime.approach_observability->wait_reason = reason_code;
  }
  runtime.status_reason_override = std::move(reason_code);
}

std::string ApproachWaitCode(
    const lunar::pure_exploration::ApproachWaitReason reason) {
  switch (reason) {
    case lunar::pure_exploration::ApproachWaitReason::kWaitingForSafeStart:
      return "WAITING_FOR_SAFE_APPROACH_START";
    case lunar::pure_exploration::ApproachWaitReason::
        kWaitingForTaskMapCoverage:
      return "WAITING_FOR_TASK_MAP_COVERAGE";
    case lunar::pure_exploration::ApproachWaitReason::kNoGuidanceRoute:
      return "APPROACH_NO_GUIDANCE_ROUTE";
    case lunar::pure_exploration::ApproachWaitReason::kNoSafeCandidate:
      return "APPROACH_NO_REACHABLE_TARGET";
    case lunar::pure_exploration::ApproachWaitReason::kNone:
      return "SELECTING_APPROACH_TARGET";
  }
  throw std::invalid_argument{"unknown approach wait reason"};
}

template <typename RuntimeT>
bool WfdCompletionProvenLocked(
    const RuntimeT& runtime, const FrozenPlanningCyclePtr& cycle) {
  return cycle && runtime.active_cycle == cycle &&
         !runtime.active_approach_cycle &&
         runtime.navigation_phase == NavigationPhase::kExploreTask &&
         runtime.fully_inside_task && runtime.latest_map_message &&
         cycle->batch()->GlobalMapContentEquals(*runtime.latest_map_message) &&
         cycle->coarse_cursor() == cycle->coarse_order().size() &&
         cycle->reachable().empty();
}

template <typename RuntimeT>
void QueueBuild(const std::shared_ptr<RuntimeT>& runtime,
                const bool stationary_confirmed = false) {
  std::optional<FrozenBuildSnapshot> snapshot;
  std::optional<std::string> execution_cancel;
  std::shared_ptr<SerializedWorkQueue> queue;
  bool publish_status = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    if (runtime->state_machine.state() !=
            ExplorationState::kWaitingForInput &&
        runtime->state_machine.state() !=
            ExplorationState::kSelectingFrontier) {
      return;
    }
    if (WaitingForInputsLocked(*runtime)) {
      runtime->state_machine.WaitForInput();
      publish_status = true;
    } else if (!HasFrozenCycleLocked(*runtime) &&
               !runtime->build_in_flight) {
      if (BlockedApproachSnapshotMatchesLocked(*runtime)) {
        return;
      }
      runtime->blocked_approach_snapshot.reset();
      runtime->status_reason_override.reset();
      if (!stationary_confirmed &&
          runtime->parameters.stop_before_planning &&
          PlanningIsAdmittedLocked(*runtime)) {
        execution_cancel = RequestStationaryActionLocked(
            *runtime, RuntimeT::PendingStationaryAction::kBuildFreshBatch);
        publish_status = true;
      }
      if (!PlanningIsAdmittedLocked(*runtime)) {
        publish_status = true;
      } else {
        if (runtime->state_machine.state() ==
            ExplorationState::kWaitingForInput) {
          runtime->state_machine.BeginSelection();
        }
        try {
          snapshot.emplace(FrozenBuildSnapshot{
              .task_id = runtime->state_machine.task_id(),
              .epoch = runtime->epoch,
              .generation = runtime->build_generation,
              .map = *runtime->latest_map,
              .boundary = *runtime->task_boundary,
              .robot_pose = *runtime->pose_resolver.LatestPoseInMap(),
              .pose_generation = runtime->pose_generation,
              .task_generation = runtime->task_generation,
              .failure_memory = runtime->failure_memory,
              .snapshot_to_goal_start = SteadyNowLocked(*runtime)});
          queue = runtime->work_queue;
          runtime->build_in_flight = static_cast<bool>(queue);
        } catch (const std::length_error& error) {
          FailLocked(*runtime, ResourceReason(error));
        } catch (const std::exception&) {
          FailLocked(*runtime, "INVALID_EXPLORATION_INPUT");
        }
        publish_status = true;
      }
    }
  }
  if (execution_cancel) {
    std_msgs::msg::String message;
    message.data = *execution_cancel;
    runtime->cancel_publisher->publish(std::move(message));
  }
  if (publish_status) {
    PublishStatus(runtime);
  }
  if (!snapshot || !queue) {
    return;
  }

  const std::weak_ptr<RuntimeT> weak_runtime{runtime};
  const bool submitted = queue->Submit(
      [weak_runtime, snapshot = std::move(*snapshot)]() mutable {
        const auto locked = weak_runtime.lock();
        if (!locked) {
          return;
        }

        bool retry_build = false;
        {
          std::scoped_lock lock{locked->mutex};
          if (locked->teardown || !locked->build_in_flight) {
            return;
          }
          if (locked->epoch != snapshot.epoch ||
              locked->build_generation != snapshot.generation ||
              locked->state_machine.task_id() != snapshot.task_id ||
              HasFrozenCycleLocked(*locked) ||
              (locked->state_machine.state() !=
                   ExplorationState::kWaitingForInput &&
               locked->state_machine.state() !=
                   ExplorationState::kSelectingFrontier)) {
            locked->build_in_flight = false;
            retry_build = !locked->teardown;
          }
        }
        if (retry_build) {
          QueueBuild(locked);
          return;
        }

        FrozenBuildProduct product{
            .failure_memory = std::move(snapshot.failure_memory)};
        try {
          OccupancyGridView map(snapshot.map.geometry, snapshot.map.data,
                                locked->parameters.global_occupied_threshold);
          product.raster.emplace(TaskRaster::Build(
              map, snapshot.boundary,
              locked->parameters.task_raster_limits));
          product.coverage = CalculateCoverage(*product.raster);
          std::optional<lunar::pure_exploration::BoundaryGuidanceResult>
              guidance;
          try {
            guidance.emplace(locked->boundary_guidance.Build(
                map, *product.raster, snapshot.robot_pose));
            if (guidance->phase == NavigationPhase::kApproachTask &&
                locked->parameters.pipeline_seams &&
                locked->parameters.pipeline_seams
                    ->after_boundary_guidance_build) {
              locked->parameters.pipeline_seams
                  ->after_boundary_guidance_build();
            }
          } catch (const std::length_error& error) {
            product.error = GuidanceResourceReason(error);
          }
          if (guidance) {
          product.navigation_phase = guidance->phase;
          product.fully_inside_task = guidance->fully_inside_task;
          product.approach_wait_reason = guidance->wait_reason;
          if (guidance->phase == NavigationPhase::kApproachTask) {
            try {
              product.approach_markers =
                  FreezeApproachMarkers(*guidance, *product.raster);
            } catch (const std::length_error&) {
              product.error = "RESOURCE_RESULT_COPY";
            } catch (const std::bad_alloc&) {
              product.error = "RESOURCE_RESULT_COPY";
            }
            auto order = locked->boundary_guidance.CoarseOrder(
                *guidance, snapshot.robot_pose);
            product.candidate_count = guidance->candidates.size();
            if (product.error.empty()) {
              product.approach_cycle = std::make_shared<FrozenApproachCycle>(
                  snapshot.map, std::move(*guidance), snapshot.robot_pose,
                  snapshot.map.geometry.resolution, std::move(order));
            }
          } else {
            const auto robot_cell = product.raster->WorldToCell(
                {snapshot.robot_pose.x, snapshot.robot_pose.y});
            if (!robot_cell) {
              throw std::invalid_argument{
                  "robot map pose cannot resolve to a cell"};
            }
            FrontierDetector detector(FrontierParameters{
                .minimum_cluster_length_m = std::max(
                    locked->candidate_generator.platform_width_m(),
                    2.0 * snapshot.map.geometry.resolution)});
            const auto frontier_start = SteadyNowLocked(*locked);
            auto detection = detector.Detect(*product.raster, *robot_cell);
            RecordLocalTiming(product.frontier_detection_timing,
                              SteadyElapsedMs(*locked, frontier_start));
            product.has_reachable_free_start =
                detection.has_reachable_free_start;
            if (product.has_reachable_free_start) {
            auto frontiers = std::move(detection.clusters);
            const bool controlled_candidates =
                locked->parameters.pipeline_seams &&
                locked->parameters.pipeline_seams->generate_candidates;
            std::vector<lunar::pure_exploration::CandidateView> candidates;
            if (controlled_candidates) {
              candidates = locked->parameters.pipeline_seams
                               ->generate_candidates(*product.raster,
                                                     frontiers);
            } else if (locked->parameters.filter_global_goal_cell) {
              candidates = locked->candidate_generator.Generate(
                  *product.raster, frontiers,
                  [&](const Pose2 pose) {
                    return GlobalGoalCellFeasible(
                        snapshot.map, pose,
                        locked->candidate_generator.minimum_standoff_m(),
                        locked->parameters.global_occupied_threshold);
                  });
            } else {
              candidates = locked->candidate_generator.Generate(
                  *product.raster, frontiers);
            }

            // Test seams inject completed candidate lists and therefore
            // cannot participate in the generator's retry predicate.
            if (locked->parameters.filter_global_goal_cell &&
                controlled_candidates) {
              std::erase_if(candidates, [&](const auto& candidate) {
                return !GlobalGoalCellFeasible(
                    snapshot.map, candidate.pose,
                    locked->candidate_generator.minimum_standoff_m(),
                    locked->parameters.global_occupied_threshold);
              });
            }

            std::vector<lunar::pure_exploration::CandidateView> unsuppressed;
            unsuppressed.reserve(candidates.size());
            for (auto& candidate : candidates) {
              if (!product.failure_memory.IsSuppressed(candidate,
                                                       *product.raster)) {
                unsuppressed.push_back(std::move(candidate));
              }
            }
            candidates = std::move(unsuppressed);

            std::vector<CandidateGain> gains;
            gains.reserve(candidates.size());
            for (std::size_t index = 0U; index < candidates.size(); ++index) {
              const auto gain_start = SteadyNowLocked(*locked);
              const double gain =
                  locked->parameters.pipeline_seams &&
                          locked->parameters.pipeline_seams->evaluate_gain
                      ? locked->parameters.pipeline_seams->evaluate_gain(
                            *product.raster, candidates[index])
                      : locked->information_gain
                            .Evaluate(*product.raster, candidates[index])
                            .visible_unknown_area_m2;
              RecordLocalTiming(product.information_gain_timing,
                                SteadyElapsedMs(*locked, gain_start));
              gains.push_back({index, gain});
            }
            auto batch = std::make_shared<FrozenCandidateBatch>(
                snapshot.map, std::move(frontiers), std::move(candidates));
            const auto coarse_start = SteadyNowLocked(*locked);
            auto order = locked->ranker.CoarseRank(
                batch->candidates(), batch->frontiers(), gains,
                snapshot.robot_pose);
            RecordLocalTiming(product.coarse_rank_timing,
                              SteadyElapsedMs(*locked, coarse_start));
            product.frontier_count = batch->frontiers().size();
            product.candidate_count = batch->candidates().size();
            product.cycle = std::make_shared<FrozenPlanningCycle>(
                std::move(batch), snapshot.robot_pose,
                snapshot.map.geometry.resolution, std::move(gains),
                std::move(order));
            }
          }
          }
        } catch (const std::length_error& error) {
          product.error = ResourceReason(error);
        } catch (const std::overflow_error&) {
          product.error = "NUMERICAL_OVERFLOW";
        } catch (const std::exception&) {
          product.error = "INVALID_EXPLORATION_INPUT";
        }

        bool committed = false;
        bool pump = false;
        bool retry_after_stale = false;
        {
          std::scoped_lock lock{locked->mutex};
          if (locked->teardown || locked->epoch != snapshot.epoch ||
              locked->build_generation != snapshot.generation ||
              locked->state_machine.task_id() != snapshot.task_id ||
              HasFrozenCycleLocked(*locked) ||
              (locked->state_machine.state() !=
                   ExplorationState::kWaitingForInput &&
               locked->state_machine.state() !=
                   ExplorationState::kSelectingFrontier)) {
            locked->build_in_flight = false;
            retry_after_stale = !locked->teardown;
          } else if (product.error.empty() &&
                     product.navigation_phase == NavigationPhase::kApproachTask &&
                     (snapshot.task_generation != locked->task_generation ||
                      !locked->latest_map_message ||
                      !product.approach_cycle ||
                      !product.approach_cycle->GlobalMapContentEquals(
                          *locked->latest_map_message))) {
            locked->build_in_flight = false;
            ++locked->build_generation;
            retry_after_stale = true;
          } else {
            locked->build_in_flight = false;
            committed = true;
            if (!product.error.empty()) {
              FailLocked(*locked, std::move(product.error));
            } else {
              locked->failure_memory = std::move(product.failure_memory);
              locked->active_raster = std::move(product.raster);
              locked->coverage = product.coverage;
              MergeLocalTiming(locked->frontier_detection_timing,
                               product.frontier_detection_timing);
              MergeLocalTiming(locked->information_gain_timing,
                               product.information_gain_timing);
              MergeLocalTiming(locked->coarse_rank_timing,
                               product.coarse_rank_timing);
              locked->frontier_count = product.frontier_count;
              locked->candidate_count = product.candidate_count;
              locked->navigation_phase = product.navigation_phase;
              locked->fully_inside_task = product.fully_inside_task;
              locked->current_batch.clear();
              locked->current_batch_cursor = 0U;
              locked->current_candidate.reset();
              locked->current_candidate_retryable_retries = 0U;
              locked->blocked_approach_snapshot.reset();
              if (product.navigation_phase == NavigationPhase::kApproachTask) {
                locked->active_cycle.reset();
                if (!product.approach_markers) {
                  FailLocked(*locked, "RESOURCE_RESULT_COPY");
                } else {
                  locked->approach_observability.emplace(
                      typename RuntimeT::ApproachObservabilitySnapshot{
                          .markers = std::move(*product.approach_markers),
                          .remaining_guidance_m = std::nullopt,
                          .guidance_unknown_cell_count = std::nullopt,
                          .wait_reason = std::nullopt});
                  locked->active_approach_cycle =
                      std::move(product.approach_cycle);
                  if (!locked->active_approach_cycle) {
                    FailLocked(*locked, "INVALID_EXPLORATION_INPUT");
                  } else if (locked->active_approach_cycle->coarse_order().empty()) {
                    const auto wait_reason =
                        locked->active_approach_cycle->guidance().wait_reason;
                    locked->active_approach_cycle.reset();
                    BlockApproachSnapshotLocked(
                        *locked, ApproachWaitCode(wait_reason));
                  } else {
                    locked->status_reason_override =
                        "SELECTING_APPROACH_TARGET";
                    locked->snapshot_to_goal_start.emplace(
                        typename RuntimeT::SnapshotToGoalStart{
                            .approach_cycle = locked->active_approach_cycle,
                            .epoch = snapshot.epoch,
                            .build_generation = snapshot.generation,
                            .start = snapshot.snapshot_to_goal_start});
                    pump = true;
                  }
                }
              } else {
                locked->approach_observability.reset();
                locked->active_approach_cycle.reset();
                locked->status_reason_override.reset();
                if (!product.has_reachable_free_start) {
                  locked->active_cycle.reset();
                } else {
                  locked->active_cycle = std::move(product.cycle);
                  if (!locked->active_cycle) {
                    FailLocked(*locked, "INVALID_EXPLORATION_INPUT");
                  } else if (locked->active_cycle->coarse_order().empty()) {
                    if (!WfdCompletionProvenLocked(
                            *locked, locked->active_cycle)) {
                      locked->active_cycle.reset();
                      locked->active_raster.reset();
                      ++locked->build_generation;
                      pump = true;
                    } else {
                      locked->state_machine.CompleteNoReachableFrontier();
                    }
                  } else {
                    locked->snapshot_to_goal_start.emplace(
                        typename RuntimeT::SnapshotToGoalStart{
                            .cycle = locked->active_cycle,
                            .epoch = snapshot.epoch,
                            .build_generation = snapshot.generation,
                            .start = snapshot.snapshot_to_goal_start});
                    pump = true;
                  }
                }
              }
            }
          }
        }
        if (retry_after_stale) {
          QueueBuild(locked);
          return;
        }
        if (committed) {
          PublishStatus(locked);
        }
        if (pump) {
          bool has_cycle = false;
          bool has_approach_cycle = false;
          {
            std::scoped_lock lock{locked->mutex};
            has_cycle = static_cast<bool>(locked->active_cycle);
            has_approach_cycle =
                static_cast<bool>(locked->active_approach_cycle);
          }
          if (has_cycle) {
            Pump(locked);
          } else if (has_approach_cycle) {
            PumpApproach(locked);
          } else {
            QueueBuild(locked);
          }
        }
      });
  if (!submitted) {
    {
      std::scoped_lock lock{runtime->mutex};
      if (!runtime->teardown) {
        runtime->build_in_flight = false;
        FailLocked(*runtime, "WORK_QUEUE_UNAVAILABLE");
      }
    }
    PublishStatus(runtime);
  }
}

template <typename RuntimeT>
void ApplyStartLocked(RuntimeT& runtime,
                      typename RuntimeT::PendingStart start) {
  const bool retain_execution_until_stationary =
      runtime.parameters.stop_before_planning &&
      runtime.active_reference.has_value();
  CancelPendingStationaryActionLocked(runtime);
  ResetStopWaitTaskLifetimeLocked(runtime);
  runtime.state_machine.Start(start.task_id);
  ++runtime.task_generation;
  ResetActiveElapsedLocked(runtime);
  runtime.timing_provenance_order.clear();
  runtime.timing_provenance.clear();
  runtime.consumed_timing_order.clear();
  runtime.consumed_timing.clear();
  runtime.planner_timing = {};
  runtime.frontier_detection_timing = {};
  runtime.information_gain_timing = {};
  runtime.coarse_rank_timing = {};
  runtime.final_rank_timing = {};
  runtime.snapshot_to_goal_timing = {};
  runtime.snapshot_to_goal_start.reset();
  runtime.execution_replan_request_id.reset();
  runtime.execution_replan_bucket.reset();
  runtime.failure_memory.BeginTask(start.task_id);
  runtime.task_boundary = std::move(start.boundary);
  ClearSelectionCycleLocked(runtime);
  runtime.approach_observability.reset();
  runtime.execution_replan_waiting_terminal = false;
  runtime.execution_replan_retry_pending = false;
  runtime.pending_map_rebuild = false;
  runtime.pending_stuck_rebuild = false;
  runtime.frontier_count = 0U;
  runtime.candidate_count = 0U;
  runtime.coverage = {};
  runtime.completed_goal_positions.clear();
  if (!retain_execution_until_stationary) {
    ClearExecutionPayloadsLocked(runtime);
  }
  ++runtime.epoch;
  ++runtime.build_generation;
}

template <typename RuntimeT>
bool ApplyPendingControlLocked(RuntimeT& runtime) {
  const auto pending = runtime.pending_control;
  const bool retain_execution_until_stationary =
      pending == RuntimeT::PendingControl::kReplacementStart &&
      runtime.parameters.stop_before_planning &&
      runtime.active_reference.has_value();
  runtime.pending_control = RuntimeT::PendingControl::kNone;
  if (!retain_execution_until_stationary) {
    runtime.pending_execution_cancel_sent = false;
  }
  CancelPendingStationaryActionLocked(runtime);
  ClearSelectionCycleLocked(runtime);
  runtime.execution_replan_request_id.reset();
  runtime.execution_replan_bucket.reset();
  runtime.execution_replan_waiting_terminal = false;
  runtime.execution_replan_retry_pending = false;
  runtime.pending_map_rebuild = false;
  runtime.pending_stuck_rebuild = false;
  if (!retain_execution_until_stationary) {
    ClearExecutionPayloadsLocked(runtime);
  }
  ++runtime.epoch;
  ++runtime.build_generation;
  switch (pending) {
    case RuntimeT::PendingControl::kNone:
      return false;
    case RuntimeT::PendingControl::kPause:
      runtime.pending_start.reset();
      FreezeActiveElapsedLocked(runtime);
      runtime.state_machine.Pause();
      return false;
    case RuntimeT::PendingControl::kCancel:
      runtime.pending_start.reset();
      FreezeActiveElapsedLocked(runtime);
      runtime.state_machine.Cancel();
      runtime.approach_observability.reset();
      ResetStopWaitTaskLifetimeLocked(runtime);
      runtime.active_elapsed_s = 0.0;
      runtime.task_boundary.reset();
      runtime.coverage = {};
      runtime.frontier_count = 0U;
      runtime.candidate_count = 0U;
      runtime.completed_goal_positions.clear();
      return false;
    case RuntimeT::PendingControl::kReplacementStart:
      if (!runtime.pending_start) {
        FailLocked(runtime, "MISSING_REPLACEMENT_TASK");
        return false;
      }
      ApplyStartLocked(runtime, std::move(*runtime.pending_start));
      runtime.pending_start.reset();
      return true;
  }
  return false;
}

template <typename RuntimeT>
void HandleEvaluation(const std::weak_ptr<RuntimeT>& weak_runtime,
                      const FrozenPlanningCyclePtr& cycle,
                      const std::uint64_t epoch,
                      PlannerEvaluation evaluation) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  bool pump = false;
  bool build = false;
  bool start_execution_replan = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    runtime->planner_in_flight = false;
    if (runtime->execution_replan_waiting_terminal) {
      runtime->execution_replan_waiting_terminal = false;
      start_execution_replan =
          runtime->state_machine.state() == ExplorationState::kReplanning &&
          runtime->state_machine.active_goal().has_value();
    } else if (runtime->pending_control != RuntimeT::PendingControl::kNone) {
      build = ApplyPendingControlLocked(*runtime);
    } else if (const auto pending_rebuild =
                   ResolvePendingRebuildAfterPlannerTerminalLocked(*runtime);
               pending_rebuild != PlannerTerminalRebuild::kNone) {
      build = pending_rebuild == PlannerTerminalRebuild::kBuildNow;
    } else if (runtime->active_cycle != cycle || runtime->epoch != epoch) {
      if (runtime->state_machine.state() ==
              ExplorationState::kSelectingFrontier &&
          PlanningIsAdmittedLocked(*runtime)) {
        pump = static_cast<bool>(runtime->active_cycle);
        build = !HasFrozenCycleLocked(*runtime) &&
                !runtime->build_in_flight;
      }
    } else {
      const auto associated =
          cycle->batch()->CandidateIndexForRequest(evaluation.request_id);
      if (!associated || !runtime->current_candidate ||
          *associated != *runtime->current_candidate) {
        FailLocked(*runtime, "UNKNOWN_PLANNER_REQUEST_ID");
      } else {
        switch (evaluation.kind) {
          case PlannerEvaluationKind::kReachable:
            if (!evaluation.path_length_m || !evaluation.reference) {
              FailLocked(*runtime, "PLANNER_REACHABLE_CONTRACT_ERROR");
              break;
            }
            try {
              static_cast<void>(ValidateExecutableReference(
                  *evaluation.reference,
                  runtime->parameters.platform.platform_type,
                  runtime->parameters.maximum_executable_path_points));
              cycle->AddReachable(
                  {.metrics = {*associated, *evaluation.path_length_m},
                   .reference = std::move(*evaluation.reference)});
              runtime->current_candidate.reset();
              runtime->current_candidate_retryable_retries = 0U;
              ++runtime->current_batch_cursor;
              pump = true;
            } catch (const std::length_error&) {
              FailLocked(*runtime, "RESOURCE_EXECUTABLE_PATH_LIMIT");
            } catch (const std::exception&) {
              FailLocked(*runtime, "INVALID_EXECUTABLE_REFERENCE");
            }
            break;
          case PlannerEvaluationKind::kExhaustiveNoPath:
            runtime->current_candidate.reset();
            runtime->current_candidate_retryable_retries = 0U;
            ++runtime->current_batch_cursor;
            pump = true;
            break;
          case PlannerEvaluationKind::kRetryable:
            if (runtime->current_candidate_retryable_retries <
                runtime->parameters.maximum_candidate_retryable_retries) {
              ++runtime->current_candidate_retryable_retries;
              runtime->candidate_retry_deadline =
                  SteadyNowLocked(*runtime) + std::chrono::milliseconds{100};
            } else {
              runtime->current_candidate.reset();
              runtime->current_candidate_retryable_retries = 0U;
              ++runtime->current_batch_cursor;
              pump = true;
            }
            break;
          case PlannerEvaluationKind::kCanceled:
            FailLocked(*runtime, "UNEXPECTED_PLANNER_CANCELED");
            break;
          case PlannerEvaluationKind::kContractError:
            FailLocked(*runtime,
                       "PLANNER_CONTRACT_" + evaluation.reason_code);
            break;
          case PlannerEvaluationKind::kResourceError:
            switch (evaluation.resource_kind) {
              case PlannerResourceKind::kPathPreviewPoses:
                FailLocked(*runtime, "RESOURCE_PATH_PREVIEW_LIMIT");
                break;
              case PlannerResourceKind::kExecutablePathPoints:
                FailLocked(*runtime, "RESOURCE_EXECUTABLE_PATH_LIMIT");
                break;
              case PlannerResourceKind::kResultCopyFailure:
                FailLocked(*runtime, "RESOURCE_RESULT_COPY");
                break;
              case PlannerResourceKind::kNone:
                FailLocked(*runtime,
                           "PLANNER_RESOURCE_CLASSIFICATION_ERROR");
                break;
            }
            break;
        }
      }
    }
  }
  PublishStatus(runtime);
  if (build) {
    QueueBuild(runtime);
  } else if (start_execution_replan) {
    StartExecutionReplan(runtime, true);
  } else if (pump) {
    Pump(runtime);
  }
}

template <typename RuntimeT>
void HandleApproachEvaluation(
    const std::weak_ptr<RuntimeT>& weak_runtime,
    const FrozenApproachCyclePtr& cycle, const std::uint64_t epoch,
    PlannerEvaluation evaluation) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  bool pump = false;
  bool build = false;
  bool start_execution_replan = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    runtime->planner_in_flight = false;
    if (runtime->execution_replan_waiting_terminal) {
      runtime->execution_replan_waiting_terminal = false;
      start_execution_replan =
          runtime->state_machine.state() == ExplorationState::kReplanning &&
          runtime->state_machine.active_goal().has_value();
    } else if (runtime->pending_control != RuntimeT::PendingControl::kNone) {
      build = ApplyPendingControlLocked(*runtime);
    } else if (const auto pending_rebuild =
                   ResolvePendingRebuildAfterPlannerTerminalLocked(*runtime);
               pending_rebuild != PlannerTerminalRebuild::kNone) {
      build = pending_rebuild == PlannerTerminalRebuild::kBuildNow;
    } else if (runtime->active_approach_cycle != cycle ||
               runtime->epoch != epoch) {
      if (runtime->state_machine.state() ==
              ExplorationState::kSelectingFrontier &&
          PlanningIsAdmittedLocked(*runtime)) {
        pump = static_cast<bool>(runtime->active_approach_cycle);
        build = !HasFrozenCycleLocked(*runtime) &&
                !runtime->build_in_flight;
      }
    } else {
      const auto associated =
          cycle->CandidateIndexForRequest(evaluation.request_id);
      if (!associated || !runtime->current_candidate ||
          *associated != *runtime->current_candidate) {
        FailLocked(*runtime, "UNKNOWN_PLANNER_REQUEST_ID");
      } else {
        switch (evaluation.kind) {
          case PlannerEvaluationKind::kReachable:
            if (!evaluation.path_length_m || !evaluation.reference) {
              FailLocked(*runtime, "PLANNER_REACHABLE_CONTRACT_ERROR");
              break;
            }
            try {
              static_cast<void>(ValidateExecutableReference(
                  *evaluation.reference,
                  runtime->parameters.platform.platform_type,
                  runtime->parameters.maximum_executable_path_points));
              cycle->AddReachable(
                  {.metrics = {*associated, *evaluation.path_length_m},
                   .reference = std::move(*evaluation.reference)});
              runtime->current_candidate.reset();
              runtime->current_candidate_retryable_retries = 0U;
              ++runtime->current_batch_cursor;
              pump = true;
            } catch (const std::length_error&) {
              FailLocked(*runtime, "RESOURCE_EXECUTABLE_PATH_LIMIT");
            } catch (const std::exception&) {
              FailLocked(*runtime, "INVALID_EXECUTABLE_REFERENCE");
            }
            break;
          case PlannerEvaluationKind::kExhaustiveNoPath:
            runtime->current_candidate.reset();
            runtime->current_candidate_retryable_retries = 0U;
            ++runtime->current_batch_cursor;
            pump = true;
            break;
          case PlannerEvaluationKind::kRetryable:
            if (runtime->current_candidate_retryable_retries <
                runtime->parameters.maximum_candidate_retryable_retries) {
              ++runtime->current_candidate_retryable_retries;
              runtime->candidate_retry_deadline =
                  SteadyNowLocked(*runtime) + std::chrono::milliseconds{100};
            } else {
              runtime->current_candidate.reset();
              runtime->current_candidate_retryable_retries = 0U;
              ++runtime->current_batch_cursor;
              pump = true;
            }
            break;
          case PlannerEvaluationKind::kCanceled:
            FailLocked(*runtime, "UNEXPECTED_PLANNER_CANCELED");
            break;
          case PlannerEvaluationKind::kContractError:
            FailLocked(*runtime,
                       "PLANNER_CONTRACT_" + evaluation.reason_code);
            break;
          case PlannerEvaluationKind::kResourceError:
            switch (evaluation.resource_kind) {
              case PlannerResourceKind::kPathPreviewPoses:
                FailLocked(*runtime, "RESOURCE_PATH_PREVIEW_LIMIT");
                break;
              case PlannerResourceKind::kExecutablePathPoints:
                FailLocked(*runtime, "RESOURCE_EXECUTABLE_PATH_LIMIT");
                break;
              case PlannerResourceKind::kResultCopyFailure:
                FailLocked(*runtime, "RESOURCE_RESULT_COPY");
                break;
              case PlannerResourceKind::kNone:
                FailLocked(*runtime,
                           "PLANNER_RESOURCE_CLASSIFICATION_ERROR");
                break;
            }
            break;
        }
      }
    }
  }
  PublishStatus(runtime);
  if (build) {
    QueueBuild(runtime);
  } else if (start_execution_replan) {
    StartExecutionReplan(runtime, true);
  } else if (pump) {
    PumpApproach(runtime);
  }
}

template <typename RuntimeT>
void QueueFinalRank(const std::shared_ptr<RuntimeT>& runtime,
                    const FrozenPlanningCyclePtr& cycle,
                    const std::uint64_t epoch,
                    std::vector<PlannedCandidate> planned,
                    std::vector<Vec2> completed_goal_positions) {
  std::shared_ptr<SerializedWorkQueue> queue;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown || runtime->active_cycle != cycle ||
        runtime->epoch != epoch || !runtime->final_rank_in_flight) {
      return;
    }
    queue = runtime->work_queue;
  }

  const std::weak_ptr<RuntimeT> weak_runtime{runtime};
  const bool submitted = queue && queue->Submit(
      [weak_runtime, cycle, epoch, planned = std::move(planned),
       completed_goal_positions = std::move(completed_goal_positions)]() mutable {
        struct FinalSelection final {
          std::optional<ActiveGoal> goal;
          std::optional<lunar::pure_exploration::CandidateView> candidate;
          std::optional<MotionReference> active_reference;
          std::optional<MotionReference> publish_reference;
          std::vector<Vec2> executable_polyline;
          std::optional<Vec2> executable_endpoint;
          std::string error;
        };

        const auto locked = weak_runtime.lock();
        if (!locked) {
          return;
        }
        FinalSelection selection;
        double final_rank_elapsed_ms = -1.0;
        try {
          const auto final_rank_start = SteadyNowLocked(*locked);
          const auto final =
              locked->parameters.pipeline_seams &&
                      locked->parameters.pipeline_seams->final_rank
                  ? locked->parameters.pipeline_seams->final_rank(
                        cycle->batch()->candidates(),
                        cycle->batch()->frontiers(), cycle->complete_gains(),
                        planned, cycle->frozen_robot_pose(),
                        completed_goal_positions,
                        cycle->frozen_resolution_m())
                  : locked->ranker.FinalRank(
                        cycle->batch()->candidates(),
                        cycle->batch()->frontiers(), cycle->complete_gains(),
                        planned, cycle->frozen_robot_pose(),
                        completed_goal_positions,
                        cycle->frozen_resolution_m());
          final_rank_elapsed_ms = SteadyElapsedMs(*locked, final_rank_start);
          if (final.empty()) {
            throw std::logic_error{"reachable rank unexpectedly empty"};
          }
          const std::size_t selected_index = final.front().candidate_index;
          const auto selected = std::ranges::find_if(
              cycle->reachable(), [&](const auto& row) {
                return row.metrics.candidate_index == selected_index;
              });
          if (selected == cycle->reachable().end()) {
            throw std::logic_error{"selected reachable row is missing"};
          }
          const auto selected_request_id =
              cycle->batch()->LatestRequestIdForCandidate(selected_index);
          if (!selected_request_id) {
            throw std::logic_error{"selected request identity is missing"};
          }
          const ExecutableReference executable = ValidateExecutableReference(
              selected->reference,
              locked->parameters.platform.platform_type,
              locked->parameters.maximum_executable_path_points);
          selection.goal.emplace(MakeActiveGoal(
              cycle->batch()->candidates()[selected_index],
              cycle->batch()->frontiers(), *selected_request_id));
          selection.candidate.emplace(
              cycle->batch()->candidates()[selected_index]);
          try {
            const auto copy_reference = [&](const MotionReference& source) {
              return locked->parameters.pipeline_seams &&
                             locked->parameters.pipeline_seams->copy_reference
                         ? locked->parameters.pipeline_seams->copy_reference(
                               source)
                         : MotionReference{source};
            };
            selection.active_reference.emplace(
                copy_reference(selected->reference));
            selection.publish_reference.emplace(
                copy_reference(selected->reference));
            selection.executable_polyline = executable.polyline;
            selection.executable_endpoint = executable.endpoint;
          } catch (const std::length_error&) {
            selection.error = "RESOURCE_RESULT_COPY";
          } catch (const std::bad_alloc&) {
            selection.error = "RESOURCE_RESULT_COPY";
          }
        } catch (const std::length_error&) {
          selection.error = "RESOURCE_EXECUTABLE_PATH_LIMIT";
        } catch (const std::bad_alloc&) {
          selection.error = "RESOURCE_RESULT_COPY";
        } catch (const std::overflow_error&) {
          selection.error = "NUMERICAL_OVERFLOW";
        } catch (const std::exception&) {
          selection.error = "FINAL_SELECTION_ERROR";
        }

        struct LatestMapValidation final {
          std::uint64_t generation{0U};
          FrozenGlobalMapContent map;
          Polygon2 boundary;
          Pose2 pose;
          lunar::pure_exploration::CandidateView candidate;
        };
        std::optional<LatestMapValidation> validation;
        bool map_validation_unavailable = false;
        std::uint64_t observed_map_generation = 0U;
        {
          std::scoped_lock lock{locked->mutex};
          if (locked->teardown || locked->active_cycle != cycle ||
              locked->epoch != epoch || !locked->final_rank_in_flight ||
              locked->state_machine.state() !=
                  ExplorationState::kSelectingFrontier) {
            return;
          }
          observed_map_generation = locked->map_generation;
          if (!locked->latest_map_message || !selection.candidate) {
            map_validation_unavailable = true;
          } else if (!cycle->batch()->GlobalMapContentEquals(
                         *locked->latest_map_message)) {
            const auto pose = locked->pose_resolver.LatestPoseInMap();
            if (!locked->latest_map || !locked->task_boundary || !pose) {
              map_validation_unavailable = true;
            } else {
              validation.emplace(LatestMapValidation{
                  .generation = locked->map_generation,
                  .map = *locked->latest_map,
                  .boundary = *locked->task_boundary,
                  .pose = *pose,
                  .candidate = *selection.candidate});
            }
          }
        }
        if (locked->parameters.pipeline_seams &&
            locked->parameters.pipeline_seams->after_final_rank_map_observed) {
          locked->parameters.pipeline_seams->after_final_rank_map_observed();
        }
        bool map_validation_valid = true;
        std::string map_validation_error;
        if (validation) {
          try {
            map_validation_valid = CandidateRemainsValidOnLatestMap(
                *locked, validation->map, validation->boundary,
                validation->pose, validation->candidate);
          } catch (const std::length_error& error) {
            map_validation_error = ResourceReason(error);
          } catch (const std::bad_alloc&) {
            map_validation_error = "RESOURCE_RESULT_COPY";
          } catch (const std::exception&) {
            map_validation_error = "FINAL_MAP_VALIDATION_ERROR";
          }
        }

        bool rebuild = false;
        {
          std::unique_lock lock{locked->mutex};
          if (locked->teardown || locked->active_cycle != cycle ||
              locked->epoch != epoch || !locked->final_rank_in_flight ||
              locked->state_machine.state() !=
                  ExplorationState::kSelectingFrontier) {
            return;
          }
          RecordLocalTiming(locked->final_rank_timing, final_rank_elapsed_ms);
          if (!selection.error.empty()) {
            FailLocked(*locked, std::move(selection.error));
          } else if (!selection.goal || !selection.active_reference ||
                     !selection.publish_reference || !selection.candidate) {
            FailLocked(*locked, "FINAL_SELECTION_ERROR");
          } else if (!map_validation_error.empty()) {
            FailLocked(*locked, std::move(map_validation_error));
          } else if (map_validation_unavailable ||
                     locked->map_generation != observed_map_generation ||
                     (validation &&
                      (locked->map_generation != validation->generation ||
                       !map_validation_valid))) {
            locked->active_cycle.reset();
            locked->active_raster.reset();
            locked->final_rank_in_flight = false;
            ++locked->epoch;
            ++locked->build_generation;
            rebuild = true;
          } else {
            try {
              locked->active_reference.emplace(
                  std::move(*selection.active_reference));
              locked->active_executable_polyline =
                  std::move(selection.executable_polyline);
              locked->active_executable_endpoint =
                  selection.executable_endpoint;
              const auto initial_pose =
                  locked->pose_resolver.LatestPoseInMap();
              if (!initial_pose) {
                throw std::logic_error{"missing execution pose"};
              }
              const auto now = locked->parameters.steady_now
                                   ? locked->parameters.steady_now()
                                   : std::chrono::steady_clock::now();
              locked->execution_monitor.emplace(ExecutionMonitorParameters{
                  .maximum_executable_path_points =
                      locked->parameters.maximum_executable_path_points,
                  .position_tolerance_m = std::max(
                      0.5 * cycle->frozen_resolution_m(),
                      0.25 * locked->candidate_generator.platform_width_m()),
                  .yaw_tolerance_rad = locked->parameters.goal_yaw_tolerance_rad,
                  .stuck_window = std::chrono::seconds{30},
                  .minimum_progress_m = 0.2});
              locked->execution_monitor->ResetProgress(
                  now, locked->active_executable_polyline,
                  {initial_pose->x, initial_pose->y});
              locked->state_machine.BeginPlanning();
              Status planning_status = MakeStatusLocked(*locked);
              const auto planning_status_publisher = locked->status_publisher;
              lock.unlock();
              try {
                planning_status_publisher->publish(std::move(planning_status));
              } catch (...) {
                lock.lock();
                throw;
              }
              lock.lock();
              if (locked->teardown || locked->active_cycle != cycle ||
                  locked->epoch != epoch ||
                  locked->state_machine.state() !=
                      ExplorationState::kPlanning) {
                return;
              }
              locked->state_machine.CommitGoal(std::move(*selection.goal));
              if (locked->snapshot_to_goal_start &&
                  locked->snapshot_to_goal_start->cycle.lock() == cycle &&
                  locked->snapshot_to_goal_start->epoch == locked->epoch &&
                  locked->snapshot_to_goal_start->build_generation ==
                      locked->build_generation) {
                RecordLocalTiming(
                    locked->snapshot_to_goal_timing,
                    SteadyElapsedMs(*locked,
                                    locked->snapshot_to_goal_start->start));
                locked->snapshot_to_goal_start.reset();
              }
              locked->pending_execution_cancel_sent = false;
              locked->final_rank_in_flight = false;
              if (locked->parameters.pipeline_seams &&
                  locked->parameters.pipeline_seams->before_reference_publish) {
                locked->parameters.pipeline_seams->before_reference_publish();
              }
              locked->reference_publisher->publish(
                  std::move(*selection.publish_reference));
            } catch (const std::length_error&) {
              FailLocked(*locked, "RESOURCE_RESULT_COPY");
            } catch (const std::bad_alloc&) {
              FailLocked(*locked, "RESOURCE_RESULT_COPY");
            } catch (const std::exception&) {
              FailLocked(*locked, "FINAL_SELECTION_ERROR");
            }
          }
        }
        PublishStatus(locked);
        if (rebuild) {
          QueueBuild(locked);
        }
      });

  if (!submitted) {
    {
      std::scoped_lock lock{runtime->mutex};
      if (!runtime->teardown && runtime->active_cycle == cycle &&
          runtime->epoch == epoch && runtime->final_rank_in_flight) {
        FailLocked(*runtime, "SERIAL_WORK_QUEUE_STOPPED");
      }
    }
    PublishStatus(runtime);
  }
}

template <typename RuntimeT>
void QueueApproachFinalRank(const std::shared_ptr<RuntimeT>& runtime,
                            const FrozenApproachCyclePtr& cycle,
                            const std::uint64_t epoch,
                            std::vector<PlannedCandidate> planned) {
  std::shared_ptr<SerializedWorkQueue> queue;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown || runtime->active_approach_cycle != cycle ||
        runtime->epoch != epoch || !runtime->final_rank_in_flight) {
      return;
    }
    queue = runtime->work_queue;
  }

  const std::weak_ptr<RuntimeT> weak_runtime{runtime};
  const bool submitted = queue && queue->Submit(
      [weak_runtime, cycle, epoch, planned = std::move(planned)]() mutable {
        struct FinalSelection final {
          std::optional<ActiveGoal> goal;
          std::optional<lunar::pure_exploration::ApproachCandidate> candidate;
          std::optional<MotionReference> active_reference;
          std::optional<MotionReference> publish_reference;
          std::vector<Vec2> executable_polyline;
          std::optional<Vec2> executable_endpoint;
          std::string error;
        };
        struct LatestValidation final {
          FrozenGlobalMapContent map;
          Polygon2 boundary;
        };

        const auto locked = weak_runtime.lock();
        if (!locked) {
          return;
        }
        FinalSelection selection;
        double final_rank_elapsed_ms = -1.0;
        try {
          const auto rank_start = SteadyNowLocked(*locked);
          const auto order = locked->boundary_guidance.FinalOrder(
              cycle->guidance(), planned, cycle->frozen_robot_pose());
          final_rank_elapsed_ms = SteadyElapsedMs(*locked, rank_start);
          if (order.empty()) {
            throw std::logic_error{"reachable approach rank is empty"};
          }
          const std::size_t selected_index = order.front();
          const auto selected = std::ranges::find_if(
              cycle->reachable(), [&](const auto& row) {
                return row.metrics.candidate_index == selected_index;
              });
          if (selected == cycle->reachable().end()) {
            throw std::logic_error{"selected approach reference is missing"};
          }
          const auto request_id =
              cycle->LatestRequestIdForCandidate(selected_index);
          if (!request_id) {
            throw std::logic_error{"selected approach request is missing"};
          }
          const auto& candidate =
              cycle->guidance().candidates.at(selected_index);
          const ExecutableReference executable = ValidateExecutableReference(
              selected->reference, locked->parameters.platform.platform_type,
              locked->parameters.maximum_executable_path_points);
          selection.goal.emplace(MakeBoundaryApproachActiveGoal(
              candidate.id, candidate.identity, candidate.pose, *request_id));
          selection.candidate = candidate;
          const auto copy_reference = [&](const MotionReference& source) {
            return locked->parameters.pipeline_seams &&
                           locked->parameters.pipeline_seams->copy_reference
                       ? locked->parameters.pipeline_seams->copy_reference(
                             source)
                       : MotionReference{source};
          };
          selection.active_reference.emplace(copy_reference(selected->reference));
          selection.publish_reference.emplace(copy_reference(selected->reference));
          selection.executable_polyline = executable.polyline;
          selection.executable_endpoint = executable.endpoint;
        } catch (const std::length_error&) {
          selection.error = "RESOURCE_EXECUTABLE_PATH_LIMIT";
        } catch (const std::bad_alloc&) {
          selection.error = "RESOURCE_RESULT_COPY";
        } catch (const std::overflow_error&) {
          selection.error = "NUMERICAL_OVERFLOW";
        } catch (const std::exception&) {
          selection.error = "FINAL_SELECTION_ERROR";
        }

        std::optional<LatestValidation> validation;
        {
          std::scoped_lock lock{locked->mutex};
          if (locked->teardown || locked->active_approach_cycle != cycle ||
              locked->epoch != epoch || !locked->final_rank_in_flight ||
              locked->state_machine.state() !=
                  ExplorationState::kSelectingFrontier) {
            return;
          }
          if (!locked->latest_map || !locked->task_boundary ||
              !selection.candidate) {
            selection.error = "FINAL_MAP_VALIDATION_ERROR";
          } else {
            validation.emplace(LatestValidation{
                .map = *locked->latest_map,
                .boundary = *locked->task_boundary});
          }
        }
        if (locked->parameters.pipeline_seams &&
            locked->parameters.pipeline_seams->after_final_rank_map_observed) {
          locked->parameters.pipeline_seams->after_final_rank_map_observed();
        }

        bool candidate_valid = false;
        std::string validation_error;
        if (validation && selection.error.empty()) {
          try {
            OccupancyGridView latest_view(
                validation->map.geometry, validation->map.data,
                locked->parameters.global_occupied_threshold);
            const TaskRaster latest_raster = TaskRaster::Build(
                latest_view, validation->boundary,
                locked->parameters.task_raster_limits);
            try {
              candidate_valid =
                  locked->boundary_guidance.IsCandidateStillValid(
                      latest_view, latest_raster, *selection.candidate);
            } catch (const std::length_error& error) {
              validation_error = GuidanceResourceReason(error);
            }
          } catch (const std::length_error& error) {
            validation_error = ResourceReason(error);
          } catch (const std::bad_alloc&) {
            validation_error = "RESOURCE_RESULT_COPY";
          } catch (const std::exception&) {
            validation_error = "FINAL_MAP_VALIDATION_ERROR";
          }
        }

        bool rebuild = false;
        {
          std::unique_lock lock{locked->mutex};
          if (locked->teardown || locked->active_approach_cycle != cycle ||
              locked->epoch != epoch || !locked->final_rank_in_flight ||
              locked->state_machine.state() !=
                  ExplorationState::kSelectingFrontier) {
            return;
          }
          RecordLocalTiming(locked->final_rank_timing, final_rank_elapsed_ms);
          if (!selection.error.empty()) {
            FailLocked(*locked, std::move(selection.error));
          } else if (!validation_error.empty()) {
            FailLocked(*locked, std::move(validation_error));
          } else if (!validation || !locked->latest_map ||
                     !SameGeometry(validation->map.geometry,
                                   locked->latest_map->geometry) ||
                     validation->map.data != locked->latest_map->data ||
                     !candidate_valid) {
            ClearSelectionCycleLocked(*locked);
            ++locked->epoch;
            ++locked->build_generation;
            rebuild = true;
          } else if (!selection.goal || !selection.active_reference ||
                     !selection.publish_reference || !selection.candidate) {
            FailLocked(*locked, "FINAL_SELECTION_ERROR");
          } else {
            try {
              locked->active_reference.emplace(
                  std::move(*selection.active_reference));
              locked->active_executable_polyline =
                  std::move(selection.executable_polyline);
              locked->active_executable_endpoint =
                  selection.executable_endpoint;
              const auto pose = locked->pose_resolver.LatestPoseInMap();
              if (!pose) {
                throw std::logic_error{"missing execution pose"};
              }
              const auto now = SteadyNowLocked(*locked);
              locked->execution_monitor.emplace(ExecutionMonitorParameters{
                  .maximum_executable_path_points =
                      locked->parameters.maximum_executable_path_points,
                  .position_tolerance_m = std::max(
                      0.5 * cycle->frozen_resolution_m(),
                      0.25 * locked->candidate_generator.platform_width_m()),
                  .yaw_tolerance_rad =
                      locked->parameters.goal_yaw_tolerance_rad,
                  .stuck_window = std::chrono::seconds{30},
                  .minimum_progress_m = 0.2});
              locked->execution_monitor->ResetProgress(
                  now, locked->active_executable_polyline,
                  {pose->x, pose->y});
              if (!locked->approach_observability) {
                throw std::logic_error{
                    "missing approach observability authority"};
              }
              locked->approach_observability->markers.selected_identity =
                  selection.candidate->identity;
              locked->approach_observability->markers.selected_guidance =
                  selection.candidate->guidance_route;
              locked->approach_observability->remaining_guidance_m =
                  selection.candidate->remaining_cost.path_length_m;
              locked->approach_observability
                  ->guidance_unknown_cell_count =
                  selection.candidate->guidance_unknown_cell_count;
              locked->approach_observability->wait_reason.reset();
              locked->state_machine.BeginPlanning();
              locked->status_reason_override = "PLANNING_APPROACH";
              Status planning_status = MakeStatusLocked(*locked);
              const auto status_publisher = locked->status_publisher;
              status_publisher->publish(std::move(planning_status));
              if (locked->parameters.pipeline_seams &&
                  locked->parameters.pipeline_seams
                      ->before_approach_goal_commit) {
                locked->parameters.pipeline_seams
                    ->before_approach_goal_commit();
              }
              if (locked->teardown ||
                  locked->active_approach_cycle != cycle ||
                  locked->epoch != epoch ||
                  locked->state_machine.state() !=
                      ExplorationState::kPlanning) {
                return;
              }
              locked->state_machine.CommitGoal(std::move(*selection.goal));
              locked->status_reason_override = "EXECUTING_APPROACH";
              if (locked->snapshot_to_goal_start &&
                  locked->snapshot_to_goal_start->approach_cycle.lock() ==
                      cycle &&
                  locked->snapshot_to_goal_start->epoch == locked->epoch &&
                  locked->snapshot_to_goal_start->build_generation ==
                      locked->build_generation) {
                RecordLocalTiming(
                    locked->snapshot_to_goal_timing,
                    SteadyElapsedMs(*locked,
                                    locked->snapshot_to_goal_start->start));
                locked->snapshot_to_goal_start.reset();
              }
              locked->pending_execution_cancel_sent = false;
              locked->final_rank_in_flight = false;
              if (locked->parameters.pipeline_seams &&
                  locked->parameters.pipeline_seams->before_reference_publish) {
                locked->parameters.pipeline_seams->before_reference_publish();
              }
              locked->reference_publisher->publish(
                  std::move(*selection.publish_reference));
            } catch (const std::length_error&) {
              FailLocked(*locked, "RESOURCE_RESULT_COPY");
            } catch (const std::bad_alloc&) {
              FailLocked(*locked, "RESOURCE_RESULT_COPY");
            } catch (const std::exception&) {
              FailLocked(*locked, "FINAL_SELECTION_ERROR");
            }
          }
        }
        PublishStatus(locked);
        if (rebuild) {
          QueueBuild(locked);
        }
      });

  if (!submitted) {
    {
      std::scoped_lock lock{runtime->mutex};
      if (!runtime->teardown && runtime->active_approach_cycle == cycle &&
          runtime->epoch == epoch && runtime->final_rank_in_flight) {
        FailLocked(*runtime, "SERIAL_WORK_QUEUE_STOPPED");
      }
    }
    PublishStatus(runtime);
  }
}

template <typename RuntimeT>
void PumpApproach(const std::shared_ptr<RuntimeT>& runtime) {
  struct EvaluationCall final {
    std::string task_id;
    std::string request_id;
    FrozenApproachCyclePtr cycle;
    std::size_t candidate_index;
    double position_tolerance_m;
    double yaw_tolerance_rad;
    std::uint64_t epoch;
  };
  struct FinalRankCall final {
    FrozenApproachCyclePtr cycle;
    std::uint64_t epoch;
    std::vector<PlannedCandidate> planned;
  };
  std::optional<EvaluationCall> call;
  std::optional<FinalRankCall> final_rank_call;
  bool rebuild = false;
  bool publish_status = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown || runtime->planner_in_flight ||
        runtime->final_rank_in_flight ||
        runtime->pending_control != RuntimeT::PendingControl::kNone ||
        !PlanningIsAdmittedLocked(*runtime) ||
        !runtime->active_approach_cycle || runtime->active_cycle ||
        runtime->state_machine.state() !=
            ExplorationState::kSelectingFrontier) {
      return;
    }
    auto cycle = runtime->active_approach_cycle;
    if (!runtime->current_candidate) {
      if (runtime->current_batch_cursor >= runtime->current_batch.size()) {
        runtime->current_batch = cycle->TakeNextCandidateIndices();
        runtime->current_batch_cursor = 0U;
        if (runtime->current_batch.empty()) {
          if (!runtime->latest_map_message ||
              !cycle->GlobalMapContentEquals(*runtime->latest_map_message)) {
            runtime->active_approach_cycle.reset();
            runtime->active_raster.reset();
            runtime->status_reason_override.reset();
            ++runtime->build_generation;
            rebuild = true;
          } else if (!cycle->reachable().empty()) {
            std::vector<PlannedCandidate> planned;
            planned.reserve(cycle->reachable().size());
            for (const auto& reachable : cycle->reachable()) {
              planned.push_back(reachable.metrics);
            }
            runtime->final_rank_in_flight = true;
            final_rank_call.emplace(FinalRankCall{
                .cycle = cycle,
                .epoch = runtime->epoch,
                .planned = std::move(planned)});
          } else {
            runtime->active_approach_cycle.reset();
            runtime->active_raster.reset();
            BlockApproachSnapshotLocked(
                *runtime, "APPROACH_NO_REACHABLE_TARGET");
            publish_status = true;
          }
        }
      }
      if (!final_rank_call && !rebuild &&
          runtime->active_approach_cycle &&
          runtime->current_batch_cursor < runtime->current_batch.size()) {
        runtime->current_candidate =
            runtime->current_batch[runtime->current_batch_cursor];
        runtime->current_candidate_retryable_retries = 0U;
      }
    }

    if (!final_rank_call && !rebuild && runtime->current_candidate &&
        runtime->active_approach_cycle) {
      const std::size_t candidate_index = *runtime->current_candidate;
      if (runtime->request_sequence ==
          std::numeric_limits<std::uint64_t>::max()) {
        FailLocked(*runtime, "REQUEST_ID_SEQUENCE_EXHAUSTED");
        publish_status = true;
      }
      const std::string request_id =
          publish_status
              ? std::string{}
              : PlannerClient::MakeRequestId(
                    runtime->state_machine.task_id(),
                    runtime->request_sequence++);
      if (!publish_status) {
        try {
          cycle->RegisterRequest(request_id, candidate_index);
          RegisterPlannerTimingRequestLocked(
              *runtime, request_id,
              RuntimeT::PlannerTimingBucket::kCandidate);
        } catch (const std::exception&) {
          FailLocked(*runtime, "REQUEST_REGISTRATION_ERROR");
          publish_status = true;
        }
      }
      if (!publish_status) {
        runtime->planner_in_flight = true;
        runtime->candidate_retry_deadline.reset();
        runtime->status_reason_override = "PLANNING_APPROACH";
        call.emplace(EvaluationCall{
            .task_id = runtime->state_machine.task_id(),
            .request_id = request_id,
            .cycle = cycle,
            .candidate_index = candidate_index,
            .position_tolerance_m = std::max(
                0.5 * cycle->frozen_resolution_m(),
                0.25 * runtime->candidate_generator.platform_width_m()),
            .yaw_tolerance_rad =
                runtime->parameters.goal_yaw_tolerance_rad,
            .epoch = runtime->epoch});
      }
    }
  }

  if (publish_status) {
    PublishStatus(runtime);
  }
  if (final_rank_call) {
    QueueApproachFinalRank(runtime, final_rank_call->cycle,
                           final_rank_call->epoch,
                           std::move(final_rank_call->planned));
    return;
  }
  if (rebuild) {
    QueueBuild(runtime);
    return;
  }
  if (!call) {
    return;
  }
  const auto& candidate =
      call->cycle->guidance().candidates.at(call->candidate_index);
  const std::weak_ptr<RuntimeT> weak_runtime{runtime};
  try {
    runtime->planner_client->Evaluate(
        call->task_id, call->request_id,
        PlannerTarget{.display_id = candidate.id, .pose = candidate.pose},
        call->position_tolerance_m, call->yaw_tolerance_rad,
        [weak_runtime, cycle = call->cycle, epoch = call->epoch](
            PlannerEvaluation evaluation) mutable {
          HandleApproachEvaluation(weak_runtime, cycle, epoch,
                                   std::move(evaluation));
        });
  } catch (const std::exception&) {
    {
      std::scoped_lock lock{runtime->mutex};
      if (!runtime->teardown &&
          runtime->active_approach_cycle == call->cycle &&
          runtime->epoch == call->epoch) {
        runtime->planner_in_flight = false;
        FailLocked(*runtime, "PLANNER_CLIENT_ERROR");
      }
    }
    PublishStatus(runtime);
  }
}

template <typename RuntimeT>
void Pump(const std::shared_ptr<RuntimeT>& runtime) {
  struct EvaluationCall {
    std::string task_id;
    std::string request_id;
    FrozenPlanningCyclePtr cycle;
    std::size_t candidate_index;
    double position_tolerance_m;
    double yaw_tolerance_rad;
    std::uint64_t epoch;
  };
  std::optional<EvaluationCall> call;
  struct FinalRankCall final {
    FrozenPlanningCyclePtr cycle;
    std::uint64_t epoch;
    std::vector<PlannedCandidate> planned;
    std::vector<Vec2> completed_goal_positions;
  };
  std::optional<FinalRankCall> final_rank_call;
  bool rebuild = false;
  bool publish_status = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown || runtime->planner_in_flight ||
        runtime->final_rank_in_flight ||
        runtime->pending_control != RuntimeT::PendingControl::kNone ||
        !PlanningIsAdmittedLocked(*runtime) ||
        !runtime->active_cycle ||
        runtime->state_machine.state() != ExplorationState::kSelectingFrontier) {
      return;
    }
    auto cycle = runtime->active_cycle;
    if (!runtime->current_candidate) {
      if (runtime->current_batch_cursor >= runtime->current_batch.size()) {
        if (!cycle->reachable().empty()) {
          std::vector<PlannedCandidate> planned;
          planned.reserve(cycle->reachable().size());
          for (const auto& reachable : cycle->reachable()) {
            planned.push_back(reachable.metrics);
          }
          runtime->final_rank_in_flight = true;
          final_rank_call.emplace(FinalRankCall{
              .cycle = cycle,
              .epoch = runtime->epoch,
              .planned = std::move(planned),
              .completed_goal_positions =
                  runtime->completed_goal_positions});
        } else {
          runtime->current_batch = cycle->TakeNextCandidateIndices();
          runtime->current_batch_cursor = 0U;
          if (runtime->current_batch.empty()) {
            if (!runtime->latest_map_message ||
                !cycle->batch()->GlobalMapContentEquals(
                    *runtime->latest_map_message)) {
              runtime->active_cycle.reset();
              runtime->active_raster.reset();
              rebuild = true;
            } else if (WfdCompletionProvenLocked(*runtime, cycle)) {
              runtime->state_machine.CompleteNoReachableFrontier();
              publish_status = true;
            } else {
              FailLocked(*runtime, "INVALID_COMPLETION_AUTHORITY");
              publish_status = true;
            }
          }
        }
      }
      if (!final_rank_call && !rebuild &&
          runtime->state_machine.state() ==
              ExplorationState::kSelectingFrontier &&
          runtime->current_batch_cursor < runtime->current_batch.size()) {
        runtime->current_candidate =
            runtime->current_batch[runtime->current_batch_cursor];
        runtime->current_candidate_retryable_retries = 0U;
      }
    }

    if (!final_rank_call && !rebuild && runtime->current_candidate &&
        runtime->state_machine.state() ==
            ExplorationState::kSelectingFrontier) {
      const std::size_t candidate_index = *runtime->current_candidate;
      if (runtime->request_sequence ==
          std::numeric_limits<std::uint64_t>::max()) {
        FailLocked(*runtime, "REQUEST_ID_SEQUENCE_EXHAUSTED");
        publish_status = true;
      }
      const std::string request_id =
          publish_status
              ? std::string{}
              : PlannerClient::MakeRequestId(
                    runtime->state_machine.task_id(),
                    runtime->request_sequence++);
      if (!publish_status) {
        try {
          cycle->batch()->RegisterRequest(request_id, candidate_index);
          RegisterPlannerTimingRequestLocked(
              *runtime, request_id,
              RuntimeT::PlannerTimingBucket::kCandidate);
        } catch (const std::exception&) {
          FailLocked(*runtime, "REQUEST_REGISTRATION_ERROR");
          publish_status = true;
        }
      }
      if (!publish_status) {
        runtime->planner_in_flight = true;
        runtime->candidate_retry_deadline.reset();
        call = EvaluationCall{
            .task_id = runtime->state_machine.task_id(),
            .request_id = request_id,
            .cycle = cycle,
            .candidate_index = candidate_index,
            .position_tolerance_m = std::max(
                0.5 * cycle->frozen_resolution_m(),
                0.25 * runtime->candidate_generator.platform_width_m()),
            .yaw_tolerance_rad = runtime->parameters.goal_yaw_tolerance_rad,
            .epoch = runtime->epoch};
      }
    }
  }

  if (publish_status) {
    PublishStatus(runtime);
  }
  if (final_rank_call) {
    QueueFinalRank(runtime, final_rank_call->cycle, final_rank_call->epoch,
                   std::move(final_rank_call->planned),
                   std::move(final_rank_call->completed_goal_positions));
    return;
  }
  if (rebuild) {
    QueueBuild(runtime);
    return;
  }
  if (!call) {
    return;
  }
  const std::weak_ptr<RuntimeT> weak_runtime{runtime};
  const auto cycle = call->cycle;
  const auto candidate =
      cycle->batch()->candidates()[call->candidate_index];
  try {
    runtime->planner_client->Evaluate(
        call->task_id, call->request_id,
        PlannerTarget{.display_id = candidate.id, .pose = candidate.pose},
        call->position_tolerance_m, call->yaw_tolerance_rad,
        [weak_runtime, cycle, epoch = call->epoch](
            PlannerEvaluation evaluation) mutable {
          HandleEvaluation(weak_runtime, cycle, epoch,
                           std::move(evaluation));
        });
  } catch (const std::exception&) {
    {
      std::scoped_lock lock{runtime->mutex};
      if (!runtime->teardown && runtime->active_cycle == cycle &&
          runtime->epoch == call->epoch) {
        runtime->planner_in_flight = false;
        FailLocked(*runtime, "PLANNER_CLIENT_ERROR");
      }
    }
    PublishStatus(runtime);
  }
}

template <typename RuntimeT>
void StartExecutionReplan(const std::shared_ptr<RuntimeT>& runtime,
                          const bool stationary_confirmed) {
  struct Call {
    std::string task_id;
    std::string request_id;
    FrozenPlanningCyclePtr wfd_cycle;
    FrozenApproachCyclePtr approach_cycle;
    PlannerTarget target;
    bool approach;
    double resolution_m;
    double position_tolerance_m;
    double yaw_tolerance_rad;
    std::uint64_t epoch;
  };
  std::optional<Call> call;
  std::optional<std::string> execution_cancel;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown || runtime->planner_in_flight ||
        runtime->state_machine.state() != ExplorationState::kReplanning ||
        !HasFrozenCycleLocked(*runtime) ||
        !runtime->state_machine.active_goal()) {
      return;
    }
    if (!stationary_confirmed &&
        runtime->parameters.stop_before_planning &&
        PlanningIsAdmittedLocked(*runtime)) {
      execution_cancel = RequestStationaryActionLocked(
          *runtime, RuntimeT::PendingStationaryAction::kReplanActiveGoal);
    }
    if (!PlanningIsAdmittedLocked(*runtime)) {
      // The odometry handler dispatches the replan after confirmation.
    } else {
      const bool approach =
          runtime->state_machine.active_goal()->kind() ==
          lunar::pure_exploration::GoalKind::kBoundaryApproach;
      std::optional<std::size_t> selected_index;
      PlannerTarget target{};
      double resolution_m = 0.0;
      if (approach && runtime->active_approach_cycle) {
        const auto* identity = runtime->state_machine.active_goal()
                                   ->boundary_approach_identity();
        const auto& candidates =
            runtime->active_approach_cycle->guidance().candidates;
        const auto selected = identity
                                  ? std::ranges::find_if(
                                        candidates, [&](const auto& candidate) {
                                          return candidate.identity == *identity;
                                        })
                                  : candidates.end();
        if (selected != candidates.end()) {
          selected_index = static_cast<std::size_t>(
              selected - candidates.begin());
          target = {.display_id = selected->id, .pose = selected->pose};
          resolution_m =
              runtime->active_approach_cycle->frozen_resolution_m();
        }
      } else if (!approach && runtime->active_cycle) {
        const auto& candidates = runtime->active_cycle->batch()->candidates();
        const auto selected = std::ranges::find_if(
            candidates, [&](const auto& candidate) {
              return candidate.key ==
                     runtime->state_machine.active_goal()->candidate_key();
            });
        if (selected != candidates.end()) {
          selected_index = static_cast<std::size_t>(
              selected - candidates.begin());
          target = {.display_id = selected->id, .pose = selected->pose};
          resolution_m = runtime->active_cycle->frozen_resolution_m();
        }
      }
      if (!selected_index ||
          runtime->request_sequence ==
              std::numeric_limits<std::uint64_t>::max()) {
        FailLocked(*runtime, "EXECUTION_REPLAN_REQUEST_ERROR");
        return;
      }
      const std::string request_id = PlannerClient::MakeRequestId(
          runtime->state_machine.task_id(), runtime->request_sequence++);
      const auto bucket = runtime->state_machine.reason_code() == "STUCK_RETRY"
                              ? RuntimeT::PlannerTimingBucket::kStuck
                              : RuntimeT::PlannerTimingBucket::kRolling;
      try {
        if (approach) {
          runtime->active_approach_cycle->RegisterRequest(request_id,
                                                          *selected_index);
        } else {
          runtime->active_cycle->batch()->RegisterRequest(request_id,
                                                          *selected_index);
        }
        RegisterPlannerTimingRequestLocked(*runtime, request_id, bucket);
      } catch (const std::exception&) {
        FailLocked(*runtime, "EXECUTION_REPLAN_REQUEST_ERROR");
        return;
      }
      runtime->planner_in_flight = true;
      runtime->execution_replan_request_id = request_id;
      runtime->execution_replan_bucket = bucket;
      runtime->execution_replan_retry_pending = false;
      if (approach) {
        runtime->status_reason_override = "PLANNING_APPROACH";
      }
      call.emplace(
          Call{.task_id = runtime->state_machine.task_id(),
               .request_id = request_id,
               .wfd_cycle = runtime->active_cycle,
               .approach_cycle = runtime->active_approach_cycle,
               .target = target,
               .approach = approach,
               .resolution_m = resolution_m,
               .position_tolerance_m = std::max(
                   0.5 * resolution_m,
                   0.25 * runtime->candidate_generator.platform_width_m()),
               .yaw_tolerance_rad =
                   runtime->parameters.goal_yaw_tolerance_rad,
               .epoch = runtime->epoch});
    }
  }
  if (execution_cancel) {
    std_msgs::msg::String message;
    message.data = *execution_cancel;
    runtime->cancel_publisher->publish(std::move(message));
  }
  if (!call) {
    PublishStatus(runtime);
    return;
  }
  const std::weak_ptr<RuntimeT> weak_runtime{runtime};
  try {
    runtime->planner_client->Evaluate(
        call->task_id, call->request_id, call->target,
        call->position_tolerance_m, call->yaw_tolerance_rad,
        [weak_runtime, wfd_cycle = call->wfd_cycle,
         approach_cycle = call->approach_cycle, approach = call->approach,
         resolution_m = call->resolution_m, epoch = call->epoch](
            PlannerEvaluation evaluation) mutable {
          const auto locked = weak_runtime.lock();
          if (!locked) {
            return;
          }
          std::optional<MotionReference> publish_reference;
          bool rebuild = false;
          bool build = false;
          bool pump = false;
          {
            std::scoped_lock lock{locked->mutex};
            if (locked->teardown) {
              return;
            }
            locked->planner_in_flight = false;
            if (locked->pending_control != RuntimeT::PendingControl::kNone) {
              build = ApplyPendingControlLocked(*locked);
            } else if (const auto pending_rebuild =
                           ResolvePendingRebuildAfterPlannerTerminalLocked(
                               *locked);
                       pending_rebuild != PlannerTerminalRebuild::kNone) {
              rebuild = pending_rebuild == PlannerTerminalRebuild::kBuildNow;
            } else if ((approach
                            ? locked->active_approach_cycle != approach_cycle
                            : locked->active_cycle != wfd_cycle) ||
                       locked->epoch != epoch ||
                       locked->state_machine.state() !=
                           ExplorationState::kReplanning ||
                       !locked->state_machine.active_goal()) {
              if (locked->state_machine.state() ==
                      ExplorationState::kSelectingFrontier &&
                  PlanningIsAdmittedLocked(*locked)) {
                pump = approach
                           ? static_cast<bool>(locked->active_approach_cycle)
                           : static_cast<bool>(locked->active_cycle);
                build = !HasFrozenCycleLocked(*locked) &&
                        !locked->build_in_flight;
              }
            } else if (evaluation.kind == PlannerEvaluationKind::kReachable) {
              if (!evaluation.path_length_m || !evaluation.reference) {
                FailLocked(*locked, "PLANNER_REACHABLE_CONTRACT_ERROR");
              } else {
              try {
                const ExecutableReference executable = ValidateExecutableReference(
                    *evaluation.reference, locked->parameters.platform.platform_type,
                    locked->parameters.maximum_executable_path_points);
                const auto pose = locked->pose_resolver.LatestPoseInMap();
                if (!pose) {
                  throw std::logic_error{"missing execution pose"};
                }
                const auto now = locked->parameters.steady_now
                                     ? locked->parameters.steady_now()
                                     : std::chrono::steady_clock::now();
                locked->execution_monitor.emplace(ExecutionMonitorParameters{
                    .maximum_executable_path_points =
                        locked->parameters.maximum_executable_path_points,
                    .position_tolerance_m = std::max(
                        0.5 * resolution_m,
                        0.25 * locked->candidate_generator.platform_width_m()),
                    .yaw_tolerance_rad = locked->parameters.goal_yaw_tolerance_rad,
                    .stuck_window = std::chrono::seconds{30},
                    .minimum_progress_m = 0.2});
                locked->execution_monitor->ResetProgress(
                    now, executable.polyline, {pose->x, pose->y});
                locked->active_reference = std::move(*evaluation.reference);
                locked->active_executable_polyline = executable.polyline;
                locked->active_executable_endpoint = executable.endpoint;
                publish_reference = *locked->active_reference;
                locked->pending_execution_cancel_sent = false;
                locked->state_machine.ResumeExecution();
                if (approach) {
                  locked->status_reason_override = "EXECUTING_APPROACH";
                }
              } catch (const std::length_error&) {
                FailLocked(*locked, "RESOURCE_EXECUTABLE_PATH_LIMIT");
              } catch (const std::bad_alloc&) {
                FailLocked(*locked, "RESOURCE_RESULT_COPY");
              } catch (const std::exception&) {
                FailLocked(*locked, "INVALID_EXECUTABLE_REFERENCE");
              }
              }
            } else if (evaluation.kind ==
                       PlannerEvaluationKind::kExhaustiveNoPath) {
              locked->state_machine.ReleaseGoal(
                  lunar::pure_exploration::GoalReleaseReason::kNoPath);
              ClearExecutionPayloadsLocked(*locked);
              ClearSelectionCycleLocked(*locked);
              locked->execution_replan_request_id.reset();
              locked->execution_replan_bucket.reset();
              locked->execution_replan_waiting_terminal = false;
              locked->execution_replan_retry_pending = false;
              ++locked->epoch;
              ++locked->build_generation;
              if (approach) {
                BlockApproachSnapshotLocked(*locked, "APPROACH_STALLED");
              } else {
                rebuild = true;
              }
            } else if (evaluation.kind == PlannerEvaluationKind::kRetryable) {
              locked->execution_replan_retry_pending = true;
            } else if (evaluation.kind == PlannerEvaluationKind::kCanceled) {
              FailLocked(*locked, "UNEXPECTED_PLANNER_CANCELED");
            } else if (evaluation.kind ==
                       PlannerEvaluationKind::kContractError) {
              FailLocked(*locked,
                         "PLANNER_CONTRACT_" + evaluation.reason_code);
            } else if (evaluation.kind == PlannerEvaluationKind::kResourceError) {
              switch (evaluation.resource_kind) {
                case PlannerResourceKind::kPathPreviewPoses:
                  FailLocked(*locked, "RESOURCE_PATH_PREVIEW_LIMIT");
                  break;
                case PlannerResourceKind::kExecutablePathPoints:
                  FailLocked(*locked, "RESOURCE_EXECUTABLE_PATH_LIMIT");
                  break;
                case PlannerResourceKind::kResultCopyFailure:
                  FailLocked(*locked, "RESOURCE_RESULT_COPY");
                  break;
                case PlannerResourceKind::kNone:
                  FailLocked(*locked,
                             "PLANNER_RESOURCE_CLASSIFICATION_ERROR");
                  break;
              }
            }
          }
          if (publish_reference) {
            locked->reference_publisher->publish(*publish_reference);
          }
          PublishStatus(locked);
          if (rebuild) {
            QueueBuild(locked);
          } else if (build) {
            QueueBuild(locked);
          } else if (pump) {
            if (approach) {
              PumpApproach(locked);
            } else {
              Pump(locked);
            }
          }
        });
  } catch (const std::exception&) {
    std::scoped_lock lock{runtime->mutex};
    runtime->planner_in_flight = false;
    FailLocked(*runtime, "PLANNER_CLIENT_ERROR");
  }
}

template <typename RuntimeT>
bool CandidateRemainsValidOnLatestMap(
    RuntimeT& runtime, const FrozenGlobalMapContent& map,
    const Polygon2& boundary, const Pose2& robot_pose,
    const lunar::pure_exploration::CandidateView& candidate) {
  OccupancyGridView view(map.geometry, map.data,
                         runtime.parameters.global_occupied_threshold);
  const TaskRaster raster = TaskRaster::Build(
      view, boundary, runtime.parameters.task_raster_limits);
  const auto robot_cell = raster.WorldToCell({robot_pose.x, robot_pose.y});
  if (!robot_cell) {
    return false;
  }
  FrontierDetector detector(FrontierParameters{
      .minimum_cluster_length_m = std::max(
          runtime.candidate_generator.platform_width_m(),
          2.0 * map.geometry.resolution)});
  const auto detection = detector.Detect(raster, *robot_cell);
  if (!detection.has_reachable_free_start) {
    return false;
  }
  if (runtime.parameters.filter_global_goal_cell &&
      !GlobalGoalCellFeasible(
          map, candidate.pose, runtime.candidate_generator.minimum_standoff_m(),
          runtime.parameters.global_occupied_threshold)) {
    return false;
  }
  const auto candidates =
      runtime.parameters.pipeline_seams &&
              runtime.parameters.pipeline_seams->generate_candidates
          ? runtime.parameters.pipeline_seams->generate_candidates(
                raster, detection.clusters)
          : runtime.candidate_generator.Generate(raster, detection.clusters);
  const auto current = std::ranges::find_if(
      candidates, [&](const auto& current_candidate) {
        return current_candidate.key == candidate.key &&
               current_candidate.frontier_canonical_key &&
               candidate.frontier_canonical_key &&
               *current_candidate.frontier_canonical_key ==
                   *candidate.frontier_canonical_key;
      });
  if (current == candidates.end()) {
    return false;
  }
  const double gain = runtime.parameters.pipeline_seams &&
                              runtime.parameters.pipeline_seams->evaluate_gain
                          ? runtime.parameters.pipeline_seams->evaluate_gain(
                                raster, *current)
                          : runtime.information_gain.Evaluate(raster, *current)
                                .visible_unknown_area_m2;
  return std::isfinite(gain) && gain > 0.0;
}

template <typename RuntimeT>
void HandleGlobalMap(const std::weak_ptr<RuntimeT>& weak_runtime,
                     const nav_msgs::msg::OccupancyGrid& message) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  bool cancel = false;
  bool needs_build = false;
  bool failed = false;
  std::optional<std::string> execution_cancel;
  struct ActiveGoalValidation final {
    std::uint64_t generation{0U};
    std::uint64_t epoch{0U};
    std::string task_id;
    std::string plan_id;
    std::string goal_request_id;
    lunar::pure_exploration::GoalKind kind;
    FrozenGlobalMapContent map;
    Polygon2 boundary;
    std::optional<Pose2> pose;
    std::optional<lunar::pure_exploration::CandidateView> wfd_candidate;
    std::optional<lunar::pure_exploration::ApproachCandidate>
        approach_candidate;
  };
  std::optional<ActiveGoalValidation> validation;
  bool validation_authority_error = false;
  try {
    auto content = FreezeGlobalMap(
        message, runtime->parameters.global_occupied_threshold);
    {
      std::scoped_lock lock{runtime->mutex};
      if (runtime->teardown) {
        return;
      }
      runtime->latest_map = std::move(content);
      runtime->latest_map_message = message;
      ++runtime->map_generation;
      if (runtime->state_machine.state() ==
              ExplorationState::kSelectingFrontier &&
          runtime->active_approach_cycle &&
          !runtime->active_approach_cycle->GlobalMapContentEquals(message)) {
        runtime->status_reason_override.reset();
        if (runtime->planner_in_flight) {
          runtime->pending_map_rebuild = true;
          cancel = true;
        } else {
          ClearSelectionCycleLocked(*runtime);
          ++runtime->epoch;
          ++runtime->build_generation;
          needs_build = true;
        }
      }
      if (runtime->pending_stationary_action !=
              RuntimeT::PendingStationaryAction::kBuildFreshBatch &&
          runtime->state_machine.active_goal()) {
        const auto& active_goal = *runtime->state_machine.active_goal();
        const auto pose = runtime->pose_resolver.LatestPoseInMap();
        // Stationary-confirmed execution replanning intentionally clears the
        // old reference before dispatching its replacement.  The frozen goal
        // and cycle remain authoritative during that bounded transition.  A
        // changed map must cancel the in-flight replan and enter the fresh-map
        // stop/rebuild gate; same-content input needs no reference validation
        // because no reference is executing yet.
        const bool execution_replan_without_reference =
            runtime->state_machine.state() ==
                ExplorationState::kReplanning &&
            runtime->planner_in_flight &&
            runtime->execution_replan_request_id &&
            !runtime->active_reference;
        validation_authority_error =
            !runtime->task_boundary || !pose ||
            (!runtime->active_reference &&
             !execution_replan_without_reference);
        if (!validation_authority_error &&
            active_goal.kind() ==
                lunar::pure_exploration::GoalKind::kTaskFrontier) {
          if (!runtime->active_cycle) {
            validation_authority_error = true;
          } else {
            const auto selected = std::ranges::find_if(
                runtime->active_cycle->batch()->candidates(),
                [&](const auto& candidate) {
                  return candidate.key == active_goal.candidate_key() &&
                         candidate.frontier_canonical_key &&
                         std::ranges::equal(
                             *candidate.frontier_canonical_key,
                             active_goal.frontier_canonical_key());
                });
            validation_authority_error =
                selected ==
                runtime->active_cycle->batch()->candidates().end();
            const bool map_changed =
                !runtime->active_cycle->batch()->GlobalMapContentEquals(
                    message);
            if (!validation_authority_error &&
                execution_replan_without_reference && map_changed) {
              auto disposition =
                  InvalidateActiveGoalForMapLocked(*runtime);
              cancel = cancel || disposition.cancel_planner;
              needs_build = needs_build || disposition.build_now;
              if (disposition.execution_cancel) {
                execution_cancel =
                    std::move(disposition.execution_cancel);
              }
            } else if (!validation_authority_error && map_changed) {
              validation.emplace(ActiveGoalValidation{
                  .generation = runtime->map_generation,
                  .epoch = runtime->epoch,
                  .task_id = runtime->state_machine.task_id(),
                  .plan_id = runtime->active_reference->plan_id,
                  .goal_request_id = active_goal.request_id(),
                  .kind = active_goal.kind(),
                  .map = *runtime->latest_map,
                  .boundary = *runtime->task_boundary,
                  .pose = *pose,
                  .wfd_candidate = *selected,
                  .approach_candidate = std::nullopt});
            }
          }
        } else if (!validation_authority_error) {
          const auto* identity = active_goal.boundary_approach_identity();
          if (!runtime->active_approach_cycle || !identity) {
            validation_authority_error = true;
          } else {
            const auto& approach_candidates =
                runtime->active_approach_cycle->guidance().candidates;
            const auto selected = std::ranges::find_if(
                approach_candidates, [&](const auto& candidate) {
                  return candidate.identity == *identity;
                });
            validation_authority_error =
                selected == approach_candidates.end();
            const bool map_changed =
                !runtime->active_approach_cycle->GlobalMapContentEquals(
                    message);
            if (!validation_authority_error &&
                execution_replan_without_reference && map_changed) {
              auto disposition =
                  InvalidateActiveGoalForMapLocked(*runtime);
              cancel = cancel || disposition.cancel_planner;
              needs_build = needs_build || disposition.build_now;
              if (disposition.execution_cancel) {
                execution_cancel =
                    std::move(disposition.execution_cancel);
              }
            } else if (!validation_authority_error && map_changed) {
              validation.emplace(ActiveGoalValidation{
                  .generation = runtime->map_generation,
                  .epoch = runtime->epoch,
                  .task_id = runtime->state_machine.task_id(),
                  .plan_id = runtime->active_reference->plan_id,
                  .goal_request_id = active_goal.request_id(),
                  .kind = active_goal.kind(),
                  .map = *runtime->latest_map,
                  .boundary = *runtime->task_boundary,
                  .pose = *pose,
                  .wfd_candidate = std::nullopt,
                  .approach_candidate = *selected});
            }
          }
        }
        if (validation_authority_error) {
          cancel = runtime->planner_in_flight;
          FailLocked(*runtime, "FINAL_MAP_VALIDATION_ERROR");
          failed = true;
        }
      }
    }
    if (validation) {
      const std::weak_ptr<RuntimeT> weak_runtime{runtime};
      const auto queue = runtime->map_validation_queue;
      const bool submitted = queue && queue->Submit(
          [weak_runtime, validation = std::move(*validation)]() mutable {
            const auto locked = weak_runtime.lock();
            if (!locked) {
              return;
            }
            bool invalid = false;
            std::string error;
            try {
              if (validation.kind ==
                  lunar::pure_exploration::GoalKind::kTaskFrontier) {
                if (!validation.pose || !validation.wfd_candidate) {
                  throw std::logic_error{
                      "missing WFD map-validation authority"};
                }
                invalid = !CandidateRemainsValidOnLatestMap(
                    *locked, validation.map, validation.boundary,
                    *validation.pose, *validation.wfd_candidate);
              } else {
                if (!validation.approach_candidate) {
                  throw std::logic_error{
                      "missing approach map-validation authority"};
                }
                OccupancyGridView latest_view(
                    validation.map.geometry, validation.map.data,
                    locked->parameters.global_occupied_threshold);
                const TaskRaster latest_raster = TaskRaster::Build(
                    latest_view, validation.boundary,
                    locked->parameters.task_raster_limits);
                try {
                  invalid = !locked->boundary_guidance.IsCandidateStillValid(
                      latest_view, latest_raster,
                      *validation.approach_candidate);
                } catch (const std::length_error& exception) {
                  error = GuidanceResourceReason(exception);
                }
              }
            } catch (const std::length_error& exception) {
              error = ResourceReason(exception);
            } catch (const std::bad_alloc&) {
              error = "RESOURCE_RESULT_COPY";
            } catch (const std::exception&) {
              error = "FINAL_MAP_VALIDATION_ERROR";
            }

            if (locked->parameters.pipeline_seams &&
                locked->parameters.pipeline_seams
                    ->before_active_map_validation_commit) {
              locked->parameters.pipeline_seams
                  ->before_active_map_validation_commit();
            }
            bool cancel = false;
            bool needs_build = false;
            bool stale = false;
            std::optional<std::string> execution_cancel;
            {
              std::scoped_lock lock{locked->mutex};
              if (locked->teardown ||
                  locked->map_generation != validation.generation ||
                  locked->state_machine.task_id() != validation.task_id ||
                  !locked->state_machine.active_goal() ||
                  locked->state_machine.active_goal()->kind() !=
                      validation.kind ||
                  locked->state_machine.active_goal()->request_id() !=
                      validation.goal_request_id) {
                stale = true;
              }
              bool goal_identity_matches = false;
              if (!stale && validation.kind ==
                                lunar::pure_exploration::GoalKind::
                                    kTaskFrontier) {
                goal_identity_matches =
                    validation.wfd_candidate &&
                    locked->state_machine.active_goal()->candidate_key() ==
                        validation.wfd_candidate->key &&
                    validation.wfd_candidate->frontier_canonical_key &&
                    std::ranges::equal(
                        *validation.wfd_candidate->frontier_canonical_key,
                        locked->state_machine.active_goal()
                            ->frontier_canonical_key());
              } else if (!stale) {
                const auto* identity = locked->state_machine.active_goal()
                                           ->boundary_approach_identity();
                goal_identity_matches =
                    identity && validation.approach_candidate &&
                    *identity == validation.approach_candidate->identity;
              }
              if (!stale && !goal_identity_matches) {
                stale = true;
              }
              const bool local_endpoint_refresh_owns_release =
                  !stale &&
                  locked->pending_stationary_action ==
                      RuntimeT::PendingStationaryAction::kBuildFreshBatch &&
                  locked->pending_fresh_batch_release_reason ==
                      lunar::pure_exploration::GoalReleaseReason::
                          kLocalSegmentCompleted;
              if (!stale && !local_endpoint_refresh_owns_release) {
                const auto current_pose =
                    locked->pose_resolver.LatestPoseInMap();
                bool current_frozen_authority = false;
                if (validation.kind ==
                    lunar::pure_exploration::GoalKind::kTaskFrontier) {
                  if (locked->active_cycle && validation.wfd_candidate) {
                    const auto& candidates =
                        locked->active_cycle->batch()->candidates();
                    current_frozen_authority = std::ranges::any_of(
                        candidates, [&](const auto& candidate) {
                          return candidate.key ==
                                     validation.wfd_candidate->key &&
                                 candidate.frontier_canonical_key &&
                                 validation.wfd_candidate
                                     ->frontier_canonical_key &&
                                 std::ranges::equal(
                                     *candidate.frontier_canonical_key,
                                     *validation.wfd_candidate
                                          ->frontier_canonical_key);
                        });
                  }
                } else if (locked->active_approach_cycle &&
                           validation.approach_candidate) {
                  current_frozen_authority = std::ranges::any_of(
                      locked->active_approach_cycle->guidance().candidates,
                      [&](const auto& candidate) {
                        return candidate.identity ==
                               validation.approach_candidate->identity;
                      });
                }
                const bool current_authority_missing =
                    !locked->task_boundary || !current_pose ||
                    !locked->active_reference ||
                    !current_frozen_authority;
                if (current_authority_missing) {
                  cancel = locked->planner_in_flight;
                  FailLocked(*locked, "FINAL_MAP_VALIDATION_ERROR");
                } else {
                  // Candidate safety is a property of this exact frozen goal
                  // and map snapshot, not of a particular rolling-reference
                  // plan ID.  A newer reference may therefore consume the
                  // completed result across a stationary-confirmed replan
                  // epoch only while task, goal request, kind, and full typed
                  // identity all remain unchanged.  The captured plan itself
                  // remains valid only in its original epoch.
                  const bool exact_plan =
                      locked->active_reference->plan_id == validation.plan_id;
                  const bool exact_plan_and_epoch =
                      exact_plan && locked->epoch == validation.epoch;
                  const bool same_goal_successor =
                      !exact_plan &&
                      locked->state_machine.state() ==
                          ExplorationState::kExecuting;
                  if (!exact_plan_and_epoch && !same_goal_successor) {
                    stale = true;
                  } else if (!error.empty()) {
                    cancel = locked->planner_in_flight;
                    FailLocked(*locked, std::move(error));
                  } else if (invalid) {
                    auto disposition =
                        InvalidateActiveGoalForMapLocked(*locked);
                    cancel = disposition.cancel_planner;
                    needs_build = disposition.build_now;
                    execution_cancel =
                        std::move(disposition.execution_cancel);
                  }
                }
              }
            }
            if (!stale) {
              if (execution_cancel) {
                std_msgs::msg::String message;
                message.data = std::move(*execution_cancel);
                locked->cancel_publisher->publish(std::move(message));
              }
              if (cancel) {
                locked->planner_client->CancelActive();
              }
              if (needs_build) {
                QueueBuild(locked);
              } else {
                PumpCurrentCycle(locked);
                PublishStatus(locked);
              }
            }
            if (locked->parameters.pipeline_seams &&
                locked->parameters.pipeline_seams
                    ->after_active_map_validation_commit) {
              locked->parameters.pipeline_seams
                  ->after_active_map_validation_commit();
            }
          });
      if (!submitted) {
        std::scoped_lock lock{runtime->mutex};
        cancel = runtime->planner_in_flight;
        FailLocked(*runtime, "SERIAL_WORK_QUEUE_STOPPED");
        failed = true;
      } else {
        if (runtime->parameters.pipeline_seams &&
            runtime->parameters.pipeline_seams->after_global_map_callback) {
          runtime->parameters.pipeline_seams->after_global_map_callback();
        }
        return;
      }
    }
    {
      std::scoped_lock lock{runtime->mutex};
      if (runtime->teardown ||
          (validation && runtime->map_generation != validation->generation)) {
        return;
      }
      const bool waiting_for_input =
          runtime->state_machine.state() == ExplorationState::kWaitingForInput;
      const bool selecting_without_frozen_cycle =
          runtime->state_machine.state() ==
              ExplorationState::kSelectingFrontier &&
          !HasFrozenCycleLocked(*runtime);
      needs_build =
          needs_build || waiting_for_input || selecting_without_frozen_cycle;
    }
  } catch (const std::length_error& error) {
    std::scoped_lock lock{runtime->mutex};
    cancel = runtime->planner_in_flight;
    FailLocked(*runtime, ResourceReason(error));
    failed = true;
  } catch (const std::exception&) {
    std::scoped_lock lock{runtime->mutex};
    cancel = runtime->planner_in_flight;
    FailLocked(*runtime, "INVALID_GLOBAL_MAP");
    failed = true;
  }
  if (execution_cancel) {
    std_msgs::msg::String message_out;
    message_out.data = *execution_cancel;
    runtime->cancel_publisher->publish(std::move(message_out));
  }
  if (cancel) {
    runtime->planner_client->CancelActive();
  }
  if (needs_build) {
    QueueBuild(runtime);
  } else if (!failed && !cancel) {
    PumpCurrentCycle(runtime);
    PublishStatus(runtime);
  }
  if (failed) {
    PublishStatus(runtime);
  }
  if (runtime->parameters.pipeline_seams &&
      runtime->parameters.pipeline_seams->after_global_map_callback) {
    runtime->parameters.pipeline_seams->after_global_map_callback();
  }
}

template <typename RuntimeT, typename Update>
void HandlePoseInput(const std::weak_ptr<RuntimeT>& weak_runtime,
                     Update&& update) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  bool cancel = false;
  bool needs_build = false;
  try {
    {
      std::scoped_lock lock{runtime->mutex};
      if (runtime->teardown) {
        return;
      }
      update(runtime->pose_resolver);
      ++runtime->pose_generation;
      needs_build = needs_build ||
          runtime->state_machine.state() ==
              ExplorationState::kWaitingForInput ||
          (runtime->state_machine.state() ==
               ExplorationState::kSelectingFrontier &&
           !HasFrozenCycleLocked(*runtime));
    }
    if (needs_build) {
      QueueBuild(runtime);
    } else {
      PumpCurrentCycle(runtime);
      PublishStatus(runtime);
    }
  } catch (const std::exception&) {
    {
      std::scoped_lock lock{runtime->mutex};
      cancel = runtime->planner_in_flight;
      FailLocked(*runtime, "INVALID_POSE_INPUT");
    }
    if (cancel) {
      runtime->planner_client->CancelActive();
    }
    PublishStatus(runtime);
  }
}

template <typename RuntimeT>
void HandleOdometry(const std::weak_ptr<RuntimeT>& weak_runtime,
                    const nav_msgs::msg::Odometry& message) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  const auto& linear = message.twist.twist.linear;
  const auto& angular = message.twist.twist.angular;
  const SpeedObservation speed{
      .linear_speed_mps = std::hypot(linear.x, linear.y, linear.z),
      .angular_speed_radps = std::hypot(angular.x, angular.y, angular.z),
  };
  auto confirmed_action = RuntimeT::PendingStationaryAction::kNone;
  bool cancel_planner = false;
  bool needs_build = false;
  bool stop_wait_diagnostic_due = false;
  try {
    {
      std::scoped_lock lock{runtime->mutex};
      if (runtime->teardown) {
        return;
      }
      runtime->pose_resolver.UpdateOdometry(message);
      runtime->latest_speed = speed;
      ++runtime->pose_generation;
      if (runtime->pending_stationary_action !=
          RuntimeT::PendingStationaryAction::kNone) {
        const auto update = runtime->stationary_gate.Observe(
            SteadyNowLocked(*runtime), speed);
        if (runtime->current_stop_wait) {
          runtime->current_stop_wait->elapsed = update.elapsed;
          runtime->current_stop_wait->confirmation_samples =
              update.consecutive_samples;
        }
        stop_wait_diagnostic_due = update.diagnostic_due;
        if (update.confirmed) {
          if (runtime->current_stop_wait) {
            runtime->current_stop_wait->confirmed_speed = update.observation;
            runtime->current_stop_wait->confirmation_samples =
                update.consecutive_samples;
          }
          ++runtime->planning_epoch;
          runtime->last_stop_wait = std::move(runtime->current_stop_wait);
          runtime->current_stop_wait.reset();
          confirmed_action =
              ConsumeConfirmedStationaryActionLocked(*runtime);
          if (confirmed_action ==
                  RuntimeT::PendingStationaryAction::kBuildFreshBatch &&
              runtime->pending_fresh_batch_release_reason &&
              runtime->state_machine.active_goal()) {
            runtime->state_machine.ReleaseGoal(
                *runtime->pending_fresh_batch_release_reason);
          }
          runtime->pending_fresh_batch_release_reason.reset();
          ClearExecutionPayloadsLocked(*runtime);
          ++runtime->epoch;
          if (confirmed_action ==
              RuntimeT::PendingStationaryAction::kBuildFreshBatch) {
            runtime->pending_map_rebuild = false;
            runtime->pending_stuck_rebuild = false;
            runtime->execution_replan_request_id.reset();
            runtime->execution_replan_bucket.reset();
            runtime->execution_replan_waiting_terminal = false;
            runtime->execution_replan_retry_pending = false;
            ClearSelectionCycleLocked(*runtime);
            ++runtime->build_generation;
          }
        }
      } else {
        needs_build =
            runtime->state_machine.state() ==
                ExplorationState::kWaitingForInput ||
            (runtime->state_machine.state() ==
                 ExplorationState::kSelectingFrontier &&
             !HasFrozenCycleLocked(*runtime));
      }
    }
    if (confirmed_action ==
        RuntimeT::PendingStationaryAction::kBuildFreshBatch) {
      QueueBuild(runtime, true);
    } else if (confirmed_action ==
               RuntimeT::PendingStationaryAction::kReplanActiveGoal) {
      StartExecutionReplan(runtime, true);
    } else if (needs_build) {
      QueueBuild(runtime);
    } else {
      if (stop_wait_diagnostic_due) {
        // Observability only: publish the captured gate update without
        // changing admission, failure memory, or task state.
        PublishStatus(runtime);
      } else {
        PumpCurrentCycle(runtime);
        PublishStatus(runtime);
      }
    }
  } catch (const std::exception&) {
    {
      std::scoped_lock lock{runtime->mutex};
      cancel_planner = runtime->planner_in_flight;
      FailLocked(*runtime, "INVALID_POSE_INPUT");
    }
    if (cancel_planner) {
      runtime->planner_client->CancelActive();
    }
    PublishStatus(runtime);
  }
}

template <typename RuntimeT>
std::optional<std::string> ExecutionCancelLocked(RuntimeT& runtime) {
  if (runtime.pending_execution_cancel_sent || !runtime.active_reference ||
      runtime.active_reference->plan_id.empty()) {
    return std::nullopt;
  }
  runtime.pending_execution_cancel_sent = true;
  return runtime.active_reference->plan_id;
}

template <typename RuntimeT>
void HandleTask(const std::weak_ptr<RuntimeT>& weak_runtime,
                const Task& task) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  std::optional<typename RuntimeT::PendingStart> start;
  if (task.command == Task::START) {
    try {
      start = typename RuntimeT::PendingStart{task.task_id,
                                               TaskBoundary(task)};
    } catch (const std::length_error& error) {
      bool cancel_planner = false;
      {
        std::scoped_lock lock{runtime->mutex};
        cancel_planner = runtime->planner_in_flight;
        FailLocked(*runtime, ResourceReason(error));
      }
      if (cancel_planner) {
        runtime->planner_client->CancelActive();
      }
      PublishStatus(runtime);
      return;
    } catch (const std::exception&) {
      bool cancel_planner = false;
      {
        std::scoped_lock lock{runtime->mutex};
        cancel_planner = runtime->planner_in_flight;
        FailLocked(*runtime, "INVALID_EXPLORATION_TASK");
      }
      if (cancel_planner) {
        runtime->planner_client->CancelActive();
      }
      PublishStatus(runtime);
      return;
    }
  }

  bool cancel_planner = false;
  bool build = false;
  std::optional<std::string> execution_cancel;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    try {
      switch (task.command) {
        case Task::START:
          if (runtime->planner_in_flight) {
            cancel_planner =
                runtime->pending_control == RuntimeT::PendingControl::kNone;
            runtime->pending_control =
                RuntimeT::PendingControl::kReplacementStart;
            runtime->pending_start = std::move(start);
            execution_cancel = ExecutionCancelLocked(*runtime);
          } else {
            execution_cancel = ExecutionCancelLocked(*runtime);
            ApplyStartLocked(*runtime, std::move(*start));
            if (!runtime->parameters.stop_before_planning) {
              runtime->pending_execution_cancel_sent = false;
            }
            build = true;
          }
          break;
        case Task::PAUSE:
          if (runtime->planner_in_flight) {
            if (runtime->pending_control ==
                RuntimeT::PendingControl::kNone) {
              FreezeActiveElapsedLocked(*runtime);
              runtime->pending_control = RuntimeT::PendingControl::kPause;
              execution_cancel = ExecutionCancelLocked(*runtime);
              cancel_planner = true;
            }
          } else {
            execution_cancel = ExecutionCancelLocked(*runtime);
            CancelPendingStationaryActionLocked(*runtime);
            FreezeActiveElapsedLocked(*runtime);
            runtime->state_machine.Pause();
            ClearSelectionCycleLocked(*runtime);
            runtime->execution_replan_request_id.reset();
            runtime->execution_replan_bucket.reset();
            runtime->execution_replan_waiting_terminal = false;
            runtime->execution_replan_retry_pending = false;
            ClearExecutionPayloadsLocked(*runtime);
            ++runtime->epoch;
            ++runtime->build_generation;
            runtime->pending_execution_cancel_sent = false;
          }
          break;
        case Task::RESUME:
          CancelPendingStationaryActionLocked(*runtime);
          runtime->state_machine.Resume();
          runtime->active_elapsed_start = SteadyNowLocked(*runtime);
          ClearSelectionCycleLocked(*runtime);
          runtime->execution_replan_request_id.reset();
          runtime->execution_replan_bucket.reset();
          runtime->execution_replan_waiting_terminal = false;
          runtime->execution_replan_retry_pending = false;
          ++runtime->epoch;
          ++runtime->build_generation;
          build = true;
          break;
        case Task::CANCEL:
          if (runtime->planner_in_flight) {
            cancel_planner =
                runtime->pending_control == RuntimeT::PendingControl::kNone;
            runtime->pending_control = RuntimeT::PendingControl::kCancel;
            runtime->pending_start.reset();
            execution_cancel = ExecutionCancelLocked(*runtime);
          } else {
            execution_cancel = ExecutionCancelLocked(*runtime);
            CancelPendingStationaryActionLocked(*runtime);
            FreezeActiveElapsedLocked(*runtime);
            runtime->state_machine.Cancel();
            runtime->approach_observability.reset();
            ResetStopWaitTaskLifetimeLocked(*runtime);
            runtime->active_elapsed_s = 0.0;
            runtime->task_boundary.reset();
            ClearSelectionCycleLocked(*runtime);
            runtime->execution_replan_request_id.reset();
            runtime->execution_replan_bucket.reset();
            runtime->execution_replan_waiting_terminal = false;
            runtime->execution_replan_retry_pending = false;
            runtime->coverage = {};
            runtime->frontier_count = 0U;
            runtime->candidate_count = 0U;
            runtime->completed_goal_positions.clear();
            ClearExecutionPayloadsLocked(*runtime);
            ++runtime->epoch;
            ++runtime->build_generation;
            runtime->pending_execution_cancel_sent = false;
          }
          break;
        default:
          cancel_planner = runtime->planner_in_flight;
          FailLocked(*runtime, "INVALID_EXPLORATION_COMMAND");
          break;
      }
    } catch (const std::length_error& error) {
      cancel_planner = runtime->planner_in_flight;
      FailLocked(*runtime, ResourceReason(error));
    } catch (const std::exception&) {
      cancel_planner = runtime->planner_in_flight;
      FailLocked(*runtime, "INVALID_CONTROL_TRANSITION");
    }
  }

  if (execution_cancel) {
    std_msgs::msg::String message;
    message.data = *execution_cancel;
    runtime->cancel_publisher->publish(std::move(message));
  }
  if (cancel_planner) {
    runtime->planner_client->CancelActive();
  }
  PublishStatus(runtime);
  if (build) {
    QueueBuild(runtime);
  }
}

}  // namespace

ExplorationNode::ExplorationNode(ExplorationNodeParameters parameters,
                                 const rclcpp::NodeOptions& options)
    : rclcpp::Node("pure_exploration", options) {
  Initialize(std::move(parameters));
}

ExplorationNode::ExplorationNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("pure_exploration", options) {
  Initialize(LoadParameters(*this));
}

void ExplorationNode::Initialize(ExplorationNodeParameters parameters) {
  auto runtime = std::make_shared<Runtime>(*this, std::move(parameters));
  const std::weak_ptr<Runtime> weak_runtime{runtime};
  global_map_subscription_ =
      create_subscription<nav_msgs::msg::OccupancyGrid>(
          runtime->parameters.global_map_topic, rclcpp::QoS{1}.reliable(),
          [weak_runtime](nav_msgs::msg::OccupancyGrid::SharedPtr message) {
            HandleGlobalMap(weak_runtime, *message);
          });
  odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      runtime->parameters.odometry_topic, rclcpp::QoS{1}.reliable(),
      [weak_runtime](nav_msgs::msg::Odometry::SharedPtr message) {
        HandleOdometry(weak_runtime, *message);
      });
  tf_subscription_ = create_subscription<tf2_msgs::msg::TFMessage>(
      runtime->parameters.tf_topic, rclcpp::QoS{10}.reliable(),
      [weak_runtime](tf2_msgs::msg::TFMessage::SharedPtr message) {
        HandlePoseInput(weak_runtime, [message = std::move(message)](
                                          PoseResolver& resolver) {
          resolver.UpdateTransforms(*message);
        });
      });
  task_subscription_ = create_subscription<Task>(
      runtime->parameters.task_topic, rclcpp::QoS{10}.reliable(),
      [weak_runtime](Task::SharedPtr message) {
        HandleTask(weak_runtime, *message);
      });
  planner_diagnostics_subscription_ =
      create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
          runtime->parameters.planner_diagnostics_topic,
          rclcpp::QoS{10}.reliable(),
          [weak_runtime](
              diagnostic_msgs::msg::DiagnosticArray::SharedPtr message) {
            const auto locked = weak_runtime.lock();
            if (!locked) {
              return;
            }
            bool publish = false;
            {
              std::scoped_lock lock{locked->mutex};
              if (!locked->teardown) {
                const TimingIngestResult result = locked->timing.Ingest(*message);
                if (result != TimingIngestResult::kRejected &&
                    message->status.size() == 1U) {
                  const auto request = std::ranges::find_if(
                      message->status.front().values, [](const KeyValue& value) {
                        return value.key == "request_id";
                      });
                  if (request != message->status.front().values.end()) {
                    ConsumePlannerTimingLocked(*locked, request->value);
                    publish = true;
                  }
                }
              }
            }
            if (publish) {
              PublishStatus(locked);
            }
          });
  runtime_ = std::move(runtime);
  execution_timer_ = create_wall_timer(std::chrono::milliseconds{100}, [this] {
    PollExecution();
  });
  PublishStatus(runtime_);
}

ExplorationNode::~ExplorationNode() noexcept {
  execution_timer_.reset();
  auto runtime = std::move(runtime_);
  if (!runtime) {
    return;
  }
  std::shared_ptr<SerializedWorkQueue> work_queue;
  std::shared_ptr<SerializedWorkQueue> map_validation_queue;
  {
    std::scoped_lock lock{runtime->mutex};
    runtime->teardown = true;
    ClearSelectionCycleLocked(*runtime);
    runtime->approach_observability.reset();
    runtime->execution_replan_request_id.reset();
    runtime->execution_replan_bucket.reset();
    runtime->execution_replan_waiting_terminal = false;
    runtime->execution_replan_retry_pending = false;
    work_queue = std::move(runtime->work_queue);
    map_validation_queue = std::move(runtime->map_validation_queue);
  }
  if (work_queue) {
    work_queue->Stop();
  }
  if (map_validation_queue) {
    map_validation_queue->Stop();
  }
  runtime.reset();
}

void ExplorationNode::PollExecution() {
  const auto runtime = runtime_;
  if (!runtime) {
    return;
  }
  std::function<std::chrono::steady_clock::time_point()> now_source;
  bool retry_replan = false;
  bool retry_candidate = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    now_source = runtime->parameters.steady_now;
    retry_replan =
        runtime->execution_replan_retry_pending &&
        !runtime->planner_in_flight &&
        PlanningIsAdmittedLocked(*runtime) &&
        runtime->pending_control == Runtime::PendingControl::kNone &&
        !runtime->execution_replan_waiting_terminal &&
        !runtime->pending_map_rebuild && !runtime->pending_stuck_rebuild &&
        runtime->state_machine.state() == ExplorationState::kReplanning &&
        runtime->state_machine.active_goal() &&
        HasFrozenCycleLocked(*runtime);
    retry_candidate =
        runtime->candidate_retry_deadline &&
        SteadyNowLocked(*runtime) >= *runtime->candidate_retry_deadline &&
        !runtime->planner_in_flight &&
        PlanningIsAdmittedLocked(*runtime) &&
        runtime->pending_control == Runtime::PendingControl::kNone &&
        runtime->state_machine.state() == ExplorationState::kSelectingFrontier &&
        HasFrozenCycleLocked(*runtime) && runtime->current_candidate;
    if (retry_candidate) {
      runtime->candidate_retry_deadline.reset();
    }
  }
  if (retry_replan) {
    StartExecutionReplan(runtime);
    PublishStatus(runtime);
    return;
  }
  if (retry_candidate) {
    PumpCurrentCycle(runtime);
    PublishStatus(runtime);
    return;
  }
  const auto now = now_source ? now_source() : std::chrono::steady_clock::now();
  std::optional<std::string> execution_cancel;
  bool start_replan = false;
  bool cancel_planner = false;
  bool rebuild = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown ||
        runtime->state_machine.state() != ExplorationState::kExecuting ||
        !runtime->state_machine.active_goal() || !runtime->execution_monitor) {
      return;
    }
    const auto pose = runtime->pose_resolver.LatestPoseInMap();
    if (!pose) {
      return;
    }
    const Vec2 position{pose->x, pose->y};
    const auto goal_kind = runtime->state_machine.active_goal()->kind();
    const bool approach_goal =
        goal_kind == lunar::pure_exploration::GoalKind::kBoundaryApproach;
    if (runtime->execution_monitor->ReachedFinalGoal(
            *pose, runtime->state_machine.active_goal()->target())) {
      if (!approach_goal) {
        runtime->completed_goal_positions.push_back(position);
      }
      runtime->state_machine.ReleaseGoal(
          lunar::pure_exploration::GoalReleaseReason::kArrived);
      ClearExecutionPayloadsLocked(*runtime);
      ClearSelectionCycleLocked(*runtime);
      runtime->execution_replan_request_id.reset();
      runtime->execution_replan_bucket.reset();
      runtime->execution_replan_waiting_terminal = false;
      runtime->execution_replan_retry_pending = false;
      ++runtime->epoch;
      ++runtime->build_generation;
      rebuild = true;
    } else {
      const bool at_executable_endpoint = runtime->active_executable_endpoint &&
          runtime->execution_monitor->ReachedSegmentEndpoint(
              position, *runtime->active_executable_endpoint);
      const auto& goal = runtime->state_machine.active_goal()->target();
      const bool endpoint_is_final_goal =
          runtime->active_executable_endpoint &&
          runtime->execution_monitor->ReachedSegmentEndpoint(
              {goal.x, goal.y}, *runtime->active_executable_endpoint);
      const bool rolling = at_executable_endpoint && !endpoint_is_final_goal;
      bool stuck = !rolling && runtime->execution_monitor->UpdateProgress(
          now, position, false);
      if (rolling || stuck) {
        if (!rolling || approach_goal) {
          execution_cancel = ExecutionCancelLocked(*runtime);
        }
        std::optional<std::size_t> committed_candidate;
        if (!rolling && !approach_goal && runtime->active_cycle) {
          const auto& candidates = runtime->active_cycle->batch()->candidates();
          const auto selected = std::ranges::find_if(
              candidates, [&](const auto& candidate) {
                return candidate.key ==
                    runtime->state_machine.active_goal()->candidate_key();
              });
          if (selected != candidates.end()) {
            committed_candidate = static_cast<std::size_t>(
                selected - candidates.begin());
          }
        }
        const auto result = runtime->state_machine.BeginReplanning(
            rolling ? lunar::pure_exploration::ReplanCause::kRollingSegment
                    : lunar::pure_exploration::ReplanCause::kStuckRecovery);
        if (rolling) {
          if (approach_goal) {
            runtime->status_reason_override = "PLANNING_APPROACH";
            if (runtime->parameters.stop_before_planning) {
              const auto stop_cancel = RequestStationaryActionLocked(
                  *runtime,
                  Runtime::PendingStationaryAction::kReplanActiveGoal);
              if (stop_cancel) {
                execution_cancel = std::move(stop_cancel);
              }
              if (runtime->planner_in_flight) {
                runtime->execution_replan_waiting_terminal = true;
                cancel_planner = true;
              }
            } else if (runtime->planner_in_flight) {
              runtime->execution_replan_waiting_terminal = true;
              cancel_planner = true;
            } else {
              start_replan = true;
            }
          } else {
            runtime->pending_fresh_batch_release_reason =
                lunar::pure_exploration::GoalReleaseReason::
                    kLocalSegmentCompleted;
            if (runtime->parameters.stop_before_planning) {
              execution_cancel = RequestStationaryActionLocked(
                  *runtime,
                  Runtime::PendingStationaryAction::kBuildFreshBatch);
            } else {
              execution_cancel = ExecutionCancelLocked(*runtime);
              runtime->state_machine.ReleaseGoal(
                  *runtime->pending_fresh_batch_release_reason);
              runtime->pending_fresh_batch_release_reason.reset();
              ClearExecutionPayloadsLocked(*runtime);
              ClearSelectionCycleLocked(*runtime);
              ++runtime->epoch;
              ++runtime->build_generation;
              rebuild = true;
            }
          }
        } else if (result ==
                   lunar::pure_exploration::ReplanResult::kExhausted) {
          if (approach_goal) {
            ClearExecutionPayloadsLocked(*runtime);
            if (runtime->planner_in_flight) {
              cancel_planner = true;
            }
            ClearSelectionCycleLocked(*runtime);
            runtime->execution_replan_request_id.reset();
            runtime->execution_replan_bucket.reset();
            runtime->execution_replan_waiting_terminal = false;
            runtime->execution_replan_retry_pending = false;
            runtime->pending_map_rebuild = false;
            runtime->pending_stuck_rebuild = false;
            ++runtime->epoch;
            ++runtime->build_generation;
            BlockApproachSnapshotLocked(*runtime, "APPROACH_STALLED");
          } else {
            bool failure_memory_failed = false;
            try {
              if (!runtime->active_cycle || !runtime->active_raster) {
                throw std::logic_error{"missing persistent failure authority"};
              }
              if (!committed_candidate) {
                throw std::logic_error{"missing committed candidate"};
              }
              runtime->failure_memory.RecordPersistentFailure(
                  runtime->active_cycle->batch()->candidates()[
                      *committed_candidate],
                  lunar::pure_exploration::PersistentFailureReason::
                      kExecutionReplansExhausted,
                  *runtime->active_raster);
            } catch (const std::exception&) {
              FailLocked(*runtime, "EXECUTION_REPLAN_FAILURE_MEMORY_ERROR");
              failure_memory_failed = true;
            }
            if (!failure_memory_failed) {
              if (runtime->parameters.stop_before_planning) {
                const auto stop_cancel = RequestStationaryActionLocked(
                    *runtime,
                    Runtime::PendingStationaryAction::kBuildFreshBatch);
                if (stop_cancel) {
                  execution_cancel = std::move(stop_cancel);
                }
                if (runtime->planner_in_flight) {
                  runtime->pending_stuck_rebuild = true;
                  cancel_planner = true;
                }
              } else {
                ClearExecutionPayloadsLocked(*runtime);
                if (runtime->planner_in_flight) {
                  runtime->pending_stuck_rebuild = true;
                  cancel_planner = true;
                } else {
                  ClearSelectionCycleLocked(*runtime);
                  ++runtime->epoch;
                  ++runtime->build_generation;
                  rebuild = true;
                }
              }
            }
          }
        } else {
          if (approach_goal) {
            runtime->status_reason_override = "PLANNING_APPROACH";
          }
          if (runtime->parameters.stop_before_planning) {
            const auto stop_cancel = RequestStationaryActionLocked(
                *runtime,
                Runtime::PendingStationaryAction::kReplanActiveGoal);
            if (stop_cancel) {
              execution_cancel = std::move(stop_cancel);
            }
            if (runtime->planner_in_flight) {
              runtime->execution_replan_waiting_terminal = true;
              cancel_planner = true;
            }
          } else {
            if (runtime->planner_in_flight) {
              runtime->execution_replan_waiting_terminal = true;
              cancel_planner = true;
            } else {
              start_replan = true;
            }
          }
        }
      }
    }
  }
  if (execution_cancel) {
    std_msgs::msg::String message;
    message.data = *execution_cancel;
    runtime->cancel_publisher->publish(std::move(message));
  }
  if (cancel_planner) {
    runtime->planner_client->CancelActive();
  }
  if (start_replan) {
    StartExecutionReplan(runtime);
  }
  PublishStatus(runtime);
  if (rebuild) {
    QueueBuild(runtime);
  }
}

FrozenPlanningCyclePtr ExplorationNode::SnapshotActiveCycleForTest() const {
  const auto runtime = runtime_;
  if (!runtime) {
    return nullptr;
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->active_cycle;
}

std::optional<lunar_planning_msgs::msg::MotionReference>
ExplorationNode::SnapshotActiveReferenceForTest() const {
  const auto runtime = runtime_;
  if (!runtime) {
    return std::nullopt;
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->active_reference;
}

std::optional<std::string> ExplorationNode::SnapshotActiveRequestIdForTest()
    const {
  const auto runtime = runtime_;
  if (!runtime) {
    return std::nullopt;
  }
  std::scoped_lock lock{runtime->mutex};
  if (!runtime->state_machine.active_goal()) {
    return std::nullopt;
  }
  return runtime->state_machine.active_goal()->request_id();
}

std::optional<lunar::pure_exploration::GoalKind>
ExplorationNode::SnapshotActiveGoalKindForTest() const {
  const auto runtime = runtime_;
  if (!runtime) {
    return std::nullopt;
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->state_machine.active_goal()
             ? std::optional<lunar::pure_exploration::GoalKind>{
                   runtime->state_machine.active_goal()->kind()}
             : std::nullopt;
}

std::optional<lunar::pure_exploration::BoundaryApproachGoalIdentity>
ExplorationNode::SnapshotBoundaryApproachIdentityForTest() const {
  const auto runtime = runtime_;
  if (!runtime) {
    return std::nullopt;
  }
  std::scoped_lock lock{runtime->mutex};
  if (!runtime->state_machine.active_goal()) {
    return std::nullopt;
  }
  const auto* identity = runtime->state_machine.active_goal()
                             ->boundary_approach_identity();
  return identity ? std::optional{*identity} : std::nullopt;
}

std::optional<Pose2> ExplorationNode::SnapshotActiveTargetForTest() const {
  const auto runtime = runtime_;
  if (!runtime) return std::nullopt;
  std::scoped_lock lock{runtime->mutex};
  return runtime->state_machine.active_goal()
             ? std::optional<Pose2>{runtime->state_machine.active_goal()->target()}
             : std::nullopt;
}

std::vector<Vec2> ExplorationNode::SnapshotExecutablePolylineForTest() const {
  const auto runtime = runtime_;
  if (!runtime) {
    return {};
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->active_executable_polyline;
}

std::optional<Vec2> ExplorationNode::SnapshotExecutableEndpointForTest()
    const {
  const auto runtime = runtime_;
  if (!runtime) {
    return std::nullopt;
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->active_executable_endpoint;
}

void ExplorationNode::ResetActiveCycleForTest() {
  const auto runtime = runtime_;
  if (!runtime) {
    return;
  }
  std::scoped_lock lock{runtime->mutex};
  runtime->active_cycle.reset();
  runtime->active_approach_cycle.reset();
}

void ExplorationNode::ResetMapValidationAuthorityForTest(
    const MapValidationAuthorityForTest authority) {
  const auto runtime = runtime_;
  if (!runtime) {
    return;
  }
  std::scoped_lock lock{runtime->mutex};
  switch (authority) {
    case MapValidationAuthorityForTest::kTaskBoundary:
      runtime->task_boundary.reset();
      return;
    case MapValidationAuthorityForTest::kActiveReference:
      runtime->active_reference.reset();
      return;
    case MapValidationAuthorityForTest::kPose:
      runtime->pose_resolver = PoseResolver{};
      return;
  }
  throw std::invalid_argument{"unknown map-validation authority"};
}

double ExplorationNode::PlatformWidthForTest() const {
  const auto runtime = runtime_;
  if (!runtime) {
    throw std::logic_error{"exploration runtime is unavailable"};
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->candidate_generator.platform_width_m();
}

std::size_t ExplorationNode::CoarseCursorForTest() const {
  const auto runtime = runtime_;
  if (!runtime) {
    return 0U;
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->active_cycle ? runtime->active_cycle->coarse_cursor() : 0U;
}

std::optional<std::size_t>
ExplorationNode::CandidateIndexForRequestForTest(
    const std::string& request_id) const {
  const auto runtime = runtime_;
  if (!runtime) {
    return std::nullopt;
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->active_cycle
             ? runtime->active_cycle->batch()->CandidateIndexForRequest(
                   request_id)
             : std::nullopt;
}

std::optional<Pose2> ExplorationNode::LatestPoseForTest() const {
  const auto runtime = runtime_;
  if (!runtime) {
    return std::nullopt;
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->pose_resolver.LatestPoseInMap();
}

std::optional<double> ExplorationNode::LatestMapResolutionForTest() const {
  const auto runtime = runtime_;
  if (!runtime) {
    return std::nullopt;
  }
  std::scoped_lock lock{runtime->mutex};
  return runtime->latest_map
             ? std::optional<double>{runtime->latest_map->geometry.resolution}
             : std::nullopt;
}

void ExplorationNode::InjectEvaluationForTest(PlannerEvaluation evaluation) {
  const auto runtime = runtime_;
  if (!runtime) {
    return;
  }
  FrozenPlanningCyclePtr cycle;
  std::uint64_t epoch = 0U;
  {
    std::scoped_lock lock{runtime->mutex};
    cycle = runtime->active_cycle;
    epoch = runtime->epoch;
  }
  if (cycle) {
    HandleEvaluation(std::weak_ptr<Runtime>{runtime}, cycle, epoch,
                     std::move(evaluation));
  }
}

}  // namespace lunar::pure_exploration_ros
