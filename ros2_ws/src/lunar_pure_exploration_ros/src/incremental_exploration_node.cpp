#include "lunar_pure_exploration_ros/incremental_exploration_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <lunar_planning_msgs/action/navigate_to_pose.hpp>
#include <lunar_pure_exploration_core/coverage.hpp>
#include <lunar_pure_exploration_core/frontier_detector.hpp>
#include <lunar_pure_exploration_core/occupancy_grid.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>
#include <rclcpp/qos.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_pure_exploration_ros/marker_builder.hpp"
#include "lunar_pure_exploration_ros/navigation_client.hpp"
#include "lunar_pure_exploration_ros/platform_config_loader.hpp"
#include "lunar_pure_exploration_ros/pose_resolver.hpp"
#include "lunar_pure_exploration_ros/task_map_marker_builder.hpp"

namespace lunar::pure_exploration_ros {
namespace {

using Action = lunar_planning_msgs::action::NavigateToPose;
using DiagnosticArray = diagnostic_msgs::msg::DiagnosticArray;
using DiagnosticStatus = diagnostic_msgs::msg::DiagnosticStatus;
using KeyValue = diagnostic_msgs::msg::KeyValue;
using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;
using Task = lunar_pure_exploration_msgs::msg::PureExplorationTask;
using lunar::pure_exploration::BoundaryGuidanceResult;
using lunar::pure_exploration::CalculateCoverage;
using lunar::pure_exploration::CandidateGain;
using lunar::pure_exploration::CandidateView;
using lunar::pure_exploration::CellState;
using lunar::pure_exploration::CoverageStats;
using lunar::pure_exploration::FrontierCluster;
using lunar::pure_exploration::FrontierDetection;
using lunar::pure_exploration::FrontierDetector;
using lunar::pure_exploration::FrontierParameters;
using lunar::pure_exploration::GridGeometry;
using lunar::pure_exploration::GridIndex;
using lunar::pure_exploration::NavigationPhase;
using lunar::pure_exploration::OccupancyGridView;
using lunar::pure_exploration::PersistentFailureReason;
using lunar::pure_exploration::Polygon2;
using lunar::pure_exploration::Pose2;
using lunar::pure_exploration::TaskRaster;
using lunar::pure_exploration::Vec2;

std::string AbsoluteTopic(rclcpp::Node& node, const std::string& name,
                          const std::string& default_value) {
  auto result = node.declare_parameter<std::string>(name, default_value);
  if (result.empty() || result.front() != '/') {
    throw std::invalid_argument{name + " must be absolute"};
  }
  return result;
}

std::size_t PositiveSize(rclcpp::Node& node, const std::string& name,
                         const std::int64_t default_value) {
  const auto value = node.declare_parameter<std::int64_t>(name, default_value);
  if (value <= 0 || static_cast<std::uint64_t>(value) >
                        std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument{name + " must be positive"};
  }
  return static_cast<std::size_t>(value);
}

IncrementalExplorationNodeParameters LoadParameters(rclcpp::Node& node) {
  const auto platform_selector =
      node.declare_parameter<std::string>("platform_selector", "wheel");
  const auto platform_config =
      node.declare_parameter<std::string>("platform_config", "");
  if (platform_config.empty()) {
    throw std::invalid_argument{"platform_config is required"};
  }
  auto platform =
      LoadIncrementalPlatformConfig(std::filesystem::path{platform_config},
                                    platform_selector)
          .geometry;

  const auto threshold =
      node.declare_parameter<std::int64_t>("occupied_threshold", 50);
  const auto minimum_frontier =
      node.declare_parameter<double>("minimum_frontier_length_m", 1.0);
  const auto coverage_target =
      node.declare_parameter<double>("coverage_target", 0.80);
  const auto sensor_range =
      node.declare_parameter<double>("sensor_range_m", 10.0);
  const auto sensor_fov_deg =
      node.declare_parameter<double>("sensor_fov_deg", 90.0);
  const auto goal_yaw_tolerance_deg =
      node.declare_parameter<double>("goal_yaw_tolerance_deg", 11.25);
  const auto yaw_offsets_deg = node.declare_parameter<std::vector<double>>(
      "yaw_offsets_deg", {-45.0, -22.5, 0.0, 22.5, 45.0});
  if (threshold < 0 || threshold > 100 || yaw_offsets_deg.size() != 5U ||
      !std::isfinite(minimum_frontier) || minimum_frontier <= 0.0 ||
      !std::isfinite(coverage_target) || coverage_target <= 0.0 ||
      coverage_target > 1.0 || !std::isfinite(sensor_range) ||
      sensor_range <= 0.0 || !std::isfinite(sensor_fov_deg) ||
      sensor_fov_deg <= 0.0 || !std::isfinite(goal_yaw_tolerance_deg) ||
      goal_yaw_tolerance_deg < 0.0) {
    throw std::invalid_argument{"invalid incremental exploration parameters"};
  }
  lunar::pure_exploration::CandidateParameters candidate_parameters;
  for (std::size_t index = 0U; index < yaw_offsets_deg.size(); ++index) {
    if (!std::isfinite(yaw_offsets_deg[index])) {
      throw std::invalid_argument{"yaw offsets must be finite"};
    }
    candidate_parameters.yaw_offsets_rad[index] =
        yaw_offsets_deg[index] * std::numbers::pi / 180.0;
  }

  const auto maximum_task_cells =
      PositiveSize(node, "maximum_task_raster_cells", 1048576);
  const auto maximum_candidates =
      PositiveSize(node, "maximum_candidate_views", 10000);
  return {
      .platform = std::move(platform),
      .candidate_parameters = candidate_parameters,
      .candidate_limits =
          {PositiveSize(node, "maximum_position_probes", 100000),
           maximum_candidates,
           PositiveSize(node, "maximum_collision_work_units", 1000000)},
      .task_raster_limits = {maximum_task_cells},
      .boundary_guidance_limits =
          {PositiveSize(node, "maximum_guidance_grid_cells",
                        static_cast<std::int64_t>(maximum_task_cells)),
           PositiveSize(node, "maximum_guidance_work_units", 8388608),
           PositiveSize(node, "maximum_approach_candidates",
                        static_cast<std::int64_t>(maximum_candidates))},
      .sensor_model =
          {sensor_range, sensor_fov_deg * std::numbers::pi / 180.0},
      .information_gain_limits =
          {PositiveSize(node, "maximum_visibility_work_units", 1000000)},
      .score_weights =
          {node.declare_parameter<double>("score_weights.information_gain",
                                          0.60),
           node.declare_parameter<double>("score_weights.global_path_length",
                                          0.30),
           node.declare_parameter<double>("score_weights.heading_change",
                                          0.05),
           node.declare_parameter<double>("score_weights.revisit", 0.05)},
      .failure_memory_limits =
          {PositiveSize(node, "maximum_failure_entries", 1000),
           PositiveSize(node, "maximum_failure_patch_cells_per_entry", 10000),
           PositiveSize(node, "maximum_failure_total_patch_cells", 1000000)},
      .occupied_threshold = static_cast<std::int8_t>(threshold),
      .minimum_frontier_length_m = minimum_frontier,
      .coverage_target = coverage_target,
      .goal_yaw_tolerance_rad =
          goal_yaw_tolerance_deg * std::numbers::pi / 180.0,
      .exploration_map_topic = AbsoluteTopic(
          node, "exploration_map_topic", "/Car/T4/mapping/exploration_map"),
      .odometry_topic = AbsoluteTopic(
          node, "odometry_topic", "/Car/T3/localization/odometry"),
      .tf_topic = AbsoluteTopic(node, "tf_topic", "/tf"),
      .task_topic = AbsoluteTopic(
          node, "task_topic", "/Car/T4/exploration/task"),
      .navigation_action = AbsoluteTopic(
          node, "navigation_action", "/Car/T4/navigation/navigate_to_pose"),
      .status_topic = AbsoluteTopic(
          node, "status_topic", "/Car/T4/exploration/status"),
      .task_boundary_topic = AbsoluteTopic(
          node, "task_boundary_topic", "/Car/T4/exploration/task_boundary"),
      .task_map_markers_topic = AbsoluteTopic(
          node, "task_map_markers_topic",
          "/Car/T4/exploration/task_map_markers"),
      .current_goal_topic = AbsoluteTopic(
          node, "current_goal_topic", "/Car/T4/exploration/current_goal"),
      .frontiers_topic = AbsoluteTopic(
          node, "frontiers_topic", "/Car/T4/exploration/frontiers"),
      .diagnostics_topic = AbsoluteTopic(
          node, "diagnostics_topic", "/Car/T4/exploration/diagnostics"),
  };
}

double QuaternionYaw(const geometry_msgs::msg::Quaternion& quaternion) {
  if (!std::isfinite(quaternion.x) || !std::isfinite(quaternion.y) ||
      !std::isfinite(quaternion.z) || !std::isfinite(quaternion.w)) {
    throw std::invalid_argument{"exploration map quaternion must be finite"};
  }
  const double norm = std::hypot(
      std::hypot(quaternion.x, quaternion.y),
      std::hypot(quaternion.z, quaternion.w));
  if (!std::isfinite(norm) || norm == 0.0) {
    throw std::invalid_argument{"exploration map quaternion must be nonzero"};
  }
  const double x = quaternion.x / norm;
  const double y = quaternion.y / norm;
  const double z = quaternion.z / norm;
  const double w = quaternion.w / norm;
  return std::atan2(2.0 * (w * z + x * y),
                    1.0 - 2.0 * (y * y + z * z));
}

GridGeometry MapGeometry(const nav_msgs::msg::OccupancyGrid& map) {
  if (map.header.frame_id != "map") {
    throw std::invalid_argument{"exploration map frame must be map"};
  }
  return {.width = map.info.width,
          .height = map.info.height,
          .resolution = static_cast<double>(map.info.resolution),
          .origin_x = map.info.origin.position.x,
          .origin_y = map.info.origin.position.y,
          .origin_yaw = QuaternionYaw(map.info.origin.orientation)};
}

GridGeometry FixedTaskGeometry(const GridGeometry& source,
                               const Polygon2& boundary) {
  long double minimum_x = 0.0L;
  long double minimum_y = 0.0L;
  long double maximum_x = static_cast<long double>(source.width);
  long double maximum_y = static_cast<long double>(source.height);
  for (const Vec2 vertex : boundary.vertices) {
    const auto grid = OccupancyGridView::WorldToGrid(source, vertex);
    if (!grid) {
      throw std::overflow_error{"task boundary grid transform overflow"};
    }
    minimum_x = std::min(minimum_x,
                         std::floor(static_cast<long double>(grid->x)) - 1.0L);
    minimum_y = std::min(minimum_y,
                         std::floor(static_cast<long double>(grid->y)) - 1.0L);
    maximum_x = std::max(maximum_x,
                         std::ceil(static_cast<long double>(grid->x)) + 1.0L);
    maximum_y = std::max(maximum_y,
                         std::ceil(static_cast<long double>(grid->y)) + 1.0L);
  }
  const long double width = maximum_x - minimum_x;
  const long double height = maximum_y - minimum_y;
  constexpr auto kMaximumExtent =
      static_cast<long double>(std::numeric_limits<std::uint32_t>::max());
  if (!std::isfinite(width) || !std::isfinite(height) || width <= 0.0L ||
      height <= 0.0L || width > kMaximumExtent || height > kMaximumExtent) {
    throw std::overflow_error{"fixed task grid extent overflow"};
  }
  const auto origin = OccupancyGridView::GridToWorld(
      source, Vec2{static_cast<double>(minimum_x),
                   static_cast<double>(minimum_y)});
  if (!origin) {
    throw std::overflow_error{"fixed task grid origin overflow"};
  }
  return {.width = static_cast<std::uint32_t>(width),
          .height = static_cast<std::uint32_t>(height),
          .resolution = source.resolution,
          .origin_x = origin->x,
          .origin_y = origin->y,
          .origin_yaw = source.origin_yaw};
}

std::vector<std::int8_t> ProjectToFixedGrid(
    const OccupancyGridView& source, const GridGeometry& fixed) {
  const auto width = static_cast<std::size_t>(fixed.width);
  const auto height = static_cast<std::size_t>(fixed.height);
  if (height != 0U && width > std::numeric_limits<std::size_t>::max() / height) {
    throw std::overflow_error{"fixed task grid cell count overflow"};
  }
  std::vector<std::int8_t> data(width * height, -1);
  for (std::uint32_t y = 0U; y < fixed.height; ++y) {
    for (std::uint32_t x = 0U; x < fixed.width; ++x) {
      const auto world = OccupancyGridView::GridToWorld(
          fixed, Vec2{static_cast<double>(x) + 0.5,
                      static_cast<double>(y) + 0.5});
      if (!world) {
        throw std::overflow_error{"fixed task cell center overflow"};
      }
      const auto source_cell = source.WorldToCell(*world);
      if (!source_cell) {
        continue;
      }
      const auto value = source.RawValue(*source_cell);
      if (value) {
        data[static_cast<std::size_t>(y) * width + x] = *value;
      }
    }
  }
  return data;
}

Polygon2 TaskBoundary(const Task& task) {
  if (task.task_id.empty() || task.header.frame_id != "map" ||
      task.boundary.points.size() < 3U) {
    throw std::invalid_argument{"invalid incremental exploration START"};
  }
  Polygon2 result;
  result.vertices.reserve(task.boundary.points.size());
  for (const auto& point : task.boundary.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
      throw std::invalid_argument{"task boundary must be finite"};
    }
    result.vertices.push_back({point.x, point.y});
  }
  return result;
}

std::uint32_t ToU32(const std::size_t value) {
  return static_cast<std::uint32_t>(
      std::min(value, static_cast<std::size_t>(
                          std::numeric_limits<std::uint32_t>::max())));
}

std::string ApproachWaitReason(
    const lunar::pure_exploration::ApproachWaitReason reason) {
  using Reason = lunar::pure_exploration::ApproachWaitReason;
  switch (reason) {
    case Reason::kNone:
      return "APPROACH_READY";
    case Reason::kWaitingForSafeStart:
      return "WAITING_FOR_SAFE_START";
    case Reason::kWaitingForTaskMapCoverage:
      return "WAITING_FOR_TASK_MAP_COVERAGE";
    case Reason::kNoGuidanceRoute:
      return "NO_GUIDANCE_ROUTE";
    case Reason::kNoSafeCandidate:
      return "NO_SAFE_APPROACH_CANDIDATE";
  }
  return "APPROACH_WAITING";
}

visualization_msgs::msg::MarkerArray BoundaryMarkers(
    const Polygon2& boundary) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "map";
  marker.ns = "task_boundary";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.08;
  marker.color.r = 1.0F;
  marker.color.g = 1.0F;
  marker.color.a = 1.0F;
  marker.points.reserve(boundary.vertices.size() + 1U);
  for (const Vec2 vertex : boundary.vertices) {
    geometry_msgs::msg::Point point;
    point.x = vertex.x;
    point.y = vertex.y;
    marker.points.push_back(point);
  }
  if (!marker.points.empty()) {
    marker.points.push_back(marker.points.front());
  }
  visualization_msgs::msg::MarkerArray result;
  result.markers.push_back(std::move(marker));
  return result;
}

}  // namespace

struct IncrementalExplorationNode::Runtime final {
  struct ActiveGoal final {
    CandidateView candidate;
    std::uint64_t map_sequence;
    std::uint64_t task_generation;
  };

  Runtime(rclcpp::Node& node, IncrementalExplorationNodeParameters input)
      : parameters(std::move(input)),
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
        clock(node.get_clock()),
        status_publisher(node.create_publisher<Status>(
            parameters.status_topic,
            rclcpp::QoS{10}.reliable().transient_local())),
        task_boundary_publisher(
            node.create_publisher<visualization_msgs::msg::MarkerArray>(
                parameters.task_boundary_topic,
                rclcpp::QoS{1}.reliable().transient_local())),
        task_map_publisher(
            node.create_publisher<visualization_msgs::msg::MarkerArray>(
                parameters.task_map_markers_topic,
                rclcpp::QoS{1}.reliable().transient_local())),
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
            rclcpp::QoS{1}.reliable().transient_local())) {}

  std::mutex mutex;
  bool teardown{false};
  IncrementalExplorationNodeParameters parameters;
  lunar::pure_exploration::CandidateGenerator candidate_generator;
  lunar::pure_exploration::BoundaryGuidance boundary_guidance;
  lunar::pure_exploration::InformationGainEvaluator information_gain;
  lunar::pure_exploration::CandidateRanker ranker;
  lunar::pure_exploration::FailureMemory failure_memory;
  PoseResolver pose_resolver;
  std::unique_ptr<NavigationClient> navigation_client;
  std::optional<nav_msgs::msg::OccupancyGrid> latest_map;
  std::optional<Polygon2> task_boundary;
  std::optional<GridGeometry> task_grid_geometry;
  std::optional<TaskRaster> raster;
  CoverageStats coverage{};
  std::vector<FrontierCluster> frontiers;
  std::vector<CandidateView> candidates;
  std::vector<std::size_t> decision_order;
  std::optional<BoundaryGuidanceResult> approach;
  std::optional<ActiveGoal> active_goal;
  std::optional<std::uint64_t> minimum_decision_map_sequence;
  std::string task_id;
  std::string reason_code{"IDLE"};
  std::uint8_t state{Status::IDLE};
  std::uint64_t exploration_map_sequence{0U};
  std::uint64_t task_generation{0U};
  std::uint32_t completed_goal_count{0U};
  std::uint32_t reachable_candidate_count{0U};
  bool cancel_expected{false};
  MarkerBuilder marker_builder;
  TaskMapMarkerBuilder task_map_marker_builder;
  rclcpp::Clock::SharedPtr clock;
  rclcpp::Publisher<Status>::SharedPtr status_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      task_boundary_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      task_map_publisher;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      current_goal_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      frontiers_publisher;
  rclcpp::Publisher<DiagnosticArray>::SharedPtr diagnostics_publisher;
};

namespace {

using Runtime = IncrementalExplorationNode::Runtime;

ApproachMarkers BuildApproachMarkers(const BoundaryGuidanceResult& result) {
  ApproachMarkers markers;
  markers.intent_points.reserve(result.intents.size());
  for (const auto& intent : result.intents) {
    markers.intent_points.push_back(
        Vec2{static_cast<double>(intent.cell.x),
             static_cast<double>(intent.cell.y)});
  }
  markers.candidates.reserve(result.candidates.size());
  for (const auto& candidate : result.candidates) {
    markers.candidates.push_back(
        {.identity = candidate.identity, .pose = candidate.pose});
  }
  return markers;
}

void PublishLocked(Runtime& runtime) {
  Status status;
  status.header.stamp = runtime.clock->now();
  status.header.frame_id = "map";
  status.task_id = runtime.task_id;
  status.state = runtime.state;
  status.reason_code = runtime.reason_code;
  status.polygon_area_m2 = runtime.coverage.polygon_area_m2;
  status.task_raster_area_m2 = runtime.coverage.task_raster_area_m2;
  status.known_free_area_m2 = runtime.coverage.known_free_area_m2;
  status.known_occupied_area_m2 = runtime.coverage.known_occupied_area_m2;
  status.unknown_area_m2 = runtime.coverage.unknown_area_m2;
  status.outside_map_area_m2 = runtime.coverage.outside_map_area_m2;
  status.coverage_ratio = runtime.coverage.coverage_ratio;
  status.frontier_cluster_count = ToU32(runtime.frontiers.size());
  status.candidate_count = ToU32(runtime.candidates.size());
  status.reachable_candidate_count = runtime.reachable_candidate_count;
  status.failed_candidate_count = ToU32(runtime.failure_memory.size());
  status.completed_goal_count = runtime.completed_goal_count;
  if (runtime.active_goal) {
    status.current_goal.position.x = runtime.active_goal->candidate.pose.x;
    status.current_goal.position.y = runtime.active_goal->candidate.pose.y;
    status.current_goal.orientation.z =
        std::sin(runtime.active_goal->candidate.pose.yaw / 2.0);
    status.current_goal.orientation.w =
        std::cos(runtime.active_goal->candidate.pose.yaw / 2.0);
  }
  runtime.status_publisher->publish(status);

  if (runtime.task_boundary) {
    runtime.task_boundary_publisher->publish(
        BoundaryMarkers(*runtime.task_boundary));
  }
  if (runtime.raster) {
    runtime.task_map_publisher->publish(
        runtime.task_map_marker_builder.Build(*runtime.raster, "map"));
  }
  if (runtime.approach) {
    auto approach_markers = BuildApproachMarkers(*runtime.approach);
    runtime.frontiers_publisher->publish(runtime.marker_builder.Build(
        {}, {}, std::nullopt, &approach_markers));
  } else {
    std::optional<MarkerSelection> selected;
    if (runtime.active_goal &&
        runtime.active_goal->candidate.frontier_canonical_key) {
      selected = MarkerSelection{
          .candidate_key = runtime.active_goal->candidate.key,
          .frontier_canonical_key =
              *runtime.active_goal->candidate.frontier_canonical_key,
          .target = runtime.active_goal->candidate.pose};
    }
    runtime.frontiers_publisher->publish(runtime.marker_builder.Build(
        runtime.frontiers, runtime.candidates, selected));
  }

  DiagnosticArray diagnostics;
  diagnostics.header = status.header;
  DiagnosticStatus diagnostic;
  diagnostic.level = runtime.state == Status::ERROR
                         ? DiagnosticStatus::ERROR
                         : DiagnosticStatus::OK;
  diagnostic.name = "incremental_exploration";
  diagnostic.message = runtime.reason_code;
  KeyValue map_sequence;
  map_sequence.key = "exploration_map_sequence";
  map_sequence.value = std::to_string(runtime.exploration_map_sequence);
  diagnostic.values.push_back(std::move(map_sequence));
  KeyValue candidate_count;
  candidate_count.key = "candidate_count";
  candidate_count.value = std::to_string(runtime.candidates.size());
  diagnostic.values.push_back(std::move(candidate_count));
  KeyValue active_navigation;
  active_navigation.key = "active_navigation";
  active_navigation.value = runtime.active_goal ? "true" : "false";
  diagnostic.values.push_back(std::move(active_navigation));
  diagnostics.status.push_back(std::move(diagnostic));
  runtime.diagnostics_publisher->publish(std::move(diagnostics));
}

void RefreshDecisionLocked(Runtime& runtime) {
  runtime.frontiers.clear();
  runtime.candidates.clear();
  runtime.decision_order.clear();
  runtime.approach.reset();
  runtime.reachable_candidate_count = 0U;
  if (!runtime.latest_map || !runtime.task_boundary) {
    runtime.raster.reset();
    runtime.coverage = {};
    return;
  }

  const auto geometry = MapGeometry(*runtime.latest_map);
  const OccupancyGridView source_map(
      geometry, runtime.latest_map->data,
      runtime.parameters.occupied_threshold);
  if (!runtime.task_grid_geometry) {
    runtime.task_grid_geometry =
        FixedTaskGeometry(geometry, *runtime.task_boundary);
  }
  const auto projected_data =
      ProjectToFixedGrid(source_map, *runtime.task_grid_geometry);
  const OccupancyGridView map(*runtime.task_grid_geometry, projected_data,
                              runtime.parameters.occupied_threshold);
  runtime.raster = TaskRaster::Build(
      map, *runtime.task_boundary, runtime.parameters.task_raster_limits);
  runtime.coverage = CalculateCoverage(*runtime.raster);
  const auto robot_pose = runtime.pose_resolver.LatestPoseInMap();
  if (!robot_pose) {
    return;
  }

  auto guidance = runtime.boundary_guidance.Build(
      map, *runtime.raster, *robot_pose);
  if (guidance.phase == NavigationPhase::kApproachTask) {
    runtime.decision_order =
        runtime.boundary_guidance.CoarseOrder(guidance, *robot_pose);
    runtime.candidates.reserve(guidance.candidates.size());
    for (std::size_t index = 0U; index < guidance.candidates.size(); ++index) {
      const auto& candidate = guidance.candidates[index];
      runtime.candidates.push_back(
          {.id = candidate.id,
           .frontier_id = 0U,
           .frontier_index = index,
           .key = candidate.identity.candidate_key,
           .pose = candidate.pose,
           .frontier_distance_m = candidate.remaining_cost.path_length_m});
    }
    runtime.reason_code = ApproachWaitReason(guidance.wait_reason);
    runtime.approach = std::move(guidance);
  } else {
    const auto robot_cell = runtime.raster->WorldToCell(
        Vec2{robot_pose->x, robot_pose->y});
    if (!robot_cell ||
        runtime.raster->Classify(*robot_cell) != CellState::kFree) {
      runtime.reason_code = "WAITING_FOR_REACHABLE_KNOWN_START";
      return;
    }
    const FrontierDetection detection = FrontierDetector(
        FrontierParameters{runtime.parameters.minimum_frontier_length_m})
                                             .Detect(*runtime.raster,
                                                     *robot_cell);
    runtime.frontiers = detection.clusters;
    if (!detection.has_reachable_free_start) {
      runtime.reason_code = "WAITING_FOR_REACHABLE_KNOWN_START";
      return;
    }
    runtime.candidates = runtime.candidate_generator.Generate(
        *runtime.raster, runtime.frontiers);
    std::vector<CandidateGain> gains;
    gains.reserve(runtime.candidates.size());
    for (std::size_t index = 0U; index < runtime.candidates.size(); ++index) {
      gains.push_back(
          {index, runtime.information_gain
                      .Evaluate(*runtime.raster, runtime.candidates[index])
                      .visible_unknown_area_m2});
    }
    const auto ranked = runtime.ranker.CoarseRank(
        runtime.candidates, runtime.frontiers, gains, *robot_pose);
    runtime.decision_order.reserve(ranked.size());
    for (const auto& row : ranked) {
      runtime.decision_order.push_back(row.candidate_index);
    }
  }

  for (const std::size_t index : runtime.decision_order) {
    if (!runtime.failure_memory.IsSuppressed(runtime.candidates.at(index),
                                             *runtime.raster)) {
      ++runtime.reachable_candidate_count;
    }
  }
}

void Decide(const std::shared_ptr<Runtime>& runtime) {
  std::optional<NavigationTarget> target;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown || runtime->task_id.empty() ||
        runtime->state == Status::PAUSED || runtime->state == Status::IDLE ||
        runtime->state == Status::ERROR || runtime->active_goal) {
      return;
    }
    if (!runtime->latest_map || !runtime->task_boundary ||
        !runtime->pose_resolver.LatestPoseInMap() ||
        (runtime->minimum_decision_map_sequence &&
         runtime->exploration_map_sequence <
             *runtime->minimum_decision_map_sequence)) {
      runtime->state = Status::WAITING_FOR_INPUT;
      runtime->reason_code = "WAITING_FOR_INPUT";
      PublishLocked(*runtime);
      return;
    }

    try {
      runtime->state = Status::SELECTING_FRONTIER;
      runtime->reason_code = "SELECTING_FRONTIER";
      RefreshDecisionLocked(*runtime);
      if (runtime->coverage.coverage_ratio >=
          runtime->parameters.coverage_target) {
        runtime->state = Status::COMPLETED;
        runtime->reason_code = "COVERAGE_TARGET_REACHED";
        PublishLocked(*runtime);
        return;
      }
      if (!runtime->approach && runtime->frontiers.empty()) {
        runtime->state = Status::COMPLETED;
        runtime->reason_code = "COMPLETED_NO_REACHABLE_FRONTIER";
        PublishLocked(*runtime);
        return;
      }

      std::optional<std::size_t> selected;
      for (const auto index : runtime->decision_order) {
        if (!runtime->failure_memory.IsSuppressed(
                runtime->candidates.at(index), *runtime->raster)) {
          selected = index;
          break;
        }
      }
      if (!selected) {
        runtime->state = Status::WAITING_FOR_INPUT;
        runtime->reason_code = runtime->approach
                                   ? ApproachWaitReason(
                                         runtime->approach->wait_reason)
                                   : "WAITING_FOR_MAP_CHANGE";
        runtime->minimum_decision_map_sequence =
            runtime->exploration_map_sequence + 1U;
        PublishLocked(*runtime);
        return;
      }

      runtime->minimum_decision_map_sequence.reset();
      runtime->active_goal = Runtime::ActiveGoal{
          .candidate = runtime->candidates.at(*selected),
          .map_sequence = runtime->exploration_map_sequence,
          .task_generation = runtime->task_generation};
      runtime->state = Status::PLANNING;
      runtime->reason_code = "NAVIGATION_GOAL_SUBMITTED";
      target = NavigationTarget{.pose = runtime->active_goal->candidate.pose};
      geometry_msgs::msg::PoseStamped goal;
      goal.header.stamp = runtime->clock->now();
      goal.header.frame_id = "map";
      goal.pose.position.x = target->pose.x;
      goal.pose.position.y = target->pose.y;
      goal.pose.orientation.z = std::sin(target->pose.yaw / 2.0);
      goal.pose.orientation.w = std::cos(target->pose.yaw / 2.0);
      runtime->current_goal_publisher->publish(goal);
      PublishLocked(*runtime);
    } catch (const std::exception& error) {
      runtime->state = Status::WAITING_FOR_INPUT;
      runtime->reason_code = std::string{"DECISION_UNAVAILABLE: "} +
                             error.what();
      PublishLocked(*runtime);
      return;
    }
  }

  try {
    runtime->navigation_client->Navigate(*target);
  } catch (const std::exception& error) {
    std::scoped_lock lock{runtime->mutex};
    runtime->active_goal.reset();
    runtime->state = Status::WAITING_FOR_INPUT;
    runtime->reason_code =
        std::string{"NAVIGATION_GOAL_NOT_SENT: "} + error.what();
    PublishLocked(*runtime);
  }
}

void HandleFeedback(const std::weak_ptr<Runtime>& weak_runtime,
                    const NavigationFeedbackState feedback) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  std::scoped_lock lock{runtime->mutex};
  if (runtime->teardown || !runtime->active_goal ||
      runtime->active_goal->task_generation != runtime->task_generation ||
      runtime->state == Status::PAUSED || runtime->state == Status::IDLE) {
    return;
  }
  switch (feedback) {
    case NavigationFeedbackState::kPlanning:
      runtime->state = Status::PLANNING;
      runtime->reason_code = "NAVIGATION_PLANNING";
      break;
    case NavigationFeedbackState::kExecuting:
      runtime->state = Status::EXECUTING;
      runtime->reason_code = "NAVIGATION_EXECUTING";
      break;
    case NavigationFeedbackState::kReplanning:
      runtime->state = Status::REPLANNING;
      runtime->reason_code = "NAVIGATION_REPLANNING";
      break;
  }
  PublishLocked(*runtime);
}

void HandleTerminal(const std::weak_ptr<Runtime>& weak_runtime,
                    NavigationTerminal terminal) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  bool decide = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown || !runtime->active_goal) {
      return;
    }
    const auto finished = std::move(*runtime->active_goal);
    runtime->active_goal.reset();
    const bool current_task =
        finished.task_generation == runtime->task_generation;
    const bool expected_cancel = runtime->cancel_expected;
    runtime->cancel_expected = false;
    if (!current_task) {
      decide = runtime->state != Status::IDLE &&
               runtime->state != Status::PAUSED &&
               runtime->state != Status::ERROR;
    } else {
      switch (terminal.outcome) {
        case Action::Result::GOAL_REACHED:
          if (runtime->completed_goal_count !=
              std::numeric_limits<std::uint32_t>::max()) {
            ++runtime->completed_goal_count;
          }
          runtime->state = Status::WAITING_FOR_INPUT;
          runtime->reason_code = "WAITING_FOR_MAP_AFTER_GOAL";
          runtime->minimum_decision_map_sequence =
              finished.map_sequence + 1U;
          decide = runtime->exploration_map_sequence > finished.map_sequence;
          break;
        case Action::Result::NO_PATH:
        case Action::Result::INVALID_GOAL:
          if (runtime->raster) {
            runtime->failure_memory.RecordPersistentFailure(
                finished.candidate,
                PersistentFailureReason::kNavigationNoPath,
                *runtime->raster);
          }
          runtime->state = Status::SELECTING_FRONTIER;
          runtime->reason_code = terminal.reason_code;
          decide = true;
          break;
        case Action::Result::MAP_UNAVAILABLE:
          runtime->state = Status::WAITING_FOR_INPUT;
          runtime->reason_code = "MAP_UNAVAILABLE";
          runtime->minimum_decision_map_sequence =
              runtime->exploration_map_sequence + 1U;
          break;
        case Action::Result::TIMEOUT:
          if (runtime->raster) {
            runtime->failure_memory.RecordPersistentFailure(
                finished.candidate,
                PersistentFailureReason::kNavigationTimeout,
                *runtime->raster);
          }
          runtime->state = Status::SELECTING_FRONTIER;
          runtime->reason_code = "NAVIGATION_TIMEOUT";
          decide = true;
          break;
        case Action::Result::CANCELED:
          if (!expected_cancel) {
            runtime->state = Status::WAITING_FOR_INPUT;
            runtime->reason_code = "UNEXPECTED_NAVIGATION_CANCELED";
            runtime->minimum_decision_map_sequence =
                runtime->exploration_map_sequence + 1U;
          } else {
            decide = runtime->state != Status::IDLE &&
                     runtime->state != Status::PAUSED;
          }
          break;
        case Action::Result::INTERNAL_ERROR:
        default:
          runtime->state = Status::ERROR;
          runtime->reason_code = terminal.reason_code.empty()
                                     ? "NAVIGATION_INTERNAL_ERROR"
                                     : terminal.reason_code;
          break;
      }
    }
    PublishLocked(*runtime);
  }
  if (decide) {
    Decide(runtime);
  }
}

void HandleMap(const std::weak_ptr<Runtime>& weak_runtime,
               const nav_msgs::msg::OccupancyGrid& map) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  bool decide = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    try {
      const auto geometry = MapGeometry(map);
      static_cast<void>(OccupancyGridView(
          geometry, map.data, runtime->parameters.occupied_threshold));
      if (runtime->exploration_map_sequence ==
          std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error{"exploration map sequence exhausted"};
      }
      runtime->latest_map = map;
      ++runtime->exploration_map_sequence;
      if (runtime->task_boundary) {
        RefreshDecisionLocked(*runtime);
      }
      decide = !runtime->task_id.empty() && !runtime->active_goal &&
               runtime->state != Status::PAUSED &&
               runtime->state != Status::IDLE &&
               runtime->state != Status::ERROR;
    } catch (const std::exception& error) {
      if (!runtime->active_goal) {
        runtime->state = Status::WAITING_FOR_INPUT;
      }
      runtime->reason_code = std::string{"INVALID_EXPLORATION_MAP: "} +
                             error.what();
    }
    PublishLocked(*runtime);
  }
  if (decide) {
    Decide(runtime);
  }
}

template <typename Update>
void HandlePose(const std::weak_ptr<Runtime>& weak_runtime, Update&& update) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  bool decide = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    try {
      update(runtime->pose_resolver);
      decide = !runtime->task_id.empty() && !runtime->active_goal &&
               runtime->state != Status::PAUSED &&
               runtime->state != Status::IDLE &&
               runtime->state != Status::ERROR;
    } catch (const std::exception& error) {
      runtime->reason_code = std::string{"INVALID_POSE_INPUT: "} + error.what();
      if (!runtime->active_goal) {
        runtime->state = Status::WAITING_FOR_INPUT;
      }
    }
    PublishLocked(*runtime);
  }
  if (decide) {
    Decide(runtime);
  }
}

void HandleTask(const std::weak_ptr<Runtime>& weak_runtime,
                const Task& task) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  std::optional<Polygon2> start_boundary;
  if (task.command == Task::START) {
    try {
      start_boundary = TaskBoundary(task);
    } catch (const std::exception& error) {
      std::scoped_lock lock{runtime->mutex};
      runtime->reason_code =
          std::string{"INVALID_START: "} + error.what();
      PublishLocked(*runtime);
      return;
    }
  }

  bool cancel = false;
  bool decide = false;
  {
    std::scoped_lock lock{runtime->mutex};
    if (runtime->teardown) {
      return;
    }
    switch (task.command) {
      case Task::START:
        cancel = true;
        runtime->cancel_expected = runtime->active_goal.has_value();
        ++runtime->task_generation;
        runtime->task_id = task.task_id;
        runtime->task_boundary = std::move(start_boundary);
        runtime->task_grid_geometry.reset();
        runtime->failure_memory.BeginTask(runtime->task_id);
        runtime->completed_goal_count = 0U;
        runtime->minimum_decision_map_sequence.reset();
        runtime->state = Status::WAITING_FOR_INPUT;
        runtime->reason_code = "WAITING_FOR_INPUT";
        if (!runtime->active_goal && runtime->latest_map) {
          RefreshDecisionLocked(*runtime);
          decide = true;
        }
        break;
      case Task::PAUSE:
        if (!runtime->task_id.empty()) {
          cancel = true;
          runtime->cancel_expected = runtime->active_goal.has_value();
          runtime->state = Status::PAUSED;
          runtime->reason_code = "PAUSED";
        }
        break;
      case Task::RESUME:
        if (!runtime->task_id.empty()) {
          cancel = true;
          runtime->cancel_expected = runtime->active_goal.has_value();
          runtime->minimum_decision_map_sequence.reset();
          runtime->state = Status::WAITING_FOR_INPUT;
          runtime->reason_code = "RESUMED";
          decide = !runtime->active_goal;
        }
        break;
      case Task::CANCEL:
        cancel = true;
        runtime->cancel_expected = runtime->active_goal.has_value();
        ++runtime->task_generation;
        runtime->task_id.clear();
        runtime->task_boundary.reset();
        runtime->task_grid_geometry.reset();
        runtime->raster.reset();
        runtime->frontiers.clear();
        runtime->candidates.clear();
        runtime->decision_order.clear();
        runtime->approach.reset();
        runtime->minimum_decision_map_sequence.reset();
        runtime->state = Status::IDLE;
        runtime->reason_code = "CANCELED";
        break;
      default:
        runtime->reason_code = "INVALID_EXPLORATION_COMMAND";
        break;
    }
    PublishLocked(*runtime);
  }
  if (cancel) {
    runtime->navigation_client->CancelActive();
  }
  if (decide) {
    Decide(runtime);
  }
}

}  // namespace

IncrementalExplorationNode::IncrementalExplorationNode(
    IncrementalExplorationNodeParameters parameters,
    const rclcpp::NodeOptions& options)
    : rclcpp::Node("incremental_exploration", options) {
  Initialize(std::move(parameters));
}

IncrementalExplorationNode::IncrementalExplorationNode(
    const rclcpp::NodeOptions& options)
    : rclcpp::Node("incremental_exploration", options) {
  Initialize(LoadParameters(*this));
}

void IncrementalExplorationNode::Initialize(
    IncrementalExplorationNodeParameters parameters) {
  auto runtime = std::make_shared<Runtime>(*this, std::move(parameters));
  const std::weak_ptr<Runtime> weak_runtime{runtime};
  runtime->navigation_client = std::make_unique<NavigationClient>(
      *this, runtime->parameters.navigation_action,
      [weak_runtime](const NavigationFeedbackState feedback) {
        HandleFeedback(weak_runtime, feedback);
      },
      [weak_runtime](NavigationTerminal terminal) {
        HandleTerminal(weak_runtime, std::move(terminal));
      });
  exploration_map_subscription_ =
      create_subscription<nav_msgs::msg::OccupancyGrid>(
          runtime->parameters.exploration_map_topic,
          rclcpp::QoS{1}.reliable().transient_local(),
          [weak_runtime](nav_msgs::msg::OccupancyGrid::SharedPtr map) {
            HandleMap(weak_runtime, *map);
          });
  odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      runtime->parameters.odometry_topic, rclcpp::QoS{1}.reliable(),
      [weak_runtime](nav_msgs::msg::Odometry::SharedPtr odometry) {
        HandlePose(weak_runtime, [odometry = std::move(odometry)](
                                     PoseResolver& resolver) {
          resolver.UpdateOdometry(*odometry);
        });
      });
  tf_subscription_ = create_subscription<tf2_msgs::msg::TFMessage>(
      runtime->parameters.tf_topic, rclcpp::QoS{10}.reliable(),
      [weak_runtime](tf2_msgs::msg::TFMessage::SharedPtr transforms) {
        HandlePose(weak_runtime, [transforms = std::move(transforms)](
                                     PoseResolver& resolver) {
          resolver.UpdateTransforms(*transforms);
        });
      });
  task_subscription_ = create_subscription<Task>(
      runtime->parameters.task_topic, rclcpp::QoS{10}.reliable(),
      [weak_runtime](Task::SharedPtr task) {
        HandleTask(weak_runtime, *task);
      });
  runtime_ = std::move(runtime);
  std::scoped_lock lock{runtime_->mutex};
  PublishLocked(*runtime_);
}

IncrementalExplorationNode::~IncrementalExplorationNode() noexcept {
  if (!runtime_) {
    return;
  }
  {
    std::scoped_lock lock{runtime_->mutex};
    runtime_->teardown = true;
  }
  runtime_->navigation_client.reset();
  runtime_.reset();
}

}  // namespace lunar::pure_exploration_ros
