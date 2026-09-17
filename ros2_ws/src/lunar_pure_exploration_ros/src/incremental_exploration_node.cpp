#include "lunar_pure_exploration_ros/incremental_exploration_node.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <list>
#include <map>
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
using lunar::pure_exploration::CalculateCoverage;
using lunar::pure_exploration::CandidateGain;
using lunar::pure_exploration::CandidateKey;
using lunar::pure_exploration::CandidateView;
using lunar::pure_exploration::CoverageStats;
using lunar::pure_exploration::FrontierCluster;
using lunar::pure_exploration::FrontierDetector;
using lunar::pure_exploration::FrontierParameters;
using lunar::pure_exploration::GridGeometry;
using lunar::pure_exploration::OccupancyGridView;
using lunar::pure_exploration::PersistentFailureReason;
using lunar::pure_exploration::Polygon2;
using lunar::pure_exploration::Pose2;
using lunar::pure_exploration::TaskRaster;
using lunar::pure_exploration::Vec2;

constexpr std::int8_t kPublishedExplorationMapOccupiedValue{100};

void RequirePublishedExplorationMapValues(
    const std::vector<std::int8_t>& values) {
  for (const std::int8_t value : values) {
    if (value != -1 && value != 0 && value != 100) {
      throw std::invalid_argument{
          "exploration map values must be exactly -1, 0, or 100"};
    }
  }
}

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

  const auto minimum_frontier =
      node.declare_parameter<double>("minimum_frontier_length_m", 1.0);
  const auto coverage_target =
      node.declare_parameter<double>("coverage_target", 0.80);
  const auto sensor_range =
      node.declare_parameter<double>("sensor_range_m", 10.0);
  const auto sensor_fov_deg =
      node.declare_parameter<double>("sensor_fov_deg", 90.0);
  const auto yaw_offsets_deg = node.declare_parameter<std::vector<double>>(
      "yaw_offsets_deg", {-45.0, -22.5, 0.0, 22.5, 45.0});
  const auto require_positive = [](double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::invalid_argument(std::string{name} + ": must be finite and > 0");
    }
  };
  require_positive(minimum_frontier, "minimum_frontier_length_m");
  require_positive(coverage_target, "coverage_target");
  if (coverage_target > 1.0) throw std::invalid_argument("coverage_target: must be <= 1");
  require_positive(sensor_range, "sensor_range_m");
  require_positive(sensor_fov_deg, "sensor_fov_deg");
  if (sensor_fov_deg > 360.0) throw std::invalid_argument("sensor_fov_deg: must be <= 360");
  if (yaw_offsets_deg.empty()) throw std::invalid_argument("yaw_offsets_deg: must be nonempty");
  lunar::pure_exploration::CandidateParameters candidate_parameters;
  candidate_parameters.yaw_offsets_rad.resize(yaw_offsets_deg.size());
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
      .minimum_frontier_length_m = minimum_frontier,
      .coverage_target = coverage_target,
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
  if (map.header.frame_id.empty()) {
    throw std::invalid_argument{"exploration map frame must be nonempty"};
  }
  return {.width = map.info.width,
          .height = map.info.height,
          .resolution = static_cast<double>(map.info.resolution),
          .origin_x = map.info.origin.position.x,
          .origin_y = map.info.origin.position.y,
          .origin_yaw = QuaternionYaw(map.info.origin.orientation)};
}

bool SameMapEvidence(const nav_msgs::msg::OccupancyGrid& left,
                     const nav_msgs::msg::OccupancyGrid& right) {
  const auto a = MapGeometry(left);
  const auto b = MapGeometry(right);
  return a.width == b.width && a.height == b.height &&
         a.resolution == b.resolution && a.origin_x == b.origin_x &&
         a.origin_y == b.origin_y && a.origin_yaw == b.origin_yaw &&
         left.data == right.data;
}

// A successful observation is separate from a navigation failure. Keep the
// coarse cells intersecting the full sensor-range disk, including UNKNOWN and
// OutsideMap, so an unrelated map update cannot revive the same pose/yaw.
class RecentObservationMemory final {
 public:
  RecentObservationMemory(const double range_m, const std::size_t entry_limit,
                          const std::size_t patch_cell_limit)
      : range_m_(range_m), entry_limit_(entry_limit),
        patch_cell_limit_(patch_cell_limit) {}

  void Clear() {
    entries_.clear();
    by_key_.clear();
    patch_cells_ = 0U;
    eviction_count_ = 0U;
    invalidated_count_ = 0U;
  }

  bool Contains(const CandidateKey& key) const { return by_key_.contains(key); }
  std::size_t size() const { return entries_.size(); }
  std::size_t patch_cells() const { return patch_cells_; }
  std::size_t entry_limit() const { return entry_limit_; }
  std::size_t patch_cell_limit() const { return patch_cell_limit_; }
  std::uint64_t eviction_count() const { return eviction_count_; }
  std::uint64_t invalidated_count() const { return invalidated_count_; }

  void InvalidateChanged(const nav_msgs::msg::OccupancyGrid& map) {
    // Each stored cell is checked at most once per semantic map update. The
    // sum of all patches is bounded by the existing visibility work budget.
    for (auto entry = entries_.begin(); entry != entries_.end();) {
      if (Matches(*entry, map)) {
        ++entry;
      } else {
        entry = Erase(entry);
        Increment(invalidated_count_);
      }
    }
  }

  void Record(const CandidateView& candidate,
              const nav_msgs::msg::OccupancyGrid& map) {
    Entry entry{.key = candidate.key, .patch = Layout(map, candidate.pose)};
    const auto count = static_cast<std::size_t>(entry.patch.geometry.width) *
                       entry.patch.geometry.height;
    entry.states.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
      entry.states.push_back(State(map, entry.patch, index));
    }
    if (const auto found = by_key_.find(candidate.key); found != by_key_.end()) {
      Erase(found->second);
    }
    // Invalid evidence has already been removed on map receipt. Capacity
    // pressure retires the oldest actual observation, never raises the
    // navigation FailureMemory limit or labels a reached view as failed.
    while (entries_.size() >= entry_limit_ ||
           count > patch_cell_limit_ - patch_cells_) {
      Erase(entries_.begin());
      Increment(eviction_count_);
    }
    patch_cells_ += count;
    entries_.push_back(std::move(entry));
    by_key_.emplace(candidate.key, std::prev(entries_.end()));
  }

 private:
  struct Patch final {
    GridGeometry geometry;
    std::int32_t minimum_x;
    std::int32_t minimum_y;
    Vec2 center_in_grid;
    double range_cells;
  };
  struct Entry final {
    CandidateKey key;
    Patch patch;
    std::vector<std::int8_t> states;
  };

  static void Increment(std::uint64_t& count) {
    if (count != std::numeric_limits<std::uint64_t>::max()) {
      ++count;
    }
  }

  std::list<Entry>::iterator Erase(const std::list<Entry>::iterator entry) {
    patch_cells_ -= entry->states.size();
    by_key_.erase(entry->key);
    return entries_.erase(entry);
  }

  Patch Layout(const nav_msgs::msg::OccupancyGrid& map, const Pose2& pose) const {
    const auto source = MapGeometry(map);
    const auto center = OccupancyGridView::WorldToGrid(source, {pose.x, pose.y});
    const double radius = range_m_ / source.resolution;
    if (!center || !std::isfinite(radius)) {
      throw std::overflow_error{"observation patch grid transform overflow"};
    }
    const auto bound = [](const long double value) {
      const auto rounded = std::floor(value);
      if (rounded < std::numeric_limits<std::int32_t>::min() ||
          rounded > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error{"observation patch exceeds GridIndex range"};
      }
      return static_cast<std::int32_t>(rounded);
    };
    const auto minimum_x = bound(static_cast<long double>(center->x) - radius);
    const auto minimum_y = bound(static_cast<long double>(center->y) - radius);
    const auto maximum_x = bound(static_cast<long double>(center->x) + radius);
    const auto maximum_y = bound(static_cast<long double>(center->y) + radius);
    const auto width = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(maximum_x) - minimum_x) + 1U;
    const auto height = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(maximum_y) - minimum_y) + 1U;
    if (height > patch_cell_limit_ || width > patch_cell_limit_ / height ||
        width > std::numeric_limits<std::uint32_t>::max() ||
        height > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error{"observation sensor patch exceeds visibility work limit"};
    }
    const auto origin = OccupancyGridView::GridToWorld(
        source, {static_cast<double>(minimum_x), static_cast<double>(minimum_y)});
    if (!origin) {
      throw std::overflow_error{"observation patch origin overflow"};
    }
    // Geometry belongs to this local patch. Growing map width/height or
    // shifting an aligned map window elsewhere does not change its evidence.
    return {.geometry = {.width = static_cast<std::uint32_t>(width),
                         .height = static_cast<std::uint32_t>(height),
                         .resolution = source.resolution,
                         .origin_x = origin->x, .origin_y = origin->y,
                         .origin_yaw = source.origin_yaw},
            .minimum_x = minimum_x, .minimum_y = minimum_y,
            .center_in_grid = *center, .range_cells = radius};
  }

  static std::int8_t State(const nav_msgs::msg::OccupancyGrid& map,
                           const Patch& patch, const std::size_t index) {
    const auto x = static_cast<std::int64_t>(patch.minimum_x) +
                   static_cast<std::int64_t>(index % patch.geometry.width);
    const auto y = static_cast<std::int64_t>(patch.minimum_y) +
                   static_cast<std::int64_t>(index / patch.geometry.width);
    const double dx = std::max({static_cast<double>(x) - patch.center_in_grid.x,
                               patch.center_in_grid.x - static_cast<double>(x) - 1.0, 0.0});
    const double dy = std::max({static_cast<double>(y) - patch.center_in_grid.y,
                               patch.center_in_grid.y - static_cast<double>(y) - 1.0, 0.0});
    if (dx * dx + dy * dy > patch.range_cells * patch.range_cells) {
      return -3;  // Outside the sensor disk, not a map observation.
    }
    return MapState(map, x, y);
  }

  static std::int8_t MapState(const nav_msgs::msg::OccupancyGrid& map,
                              const std::int64_t x, const std::int64_t y) {
    if (x < 0 || y < 0 || x >= map.info.width || y >= map.info.height) {
      return -2;  // OutsideMap remains distinct from map-backed UNKNOWN (-1).
    }
    return map.data[static_cast<std::size_t>(y) * map.info.width +
                    static_cast<std::size_t>(x)];
  }

  bool Matches(const Entry& entry, const nav_msgs::msg::OccupancyGrid& map) const {
    const auto source = MapGeometry(map);
    const auto& previous = entry.patch.geometry;
    if (source.resolution != previous.resolution ||
        source.origin_yaw != previous.origin_yaw) {
      return false;
    }
    const auto origin_in_grid = OccupancyGridView::WorldToGrid(
        source, {previous.origin_x, previous.origin_y});
    if (!origin_in_grid) {
      return false;
    }
    // Reuse the stored patch lattice and sensor mask. Rebuilding either after
    // an aligned window shift can introduce rounding differences at its origin
    // or disk boundary. Permit only scaled floating-point roundoff, capped far
    // below one cell; a physical fractional-cell shift still invalidates it.
    const double scale = std::max({
        1.0, std::abs(source.origin_x / source.resolution),
        std::abs(source.origin_y / source.resolution),
        std::abs(previous.origin_x / source.resolution),
        std::abs(previous.origin_y / source.resolution),
        std::abs(origin_in_grid->x), std::abs(origin_in_grid->y)});
    const double tolerance = std::min(
        1.0e-7, 32.0 * std::numeric_limits<double>::epsilon() * scale);
    const auto aligned_index = [tolerance](const double coordinate)
        -> std::optional<std::int32_t> {
      const double rounded = std::round(coordinate);
      if (std::abs(coordinate - rounded) > tolerance ||
          rounded < std::numeric_limits<std::int32_t>::min() ||
          rounded > std::numeric_limits<std::int32_t>::max()) {
        return std::nullopt;
      }
      return static_cast<std::int32_t>(rounded);
    };
    const auto minimum_x = aligned_index(origin_in_grid->x);
    const auto minimum_y = aligned_index(origin_in_grid->y);
    if (!minimum_x || !minimum_y) {
      return false;
    }
    for (std::size_t index = 0U; index < entry.states.size(); ++index) {
      if (entry.states[index] == -3) {
        continue;
      }
      const auto x = static_cast<std::int64_t>(*minimum_x) +
                     static_cast<std::int64_t>(index % previous.width);
      const auto y = static_cast<std::int64_t>(*minimum_y) +
                     static_cast<std::int64_t>(index / previous.width);
      if (entry.states[index] != MapState(map, x, y)) {
        return false;
      }
    }
    return true;
  }

  double range_m_;
  std::size_t entry_limit_;
  std::size_t patch_cell_limit_;
  std::size_t patch_cells_{0U};
  std::uint64_t eviction_count_{0U};
  std::uint64_t invalidated_count_{0U};
  std::list<Entry> entries_;
  std::map<CandidateKey, std::list<Entry>::iterator> by_key_;
};

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

Polygon2 TaskBoundary(const Task& task, const std::string& frame) {
  if (task.task_id.empty() || task.header.frame_id != frame ||
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

visualization_msgs::msg::MarkerArray BoundaryMarkers(
    const Polygon2& boundary, const std::string& frame) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame;
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
    std::uint64_t task_generation;
    std::optional<rclcpp::Time> map_wait_started_at;
    std::optional<std::uint64_t> map_wait_timeout_evidence_sequence;
  };

  Runtime(rclcpp::Node& node, IncrementalExplorationNodeParameters input)
      : parameters(std::move(input)),
        candidate_generator(parameters.platform,
                            parameters.candidate_parameters,
                            parameters.candidate_limits),
        information_gain(parameters.sensor_model,
                         parameters.information_gain_limits),
        ranker(parameters.score_weights,
               candidate_generator.platform_length_m()),
        failure_memory(candidate_generator.platform_length_m(),
                       parameters.failure_memory_limits),
        recent_observations(parameters.sensor_model.range_m,
                            parameters.candidate_limits.maximum_candidate_views,
                            parameters.information_gain_limits.maximum_visibility_work_units),
        pose_resolver(parameters.map_frame, parameters.odom_frame, parameters.base_frame),
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
  lunar::pure_exploration::InformationGainEvaluator information_gain;
  lunar::pure_exploration::CandidateRanker ranker;
  lunar::pure_exploration::FailureMemory failure_memory;
  RecentObservationMemory recent_observations;
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
  std::vector<bool> candidate_suppressed;
  std::optional<ActiveGoal> active_goal;
  std::optional<std::uint64_t> minimum_decision_map_sequence;
  std::string task_id;
  std::string reason_code{"IDLE"};
  std::uint8_t state{Status::IDLE};
  std::uint64_t exploration_map_sequence{0U};
  std::uint64_t map_evidence_sequence{0U};
  std::uint64_t task_generation{0U};
  std::uint32_t completed_goal_count{0U};
  std::uint32_t navigation_map_wait_timeout_count{0U};
  std::uint32_t map_backed_free_cell_count{0U};
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

double MapWaitElapsedSecondsLocked(const Runtime& runtime) {
  if (!runtime.active_goal || !runtime.active_goal->map_wait_started_at) {
    return 0.0;
  }
  const auto now = runtime.clock->now();
  const auto started = *runtime.active_goal->map_wait_started_at;
  return now >= started ? (now - started).seconds() : 0.0;
}

void PublishLocked(Runtime& runtime) {
  Status status;
  status.header.stamp = runtime.clock->now();
  status.header.frame_id = runtime.parameters.map_frame;
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
        BoundaryMarkers(*runtime.task_boundary, runtime.parameters.map_frame));
  }
  if (runtime.raster) {
    runtime.task_map_publisher->publish(
        runtime.task_map_marker_builder.Build(*runtime.raster, runtime.parameters.map_frame));
  }
  std::optional<MarkerSelection> selected;
  if (runtime.active_goal &&
      runtime.active_goal->candidate.frontier_canonical_key) {
    selected = MarkerSelection{
        .candidate_key = runtime.active_goal->candidate.key,
        .frontier_canonical_key =
            *runtime.active_goal->candidate.frontier_canonical_key,
        .target = runtime.active_goal->candidate.pose};
  }
  auto frontier_markers = runtime.marker_builder.Build(
      runtime.frontiers, runtime.candidates, selected);
  for (auto& marker : frontier_markers.markers) {
    marker.header = status.header;
    if (marker.ns == "candidates" && marker.action == marker.ADD &&
        marker.id >= 0 && static_cast<std::size_t>(marker.id) <
                              runtime.candidate_suppressed.size()) {
      const auto index = static_cast<std::size_t>(marker.id);
      const bool active = runtime.active_goal &&
          runtime.active_goal->candidate.key == runtime.candidates[index].key;
      if (runtime.candidate_suppressed[index] && !active) {
        marker.color.r = 0.5F;
        marker.color.g = 0.5F;
        marker.color.b = 0.5F;
        marker.color.a = 0.25F;
      }
    }
  }
  runtime.frontiers_publisher->publish(std::move(frontier_markers));

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
  KeyValue successful_candidate_count;
  successful_candidate_count.key = "successful_candidate_count";
  successful_candidate_count.value =
      std::to_string(runtime.recent_observations.size());
  diagnostic.values.push_back(std::move(successful_candidate_count));
  const auto add_count = [&diagnostic](const char* key, const auto count) {
    KeyValue value;
    value.key = key;
    value.value = std::to_string(count);
    diagnostic.values.push_back(std::move(value));
  };
  add_count("eligible_candidate_count", runtime.reachable_candidate_count);
  add_count("suppressed_candidate_count", std::count(
      runtime.candidate_suppressed.begin(), runtime.candidate_suppressed.end(), true));
  add_count("recent_observation_memory_count", runtime.recent_observations.size());
  add_count("recent_observation_memory_patch_cells", runtime.recent_observations.patch_cells());
  add_count("recent_observation_memory_entry_limit", runtime.recent_observations.entry_limit());
  add_count("recent_observation_memory_patch_cell_limit", runtime.recent_observations.patch_cell_limit());
  add_count("recent_observation_memory_eviction_count", runtime.recent_observations.eviction_count());
  add_count("recent_observation_memory_invalidated_count", runtime.recent_observations.invalidated_count());
  KeyValue active_navigation;
  active_navigation.key = "active_navigation";
  active_navigation.value = runtime.active_goal ? "true" : "false";
  diagnostic.values.push_back(std::move(active_navigation));
  KeyValue map_wait_elapsed;
  map_wait_elapsed.key = "navigation_map_wait_elapsed_s";
  map_wait_elapsed.value = std::to_string(MapWaitElapsedSecondsLocked(runtime));
  diagnostic.values.push_back(std::move(map_wait_elapsed));
  KeyValue map_wait_timeout;
  map_wait_timeout.key = "navigation_map_wait_timeout_s";
  map_wait_timeout.value =
      std::to_string(runtime.parameters.navigation_map_wait_timeout_s);
  diagnostic.values.push_back(std::move(map_wait_timeout));
  KeyValue map_wait_cancel_pending;
  map_wait_cancel_pending.key = "navigation_map_wait_cancel_pending";
  map_wait_cancel_pending.value =
      runtime.active_goal &&
              runtime.active_goal->map_wait_timeout_evidence_sequence
          ? "true" : "false";
  diagnostic.values.push_back(std::move(map_wait_cancel_pending));
  KeyValue map_wait_timeout_count;
  map_wait_timeout_count.key = "navigation_map_wait_timeout_count";
  map_wait_timeout_count.value =
      std::to_string(runtime.navigation_map_wait_timeout_count);
  diagnostic.values.push_back(std::move(map_wait_timeout_count));
  diagnostics.status.push_back(std::move(diagnostic));
  runtime.diagnostics_publisher->publish(std::move(diagnostics));
}

void RefreshCandidateSuppressionLocked(Runtime& runtime) {
  runtime.candidate_suppressed.assign(runtime.candidates.size(), false);
  runtime.reachable_candidate_count = 0U;
  if (!runtime.raster) {
    return;
  }
  for (std::size_t index = 0U; index < runtime.candidates.size(); ++index) {
    runtime.candidate_suppressed[index] =
        runtime.recent_observations.Contains(runtime.candidates[index].key) ||
        runtime.failure_memory.IsSuppressed(runtime.candidates[index], *runtime.raster);
  }
  for (const auto index : runtime.decision_order) {
    if (!runtime.candidate_suppressed.at(index)) {
      ++runtime.reachable_candidate_count;
    }
  }
}

void RefreshDecisionLocked(Runtime& runtime) {
  runtime.frontiers.clear();
  runtime.candidates.clear();
  runtime.decision_order.clear();
  runtime.candidate_suppressed.clear();
  runtime.map_backed_free_cell_count = 0U;
  runtime.reachable_candidate_count = 0U;
  if (!runtime.latest_map || !runtime.task_boundary) {
    runtime.raster.reset();
    runtime.coverage = {};
    return;
  }

  const auto geometry = MapGeometry(*runtime.latest_map);
  const OccupancyGridView source_map(
      geometry, runtime.latest_map->data,
      kPublishedExplorationMapOccupiedValue);
  if (!runtime.task_grid_geometry) {
    runtime.task_grid_geometry =
        FixedTaskGeometry(geometry, *runtime.task_boundary);
  }
  const auto projected_data =
      ProjectToFixedGrid(source_map, *runtime.task_grid_geometry);
  const OccupancyGridView map(*runtime.task_grid_geometry, projected_data,
                              kPublishedExplorationMapOccupiedValue);
  runtime.raster = TaskRaster::Build(
      map, *runtime.task_boundary, runtime.parameters.task_raster_limits);
  runtime.coverage = CalculateCoverage(*runtime.raster);
  const auto robot_pose = runtime.pose_resolver.LatestPoseInMap();
  if (!robot_pose) {
    return;
  }

  const auto detection = FrontierDetector(
      FrontierParameters{runtime.parameters.minimum_frontier_length_m})
                             .DetectAll(*runtime.raster);
  runtime.frontiers = detection.clusters;
  runtime.map_backed_free_cell_count = detection.map_backed_free_cell_count;
  if (runtime.map_backed_free_cell_count == 0U) {
    runtime.reason_code = "WAITING_FOR_KNOWN_FREE_MAP_EVIDENCE";
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

  RefreshCandidateSuppressionLocked(runtime);
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
        !runtime->pose_resolver.LatestPoseInMap()) {
      runtime->state = Status::WAITING_FOR_INPUT;
      runtime->reason_code = "WAITING_FOR_INPUT";
      PublishLocked(*runtime);
      return;
    }
    if (runtime->minimum_decision_map_sequence &&
        runtime->exploration_map_sequence <
            *runtime->minimum_decision_map_sequence) {
      // Pose updates do not satisfy a map wait or erase its specific reason.
      runtime->state = Status::WAITING_FOR_INPUT;
      PublishLocked(*runtime);
      return;
    }

    try {
      runtime->state = Status::SELECTING_FRONTIER;
      runtime->reason_code = "SELECTING_FRONTIER";
      RefreshDecisionLocked(*runtime);
      if (runtime->map_backed_free_cell_count == 0U) {
        runtime->state = Status::WAITING_FOR_INPUT;
        runtime->reason_code = "WAITING_FOR_KNOWN_FREE_MAP_EVIDENCE";
        runtime->minimum_decision_map_sequence =
            runtime->exploration_map_sequence + 1U;
        PublishLocked(*runtime);
        return;
      }
      // Coverage milestones (80% / 99%) are reporting targets, not stop gates.
      if (runtime->frontiers.empty() || runtime->candidates.empty()) {
        runtime->state = Status::COMPLETED;
        runtime->reason_code = "COMPLETED_NO_REACHABLE_FRONTIER";
        PublishLocked(*runtime);
        return;
      }

      std::optional<std::size_t> selected;
      for (const auto index : runtime->decision_order) {
        if (!runtime->candidate_suppressed.at(index)) {
          selected = index;
          break;
        }
      }
      if (!selected) {
        runtime->state = Status::WAITING_FOR_INPUT;
        runtime->reason_code = "WAITING_FOR_MAP_CHANGE";
        runtime->minimum_decision_map_sequence =
            runtime->exploration_map_sequence + 1U;
        PublishLocked(*runtime);
        return;
      }

      runtime->minimum_decision_map_sequence.reset();
      runtime->active_goal = Runtime::ActiveGoal{
          .candidate = runtime->candidates.at(*selected),
          .task_generation = runtime->task_generation};
      runtime->state = Status::PLANNING;
      runtime->reason_code = "NAVIGATION_GOAL_SUBMITTED";
      target = NavigationTarget{.pose = runtime->active_goal->candidate.pose};
      geometry_msgs::msg::PoseStamped goal;
      goal.header.stamp = runtime->clock->now();
      goal.header.frame_id = runtime->parameters.map_frame;
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
      runtime->state == Status::PAUSED || runtime->state == Status::IDLE ||
      runtime->cancel_expected) {
    return;
  }
  if (feedback != NavigationFeedbackState::kWaitingForMap) {
    runtime->active_goal->map_wait_started_at.reset();
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
    case NavigationFeedbackState::kWaitingForMap:
      if (!runtime->active_goal->map_wait_started_at ||
          runtime->clock->now() < *runtime->active_goal->map_wait_started_at) {
        runtime->active_goal->map_wait_started_at = runtime->clock->now();
      }
      runtime->state = Status::WAITING_FOR_INPUT;
      runtime->reason_code = "NAVIGATION_WAITING_FOR_MAP";
      break;
  }
  PublishLocked(*runtime);
}

void CheckNavigationMapWaitTimeout(const std::weak_ptr<Runtime>& weak_runtime) {
  const auto runtime = weak_runtime.lock();
  if (!runtime) {
    return;
  }
  std::scoped_lock lock{runtime->mutex};
  if (runtime->teardown || !runtime->active_goal || runtime->cancel_expected ||
      runtime->active_goal->task_generation != runtime->task_generation ||
      !runtime->active_goal->map_wait_started_at ||
      runtime->state == Status::PAUSED || runtime->state == Status::IDLE ||
      runtime->state == Status::ERROR ||
      runtime->parameters.navigation_map_wait_timeout_s <= 0.0) {
    return;
  }
  const auto now = runtime->clock->now();
  if (now < *runtime->active_goal->map_wait_started_at) {
    runtime->active_goal->map_wait_started_at = now;
    return;
  }
  if (MapWaitElapsedSecondsLocked(*runtime) <
      runtime->parameters.navigation_map_wait_timeout_s) {
    return;
  }
  runtime->active_goal->map_wait_timeout_evidence_sequence =
      runtime->map_evidence_sequence;
  runtime->cancel_expected = true;
  runtime->state = Status::WAITING_FOR_INPUT;
  runtime->reason_code = "NAVIGATION_MAP_WAIT_TIMEOUT_CANCELING";
  // Keep ownership until the navigator confirms a terminal result. Cancel
  // acceptance alone is neither a completed goal nor evidence of no path.
  runtime->navigation_client->CancelActive();
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
          // Use the latest local evidence even when this candidate vanished
          // while navigation was active. START/CANCEL generation checks above
          // prevent an old task's success from entering the new memory.
          try {
            runtime->recent_observations.Record(finished.candidate, *runtime->latest_map);
          } catch (const std::exception& error) {
            runtime->minimum_decision_map_sequence = runtime->exploration_map_sequence + 1U;
            if (runtime->state != Status::PAUSED) {
              runtime->state = Status::WAITING_FOR_INPUT;
              runtime->reason_code = std::string{"RECENT_OBSERVATION_MEMORY_UNAVAILABLE: "} + error.what();
            }
            break;
          }
          runtime->minimum_decision_map_sequence.reset();
          if (runtime->state != Status::PAUSED) {
            runtime->state = Status::SELECTING_FRONTIER;
            runtime->reason_code = "GOAL_REACHED";
            decide = true;
          }
          break;
        case Action::Result::NO_PATH:
        case Action::Result::INVALID_GOAL:
          if (runtime->raster) {
            runtime->failure_memory.RecordPersistentFailure(
                finished.candidate,
                PersistentFailureReason::kNavigationNoPath,
                *runtime->raster);
          }
          if (runtime->state != Status::PAUSED) {
            runtime->state = Status::SELECTING_FRONTIER;
            runtime->reason_code = terminal.reason_code;
            decide = true;
          }
          break;
        case Action::Result::MAP_UNAVAILABLE:
          if (runtime->state != Status::PAUSED) {
            runtime->state = Status::WAITING_FOR_INPUT;
            runtime->reason_code = "MAP_UNAVAILABLE";
          }
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
          if (runtime->state != Status::PAUSED) {
            runtime->state = Status::SELECTING_FRONTIER;
            runtime->reason_code = "NAVIGATION_TIMEOUT";
            decide = true;
          }
          break;
        case Action::Result::CANCELED:
          if (finished.map_wait_timeout_evidence_sequence) {
            const bool same_evidence =
                *finished.map_wait_timeout_evidence_sequence ==
                runtime->map_evidence_sequence;
            if (same_evidence) {
              try {
                if (!runtime->raster) {
                  throw std::logic_error{"map-wait timeout has no task raster"};
                }
                runtime->failure_memory.RecordPersistentFailure(
                    finished.candidate,
                    PersistentFailureReason::kNavigationTimeout,
                    *runtime->raster);
              } catch (const std::exception& error) {
                runtime->state = Status::ERROR;
                runtime->reason_code =
                    std::string{"NAVIGATION_MAP_WAIT_SUPPRESSION_FAILED: "} +
                    error.what();
                break;
              }
            }
            if (runtime->navigation_map_wait_timeout_count !=
                std::numeric_limits<std::uint32_t>::max()) {
              ++runtime->navigation_map_wait_timeout_count;
            }
            runtime->minimum_decision_map_sequence.reset();
            if (runtime->state != Status::PAUSED &&
                runtime->state != Status::IDLE && runtime->state != Status::ERROR) {
              runtime->state = Status::SELECTING_FRONTIER;
              runtime->reason_code = same_evidence
                  ? "NAVIGATION_MAP_WAIT_TIMEOUT"
                  : "NAVIGATION_MAP_WAIT_CANCELED_AFTER_MAP_CHANGE";
              decide = true;
            }
          } else if (!expected_cancel) {
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
      RefreshCandidateSuppressionLocked(*runtime);
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
      if (map.header.frame_id != runtime->parameters.map_frame) {
        throw std::invalid_argument("exploration map frame mismatch: expected " + runtime->parameters.map_frame);
      }
      const auto geometry = MapGeometry(map);
      RequirePublishedExplorationMapValues(map.data);
      static_cast<void>(OccupancyGridView(
          geometry, map.data, kPublishedExplorationMapOccupiedValue));
      if (runtime->exploration_map_sequence ==
          std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error{"exploration map sequence exhausted"};
      }
      const bool evidence_changed =
          !runtime->latest_map || !SameMapEvidence(*runtime->latest_map, map);
      runtime->latest_map = map;
      ++runtime->exploration_map_sequence;
      if (evidence_changed) {
        runtime->recent_observations.InvalidateChanged(map);
        runtime->map_evidence_sequence = runtime->exploration_map_sequence;
        if (runtime->active_goal && runtime->active_goal->map_wait_started_at &&
            !runtime->cancel_expected) {
          runtime->active_goal->map_wait_started_at = runtime->clock->now();
        }
      }
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
      start_boundary = TaskBoundary(task, runtime->parameters.map_frame);
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
        runtime->recent_observations.Clear();
        runtime->candidate_suppressed.assign(runtime->candidates.size(), false);
        runtime->completed_goal_count = 0U;
        runtime->navigation_map_wait_timeout_count = 0U;
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
          if (runtime->active_goal) {
            runtime->active_goal->map_wait_started_at.reset();
          }
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
        runtime->recent_observations.Clear();
        runtime->candidate_suppressed.clear();
        runtime->map_backed_free_cell_count = 0U;
        runtime->reachable_candidate_count = 0U;
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

rclcpp::QoS ConfiguredExplorationMapQos(rclcpp::Node& node) {
  const auto reliability = node.declare_parameter<std::string>("exploration_map_qos_reliability", "reliable");
  const auto durability = node.declare_parameter<std::string>("exploration_map_qos_durability", "transient_local");
  const auto depth = node.declare_parameter<int>("exploration_map_qos_depth", 1);
  if (depth <= 0) throw std::invalid_argument("exploration_map_qos_depth must be positive");
  rclcpp::QoS qos{rclcpp::KeepLast{static_cast<std::size_t>(depth)}};
  if (reliability == "reliable") qos.reliable();
  else if (reliability == "best_effort") qos.best_effort();
  else throw std::invalid_argument("exploration_map_qos_reliability: expected reliable or best_effort");
  if (durability == "transient_local") qos.transient_local();
  else if (durability == "volatile") qos.durability_volatile();
  else throw std::invalid_argument("exploration_map_qos_durability: expected transient_local or volatile");
  return qos;
}

void IncrementalExplorationNode::Initialize(
    IncrementalExplorationNodeParameters parameters) {
  parameters.map_frame = declare_parameter<std::string>("map_frame", parameters.map_frame);
  parameters.odom_frame = declare_parameter<std::string>("odom_frame", parameters.odom_frame);
  parameters.base_frame = declare_parameter<std::string>("base_frame", parameters.base_frame);
  if (parameters.map_frame.empty() || parameters.odom_frame.empty() || parameters.base_frame.empty() || parameters.map_frame == parameters.odom_frame || parameters.odom_frame == parameters.base_frame || parameters.map_frame == parameters.base_frame) {
    throw std::invalid_argument("map_frame, odom_frame, base_frame must be nonempty distinct names");
  }
  parameters.navigation_map_wait_timeout_s = declare_parameter<double>(
      "navigation_map_wait_timeout_s", parameters.navigation_map_wait_timeout_s);
  if (!std::isfinite(parameters.navigation_map_wait_timeout_s) ||
      parameters.navigation_map_wait_timeout_s < 0.0) {
    throw std::invalid_argument{
        "navigation_map_wait_timeout_s must be finite and nonnegative"};
  }
  const auto exploration_qos = ConfiguredExplorationMapQos(*this);
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
          exploration_qos,
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

  if (runtime_->parameters.navigation_map_wait_timeout_s > 0.0) {
    navigation_map_wait_timer_ = create_wall_timer(
        std::chrono::milliseconds{50}, [weak_runtime] {
          CheckNavigationMapWaitTimeout(weak_runtime);
        });
  }
  std::scoped_lock lock{runtime_->mutex};
  PublishLocked(*runtime_);
}

IncrementalExplorationNode::~IncrementalExplorationNode() noexcept {
  navigation_map_wait_timer_.reset();
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
