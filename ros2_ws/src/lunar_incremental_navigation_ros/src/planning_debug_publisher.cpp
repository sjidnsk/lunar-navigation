#include "lunar_incremental_navigation_ros/planning_debug_publisher.hpp"

#include <stdexcept>
#include <rclcpp/expand_topic_or_service_name.hpp>
#include "lunar_incremental_navigation_core/local_goal_region.hpp"
#include <utility>

#include "lunar_incremental_navigation_ros/planning_snapshot_visualization.hpp"

namespace lunar::incremental_navigation_ros {
std::string NormalizeDebugTopicPrefix(const std::string_view prefix) {
  if (prefix.empty()) {
    throw std::invalid_argument("debug topic prefix must not be empty");
  }
  std::string normalized{prefix};
  if (normalized.front() != '/') {
    normalized.insert(normalized.begin(), '/');
  }
  while (normalized.size() > 1U && normalized.back() == '/') {
    normalized.pop_back();
  }
  if (normalized == "/") {
    throw std::invalid_argument("debug topic prefix must name a namespace");
  }
  try {
    static_cast<void>(rclcpp::expand_topic_or_service_name(
        normalized + "/fine_state", "planning_debug", "/"));
  } catch (const rclcpp::exceptions::InvalidTopicNameError& error) {
    throw std::invalid_argument("debug_topic_prefix: " + std::string(error.what()));
  }
  return normalized;
}

namespace {

[[nodiscard]] bool SamePoint(const lunar::incremental_navigation::Point2& left,
                             const lunar::incremental_navigation::Point2& right) {
  return left.x == right.x && left.y == right.y;
}

}  // namespace

rclcpp::QoS DebugVisualizationQos() {
  return rclcpp::QoS{rclcpp::KeepLast{1}}.reliable().transient_local();
}

bool ShouldPublishDebugRevision(
    const std::optional<std::uint64_t> last_revision,
    const std::uint64_t revision) noexcept {
  return !last_revision || *last_revision != revision;
}

bool ShouldPublishDebugFineSnapshot(
    const std::optional<std::uint64_t> last_revision,
    const std::optional<lunar::incremental_navigation::Point2> last_center,
    const std::uint64_t revision,
    const lunar::incremental_navigation::Point2 center) noexcept {
  return ShouldPublishDebugRevision(last_revision, revision) || !last_center ||
         !SamePoint(*last_center, center);
}

PlanningDebugPublisher::PlanningDebugPublisher(
    rclcpp::Node& node, PlanningDebugPublisherConfig config)
    : config_(std::move(config)), clock_(node.get_clock()) {
  if (!config_.enabled) {
    return;
  }
  config_.topic_prefix = NormalizeDebugTopicPrefix(config_.topic_prefix);
  fine_state_publisher_ =
      node.create_publisher<nav_msgs::msg::OccupancyGrid>(
          config_.topic_prefix + "/fine_state", DebugVisualizationQos());
  fine_cost_publisher_ =
      node.create_publisher<nav_msgs::msg::OccupancyGrid>(
          config_.topic_prefix + "/fine_cost", DebugVisualizationQos());
  execution_risk_publisher_ =
      node.create_publisher<nav_msgs::msg::OccupancyGrid>(
          config_.topic_prefix + "/execution_risk", DebugVisualizationQos());
  guidance_state_publisher_ =
      node.create_publisher<nav_msgs::msg::OccupancyGrid>(
          config_.topic_prefix + "/guidance_state", DebugVisualizationQos());
  traversability_publisher_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>(
          config_.topic_prefix + "/traversability", DebugVisualizationQos());
  local_goals_publisher_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>(
          config_.topic_prefix + "/local_goals", DebugVisualizationQos());
  start_patch_publisher_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>(
          config_.topic_prefix + "/start_patch_cells", DebugVisualizationQos());
}

void PlanningDebugPublisher::PublishLocalGoals(
    const lunar::incremental_navigation::LocalTarget& target,
    const lunar::incremental_navigation::LocalPlanResult& result,
    const std::string& frame_id) {
  if (!enabled()) return;
  using Marker = visualization_msgs::msg::Marker;
  visualization_msgs::msg::MarkerArray array;
  Marker clear;
  clear.action = Marker::DELETEALL;
  clear.header.frame_id = frame_id;
  clear.header.stamp = clock_->now();
  array.markers.push_back(clear);
  Marker candidates;
  candidates.header.frame_id = frame_id;
  candidates.header.stamp = clock_->now();
  candidates.ns = "candidates";
  candidates.id = 0;
  candidates.type = Marker::POINTS;
  candidates.action = Marker::ADD;
  candidates.pose.orientation.w = 1.;
  candidates.scale.x = candidates.scale.y = 0.16;
  candidates.color.g = candidates.color.b = candidates.color.a = 1.;
  if (target.region) {
    for (const auto& candidate : target.region->candidates) {
      geometry_msgs::msg::Point point;
      point.x = candidate.target.center.x;
      point.y = candidate.target.center.y;
      point.z = 0.15;
      candidates.points.push_back(point);
    }
  }
  array.markers.push_back(candidates);
  const auto& path = result.path.empty() ? result.raw_path : result.path;
  if (result.status == lunar::incremental_navigation::LocalPlanResult::Status::kPlanFound &&
      !path.empty()) {
    Marker selected = candidates;
    selected.ns = "selected";
    selected.type = Marker::SPHERE;
    selected.points.clear();
    selected.pose.position.x = path.back().pose.position_m.x;
    selected.pose.position.y = path.back().pose.position_m.y;
    selected.pose.position.z = 0.25;
    selected.scale.x = selected.scale.y = selected.scale.z = 0.4;
    selected.color.r = 1.;
    selected.color.g = 0.;
    array.markers.push_back(selected);
  }
  local_goals_publisher_->publish(array);
}

bool PlanningDebugPublisher::enabled() const noexcept {
  return config_.enabled;
}

void PlanningDebugPublisher::PublishSnapshots(
    const lunar::incremental_navigation::SnapshotBundle& bundle,
    const lunar::incremental_navigation::Point2 center) {
  if (!enabled()) {
    return;
  }
  if (bundle.fine) {
    const std::uint64_t revision = bundle.fine->fine_traversability_revision();
    if (ShouldPublishDebugFineSnapshot(last_fine_revision_, last_fine_center_,
                                       revision, center)) {
      FineVisualization visualization = ProjectFineVisualization(
          *bundle.fine, {.center_map_m = center, .length_m = config_.fine_window_m},
          config_.cost_display_max);
      visualization.state.header.stamp = clock_->now();
      visualization.cost.header.stamp = visualization.state.header.stamp;
      visualization.risk.header.stamp = visualization.state.header.stamp;
      fine_state_publisher_->publish(visualization.state);
      fine_cost_publisher_->publish(visualization.cost);
      execution_risk_publisher_->publish(visualization.risk);
      auto traversability = ProjectTraversabilityVisualization(
          *bundle.fine, {.center_map_m = center, .length_m = config_.fine_window_m},
          config_.cost_display_max, bundle.guidance.get());
      for (auto& marker : traversability.markers) marker.header.stamp = visualization.state.header.stamp;
      traversability_publisher_->publish(traversability);
      last_fine_revision_ = revision;
      last_fine_center_ = center;
    }
  }
  if (bundle.guidance) {
    const std::uint64_t revision = bundle.guidance->global_guidance_revision();
    if (ShouldPublishDebugRevision(last_guidance_revision_, revision)) {
      auto visualization = ProjectGuidanceVisualization(*bundle.guidance);
      visualization.header.stamp = clock_->now();
      guidance_state_publisher_->publish(visualization);
      last_guidance_revision_ = revision;
    }
  }
}

void PlanningDebugPublisher::PublishStartPatch(
    const lunar::incremental_navigation::RequestLocalPlanningView& view,
    const double hard_inflation_radius_m) {
  if (!enabled()) {
    return;
  }
  const auto& fine = *view.base();
  const std::uint64_t revision = fine.fine_traversability_revision();
  const auto center = view.patch_anchor().position_m;
  if (!ShouldPublishDebugRevision(last_patch_fine_revision_, revision) &&
      last_patch_center_ && SamePoint(*last_patch_center_, center)) {
    return;
  }
  auto markers = ProjectStartPatchVisualization(view, hard_inflation_radius_m);
  const auto stamp = clock_->now();
  for (auto& marker : markers.markers) {
    marker.header.stamp = stamp;
  }
  start_patch_publisher_->publish(markers);
  last_patch_fine_revision_ = revision;
  last_patch_center_ = center;
}

}  // namespace lunar::incremental_navigation_ros
