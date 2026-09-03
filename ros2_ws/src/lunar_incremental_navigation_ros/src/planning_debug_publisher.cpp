#include "lunar_incremental_navigation_ros/planning_debug_publisher.hpp"

#include <stdexcept>
#include <utility>

#include "lunar_incremental_navigation_ros/planning_snapshot_visualization.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

[[nodiscard]] bool IsPlanningDemoTopicPrefix(
    const std::string_view prefix) noexcept {
  constexpr std::string_view kRoot{"/planning_demo"};
  return prefix.size() > kRoot.size() && prefix.starts_with(kRoot) &&
         prefix[kRoot.size()] == '/';
}

}  // namespace

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
  if (!IsPlanningDemoTopicPrefix(normalized)) {
    throw std::invalid_argument(
        "debug topic prefix must be below /planning_demo/");
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
  start_patch_publisher_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>(
          config_.topic_prefix + "/start_patch_cells", DebugVisualizationQos());
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
