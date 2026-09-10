#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/node.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_core/local_target_selector.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation_ros {

struct PlanningDebugPublisherConfig final {
  bool enabled{};
  std::string topic_prefix;
  double fine_window_m{40.0};
  double cost_display_max{2.0};
};

[[nodiscard]] rclcpp::QoS DebugVisualizationQos();
[[nodiscard]] std::string NormalizeDebugTopicPrefix(std::string_view prefix);
[[nodiscard]] bool ShouldPublishDebugRevision(
    std::optional<std::uint64_t> last_revision,
    std::uint64_t revision) noexcept;
[[nodiscard]] bool ShouldPublishDebugFineSnapshot(
    std::optional<std::uint64_t> last_revision,
    std::optional<lunar::incremental_navigation::Point2> last_center,
    std::uint64_t revision, lunar::incremental_navigation::Point2 center) noexcept;

class PlanningDebugPublisher final {
 public:
  PlanningDebugPublisher(rclcpp::Node& node,
                         PlanningDebugPublisherConfig config);

  [[nodiscard]] bool enabled() const noexcept;
  void PublishSnapshots(const lunar::incremental_navigation::SnapshotBundle& bundle,
                        lunar::incremental_navigation::Point2 center);
  void PublishStartPatch(
      const lunar::incremental_navigation::RequestLocalPlanningView& view,
      double hard_inflation_radius_m);

  void PublishLocalGoals(
      const lunar::incremental_navigation::LocalTarget& target,
      const lunar::incremental_navigation::LocalPlanResult& result,
      const std::string& frame_id);

 private:
  PlanningDebugPublisherConfig config_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
      fine_state_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
      fine_cost_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
      execution_risk_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
      guidance_state_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      traversability_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      start_patch_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      local_goals_publisher_;
  std::optional<std::uint64_t> last_fine_revision_;
  std::optional<lunar::incremental_navigation::Point2> last_fine_center_;
  std::optional<std::uint64_t> last_guidance_revision_;
  std::optional<std::uint64_t> last_patch_fine_revision_;
  std::optional<lunar::incremental_navigation::Point2> last_patch_center_;
};

}  // namespace lunar::incremental_navigation_ros
