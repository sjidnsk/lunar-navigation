#include "lunar_planner_ros/route_marker_publisher.hpp"

#include <chrono>
#include <ranges>
#include <string>

#include <gtest/gtest.h>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::planning::ros {
namespace {

using namespace std::chrono_literals;

lunar::planning::CertifiedHopPreview Hop(
    std::string segment_id, const double launch_x) {
  return lunar::planning::CertifiedHopPreview{
      .segment_id = std::move(segment_id),
      .launch_pose_map = {
          .position_m = {launch_x, 0.0, 0.5},
          .orientation = {},
      },
      .landing_pose_map = {
          .position_m = {launch_x + 4.0, 0.0, 0.0},
          .orientation = {},
      },
      .launch_velocity_mps = {2.0, 0.0, 2.0},
      .flight_time = 2s,
      .flight_tube_radius_m = 0.25,
      .landing_region_map = {
          {launch_x + 3.5, -0.5, 0.0},
          {launch_x + 4.5, -0.5, 0.0},
          {launch_x + 4.5, 0.5, 0.0},
          {launch_x + 3.5, 0.5, 0.0},
      },
      .promotion_region_map = {
          {launch_x + 3.7, -0.3, 0.0},
          {launch_x + 4.3, -0.3, 0.0},
          {launch_x + 4.3, 0.3, 0.0},
          {launch_x + 3.7, 0.3, 0.0},
      },
      .position_uncertainty_m = 0.1,
      .velocity_uncertainty_mps = 0.1,
  };
}

TEST(RouteMarkerPublisher, RendersFutureArcTubeLandingAndPromotionEvidence) {
  RouteMarkerPublisher publisher;
  const RouteMarkerContext context{
      .stamp = builtin_interfaces::msg::Time{}.set__sec(10),
      .route_id = "hopper-route/request-1",
      .reference_plan_id = "hopper/request-1",
      .authorized_segment_id = "hop-1",
      .map_from_odom = {
          .parent_frame = "map",
          .child_frame = "odom",
          .stamp = {10'000'000'000LL},
          .translation_m = {1.0, 0.0, 0.0},
          .rotation = {},
      },
      .gravity_mps2 = {0.0, 0.0, -1.62},
  };
  const auto markers = publisher.Replace(
      {Hop("hop-1", 0.0), Hop("hop-2", 4.0)}, context);

  const auto has_namespace = [&](const std::string& marker_namespace) {
    return std::ranges::any_of(markers.markers, [&](const auto& marker) {
      return marker.action == visualization_msgs::msg::Marker::ADD &&
          marker.ns == marker_namespace;
    });
  };
  EXPECT_TRUE(has_namespace("certified_hop_arc"));
  EXPECT_TRUE(has_namespace("certified_hop_tube"));
  EXPECT_TRUE(has_namespace("certified_hop_landing_region"));
  EXPECT_TRUE(has_namespace("certified_hop_promotion_region"));
  EXPECT_TRUE(has_namespace("authorized_hop_arc"));
  EXPECT_TRUE(std::ranges::any_of(markers.markers, [](const auto& marker) {
    return marker.header.frame_id == "map";
  }));
  EXPECT_TRUE(std::ranges::any_of(markers.markers, [](const auto& marker) {
    return marker.header.frame_id == "odom";
  }));
}

TEST(RouteMarkerPublisher, DeletesOnlyPreviouslyOwnedNamespaceAndIds) {
  RouteMarkerPublisher publisher;
  RouteMarkerContext context{
      .stamp = builtin_interfaces::msg::Time{}.set__sec(10),
      .route_id = "route-1",
      .reference_plan_id = "plan-1",
      .authorized_segment_id = "hop-1",
      .map_from_odom = {
          .parent_frame = "map",
          .child_frame = "odom",
          .stamp = {10'000'000'000LL},
          .translation_m = {},
          .rotation = {},
      },
      .gravity_mps2 = {0.0, 0.0, -1.62},
  };
  const auto added = publisher.Replace({Hop("hop-1", 0.0)}, context);
  const std::size_t owned = std::ranges::count_if(
      added.markers, [](const auto& marker) {
        return marker.action == visualization_msgs::msg::Marker::ADD;
      });

  const auto deleted = publisher.DeleteOwned(
      builtin_interfaces::msg::Time{}.set__sec(11));
  EXPECT_EQ(deleted.markers.size(), owned);
  EXPECT_TRUE(std::ranges::all_of(deleted.markers, [](const auto& marker) {
    return marker.action == visualization_msgs::msg::Marker::DELETE &&
        marker.action != visualization_msgs::msg::Marker::DELETEALL;
  }));
  EXPECT_TRUE(publisher.DeleteOwned(
      builtin_interfaces::msg::Time{}.set__sec(12)).markers.empty());
}

}  // namespace
}  // namespace lunar::planning::ros
