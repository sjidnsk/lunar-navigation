#include "lunar_planner_ros/provisional_route_marker_publisher.hpp"

#include <algorithm>

#include <gtest/gtest.h>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::planning::ros {
namespace {

TEST(ProvisionalRouteMarkerPublisher, UsesOwnedCyanDashedMarkers) {
  ProvisionalRouteMarkerPublisher publisher;
  const builtin_interfaces::msg::Time stamp;
  const ProvisionalGlobalRoute route{
      .request_id = "request-1",
      .route_id = "wheel-route/request-1",
      .platform_type = PlatformType::kWheeled,
      .poses_map = {
          Pose3{.position_m = {0.0, 0.0, 0.0}},
          Pose3{.position_m = {1.0, 0.0, 0.0}},
          Pose3{.position_m = {2.0, 0.0, 0.0}},
          Pose3{.position_m = {3.0, 0.0, 0.0}},
      },
  };

  const auto added = publisher.Replace(route, stamp);

  const auto line = std::ranges::find_if(added.markers, [](const auto& marker) {
    return marker.ns == "provisional_global_route" &&
        marker.action == visualization_msgs::msg::Marker::ADD;
  });
  ASSERT_NE(line, added.markers.end());
  EXPECT_EQ(line->type, visualization_msgs::msg::Marker::LINE_LIST);
  EXPECT_EQ(line->header.frame_id, "map");
  EXPECT_FLOAT_EQ(line->color.r, 0.0F);
  EXPECT_FLOAT_EQ(line->color.g, 0.85F);
  EXPECT_FLOAT_EQ(line->color.b, 1.0F);
  EXPECT_FLOAT_EQ(line->color.a, 0.55F);
  EXPECT_DOUBLE_EQ(line->scale.x, 0.04);
  ASSERT_EQ(line->points.size(), 4U);

  const auto deleted = publisher.DeleteOwned(stamp);
  EXPECT_EQ(deleted.markers.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(deleted.markers, [](const auto& marker) {
    return marker.action == visualization_msgs::msg::Marker::DELETE;
  }));
  EXPECT_TRUE(publisher.DeleteOwned(stamp).markers.empty());
}

}  // namespace
}  // namespace lunar::planning::ros
