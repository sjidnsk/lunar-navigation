#include "lunar_planner_ros/provisional_route_marker_publisher.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::planning::ros {
namespace {

using Marker = visualization_msgs::msg::Marker;
constexpr const char* kRouteNamespace = "provisional_global_route";
constexpr const char* kLabelNamespace = "provisional_global_route_label";

[[nodiscard]] geometry_msgs::msg::Point Point(const Vec3& value) {
  geometry_msgs::msg::Point point;
  point.x = value.x;
  point.y = value.y;
  point.z = value.z + 0.08;
  return point;
}

}  // namespace

visualization_msgs::msg::MarkerArray ProvisionalRouteMarkerPublisher::Replace(
    const ProvisionalGlobalRoute& route,
    const builtin_interfaces::msg::Time& stamp) {
  visualization_msgs::msg::MarkerArray result = DeleteOwned(stamp);
  if (route.poses_map.empty()) {
    return result;
  }

  Marker line;
  line.header.frame_id = "map";
  line.header.stamp = stamp;
  line.ns = kRouteNamespace;
  line.id = 0;
  line.type = Marker::LINE_LIST;
  line.action = Marker::ADD;
  line.pose.orientation.w = 1.0;
  line.scale.x = 0.04;
  line.color.r = 0.0F;
  line.color.g = 0.85F;
  line.color.b = 1.0F;
  line.color.a = 0.55F;
  for (std::size_t index = 1U; index < route.poses_map.size(); ++index) {
    if ((index - 1U) % 2U != 0U) {
      continue;
    }
    line.points.push_back(Point(route.poses_map[index - 1U].position_m));
    line.points.push_back(Point(route.poses_map[index].position_m));
  }
  if (!line.points.empty()) {
    owned_markers_.emplace(line.ns, line.id, line.header.frame_id);
    result.markers.push_back(std::move(line));
  }

  Marker label;
  label.header.frame_id = "map";
  label.header.stamp = stamp;
  label.ns = kLabelNamespace;
  label.id = 0;
  label.type = Marker::TEXT_VIEW_FACING;
  label.action = Marker::ADD;
  label.pose.position = Point(route.poses_map.front().position_m);
  label.pose.position.z += 0.35;
  label.pose.orientation.w = 1.0;
  label.scale.z = 0.22;
  label.color.r = 0.0F;
  label.color.g = 0.85F;
  label.color.b = 1.0F;
  label.color.a = 0.75F;
  label.text = "provisional " + route.route_id + " request=" +
      route.request_id;
  owned_markers_.emplace(label.ns, label.id, label.header.frame_id);
  result.markers.push_back(std::move(label));
  return result;
}

visualization_msgs::msg::MarkerArray
ProvisionalRouteMarkerPublisher::DeleteOwned(
    const builtin_interfaces::msg::Time& stamp) {
  visualization_msgs::msg::MarkerArray result;
  result.markers.reserve(owned_markers_.size());
  for (const auto& [marker_namespace, id, frame] : owned_markers_) {
    Marker marker;
    marker.header.frame_id = frame;
    marker.header.stamp = stamp;
    marker.ns = marker_namespace;
    marker.id = id;
    marker.action = Marker::DELETE;
    result.markers.push_back(std::move(marker));
  }
  owned_markers_.clear();
  return result;
}

}  // namespace lunar::planning::ros
