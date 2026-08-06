#include "lunar_planner_ros/route_marker_publisher.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::planning::ros {
namespace {

using Marker = visualization_msgs::msg::Marker;

[[nodiscard]] std_msgs::msg::ColorRGBA Color(
    const float red, const float green, const float blue, const float alpha) {
  std_msgs::msg::ColorRGBA color;
  color.r = red;
  color.g = green;
  color.b = blue;
  color.a = alpha;
  return color;
}

[[nodiscard]] geometry_msgs::msg::Point Point(
    const lunar::planning::Vec3& value) {
  geometry_msgs::msg::Point point;
  point.x = value.x;
  point.y = value.y;
  point.z = value.z;
  return point;
}

[[nodiscard]] lunar::planning::Vec3 RotateInverse(
    const lunar::planning::Quaternion& rotation,
    const lunar::planning::Vec3& value) noexcept {
  const lunar::planning::Vec3 cross{
      rotation.y * value.z - rotation.z * value.y,
      rotation.z * value.x - rotation.x * value.z,
      rotation.x * value.y - rotation.y * value.x,
  };
  const lunar::planning::Vec3 second_cross{
      rotation.y * cross.z - rotation.z * cross.y,
      rotation.z * cross.x - rotation.x * cross.z,
      rotation.x * cross.y - rotation.y * cross.x,
  };
  return lunar::planning::Vec3{
      .x = value.x - 2.0 * rotation.w * cross.x + 2.0 * second_cross.x,
      .y = value.y - 2.0 * rotation.w * cross.y + 2.0 * second_cross.y,
      .z = value.z - 2.0 * rotation.w * cross.z + 2.0 * second_cross.z,
  };
}

[[nodiscard]] lunar::planning::Vec3 ToOdom(
    const lunar::planning::Vec3& map_point,
    const lunar::planning::RigidTransform& map_from_odom) noexcept {
  return RotateInverse(
      map_from_odom.rotation,
      lunar::planning::Vec3{
          .x = map_point.x - map_from_odom.translation_m.x,
          .y = map_point.y - map_from_odom.translation_m.y,
          .z = map_point.z - map_from_odom.translation_m.z,
      });
}

[[nodiscard]] std::vector<lunar::planning::Vec3> BallisticArc(
    const lunar::planning::CertifiedHopPreview& hop,
    const lunar::planning::Vec3& gravity) {
  const double seconds = std::chrono::duration<double>(hop.flight_time).count();
  const std::size_t samples = std::max<std::size_t>(
      2U, static_cast<std::size_t>(std::ceil(seconds / 0.1)) + 1U);
  std::vector<lunar::planning::Vec3> points;
  points.reserve(samples);
  for (std::size_t index = 0U; index < samples; ++index) {
    const double time = seconds * static_cast<double>(index) /
        static_cast<double>(samples - 1U);
    points.push_back(lunar::planning::Vec3{
        .x = hop.launch_pose_map.position_m.x +
            hop.launch_velocity_mps.x * time +
            0.5 * gravity.x * time * time,
        .y = hop.launch_pose_map.position_m.y +
            hop.launch_velocity_mps.y * time +
            0.5 * gravity.y * time * time,
        .z = hop.launch_pose_map.position_m.z +
            hop.launch_velocity_mps.z * time +
            0.5 * gravity.z * time * time,
    });
  }
  return points;
}

[[nodiscard]] Marker LineMarker(
    std::string marker_namespace,
    const std::int32_t id,
    std::string frame,
    const builtin_interfaces::msg::Time& stamp,
    const double width,
    const std_msgs::msg::ColorRGBA& color,
    const std::vector<lunar::planning::Vec3>& points,
    const bool close) {
  Marker marker;
  marker.header.frame_id = std::move(frame);
  marker.header.stamp = stamp;
  marker.ns = std::move(marker_namespace);
  marker.id = id;
  marker.type = Marker::LINE_STRIP;
  marker.action = Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = width;
  marker.color = color;
  marker.points.reserve(points.size() + (close && !points.empty() ? 1U : 0U));
  for (const auto& value : points) {
    marker.points.push_back(Point(value));
  }
  if (close && !points.empty()) {
    marker.points.push_back(Point(points.front()));
  }
  return marker;
}

}  // namespace

visualization_msgs::msg::MarkerArray RouteMarkerPublisher::Replace(
    const std::vector<lunar::planning::CertifiedHopPreview>& hops,
    const RouteMarkerContext& context) {
  visualization_msgs::msg::MarkerArray result = DeleteOwned(context.stamp);
  const auto add = [this, &result](Marker marker) {
    owned_markers_.emplace(marker.ns, marker.id, marker.header.frame_id);
    result.markers.push_back(std::move(marker));
  };

  for (std::size_t index = 0U; index < hops.size(); ++index) {
    const auto& hop = hops[index];
    const auto id = static_cast<std::int32_t>(index);
    const bool authorized = hop.segment_id == context.authorized_segment_id;
    const auto arc = BallisticArc(hop, context.gravity_mps2);
    add(LineMarker(
        "certified_hop_arc", id, "map", context.stamp, 0.08,
        authorized ? Color(1.0F, 0.75F, 0.1F, 1.0F) :
                     Color(0.45F, 0.75F, 1.0F, 0.75F),
        arc, false));
    add(LineMarker(
        "certified_hop_tube", id, "map", context.stamp,
        std::max(0.02, 2.0 * hop.flight_tube_radius_m),
        authorized ? Color(1.0F, 0.55F, 0.1F, 0.18F) :
                     Color(0.35F, 0.65F, 1.0F, 0.12F),
        arc, false));
    add(LineMarker(
        "certified_hop_landing_region", id, "map", context.stamp, 0.06,
        Color(0.25F, 1.0F, 0.35F, authorized ? 1.0F : 0.65F),
        hop.landing_region_map, true));
    add(LineMarker(
        "certified_hop_promotion_region", id, "map", context.stamp, 0.045,
        Color(0.95F, 0.25F, 1.0F, authorized ? 1.0F : 0.65F),
        hop.promotion_region_map, true));

    Marker identity;
    identity.header.frame_id = "map";
    identity.header.stamp = context.stamp;
    identity.ns = "certified_hop_identity";
    identity.id = id;
    identity.type = Marker::TEXT_VIEW_FACING;
    identity.action = Marker::ADD;
    identity.pose.position = Point(hop.launch_pose_map.position_m);
    identity.pose.position.z += 0.6;
    identity.pose.orientation.w = 1.0;
    identity.scale.z = 0.3;
    identity.color = Color(0.9F, 0.95F, 1.0F, 1.0F);
    identity.text = "route=" + context.route_id + " reference=" +
        context.reference_plan_id + " segment=" + hop.segment_id;
    add(std::move(identity));

    if (authorized) {
      std::vector<lunar::planning::Vec3> odom_arc;
      odom_arc.reserve(arc.size());
      for (const auto& point : arc) {
        odom_arc.push_back(ToOdom(point, context.map_from_odom));
      }
      add(LineMarker(
          "authorized_hop_arc", id, "odom", context.stamp, 0.11,
          Color(1.0F, 0.8F, 0.05F, 1.0F), odom_arc, false));
      add(LineMarker(
          "authorized_hop_tube", id, "odom", context.stamp,
          std::max(0.02, 2.0 * hop.flight_tube_radius_m),
          Color(1.0F, 0.5F, 0.05F, 0.2F), odom_arc, false));
    }
  }
  return result;
}

visualization_msgs::msg::MarkerArray RouteMarkerPublisher::DeleteOwned(
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
