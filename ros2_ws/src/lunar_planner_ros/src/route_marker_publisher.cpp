#include "lunar_planner_ros/route_marker_publisher.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
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
    const lunar::planning::Vec3 value) {
  geometry_msgs::msg::Point point;
  point.x = value.x;
  point.y = value.y;
  point.z = value.z;
  return point;
}

[[nodiscard]] geometry_msgs::msg::Quaternion Quaternion(
    const lunar::planning::Quaternion value) {
  geometry_msgs::msg::Quaternion result;
  result.w = value.w;
  result.x = value.x;
  result.y = value.y;
  result.z = value.z;
  return result;
}

[[nodiscard]] geometry_msgs::msg::Pose Pose(
    const lunar::planning::Pose3& value) {
  geometry_msgs::msg::Pose pose;
  pose.position = Point(value.position_m);
  pose.orientation = Quaternion(value.orientation);
  return pose;
}

[[nodiscard]] lunar::planning::Quaternion Multiply(
    const lunar::planning::Quaternion lhs,
    const lunar::planning::Quaternion rhs) noexcept {
  return {
      .w = lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
      .x = lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
      .y = lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
      .z = lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
  };
}

[[nodiscard]] lunar::planning::Vec3 Rotate(
    const lunar::planning::Quaternion q,
    const lunar::planning::Vec3 value) noexcept {
  const lunar::planning::Vec3 axis{q.x, q.y, q.z};
  const lunar::planning::Vec3 first{
      axis.y * value.z - axis.z * value.y,
      axis.z * value.x - axis.x * value.z,
      axis.x * value.y - axis.y * value.x,
  };
  const lunar::planning::Vec3 second{
      axis.y * first.z - axis.z * first.y,
      axis.z * first.x - axis.x * first.z,
      axis.x * first.y - axis.y * first.x,
  };
  return {
      .x = value.x + 2.0 * q.w * first.x + 2.0 * second.x,
      .y = value.y + 2.0 * q.w * first.y + 2.0 * second.y,
      .z = value.z + 2.0 * q.w * first.z + 2.0 * second.z,
  };
}

[[nodiscard]] lunar::planning::Pose3 OffsetPose(
    const lunar::planning::Pose3& base,
    const lunar::planning::Vec3 offset) noexcept {
  const lunar::planning::Vec3 rotated = Rotate(base.orientation, offset);
  return {
      .position_m = {
          base.position_m.x + rotated.x,
          base.position_m.y + rotated.y,
          base.position_m.z + rotated.z,
      },
      .orientation = base.orientation,
  };
}

[[nodiscard]] Marker BaseMarker(
    std::string marker_namespace, const std::int32_t id,
    const std::int32_t type, std::string frame,
    const builtin_interfaces::msg::Time& stamp,
    const std_msgs::msg::ColorRGBA& color) {
  Marker marker;
  marker.header.frame_id = std::move(frame);
  marker.header.stamp = stamp;
  marker.ns = std::move(marker_namespace);
  marker.id = id;
  marker.type = type;
  marker.action = Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.color = color;
  return marker;
}

[[nodiscard]] Marker LineMarker(
    std::string marker_namespace, const std::int32_t id,
    std::string frame, const builtin_interfaces::msg::Time& stamp,
    const double width, const std_msgs::msg::ColorRGBA& color,
    const std::vector<lunar::planning::Vec3>& points,
    const bool close = false) {
  Marker marker = BaseMarker(
      std::move(marker_namespace), id, Marker::LINE_STRIP,
      std::move(frame), stamp, color);
  marker.scale.x = width;
  marker.points.reserve(points.size() + (close && !points.empty() ? 1U : 0U));
  for (const auto point : points) {
    marker.points.push_back(Point(point));
  }
  if (close && !points.empty()) {
    marker.points.push_back(Point(points.front()));
  }
  return marker;
}

void AddCommonReferenceMarkers(
    std::vector<Marker>& markers,
    const lunar::planning::PlannerOutput& output,
    const lunar::planning::PlannerInput& input,
    const RouteMarkerContext& context) {
  if (!output.reference.has_value()) {
    return;
  }
  const auto& reference = *output.reference;
  if (!reference.preview.poses_map.empty()) {
    Marker start = BaseMarker(
        "planning_start", 0, Marker::SPHERE, "map", context.stamp,
        Color(0.15F, 0.95F, 0.25F, 1.0F));
    start.pose = Pose(reference.preview.poses_map.front());
    start.scale.x = 0.35;
    start.scale.y = 0.35;
    start.scale.z = 0.35;
    markers.push_back(std::move(start));

    if (reference.platform_type != lunar::planning::PlatformType::kHopper) {
      const std::array<std::pair<const char*, std_msgs::msg::ColorRGBA>, 3>
          semantics{
              std::pair{
                  "roughness: cost/speed diagnostic",
                  Color(0.10F, 0.90F, 0.95F, 1.0F)},
              std::pair{
                  "obstacle: blocked",
                  Color(1.0F, 0.10F, 0.08F, 1.0F)},
              std::pair{
                  "unknown: blocked",
                  Color(0.45F, 0.45F, 0.48F, 1.0F)},
          };
      for (std::size_t index = 0U; index < semantics.size(); ++index) {
        Marker legend = BaseMarker(
            "terrain_semantics", static_cast<std::int32_t>(index),
            Marker::TEXT_VIEW_FACING, "map", context.stamp,
            semantics[index].second);
        legend.pose.position = Point(reference.preview.poses_map.front().position_m);
        legend.pose.position.z += 0.55 + 0.24 * static_cast<double>(index);
        legend.scale.z = 0.18;
        legend.text = semantics[index].first;
        markers.push_back(std::move(legend));
      }
    }
  }

  if (const auto* point = std::get_if<lunar::planning::PointGoal>(
          &input.goal_map.target)) {
    Marker goal = BaseMarker(
        "planning_goal", 0, Marker::SPHERE, "map", context.stamp,
        Color(1.0F, 0.25F, 0.2F, 1.0F));
    goal.pose.position = Point(point->position_m);
    goal.scale.x = 0.45;
    goal.scale.y = 0.45;
    goal.scale.z = 0.45;
    markers.push_back(std::move(goal));
  } else if (const auto* region =
                 std::get_if<lunar::planning::PlanarRegionGoal>(
                     &input.goal_map.target)) {
    markers.push_back(LineMarker(
        "planning_goal", 0, "map", context.stamp, 0.08,
        Color(1.0F, 0.25F, 0.2F, 1.0F), region->boundary_m, true));
  }

  const auto* trajectory =
      std::get_if<lunar::planning::TrajectoryReference>(&reference.data);
  if (trajectory == nullptr || trajectory->points.empty()) {
    return;
  }
  std::vector<lunar::planning::Vec3> local;
  local.reserve(trajectory->points.size());
  for (const auto& point : trajectory->points) {
    local.push_back(point.pose.position_m);
  }
  markers.push_back(LineMarker(
      "certified_local_execution", 0, "odom", context.stamp, 0.12,
      Color(0.10F, 1.0F, 0.25F, 1.0F), local));
  if (local.size() >= 2U && local.front() != local.back()) {
    Marker direction = BaseMarker(
        "planning_direction", 0, Marker::ARROW, "odom", context.stamp,
        Color(1.0F, 0.9F, 0.1F, 1.0F));
    direction.points = {Point(local.front()), Point(local.back())};
    direction.scale.x = 0.08;
    direction.scale.y = 0.16;
    direction.scale.z = 0.24;
    markers.push_back(std::move(direction));
  }
}

void AddWheeledMarkers(
    std::vector<Marker>& markers,
    const lunar::planning::PlannerInput& input,
    const RouteMarkerContext& context) {
  const auto* capability =
      std::get_if<lunar::planning::WheeledCapability>(&input.capability);
  const auto* state =
      std::get_if<lunar::planning::WheeledState>(&input.current_state);
  if (capability == nullptr || state == nullptr) {
    return;
  }

  Marker body = BaseMarker(
      "wheeled_platform_body", 0, Marker::CUBE, "odom", context.stamp,
      Color(0.20F, 0.55F, 0.95F, 0.75F));
  body.pose = Pose(OffsetPose(
      state->pose, {0.0, 0.0, 0.5 * capability->body_extent_m.z}));
  body.scale.x = capability->body_extent_m.x;
  body.scale.y = capability->body_extent_m.y;
  body.scale.z = capability->body_extent_m.z;
  markers.push_back(std::move(body));

  constexpr double kSqrtHalf = 0.7071067811865475244;
  const lunar::planning::Quaternion wheel_rotation{
      .w = kSqrtHalf,
      .x = kSqrtHalf,
  };
  const std::array<double, 2> longitudinal{
      -0.5 * capability->wheelbase_m,
      0.5 * capability->wheelbase_m,
  };
  const std::array<double, 2> lateral{
      -0.5 * capability->track_width_m,
      0.5 * capability->track_width_m,
  };
  std::int32_t wheel_id = 0;
  for (const double x : longitudinal) {
    for (const double y : lateral) {
      lunar::planning::Pose3 wheel_pose = OffsetPose(
          state->pose, {x, y, 0.5 * capability->wheel_diameter_m});
      wheel_pose.orientation = Multiply(
          state->pose.orientation, wheel_rotation);
      Marker wheel = BaseMarker(
          "wheeled_platform_wheel", wheel_id++, Marker::CYLINDER,
          "odom", context.stamp, Color(0.08F, 0.08F, 0.10F, 1.0F));
      wheel.pose = Pose(wheel_pose);
      wheel.scale.x = capability->wheel_diameter_m;
      wheel.scale.y = capability->wheel_diameter_m;
      wheel.scale.z = capability->wheel_width_m;
      markers.push_back(std::move(wheel));
    }
  }

  std::vector<lunar::planning::Vec3> footprint;
  footprint.reserve(capability->footprint_xy_m.size());
  double minimum_x = std::numeric_limits<double>::infinity();
  double maximum_x = -std::numeric_limits<double>::infinity();
  double minimum_y = std::numeric_limits<double>::infinity();
  double maximum_y = -std::numeric_limits<double>::infinity();
  for (const auto point : capability->footprint_xy_m) {
    minimum_x = std::min(minimum_x, point.x);
    maximum_x = std::max(maximum_x, point.x);
    minimum_y = std::min(minimum_y, point.y);
    maximum_y = std::max(maximum_y, point.y);
    footprint.push_back(
        OffsetPose(state->pose, {point.x, point.y, 0.02}).position_m);
  }
  if (footprint.size() >= 3U) {
    markers.push_back(LineMarker(
        "wheeled_raw_footprint", 0, "odom", context.stamp, 0.035,
        Color(0.95F, 0.95F, 0.95F, 1.0F), footprint, true));
  }
  if (std::isfinite(minimum_x) && std::isfinite(maximum_x) &&
      std::isfinite(minimum_y) && std::isfinite(maximum_y) &&
      capability->minimum_clearance_m >= 0.0) {
    const double margin = capability->minimum_clearance_m;
    const std::array<lunar::planning::Vec3, 4> local_margin{
        lunar::planning::Vec3{minimum_x - margin, minimum_y - margin, 0.025},
        lunar::planning::Vec3{maximum_x + margin, minimum_y - margin, 0.025},
        lunar::planning::Vec3{maximum_x + margin, maximum_y + margin, 0.025},
        lunar::planning::Vec3{minimum_x - margin, maximum_y + margin, 0.025},
    };
    std::vector<lunar::planning::Vec3> clearance;
    clearance.reserve(local_margin.size());
    for (const auto point : local_margin) {
      clearance.push_back(OffsetPose(state->pose, point).position_m);
    }
    markers.push_back(LineMarker(
        "wheeled_clearance_margin", 0, "odom", context.stamp, 0.045,
        Color(1.0F, 0.45F, 0.05F, 0.95F), clearance, true));
  }
}

void AddLeggedMarkers(
    std::vector<Marker>& markers,
    const lunar::planning::PlannerInput& input,
    const RouteMarkerContext& context) {
  const auto* capability =
      std::get_if<lunar::planning::LeggedCapability>(&input.capability);
  const auto* state =
      std::get_if<lunar::planning::LeggedState>(&input.current_state);
  if (capability == nullptr || state == nullptr) {
    return;
  }
  Marker body = BaseMarker(
      "legged_platform_body", 0, Marker::CUBE, "odom", context.stamp,
      Color(0.60F, 0.25F, 0.95F, 0.78F));
  body.pose = Pose(state->body_pose);
  body.scale.x = capability->body_extent_m.x;
  body.scale.y = capability->body_extent_m.y;
  body.scale.z = capability->body_extent_m.z;
  markers.push_back(std::move(body));
}

[[nodiscard]] std::vector<lunar::planning::Vec3> BallisticArc(
    const lunar::planning::CertifiedHopPreview& hop) {
  constexpr lunar::planning::Vec3 kGravity{0.0, 0.0, -1.62};
  const double seconds = std::chrono::duration<double>(hop.flight_time).count();
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    return {};
  }
  const std::size_t samples = std::max<std::size_t>(
      2U, static_cast<std::size_t>(std::ceil(seconds / 0.1)) + 1U);
  std::vector<lunar::planning::Vec3> points;
  points.reserve(samples);
  for (std::size_t index = 0U; index < samples; ++index) {
    const double time = seconds * static_cast<double>(index) /
        static_cast<double>(samples - 1U);
    points.push_back({
        .x = hop.launch_pose_map.position_m.x +
            hop.launch_velocity_mps.x * time +
            0.5 * kGravity.x * time * time,
        .y = hop.launch_pose_map.position_m.y +
            hop.launch_velocity_mps.y * time +
            0.5 * kGravity.y * time * time,
        .z = hop.launch_pose_map.position_m.z +
            hop.launch_velocity_mps.z * time +
            0.5 * kGravity.z * time * time,
    });
  }
  return points;
}

void AddHopperMarkers(
    std::vector<Marker>& markers,
    const lunar::planning::PlannerOutput& output,
    const lunar::planning::PlannerInput& input,
    const RouteMarkerContext& context) {
  const auto* capability =
      std::get_if<lunar::planning::HopperCapability>(&input.capability);
  const auto* state =
      std::get_if<lunar::planning::HopperState>(&input.current_state);
  if (capability == nullptr || state == nullptr ||
      !output.reference.has_value() || output.certified_hops.size() != 1U) {
    return;
  }
  const auto* hops = std::get_if<lunar::planning::HopReference>(
      &output.reference->data);
  if (hops == nullptr || hops->segments.size() != 1U) {
    return;
  }
  const auto& hop = output.certified_hops.front();
  const auto& segment = hops->segments.front();

  Marker platform = BaseMarker(
      "hopper_platform", 0, Marker::SPHERE, "odom", context.stamp,
      Color(0.95F, 0.45F, 0.08F, 0.75F));
  platform.pose = Pose(state->pose);
  platform.scale.x = 2.0 * capability->flight_collision_radius_m;
  platform.scale.y = platform.scale.x;
  platform.scale.z = platform.scale.x;
  markers.push_back(std::move(platform));

  Marker target = BaseMarker(
      "hopper_exact_target", 0, Marker::SPHERE, "map", context.stamp,
      Color(1.0F, 0.1F, 0.2F, 1.0F));
  target.pose.position = Point(hop.landing_pose_map.position_m);
  target.scale.x = 0.24;
  target.scale.y = 0.24;
  target.scale.z = 0.24;
  markers.push_back(std::move(target));

  Marker support = BaseMarker(
      "hopper_landing_support_disk", 0, Marker::CYLINDER, "map",
      context.stamp, Color(0.2F, 0.95F, 0.35F, 0.22F));
  support.pose.position = Point(hop.landing_pose_map.position_m);
  support.pose.position.z += 0.015;
  const double support_diameter = 2.0 *
      (capability->landing_support_radius_m +
       capability->landing_lateral_margin_m);
  support.scale.x = support_diameter;
  support.scale.y = support_diameter;
  support.scale.z = 0.03;
  markers.push_back(std::move(support));

  if (hop.landing_region_map.size() >= 3U) {
    lunar::planning::Vec3 center{};
    for (const auto point : hop.landing_region_map) {
      center.x += point.x;
      center.y += point.y;
      center.z += point.z;
    }
    const double count = static_cast<double>(hop.landing_region_map.size());
    center.x /= count;
    center.y /= count;
    center.z /= count;
    Marker filled = BaseMarker(
        "hopper_landing_region_filled", 0, Marker::TRIANGLE_LIST, "map",
        context.stamp, Color(0.15F, 1.0F, 0.30F, 0.55F));
    for (std::size_t index = 0U; index < hop.landing_region_map.size();
         ++index) {
      filled.points.push_back(Point(center));
      filled.points.push_back(Point(hop.landing_region_map[index]));
      filled.points.push_back(Point(
          hop.landing_region_map[(index + 1U) %
                                 hop.landing_region_map.size()]));
    }
    markers.push_back(std::move(filled));
    markers.push_back(LineMarker(
        "hopper_landing_region_boundary", 0, "map", context.stamp, 0.055,
        Color(0.10F, 1.0F, 0.25F, 1.0F), hop.landing_region_map, true));
  }

  Marker nominal = BaseMarker(
      "hopper_nominal_landing_point", 0, Marker::SPHERE, "map",
      context.stamp, Color(1.0F, 1.0F, 0.1F, 1.0F));
  nominal.pose.position = Point(hop.landing_pose_map.position_m);
  nominal.scale.x = 0.16;
  nominal.scale.y = 0.16;
  nominal.scale.z = 0.16;
  markers.push_back(std::move(nominal));

  const auto arc = BallisticArc(hop);
  if (!arc.empty()) {
    markers.push_back(LineMarker(
        "hopper_nominal_arc", 0, "map", context.stamp, 0.09,
        Color(1.0F, 0.82F, 0.08F, 1.0F), arc));
    markers.push_back(LineMarker(
        "hopper_flight_tube", 0, "map", context.stamp,
        2.0 * hop.flight_tube_radius_m,
        Color(1.0F, 0.45F, 0.05F, 0.18F), arc));
  }

  Marker evidence = BaseMarker(
      "hopper_certification_evidence", 0, Marker::TEXT_VIEW_FACING, "map",
      context.stamp, Color(0.95F, 0.98F, 1.0F, 1.0F));
  evidence.pose.position = Point(hop.landing_pose_map.position_m);
  evidence.pose.position.z += 0.8;
  evidence.scale.z = 0.28;
  std::ostringstream text;
  text << std::fixed << std::setprecision(6)
       << "fuel=" << segment.certified_fuel_required_kg
       << " kg ideal=" << segment.ideal_fuel_required_kg
       << " kg remaining=" << segment.expected_remaining_usable_fuel_kg
       << " kg dv=" << segment.required_delta_v_mps << "/"
       << segment.available_delta_v_mps << " m/s version="
       << segment.capability_version << " maps="
       << segment.global_map_generation << "/"
       << segment.local_map_generation << " route=" << context.route_id
       << " plan=" << output.reference->plan_id << " segment="
       << segment.segment_id;
  evidence.text = text.str();
  markers.push_back(std::move(evidence));
}

}  // namespace

visualization_msgs::msg::MarkerArray RouteMarkerPublisher::Replace(
    const lunar::planning::PlannerOutput& output,
    const lunar::planning::PlannerInput& input,
    const RouteMarkerContext& context) {
  visualization_msgs::msg::MarkerArray result = DeleteOwned(context.stamp);
  std::vector<Marker> additions;
  if (output.reference.has_value() &&
      output.reference->platform_type ==
          lunar::planning::CapabilityPlatform(input.capability)) {
    AddCommonReferenceMarkers(additions, output, input, context);
    switch (output.reference->platform_type) {
      case lunar::planning::PlatformType::kWheeled:
        AddWheeledMarkers(additions, input, context);
        break;
      case lunar::planning::PlatformType::kLegged:
        AddLeggedMarkers(additions, input, context);
        break;
      case lunar::planning::PlatformType::kHopper:
        AddHopperMarkers(additions, output, input, context);
        break;
    }
  }
  result.markers.reserve(result.markers.size() + additions.size());
  for (Marker& marker : additions) {
    owned_markers_.emplace(marker.ns, marker.id, marker.header.frame_id);
    result.markers.push_back(std::move(marker));
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
