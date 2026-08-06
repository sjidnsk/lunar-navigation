#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_navigation_msgs/msg/localization_status.hpp>
#include <lunar_planner_core/types/platform_capability.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace lunar::planning::ros::test {

inline constexpr std::size_t kMapWidth = 3U;
inline constexpr std::size_t kMapHeight = 2U;

inline builtin_interfaces::msg::Time Stamp(const std::int64_t nanoseconds) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
  stamp.nanosec = static_cast<std::uint32_t>(nanoseconds % 1'000'000'000LL);
  return stamp;
}

inline const std::array<std::string_view, 10>& RequiredLayerNames() {
  static constexpr std::array<std::string_view, 10> names{
      "elevation",
      "valid_mask",
      "obstacle",
      "obstacle_height",
      "observation_age_s",
      "observation_quality",
      "elevation_variance",
      "obstacle_variance",
      "observation_count",
      "forbidden",
  };
  return names;
}

inline std_msgs::msg::Float32MultiArray EncodeLayer(
    const std::vector<float>& row_major_values,
    const std::size_t outer_start_index,
    const std::size_t inner_start_index) {
  std_msgs::msg::Float32MultiArray layer;
  layer.layout.dim.resize(2U);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = kMapHeight;
  layer.layout.dim[0].stride = kMapWidth * kMapHeight;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = kMapWidth;
  layer.layout.dim[1].stride = kMapWidth;
  layer.data.assign(kMapWidth * kMapHeight, 0.0F);
  for (std::size_t y = 0U; y < kMapHeight; ++y) {
    for (std::size_t x = 0U; x < kMapWidth; ++x) {
      const std::size_t unwrapped_row = kMapWidth - 1U - x;
      const std::size_t unwrapped_column = kMapHeight - 1U - y;
      const std::size_t physical_row =
          (unwrapped_row + outer_start_index) % kMapWidth;
      const std::size_t physical_column =
          (unwrapped_column + inner_start_index) % kMapHeight;
      layer.data[physical_column * kMapWidth + physical_row] =
          row_major_values[y * kMapWidth + x];
    }
  }
  return layer;
}

inline grid_map_msgs::msg::GridMap MakeGridMap(
    std::string frame_id = "odom",
    const std::size_t outer_start_index = 0U,
    const std::size_t inner_start_index = 0U) {
  grid_map_msgs::msg::GridMap message;
  message.header.stamp = Stamp(10'000'000'000LL);
  message.header.frame_id = std::move(frame_id);
  message.info.resolution = 1.0;
  message.info.length_x = static_cast<double>(kMapWidth);
  message.info.length_y = static_cast<double>(kMapHeight);
  message.info.pose.position.x = 1.5;
  message.info.pose.position.y = 1.0;
  message.info.pose.orientation.w = 1.0;
  message.outer_start_index = static_cast<std::uint16_t>(outer_start_index);
  message.inner_start_index = static_cast<std::uint16_t>(inner_start_index);
  message.basic_layers = {"elevation", "valid_mask"};

  const std::vector<float> elevation{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
  const std::vector<float> ones(kMapWidth * kMapHeight, 1.0F);
  const std::vector<float> zeros(kMapWidth * kMapHeight, 0.0F);
  const std::vector<float> quality(kMapWidth * kMapHeight, 0.9F);
  const std::vector<float> variance(kMapWidth * kMapHeight, 0.01F);
  const std::vector<float> counts(kMapWidth * kMapHeight, 2.0F);

  for (const std::string_view name : RequiredLayerNames()) {
    message.layers.emplace_back(name);
    const std::vector<float>* values = &zeros;
    if (name == "elevation") {
      values = &elevation;
    } else if (name == "valid_mask") {
      values = &ones;
    } else if (name == "observation_quality") {
      values = &quality;
    } else if (name == "elevation_variance" || name == "obstacle_variance") {
      values = &variance;
    } else if (name == "observation_count") {
      values = &counts;
    }
    message.data.push_back(
        EncodeLayer(*values, outer_start_index, inner_start_index));
  }
  return message;
}

inline std::size_t LayerIndex(
    const grid_map_msgs::msg::GridMap& message,
    const std::string_view layer_name) {
  for (std::size_t index = 0U; index < message.layers.size(); ++index) {
    if (message.layers[index] == layer_name) {
      return index;
    }
  }
  return message.layers.size();
}

inline nav_msgs::msg::Odometry MakeOdometry(
    const std::int64_t nanoseconds = 10'000'000'000LL) {
  nav_msgs::msg::Odometry message;
  message.header.stamp = Stamp(nanoseconds);
  message.header.frame_id = "odom";
  message.child_frame_id = "base_link";
  message.pose.pose.position.x = 0.5;
  message.pose.pose.position.y = 0.5;
  message.pose.pose.position.z = 0.2;
  message.pose.pose.orientation.w = 1.0;
  message.twist.twist.linear.x = 0.1;
  for (std::size_t index = 0U; index < 6U; ++index) {
    message.pose.covariance[index * 6U + index] = 0.1;
    message.twist.covariance[index * 6U + index] = 0.1;
  }
  return message;
}

inline lunar_navigation_msgs::msg::LocalizationStatus MakeLocalizationStatus(
    const std::uint8_t status =
        lunar_navigation_msgs::msg::LocalizationStatus::VALID,
    const std::int64_t nanoseconds = 10'000'000'000LL) {
  lunar_navigation_msgs::msg::LocalizationStatus message;
  message.header.stamp = Stamp(nanoseconds);
  message.header.frame_id = "odom";
  message.status = status;
  return message;
}

inline geometry_msgs::msg::TransformStamped MakeTransform(
    std::string parent,
    std::string child,
    const std::int64_t nanoseconds,
    const double translation_x = 0.0) {
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = Stamp(nanoseconds);
  transform.header.frame_id = std::move(parent);
  transform.child_frame_id = std::move(child);
  transform.transform.translation.x = translation_x;
  transform.transform.rotation.w = 1.0;
  return transform;
}

inline tf2_msgs::msg::TFMessage MakeTransforms(
    const std::int64_t nanoseconds = 10'000'000'000LL) {
  tf2_msgs::msg::TFMessage message;
  message.transforms = {
      MakeTransform("map", "odom", nanoseconds, 10.0),
      MakeTransform("odom", "base_link", nanoseconds, 0.5),
  };
  return message;
}

inline lunar::planning::WheeledCapability MakeWheeledCapability() {
  return lunar::planning::WheeledCapability{
      .footprint_xy_m =
          {{-0.2, -0.2}, {0.2, -0.2}, {0.2, 0.2}, {-0.2, 0.2}},
      .body_extent_m = {1.182, 0.818, 1.29996},
      .wheel_diameter_m = 0.319,
      .wheel_width_m = 0.148,
      .wheelbase_m = 0.8175,
      .track_width_m = 0.67,
      .minimum_underbody_clearance_m = 0.21,
      .maximum_local_obstacle_relief_m = 0.2,
      .allow_unsupported_gap = false,
      .minimum_body_z_m = -0.1,
      .maximum_body_z_m = 0.5,
      .maximum_forward_speed_mps = 1.0,
      .maximum_reverse_speed_mps = 0.5,
      .maximum_spin_rate_radps = 1.0,
      .maximum_acceleration_mps2 = 1.0,
      .maximum_braking_deceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .maximum_lateral_acceleration_mps2 = 1.0,
      .maximum_curvature_per_m = 1.0,
      .maximum_slope_rad = 0.4,
      .minimum_clearance_m = 0.0,
      .motion_primitives =
          {lunar::planning::WheelMotionPrimitive{
              .primitive_id = "forward",
              .kind = lunar::planning::WheelPrimitiveKind::kForward,
              .relative_end_pose =
                  lunar::planning::Pose3{
                      .position_m = {1.0, 0.0, 0.0},
                      .orientation = {},
                  },
              .nominal_duration = std::chrono::seconds{1},
          }},
  };
}

}  // namespace lunar::planning::ros::test
