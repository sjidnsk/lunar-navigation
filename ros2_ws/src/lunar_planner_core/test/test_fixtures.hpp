#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::test {

inline GridLayer MakeFloatLayer(const std::size_t count, const float value) {
  return GridLayer{.values = std::vector<float>(count, value)};
}

inline GridLayer MakeByteLayer(
    const std::size_t count, const std::uint8_t value) {
  return GridLayer{.values = std::vector<std::uint8_t>(count, value)};
}

inline GridLayer MakeCountLayer(
    const std::size_t count, const std::uint32_t value) {
  return GridLayer{.values = std::vector<std::uint32_t>(count, value)};
}

inline GridMap MakeFlatMap(std::string frame_id) {
  constexpr std::size_t kWidth = 8;
  constexpr std::size_t kHeight = 6;
  constexpr std::size_t kCellCount = kWidth * kHeight;
  std::map<std::string, GridLayer, std::less<>> layers;
  layers.emplace("elevation", MakeFloatLayer(kCellCount, 0.0F));
  layers.emplace("valid_mask", MakeByteLayer(kCellCount, 1U));
  layers.emplace("obstacle", MakeByteLayer(kCellCount, 0U));
  layers.emplace("obstacle_height", MakeFloatLayer(kCellCount, 0.0F));
  layers.emplace("observation_age_s", MakeFloatLayer(kCellCount, 0.0F));
  layers.emplace("observation_quality", MakeFloatLayer(kCellCount, 1.0F));
  layers.emplace("elevation_variance", MakeFloatLayer(kCellCount, 0.0F));
  layers.emplace("obstacle_variance", MakeFloatLayer(kCellCount, 0.0F));
  layers.emplace("observation_count", MakeCountLayer(kCellCount, 1U));
  layers.emplace("forbidden", MakeByteLayer(kCellCount, 0U));
  return GridMap{
      .frame_id = std::move(frame_id),
      .stamp = TimePoint{.nanoseconds_since_epoch = 1'000'000'000},
      .width = kWidth,
      .height = kHeight,
      .resolution_m = 0.5,
      .origin_m = Vec3{},
      .layers = std::move(layers),
  };
}

inline PlannerInput MakeValidWheelInput() {
  PlannerInput input{
      .request_id = "public-api-wheel",
      .state_time = TimePoint{.nanoseconds_since_epoch = 1'000'000'000},
      .current_state =
          WheeledState{
              .pose = Pose3{.position_m = {1.0, 1.0, 0.0}},
          },
      .goal =
          GoalRegion{
              .goal_id = "goal",
              .target = PointGoal{.position_m = {2.0, 1.0, 0.0},
                                  .tolerance_m = 0.2},
          },
      .world =
          WorldSnapshot{
              .global_map = MakeFlatMap("map"),
              .local_map = MakeFlatMap("odom"),
              .map_from_odom =
                  RigidTransform{
                      .parent_frame = "map",
                      .child_frame = "odom",
                      .stamp = TimePoint{.nanoseconds_since_epoch = 1'000'000'000},
                  },
          },
      .capability =
          WheeledCapability{
              .footprint_xy_m =
                  {{-0.2, -0.2}, {0.2, -0.2}, {0.2, 0.2}, {-0.2, 0.2}},
              .maximum_forward_speed_mps = 1.0,
              .maximum_reverse_speed_mps = 0.5,
              .maximum_spin_rate_radps = 1.0,
              .maximum_acceleration_mps2 = 0.5,
              .maximum_braking_deceleration_mps2 = 0.5,
              .maximum_yaw_acceleration_radps2 = 1.0,
              .maximum_lateral_acceleration_mps2 = 0.5,
              .maximum_curvature_per_m = 1.0,
              .maximum_slope_rad = 0.4,
              .minimum_clearance_m = 0.1,
          },
      .config = PlannerConfig{},
  };
  return input;
}

}  // namespace lunar::planning::test
