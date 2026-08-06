#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <chrono>
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

inline GridMap MakeFlatMap(
    std::string frame_id, const std::size_t width = 12U,
    const std::size_t height = 8U, const double resolution_m = 1.0) {
  const std::size_t cell_count = width * height;
  std::map<std::string, GridLayer, std::less<>> layers;
  layers.emplace("elevation", MakeFloatLayer(cell_count, 0.0F));
  layers.emplace("valid_mask", MakeByteLayer(cell_count, 1U));
  layers.emplace("obstacle", MakeByteLayer(cell_count, 0U));
  layers.emplace("obstacle_height", MakeFloatLayer(cell_count, 0.0F));
  layers.emplace("observation_age_s", MakeFloatLayer(cell_count, 0.0F));
  layers.emplace("observation_quality", MakeFloatLayer(cell_count, 1.0F));
  layers.emplace("elevation_variance", MakeFloatLayer(cell_count, 0.0F));
  layers.emplace("obstacle_variance", MakeFloatLayer(cell_count, 0.0F));
  layers.emplace("observation_count", MakeCountLayer(cell_count, 1U));
  layers.emplace("forbidden", MakeByteLayer(cell_count, 0U));
  return GridMap{
      .frame_id = std::move(frame_id),
      .stamp = TimePoint{.nanoseconds_since_epoch = 1'000'000'000},
      .width = width,
      .height = height,
      .resolution_m = resolution_m,
      .origin_m = Vec3{},
      .layers = std::move(layers),
  };
}

inline Quaternion YawQuaternion(const double yaw_rad) {
  return Quaternion{
      .w = std::cos(yaw_rad / 2.0),
      .z = std::sin(yaw_rad / 2.0),
  };
}

inline PlannerInput MakeValidWheelInput() {
  PlannerInput input{
      .request_id = "public-api-wheel",
      .mission_id = "test-mission",
      .mission_revision = 1U,
      .platform_id = "test-wheel",
      .capability_version = "test-wheel-capability-v1",
      .global_map_generation = 1U,
      .local_map_generation = 1U,
      .map_from_odom_generation = 1U,
      .state_time = TimePoint{.nanoseconds_since_epoch = 1'000'000'000},
      .current_state =
          WheeledState{
              .pose = Pose3{.position_m = {2.5, 3.5, 0.0}},
          },
      .goal_map =
          GoalRegion{
              .goal_id = "goal",
              .target = PointGoal{.position_m = {4.5, 3.5, 0.0},
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
                  {{-0.591, -0.409},
                   {0.591, -0.409},
                   {0.591, 0.409},
                   {-0.591, 0.409}},
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
              .maximum_reverse_speed_mps = 0.8,
              .maximum_spin_rate_radps = 1.0,
              .maximum_acceleration_mps2 = 1.0,
              .maximum_braking_deceleration_mps2 = 1.0,
              .maximum_yaw_acceleration_radps2 = 1.0,
              .maximum_lateral_acceleration_mps2 = 1.0,
              .maximum_curvature_per_m = 1.0,
              .maximum_slope_rad = 0.5,
              .minimum_clearance_m = 0.0,
              .motion_primitives =
                  {
                      WheelMotionPrimitive{
                          .primitive_id = "forward",
                          .kind = WheelPrimitiveKind::kForward,
                          .relative_end_pose = Pose3{.position_m = {1.0, 0.0, 0.0}},
                      },
                      WheelMotionPrimitive{
                          .primitive_id = "forward-arc",
                          .kind = WheelPrimitiveKind::kForwardArc,
                          .relative_end_pose =
                              Pose3{
                                  .position_m = {1.0, 0.0, 0.0},
                                  .orientation =
                                      YawQuaternion(1.5707963267948966),
                              },
                      },
                      WheelMotionPrimitive{
                          .primitive_id = "reverse",
                          .kind = WheelPrimitiveKind::kReverse,
                          .relative_end_pose = Pose3{.position_m = {-1.0, 0.0, 0.0}},
                      },
                      WheelMotionPrimitive{
                          .primitive_id = "reverse-arc",
                          .kind = WheelPrimitiveKind::kReverseArc,
                          .relative_end_pose =
                              Pose3{
                                  .position_m = {-1.0, 0.0, 0.0},
                                  .orientation =
                                      YawQuaternion(-1.5707963267948966),
                              },
                      },
                      WheelMotionPrimitive{
                          .primitive_id = "spin-left",
                          .kind = WheelPrimitiveKind::kSpinCounterclockwise,
                          .relative_end_pose =
                              Pose3{.orientation = YawQuaternion(1.5707963267948966)},
                      },
                      WheelMotionPrimitive{
                          .primitive_id = "spin-right",
                          .kind = WheelPrimitiveKind::kSpinClockwise,
                          .relative_end_pose =
                              Pose3{.orientation = YawQuaternion(-1.5707963267948966)},
                      },
                      WheelMotionPrimitive{
                          .primitive_id = "stop-switch",
                          .kind = WheelPrimitiveKind::kStopAndSwitch,
                      },
                  },
          },
      .config = PlannerConfig{},
      .position_uncertainty_m = 0.05,
      .velocity_uncertainty_mps = 0.02,
  };
  input.config.wheel.xy_resolution_m = 1.0;
  input.config.legged.xy_resolution_m = 1.0;
  input.config.global_map.base_resolution_m = 1.0;
  return input;
}

inline PlannerInput MakeValidLeggedInput() {
  PlannerInput input = MakeValidWheelInput();
  input.request_id = "public-api-legged";
  input.platform_id = "test-legged";
  input.capability_version = "test-legged-capability-v1";
  input.current_state = LeggedState{
      .body_pose = Pose3{.position_m = {2.5, 3.5, 0.5}},
  };
  input.capability = LeggedCapability{
      .body_extent_m = {0.68, 0.33, 0.35},
      .platform_mass_kg = 15.89,
      .maximum_payload_kg = 10.0,
      .maximum_slope_rad = 0.5235987755982988,
      .maximum_step_height_m = 0.5,
      .maximum_gap_width_m = 0.3,
      .minimum_body_clearance_m = 0.3,
      .step_vertical_rate_mps = 0.1,
      .body_height_m = {0.4, 0.6},
      .forward_speed_mps = {-1.5, 1.5},
      .lateral_speed_mps = {-0.8, 0.8},
      .yaw_rate_radps = {-1.0, 1.0},
      .maximum_linear_acceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .motion_primitives =
          {
              LeggedBodyPrimitive{
                  .primitive_id = "forward",
                  .kind = LeggedPrimitiveKind::kForward,
                  .body_frame_displacement_m = {1.0, 0.0, 0.0},
              },
              LeggedBodyPrimitive{
                  .primitive_id = "backward",
                  .kind = LeggedPrimitiveKind::kBackward,
                  .body_frame_displacement_m = {-1.0, 0.0, 0.0},
              },
              LeggedBodyPrimitive{
                  .primitive_id = "left",
                  .kind = LeggedPrimitiveKind::kLateralLeft,
                  .body_frame_displacement_m = {0.0, 1.0, 0.0},
              },
              LeggedBodyPrimitive{
                  .primitive_id = "right",
                  .kind = LeggedPrimitiveKind::kLateralRight,
                  .body_frame_displacement_m = {0.0, -1.0, 0.0},
              },
              LeggedBodyPrimitive{
                  .primitive_id = "spin",
                  .kind = LeggedPrimitiveKind::kSpin,
                  .yaw_change_rad = 1.5707963267948966,
              },
          },
  };
  return input;
}

inline PlannerInput MakeValidHopperInput() {
  PlannerInput input = MakeValidWheelInput();
  input.request_id = "public-api-hopper";
  input.platform_id = "test-hopper";
  input.capability_version = "test-hopper-capability-v1";
  input.current_state = HopperState{
      .pose = Pose3{.position_m = {3.0, 3.0, 0.5}},
  };
  input.goal_map.target = PointGoal{
      .position_m = {4.0, 3.0, 0.0},
      .tolerance_m = 0.0,
  };
  input.goal_map.yaw_rad = std::nullopt;
  input.goal_map.yaw_tolerance_rad = 0.0;
  input.world.global_map = MakeFlatMap("map", 16U, 12U, 0.5);
  input.world.local_map = MakeFlatMap("odom", 16U, 12U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  input.hopper_propellant = HopperPropellantState{
      .stamp = input.state_time,
      .platform_id = input.platform_id,
      .capability_version = input.capability_version,
      .total_mass_kg = 20.0,
      .remaining_usable_fuel_mass_kg = 0.20,
  };
  input.capability = HopperCapability{
      .specific_impulse_s = 301.0,
      .landing_support_radius_m = 0.45,
      .flight_collision_radius_m = 0.55,
      .maximum_landing_plane_residual_m = 0.05,
      .landing_lateral_margin_m = 0.2,
      .flight_map_margin_m = 0.2,
      .reachability_delta_v_margin_ratio = 0.1,
      .standard_gravity_mps2 = 9.80665,
      .maximum_landing_slope_rad = 0.174533,
  };
  return input;
}

}  // namespace lunar::planning::test
