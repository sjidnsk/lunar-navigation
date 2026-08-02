#include "migration/legacy_v3_adapter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/api/planner_v3.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/hopper/hopper_config.hpp"

namespace lunar::planning {
namespace {

namespace legacy = ::lunar::planning::v3;
using namespace std::chrono_literals;

class ConversionError final : public std::runtime_error {
 public:
  explicit ConversionError(std::string reason_code)
      : std::runtime_error(reason_code), reason_code_(std::move(reason_code)) {}

  [[nodiscard]] const std::string& reason_code() const noexcept {
    return reason_code_;
  }

 private:
  std::string reason_code_;
};

[[nodiscard]] legacy::ContentRef Ref(std::string id, const char digit) {
  return legacy::ContentRef{
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digit),
  };
}

[[nodiscard]] legacy::AxisAlignedBox3 ZeroBox() {
  return legacy::AxisAlignedBox3{};
}

[[nodiscard]] legacy::WheeledOrLeggedErrorBounds ZeroGroundError() {
  return legacy::WheeledOrLeggedErrorBounds{
      .position_bound_m = ZeroBox(),
      .yaw_bound_rad = {},
      .linear_velocity_bound_mps = ZeroBox(),
      .yaw_rate_bound_radps = {},
  };
}

[[nodiscard]] legacy::HopperErrorBounds ZeroHopperError() {
  return legacy::HopperErrorBounds{
      .position_bound_m = ZeroBox(),
      .orientation_bound = {},
      .linear_velocity_bound_mps = ZeroBox(),
      .angular_velocity_bound_radps = ZeroBox(),
  };
}

[[nodiscard]] legacy::Vec2 ToLegacy(const Vec2& value) {
  return {value.x, value.y};
}

[[nodiscard]] legacy::Vec3 ToLegacy(const Vec3& value) {
  return {value.x, value.y, value.z};
}

[[nodiscard]] legacy::Quaternion ToLegacy(const Quaternion& value) {
  return {value.w, value.x, value.y, value.z};
}

[[nodiscard]] Vec3 ToPublic(const legacy::Vec3& value) {
  return {value.x, value.y, value.z};
}

[[nodiscard]] Quaternion ToPublic(const legacy::Quaternion& value) {
  return {value.w, value.x, value.y, value.z};
}

[[nodiscard]] double YawFromQuaternion(const Quaternion& value) noexcept {
  return std::atan2(
      2.0 * (value.w * value.z + value.x * value.y),
      1.0 - 2.0 * (value.y * value.y + value.z * value.z));
}

[[nodiscard]] Quaternion QuaternionFromYaw(const double yaw_rad) noexcept {
  return Quaternion{
      .w = std::cos(yaw_rad / 2.0),
      .z = std::sin(yaw_rad / 2.0),
  };
}

[[nodiscard]] std::string UpperToken(const std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char character : value) {
    if (character >= 'a' && character <= 'z') {
      result.push_back(static_cast<char>(character - 'a' + 'A'));
    } else if ((character >= 'A' && character <= 'Z') ||
               (character >= '0' && character <= '9')) {
      result.push_back(character);
    } else {
      result.push_back('_');
    }
  }
  return result;
}

template <typename Value>
[[nodiscard]] const std::vector<Value>& LayerValues(
    const GridMap& map, const std::string_view name) {
  const auto found = map.layers.find(name);
  if (found == map.layers.end()) {
    throw ConversionError{"MISSING_MAP_LAYER_" + UpperToken(name)};
  }
  const auto* values = std::get_if<std::vector<Value>>(&found->second.values);
  if (values == nullptr) {
    throw ConversionError{"INVALID_MAP_LAYER_TYPE_" + UpperToken(name)};
  }
  if (values->size() != map.CellCount()) {
    throw ConversionError{"INVALID_MAP_LAYER_SIZE_" + UpperToken(name)};
  }
  return *values;
}

[[nodiscard]] legacy::ClockStamp LegacyTime(const TimePoint time) {
  return legacy::ClockStamp{
      .clock_id = "planner_input",
      .tick = std::chrono::nanoseconds{time.nanoseconds_since_epoch},
  };
}

[[nodiscard]] legacy::MapSnapshotInput ConvertMap(
    const PlannerInput& input) {
  const GridMap& map = input.world.local_map;
  if (map.CellCount() == 0U || !std::isfinite(map.resolution_m) ||
      map.resolution_m <= 0.0 || map.frame_id.empty()) {
    throw ConversionError{"INVALID_MAP_GEOMETRY"};
  }
  const auto& elevation = LayerValues<float>(map, "elevation");
  const auto& valid = LayerValues<std::uint8_t>(map, "valid_mask");
  const auto& obstacle = LayerValues<std::uint8_t>(map, "obstacle");
  const auto& obstacle_height = LayerValues<float>(map, "obstacle_height");
  const auto& age = LayerValues<float>(map, "observation_age_s");
  const auto& quality = LayerValues<float>(map, "observation_quality");
  const auto& elevation_variance =
      LayerValues<float>(map, "elevation_variance");
  const auto& obstacle_variance =
      LayerValues<float>(map, "obstacle_variance");
  const auto& observation_count =
      LayerValues<std::uint32_t>(map, "observation_count");
  const auto& forbidden = LayerValues<std::uint8_t>(map, "forbidden");

  const std::size_t count = map.CellCount();
  legacy::MapSnapshotInput result{
      .snapshot_ref = Ref("adapter-map", '1'),
      .map_revision = 1U,
      .immutable_data_handle = "adapter-map-handle",
      .source_time = LegacyTime(map.stamp),
      .bounds =
          legacy::MapBounds{
              .minimum_m = ToLegacy(map.origin_m),
              .maximum_m =
                  legacy::Vec3{
                      map.origin_m.x +
                          static_cast<double>(map.width) * map.resolution_m,
                      map.origin_m.y +
                          static_cast<double>(map.height) * map.resolution_m,
                      map.origin_m.z + 100.0,
                  },
          },
      .geometry =
          legacy::GridGeometry{
              .width = map.width,
              .height = map.height,
              .resolution_m = map.resolution_m,
              .origin_m = legacy::Vec2{map.origin_m.x, map.origin_m.y},
              .frame_id = map.frame_id,
          },
      .layer_manifest =
          {
              {legacy::LayerKind::kKnownMask, Ref("adapter-known", '2')},
              {legacy::LayerKind::kElevation, Ref("adapter-elevation", '3')},
              {legacy::LayerKind::kTerrainNormal, Ref("adapter-normal", '4')},
              {legacy::LayerKind::kRoughness, Ref("adapter-roughness", '5')},
              {legacy::LayerKind::kHardObstacle, Ref("adapter-obstacle", '6')},
              {legacy::LayerKind::kConfidence, Ref("adapter-confidence", '7')},
          },
      .known_mask = std::vector<std::uint8_t>(count, 0U),
      .elevation_m = elevation,
      .normal_x = std::vector<float>(count, 0.0F),
      .normal_y = std::vector<float>(count, 0.0F),
      .normal_z = std::vector<float>(count, 1.0F),
      .roughness_m = std::vector<float>(count, 0.0F),
      .hard_obstacle_mask = std::vector<std::uint8_t>(count, 0U),
      .confidence = quality,
  };

  const MapSafetyConfig& safety = input.config.map_safety;
  const auto elevation_at = [&](const std::size_t x, const std::size_t y) {
    return static_cast<double>(elevation[y * map.width + x]);
  };
  for (std::size_t y = 0U; y < map.height; ++y) {
    for (std::size_t x = 0U; x < map.width; ++x) {
      const std::size_t index = y * map.width + x;
      const bool finite =
          std::isfinite(elevation[index]) &&
          std::isfinite(obstacle_height[index]) && std::isfinite(age[index]) &&
          std::isfinite(quality[index]) &&
          std::isfinite(elevation_variance[index]) &&
          std::isfinite(obstacle_variance[index]);
      const bool known =
          finite && valid[index] != 0U &&
          age[index] <= safety.maximum_observation_age_s &&
          quality[index] >= safety.minimum_observation_quality &&
          elevation_variance[index] <= safety.maximum_elevation_variance_m2 &&
          obstacle_variance[index] <= safety.maximum_obstacle_variance_m2 &&
          observation_count[index] >= safety.minimum_observation_count;
      result.known_mask[index] = known ? 1U : 0U;
      result.confidence[index] = known ? quality[index] : 0.0F;
      result.hard_obstacle_mask[index] =
          obstacle[index] != 0U || forbidden[index] != 0U ? 1U : 0U;
      if (!finite || elevation_variance[index] < 0.0F) {
        result.roughness_m[index] = std::numeric_limits<float>::quiet_NaN();
        continue;
      }
      result.roughness_m[index] = std::sqrt(elevation_variance[index]);

      const std::size_t left = x == 0U ? x : x - 1U;
      const std::size_t right = x + 1U == map.width ? x : x + 1U;
      const std::size_t down = y == 0U ? y : y - 1U;
      const std::size_t up = y + 1U == map.height ? y : y + 1U;
      const double dx = static_cast<double>(right - left) * map.resolution_m;
      const double dy = static_cast<double>(up - down) * map.resolution_m;
      const double dz_dx = dx > 0.0
                               ? (elevation_at(right, y) - elevation_at(left, y)) /
                                     dx
                               : 0.0;
      const double dz_dy = dy > 0.0
                               ? (elevation_at(x, up) - elevation_at(x, down)) /
                                     dy
                               : 0.0;
      const double norm = std::hypot(dz_dx, dz_dy, 1.0);
      result.normal_x[index] = static_cast<float>(-dz_dx / norm);
      result.normal_y[index] = static_cast<float>(-dz_dy / norm);
      result.normal_z[index] = static_cast<float>(1.0 / norm);
    }
  }
  return result;
}

[[nodiscard]] legacy::ConvexPolytope3 BoxPolytope(const Vec3& half_extent) {
  return legacy::ConvexPolytope3{
      .halfspaces =
          {
              {{1.0, 0.0, 0.0}, half_extent.x},
              {{-1.0, 0.0, 0.0}, half_extent.x},
              {{0.0, 1.0, 0.0}, half_extent.y},
              {{0.0, -1.0, 0.0}, half_extent.y},
              {{0.0, 0.0, 1.0}, half_extent.z},
              {{0.0, 0.0, -1.0}, half_extent.z},
          },
  };
}

[[nodiscard]] legacy::WheelMotionPrimitive::Kind ConvertWheelKind(
    const WheelPrimitiveKind kind) {
  using NewKind = WheelPrimitiveKind;
  using OldKind = legacy::WheelMotionPrimitive::Kind;
  switch (kind) {
    case NewKind::kForward:
      return OldKind::kDriveForwardLine;
    case NewKind::kReverse:
      return OldKind::kDriveReverseLine;
    case NewKind::kForwardArc:
      return OldKind::kDriveForwardArc;
    case NewKind::kReverseArc:
      return OldKind::kDriveReverseArc;
    case NewKind::kSpinClockwise:
      return OldKind::kSpinCw;
    case NewKind::kSpinCounterclockwise:
      return OldKind::kSpinCcw;
    case NewKind::kStopAndSwitch:
      return OldKind::kStopAndSwitch;
  }
  throw ConversionError{"INVALID_WHEEL_PRIMITIVE_KIND"};
}

[[nodiscard]] legacy::LeggedBodyPrimitive::Kind ConvertLeggedKind(
    const LeggedPrimitiveKind kind) {
  using NewKind = LeggedPrimitiveKind;
  using OldKind = legacy::LeggedBodyPrimitive::Kind;
  switch (kind) {
    case NewKind::kForward:
      return OldKind::kForward;
    case NewKind::kBackward:
      return OldKind::kBackward;
    case NewKind::kLateralLeft:
    case NewKind::kLateralRight:
      return OldKind::kLateral;
    case NewKind::kSpin:
      return OldKind::kSpin;
    case NewKind::kCoupled:
      return OldKind::kCoupled;
  }
  throw ConversionError{"INVALID_LEGGED_PRIMITIVE_KIND"};
}

[[nodiscard]] legacy::SafetyCapabilityProfile ConvertCapability(
    const PlannerInput& input) {
  const std::string& frame = input.world.local_map.frame_id;
  return std::visit(
      [&](const auto& capability) -> legacy::SafetyCapabilityProfile {
        using Capability = std::decay_t<decltype(capability)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability>) {
          if (capability.footprint_xy_m.size() < 3U ||
              capability.motion_primitives.empty()) {
            throw ConversionError{"INVALID_WHEELED_CAPABILITY"};
          }
          std::vector<legacy::WheelMotionPrimitive> primitives;
          primitives.reserve(capability.motion_primitives.size());
          for (std::size_t index = 0U;
               index < capability.motion_primitives.size(); ++index) {
            const WheelMotionPrimitive& primitive =
                capability.motion_primitives[index];
            primitives.push_back(legacy::WheelMotionPrimitive{
                .primitive_id = primitive.primitive_id,
                .kind = ConvertWheelKind(primitive.kind),
                .relative_end_pose =
                    legacy::PoseXyzYaw{
                        .position_m = ToLegacy(primitive.relative_end_pose.position_m),
                        .yaw_rad =
                            YawFromQuaternion(primitive.relative_end_pose.orientation),
                    },
                .nominal_duration =
                    legacy::DurationNanoseconds{primitive.nominal_duration},
                .swept_geometry_ref =
                    Ref("adapter-wheel-sweep-" + std::to_string(index), '8'),
            });
          }
          std::vector<legacy::Vec2> footprint;
          footprint.reserve(capability.footprint_xy_m.size());
          std::transform(
              capability.footprint_xy_m.begin(),
              capability.footprint_xy_m.end(),
              std::back_inserter(footprint),
              [](const Vec2& point) { return ToLegacy(point); });
          return legacy::SafetyCapabilityProfile{
              .content_ref = Ref("adapter-wheel-capability", 'a'),
              .content =
                  legacy::WheeledCapability{
                      .frame_id = frame,
                      .collision_envelope =
                          legacy::ExtrudedConvexFootprint{
                              .vertices_xy_m = std::move(footprint),
                              .minimum_z_m = capability.minimum_body_z_m,
                              .maximum_z_m = capability.maximum_body_z_m,
                          },
                      .motion_model_ref = Ref("adapter-wheel-motion", 'b'),
                      .analytic_cost_model_ref = Ref("adapter-wheel-cost", 'c'),
                      .hard_limits =
                          legacy::WheelHardLimits{
                              .maximum_forward_speed_mps =
                                  capability.maximum_forward_speed_mps,
                              .maximum_reverse_speed_mps =
                                  capability.maximum_reverse_speed_mps,
                              .maximum_spin_rate_radps =
                                  capability.maximum_spin_rate_radps,
                              .maximum_forward_acceleration_mps2 =
                                  capability.maximum_acceleration_mps2,
                              .maximum_braking_deceleration_mps2 =
                                  capability.maximum_braking_deceleration_mps2,
                              .maximum_yaw_acceleration_radps2 =
                                  capability.maximum_yaw_acceleration_radps2,
                              .maximum_lateral_acceleration_mps2 =
                                  capability.maximum_lateral_acceleration_mps2,
                              .maximum_drive_curvature_per_m =
                                  capability.maximum_curvature_per_m,
                              .maximum_slope_rad =
                                  std::min(
                                      capability.maximum_slope_rad,
                                      input.config.map_safety
                                          .project_maximum_slope_rad),
                              .minimum_clearance_m =
                                  capability.minimum_clearance_m,
                          },
                      .certified_state_error_bounds = ZeroGroundError(),
                      .motion_primitives = std::move(primitives),
                  },
          };
        } else if constexpr (std::is_same_v<Capability, LeggedCapability>) {
          if (capability.motion_primitives.empty()) {
            throw ConversionError{"INVALID_LEGGED_CAPABILITY"};
          }
          std::vector<legacy::LeggedBodyPrimitive> primitives;
          primitives.reserve(capability.motion_primitives.size());
          for (std::size_t index = 0U;
               index < capability.motion_primitives.size(); ++index) {
            const LeggedBodyPrimitive& primitive =
                capability.motion_primitives[index];
            primitives.push_back(legacy::LeggedBodyPrimitive{
                .primitive_id = primitive.primitive_id,
                .kind = ConvertLeggedKind(primitive.kind),
                .body_frame_displacement_m =
                    ToLegacy(primitive.body_frame_displacement_m),
                .yaw_change_rad = primitive.yaw_change_rad,
                .nominal_duration =
                    legacy::DurationNanoseconds{primitive.nominal_duration},
                .sampled_body_sweep_ref =
                    Ref("adapter-legged-sweep-" + std::to_string(index), 'd'),
            });
          }
          return legacy::SafetyCapabilityProfile{
              .content_ref = Ref("adapter-legged-capability", 'd'),
              .content =
                  legacy::LeggedCapability{
                      .frame_id = frame,
                      .reference_point_id = "body_reference",
                      .reference_point_definition =
                          legacy::LeggedCapability::ReferencePointDefinition::
                              kFixedNominalCom,
                      .collision_envelope =
                          legacy::BodyConvexPolytope{
                              .body_frame_halfspaces =
                                  BoxPolytope(capability.body_half_extent_m),
                          },
                      .motion_model_ref = Ref("adapter-legged-motion", 'e'),
                      .analytic_cost_model_ref = Ref("adapter-legged-cost", 'f'),
                      .terrain_thresholds =
                          legacy::LeggedTerrainThresholds{
                              .maximum_slope_rad =
                                  std::min(
                                      capability.maximum_slope_rad,
                                      input.config.map_safety
                                          .project_maximum_slope_rad),
                              .maximum_roughness_m =
                                  capability.maximum_roughness_m,
                              .maximum_step_height_m =
                                  capability.maximum_step_height_m,
                              .maximum_gap_width_m =
                                  capability.maximum_gap_width_m,
                              .minimum_confidence = capability.minimum_confidence,
                              .minimum_body_clearance_m =
                                  capability.minimum_body_clearance_m,
                              .minimum_body_height_m =
                                  capability.body_height_m.lower,
                              .maximum_body_height_m =
                                  capability.body_height_m.upper,
                          },
                      .body_velocity_limits =
                          legacy::LeggedBodyVelocityLimits{
                              .forward_mps =
                                  {capability.forward_speed_mps.lower,
                                   capability.forward_speed_mps.upper},
                              .lateral_mps =
                                  {capability.lateral_speed_mps.lower,
                                   capability.lateral_speed_mps.upper},
                              .vertical_mps =
                                  {capability.vertical_speed_mps.lower,
                                   capability.vertical_speed_mps.upper},
                              .yaw_rate_radps =
                                  {capability.yaw_rate_radps.lower,
                                   capability.yaw_rate_radps.upper},
                              .linear_acceleration_mps2 =
                                  capability.maximum_linear_acceleration_mps2,
                              .yaw_acceleration_radps2 =
                                  capability.maximum_yaw_acceleration_radps2,
                          },
                      .certified_state_error_bounds = ZeroGroundError(),
                      .motion_primitives = std::move(primitives),
                  },
          };
        } else {
          return legacy::SafetyCapabilityProfile{
              .content_ref = Ref("adapter-hopper-capability", '1'),
              .content =
                  legacy::HopperCapability{
                      .frame_id = frame,
                      .collision_envelope =
                          legacy::BodyConvexPolytope{
                              .body_frame_halfspaces =
                                  BoxPolytope(capability.body_half_extent_m),
                          },
                      .motion_model_ref = Ref("adapter-hopper-motion", '2'),
                      .analytic_cost_model_ref = Ref("adapter-hopper-cost", '3'),
                      .gravity_model_ref = Ref("adapter-hopper-gravity", '4'),
                      .landing_terrain_thresholds =
                          legacy::HopperLandingTerrainThresholds{
                              .maximum_slope_rad =
                                  std::min(
                                      capability.maximum_landing_slope_rad,
                                      input.config.map_safety
                                          .project_maximum_slope_rad),
                              .maximum_roughness_m =
                                  capability.maximum_landing_roughness_m,
                              .maximum_plane_residual_m =
                                  capability.maximum_plane_residual_m,
                              .minimum_overhead_clearance_m =
                                  capability.minimum_overhead_clearance_m,
                              .minimum_lateral_clearance_m =
                                  capability.minimum_lateral_clearance_m,
                              .minimum_landing_region_area_m2 =
                                  capability.minimum_landing_region_area_m2,
                          },
                      .launch_limits =
                          legacy::HopperLaunchLimits{
                              .maximum_launch_speed_mps =
                                  capability.maximum_launch_speed_mps,
                              .maximum_launch_impulse_newton_seconds =
                                  capability.maximum_launch_impulse_newton_seconds,
                              .minimum_flight_time =
                                  legacy::DurationNanoseconds{
                                      capability.minimum_flight_time},
                              .maximum_flight_time =
                                  legacy::DurationNanoseconds{
                                      capability.maximum_flight_time},
                              .maximum_landing_speed_mps =
                                  capability.maximum_landing_speed_mps,
                              .minimum_downward_impact_speed_mps =
                                  capability.minimum_downward_impact_speed_mps,
                              .minimum_landing_clearance_m =
                                  capability.minimum_landing_clearance_m,
                          },
                      .attitude_envelope =
                          legacy::ArbitraryAxisAttitudeEnvelope{
                              .maximum_angular_speed_radps =
                                  capability.maximum_angular_speed_radps,
                              .maximum_angular_acceleration_radps2 =
                                  capability.maximum_angular_acceleration_radps2,
                              .maximum_initial_angular_speed_radps =
                                  capability.maximum_initial_angular_speed_radps,
                              .minimum_settle_guard =
                                  legacy::DurationNanoseconds{
                                      capability.minimum_settle_guard},
                          },
                      .certified_state_error_bounds = ZeroHopperError(),
                  },
          };
        }
      },
      input.capability);
}

[[nodiscard]] legacy::PlannerAlgorithmConfig ConvertConfig(
    const PlannerConfig& config) {
  const auto old_resources = legacy::ResourceCaps{
      .maximum_expanded_states = config.search.resources.maximum_expanded_states,
      .maximum_reopened_states = config.search.resources.maximum_reopened_states,
      .maximum_generated_candidates =
          config.search.resources.maximum_generated_candidates,
      .maximum_open_states = config.search.resources.maximum_open_states,
      .maximum_memory_bytes = config.search.resources.maximum_memory_bytes,
  };
  const auto old_search = legacy::AraStarConfig{
      .initial_epsilon = config.search.initial_epsilon,
      .epsilon_decrement = config.search.epsilon_decrement,
      .target_epsilon = config.search.target_epsilon,
      .resource_caps = old_resources,
  };
  const auto old_corridor = legacy::CorridorConfig{
      .maximum_regions = config.corridor.maximum_regions,
      .maximum_inflation_iterations =
          config.corridor.maximum_inflation_iterations,
      .maximum_halfplanes_per_region =
          config.corridor.maximum_halfplanes_per_region,
      .maximum_split_depth = config.corridor.maximum_split_depth,
      .minimum_overlap_m = config.corridor.minimum_overlap_m,
      .sampling_spacing_m = config.corridor.sampling_spacing_m,
  };
  const auto smoothing = legacy::SmoothingConfig{
      .maximum_scp_iterations = config.optimization.maximum_iterations,
      .maximum_trust_region_reductions =
          config.optimization.maximum_trust_region_reductions,
      .initial_trust_region_m = config.optimization.initial_trust_region_m,
      .minimum_trust_region_m = config.optimization.minimum_trust_region_m,
      .constraint_tolerance = config.optimization.constraint_tolerance,
      .maximum_time_increase = legacy::DurationNanoseconds{2s},
  };
  const auto timing = legacy::TimeScalingConfig{
      .maximum_adaptive_samples = 256U,
      .minimum_parameter_step = 0.005,
      .maximum_forward_passes = 4U,
      .maximum_backward_passes = 4U,
      .enable_jerk_smoothing = false,
      .maximum_jerk_smoothing_iterations = 0U,
  };
  return legacy::PlannerAlgorithmConfig{
      .content_ref = Ref("adapter-algorithm-config", 'a'),
      .time_equivalence_tolerance = legacy::DurationNanoseconds{100ms},
      .max_input_skew = legacy::DurationNanoseconds{config.maximum_input_skew},
      .error_bound_model_id = "adapter-zero-error",
      .projection_cache_capacity = 8U,
      .ara_star = old_search,
      .wheeled =
          legacy::WheeledAlgorithmConfig{
              .state_lattice =
                  legacy::GridConfig{
                      .xy_resolution_m = config.wheel.xy_resolution_m,
                      .yaw_bin_count = config.wheel.yaw_bin_count,
                      .maximum_terminal_candidates =
                          config.wheel.maximum_terminal_candidates,
                  },
              .corridor = old_corridor,
              .smoothing = smoothing,
              .time_scaling = timing,
              .continuous_validation_maximum_subdivisions =
                  std::min<std::size_t>(
                      config.wheel.continuous_validation_maximum_subdivisions,
                      16U),
          },
      .legged =
          legacy::LeggedAlgorithmConfig{
              .pose_lattice =
                  legacy::GridConfig{
                      .xy_resolution_m = config.legged.xy_resolution_m,
                      .yaw_bin_count = config.legged.yaw_bin_count,
                      .maximum_terminal_candidates =
                          config.legged.maximum_terminal_candidates,
                  },
              .maximum_height_interval_splits =
                  config.legged.maximum_height_interval_splits,
              .corridor = old_corridor,
              .smoothing = smoothing,
              .time_scaling = timing,
              .continuous_validation_maximum_subdivisions =
                  config.legged.continuous_validation_maximum_subdivisions,
          },
      .hopper =
          legacy::HopperAlgorithmConfig{
              .maximum_landing_regions = config.hopper.maximum_landing_regions,
              .maximum_graph_nodes = config.hopper.maximum_graph_nodes,
              .maximum_graph_out_degree =
                  config.hopper.maximum_graph_out_degree,
              .yaw_partition_count = 4U,
              .support_direction_count = 16U,
              .maximum_nominal_aim_points_per_region =
                  config.hopper.maximum_nominal_aim_points_per_region,
              .maximum_full_certification_attempts =
                  config.hopper.maximum_certification_attempts,
              .maximum_interval_subdivision_depth = 8U,
              .maximum_collision_subdivision_depth = 8U,
              .maximum_root_iterations = 64U,
              .maximum_flight_tube_sections =
                  config.hopper.maximum_flight_tube_sections,
              .landing_region_inflation_iterations = 16U,
              .landing_region_maximum_split_depth = 4U,
              .landing_region_maximum_vertices = 16U,
          },
      .learned_cost_policy =
          legacy::LearnedCostPolicy{
              .mode = legacy::LearnedCostPolicy::Mode::kDisabled,
          },
      .deterministic_execution =
          legacy::DeterministicExecutionConfig{
              .fixed_thread_count = 1U,
              .stable_candidate_order = config.stable_candidate_order,
              .preallocated_memory_pools = true,
          },
  };
}

template <class Object>
[[nodiscard]] std::shared_ptr<const Object> OpaqueObject() {
  auto owner = std::make_shared<int>(1);
  const auto* opaque = reinterpret_cast<const Object*>(owner.get());
  return std::shared_ptr<const Object>(
      std::move(owner), opaque);
}

[[nodiscard]] legacy::ResolvedCapabilityBindings ConvertBindings(
    const legacy::SafetyCapabilityProfile& profile,
    const PlannerInput& input) {
  return std::visit(
      [&](const auto& capability) {
        using Capability = std::decay_t<decltype(capability)>;
        legacy::ResolvedCapabilityBindings bindings{
            .motion_model =
                {capability.motion_model_ref,
                 OpaqueObject<legacy::MotionModel>()},
            .analytic_cost_model =
                {capability.analytic_cost_model_ref,
                 OpaqueObject<legacy::AnalyticCostModel>()},
        };
        if constexpr (std::is_same_v<Capability, legacy::HopperCapability>) {
          const auto& public_capability = std::get<HopperCapability>(input.capability);
          auto gravity = std::make_shared<const legacy::GravityModel>(
              legacy::GravityModel{
                  .content_ref = capability.gravity_model_ref,
                  .frame_id = capability.frame_id,
                  .nominal_acceleration_mps2 =
                      Eigen::Vector3d{
                          public_capability.gravity_mps2.x,
                          public_capability.gravity_mps2.y,
                          public_capability.gravity_mps2.z},
                  .acceleration_error_mps2 = ZeroBox(),
                  .spatial_validity_m =
                      legacy::AxisAlignedBox3{
                          .center = {},
                          .half_extent = {10'000.0, 10'000.0, 10'000.0},
                      },
                  .valid_from =
                      legacy::ClockStamp{"planner_input", 0ns},
                  .valid_until =
                      legacy::ClockStamp{"planner_input", 24h},
              });
          auto error =
              std::make_shared<const legacy::DeterministicErrorModel>(
                  legacy::DeterministicErrorModel{
                      .content_ref = Ref("adapter-hopper-error", '5'),
                      .initial_position_error_m =
                          legacy::AxisAlignedBox3{
                              .half_extent = {0.01, 0.01, 0.01},
                          },
                      .initial_velocity_error_mps =
                          legacy::AxisAlignedBox3{
                              .half_extent = {0.01, 0.01, 0.01},
                          },
                      .launch_execution_velocity_error_mps =
                          legacy::AxisAlignedBox3{
                              .half_extent = {0.01, 0.01, 0.01},
                          },
                      .gravity_error_mps2 =
                          legacy::AxisAlignedBox3{
                              .half_extent = {0.0, 0.0, 0.01},
                          },
                      .landing_plane_origin_error_m =
                          legacy::AxisAlignedBox3{
                              .half_extent = {0.01, 0.01, 0.01},
                          },
                      .landing_plane_normal_error =
                          legacy::RotationVectorBall{0.01},
                      .landing_plane_residual_error_m =
                          legacy::SymmetricScalarInterval{0.0, 0.01},
                  });
          auto actuator =
              std::make_shared<const legacy::ActuatorOrImpulseProfile>(
                  legacy::ActuatorOrImpulseProfile{
                      .content_ref = Ref("adapter-hopper-actuator", '6'),
                      .platform_mass_kg = public_capability.platform_mass_kg,
                      .launch_preparation_time =
                          legacy::DurationNanoseconds{100ms},
                      .landing_settle_time =
                          legacy::DurationNanoseconds{200ms},
                      .nominal_landing_center_normal_offset_m = 0.5,
                  });
          auto rotation = std::make_shared<const legacy::BodyRotationEnvelope>(
              legacy::BodyRotationEnvelope{
                  .content_ref = Ref("adapter-hopper-rotation", '7'),
                  .arbitrary_attitude_body_envelope =
                      BoxPolytope(Vec3{0.45, 0.45, 0.45}),
              });
          bindings.gravity_model =
              legacy::ResolvedBinding<legacy::GravityModel>{
                  gravity->content_ref, gravity};
          bindings.error_model =
              legacy::ResolvedBinding<legacy::DeterministicErrorModel>{
                  error->content_ref, error};
          bindings.actuator_or_impulse_profile =
              legacy::ResolvedBinding<legacy::ActuatorOrImpulseProfile>{
                  actuator->content_ref, actuator};
          bindings.body_rotation_envelope =
              legacy::ResolvedBinding<legacy::BodyRotationEnvelope>{
                  rotation->content_ref, rotation};
        }
        return bindings;
      },
      profile.content);
}

class AdapterContractObject final : public legacy::ImmutableContractObject {
 public:
  AdapterContractObject(
      legacy::ContentRef ref, const legacy::ContractObjectKind kind)
      : ref_(std::move(ref)), kind_(kind) {}

  [[nodiscard]] legacy::ContentRef content_ref() const override { return ref_; }
  [[nodiscard]] legacy::ContractObjectKind kind() const override { return kind_; }

 private:
  legacy::ContentRef ref_;
  legacy::ContractObjectKind kind_;
};

class AdapterCertificate final : public legacy::CertificationObject {
 public:
  AdapterCertificate(
      legacy::ContentRef ref, legacy::CertificationProvenance provenance)
      : ref_(std::move(ref)), provenance_(std::move(provenance)) {}

  [[nodiscard]] legacy::ContentRef content_ref() const override { return ref_; }
  [[nodiscard]] legacy::ContractObjectKind kind() const override {
    return legacy::ContractObjectKind::kCertification;
  }
  [[nodiscard]] const legacy::CertificationProvenance& provenance()
      const override {
    return provenance_;
  }

 private:
  legacy::ContentRef ref_;
  legacy::CertificationProvenance provenance_;
};

class AdapterRegistry final : public legacy::ContractObjectRegistry {
 public:
  AdapterRegistry(
      std::shared_ptr<const legacy::ImmutableMapSnapshot> map,
      std::shared_ptr<const legacy::SafetyCapabilityProfile> capability,
      std::shared_ptr<const legacy::PlannerAlgorithmConfig> config,
      legacy::ResolvedCapabilityBindings bindings,
      const bool throw_on_map_lookup)
      : map_(std::move(map)),
        capability_(std::move(capability)),
        config_(std::move(config)),
        bindings_(std::move(bindings)),
        throw_on_map_lookup_(throw_on_map_lookup) {}

  [[nodiscard]] std::shared_ptr<const legacy::ImmutableMapSnapshot>
  FindMapSnapshot(
      const legacy::ContentRef& ref,
      const std::string_view handle) const override {
    if (throw_on_map_lookup_) {
      throw std::runtime_error{"injected legacy registry failure"};
    }
    return map_->snapshot_ref() == ref && map_->immutable_data_handle() == handle
               ? map_
               : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const legacy::SafetyCapabilityProfile>
  FindSafetyCapability(const legacy::ContentRef& ref) const override {
    return capability_->content_ref == ref ? capability_ : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const legacy::PlannerAlgorithmConfig>
  FindAlgorithmConfig(const legacy::ContentRef& ref) const override {
    return config_->content_ref == ref ? config_ : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const legacy::LearnedCostSnapshot>
  FindLearnedCost(
      const legacy::ContentRef&, std::string_view) const override {
    return nullptr;
  }

  [[nodiscard]] legacy::Result<legacy::ResolvedCapabilityBindings>
  ResolveCapabilityBindings(
      const legacy::SafetyCapabilityProfile& profile) const override {
    if (profile.content_ref != capability_->content_ref) {
      return legacy::Error{
          legacy::ErrorCode::kMissingRegistryObject,
          "safety_capability",
          "adapter capability is absent",
      };
    }
    return bindings_;
  }

  [[nodiscard]] legacy::Result<
      std::shared_ptr<const legacy::ImmutableContractObject>>
  Resolve(
      const legacy::ContentRef& ref,
      const legacy::ContractObjectKind expected_kind) const override {
    if (expected_kind == legacy::ContractObjectKind::kCertification) {
      std::vector<legacy::ContentRef> inputs{
          bindings_.motion_model.content_ref,
          bindings_.analytic_cost_model.content_ref,
      };
      const auto append = [&inputs](const auto& binding) {
        if (binding.has_value()) {
          inputs.push_back(binding->content_ref);
        }
      };
      append(bindings_.gravity_model);
      append(bindings_.error_model);
      append(bindings_.actuator_or_impulse_profile);
      append(bindings_.body_rotation_envelope);
      append(bindings_.attitude_tightening_table);
      return std::shared_ptr<const legacy::ImmutableContractObject>(
          std::make_shared<const AdapterCertificate>(
              ref,
              legacy::CertificationProvenance{
                  .source_map_snapshot_ref = map_->snapshot_ref(),
                  .source_safety_capability_ref = capability_->content_ref,
                  .source_algorithm_config_ref = config_->content_ref,
                  .input_refs = std::move(inputs),
                  .certification_purpose = "temporary-legacy-adapter",
              }));
    }
    return std::shared_ptr<const legacy::ImmutableContractObject>(
        std::make_shared<const AdapterContractObject>(ref, expected_kind));
  }

 private:
  std::shared_ptr<const legacy::ImmutableMapSnapshot> map_;
  std::shared_ptr<const legacy::SafetyCapabilityProfile> capability_;
  std::shared_ptr<const legacy::PlannerAlgorithmConfig> config_;
  legacy::ResolvedCapabilityBindings bindings_;
  bool throw_on_map_lookup_{};
};

[[nodiscard]] legacy::PlatformType ConvertPlatform(const PlatformCapability& value) {
  switch (CapabilityPlatform(value)) {
    case PlatformType::kWheeled:
      return legacy::PlatformType::kWheeled;
    case PlatformType::kLegged:
      return legacy::PlatformType::kLegged;
    case PlatformType::kHopper:
      return legacy::PlatformType::kHopper;
  }
  throw ConversionError{"INVALID_PLATFORM_TYPE"};
}

[[nodiscard]] legacy::PlatformState ConvertState(const PlannerInput& input) {
  const PlatformType capability_platform = CapabilityPlatform(input.capability);
  return std::visit(
      [&](const auto& state) -> legacy::PlatformState {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, WheeledState>) {
          if (capability_platform != PlatformType::kWheeled) {
            throw ConversionError{"PLATFORM_STATE_CAPABILITY_MISMATCH"};
          }
          return legacy::WheeledOrLeggedState{
              .position_m = ToLegacy(state.pose.position_m),
              .yaw_rad = YawFromQuaternion(state.pose.orientation),
              .linear_velocity_mps = ToLegacy(state.velocity.linear_mps),
              .yaw_rate_radps = state.velocity.angular_radps.z,
              .error_bounds = ZeroGroundError(),
          };
        } else if constexpr (std::is_same_v<State, LeggedState>) {
          if (capability_platform != PlatformType::kLegged) {
            throw ConversionError{"PLATFORM_STATE_CAPABILITY_MISMATCH"};
          }
          return legacy::WheeledOrLeggedState{
              .position_m = ToLegacy(state.body_pose.position_m),
              .yaw_rad = YawFromQuaternion(state.body_pose.orientation),
              .linear_velocity_mps = ToLegacy(state.body_velocity.linear_mps),
              .yaw_rate_radps = state.body_velocity.angular_radps.z,
              .error_bounds = ZeroGroundError(),
          };
        } else {
          if (capability_platform != PlatformType::kHopper) {
            throw ConversionError{"PLATFORM_STATE_CAPABILITY_MISMATCH"};
          }
          return legacy::HopperState{
              .position_m = ToLegacy(state.pose.position_m),
              .orientation_body_to_frame = ToLegacy(state.pose.orientation),
              .linear_velocity_mps = ToLegacy(state.velocity.linear_mps),
              .angular_velocity_radps = ToLegacy(state.velocity.angular_radps),
              .error_bounds = ZeroHopperError(),
          };
        }
      },
      input.current_state);
}

[[nodiscard]] double Dot(const Vec3& left, const Vec3& right) noexcept {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

[[nodiscard]] Vec3 Subtract(const Vec3& left, const Vec3& right) noexcept {
  return {left.x - right.x, left.y - right.y, left.z - right.z};
}

[[nodiscard]] Vec3 Cross(const Vec3& left, const Vec3& right) noexcept {
  return {
      left.y * right.z - left.z * right.y,
      left.z * right.x - left.x * right.z,
      left.x * right.y - left.y * right.x,
  };
}

[[nodiscard]] Vec3 Normalize(const Vec3& value) {
  const double norm = std::hypot(value.x, value.y, value.z);
  if (!std::isfinite(norm) || norm <= 1.0e-12) {
    throw ConversionError{"INVALID_PLANAR_GOAL_GEOMETRY"};
  }
  return {value.x / norm, value.y / norm, value.z / norm};
}

[[nodiscard]] legacy::GoalRegion ConvertGoal(const GoalRegion& goal) {
  legacy::GoalTarget target = std::visit(
      [](const auto& typed_goal) -> legacy::GoalTarget {
        using Goal = std::decay_t<decltype(typed_goal)>;
        if constexpr (std::is_same_v<Goal, PointGoal>) {
          return legacy::PointGoal{
              .position_m = ToLegacy(typed_goal.position_m),
              .position_tolerance_m = typed_goal.tolerance_m,
          };
        } else {
          if (typed_goal.boundary_m.size() < 3U) {
            throw ConversionError{"INVALID_PLANAR_GOAL_GEOMETRY"};
          }
          const Vec3 origin = typed_goal.boundary_m.front();
          const Vec3 basis_u =
              Normalize(Subtract(typed_goal.boundary_m[1], origin));
          const Vec3 normal = Normalize(Cross(
              basis_u, Subtract(typed_goal.boundary_m[2], origin)));
          const Vec3 basis_v = Cross(normal, basis_u);
          std::vector<legacy::Vec2> vertices;
          vertices.reserve(typed_goal.boundary_m.size());
          for (const Vec3& point : typed_goal.boundary_m) {
            const Vec3 delta = Subtract(point, origin);
            vertices.push_back({Dot(delta, basis_u), Dot(delta, basis_v)});
          }
          double twice_area = 0.0;
          for (std::size_t index = 0U; index < vertices.size(); ++index) {
            const auto& a = vertices[index];
            const auto& b = vertices[(index + 1U) % vertices.size()];
            twice_area += a.x * b.y - a.y * b.x;
          }
          if (twice_area < 0.0) {
            std::reverse(vertices.begin(), vertices.end());
          }
          return legacy::PlanarRegionGoal{
              .plane =
                  legacy::LandingPlane{
                      .origin_m = ToLegacy(origin),
                      .normal = ToLegacy(normal),
                      .basis_u = ToLegacy(basis_u),
                      .basis_v = ToLegacy(basis_v),
                      .residual_bound_m = 0.0,
                  },
              .polygon = legacy::ConvexPolygonUv{.vertices_uv = std::move(vertices)},
              .normal_tolerance_m = typed_goal.normal_tolerance_m,
          };
        }
      },
      goal.target);
  std::optional<legacy::CircularYawInterval> yaw;
  if (goal.yaw_rad.has_value()) {
    yaw = legacy::CircularYawInterval{
        .start_rad = *goal.yaw_rad - goal.yaw_tolerance_rad,
        .span_rad = 2.0 * goal.yaw_tolerance_rad,
    };
  }
  return legacy::GoalRegion{
      .goal_id = goal.goal_id,
      .target = std::move(target),
      .optional_yaw_interval = std::move(yaw),
  };
}

[[nodiscard]] std::optional<legacy::PreviousExecutionContext>
ConvertExecutionContext(
    const PlannerInput& input,
    const legacy::ContentRef& map_ref,
    const legacy::ContentRef& capability_ref) {
  if (!input.previous_execution.has_value()) {
    return std::nullopt;
  }
  return std::visit(
      [&](const auto& context)
          -> std::optional<legacy::PreviousExecutionContext> {
        using Context = std::decay_t<decltype(context)>;
        const std::string plan_id =
            context.active_plan_id.value_or("adapter-active-plan");
        if constexpr (std::is_same_v<Context, GroundExecutionContext>) {
          if (CapabilityPlatform(input.capability) == PlatformType::kHopper) {
            throw ConversionError{"EXECUTION_CONTEXT_PLATFORM_MISMATCH"};
          }
          legacy::ControllerStatus status = legacy::ControllerStatus::kReady;
          if (context.state == GroundExecutionState::kExecuting) {
            status = legacy::ControllerStatus::kExecuting;
          } else if (context.state == GroundExecutionState::kHolding) {
            status = legacy::ControllerStatus::kHolding;
          } else if (context.state == GroundExecutionState::kFault) {
            status = legacy::ControllerStatus::kFault;
          }
          return legacy::PreviousExecutionContext{
              .active_bundle_ref = Ref(plan_id, 'e'),
              .active_bundle_handle = "adapter-active-bundle-handle",
              .commit_boundary = legacy::TimeCommitBoundary{
                  legacy::DurationNanoseconds{0ns}},
              .execution_cursor = legacy::TimeExecutionCursor{
                  .offset = legacy::DurationNanoseconds{0ns},
                  .segment_id = context.active_segment_id,
              },
              .controller_status = status,
              .source_map_snapshot_ref = map_ref,
              .source_capability_ref = capability_ref,
          };
        } else {
          if (CapabilityPlatform(input.capability) != PlatformType::kHopper) {
            throw ConversionError{"EXECUTION_CONTEXT_PLATFORM_MISMATCH"};
          }
          const std::string boundary_id =
              context.active_segment_id.value_or("adapter-active-hop");
          const bool locked =
              context.state == HopperExecutionState::kJumpCommitted ||
              context.state == HopperExecutionState::kInFlight;
          legacy::JumpExecutionState jump_state =
              legacy::JumpExecutionState::kGroundHold;
          legacy::ControllerStatus controller = legacy::ControllerStatus::kHolding;
          switch (context.state) {
            case HopperExecutionState::kGroundHold:
            case HopperExecutionState::kLandedHold:
              break;
            case HopperExecutionState::kJumpReady:
              jump_state = legacy::JumpExecutionState::kJumpReady;
              controller = legacy::ControllerStatus::kReady;
              break;
            case HopperExecutionState::kJumpCommitted:
              jump_state = legacy::JumpExecutionState::kJumpCommitted;
              controller = legacy::ControllerStatus::kCommitted;
              break;
            case HopperExecutionState::kInFlight:
              jump_state = legacy::JumpExecutionState::kInFlight;
              controller = legacy::ControllerStatus::kInFlight;
              break;
            case HopperExecutionState::kEmergencyDelegated:
              controller = legacy::ControllerStatus::kFault;
              break;
          }
          return legacy::PreviousExecutionContext{
              .active_bundle_ref = Ref(plan_id, 'e'),
              .active_bundle_handle = "adapter-active-bundle-handle",
              .commit_boundary = legacy::JumpCommitBoundary{
                  .boundary_id = boundary_id,
                  .locked = locked,
              },
              .execution_cursor = legacy::JumpExecutionCursor{
                  .jump_state = jump_state,
                  .boundary_id = boundary_id,
              },
              .controller_status = controller,
              .source_map_snapshot_ref = map_ref,
              .source_capability_ref = capability_ref,
          };
        }
      },
      *input.previous_execution);
}

struct LegacyCall final {
  legacy::PlanningRequest request;
  std::shared_ptr<const AdapterRegistry> registry;
  std::unique_ptr<legacy::SafeProjectionCache> cache;
};

[[nodiscard]] LegacyCall ConvertRequest(
    const PlannerInput& input, const LegacyV3FaultMode fault_mode) {
  auto map_result = legacy::ImmutableMapSnapshot::Create(ConvertMap(input));
  if (!legacy::IsOk(map_result)) {
    const legacy::Error& error = std::get<legacy::Error>(map_result);
    throw ConversionError{"INVALID_MAP_" + UpperToken(error.field_path)};
  }
  auto map = std::get<std::shared_ptr<const legacy::ImmutableMapSnapshot>>(
      std::move(map_result));
  auto capability = std::make_shared<const legacy::SafetyCapabilityProfile>(
      ConvertCapability(input));
  auto config = std::make_shared<const legacy::PlannerAlgorithmConfig>(
      ConvertConfig(input.config));
  legacy::ResolvedCapabilityBindings bindings =
      ConvertBindings(*capability, input);
  auto registry = std::make_shared<const AdapterRegistry>(
      map, capability, config, bindings,
      fault_mode == LegacyV3FaultMode::kThrowingRegistry);
  legacy::PlanningRequest request{
      .request_id = input.request_id,
      .request_time = LegacyTime(input.state_time),
      .state_time = LegacyTime(input.state_time),
      .frame_id = input.world.local_map.frame_id,
      .platform_type = ConvertPlatform(input.capability),
      .current_state = ConvertState(input),
      .goal = ConvertGoal(input.goal),
      .map_snapshot = std::move(map),
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(config),
      .capability_bindings = std::move(bindings),
  };
  request.previous_execution_context = ConvertExecutionContext(
      input,
      request.map_snapshot->snapshot_ref(),
      request.safety_capability->content_ref);
  return LegacyCall{
      .request = std::move(request),
      .registry = std::move(registry),
      .cache = std::make_unique<legacy::SafeProjectionCache>(8U),
  };
}

[[nodiscard]] PlanningOutcome ConvertOutcome(
    const legacy::PlanningOutcome outcome) {
  switch (outcome) {
    case legacy::PlanningOutcome::kNewReferenceReady:
      return PlanningOutcome::kNewReferenceAvailable;
    case legacy::PlanningOutcome::kSafeFrontierReferenceReady:
      return PlanningOutcome::kSafeFrontierReferenceAvailable;
    case legacy::PlanningOutcome::kNoKnownSafeRoute:
      return PlanningOutcome::kNoKnownSafeRoute;
    case legacy::PlanningOutcome::kGoalInfeasible:
      return PlanningOutcome::kGoalInfeasible;
    case legacy::PlanningOutcome::kInvalidRequest:
      return PlanningOutcome::kInvalidRequest;
    case legacy::PlanningOutcome::kStaleInput:
      return PlanningOutcome::kStaleInput;
    case legacy::PlanningOutcome::kNumericalFailure:
      return PlanningOutcome::kNumericalFailure;
    case legacy::PlanningOutcome::kResourceLimit:
      return PlanningOutcome::kResourceExhausted;
    case legacy::PlanningOutcome::kActiveReferenceInvalidated:
      return PlanningOutcome::kActiveReferenceInvalidated;
  }
  return PlanningOutcome::kNumericalFailure;
}

[[nodiscard]] ExecutionDirective ConvertDirective(
    const legacy::ExecutionDirective directive) {
  switch (directive) {
    case legacy::ExecutionDirective::kActivateNewBundle:
      return ExecutionDirective::kActivateNewReference;
    case legacy::ExecutionDirective::kContinueActiveBundle:
      return ExecutionDirective::kContinueActiveReference;
    case legacy::ExecutionDirective::kHoldStationary:
      return ExecutionDirective::kHoldPosition;
    case legacy::ExecutionDirective::kContinueCommittedJump:
      return ExecutionDirective::kContinueCommittedHop;
    case legacy::ExecutionDirective::kNoSafePlannerReference:
      return ExecutionDirective::kNoSafeReference;
  }
  return ExecutionDirective::kNoSafeReference;
}

[[nodiscard]] Pose3 PublicPose(const legacy::PoseXyzYaw& pose) {
  return Pose3{
      .position_m = ToPublic(pose.position_m),
      .orientation = QuaternionFromYaw(pose.yaw_rad),
  };
}

[[nodiscard]] TrajectoryReference ConvertTrajectory(
    const legacy::ReferenceBundle& bundle,
    const legacy::PlanningResponse& response) {
  TrajectoryReference trajectory{
      .semantics =
          bundle.platform_type == legacy::PlatformType::kWheeled
              ? TrajectorySemantics::kWheeledBase
              : TrajectorySemantics::kLeggedBodyReference,
  };
  const auto& waypoints = bundle.route_skeleton.content.waypoints;
  trajectory.points.reserve(waypoints.size());
  const std::chrono::nanoseconds duration =
      response.call_diagnostics.expected_execution_time.has_value()
          ? response.call_diagnostics.expected_execution_time->value
          : 0ns;
  for (std::size_t index = 0U; index < waypoints.size(); ++index) {
    const auto time = waypoints.size() > 1U
                          ? duration * static_cast<std::int64_t>(index) /
                                static_cast<std::int64_t>(waypoints.size() - 1U)
                          : 0ns;
    trajectory.points.push_back(TrajectoryPoint{
        .time_from_start = time,
        .pose = PublicPose(waypoints[index]),
    });
  }
  return trajectory;
}

[[nodiscard]] Vec3 PlanePoint(
    const legacy::LandingPlane& plane, const legacy::Vec2& uv) {
  return {
      plane.origin_m.x + plane.basis_u.x * uv.x + plane.basis_v.x * uv.y,
      plane.origin_m.y + plane.basis_u.y * uv.x + plane.basis_v.y * uv.y,
      plane.origin_m.z + plane.basis_u.z * uv.x + plane.basis_v.z * uv.y,
  };
}

[[nodiscard]] HopReference ConvertHop(const legacy::HopperReference& reference) {
  std::vector<Vec3> landing_boundary;
  landing_boundary.reserve(
      reference.next_landing_region.convex_polygon.vertices_uv.size());
  for (const legacy::Vec2& uv :
       reference.next_landing_region.convex_polygon.vertices_uv) {
    landing_boundary.push_back(
        PlanePoint(reference.next_landing_region.landing_plane, uv));
  }
  return HopReference{
      .segments =
          {
              HopSegment{
                  .segment_id = "next-hop",
                  .launch_pose =
                      Pose3{
                          .position_m = ToPublic(
                              reference.jump_boundary.nominal_launch_state
                                  .position_m),
                          .orientation = ToPublic(
                              reference.jump_boundary.nominal_launch_state
                                  .orientation_body_to_frame),
                      },
                  .landing_region_boundary_m = std::move(landing_boundary),
                  .flight_time =
                      reference.jump_boundary.ballistic_flight_time.value,
                  .launch_velocity_mps = ToPublic(
                      reference.jump_boundary.nominal_launch_state
                          .linear_velocity_mps),
                  .flight_tube_radius_m = std::max(
                      0.0,
                      reference.certified_flight_tube
                          .minimum_certified_clearance_m),
              },
          },
  };
}

[[nodiscard]] std::optional<MotionReference> ConvertReference(
    const PlannerInput& input,
    const legacy::PlanningResponse& response) {
  if (!response.new_reference_bundle.has_value()) {
    return std::nullopt;
  }
  const legacy::ReferenceBundle& bundle = *response.new_reference_bundle;
  MotionReferenceData data = std::visit(
      [&](const auto& reference) -> MotionReferenceData {
        using Reference = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Reference, legacy::HopperReference>) {
          return ConvertHop(reference);
        } else {
          return ConvertTrajectory(bundle, response);
        }
      },
      bundle.platform_reference);
  return MotionReference{
      .plan_id = input.request_id + "-reference",
      .platform_type = CapabilityPlatform(input.capability),
      .input_time = input.state_time,
      .data = std::move(data),
  };
}

[[nodiscard]] PlannerOutput CanceledOutput() {
  return PlannerOutput{
      .outcome = PlanningOutcome::kCanceled,
      .directive = ExecutionDirective::kHoldPosition,
      .reason_code = "REQUEST_CANCELED",
  };
}

[[nodiscard]] PlannerOutput FailureOutput(
    const PlanningOutcome outcome, std::string reason) {
  return PlannerOutput{
      .outcome = outcome,
      .directive = ExecutionDirective::kNoSafeReference,
      .reason_code = std::move(reason),
  };
}

[[nodiscard]] PlannerOutput ConvertResponse(
    const PlannerInput& input,
    const legacy::PlanningResponse& response,
    const std::chrono::nanoseconds elapsed) {
  PlannerDiagnostics diagnostics{
      .elapsed = elapsed,
      .expanded_states = response.call_diagnostics.expanded_state_count,
      .warning_codes = response.call_diagnostics.message_codes,
  };
  if (response.call_diagnostics.expected_execution_time.has_value()) {
    diagnostics.best_cost =
        static_cast<double>(
            response.call_diagnostics.expected_execution_time->value.count()) /
        1.0e9;
  }
  const PlanningOutcome outcome = ConvertOutcome(response.planning_outcome);
  ExecutionDirective directive = ConvertDirective(response.execution_directive);
  if (outcome == PlanningOutcome::kNoKnownSafeRoute &&
      directive == ExecutionDirective::kHoldPosition) {
    directive = ExecutionDirective::kNoSafeReference;
  }
  return PlannerOutput{
      .outcome = outcome,
      .directive = directive,
      .reason_code = response.reason_code,
      .reference = ConvertReference(input, response),
      .diagnostics = std::move(diagnostics),
  };
}

}  // namespace

struct LegacyV3Adapter::Impl final {
  std::size_t fallback_invocations{};
  LegacyV3FaultMode fault_mode{LegacyV3FaultMode::kNone};
};

LegacyV3Adapter::LegacyV3Adapter()
    : LegacyV3Adapter(LegacyV3FaultMode::kNone) {}

LegacyV3Adapter::LegacyV3Adapter(const LegacyV3FaultMode fault_mode)
    : impl_(std::make_unique<Impl>(Impl{.fault_mode = fault_mode})) {}

LegacyV3Adapter::~LegacyV3Adapter() = default;

LegacyV3Adapter::LegacyV3Adapter(LegacyV3Adapter&&) noexcept = default;

LegacyV3Adapter& LegacyV3Adapter::operator=(LegacyV3Adapter&&) noexcept = default;

PlannerOutput LegacyV3Adapter::Plan(const PlannerInput& input) noexcept {
  const auto started = std::chrono::steady_clock::now();
  try {
    if (input.stop_token.stop_requested()) {
      return CanceledOutput();
    }
    LegacyCall call = ConvertRequest(
        input,
        impl_ == nullptr ? LegacyV3FaultMode::kNone : impl_->fault_mode);
    if (input.stop_token.stop_requested()) {
      return CanceledOutput();
    }
    static const legacy::SemanticValidator validator;
    auto planner = legacy::MakeDefaultPlannerV3(
        validator, *call.registry, *call.cache);
    const legacy::PlanningResponse response = planner->Plan(call.request);
    if (input.stop_token.stop_requested()) {
      return CanceledOutput();
    }
    return ConvertResponse(
        input,
        response,
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started));
  } catch (const ConversionError& error) {
    return FailureOutput(PlanningOutcome::kInvalidRequest, error.reason_code());
  } catch (const std::bad_alloc&) {
    return FailureOutput(
        PlanningOutcome::kResourceExhausted, "RESOURCE_EXHAUSTED");
  } catch (const std::exception&) {
    return FailureOutput(
        PlanningOutcome::kNumericalFailure, "INTERNAL_PLANNER_EXCEPTION");
  } catch (...) {
    return FailureOutput(
        PlanningOutcome::kNumericalFailure, "INTERNAL_PLANNER_EXCEPTION");
  }
}

std::size_t LegacyV3Adapter::fallback_invocations() const noexcept {
  return impl_ == nullptr ? 0U : impl_->fallback_invocations;
}

}  // namespace lunar::planning
