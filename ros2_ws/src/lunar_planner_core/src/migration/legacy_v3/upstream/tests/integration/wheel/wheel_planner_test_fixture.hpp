#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"

namespace lunar::planning::v3::test {

using namespace std::chrono_literals;

[[nodiscard]] inline ContentRef Ref(std::string id, char digit) {
  return {
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digit),
  };
}

[[nodiscard]] inline WheelMotionPrimitive Primitive(
    std::string id,
    WheelMotionPrimitive::Kind kind,
    PoseXyzYaw relative_end,
    std::chrono::nanoseconds duration) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .relative_end_pose = relative_end,
      .nominal_duration = DurationNanoseconds{duration},
      .swept_geometry_ref = Ref("wheel-sweep", 'b'),
  };
}

[[nodiscard]] inline MapSnapshotInput MapInput() {
  constexpr std::size_t kWidth = 12U;
  constexpr std::size_t kHeight = 12U;
  constexpr std::size_t kCount = kWidth * kHeight;
  return {
      .snapshot_ref = Ref("wheel-map", '1'),
      .map_revision = 1U,
      .immutable_data_handle = "wheel-planner-map",
      .source_time =
          {.clock_id = "mission", .tick = 10ns},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {12.0, 12.0, 1.0},
          },
      .geometry =
          {
              .width = kWidth,
              .height = kHeight,
              .resolution_m = 1.0,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {LayerKind::kKnownMask, Ref("known", '2')},
              {LayerKind::kElevation, Ref("elevation", '3')},
              {LayerKind::kTerrainNormal, Ref("normal", '4')},
              {LayerKind::kRoughness, Ref("roughness", '5')},
              {LayerKind::kHardObstacle, Ref("obstacle", '6')},
              {LayerKind::kConfidence, Ref("confidence", '7')},
          },
      .known_mask = std::vector<std::uint8_t>(kCount, 1U),
      .elevation_m = std::vector<float>(kCount, 0.0F),
      .normal_x = std::vector<float>(kCount, 0.0F),
      .normal_y = std::vector<float>(kCount, 0.0F),
      .normal_z = std::vector<float>(kCount, 1.0F),
      .roughness_m = std::vector<float>(kCount, 0.0F),
      .hard_obstacle_mask =
          std::vector<std::uint8_t>(kCount, 0U),
      .confidence = std::vector<float>(kCount, 1.0F),
  };
}

[[nodiscard]] inline SafetyCapabilityProfile Capability() {
  return {
      .content_ref = Ref("wheel-capability", 'a'),
      .content =
          WheeledCapability{
              .frame_id = "map",
              .collision_envelope =
                  {
                      .vertices_xy_m =
                          {{-0.2, -0.2}, {0.2, -0.2},
                           {0.2, 0.2}, {-0.2, 0.2}},
                      .minimum_z_m = -0.1,
                      .maximum_z_m = 0.5,
                  },
              .motion_model_ref = Ref("motion", 'c'),
              .analytic_cost_model_ref = Ref("cost", 'd'),
              .hard_limits =
                  {
                      .maximum_forward_speed_mps = 1.0,
                      .maximum_reverse_speed_mps = 0.8,
                      .maximum_spin_rate_radps = 1.0,
                      .maximum_forward_acceleration_mps2 = 1.0,
                      .maximum_braking_deceleration_mps2 = 1.0,
                      .maximum_yaw_acceleration_radps2 = 1.0,
                      .maximum_lateral_acceleration_mps2 = 1.0,
                      .maximum_drive_curvature_per_m = 1.0,
                      .maximum_slope_rad = 0.5,
                      .minimum_clearance_m = 0.0,
                  },
              .motion_primitives =
                  {
                      Primitive(
                          "forward",
                          WheelMotionPrimitive::Kind::
                              kDriveForwardLine,
                          {{1.0, 0.0, 0.0}, 0.0}, 1s),
                      Primitive(
                          "forward-arc",
                          WheelMotionPrimitive::Kind::
                              kDriveForwardArc,
                          {{1.0, 0.0, 0.0}, 0.0}, 100s),
                      Primitive(
                          "reverse",
                          WheelMotionPrimitive::Kind::
                              kDriveReverseLine,
                          {{-1.0, 0.0, 0.0}, 0.0}, 100s),
                      Primitive(
                          "spin-cw",
                          WheelMotionPrimitive::Kind::kSpinCw,
                          {{0.0, 0.0, 0.0},
                           -std::numbers::pi / 2.0},
                          100s),
                      Primitive(
                          "spin-ccw",
                          WheelMotionPrimitive::Kind::kSpinCcw,
                          {{0.0, 0.0, 0.0},
                           std::numbers::pi / 2.0},
                          1s),
                      Primitive(
                          "switch",
                          WheelMotionPrimitive::Kind::
                              kStopAndSwitch,
                          {{0.0, 0.0, 0.0}, 0.0}, 100ms),
                  },
          },
  };
}

[[nodiscard]] inline PlannerAlgorithmConfig Algorithm() {
  PlannerAlgorithmConfig config;
  config.content_ref = Ref("wheel-algorithm", 'e');
  config.time_equivalence_tolerance =
      DurationNanoseconds{2s};
  config.max_input_skew = DurationNanoseconds{1s};
  config.error_bound_model_id = "wheel-error-bound";
  config.projection_cache_capacity = 4U;
  config.ara_star = {
      .initial_epsilon = 1.0,
      .epsilon_decrement = 0.1,
      .target_epsilon = 1.0,
      .resource_caps =
          {
              .maximum_expanded_states = 256U,
              .maximum_reopened_states = 256U,
              .maximum_generated_candidates = 8U,
              .maximum_open_states = 256U,
              .maximum_memory_bytes = 1U << 20U,
          },
  };
  config.wheeled = {
      .state_lattice =
          {
              .xy_resolution_m = 1.0,
              .yaw_bin_count = 4U,
              .maximum_terminal_candidates = 4U,
          },
      .corridor =
          {
              .maximum_regions = 16U,
              .maximum_inflation_iterations = 8U,
              .maximum_halfplanes_per_region = 8U,
              .maximum_split_depth = 4U,
              .minimum_overlap_m = 0.05,
              .sampling_spacing_m = 0.25,
          },
      .smoothing =
          {
              .maximum_scp_iterations = 2U,
              .maximum_trust_region_reductions = 2U,
              .initial_trust_region_m = 0.2,
              .minimum_trust_region_m = 0.01,
              .constraint_tolerance = 1.0e-6,
              .maximum_time_increase =
                  DurationNanoseconds{2s},
          },
      .time_scaling =
          {
              .maximum_adaptive_samples = 128U,
              .minimum_parameter_step = 0.1,
              .maximum_forward_passes = 1U,
              .maximum_backward_passes = 1U,
              .enable_jerk_smoothing = false,
              .maximum_jerk_smoothing_iterations = 0U,
          },
      .continuous_validation_maximum_subdivisions = 8U,
  };
  config.deterministic_execution = {
      .fixed_thread_count = 1U,
      .stable_candidate_order = true,
      .preallocated_memory_pools = true,
  };
  return config;
}

struct WheelPlannerFixture final {
  PlanningRequest request;
  ResolvedTerminalSet terminal;
};

[[nodiscard]] inline WheelPlannerFixture MakeFixture(
    bool spin_only = false) {
  auto map_result = ImmutableMapSnapshot::Create(MapInput());
  auto map =
      std::get<std::shared_ptr<const ImmutableMapSnapshot>>(
          std::move(map_result));
  auto capability =
      std::make_shared<const SafetyCapabilityProfile>(
          Capability());
  auto algorithm =
      std::make_shared<const PlannerAlgorithmConfig>(
          Algorithm());

  PlanningRequest request{
      .request_id =
          spin_only ? "wheel-spin-request"
                    : "wheel-drive-request",
      .request_time =
          {.clock_id = "mission", .tick = 20ns},
      .state_time =
          {.clock_id = "mission", .tick = 20ns},
      .frame_id = "map",
      .platform_type = PlatformType::kWheeled,
      .current_state =
          WheeledOrLeggedState{
              .position_m = {4.5, 4.5, 0.0},
              .yaw_rad = 0.0,
              .linear_velocity_mps = {},
              .yaw_rate_radps = 0.0,
          },
      .goal =
          {
              .goal_id = "wheel-goal",
              .target =
                  PointGoal{
                      .position_m =
                          spin_only
                              ? Vec3{4.5, 4.5, 0.0}
                              : Vec3{5.5, 4.5, 0.0},
                      .position_tolerance_m = 0.1,
                  },
          },
      .map_snapshot = std::move(map),
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(algorithm),
  };
  ResolvedTerminalSet terminal{
      .kind = TerminalKind::kGoal,
      .candidates =
          {
              {
                  .stable_id =
                      spin_only ? "spin-goal" : "drive-goal",
                  .cell = spin_only ? Cell{4, 4} : Cell{5, 4},
                  .position_m =
                      spin_only ? Vec3{4.5, 4.5, 0.0}
                                : Vec3{5.5, 4.5, 0.0},
                  .yaw_interval =
                      spin_only
                          ? std::optional<CircularYawInterval>{
                                {
                                    .start_rad =
                                        std::numbers::pi / 2.0,
                                    .span_rad = 0.0,
                                }}
                          : std::nullopt,
                  .requires_zero_speed = true,
                  .clearance_m = 5.0,
              },
          },
      .reason_code = "GOAL_REACHABLE",
  };
  return {
      .request = std::move(request),
      .terminal = std::move(terminal),
  };
}

}  // namespace lunar::planning::v3::test
