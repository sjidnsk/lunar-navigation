#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

#include <nlohmann/json.hpp>

#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::benchmark {
namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

constexpr std::string_view kSchemaVersion =
    "lunar-three-platform-qualification/v1";
constexpr std::string_view kCapabilityFreezeSha256 =
    "60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95";
constexpr std::string_view kBuildType = LUNAR_BUILD_TYPE;
constexpr std::size_t kWarmupRuns = 1U;
constexpr std::size_t kMeasuredRuns = 20U;
constexpr double kBaseResolutionM = 0.2;

enum class Scenario : std::uint8_t {
  kWheel50,
  kWheelLocalFrontierStress,
  kLegged50,
  kWheelOneKilometer,
  kLeggedOneKilometer,
  kHopperDirect,
  kHopperAlternateTime,
  kHopperBlocked,
};

struct CaseDefinition final {
  std::string_view id;
  Scenario scenario{};
  PlatformType platform{};
  std::string_view capability_version;
  std::size_t global_width{};
  std::size_t global_height{};
  double global_resolution_m{};
  std::size_t global_level{};
  double release_threshold_s{};
};

constexpr std::array<CaseDefinition, 8U> kCases{
    CaseDefinition{
        .id = "wheel_50m_l0",
        .scenario = Scenario::kWheel50,
        .platform = PlatformType::kWheeled,
        .capability_version = "wheeled-engineering-baseline-v1",
        .global_width = 250U,
        .global_height = 250U,
        .global_resolution_m = 0.2,
        .global_level = 0U,
        .release_threshold_s = 2.0,
    },
    CaseDefinition{
        .id = "wheel_local_frontier_stress",
        .scenario = Scenario::kWheelLocalFrontierStress,
        .platform = PlatformType::kWheeled,
        .capability_version = "wheeled-engineering-baseline-v1",
        .global_width = 250U,
        .global_height = 250U,
        .global_resolution_m = 0.2,
        .global_level = 0U,
        .release_threshold_s = 2.0,
    },
    CaseDefinition{
        .id = "legged_50m_l0",
        .scenario = Scenario::kLegged50,
        .platform = PlatformType::kLegged,
        .capability_version = "quad48-approved-baseline-v1",
        .global_width = 250U,
        .global_height = 250U,
        .global_resolution_m = 0.2,
        .global_level = 0U,
        .release_threshold_s = 3.0,
    },
    CaseDefinition{
        .id = "wheel_1km_l5",
        .scenario = Scenario::kWheelOneKilometer,
        .platform = PlatformType::kWheeled,
        .capability_version = "wheeled-engineering-baseline-v1",
        .global_width = 250U,
        .global_height = 250U,
        .global_resolution_m = 4.0,
        .global_level = 5U,
        .release_threshold_s = 3.0,
    },
    CaseDefinition{
        .id = "legged_1km_l5",
        .scenario = Scenario::kLeggedOneKilometer,
        .platform = PlatformType::kLegged,
        .capability_version = "quad48-approved-baseline-v1",
        .global_width = 250U,
        .global_height = 250U,
        .global_resolution_m = 4.0,
        .global_level = 5U,
        .release_threshold_s = 3.0,
    },
    CaseDefinition{
        .id = "hopper_direct_100m",
        .scenario = Scenario::kHopperDirect,
        .platform = PlatformType::kHopper,
        .capability_version = "hopper-engineering-baseline-v1",
        .global_width = 140U,
        .global_height = 15U,
        .global_resolution_m = 0.8,
        .global_level = 2U,
        .release_threshold_s = 1.0,
    },
    CaseDefinition{
        .id = "hopper_alternate_time_100m",
        .scenario = Scenario::kHopperAlternateTime,
        .platform = PlatformType::kHopper,
        .capability_version = "hopper-engineering-baseline-v1",
        .global_width = 140U,
        .global_height = 15U,
        .global_resolution_m = 0.8,
        .global_level = 2U,
        .release_threshold_s = 2.0,
    },
    CaseDefinition{
        .id = "hopper_complete_blocked_100m",
        .scenario = Scenario::kHopperBlocked,
        .platform = PlatformType::kHopper,
        .capability_version = "hopper-engineering-baseline-v1",
        .global_width = 140U,
        .global_height = 15U,
        .global_resolution_m = 0.8,
        .global_level = 2U,
        .release_threshold_s = 5.0,
    },
};

struct Arguments final {
  std::string output_path;
  std::optional<std::string> case_filter;
};

struct Timings final {
  double p50_s{};
  double p95_s{};
  double maximum_s{};
};

struct StableMetrics final {
  std::uint64_t expanded_states{};
  std::size_t open_peak{};
  std::size_t peak_work_memory_bytes{};
  std::size_t hopper_certification_attempts{};

  auto operator<=>(const StableMetrics&) const = default;
};

struct TimingSamples final {
  std::vector<double> complete;
  std::vector<double> global;
  std::vector<double> local;
  std::vector<double> landing_field;
  std::vector<double> spatial_index;
  std::vector<double> ballistic_solve;
  std::vector<double> flight_tube;

  void Reserve() {
    complete.reserve(kMeasuredRuns);
    global.reserve(kMeasuredRuns);
    local.reserve(kMeasuredRuns);
    landing_field.reserve(kMeasuredRuns);
    spatial_index.reserve(kMeasuredRuns);
    ballistic_solve.reserve(kMeasuredRuns);
    flight_tube.reserve(kMeasuredRuns);
  }
};

[[nodiscard]] Arguments ParseArguments(const int argc, char** argv) {
  if ((argc != 3 && argc != 5) ||
      std::string_view{argv[1]} != "--output" ||
      std::string_view{argv[2]}.empty()) {
    throw std::invalid_argument{
        "usage: hierarchical_planner_benchmark --output PATH [--case ID]"};
  }
  if (argc == 5 &&
      (std::string_view{argv[3]} != "--case" ||
       std::string_view{argv[4]}.empty())) {
    throw std::invalid_argument{
        "usage: hierarchical_planner_benchmark --output PATH [--case ID]"};
  }
  return {
      .output_path = argv[2],
      .case_filter = argc == 5
                         ? std::optional<std::string>{argv[4]}
                         : std::nullopt,
  };
}

[[nodiscard]] Quaternion Yaw(const double radians) noexcept {
  return {
      .w = std::cos(0.5 * radians),
      .z = std::sin(0.5 * radians),
  };
}

[[nodiscard]] WheeledCapability ApprovedWheelCapability() {
  constexpr double arc_yaw = std::numbers::pi / 16.0;
  constexpr double arc_x = 0.19509032201612825;
  constexpr double arc_y = 0.01921471959676957;
  return {
      .footprint_xy_m = {
          {0.591, 0.409}, {0.591, -0.409},
          {-0.591, -0.409}, {-0.591, 0.409},
      },
      .body_extent_m = {1.182, 0.818, 1.29996},
      .wheel_diameter_m = 0.319,
      .wheel_width_m = 0.148,
      .wheelbase_m = 0.8175,
      .track_width_m = 0.670,
      .minimum_underbody_clearance_m = 0.210,
      .maximum_local_obstacle_relief_m = 0.20,
      .allow_unsupported_gap = false,
      .minimum_body_z_m = 0.0,
      .maximum_body_z_m = 1.29996,
      .maximum_forward_speed_mps = 1.5,
      .maximum_reverse_speed_mps = 1.5,
      .maximum_spin_rate_radps = 1.0,
      .maximum_acceleration_mps2 = 0.5,
      .maximum_braking_deceleration_mps2 = 0.5,
      .maximum_yaw_acceleration_radps2 = 0.5,
      .maximum_lateral_acceleration_mps2 = 0.5,
      .maximum_curvature_per_m = 1.0,
      .maximum_slope_rad = 0.3490658503988659,
      .minimum_clearance_m = 0.20,
      .motion_primitives = {
          WheelMotionPrimitive{
              .primitive_id = "forward",
              .kind = WheelPrimitiveKind::kForward,
              .relative_end_pose = Pose3{.position_m = {0.2, 0.0, 0.0}},
          },
          WheelMotionPrimitive{
              .primitive_id = "reverse",
              .kind = WheelPrimitiveKind::kReverse,
              .relative_end_pose = Pose3{.position_m = {-0.2, 0.0, 0.0}},
          },
          WheelMotionPrimitive{
              .primitive_id = "forward-arc-left",
              .kind = WheelPrimitiveKind::kForwardArc,
              .relative_end_pose = Pose3{
                  .position_m = {arc_x, arc_y, 0.0},
                  .orientation = Yaw(arc_yaw),
              },
          },
          WheelMotionPrimitive{
              .primitive_id = "forward-arc-right",
              .kind = WheelPrimitiveKind::kForwardArc,
              .relative_end_pose = Pose3{
                  .position_m = {arc_x, -arc_y, 0.0},
                  .orientation = Yaw(-arc_yaw),
              },
          },
          WheelMotionPrimitive{
              .primitive_id = "reverse-arc-left",
              .kind = WheelPrimitiveKind::kReverseArc,
              .relative_end_pose = Pose3{
                  .position_m = {-arc_x, arc_y, 0.0},
                  .orientation = Yaw(-arc_yaw),
              },
          },
          WheelMotionPrimitive{
              .primitive_id = "reverse-arc-right",
              .kind = WheelPrimitiveKind::kReverseArc,
              .relative_end_pose = Pose3{
                  .position_m = {-arc_x, -arc_y, 0.0},
                  .orientation = Yaw(arc_yaw),
              },
          },
          WheelMotionPrimitive{
              .primitive_id = "spin-left",
              .kind = WheelPrimitiveKind::kSpinCounterclockwise,
              .relative_end_pose = Pose3{.orientation = Yaw(arc_yaw)},
          },
          WheelMotionPrimitive{
              .primitive_id = "spin-right",
              .kind = WheelPrimitiveKind::kSpinClockwise,
              .relative_end_pose = Pose3{.orientation = Yaw(-arc_yaw)},
          },
          WheelMotionPrimitive{
              .primitive_id = "stop-switch",
              .kind = WheelPrimitiveKind::kStopAndSwitch,
          },
      },
  };
}

[[nodiscard]] LeggedCapability ApprovedLeggedCapability() {
  constexpr double yaw_step = std::numbers::pi / 16.0;
  return {
      .body_extent_m = {0.68, 0.33, 0.35},
      .platform_mass_kg = 15.89,
      .maximum_payload_kg = 10.0,
      .maximum_slope_rad = 0.5235987755982988,
      .maximum_step_height_m = 0.50,
      .maximum_gap_width_m = 0.30,
      .minimum_body_clearance_m = 0.30,
      .step_vertical_rate_mps = 0.10,
      .body_height_m = {0.28, 0.38},
      .forward_speed_mps = {-1.5, 1.5},
      .lateral_speed_mps = {-0.8, 0.8},
      .yaw_rate_radps = {-1.0, 1.0},
      .maximum_linear_acceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .motion_primitives = {
          LeggedBodyPrimitive{
              .primitive_id = "forward",
              .kind = LeggedPrimitiveKind::kForward,
              .body_frame_displacement_m = {0.2, 0.0, 0.0},
          },
          LeggedBodyPrimitive{
              .primitive_id = "backward",
              .kind = LeggedPrimitiveKind::kBackward,
              .body_frame_displacement_m = {-0.2, 0.0, 0.0},
          },
          LeggedBodyPrimitive{
              .primitive_id = "left",
              .kind = LeggedPrimitiveKind::kLateralLeft,
              .body_frame_displacement_m = {0.0, 0.2, 0.0},
          },
          LeggedBodyPrimitive{
              .primitive_id = "right",
              .kind = LeggedPrimitiveKind::kLateralRight,
              .body_frame_displacement_m = {0.0, -0.2, 0.0},
          },
          LeggedBodyPrimitive{
              .primitive_id = "spin-left",
              .kind = LeggedPrimitiveKind::kSpin,
              .yaw_change_rad = yaw_step,
          },
          LeggedBodyPrimitive{
              .primitive_id = "spin-right",
              .kind = LeggedPrimitiveKind::kSpin,
              .yaw_change_rad = -yaw_step,
          },
      },
  };
}

[[nodiscard]] HopperCapability ApprovedHopperCapability() {
  return {
      .specific_impulse_s = 301.0,
      .reference_total_mass_kg = 20.0,
      .reference_propellant_mass_kg = 0.20,
      .landing_support_radius_m = 0.45,
      .flight_collision_radius_m = 0.55,
      .maximum_landing_plane_residual_m = 0.05,
      .landing_lateral_margin_m = 0.20,
      .flight_map_margin_m = 0.20,
      .reachability_delta_v_margin_ratio = 0.10,
      .standard_gravity_mps2 = 9.80665,
      .maximum_landing_slope_rad = 0.17453292519943295,
  };
}

void SetObstacle(
    GridMap& map, const std::size_t x, const std::size_t y,
    const float height_m) {
  const std::size_t index = y * map.width + x;
  std::get<std::vector<std::uint8_t>>(
      map.layers.at("obstacle").values)[index] = 1U;
  std::get<std::vector<float>>(
      map.layers.at("obstacle_height").values)[index] = height_m;
}

void AddGroundObstacleCourse(GridMap& map) {
  const std::size_t center_y = map.height / 2U;
  const std::size_t first_x = map.width / 3U;
  const std::size_t second_x = 2U * map.width / 3U;
  constexpr std::size_t half_gap = 14U;
  for (std::size_t y = 0U; y < map.height; ++y) {
    if (y + half_gap < center_y || y > center_y + half_gap) {
      SetObstacle(map, first_x, y, 1.0F);
    }
    const std::size_t second_center = center_y - 28U;
    if (y + half_gap < second_center || y > second_center + half_gap) {
      SetObstacle(map, second_x, y, 1.0F);
    }
  }
}

void AddWheelLocalStressField(GridMap& map) {
  const std::size_t center_y = map.height / 2U;
  for (std::size_t y = 2U; y + 2U < map.height; ++y) {
    for (std::size_t x = 2U; x + 2U < map.width; ++x) {
      const std::size_t corridor_distance =
          y > center_y ? y - center_y : center_y - y;
      if (corridor_distance <= 8U) {
        continue;
      }
      const std::uint64_t mixed =
          (static_cast<std::uint64_t>(x) * 73'856'093ULL) ^
          (static_cast<std::uint64_t>(y) * 19'349'663ULL) ^
          0x9e3779b97f4a7c15ULL;
      if (mixed % 100U < 18U) {
        SetObstacle(map, x, y, 1.0F);
      }
    }
  }
}

[[nodiscard]] PlannerInput MakeGroundInput(const CaseDefinition& definition) {
  const bool wheeled = definition.platform == PlatformType::kWheeled;
  PlannerInput input = wheeled ? test::MakeValidWheelInput()
                               : test::MakeValidLeggedInput();
  input.request_id = std::string{definition.id};
  input.platform_id = wheeled ? "wheeled-lunar-explorer" : "yobotics-quad48";
  input.capability_version = std::string{definition.capability_version};
  input.global_map_generation = 31U;
  input.local_map_generation = 37U;
  input.map_from_odom_generation = 41U;
  input.world.global_map = test::MakeFlatMap(
      "map", definition.global_width, definition.global_height,
      definition.global_resolution_m);
  const std::size_t local_axis =
      definition.scenario == Scenario::kWheelLocalFrontierStress ? 56U : 100U;
  input.world.local_map = test::MakeFlatMap(
      "odom", local_axis, local_axis, kBaseResolutionM);
  input.world.map_from_odom = RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = input.state_time,
  };
  input.config.global_map.base_resolution_m = kBaseResolutionM;
  input.config.wheel.xy_resolution_m = kBaseResolutionM;
  input.config.legged.xy_resolution_m = kBaseResolutionM;

  const bool large = definition.global_level > 0U;
  const std::size_t start_cell = large ? 12U : 12U;
  const std::size_t goal_cell = large ? definition.global_width - 13U
                                      : definition.global_width - 13U;
  const std::size_t center_y = definition.global_height / 2U;
  const double start_x =
      (static_cast<double>(start_cell) + 0.5) * definition.global_resolution_m;
  const double start_y =
      (static_cast<double>(center_y) + 0.5) * definition.global_resolution_m;
  const double goal_x =
      (static_cast<double>(goal_cell) + 0.5) * definition.global_resolution_m;
  const double local_center_offset =
      (static_cast<double>(local_axis / 2U) + 0.5) * kBaseResolutionM;
  input.world.local_map.origin_m = {
      start_x - local_center_offset,
      start_y - local_center_offset,
      0.0,
  };
  input.goal_map = GoalRegion{
      .goal_id = "qualification-goal",
      .target = PointGoal{
          .position_m = {goal_x, start_y, 0.0},
          .tolerance_m = 0.05,
      },
  };
  input.position_uncertainty_m = 0.0;
  input.velocity_uncertainty_mps = 0.0;
  if (wheeled) {
    input.current_state = WheeledState{
        .pose = Pose3{.position_m = {start_x, start_y, 0.0}},
    };
    input.capability = ApprovedWheelCapability();
  } else {
    input.current_state = LeggedState{
        .body_pose = Pose3{.position_m = {start_x, start_y, 0.33}},
    };
    input.capability = ApprovedLeggedCapability();
  }
  if (!large) {
    AddGroundObstacleCourse(input.world.global_map);
  }
  if (definition.scenario == Scenario::kWheelLocalFrontierStress) {
    AddWheelLocalStressField(input.world.local_map);
  }
  return input;
}

void AddHopperBlockingColumn(GridMap& map, const float height_m) {
  constexpr double obstacle_x_m = 55.0;
  constexpr double obstacle_y_m = 5.0;
  const std::size_t center_x = static_cast<std::size_t>(
      std::floor((obstacle_x_m - map.origin_m.x) / map.resolution_m));
  const std::size_t center_y = static_cast<std::size_t>(
      std::floor((obstacle_y_m - map.origin_m.y) / map.resolution_m));
  for (std::size_t y = center_y - 4U; y <= center_y + 4U; ++y) {
    SetObstacle(map, center_x, y, height_m);
  }
}

[[nodiscard]] PlannerInput MakeHopperInput(
    const CaseDefinition& definition) {
  PlannerInput input = test::MakeValidHopperInput();
  input.request_id = std::string{definition.id};
  input.platform_id = "hopper-lunar-explorer";
  input.capability_version = std::string{definition.capability_version};
  input.global_map_generation = 31U;
  input.local_map_generation = 37U;
  input.map_from_odom_generation = 41U;
  input.world.global_map = test::MakeFlatMap(
      "map", definition.global_width, definition.global_height,
      definition.global_resolution_m);
  const auto local_width = static_cast<std::size_t>(std::llround(
      static_cast<double>(definition.global_width) *
      definition.global_resolution_m / kBaseResolutionM));
  const auto local_height = static_cast<std::size_t>(std::llround(
      static_cast<double>(definition.global_height) *
      definition.global_resolution_m / kBaseResolutionM));
  input.world.local_map = test::MakeFlatMap(
      "odom", local_width, local_height, kBaseResolutionM);
  input.world.map_from_odom = RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = input.state_time,
  };
  input.config.global_map.base_resolution_m = kBaseResolutionM;
  input.position_uncertainty_m = 0.0;
  input.velocity_uncertainty_mps = 0.0;
  input.capability = ApprovedHopperCapability();
  const Vec3 launch{5.0, 5.0, 0.0};
  const Vec3 landing{105.0, 5.0, 0.0};
  input.current_state = HopperState{.pose = Pose3{.position_m = launch}};
  input.goal_map = GoalRegion{
      .goal_id = "qualification-hopper-goal",
      .target = PointGoal{.position_m = landing, .tolerance_m = 0.0},
      .yaw_rad = std::nullopt,
      .yaw_tolerance_rad = 0.0,
  };
  if (definition.scenario == Scenario::kHopperAlternateTime) {
    AddHopperBlockingColumn(input.world.global_map, 35.0F);
    AddHopperBlockingColumn(input.world.local_map, 35.0F);
  } else if (definition.scenario == Scenario::kHopperBlocked) {
    AddHopperBlockingColumn(input.world.global_map, 100.0F);
    AddHopperBlockingColumn(input.world.local_map, 100.0F);
  }
  return input;
}

[[nodiscard]] PlannerInput MakeInput(const CaseDefinition& definition) {
  return definition.platform == PlatformType::kHopper
             ? MakeHopperInput(definition)
             : MakeGroundInput(definition);
}

[[nodiscard]] std::string PlatformName(const PlatformType platform) {
  switch (platform) {
    case PlatformType::kWheeled:
      return "WHEELED";
    case PlatformType::kLegged:
      return "LEGGED";
    case PlatformType::kHopper:
      return "HOPPER";
  }
  throw std::logic_error{"unknown platform"};
}

[[nodiscard]] std::string OutcomeName(const PlanningOutcome outcome) {
  switch (outcome) {
    case PlanningOutcome::kNewReferenceAvailable:
      return "NEW_REFERENCE_AVAILABLE";
    case PlanningOutcome::kSafeFrontierReferenceAvailable:
      return "SAFE_FRONTIER_REFERENCE_AVAILABLE";
    case PlanningOutcome::kNoKnownSafeRoute:
      return "NO_KNOWN_SAFE_ROUTE";
    case PlanningOutcome::kGoalInfeasible:
      return "GOAL_INFEASIBLE";
    case PlanningOutcome::kInvalidRequest:
      return "INVALID_REQUEST";
    case PlanningOutcome::kStaleInput:
      return "STALE_INPUT";
    case PlanningOutcome::kNumericalFailure:
      return "NUMERICAL_FAILURE";
    case PlanningOutcome::kResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case PlanningOutcome::kActiveReferenceInvalidated:
      return "ACTIVE_REFERENCE_INVALIDATED";
    case PlanningOutcome::kCanceled:
      return "CANCELED";
  }
  throw std::logic_error{"unknown planning outcome"};
}

[[nodiscard]] std::string ModeName(const PlannerOutput& output) {
  return output.diagnostics.local_trajectory.has_value()
             ? std::string{ToString(
                   output.diagnostics.local_trajectory->trajectory_mode)}
             : "NO_REFERENCE";
}

[[nodiscard]] std::optional<double> HopFlightTimeSeconds(
    const PlannerOutput& output) {
  if (!output.reference.has_value()) {
    return std::nullopt;
  }
  const auto* hops = std::get_if<HopReference>(&output.reference->data);
  if (hops == nullptr || hops->segments.size() != 1U) {
    return std::nullopt;
  }
  return std::chrono::duration<double>(
      hops->segments.front().flight_time).count();
}

void RequireExpected(
    const CaseDefinition& definition, const PlannerOutput& output) {
  if (definition.scenario == Scenario::kHopperBlocked) {
    if (output.outcome != PlanningOutcome::kNoKnownSafeRoute ||
        output.reason_code != "HOPPER_ALL_FLIGHT_TUBES_BLOCKED" ||
        output.reference.has_value()) {
      throw std::runtime_error{
          "blocked hopper result mismatch: " + output.reason_code};
    }
    return;
  }
  if (output.outcome != PlanningOutcome::kNewReferenceAvailable ||
      !output.reference.has_value()) {
    throw std::runtime_error{
        "positive case result mismatch: " + output.reason_code};
  }
  if (definition.platform == PlatformType::kWheeled &&
      output.reason_code != "WHEEL_PLAN_AVAILABLE") {
    throw std::runtime_error{"wheel positive reason mismatch"};
  }
  if (definition.platform == PlatformType::kLegged &&
      output.reason_code != "LEGGED_BODY_PLAN_AVAILABLE") {
    throw std::runtime_error{"legged positive reason mismatch"};
  }
  if (definition.platform == PlatformType::kHopper) {
    const auto* hops = std::get_if<HopReference>(&output.reference->data);
    if (output.reason_code != "HOPPER_SINGLE_HOP_AVAILABLE" ||
        hops == nullptr || hops->segments.size() != 1U ||
        output.certified_hops.size() != 1U || output.continuation != nullptr) {
      throw std::runtime_error{"hopper result is not one certified segment"};
    }
    if (definition.scenario == Scenario::kHopperAlternateTime) {
      const double seconds = std::chrono::duration<double>(
          hops->segments.front().flight_time).count();
      const double minimum_energy_time = std::sqrt(2.0 * 100.0 / 1.62);
      if (!(seconds > minimum_energy_time)) {
        throw std::runtime_error{"alternate-time fixture used minimum arc"};
      }
    }
  }
}

[[nodiscard]] Json PoseJson(const Pose3& pose) {
  return Json::array({
      pose.position_m.x,
      pose.position_m.y,
      pose.position_m.z,
      pose.orientation.w,
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z,
  });
}

[[nodiscard]] Json StableSignature(const PlannerOutput& output) {
  Json signature{
      {"outcome", OutcomeName(output.outcome)},
      {"directive", static_cast<std::uint8_t>(output.directive)},
      {"reason", output.reason_code},
      {"mode", ModeName(output)},
      {"best_cost", output.diagnostics.best_cost.has_value()
                        ? Json{*output.diagnostics.best_cost}
                        : Json{nullptr}},
      {"warnings", output.diagnostics.warning_codes},
  };
  if (!output.reference.has_value()) {
    return signature;
  }
  signature["plan_id"] = output.reference->plan_id;
  for (const auto& pose : output.reference->preview.poses_map) {
    signature["preview"].push_back(PoseJson(pose));
  }
  if (const auto* trajectory =
          std::get_if<TrajectoryReference>(&output.reference->data)) {
    for (const auto& point : trajectory->points) {
      signature["trajectory"].push_back({
          {"time_ns", point.time_from_start.count()},
          {"pose", PoseJson(point.pose)},
      });
    }
  } else if (const auto* hops =
                 std::get_if<HopReference>(&output.reference->data)) {
    for (const auto& hop : hops->segments) {
      signature["hops"].push_back({
          {"segment_id", hop.segment_id},
          {"flight_time_ns", hop.flight_time.count()},
          {"launch_velocity",
           {hop.launch_velocity_mps.x, hop.launch_velocity_mps.y,
            hop.launch_velocity_mps.z}},
          {"nominal_landing",
           {hop.nominal_landing_point_m.x, hop.nominal_landing_point_m.y,
            hop.nominal_landing_point_m.z}},
          {"available_delta_v_mps", hop.available_delta_v_mps},
          {"required_delta_v_mps", hop.required_delta_v_mps},
      });
    }
  }
  return signature;
}

[[nodiscard]] std::string HashSignature(const Json& signature) {
  constexpr std::uint64_t offset_basis = 14'695'981'039'346'656'037ULL;
  constexpr std::uint64_t prime = 1'099'511'628'211ULL;
  std::uint64_t hash = offset_basis;
  for (const unsigned char byte : signature.dump()) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= prime;
  }
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << hash;
  return stream.str();
}

[[nodiscard]] StableMetrics Metrics(const PlannerOutput& output) {
  StableMetrics metrics{.expanded_states = output.diagnostics.expanded_states};
  if (output.diagnostics.hierarchical.has_value()) {
    const auto& hierarchical = *output.diagnostics.hierarchical;
    metrics.expanded_states = std::max(
        metrics.expanded_states,
        hierarchical.global_expanded_states +
            hierarchical.local_expanded_states);
    metrics.open_peak = hierarchical.global_open_peak;
    metrics.peak_work_memory_bytes =
        hierarchical.estimated_work_memory_bytes;
    metrics.hopper_certification_attempts =
        hierarchical.hopper_certification_attempts;
  }
  return metrics;
}

[[nodiscard]] double Seconds(const Clock::duration duration) {
  return std::chrono::duration<double>(duration).count();
}

[[nodiscard]] Timings Summarize(std::vector<double> values) {
  if (values.size() != kMeasuredRuns) {
    throw std::logic_error{"benchmark timing count is invalid"};
  }
  std::ranges::sort(values);
  const std::size_t middle = values.size() / 2U;
  const double median = values.size() % 2U == 0U
                            ? 0.5 * (values[middle - 1U] + values[middle])
                            : values[middle];
  const std::size_t p95_index =
      static_cast<std::size_t>(std::ceil(0.95 * values.size())) - 1U;
  return {
      .p50_s = median,
      .p95_s = values[p95_index],
      .maximum_s = values.back(),
  };
}

[[nodiscard]] Json TimingJson(std::vector<double> values) {
  const Timings timing = Summarize(std::move(values));
  return {
      {"p50_s", timing.p50_s},
      {"p95_s", timing.p95_s},
      {"maximum_s", timing.maximum_s},
  };
}

[[nodiscard]] std::size_t PeakResidentMemoryBytes() {
#if defined(__linux__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss >= 0) {
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024U;
  }
#endif
  return 0U;
}

void RecordStages(const PlannerOutput& output, TimingSamples& samples) {
  if (!output.diagnostics.hierarchical.has_value()) {
    samples.global.push_back(0.0);
    samples.local.push_back(0.0);
    samples.landing_field.push_back(0.0);
    samples.spatial_index.push_back(0.0);
    samples.ballistic_solve.push_back(0.0);
    samples.flight_tube.push_back(0.0);
    return;
  }
  const auto& metrics = *output.diagnostics.hierarchical;
  samples.global.push_back(
      std::chrono::duration<double>(metrics.global_elapsed).count());
  samples.local.push_back(
      std::chrono::duration<double>(metrics.local_elapsed).count());
  samples.landing_field.push_back(
      std::chrono::duration<double>(metrics.landing_field_elapsed).count());
  samples.spatial_index.push_back(
      std::chrono::duration<double>(metrics.spatial_index_elapsed).count());
  samples.ballistic_solve.push_back(
      std::chrono::duration<double>(metrics.ballistic_solve_elapsed).count());
  samples.flight_tube.push_back(std::chrono::duration<double>(
      metrics.flight_tube_certification_elapsed).count());
}

[[nodiscard]] Json RunCase(const CaseDefinition& definition) {
  const PlannerInput input = MakeInput(definition);
  Planner planner;
  for (std::size_t run = 0U; run < kWarmupRuns; ++run) {
    RequireExpected(definition, planner.Plan(input));
  }

  TimingSamples samples;
  samples.Reserve();
  bool first = true;
  std::string stable_hash;
  StableMetrics stable_metrics;
  PlanningOutcome stable_outcome{PlanningOutcome::kInvalidRequest};
  std::string stable_reason;
  std::string stable_mode;
  std::optional<double> stable_hop_flight_time_s;
  for (std::size_t run = 0U; run < kMeasuredRuns; ++run) {
    const auto started = Clock::now();
    const PlannerOutput output = planner.Plan(input);
    samples.complete.push_back(Seconds(Clock::now() - started));
    RequireExpected(definition, output);
    RecordStages(output, samples);
    if (output.diagnostics.hierarchical.has_value() &&
        output.diagnostics.hierarchical->global_level !=
            definition.global_level) {
      throw std::runtime_error{"reported global map level mismatch"};
    }
    const std::string hash = HashSignature(StableSignature(output));
    const StableMetrics metrics = Metrics(output);
    const std::string mode = ModeName(output);
    if (first) {
      first = false;
      stable_hash = hash;
      stable_metrics = metrics;
      stable_outcome = output.outcome;
      stable_reason = output.reason_code;
      stable_mode = mode;
      stable_hop_flight_time_s = HopFlightTimeSeconds(output);
    } else if (hash != stable_hash || metrics != stable_metrics ||
               output.outcome != stable_outcome ||
               output.reason_code != stable_reason || mode != stable_mode) {
      throw std::runtime_error{
          "reference, result, mode, or diagnostic counts changed"};
    }
  }

  const Timings complete = Summarize(std::move(samples.complete));
  const double width_m = static_cast<double>(definition.global_width) *
      definition.global_resolution_m;
  const double height_m = static_cast<double>(definition.global_height) *
      definition.global_resolution_m;
  return {
      {"case_id", definition.id},
      {"platform", PlatformName(definition.platform)},
      {"capability_version", definition.capability_version},
      {"map",
       {
           {"width_m", width_m},
           {"height_m", height_m},
           {"resolution_m", definition.global_resolution_m},
           {"level", definition.global_level},
           {"cells", definition.global_width * definition.global_height},
       }},
      {"local_map",
       {
           {"width_m", static_cast<double>(input.world.local_map.width) *
                           input.world.local_map.resolution_m},
           {"height_m", static_cast<double>(input.world.local_map.height) *
                            input.world.local_map.resolution_m},
           {"resolution_m", input.world.local_map.resolution_m},
           {"cells", input.world.local_map.CellCount()},
       }},
      {"runs", kMeasuredRuns},
      {"p50_s", complete.p50_s},
      {"p95_s", complete.p95_s},
      {"maximum_s", complete.maximum_s},
      {"stage_timings_s",
       {
           {"global_search", TimingJson(std::move(samples.global))},
           {"local_planning", TimingJson(std::move(samples.local))},
           {"landing_field", TimingJson(std::move(samples.landing_field))},
           {"spatial_index", TimingJson(std::move(samples.spatial_index))},
           {"ballistic_solve", TimingJson(std::move(samples.ballistic_solve))},
           {"flight_tube_certification",
            TimingJson(std::move(samples.flight_tube))},
       }},
      {"expanded_states", stable_metrics.expanded_states},
      {"open_peak", stable_metrics.open_peak},
      {"peak_work_memory_bytes", stable_metrics.peak_work_memory_bytes},
      {"peak_resident_memory_bytes", PeakResidentMemoryBytes()},
      {"hopper_certification_attempts",
       stable_metrics.hopper_certification_attempts},
      {"hop_flight_time_s", stable_hop_flight_time_s.has_value()
                                ? Json(*stable_hop_flight_time_s)
                                : Json(nullptr)},
      {"reference_hash", stable_hash},
      {"planning_outcome", OutcomeName(stable_outcome)},
      {"reason_code", stable_reason},
      {"mode", stable_mode},
      {"deterministic", true},
      {"release_p95_threshold_s", definition.release_threshold_s},
      {"release_threshold_passed",
       complete.p95_s <= definition.release_threshold_s},
  };
}

[[nodiscard]] Json Run(const std::optional<std::string>& case_filter) {
  if (kBuildType != "Release") {
    throw std::runtime_error{"PERFORMANCE_BUILD_NOT_RELEASE"};
  }
  Json cases = Json::object();
  Json results = Json::array();
  for (const CaseDefinition& definition : kCases) {
    if (case_filter.has_value() && *case_filter != definition.id) {
      continue;
    }
    try {
      Json result = RunCase(definition);
      cases[std::string{definition.id}] = result;
      results.push_back(std::move(result));
    } catch (const std::exception& error) {
      throw std::runtime_error{
          "case " + std::string{definition.id} + ": " + error.what()};
    }
  }
  if (case_filter.has_value() && results.empty()) {
    throw std::runtime_error{"unknown benchmark case: " + *case_filter};
  }
  return {
      {"schema_version", kSchemaVersion},
      {"build_type", kBuildType},
      {"warmup_runs", kWarmupRuns},
      {"measured_runs", kMeasuredRuns},
      {"timing_unit", "s"},
      {"capability_authority", "approved-engineering-baseline"},
      {"capability_freeze_sha256", kCapabilityFreezeSha256},
      {"ubuntu_amd64_release_evaluated", true},
      {"jetson_agx_orin_evaluated", false},
      {"cases", std::move(cases)},
      {"results", std::move(results)},
  };
}

void Write(const std::string& path, const Json& document) {
  std::ofstream stream{path, std::ios::binary | std::ios::trunc};
  if (!stream) {
    throw std::runtime_error{"cannot open benchmark output: " + path};
  }
  stream << document.dump(2) << '\n';
  stream.flush();
  if (!stream) {
    throw std::runtime_error{"cannot write benchmark output: " + path};
  }
}

}  // namespace
}  // namespace lunar::planning::benchmark

int main(int argc, char** argv) {
  try {
    const auto arguments =
        lunar::planning::benchmark::ParseArguments(argc, argv);
    lunar::planning::benchmark::Write(
        arguments.output_path,
        lunar::planning::benchmark::Run(arguments.case_filter));
  } catch (const std::exception& error) {
    std::cerr << "hierarchical planner benchmark error: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
