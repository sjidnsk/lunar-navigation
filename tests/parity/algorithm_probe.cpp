#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(PLANNER_VARIANT_OLD)
#include "lunar_planner_core/planner.hpp"
namespace planner_api = lunar::planning;
#elif defined(PLANNER_VARIANT_PURE)
#include "lunar_pure_planner_core/planner.hpp"
namespace planner_api = lunar::pure_planning;
#else
#error "one planner variant must be selected"
#endif

namespace {

using Json = nlohmann::json;
using namespace planner_api;

constexpr std::int64_t kFixtureTimeNs = 1'000'000'000;

[[nodiscard]] Json LoadFixture(const std::string& path) {
  std::ifstream stream{path};
  if (!stream) {
    throw std::runtime_error{"cannot open fixture: " + path};
  }
  Json fixture = Json::parse(stream);
  if (fixture.at("schema_version") != "pure-planner-canonical-fixture/v1") {
    throw std::runtime_error{"unsupported fixture schema"};
  }
  return fixture;
}

[[nodiscard]] Vec3 ReadVec3(const Json& value) {
  if (!value.is_array() || value.size() != 3U) {
    throw std::runtime_error{"expected three-element vector"};
  }
  return Vec3{value.at(0).get<double>(), value.at(1).get<double>(),
              value.at(2).get<double>()};
}

[[nodiscard]] Vec2 ReadVec2(const Json& value) {
  if (!value.is_array() || value.size() != 2U) {
    throw std::runtime_error{"expected two-element vector"};
  }
  return Vec2{value.at(0).get<double>(), value.at(1).get<double>()};
}

[[nodiscard]] Interval ReadInterval(const Json& value) {
  if (!value.is_array() || value.size() != 2U) {
    throw std::runtime_error{"expected two-element interval"};
  }
  return Interval{value.at(0).get<double>(), value.at(1).get<double>()};
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw_rad) {
  return Quaternion{.w = std::cos(yaw_rad / 2.0),
                    .z = std::sin(yaw_rad / 2.0)};
}

[[nodiscard]] Pose3 ReadPose(const Json& value) {
  return Pose3{.position_m = ReadVec3(value.at("position_m")),
               .orientation = YawQuaternion(value.at("yaw_rad").get<double>())};
}

[[nodiscard]] GridLayer FloatLayer(
    const std::size_t count, const float value) {
  return GridLayer{.values = std::vector<float>(count, value)};
}

[[nodiscard]] GridLayer ByteLayer(
    const std::size_t count, const std::uint8_t value) {
  return GridLayer{.values = std::vector<std::uint8_t>(count, value)};
}

[[nodiscard]] GridLayer CountLayer(
    const std::size_t count, const std::uint32_t value) {
  return GridLayer{.values = std::vector<std::uint32_t>(count, value)};
}

[[nodiscard]] GridMap MakeMap(
    const Json& map_definition, const Json& common, std::string frame_id) {
  const std::size_t width = map_definition.at("width").get<std::size_t>();
  const std::size_t height = map_definition.at("height").get<std::size_t>();
  const std::size_t cell_count = width * height;
  std::map<std::string, GridLayer, std::less<>> layers;
  layers.emplace("elevation", FloatLayer(
      cell_count, common.at("elevation_m").get<float>()));
  layers.emplace("valid_mask", ByteLayer(cell_count, 1U));
  layers.emplace("obstacle", ByteLayer(cell_count, 0U));
  layers.emplace("obstacle_height", FloatLayer(cell_count, 0.0F));
  layers.emplace("observation_age_s", FloatLayer(cell_count, 0.0F));
  layers.emplace("observation_quality", FloatLayer(cell_count, 1.0F));
  layers.emplace("elevation_variance", FloatLayer(cell_count, 0.0F));
  layers.emplace("obstacle_variance", FloatLayer(cell_count, 0.0F));
  layers.emplace("observation_count", CountLayer(cell_count, 1U));
  layers.emplace("forbidden", ByteLayer(cell_count, 0U));
  auto& forbidden = std::get<std::vector<std::uint8_t>>(
      layers.at("forbidden").values);
  for (const Json& index_value : common.at("forbidden_indices")) {
    const auto index = index_value.get<std::size_t>();
    if (index >= cell_count) {
      throw std::runtime_error{"forbidden index outside map"};
    }
    forbidden[index] = 1U;
  }
  return GridMap{
      .frame_id = std::move(frame_id),
      .stamp = TimePoint{.nanoseconds_since_epoch = kFixtureTimeNs},
      .width = width,
      .height = height,
      .resolution_m = map_definition.at("resolution_m").get<double>(),
      .origin_m = ReadVec3(map_definition.at("origin_m")),
      .layers = std::move(layers),
  };
}

[[nodiscard]] WheelPrimitiveKind ReadWheelKind(const std::string& kind) {
  if (kind == "forward") return WheelPrimitiveKind::kForward;
  if (kind == "reverse") return WheelPrimitiveKind::kReverse;
  if (kind == "forward_arc") return WheelPrimitiveKind::kForwardArc;
  if (kind == "reverse_arc") return WheelPrimitiveKind::kReverseArc;
  if (kind == "spin_clockwise") return WheelPrimitiveKind::kSpinClockwise;
  if (kind == "spin_counterclockwise") {
    return WheelPrimitiveKind::kSpinCounterclockwise;
  }
  if (kind == "stop_and_switch") return WheelPrimitiveKind::kStopAndSwitch;
  throw std::runtime_error{"unknown wheel primitive kind: " + kind};
}

[[nodiscard]] LeggedPrimitiveKind ReadLeggedKind(const std::string& kind) {
  if (kind == "forward") return LeggedPrimitiveKind::kForward;
  if (kind == "backward") return LeggedPrimitiveKind::kBackward;
  if (kind == "lateral_left") return LeggedPrimitiveKind::kLateralLeft;
  if (kind == "lateral_right") return LeggedPrimitiveKind::kLateralRight;
  if (kind == "spin") return LeggedPrimitiveKind::kSpin;
  if (kind == "coupled") return LeggedPrimitiveKind::kCoupled;
  throw std::runtime_error{"unknown legged primitive kind: " + kind};
}

[[nodiscard]] WheeledCapability MakeWheelCapability(const Json& value) {
  std::vector<Vec2> footprint;
  for (const Json& point : value.at("footprint_xy_m")) {
    footprint.push_back(ReadVec2(point));
  }
  std::vector<WheelMotionPrimitive> primitives;
  for (const Json& primitive : value.at("motion_primitives")) {
    primitives.push_back(WheelMotionPrimitive{
        .primitive_id = primitive.at("id").get<std::string>(),
        .kind = ReadWheelKind(primitive.at("kind").get<std::string>()),
        .relative_end_pose = Pose3{
            .position_m = ReadVec3(primitive.at("position_m")),
            .orientation = YawQuaternion(primitive.at("yaw_rad").get<double>()),
        },
    });
  }
  return WheeledCapability{
      .footprint_xy_m = std::move(footprint),
      .body_extent_m = ReadVec3(value.at("body_extent_m")),
      .wheel_diameter_m = value.at("wheel_diameter_m").get<double>(),
      .wheel_width_m = value.at("wheel_width_m").get<double>(),
      .wheelbase_m = value.at("wheelbase_m").get<double>(),
      .track_width_m = value.at("track_width_m").get<double>(),
      .minimum_underbody_clearance_m = value.at("minimum_underbody_clearance_m").get<double>(),
      .maximum_local_obstacle_relief_m = value.at("maximum_local_obstacle_relief_m").get<double>(),
      .allow_unsupported_gap = false,
      .minimum_body_z_m = value.at("minimum_body_z_m").get<double>(),
      .maximum_body_z_m = value.at("maximum_body_z_m").get<double>(),
      .maximum_forward_speed_mps = value.at("maximum_forward_speed_mps").get<double>(),
      .maximum_reverse_speed_mps = value.at("maximum_reverse_speed_mps").get<double>(),
      .maximum_spin_rate_radps = value.at("maximum_spin_rate_radps").get<double>(),
      .maximum_acceleration_mps2 = value.at("maximum_acceleration_mps2").get<double>(),
      .maximum_braking_deceleration_mps2 = value.at("maximum_braking_deceleration_mps2").get<double>(),
      .maximum_yaw_acceleration_radps2 = value.at("maximum_yaw_acceleration_radps2").get<double>(),
      .maximum_lateral_acceleration_mps2 = value.at("maximum_lateral_acceleration_mps2").get<double>(),
      .maximum_curvature_per_m = value.at("maximum_curvature_per_m").get<double>(),
      .maximum_slope_rad = value.at("maximum_slope_rad").get<double>(),
      .minimum_clearance_m = value.at("minimum_clearance_m").get<double>(),
      .motion_primitives = std::move(primitives),
  };
}

[[nodiscard]] LeggedCapability MakeLeggedCapability(const Json& value) {
  std::vector<LeggedBodyPrimitive> primitives;
  for (const Json& primitive : value.at("motion_primitives")) {
    primitives.push_back(LeggedBodyPrimitive{
        .primitive_id = primitive.at("id").get<std::string>(),
        .kind = ReadLeggedKind(primitive.at("kind").get<std::string>()),
        .body_frame_displacement_m = ReadVec3(primitive.at("displacement_m")),
        .yaw_change_rad = primitive.at("yaw_rad").get<double>(),
    });
  }
  return LeggedCapability{
      .body_extent_m = ReadVec3(value.at("body_extent_m")),
      .platform_mass_kg = value.at("platform_mass_kg").get<double>(),
      .maximum_payload_kg = value.at("maximum_payload_kg").get<double>(),
      .maximum_slope_rad = value.at("maximum_slope_rad").get<double>(),
      .maximum_step_height_m = value.at("maximum_step_height_m").get<double>(),
      .maximum_gap_width_m = value.at("maximum_gap_width_m").get<double>(),
      .minimum_body_clearance_m = value.at("minimum_body_clearance_m").get<double>(),
      .step_vertical_rate_mps = value.at("step_vertical_rate_mps").get<double>(),
      .body_height_m = ReadInterval(value.at("body_height_m")),
      .forward_speed_mps = ReadInterval(value.at("forward_speed_mps")),
      .lateral_speed_mps = ReadInterval(value.at("lateral_speed_mps")),
      .yaw_rate_radps = ReadInterval(value.at("yaw_rate_radps")),
      .maximum_linear_acceleration_mps2 = value.at("maximum_linear_acceleration_mps2").get<double>(),
      .maximum_yaw_acceleration_radps2 = value.at("maximum_yaw_acceleration_radps2").get<double>(),
      .motion_primitives = std::move(primitives),
  };
}

[[nodiscard]] HopperCapability MakeHopperCapability(const Json& value) {
  return HopperCapability{
      .specific_impulse_s = value.at("specific_impulse_s").get<double>(),
      .reference_total_mass_kg = value.at("reference_total_mass_kg").get<double>(),
      .reference_propellant_mass_kg = value.at("reference_propellant_mass_kg").get<double>(),
      .gravity_mps2 = ReadVec3(value.at("gravity_mps2")),
      .reference_horizontal_range_m = value.at("reference_horizontal_range_m").get<double>(),
      .landing_support_radius_m = value.at("landing_support_radius_m").get<double>(),
      .flight_collision_radius_m = value.at("flight_collision_radius_m").get<double>(),
      .maximum_landing_plane_residual_m = value.at("maximum_landing_plane_residual_m").get<double>(),
      .landing_lateral_margin_m = value.at("landing_lateral_margin_m").get<double>(),
      .flight_map_margin_m = value.at("flight_map_margin_m").get<double>(),
      .reachability_delta_v_margin_ratio = value.at("reachability_delta_v_margin_ratio").get<double>(),
      .standard_gravity_mps2 = value.at("standard_gravity_mps2").get<double>(),
      .maximum_landing_slope_rad = value.at("maximum_landing_slope_rad").get<double>(),
  };
}

[[nodiscard]] PlannerInput MakeInput(const Json& fixture) {
  const std::string platform = fixture.at("platform").get<std::string>();
  const Json& start = fixture.at("start");
  const Pose3 start_pose = ReadPose(start);
  PlatformState state;
  PlatformCapability capability;
  if (platform == "wheel") {
    state = WheeledState{.pose = start_pose};
    capability = MakeWheelCapability(fixture.at("capability"));
  } else if (platform == "legged") {
    state = LeggedState{.body_pose = start_pose};
    capability = MakeLeggedCapability(fixture.at("capability"));
  } else if (platform == "hopper") {
    state = HopperState{.pose = start_pose};
    capability = MakeHopperCapability(fixture.at("capability"));
  } else {
    throw std::runtime_error{"unsupported platform: " + platform};
  }

  const Json& map = fixture.at("map");
  PlannerConfig config;
  const Json& config_fixture = fixture.at("planner_config");
  config.wheel.xy_resolution_m = config_fixture.at("xy_resolution_m").get<double>();
  config.legged.xy_resolution_m = config_fixture.at("xy_resolution_m").get<double>();
  config.global_map.base_resolution_m = config_fixture.at("global_base_resolution_m").get<double>();
  config.stable_candidate_order = config_fixture.at("stable_candidate_order").get<bool>();

  return PlannerInput{
      .request_id = "pure-parity-" + platform,
      .mission_id = "pure-parity-mission",
      .mission_revision = 1U,
      .platform_id = "pure-parity-" + platform,
      .capability_version = "pure-parity-capability-v1",
      .global_map_generation = 1U,
      .local_map_generation = 1U,
      .map_from_odom_generation = 1U,
      .state_time = TimePoint{.nanoseconds_since_epoch = kFixtureTimeNs},
      .current_state = std::move(state),
      .goal_map = GoalRegion{
          .goal_id = "pure-parity-goal",
          .target = PointGoal{
              .position_m = ReadVec3(fixture.at("goal").at("position_m")),
              .tolerance_m = fixture.at("goal").at("tolerance_m").get<double>(),
          },
          .yaw_rad = std::nullopt,
          .yaw_tolerance_rad = 0.0,
      },
      .world = WorldSnapshot{
          .global_map = MakeMap(map.at("global"), map, "map"),
          .local_map = MakeMap(map.at("local"), map, "odom"),
          .map_from_odom = RigidTransform{
              .parent_frame = "map",
              .child_frame = "odom",
              .stamp = TimePoint{.nanoseconds_since_epoch = kFixtureTimeNs},
          },
      },
      .capability = std::move(capability),
      .config = config,
      .position_uncertainty_m = 0.05,
      .velocity_uncertainty_mps = 0.02,
  };
}

[[nodiscard]] Json SerializeVec3(const Vec3& value) {
  return Json::array({value.x, value.y, value.z});
}

[[nodiscard]] Json NormalizeOutput(
    const PlannerOutput& output, const PlannerInput& input,
    const std::string& platform) {
  Json global_path = Json::array();
  Json local_trajectory = Json::array();
  Json hop_segments = Json::array();
  Json reference_platform = nullptr;
  std::string reference_kind = "none";
  if (output.reference.has_value()) {
    reference_platform = static_cast<int>(output.reference->platform_type);
    for (const Pose3& pose : output.reference->preview.poses_map) {
      global_path.push_back(SerializeVec3(pose.position_m));
    }
    if (const auto* trajectory =
            std::get_if<TrajectoryReference>(&output.reference->data)) {
      reference_kind = trajectory->semantics == TrajectorySemantics::kWheeledBase
                           ? "wheeled_trajectory"
                           : "legged_trajectory";
      for (const TrajectoryPoint& point : trajectory->points) {
        local_trajectory.push_back(SerializeVec3(point.pose.position_m));
      }
    } else if (const auto* hops =
                   std::get_if<HopReference>(&output.reference->data)) {
      reference_kind = "certified_hops";
      for (const HopSegment& segment : hops->segments) {
        hop_segments.push_back(Json{
            {"launch_position_m", SerializeVec3(segment.launch_pose.position_m)},
            {"nominal_landing_point_m", SerializeVec3(segment.nominal_landing_point_m)},
            {"flight_time_ns", segment.flight_time.count()},
            {"launch_velocity_mps", SerializeVec3(segment.launch_velocity_mps)},
            {"required_delta_v_mps", segment.required_delta_v_mps},
            {"available_delta_v_mps", segment.available_delta_v_mps},
        });
      }
    }
  }

  Json total_cost = nullptr;
  if (output.diagnostics.best_cost.has_value()) {
    total_cost = *output.diagnostics.best_cost;
  }
  Json search_domain_sha256 = nullptr;
  std::uint64_t global_expanded = 0U;
  std::uint64_t local_expanded = 0U;
  if (output.diagnostics.hierarchical.has_value()) {
    search_domain_sha256 =
        output.diagnostics.hierarchical->search_domain_sha256;
    global_expanded = output.diagnostics.hierarchical->global_expanded_states;
    local_expanded = output.diagnostics.hierarchical->local_expanded_states;
  }
  Json trajectory_mode = nullptr;
  if (output.diagnostics.local_trajectory.has_value()) {
    trajectory_mode = std::string{ToString(
        output.diagnostics.local_trajectory->trajectory_mode)};
  }

  return Json{
      {"schema_version", "pure-planner-algorithm-probe/v1"},
      {"platform", platform},
      {"outcome", static_cast<int>(output.outcome)},
      {"reason_code", output.reason_code},
      {"global_path_points", std::move(global_path)},
      {"local_trajectory_points", std::move(local_trajectory)},
      {"hop_segments", std::move(hop_segments)},
      {"total_cost", std::move(total_cost)},
      {"expanded_nodes", output.diagnostics.expanded_states},
      {"determinism_summary",
       Json{
           {"execution_directive", static_cast<int>(output.directive)},
           {"candidate_disposition", static_cast<int>(output.candidate_disposition)},
           {"reference_platform", reference_platform},
           {"reference_kind", reference_kind},
           {"stable_candidate_order", input.config.stable_candidate_order},
           {"search_domain_sha256", std::move(search_domain_sha256)},
           {"global_expanded_nodes", global_expanded},
           {"local_expanded_nodes", local_expanded},
           {"trajectory_mode", trajectory_mode},
       }},
  };
}

void Run(const std::string& fixture_path, const std::string& output_path) {
  const Json fixture = LoadFixture(fixture_path);
  const PlannerInput input = MakeInput(fixture);
  Planner planner;
  const PlannerOutput output = planner.Plan(input);
  const Json normalized = NormalizeOutput(
      output, input, fixture.at("platform").get<std::string>());
  std::ofstream stream{output_path};
  if (!stream) {
    throw std::runtime_error{"cannot open output: " + output_path};
  }
  stream << normalized.dump(2) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: algorithm_probe FIXTURE OUTPUT\n";
    return 2;
  }
  try {
    Run(argv[1], argv[2]);
  } catch (const std::exception& error) {
    std::cerr << "algorithm probe error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
