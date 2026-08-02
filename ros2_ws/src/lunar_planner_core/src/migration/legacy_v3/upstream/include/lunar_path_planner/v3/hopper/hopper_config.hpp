#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "lunar_path_planner/v3/contracts/planning_request.hpp"
#include "lunar_path_planner/v3/contracts/profiles.hpp"
#include "lunar_path_planner/v3/hopper/landing_geometry.hpp"

namespace lunar::planning::v3 {

// These classes complete the request contract's existing forward
// declarations. They are process-local immutable registry objects, not
// additional wire DTOs.
class GravityModel final {
 public:
  ContentRef content_ref;
  FrameId frame_id;
  Eigen::Vector3d nominal_acceleration_mps2{
      Eigen::Vector3d::Zero()};
  AxisAlignedBox3 acceleration_error_mps2;
  AxisAlignedBox3 spatial_validity_m;
  ClockStamp valid_from;
  ClockStamp valid_until;
};

class DeterministicErrorModel final {
 public:
  ContentRef content_ref;
  AxisAlignedBox3 initial_position_error_m;
  AxisAlignedBox3 initial_velocity_error_mps;
  AxisAlignedBox3 launch_execution_velocity_error_mps;
  AxisAlignedBox3 gravity_error_mps2;
  AxisAlignedBox3 landing_plane_origin_error_m;
  RotationVectorBall landing_plane_normal_error;
  SymmetricScalarInterval landing_plane_residual_error_m;
};

class ActuatorOrImpulseProfile final {
 public:
  ContentRef content_ref;
  double platform_mass_kg{};
  DurationNanoseconds launch_preparation_time;
  DurationNanoseconds landing_settle_time;
  double nominal_landing_center_normal_offset_m{};
};

class BodyRotationEnvelope final {
 public:
  ContentRef content_ref;
  ConvexPolytope3 arbitrary_attitude_body_envelope;
};

struct AttitudeTighteningEntry final {
  double maximum_rotation_angle_rad{};
  double maximum_angular_speed_radps{};
  double maximum_angular_acceleration_radps2{};
};

class AttitudeTighteningTable final {
 public:
  ContentRef content_ref;
  std::vector<AttitudeTighteningEntry> entries;
};

struct HopperPlannerLimits final {
  enum class FutureRouteAuthority {
    kMissionPreviewOnly,
  };

  std::size_t yaw_partition_count{};
  std::size_t support_direction_count{};
  std::size_t maximum_landing_regions{};
  std::size_t landing_region_maximum_vertices{};
  std::size_t landing_region_inflation_iterations{};
  std::size_t landing_region_maximum_split_depth{};
  std::size_t maximum_graph_nodes{};
  std::size_t maximum_graph_out_degree{};
  std::size_t maximum_nominal_aim_points_per_region{};
  std::size_t maximum_full_certification_attempts{};
  std::size_t maximum_interval_subdivision_depth{};
  std::size_t maximum_root_iterations{};
  std::size_t maximum_collision_subdivision_depth{};
  std::size_t maximum_flight_tube_sections{};
  DurationNanoseconds time_equivalence_tolerance;
  AraStarConfig ara_star;
  FutureRouteAuthority future_route_authority{
      FutureRouteAuthority::kMissionPreviewOnly};
};

struct HopperCapabilityView final {
  ContentRef source_safety_capability_ref;
  FrameId frame_id;
  BodyConvexPolytope collision_envelope;
  ConvexPolygon2d landing_footprint_body_xy;
  HopperLandingTerrainThresholds landing_terrain_thresholds;
  HopperLaunchLimits launch_limits;
  ArbitraryAxisAttitudeEnvelope attitude_envelope;
  HopperErrorBounds certified_state_error_bounds;
  GravityModel gravity_model;
  DeterministicErrorModel error_model;
  ActuatorOrImpulseProfile actuator_or_impulse_profile;
  BodyRotationEnvelope body_rotation_envelope;
  std::optional<AttitudeTighteningTable> attitude_tightening_table;
};

[[nodiscard]] Result<HopperCapabilityView> bind_hopper_capability(
    const SafetyCapabilityProfile& profile,
    const ResolvedCapabilityBindings& bindings);
[[nodiscard]] Result<HopperPlannerLimits> bind_hopper_limits(
    const PlannerAlgorithmConfig& config);

}  // namespace lunar::planning::v3
