#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "lunar_path_planner/v3/contracts/base_types.hpp"
#include "lunar_path_planner/v3/contracts/state.hpp"

namespace lunar::planning::v3 {

struct ExtrudedConvexFootprint final {
  enum class Shape {
    kExtrudedConvexPolygon,
  };
  enum class Winding {
    kCcw,
  };

  Shape shape{Shape::kExtrudedConvexPolygon};
  std::vector<Vec2> vertices_xy_m;
  Winding winding{Winding::kCcw};
  double minimum_z_m{};
  double maximum_z_m{};
};

struct BodyConvexPolytope final {
  enum class Shape {
    kConvexPolytope,
  };

  Shape shape{Shape::kConvexPolytope};
  ConvexPolytope3 body_frame_halfspaces;
};

struct WheelHardLimits final {
  double maximum_forward_speed_mps{};
  double maximum_reverse_speed_mps{};
  double maximum_spin_rate_radps{};
  double maximum_forward_acceleration_mps2{};
  double maximum_braking_deceleration_mps2{};
  double maximum_yaw_acceleration_radps2{};
  double maximum_lateral_acceleration_mps2{};
  double maximum_drive_curvature_per_m{};
  double maximum_slope_rad{};
  double minimum_clearance_m{};
};

struct WheelMotionPrimitive final {
  enum class Kind {
    kDriveForwardLine,
    kDriveForwardArc,
    kDriveReverseLine,
    kDriveReverseArc,
    kSpinCw,
    kSpinCcw,
    kStopAndSwitch,
  };

  PrimitiveId primitive_id;
  Kind kind{Kind::kDriveForwardLine};
  PoseXyzYaw relative_end_pose;
  DurationNanoseconds nominal_duration;
  ContentRef swept_geometry_ref;
};

struct WheeledCapability final {
  FrameId frame_id;
  ExtrudedConvexFootprint collision_envelope;
  ContentRef motion_model_ref;
  ContentRef analytic_cost_model_ref;
  WheelHardLimits hard_limits;
  WheeledOrLeggedErrorBounds certified_state_error_bounds;
  std::vector<WheelMotionPrimitive> motion_primitives;
};

struct LeggedTerrainThresholds final {
  double maximum_slope_rad{};
  double maximum_roughness_m{};
  double maximum_step_height_m{};
  double maximum_gap_width_m{};
  double minimum_confidence{};
  double minimum_body_clearance_m{};
  double minimum_body_height_m{};
  double maximum_body_height_m{};
};

struct LeggedBodyVelocityLimits final {
  ScalarBounds forward_mps;
  ScalarBounds lateral_mps;
  ScalarBounds vertical_mps;
  ScalarBounds yaw_rate_radps;
  double linear_acceleration_mps2{};
  double yaw_acceleration_radps2{};
};

struct LeggedBodyPrimitive final {
  enum class Kind {
    kForward,
    kBackward,
    kLateral,
    kDiagonal,
    kSpin,
    kCoupled,
  };

  PrimitiveId primitive_id;
  Kind kind{Kind::kForward};
  Vec3 body_frame_displacement_m;
  double yaw_change_rad{};
  DurationNanoseconds nominal_duration;
  ContentRef sampled_body_sweep_ref;
};

struct LeggedCapability final {
  enum class ReferencePointDefinition {
    kFixedBodyFramePoint,
    kFixedNominalCom,
  };
  enum class FeasibilityScope {
    kBodyGeometryAndTerrainThresholdsOnly,
  };

  FrameId frame_id;
  ReferencePointId reference_point_id;
  ReferencePointDefinition reference_point_definition{
      ReferencePointDefinition::kFixedBodyFramePoint};
  BodyConvexPolytope collision_envelope;
  ContentRef motion_model_ref;
  ContentRef analytic_cost_model_ref;
  LeggedTerrainThresholds terrain_thresholds;
  LeggedBodyVelocityLimits body_velocity_limits;
  WheeledOrLeggedErrorBounds certified_state_error_bounds;
  std::vector<LeggedBodyPrimitive> motion_primitives;
  FeasibilityScope feasibility_scope{
      FeasibilityScope::kBodyGeometryAndTerrainThresholdsOnly};
  bool footstep_feasibility_guaranteed{false};
};

struct HopperLandingTerrainThresholds final {
  double maximum_slope_rad{};
  double maximum_roughness_m{};
  double maximum_plane_residual_m{};
  double minimum_overhead_clearance_m{};
  double minimum_lateral_clearance_m{};
  double minimum_landing_region_area_m2{};
};

struct HopperLaunchLimits final {
  double maximum_launch_speed_mps{};
  double maximum_launch_impulse_newton_seconds{};
  DurationNanoseconds minimum_flight_time;
  DurationNanoseconds maximum_flight_time;
  double maximum_landing_speed_mps{};
  double minimum_downward_impact_speed_mps{};
  double minimum_landing_clearance_m{};
};

struct ArbitraryAxisAttitudeEnvelope final {
  enum class CapabilityKind {
    kCertifiedConservativeArbitraryAxis,
  };

  CapabilityKind capability_kind{
      CapabilityKind::kCertifiedConservativeArbitraryAxis};
  double maximum_angular_speed_radps{};
  double maximum_angular_acceleration_radps2{};
  double maximum_initial_angular_speed_radps{};
  DurationNanoseconds minimum_settle_guard;
};

struct HopperCapability final {
  enum class TranslationModel {
    kPureBallisticNoInflightTranslationControl,
  };
  enum class AttitudeTighteningAuthority {
    kTightenOnly,
  };

  inline static constexpr AttitudeTighteningAuthority
      kAttitudeTighteningAuthority{
          AttitudeTighteningAuthority::kTightenOnly};

  FrameId frame_id;
  BodyConvexPolytope collision_envelope;
  ContentRef motion_model_ref;
  ContentRef analytic_cost_model_ref;
  ContentRef gravity_model_ref;
  HopperLandingTerrainThresholds landing_terrain_thresholds;
  HopperLaunchLimits launch_limits;
  ArbitraryAxisAttitudeEnvelope attitude_envelope;
  std::optional<ContentRef> attitude_tightening_table_ref;
  HopperErrorBounds certified_state_error_bounds;
  TranslationModel translation_model{
      TranslationModel::kPureBallisticNoInflightTranslationControl};
};

using SafetyCapabilityContent =
    std::variant<WheeledCapability, LeggedCapability, HopperCapability>;

struct SafetyCapabilityProfile final {
  ContentRef content_ref;
  SafetyCapabilityContent content;
};

struct ResourceCaps final {
  std::size_t maximum_expanded_states{};
  std::size_t maximum_reopened_states{};
  std::size_t maximum_generated_candidates{};
  std::size_t maximum_open_states{};
  std::size_t maximum_memory_bytes{};
};

struct AraStarConfig final {
  enum class TerminationPolicy {
    kTargetEpsilonGraphExhaustionOrResourceCap,
  };
  enum class TieBreaking {
    kDeterministicLexicographicStateKey,
  };

  double initial_epsilon{};
  double epsilon_decrement{};
  double target_epsilon{};
  ResourceCaps resource_caps;
  TerminationPolicy termination_policy{
      TerminationPolicy::kTargetEpsilonGraphExhaustionOrResourceCap};
  TieBreaking tie_breaking{
      TieBreaking::kDeterministicLexicographicStateKey};
};

struct CorridorConfig final {
  enum class FailurePolicy {
    kFallBackToValidatedDiscretePrimitives,
  };

  std::size_t maximum_regions{};
  std::size_t maximum_inflation_iterations{};
  std::size_t maximum_halfplanes_per_region{};
  std::size_t maximum_split_depth{};
  double minimum_overlap_m{};
  double sampling_spacing_m{};
  FailurePolicy failure_policy{
      FailurePolicy::kFallBackToValidatedDiscretePrimitives};
};

struct SmoothingConfig final {
  enum class FailurePolicy {
    kRejectWholeSmoothedSegment,
  };

  std::size_t maximum_scp_iterations{};
  std::size_t maximum_trust_region_reductions{};
  double initial_trust_region_m{};
  double minimum_trust_region_m{};
  double constraint_tolerance{};
  DurationNanoseconds maximum_time_increase;
  FailurePolicy failure_policy{
      FailurePolicy::kRejectWholeSmoothedSegment};
};

struct TimeScalingConfig final {
  std::size_t maximum_adaptive_samples{};
  double minimum_parameter_step{};
  std::size_t maximum_forward_passes{};
  std::size_t maximum_backward_passes{};
  bool enable_jerk_smoothing{};
  std::size_t maximum_jerk_smoothing_iterations{};
};

struct GridConfig final {
  double xy_resolution_m{};
  std::size_t yaw_bin_count{};
  std::size_t maximum_terminal_candidates{};
};

struct WheeledAlgorithmConfig final {
  GridConfig state_lattice;
  CorridorConfig corridor;
  SmoothingConfig smoothing;
  TimeScalingConfig time_scaling;
  std::size_t continuous_validation_maximum_subdivisions{};
};

struct LeggedAlgorithmConfig final {
  GridConfig pose_lattice;
  std::size_t maximum_height_interval_splits{};
  CorridorConfig corridor;
  SmoothingConfig smoothing;
  TimeScalingConfig time_scaling;
  std::size_t continuous_validation_maximum_subdivisions{};
};

struct HopperAlgorithmConfig final {
  enum class FutureRouteAuthority {
    kMissionPreviewOnly,
  };

  std::size_t maximum_landing_regions{};
  std::size_t maximum_graph_nodes{};
  std::size_t maximum_graph_out_degree{};
  std::size_t yaw_partition_count{};
  std::size_t support_direction_count{};
  std::size_t maximum_nominal_aim_points_per_region{};
  std::size_t maximum_full_certification_attempts{};
  std::size_t maximum_interval_subdivision_depth{};
  std::size_t maximum_collision_subdivision_depth{};
  std::size_t maximum_root_iterations{};
  std::size_t maximum_flight_tube_sections{};
  std::size_t landing_region_inflation_iterations{};
  std::size_t landing_region_maximum_split_depth{};
  std::size_t landing_region_maximum_vertices{};
  FutureRouteAuthority future_route_authority{
      FutureRouteAuthority::kMissionPreviewOnly};
};

struct LearnedCostPolicy final {
  enum class Mode {
    kDisabled,
    kOptionalBoundedSoftCost,
  };
  enum class FailurePolicy {
    kFallBackToAnalyticCost,
  };
  enum class HardFeasibilityAuthority {
    kNone,
  };

  Mode mode{Mode::kDisabled};
  std::size_t maximum_inference_evaluations{};
  double maximum_absolute_energy_correction{};
  double maximum_absolute_nonfatal_risk_correction{};
  FailurePolicy failure_policy{
      FailurePolicy::kFallBackToAnalyticCost};
  HardFeasibilityAuthority hard_feasibility_authority{
      HardFeasibilityAuthority::kNone};
};

struct DeterministicExecutionConfig final {
  std::size_t fixed_thread_count{};
  bool stable_candidate_order{true};
  bool preallocated_memory_pools{true};
};

struct PlannerAlgorithmConfig final {
  ContentRef content_ref;
  DurationNanoseconds time_equivalence_tolerance;
  DurationNanoseconds max_input_skew;
  std::string error_bound_model_id;
  std::size_t projection_cache_capacity{};
  AraStarConfig ara_star;
  WheeledAlgorithmConfig wheeled;
  LeggedAlgorithmConfig legged;
  HopperAlgorithmConfig hopper;
  LearnedCostPolicy learned_cost_policy;
  std::optional<ContentRef> learned_cost_model_ref;
  DeterministicExecutionConfig deterministic_execution;
};

}  // namespace lunar::planning::v3
