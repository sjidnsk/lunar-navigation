#pragma once

#include <array>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/contracts/planning_request.hpp"

namespace lunar::planning::v3 {

struct TimeInterval final {
  DurationNanoseconds start_offset;
  DurationNanoseconds end_offset;
};

struct CubicPolynomialSegment final {
  DurationNanoseconds start_offset;
  DurationNanoseconds end_offset;
  std::array<double, 4> coefficients{};
};

struct PiecewiseCubicScalarTrajectory final {
  std::string value_semantics;
  std::vector<CubicPolynomialSegment> segments;
};

struct MonotoneTimeScaling final {
  std::vector<CubicPolynomialSegment> segments;
};

struct DerivedKinematicCaches final {
  double consistency_tolerance{};
  PiecewiseCubicScalarTrajectory signed_body_forward_speed_mps;
  PiecewiseCubicScalarTrajectory yaw_rate_radps;
};

struct ClampedCubicBSplinePath final {
  std::vector<double> knots;
  std::vector<PoseXyzYaw> control_points;
};

enum class PrimitiveKind {
  kDriveForward,
  kDriveReverse,
  kSpinCw,
  kSpinCcw,
  kStopAndSwitch,
  kBodyTranslation,
  kBodySpin,
  kBodyCoupled,
};

struct ValidatedPrimitive final {
  PrimitiveId primitive_id;
  PrimitiveId capability_primitive_id;
  PrimitiveKind primitive_kind{};
  PoseXyzYaw start_pose;
  PoseXyzYaw end_pose;
  DurationNanoseconds nominal_duration;
  ContentRef validation_ref;
};

struct ValidatedPrimitiveChain final {
  std::vector<ValidatedPrimitive> primitives;
};

using GeometricPath =
    std::variant<ClampedCubicBSplinePath, ValidatedPrimitiveChain>;

enum class DriveDirection {
  kForward,
  kReverse,
};

struct DriveSegment final {
  SegmentId segment_id;
  TimeInterval time_interval;
  DriveDirection direction{};
  GeometricPath geometric_path;
  MonotoneTimeScaling time_scaling;
  std::optional<DerivedKinematicCaches> derived_caches;
};

struct SpinSegment final {
  SegmentId segment_id;
  TimeInterval time_interval;
  Vec3 fixed_position_m;
  PiecewiseCubicScalarTrajectory unwrapped_yaw_rad;
};

using WheeledSegment = std::variant<DriveSegment, SpinSegment>;

struct SafeStopAnchor final {
  std::string anchor_id;
  PoseXyzYaw pose;
  double target_linear_velocity_mps{0.0};
  double target_yaw_rate_radps{0.0};
  ContentRef terrain_certification_ref;
};

struct WheeledReference final {
  ReferenceId reference_id;
  Sha256Digest reference_hash;
  ClockStamp reference_time_origin;
  std::vector<WheeledSegment> segments;
  SafeStopAnchor safe_stop_anchor;
};

struct BodyFrameVelocityEnvelope final {
  Interval forward_mps;
  Interval lateral_mps;
  Interval vertical_mps;
  Interval yaw_rate_radps;
};

struct TerrainNormalEnvelope final {
  double maximum_normal_deviation_rad{};
  ContentRef source_terrain_certification_ref;
};

struct RollPitchDiagnosticEnvelope final {
  Interval roll_rad;
  Interval pitch_rad;
};

struct LeggedBodyReference final {
  ReferenceId reference_id;
  Sha256Digest reference_hash;
  ReferencePointId reference_point_id;
  ClockStamp reference_time_origin;
  GeometricPath geometric_path;
  MonotoneTimeScaling time_scaling;
  BodyFrameVelocityEnvelope body_frame_velocity_envelope;
  TerrainNormalEnvelope terrain_normal_envelope;
  RollPitchDiagnosticEnvelope roll_pitch_diagnostic_envelope;
  SafeStopAnchor safe_stop_anchor;
  std::string feasibility_scope{
      "body_geometry_and_terrain_thresholds_only"};
  bool footstep_feasibility_guaranteed{false};
};

struct Vector3Bounds final {
  Vec3 lower;
  Vec3 upper;
};

struct NextLandingRegion final {
  LandingRegionId region_id;
  FrameId frame_id;
  LandingPlane landing_plane;
  ConvexPolygonUv convex_polygon;
  CircularYawInterval allowed_yaw_interval;
  ContentRef terrain_certification_ref;
  double inward_safety_margin_m{};
};

struct HopperKinematicState final {
  Vec3 position_m;
  Quaternion orientation_body_to_frame;
  Vec3 linear_velocity_mps;
  Vec3 angular_velocity_radps;
};

struct GroundHoldAnchor final {
  Identifier anchor_id;
  HopperKinematicState hold_state;
  HopperErrorBounds allowed_hold_state_error_set;
  ContentRef terrain_certification_ref;
};

struct JumpBoundary final {
  enum class LockEvent {
    kJumpBoundaryLock,
  };
  enum class BallisticTimeOrigin {
    kBallisticLaunchEvent,
  };

  JumpBoundaryId boundary_id;
  LockEvent lock_event{LockEvent::kJumpBoundaryLock};
  HopperKinematicState nominal_launch_state;
  HopperErrorBounds allowed_launch_state_error_set;
  ContentRef gravity_model_ref;
  BallisticTimeOrigin ballistic_time_origin{
      BallisticTimeOrigin::kBallisticLaunchEvent};
  DurationNanoseconds ballistic_flight_time;
  ContentRef actuator_or_impulse_profile_ref;
};

struct PredictedLandingFootprint final {
  LandingPlane landing_plane;
  ConvexPolygonUv convex_center_landing_polygon;
  TimeInterval landing_time_window;
  Vector3Bounds landing_velocity_bounds;
  CircularYawInterval landing_yaw_interval;
  ContentRef source_error_model_ref;
  double outer_approximation_margin_m{};
};

struct FlightTubeSection final {
  TimeInterval time_interval;
  ConvexPolytope3 envelope;
};

struct CertifiedFlightTube final {
  FrameId frame_id;
  std::vector<FlightTubeSection> sections;
  ContentRef source_map_snapshot_ref;
  ContentRef body_rotation_envelope_ref;
  ContentRef error_model_ref;
  double minimum_certified_clearance_m{};
};

struct TargetAttitudeSet final {
  Quaternion nominal_orientation_body_to_frame;
  RotationVectorBall orientation_error_set;
  CircularYawInterval allowed_yaw_interval;
};

struct AttitudeBoundary final {
  enum class TranslationAuthority {
    kNone,
  };

  RotationVectorBall initial_orientation_error_set;
  DeterministicVectorSet3 initial_angular_velocity_error_set_radps;
  TargetAttitudeSet target_attitude_set;
  DeterministicVectorSet3 landing_angular_velocity_bounds_radps;
  DurationNanoseconds settle_guard;
  TranslationAuthority center_of_mass_translation_authority{
      TranslationAuthority::kNone};
  ContentRef certification_ref;
};

struct NominalAimPoint final {
  enum class Authority {
    kNonAuthoritativeExplanatory,
  };

  Authority authority{Authority::kNonAuthoritativeExplanatory};
  Vec3 position_m;
};

enum class FutureViability {
  kViable,
  kUnknown,
  kNoCertifiedContinuation,
};

struct FutureRoutePreview final {
  enum class Authority {
    kNonAuthoritativeMissionPreview,
  };

  Authority authority{Authority::kNonAuthoritativeMissionPreview};
  FutureViability future_viability{FutureViability::kUnknown};
  std::string reason_code;
  std::vector<LandingRegionId> candidate_region_ids;
};

struct HopperReference final {
  enum class TranslationModel {
    kPureBallisticNoInflightTranslationControl,
  };

  ReferenceId reference_id;
  Sha256Digest reference_hash;
  ClockStamp reference_time_origin;
  TranslationModel translation_model{
      TranslationModel::kPureBallisticNoInflightTranslationControl};
  GroundHoldAnchor ground_hold_anchor;
  NextLandingRegion next_landing_region;
  JumpBoundary jump_boundary;
  PredictedLandingFootprint predicted_landing_footprint;
  CertifiedFlightTube certified_flight_tube;
  AttitudeBoundary attitude_boundary;
  NominalAimPoint nominal_aim_point;
  ContentRef physical_certification_ref;
  std::optional<FutureRoutePreview> future_route_preview;
};

using PlatformReference =
    std::variant<WheeledReference, LeggedBodyReference, HopperReference>;

}  // namespace lunar::planning::v3
