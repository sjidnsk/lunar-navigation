#include "lunar_path_planner/v3/wheel/wheel_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/crypto/sha256.hpp"

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-8;
constexpr std::size_t kMaximumInternalControlPoints = 128U;
constexpr std::size_t kMaximumInternalCenterlineSamples = 65536U;
constexpr std::size_t kMaximumFootprintCellsPerSample = 4096U;
constexpr std::size_t kFixedQpIterations = 256U;
constexpr double kPathDeviationWeight = 1.0;
constexpr double kSecondDifferenceWeight = 0.1;
constexpr double kMinimumQpTolerance = 1.0e-8;

[[nodiscard]] Error Failure(ErrorCode code,
                            std::string field_path,
                            std::string message) {
  return {
      .code = code,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool CheckedAdd(
    std::chrono::nanoseconds rhs,
    std::chrono::nanoseconds& lhs) noexcept {
  if (rhs.count() < 0 ||
      lhs.count() >
          std::numeric_limits<std::int64_t>::max() -
              rhs.count()) {
    return false;
  }
  lhs += rhs;
  return true;
}

[[nodiscard]] bool DriveMode(WheelMotionMode mode) noexcept {
  return mode == WheelMotionMode::kForward ||
         mode == WheelMotionMode::kReverse;
}

[[nodiscard]] bool SpinMode(WheelMotionMode mode) noexcept {
  return mode == WheelMotionMode::kSpinClockwise ||
         mode == WheelMotionMode::kSpinCounterClockwise;
}

[[nodiscard]] PrimitiveKind PrimitiveKindFor(
    const WheelLatticeEdge& edge,
    WheelMotionMode segment_mode) noexcept {
  switch (edge.primitive_kind) {
    case WheelPrimitiveKind::kModeSwitch:
      return PrimitiveKind::kStopAndSwitch;
    case WheelPrimitiveKind::kSpin:
      return segment_mode == WheelMotionMode::kSpinClockwise
                 ? PrimitiveKind::kSpinCw
                 : PrimitiveKind::kSpinCcw;
    case WheelPrimitiveKind::kDriveLine:
    case WheelPrimitiveKind::kDriveArc:
      return segment_mode == WheelMotionMode::kReverse
                 ? PrimitiveKind::kDriveReverse
                 : PrimitiveKind::kDriveForward;
  }
  return PrimitiveKind::kStopAndSwitch;
}

[[nodiscard]] Result<ValidatedPrimitiveChain>
BuildPrimitiveChain(const WheelDiscreteSegment& segment) {
  if (segment.edges.empty()) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/wheel_plan/segments",
                   "EMPTY_WHEEL_DISCRETE_SEGMENT");
  }
  ValidatedPrimitiveChain chain;
  chain.primitives.reserve(segment.edges.size());
  for (std::size_t index = 0U; index < segment.edges.size();
       ++index) {
    const WheelLatticeEdge& edge = segment.edges[index];
    if (edge.primitive_id.empty() ||
        edge.capability_primitive_id.empty() ||
        edge.validation_ref.id.empty() ||
        edge.transition_time.value.count() <= 0) {
      return Failure(ErrorCode::kInvalidArgument,
                     "/wheel_plan/segments/edges",
                     "INVALID_WHEEL_DISCRETE_EDGE");
    }
    if (index > 0U) {
      const PoseXyzYaw& prior =
          segment.edges[index - 1U].target_pose;
      if (std::hypot(
              prior.position_m.x - edge.source_pose.position_m.x,
              prior.position_m.y - edge.source_pose.position_m.y) >
              kGeometryTolerance ||
          std::abs(prior.position_m.z -
                   edge.source_pose.position_m.z) >
              kGeometryTolerance ||
          std::abs(prior.yaw_rad - edge.source_pose.yaw_rad) >
              kGeometryTolerance) {
        return Failure(ErrorCode::kInvalidArgument,
                       "/wheel_plan/segments/edges",
                       "DISCONTINUOUS_WHEEL_DISCRETE_EDGE");
      }
    }
    chain.primitives.push_back(
        {
            .primitive_id =
                edge.primitive_id + "-instance-" +
                std::to_string(index),
            .capability_primitive_id =
                edge.capability_primitive_id,
            .primitive_kind =
                PrimitiveKindFor(edge, segment.mode),
            .start_pose = edge.source_pose,
            .end_pose = edge.target_pose,
            .nominal_duration = edge.transition_time,
            .validation_ref = edge.validation_ref,
        });
  }
  return chain;
}

struct SwitchWaits final {
  std::chrono::nanoseconds leading{};
  std::chrono::nanoseconds trailing{};
};

[[nodiscard]] Result<SwitchWaits> ModeSwitchWaits(
    const WheelDiscreteSegment& segment) {
  SwitchWaits waits;
  std::size_t first_active = 0U;
  while (first_active < segment.edges.size() &&
         segment.edges[first_active].primitive_kind ==
             WheelPrimitiveKind::kModeSwitch) {
    if (!CheckedAdd(
            segment.edges[first_active].transition_time.value,
            waits.leading)) {
      return Failure(ErrorCode::kNumericalFailure,
                     "/wheel_plan/mode_switch",
                     "WHEEL_MODE_SWITCH_TIME_OVERFLOW");
    }
    ++first_active;
  }
  if (first_active == segment.edges.size()) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/wheel_plan/mode_switch",
                   "WHEEL_SEGMENT_HAS_NO_ACTIVE_PRIMITIVE");
  }
  std::size_t last_active = segment.edges.size();
  while (last_active > first_active &&
         segment.edges[last_active - 1U].primitive_kind ==
             WheelPrimitiveKind::kModeSwitch) {
    --last_active;
    if (!CheckedAdd(
            segment.edges[last_active].transition_time.value,
            waits.trailing)) {
      return Failure(ErrorCode::kNumericalFailure,
                     "/wheel_plan/mode_switch",
                     "WHEEL_MODE_SWITCH_TIME_OVERFLOW");
    }
  }
  return waits;
}

[[nodiscard]] bool ShiftOffset(
    DurationNanoseconds& offset,
    std::chrono::nanoseconds shift) noexcept {
  if (shift.count() < 0 ||
      offset.value.count() >
          std::numeric_limits<std::int64_t>::max() -
              shift.count()) {
    return false;
  }
  offset.value += shift;
  return true;
}

struct AbsoluteScaling final {
  MonotoneTimeScaling scaling;
  TimeInterval interval;
};

[[nodiscard]] Result<AbsoluteScaling> MakeAbsoluteScaling(
    MonotoneTimeScaling local,
    DurationNanoseconds segment_start,
    SwitchWaits waits) {
  if (local.segments.empty()) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/wheel_plan/time_scaling",
                   "EMPTY_WHEEL_TIME_SCALING");
  }
  MonotoneTimeScaling absolute;
  absolute.segments.reserve(
      local.segments.size() +
      (waits.leading.count() > 0 ? 1U : 0U) +
      (waits.trailing.count() > 0 ? 1U : 0U));
  std::chrono::nanoseconds movement_start =
      segment_start.value;
  if (!CheckedAdd(waits.leading, movement_start)) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/wheel_plan/time_scaling",
                   "WHEEL_TIME_SCALING_OVERFLOW");
  }
  if (waits.leading.count() > 0) {
    absolute.segments.push_back(
        {
            .start_offset = segment_start,
            .end_offset =
                DurationNanoseconds{movement_start},
            .coefficients = {0.0, 0.0, 0.0, 0.0},
        });
  }
  for (auto& polynomial : local.segments) {
    if (!ShiftOffset(polynomial.start_offset, movement_start) ||
        !ShiftOffset(polynomial.end_offset, movement_start)) {
      return Failure(ErrorCode::kNumericalFailure,
                     "/wheel_plan/time_scaling",
                     "WHEEL_TIME_SCALING_OVERFLOW");
    }
    absolute.segments.push_back(std::move(polynomial));
  }
  std::chrono::nanoseconds segment_end =
      absolute.segments.back().end_offset.value;
  if (waits.trailing.count() > 0) {
    const std::chrono::nanoseconds wait_start = segment_end;
    if (!CheckedAdd(waits.trailing, segment_end)) {
      return Failure(ErrorCode::kNumericalFailure,
                     "/wheel_plan/time_scaling",
                     "WHEEL_TIME_SCALING_OVERFLOW");
    }
    absolute.segments.push_back(
        {
            .start_offset =
                DurationNanoseconds{wait_start},
            .end_offset = DurationNanoseconds{segment_end},
            .coefficients = {1.0, 0.0, 0.0, 0.0},
        });
  }
  return AbsoluteScaling{
      .scaling = std::move(absolute),
      .interval =
          {
              .start_offset = segment_start,
              .end_offset =
                  DurationNanoseconds{segment_end},
          },
  };
}

struct AbsoluteYaw final {
  PiecewiseCubicScalarTrajectory trajectory;
  TimeInterval interval;
};

[[nodiscard]] Result<AbsoluteYaw> MakeAbsoluteYaw(
    PiecewiseCubicScalarTrajectory local,
    DurationNanoseconds segment_start,
    SwitchWaits waits,
    double yaw_start,
    double yaw_end) {
  if (local.segments.empty() || !std::isfinite(yaw_start) ||
      !std::isfinite(yaw_end)) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/wheel_plan/spin",
                   "INVALID_WHEEL_SPIN_TRAJECTORY");
  }
  PiecewiseCubicScalarTrajectory absolute{
      .value_semantics = "unwrapped_yaw_rad"};
  absolute.segments.reserve(
      local.segments.size() +
      (waits.leading.count() > 0 ? 1U : 0U) +
      (waits.trailing.count() > 0 ? 1U : 0U));
  std::chrono::nanoseconds movement_start =
      segment_start.value;
  if (!CheckedAdd(waits.leading, movement_start)) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/wheel_plan/spin",
                   "WHEEL_SPIN_TIME_OVERFLOW");
  }
  if (waits.leading.count() > 0) {
    absolute.segments.push_back(
        {
            .start_offset = segment_start,
            .end_offset =
                DurationNanoseconds{movement_start},
            .coefficients = {yaw_start, 0.0, 0.0, 0.0},
        });
  }
  for (auto& polynomial : local.segments) {
    if (!ShiftOffset(polynomial.start_offset, movement_start) ||
        !ShiftOffset(polynomial.end_offset, movement_start)) {
      return Failure(ErrorCode::kNumericalFailure,
                     "/wheel_plan/spin",
                     "WHEEL_SPIN_TIME_OVERFLOW");
    }
    absolute.segments.push_back(std::move(polynomial));
  }
  std::chrono::nanoseconds segment_end =
      absolute.segments.back().end_offset.value;
  if (waits.trailing.count() > 0) {
    const std::chrono::nanoseconds wait_start = segment_end;
    if (!CheckedAdd(waits.trailing, segment_end)) {
      return Failure(ErrorCode::kNumericalFailure,
                     "/wheel_plan/spin",
                     "WHEEL_SPIN_TIME_OVERFLOW");
    }
    absolute.segments.push_back(
        {
            .start_offset =
                DurationNanoseconds{wait_start},
            .end_offset = DurationNanoseconds{segment_end},
            .coefficients = {yaw_end, 0.0, 0.0, 0.0},
        });
  }
  return AbsoluteYaw{
      .trajectory = std::move(absolute),
      .interval =
          {
              .start_offset = segment_start,
              .end_offset =
                  DurationNanoseconds{segment_end},
          },
  };
}

[[nodiscard]] double SegmentSpeedLimit(
    const WheelDiscreteSegment& segment,
    const SafeProjection& projection,
    double fallback) noexcept {
  double limit = fallback;
  for (const WheelLatticeEdge& edge : segment.edges) {
    if (edge.primitive_kind == WheelPrimitiveKind::kModeSwitch) {
      continue;
    }
    const Cell cells[] = {
        {edge.source.ix, edge.source.iy},
        {edge.target.ix, edge.target.iy},
    };
    for (const Cell cell : cells) {
      if (!projection.InBounds(cell)) {
        return 0.0;
      }
      const std::size_t index =
          static_cast<std::size_t>(cell.y) *
              projection.geometry.width +
          static_cast<std::size_t>(cell.x);
      if (index >=
          projection.conservative_speed_limit_mps.size()) {
        return 0.0;
      }
      const double value = static_cast<double>(
          projection.conservative_speed_limit_mps[index]);
      if (!std::isfinite(value) || value <= 0.0) {
        return 0.0;
      }
      limit = std::min(limit, value);
    }
  }
  return limit;
}

[[nodiscard]] WheelTimingConfig TimingConfigFor(
    const PlannerAlgorithmConfig& algorithm,
    const WheelCapabilityView& capability) noexcept {
  return {
      .maximum_adaptive_samples =
          algorithm.wheeled.time_scaling
              .maximum_adaptive_samples,
      .minimum_parameter_step =
          algorithm.wheeled.time_scaling
              .minimum_parameter_step,
      .curvature_refinement_threshold_per_m =
          std::max(
              capability.hard_limits()
                      .maximum_drive_curvature_per_m *
                  0.25,
              1.0e-6),
      .maximum_forward_passes =
          algorithm.wheeled.time_scaling
              .maximum_forward_passes,
      .maximum_backward_passes =
          algorithm.wheeled.time_scaling
              .maximum_backward_passes,
  };
}

[[nodiscard]] WheelSweepValidationConfig SweepConfigFor(
    const PlanningRequest& request) noexcept {
  const std::size_t map_cells =
      request.map_snapshot
          ? request.map_snapshot->geometry().CellCount()
          : 0U;
  return {
      .maximum_subdivisions =
          request.algorithm_config
              ? request.algorithm_config->wheeled
                    .continuous_validation_maximum_subdivisions
              : 0U,
      .maximum_footprint_cells_per_sample =
          std::max<std::size_t>(
              1U,
              std::min(map_cells,
                       kMaximumFootprintCellsPerSample)),
      .additional_margin_m = 0.0,
  };
}

[[nodiscard]] WheelCollisionEnvelope EnvelopeFor(
    const WheelCapabilityView& capability) {
  return {
      .footprint = capability.collision_envelope(),
      .horizontal_tracking_error_m =
          capability.certified_horizontal_position_error_m(),
      .minimum_clearance_m =
          capability.hard_limits().minimum_clearance_m,
  };
}

[[nodiscard]] std::size_t SaturatingProduct(
    std::size_t lhs, std::size_t rhs,
    std::size_t cap) noexcept {
  if (lhs == 0U || rhs == 0U) {
    return 0U;
  }
  if (lhs > cap / rhs) {
    return cap;
  }
  return std::min(lhs * rhs, cap);
}

[[nodiscard]] WheelCorridorConfig CorridorConfigFor(
    const PlannerAlgorithmConfig& algorithm,
    const WheelCapabilityView& capability) noexcept {
  const auto& source = algorithm.wheeled.corridor;
  return {
      .maximum_regions = source.maximum_regions,
      .maximum_inflation_iterations =
          source.maximum_inflation_iterations,
      .maximum_halfplanes_per_region =
          source.maximum_halfplanes_per_region,
      .maximum_centerline_samples =
          std::max<std::size_t>(
              3U,
              SaturatingProduct(
                  source.maximum_regions,
                  source.maximum_halfplanes_per_region,
                  kMaximumInternalCenterlineSamples)),
      .sampling_spacing_m = source.sampling_spacing_m,
      .additional_margin_m = 0.0,
      .maximum_curvature_per_m =
          capability.hard_limits()
              .maximum_drive_curvature_per_m,
  };
}

[[nodiscard]] WheelSplineConfig SplineConfigFor(
    const PlannerAlgorithmConfig& algorithm,
    const WheelCapabilityView& capability,
    std::size_t edge_count) noexcept {
  const auto& source = algorithm.wheeled.smoothing;
  const auto tolerance =
      std::max(source.constraint_tolerance,
               kMinimumQpTolerance);
  const auto requested_control_points =
      edge_count >
              kMaximumInternalControlPoints - 3U
          ? kMaximumInternalControlPoints
          : edge_count + 3U;
  const auto time_tolerance =
      std::min(
          algorithm.time_equivalence_tolerance.value,
          source.maximum_time_increase.value);
  return {
      .maximum_control_points =
          std::max<std::size_t>(
              4U,
              std::min(requested_control_points,
                       kMaximumInternalControlPoints)),
      .maximum_scp_iterations =
          source.maximum_scp_iterations,
      .maximum_qp_iterations = kFixedQpIterations,
      .path_deviation_weight = kPathDeviationWeight,
      .second_difference_weight =
          kSecondDifferenceWeight,
      .initial_trust_region_m =
          source.initial_trust_region_m,
      .minimum_trust_region_m =
          source.minimum_trust_region_m,
      .constraint_tolerance =
          source.constraint_tolerance,
      .absolute_qp_tolerance = tolerance,
      .relative_qp_tolerance = tolerance,
      .maximum_curvature_per_m =
          capability.hard_limits()
              .maximum_drive_curvature_per_m,
      .time_equivalence_tolerance =
          DurationNanoseconds{time_tolerance},
  };
}

[[nodiscard]] Result<DriveSegment> BuildDriveSegment(
    const WheelDiscreteSegment& discrete,
    GeometricPath path,
    DurationNanoseconds start_offset,
    const WheelCapabilityView& capability,
    const PlannerAlgorithmConfig& algorithm,
    const SafeProjection& projection,
    const WheelSweepValidator& sweep_validator,
    std::string segment_id) {
  const DriveDirection direction =
      discrete.mode == WheelMotionMode::kReverse
          ? DriveDirection::kReverse
          : DriveDirection::kForward;
  const double directional_limit =
      direction == DriveDirection::kReverse
          ? capability.hard_limits()
                .maximum_reverse_speed_mps
          : capability.hard_limits()
                .maximum_forward_speed_mps;
  const double terrain_limit =
      SegmentSpeedLimit(discrete, projection,
                        directional_limit);
  const auto timing = ParameterizeWheelDrive(
      path, direction,
      {
          .maximum_forward_speed_mps =
              capability.hard_limits()
                  .maximum_forward_speed_mps,
          .maximum_reverse_speed_mps =
              capability.hard_limits()
                  .maximum_reverse_speed_mps,
          .maximum_linear_acceleration_mps2 =
              capability.hard_limits()
                  .maximum_forward_acceleration_mps2,
          .maximum_braking_deceleration_mps2 =
              capability.hard_limits()
                  .maximum_braking_deceleration_mps2,
          .maximum_yaw_rate_radps =
              capability.hard_limits()
                  .maximum_spin_rate_radps,
          .maximum_lateral_acceleration_mps2 =
              capability.hard_limits()
                  .maximum_lateral_acceleration_mps2,
          .terrain_speed_limit_mps = terrain_limit,
          .clearance_speed_limit_mps = terrain_limit,
          .traction_speed_limit_mps = terrain_limit,
      },
      TimingConfigFor(algorithm, capability));
  if (!IsOk(timing)) {
    return std::get<Error>(timing);
  }
  const auto waits = ModeSwitchWaits(discrete);
  if (!IsOk(waits)) {
    return std::get<Error>(waits);
  }
  auto absolute = MakeAbsoluteScaling(
      std::get<WheelTimingResult>(timing).time_scaling,
      start_offset, std::get<SwitchWaits>(waits));
  if (!IsOk(absolute)) {
    return std::get<Error>(absolute);
  }
  const auto& scaling =
      std::get<AbsoluteScaling>(absolute).scaling;
  const ValidationReport report = std::visit(
      [&](const auto& concrete) {
        using Path = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<
                          Path,
                          ClampedCubicBSplinePath>) {
          return sweep_validator.ValidateSpline(
              concrete, scaling);
        } else {
          return sweep_validator.ValidatePrimitiveChain(
              concrete);
        }
      },
      path);
  if (!report.ok()) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/wheel_plan/continuous_validation",
                   report.issues.front().reason_code);
  }
  auto built =
      std::get<AbsoluteScaling>(std::move(absolute));
  return DriveSegment{
      .segment_id = std::move(segment_id),
      .time_interval = built.interval,
      .direction = direction,
      .geometric_path = std::move(path),
      .time_scaling = std::move(built.scaling),
      .derived_caches = std::nullopt,
  };
}

[[nodiscard]] Result<DriveSegment> BuildPrimitiveDrive(
    const WheelDiscreteSegment& discrete,
    DurationNanoseconds start_offset,
    const WheelCapabilityView& capability,
    const PlannerAlgorithmConfig& algorithm,
    const SafeProjection& projection,
    const WheelSweepValidator& sweep_validator,
    std::string segment_id) {
  const auto chain = BuildPrimitiveChain(discrete);
  if (!IsOk(chain)) {
    return std::get<Error>(chain);
  }
  return BuildDriveSegment(
      discrete, std::get<ValidatedPrimitiveChain>(chain),
      start_offset, capability, algorithm, projection,
      sweep_validator, std::move(segment_id));
}

[[nodiscard]] Result<DriveSegment> BuildSmoothDrive(
    const WheelDiscreteSegment& discrete,
    DurationNanoseconds start_offset,
    const WheelCapabilityView& capability,
    const PlannerAlgorithmConfig& algorithm,
    const SafeProjection& projection,
    const WheelSweepValidator& sweep_validator,
    BoundedQpSolver& solver,
    WheelCorridorBuilder corridor_builder,
    std::string segment_id) {
  const WheelCollisionEnvelope envelope =
      EnvelopeFor(capability);
  const CorridorResult corridor = corridor_builder(
      {
          .edges = discrete.edges,
          .projection = projection,
          .envelope = envelope,
          .config =
              CorridorConfigFor(algorithm, capability),
      });
  if (corridor.status != CorridorStatus::kCertified) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/wheel_plan/corridor",
                   corridor.reason_code.empty()
                       ? "WHEEL_CORRIDOR_FALLBACK"
                       : corridor.reason_code);
  }
  const auto optimized = OptimizeWheelSpline(
      {
          .discrete_segment = discrete,
          .corridor = corridor,
          .capability = capability,
          .config =
              SplineConfigFor(
                  algorithm, capability,
                  discrete.edges.size()),
          .committed_points =
              std::span<const WheelFrozenControlPoint>{},
      },
      solver);
  if (!optimized.spline.has_value()) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/wheel_plan/smoothing",
                   "WHEEL_SMOOTHING_FALLBACK");
  }
  return BuildDriveSegment(
      discrete, *optimized.spline, start_offset,
      capability, algorithm, projection, sweep_validator,
      std::move(segment_id));
}

[[nodiscard]] Result<SpinSegment> BuildSpinSegment(
    const WheelDiscreteSegment& discrete,
    DurationNanoseconds start_offset,
    const WheelCapabilityView& capability,
    const WheelSweepValidator& sweep_validator,
    std::string segment_id) {
  const auto waits = ModeSwitchWaits(discrete);
  if (!IsOk(waits)) {
    return std::get<Error>(waits);
  }
  const WheelLatticeEdge* first_spin = nullptr;
  const WheelLatticeEdge* last_spin = nullptr;
  Vec3 fixed_position{};
  for (const WheelLatticeEdge& edge : discrete.edges) {
    if (edge.primitive_kind == WheelPrimitiveKind::kModeSwitch) {
      continue;
    }
    if (edge.primitive_kind != WheelPrimitiveKind::kSpin) {
      return Failure(ErrorCode::kInvalidArgument,
                     "/wheel_plan/spin",
                     "NON_SPIN_EDGE_IN_SPIN_SEGMENT");
    }
    if (first_spin == nullptr) {
      first_spin = &edge;
      fixed_position = edge.source_pose.position_m;
    }
    if (std::hypot(
            fixed_position.x -
                edge.source_pose.position_m.x,
            fixed_position.y -
                edge.source_pose.position_m.y) >
            kGeometryTolerance ||
        std::hypot(
            fixed_position.x -
                edge.target_pose.position_m.x,
            fixed_position.y -
                edge.target_pose.position_m.y) >
            kGeometryTolerance) {
      return Failure(ErrorCode::kInvalidArgument,
                     "/wheel_plan/spin",
                     "SPIN_POSITION_IS_NOT_FIXED");
    }
    last_spin = &edge;
  }
  if (first_spin == nullptr || last_spin == nullptr) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/wheel_plan/spin",
                   "EMPTY_WHEEL_SPIN_SEGMENT");
  }
  const double yaw_start = first_spin->source_pose.yaw_rad;
  const double yaw_end = last_spin->target_pose.yaw_rad;
  const auto local = ParameterizeWheelSpin(
      yaw_start, yaw_end,
      {
          .maximum_yaw_rate_radps =
              capability.hard_limits()
                  .maximum_spin_rate_radps,
          .maximum_yaw_acceleration_radps2 =
              capability.hard_limits()
                  .maximum_yaw_acceleration_radps2,
      });
  if (!IsOk(local)) {
    return std::get<Error>(local);
  }
  auto absolute = MakeAbsoluteYaw(
      std::get<PiecewiseCubicScalarTrajectory>(local),
      start_offset, std::get<SwitchWaits>(waits),
      yaw_start, yaw_end);
  if (!IsOk(absolute)) {
    return std::get<Error>(absolute);
  }
  const auto& yaw = std::get<AbsoluteYaw>(absolute).trajectory;
  const auto report =
      sweep_validator.ValidateSpin(fixed_position, yaw);
  if (!report.ok()) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/wheel_plan/continuous_validation",
                   report.issues.front().reason_code);
  }
  auto built = std::get<AbsoluteYaw>(std::move(absolute));
  return SpinSegment{
      .segment_id = std::move(segment_id),
      .time_interval = built.interval,
      .fixed_position_m = fixed_position,
      .unwrapped_yaw_rad = std::move(built.trajectory),
  };
}

struct SegmentAssembly final {
  std::vector<WheeledSegment> segments;
  DurationNanoseconds end_offset;
};

[[nodiscard]] Result<SegmentAssembly> AssembleSegments(
    const WheelDiscretePlan& discrete_plan,
    bool smooth_drives,
    const WheelCapabilityView& capability,
    const PlannerAlgorithmConfig& algorithm,
    const SafeProjection& projection,
    const WheelSweepValidator& sweep_validator,
    BoundedQpSolver* qp_solver,
    WheelCorridorBuilder corridor_builder,
    std::string_view stable_prefix) {
  SegmentAssembly assembly;
  assembly.segments.reserve(discrete_plan.segments.size());
  for (std::size_t index = 0U;
       index < discrete_plan.segments.size(); ++index) {
    const WheelDiscreteSegment& discrete =
        discrete_plan.segments[index];
    const std::string segment_id =
        std::string{stable_prefix} + "-segment-" +
        std::to_string(index);
    if (DriveMode(discrete.mode)) {
      Result<DriveSegment> drive =
          smooth_drives && qp_solver != nullptr
              ? BuildSmoothDrive(
                    discrete, assembly.end_offset, capability,
                    algorithm, projection, sweep_validator,
                    *qp_solver, corridor_builder, segment_id)
              : BuildPrimitiveDrive(
                    discrete, assembly.end_offset, capability,
                    algorithm, projection, sweep_validator,
                    segment_id);
      if (!IsOk(drive)) {
        return std::get<Error>(drive);
      }
      auto built = std::get<DriveSegment>(std::move(drive));
      assembly.end_offset =
          built.time_interval.end_offset;
      assembly.segments.emplace_back(std::move(built));
    } else if (SpinMode(discrete.mode)) {
      auto spin = BuildSpinSegment(
          discrete, assembly.end_offset, capability,
          sweep_validator, segment_id);
      if (!IsOk(spin)) {
        return std::get<Error>(spin);
      }
      auto built = std::get<SpinSegment>(std::move(spin));
      assembly.end_offset =
          built.time_interval.end_offset;
      assembly.segments.emplace_back(std::move(built));
    } else {
      return Failure(ErrorCode::kInvalidArgument,
                     "/wheel_plan/segments",
                     "INVALID_WHEEL_SEGMENT_MODE");
    }
  }
  if (assembly.segments.empty()) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/wheel_plan/segments",
                   "EMPTY_WHEEL_REFERENCE");
  }
  return assembly;
}

[[nodiscard]] std::optional<PoseXyzYaw> SegmentEndPose(
    const WheeledSegment& segment) {
  return std::visit(
      [](const auto& concrete) -> std::optional<PoseXyzYaw> {
        using Segment = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<Segment, DriveSegment>) {
          return std::visit(
              [](const auto& path) -> std::optional<PoseXyzYaw> {
                using Path = std::decay_t<decltype(path)>;
                if constexpr (std::is_same_v<
                                  Path,
                                  ClampedCubicBSplinePath>) {
                  if (path.control_points.empty()) {
                    return std::nullopt;
                  }
                  return path.control_points.back();
                } else {
                  if (path.primitives.empty()) {
                    return std::nullopt;
                  }
                  return path.primitives.back().end_pose;
                }
              },
              concrete.geometric_path);
        } else {
          if (concrete.unwrapped_yaw_rad.segments.empty()) {
            return std::nullopt;
          }
          const auto finish =
              concrete.unwrapped_yaw_rad.segments.back().end_offset;
          const auto yaw = EvaluateScalarTrajectory(
              concrete.unwrapped_yaw_rad, finish);
          if (!yaw.has_value()) {
            return std::nullopt;
          }
          return PoseXyzYaw{
              .position_m = concrete.fixed_position_m,
              .yaw_rad = *yaw,
          };
        }
      },
      segment);
}

[[nodiscard]] bool SamePose(
    const PoseXyzYaw& lhs,
    const PoseXyzYaw& rhs) noexcept {
  return std::hypot(lhs.position_m.x - rhs.position_m.x,
                    lhs.position_m.y - rhs.position_m.y) <=
             kGeometryTolerance &&
         std::abs(lhs.position_m.z - rhs.position_m.z) <=
             kGeometryTolerance &&
         std::abs(lhs.yaw_rad - rhs.yaw_rad) <=
             kGeometryTolerance;
}

[[nodiscard]] bool EndsStopped(
    const WheeledSegment& segment) noexcept {
  return std::visit(
      [](const auto& concrete) {
        using Segment = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<Segment, DriveSegment>) {
          const auto rate =
              EvaluateMonotoneTimeScalingDerivative(
                  concrete.time_scaling,
                  concrete.time_interval.end_offset);
          return rate.has_value() &&
                 std::abs(*rate) <= kGeometryTolerance;
        } else {
          const auto rate =
              EvaluateScalarTrajectoryDerivative(
                  concrete.unwrapped_yaw_rad,
                  concrete.time_interval.end_offset);
          return rate.has_value() &&
                 std::abs(*rate) <= kGeometryTolerance;
        }
      },
      segment);
}

[[nodiscard]] Result<WheeledReference> PlanImpl(
    BoundedQpSolver* qp_solver,
    WheelCorridorBuilder corridor_builder,
    const PlanningRequest& request,
    const ResolvedTerminalSet& terminal_set) {
  if (request.platform_type != PlatformType::kWheeled ||
      !std::holds_alternative<WheeledOrLeggedState>(
          request.current_state) ||
      !request.map_snapshot || !request.safety_capability ||
      !request.algorithm_config ||
      corridor_builder == nullptr) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/planning_request",
                   "INVALID_WHEEL_PLANNING_REQUEST");
  }
  const auto capability =
      WheelCapabilityView::Create(*request.safety_capability);
  if (!IsOk(capability)) {
    return std::get<Error>(capability);
  }
  const auto primitives =
      WheelPrimitiveCatalog::Create(*request.safety_capability);
  if (!IsOk(primitives)) {
    return std::get<Error>(primitives);
  }
  std::shared_ptr<const LearnedCostSnapshot> learned;
  if (request.learned_cost_snapshot.has_value()) {
    learned =
        request.learned_cost_snapshot->resolved_snapshot;
  }
  const auto projection = BuildSafeProjection(
      {
          .map = request.map_snapshot,
          .capability = request.safety_capability,
          .algorithm_config = request.algorithm_config,
          .learned_cost = std::move(learned),
      });
  if (!IsOk(projection)) {
    return std::get<Error>(projection);
  }
  const auto discrete = PlanWheelDiscrete(
      WheelPlanningProblem{
          std::get<SafeProjection>(projection),
          std::get<WheeledOrLeggedState>(
              request.current_state),
          terminal_set,
          *request.safety_capability,
          *request.algorithm_config,
      },
      std::get<WheelPrimitiveCatalog>(primitives),
      request.algorithm_config->ara_star);
  if (!IsOk(discrete)) {
    return std::get<Error>(discrete);
  }
  const auto& discrete_plan =
      std::get<WheelDiscretePlan>(discrete);
  const auto stable_hash = Sha256Hex(
      request.request_id + "\n" +
      discrete_plan.selected_candidate_id +
      "\nWHEELED");
  if (!IsOk(stable_hash)) {
    return std::get<Error>(stable_hash);
  }
  const std::string stable_prefix =
      "wheel-" +
      std::get<Sha256Digest>(stable_hash).substr(0U, 32U);
  const auto& projection_value =
      std::get<SafeProjection>(projection);
  const auto& capability_value =
      std::get<WheelCapabilityView>(capability);
  const WheelSweepValidator sweep_validator{
      projection_value, EnvelopeFor(capability_value),
      SweepConfigFor(request)};

  Result<SegmentAssembly> assembled =
      qp_solver != nullptr
          ? AssembleSegments(
                discrete_plan, true, capability_value,
                *request.algorithm_config, projection_value,
                sweep_validator, qp_solver, corridor_builder,
                stable_prefix)
          : AssembleSegments(
                discrete_plan, false, capability_value,
                *request.algorithm_config, projection_value,
                sweep_validator, qp_solver, corridor_builder,
                stable_prefix);
  if (qp_solver != nullptr && !IsOk(assembled)) {
    assembled = AssembleSegments(
        discrete_plan, false, capability_value,
        *request.algorithm_config, projection_value,
        sweep_validator, qp_solver, corridor_builder,
        stable_prefix);
  }
  if (!IsOk(assembled)) {
    return std::get<Error>(assembled);
  }
  auto segments =
      std::get<SegmentAssembly>(std::move(assembled)).segments;
  const auto end_pose = SegmentEndPose(segments.back());
  if (!end_pose.has_value() ||
      !SamePose(*end_pose,
                discrete_plan.safe_stop_anchor.pose) ||
      !EndsStopped(segments.back())) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/platform_reference/safe_stop_anchor",
                   "WHEEL_REFERENCE_DOES_NOT_END_STOPPED_AT_ANCHOR");
  }
  WheeledReference reference{
      .reference_id = stable_prefix,
      .reference_hash = std::string(64U, '0'),
      .reference_time_origin = request.request_time,
      .segments = std::move(segments),
      .safe_stop_anchor = discrete_plan.safe_stop_anchor,
  };
  const auto reference_hash =
      CanonicalReferenceHash(PlatformReference{reference});
  if (!IsOk(reference_hash)) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/platform_reference/reference_hash",
                   "WHEEL_REFERENCE_HASH_FAILED");
  }
  reference.reference_hash =
      std::get<Sha256Digest>(reference_hash);
  const auto report =
      SemanticValidator{}.Validate(PlatformReference{reference});
  if (!report.ok()) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/platform_reference",
                   report.issues.front().reason_code);
  }
  return reference;
}

}  // namespace

WheelPlanner::WheelPlanner() noexcept
    : WheelPlanner(nullptr, &BuildWheelCorridor) {}

WheelPlanner::WheelPlanner(
    BoundedQpSolver* qp_solver,
    WheelCorridorBuilder corridor_builder) noexcept
    : qp_solver_(qp_solver),
      corridor_builder_(corridor_builder) {}

Result<WheeledReference> WheelPlanner::Plan(
    const PlanningRequest& request,
    const ResolvedTerminalSet& terminal_set) noexcept {
  try {
    return PlanImpl(
        qp_solver_, corridor_builder_, request, terminal_set);
  } catch (const std::exception& exception) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/wheel_planner",
                   exception.what());
  } catch (...) {
    return Failure(ErrorCode::kNumericalFailure,
                   "/wheel_planner",
                   "UNKNOWN_WHEEL_PLANNER_FAILURE");
  }
}

}  // namespace lunar::planning::v3
