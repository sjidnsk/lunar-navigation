#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "lunar_path_planner/v3/contracts/platform_reference.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

struct WheelTimingLimits final {
  double maximum_forward_speed_mps{};
  double maximum_reverse_speed_mps{};
  double maximum_linear_acceleration_mps2{};
  double maximum_braking_deceleration_mps2{};
  double maximum_yaw_rate_radps{};
  double maximum_lateral_acceleration_mps2{};
  double terrain_speed_limit_mps{};
  double clearance_speed_limit_mps{};
  double traction_speed_limit_mps{};
};

struct WheelTimingConfig final {
  std::size_t maximum_adaptive_samples{};
  double minimum_parameter_step{};
  double curvature_refinement_threshold_per_m{};
  std::size_t maximum_forward_passes{};
  std::size_t maximum_backward_passes{};
};

struct SpinTimingLimits final {
  double maximum_yaw_rate_radps{};
  double maximum_yaw_acceleration_radps2{};
};

struct WheelTimingDiagnostics final {
  std::size_t sample_count{};
  std::size_t forward_passes{};
  std::size_t backward_passes{};
  std::string termination_reason;
};

struct WheelTimingResult final {
  MonotoneTimeScaling time_scaling;
  DurationNanoseconds duration;
  double path_length_m{};
  WheelTimingDiagnostics diagnostics;
};

[[nodiscard]] Result<WheelTimingResult> ParameterizeWheelDrive(
    const GeometricPath& path,
    DriveDirection direction,
    const WheelTimingLimits& limits,
    const WheelTimingConfig& config);

[[nodiscard]] Result<PiecewiseCubicScalarTrajectory>
ParameterizeWheelSpin(
    double yaw_start_rad,
    double yaw_end_unwrapped_rad,
    const SpinTimingLimits& limits);

[[nodiscard]] std::optional<double>
EvaluateMonotoneTimeScaling(
    const MonotoneTimeScaling& scaling,
    DurationNanoseconds offset) noexcept;

[[nodiscard]] std::optional<double>
EvaluateMonotoneTimeScalingDerivative(
    const MonotoneTimeScaling& scaling,
    DurationNanoseconds offset) noexcept;

[[nodiscard]] std::optional<double> EvaluateScalarTrajectory(
    const PiecewiseCubicScalarTrajectory& trajectory,
    DurationNanoseconds offset) noexcept;

[[nodiscard]] std::optional<double>
EvaluateScalarTrajectoryDerivative(
    const PiecewiseCubicScalarTrajectory& trajectory,
    DurationNanoseconds offset) noexcept;

[[nodiscard]] double SignedBodyForwardSpeed(
    double path_parameter_rate_per_s,
    double path_length_m,
    DriveDirection direction) noexcept;

}  // namespace lunar::planning::v3
