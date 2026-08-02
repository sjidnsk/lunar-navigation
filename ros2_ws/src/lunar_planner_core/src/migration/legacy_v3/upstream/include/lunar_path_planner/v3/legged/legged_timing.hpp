#pragma once

#include <cstddef>
#include <string>

#include "lunar_path_planner/v3/contracts/platform_reference.hpp"
#include "lunar_path_planner/v3/contracts/profiles.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

struct LeggedTimingConfig final {
  TimeScalingConfig sampling;
  double maximum_linear_acceleration_mps2{};
  double maximum_yaw_acceleration_radps2{};
};

struct LeggedTimingDiagnostics final {
  std::size_t sample_count{};
  std::size_t forward_passes{};
  std::size_t backward_passes{};
  double maximum_body_rate_ratio{};
  double maximum_yaw_rate_ratio{};
  bool ends_stopped{false};
  std::string termination_reason;
};

struct LeggedTimingResult final {
  MonotoneTimeScaling time_scaling;
  DurationNanoseconds duration;
  LeggedTimingDiagnostics diagnostics;
};

// Parameterizes only the fixed body/CoM xyz+yaw path. Foot placement,
// gait, contacts, and contact forces remain downstream controller concerns.
[[nodiscard]] Result<LeggedTimingResult> ParameterizeLeggedBodyPath(
    const GeometricPath& path,
    const BodyFrameVelocityEnvelope& velocity_envelope,
    const LeggedTimingConfig& config);

}  // namespace lunar::planning::v3
