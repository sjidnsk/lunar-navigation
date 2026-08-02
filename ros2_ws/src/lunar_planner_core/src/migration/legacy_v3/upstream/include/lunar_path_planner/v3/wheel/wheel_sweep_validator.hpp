#pragma once

#include <cstddef>

#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/wheel/wheel_corridor.hpp"

namespace lunar::planning::v3 {

struct WheelSweepValidationConfig final {
  std::size_t maximum_subdivisions{};
  std::size_t maximum_footprint_cells_per_sample{};
  double additional_margin_m{};
};

class WheelSweepValidator final {
 public:
  WheelSweepValidator(
      const SafeProjection& projection,
      WheelCollisionEnvelope envelope,
      WheelSweepValidationConfig config);

  [[nodiscard]] ValidationReport ValidatePrimitiveChain(
      const ValidatedPrimitiveChain& chain) const;
  [[nodiscard]] ValidationReport ValidateSpline(
      const ClampedCubicBSplinePath& spline,
      const MonotoneTimeScaling& time_scaling) const;
  [[nodiscard]] ValidationReport ValidateSpin(
      const Vec3& fixed_position,
      const PiecewiseCubicScalarTrajectory& yaw) const;

 private:
  const SafeProjection* projection_{};
  WheelCollisionEnvelope envelope_;
  WheelSweepValidationConfig config_;
};

}  // namespace lunar::planning::v3
