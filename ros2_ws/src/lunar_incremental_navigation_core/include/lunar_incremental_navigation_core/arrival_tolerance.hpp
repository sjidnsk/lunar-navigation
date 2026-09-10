#pragma once

namespace lunar::incremental_navigation {

// Numerical allowance only. The caller owns the task's position/yaw tolerance;
// this does not enlarge it by a physically meaningful tracking distance.
inline constexpr double kArrivalComparisonTolerance = 1.0e-9;

[[nodiscard]] inline bool WithinArrivalTolerance(double error,
                                                  double tolerance) noexcept {
  return error <= tolerance + kArrivalComparisonTolerance;
}

}  // namespace lunar::incremental_navigation
