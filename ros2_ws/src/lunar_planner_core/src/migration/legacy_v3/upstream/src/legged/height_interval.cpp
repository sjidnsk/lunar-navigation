#include "lunar_path_planner/v3/legged/height_interval.hpp"

#include <algorithm>
#include <cmath>

namespace lunar::planning::v3 {

bool IsValidHeightInterval(const HeightInterval& interval) noexcept {
  return std::isfinite(interval.min_m) &&
         std::isfinite(interval.max_m) &&
         interval.min_m <= interval.max_m;
}

std::optional<HeightInterval> IntersectHeightIntervals(
    const HeightInterval& lhs,
    const HeightInterval& rhs) noexcept {
  if (!IsValidHeightInterval(lhs) || !IsValidHeightInterval(rhs)) {
    return std::nullopt;
  }
  const HeightInterval result{
      .min_m = std::max(lhs.min_m, rhs.min_m),
      .max_m = std::min(lhs.max_m, rhs.max_m),
  };
  return IsValidHeightInterval(result)
             ? std::optional<HeightInterval>{result}
             : std::nullopt;
}

std::optional<HeightInterval> ExpandHeightInterval(
    const HeightInterval& source,
    double maximum_down_delta_m,
    double maximum_up_delta_m) noexcept {
  if (!IsValidHeightInterval(source) ||
      !std::isfinite(maximum_down_delta_m) ||
      maximum_down_delta_m < 0.0 ||
      !std::isfinite(maximum_up_delta_m) ||
      maximum_up_delta_m < 0.0) {
    return std::nullopt;
  }
  const HeightInterval expanded{
      .min_m = source.min_m - maximum_down_delta_m,
      .max_m = source.max_m + maximum_up_delta_m,
  };
  return IsValidHeightInterval(expanded)
             ? std::optional<HeightInterval>{expanded}
             : std::nullopt;
}

}  // namespace lunar::planning::v3
