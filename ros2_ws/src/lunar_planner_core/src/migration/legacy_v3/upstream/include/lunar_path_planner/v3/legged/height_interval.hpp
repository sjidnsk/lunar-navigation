#pragma once

#include <optional>

namespace lunar::planning::v3 {

struct HeightInterval final {
  double min_m{};
  double max_m{};
};

[[nodiscard]] bool IsValidHeightInterval(
    const HeightInterval& interval) noexcept;

[[nodiscard]] std::optional<HeightInterval> IntersectHeightIntervals(
    const HeightInterval& lhs,
    const HeightInterval& rhs) noexcept;

[[nodiscard]] std::optional<HeightInterval> ExpandHeightInterval(
    const HeightInterval& source,
    double maximum_down_delta_m,
    double maximum_up_delta_m) noexcept;

}  // namespace lunar::planning::v3
