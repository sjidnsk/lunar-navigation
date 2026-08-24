#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace lunar::pure_planner_ros {

struct CenterDistanceResult final {
  std::vector<double> squared_cells;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept { return reason_code.empty(); }
};

[[nodiscard]] CenterDistanceResult BuildCenterSquaredDistance(
    std::size_t width, std::size_t height,
    std::span<const std::uint8_t> seeds);

}  // namespace lunar::pure_planner_ros
