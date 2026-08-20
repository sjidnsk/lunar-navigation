#pragma once

#include "luna_t3_map_adapter/roi_level.hpp"

#include <cstddef>
#include <vector>

namespace luna::task3 {

struct FineCell final {
  double elevation_m{};
  bool valid_mask{};
  bool obstacle{};
  double obstacle_height_m{};
  double observation_age_s{};
  double observation_quality{};
  double elevation_variance_m2{};
  double obstacle_variance_m2{};
  std::size_t observation_count{};
  bool forbidden{};
};

struct FineGrid final {
  double resolution_m{};
  std::size_t width{};
  std::size_t height{};
  std::vector<FineCell> cells;

  [[nodiscard]] const FineCell& At(const std::size_t x,
                                   const std::size_t y) const noexcept {
    return cells.at(y * width + x);
  }
};

[[nodiscard]] Result<FineGrid> AggregateConservatively(
    const FineGrid& source,
    const SelectedGlobalLevel& target);

}  // namespace luna::task3
