#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"

namespace lunar::pure_planning::shared {

struct CellAreaClearanceResult final {
  std::vector<float> clearance_m;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept { return reason_code.empty(); }
};

struct CellAreaOffset final {
  std::int32_t dx{};
  std::int32_t dy{};
};

[[nodiscard]] CellAreaClearanceResult BuildCellAreaClearance(
    std::size_t width, std::size_t height, double resolution_m,
    std::span<const std::uint8_t> hazard_mask, SearchControl control = {});

[[nodiscard]] std::vector<CellAreaOffset> BuildCellAreaInflationStencil(
    double resolution_m, double inflation_m);

}  // namespace lunar::pure_planning::shared
