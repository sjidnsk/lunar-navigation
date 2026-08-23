#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "lunar_pure_exploration_core/coverage.hpp"
#include "lunar_pure_exploration_core/types.hpp"

namespace lunar::pure_exploration::detail {

struct CoverageCounts {
  std::size_t free{0U};
  std::size_t occupied{0U};
  std::size_t unknown{0U};
  std::size_t outside_map{0U};
};

inline void CheckedIncrement(std::size_t& count) {
  if (count == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("coverage state count overflow");
  }
  ++count;
}

inline std::size_t CheckedCountSum(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error("coverage cell count overflow");
  }
  return left + right;
}

inline void AccumulateCoverageState(CellState state, CoverageCounts& counts) {
  switch (state) {
    case CellState::kOutsideTask:
      throw std::logic_error("task cell unexpectedly outside task");
    case CellState::kOutsideMap:
      CheckedIncrement(counts.outside_map);
      break;
    case CellState::kUnknown:
      CheckedIncrement(counts.unknown);
      break;
    case CellState::kFree:
      CheckedIncrement(counts.free);
      break;
    case CellState::kOccupied:
      CheckedIncrement(counts.occupied);
      break;
  }
}

inline double CheckedCoverageArea(std::size_t cell_count,
                                  double resolution_m) {
  if (!std::isfinite(resolution_m) || resolution_m <= 0.0) {
    throw std::invalid_argument("coverage resolution must be finite positive");
  }
  const long double area_ld =
      static_cast<long double>(cell_count) *
      static_cast<long double>(resolution_m) *
      static_cast<long double>(resolution_m);
  if (!std::isfinite(area_ld) ||
      (cell_count != 0U && !(area_ld > 0.0L))) {
    throw std::overflow_error("coverage area exceeds long double range");
  }
  const double area = static_cast<double>(area_ld);
  if (!std::isfinite(area) || (cell_count != 0U && area <= 0.0)) {
    throw std::overflow_error("coverage area is not representable as double");
  }
  return area;
}

inline CoverageStats FinalizeCoverage(double polygon_area_m2,
                                      double resolution_m,
                                      const CoverageCounts& counts) {
  const std::size_t known_cell_count =
      CheckedCountSum(counts.free, counts.occupied);
  const std::size_t unknown_cell_count =
      CheckedCountSum(counts.unknown, counts.outside_map);
  const std::size_t task_cell_count =
      CheckedCountSum(known_cell_count, unknown_cell_count);

  double coverage_ratio = 0.0;
  if (task_cell_count != 0U) {
    coverage_ratio = static_cast<double>(known_cell_count) /
                     static_cast<double>(task_cell_count);
    if (!std::isfinite(coverage_ratio) || coverage_ratio < 0.0 ||
        coverage_ratio > 1.0) {
      throw std::overflow_error("coverage ratio is outside its numeric range");
    }
  }

  return CoverageStats{
      .polygon_area_m2 = polygon_area_m2,
      .task_raster_area_m2 =
          CheckedCoverageArea(task_cell_count, resolution_m),
      .known_free_area_m2 = CheckedCoverageArea(counts.free, resolution_m),
      .known_occupied_area_m2 =
          CheckedCoverageArea(counts.occupied, resolution_m),
      .unknown_area_m2 = CheckedCoverageArea(counts.unknown, resolution_m),
      .outside_map_area_m2 =
          CheckedCoverageArea(counts.outside_map, resolution_m),
      .coverage_ratio = coverage_ratio,
  };
}

}  // namespace lunar::pure_exploration::detail
