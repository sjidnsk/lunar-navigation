#include "lunar_pure_exploration_core/coverage.hpp"

#include "detail/coverage_statistics.hpp"

namespace lunar::pure_exploration {

CoverageStats CalculateCoverage(const TaskRaster& raster) {
  detail::CoverageCounts counts{};
  for (const GridIndex cell : raster.task_cells()) {
    detail::AccumulateCoverageState(raster.Classify(cell), counts);
  }
  return detail::FinalizeCoverage(raster.polygon_area_m2(),
                                  raster.geometry().resolution, counts);
}

}  // namespace lunar::pure_exploration
