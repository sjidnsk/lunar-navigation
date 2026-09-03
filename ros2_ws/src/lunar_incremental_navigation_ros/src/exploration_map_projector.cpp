#include "lunar_incremental_navigation_ros/exploration_map_projector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lunar::incremental_navigation_ros {

namespace {

namespace core = lunar::incremental_navigation;

[[nodiscard]] double CellMinimum(const double origin,
                                 const double resolution,
                                 const std::int64_t index) {
  return std::fma(static_cast<double>(index), resolution, origin);
}

[[nodiscard]] double OverlapLength(const double first_min,
                                   const double first_max,
                                   const double second_min,
                                   const double second_max) {
  return std::max(0.0,
                  std::min(first_max, second_max) -
                      std::max(first_min, second_min));
}

}  // namespace

ExplorationMapProjection ExplorationMapProjector::Project(
    const core::FineTraversabilitySnapshot& fine,
    const core::SparseGridGeometry& coarse_geometry) const {
  const core::SparseGridGeometry& fine_geometry = fine.geometry();
  if (!coarse_geometry.valid() ||
      coarse_geometry.frame_id() != fine_geometry.frame_id()) {
    throw std::invalid_argument(
        "exploration projection requires compatible map geometry");
  }

  ExplorationMapProjection result{
      .geometry = coarse_geometry,
      .source_fine_traversability_revision =
          fine.fine_traversability_revision(),
      .data = std::vector<std::int8_t>(coarse_geometry.CellCount(), -1)};
  const core::Vec3 fine_origin = fine_geometry.origin_m();
  const core::Vec3 coarse_origin = coarse_geometry.origin_m();
  const double fine_resolution = fine_geometry.resolution_m();
  const double coarse_resolution = coarse_geometry.resolution_m();
  const double coarse_area = coarse_resolution * coarse_resolution;
  const double area_tolerance = coarse_area * 1.0e-9;
  const core::GridIndex coarse_min = coarse_geometry.min_inclusive();
  const core::GridIndex coarse_max = coarse_geometry.max_exclusive();

  std::size_t output_index = 0U;
  for (std::int64_t coarse_y = coarse_min.y; coarse_y < coarse_max.y;
       ++coarse_y) {
    const double coarse_y_min =
        CellMinimum(coarse_origin.y, coarse_resolution, coarse_y);
    const double coarse_y_max = coarse_y_min + coarse_resolution;
    const std::int64_t fine_y_begin = std::max(
        fine_geometry.min_inclusive().y,
        static_cast<std::int64_t>(
            std::floor((coarse_y_min - fine_origin.y) / fine_resolution)));
    const std::int64_t fine_y_end = std::min(
        fine_geometry.max_exclusive().y,
        static_cast<std::int64_t>(
            std::ceil((coarse_y_max - fine_origin.y) / fine_resolution)));

    for (std::int64_t coarse_x = coarse_min.x; coarse_x < coarse_max.x;
         ++coarse_x, ++output_index) {
      const double coarse_x_min =
          CellMinimum(coarse_origin.x, coarse_resolution, coarse_x);
      const double coarse_x_max = coarse_x_min + coarse_resolution;
      const std::int64_t fine_x_begin = std::max(
          fine_geometry.min_inclusive().x,
          static_cast<std::int64_t>(
              std::floor((coarse_x_min - fine_origin.x) / fine_resolution)));
      const std::int64_t fine_x_end = std::min(
          fine_geometry.max_exclusive().x,
          static_cast<std::int64_t>(
              std::ceil((coarse_x_max - fine_origin.x) / fine_resolution)));

      bool has_free = false;
      double blocked_area = 0.0;
      for (std::int64_t fine_y = fine_y_begin;
           fine_y < fine_y_end && !has_free; ++fine_y) {
        const double fine_y_min =
            CellMinimum(fine_origin.y, fine_resolution, fine_y);
        const double overlap_y = OverlapLength(
            coarse_y_min, coarse_y_max, fine_y_min,
            fine_y_min + fine_resolution);
        if (overlap_y <= 0.0) {
          continue;
        }
        for (std::int64_t fine_x = fine_x_begin; fine_x < fine_x_end;
             ++fine_x) {
          const double fine_x_min =
              CellMinimum(fine_origin.x, fine_resolution, fine_x);
          const double overlap_x = OverlapLength(
              coarse_x_min, coarse_x_max, fine_x_min,
              fine_x_min + fine_resolution);
          if (overlap_x <= 0.0) {
            continue;
          }
          switch (fine.State({.x = fine_x, .y = fine_y})) {
            case core::FineCellState::kFree:
              has_free = true;
              break;
            case core::FineCellState::kBlocked:
              blocked_area += overlap_x * overlap_y;
              break;
            case core::FineCellState::kUnknown:
              break;
          }
          if (has_free) {
            break;
          }
        }
      }
      if (has_free) {
        result.data[output_index] = 0;
      } else if (blocked_area + area_tolerance >= coarse_area) {
        result.data[output_index] = 100;
      }
    }
  }
  return result;
}

}  // namespace lunar::incremental_navigation_ros
