#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "lunar_pure_exploration_core/occupancy_grid.hpp"

namespace lunar::pure_exploration {

class TaskRaster {
 public:
  struct Limits {
    std::size_t maximum_raster_cell_count{1048576U};
  };

  static TaskRaster Build(const OccupancyGridView& map,
                          const Polygon2& polygon,
                          Limits limits = Limits{1048576U});

  CellState Classify(GridIndex logical_index) const;
  bool ContainsCellCenter(GridIndex logical_index) const;
  bool IsMapBacked(GridIndex logical_index) const;
  std::optional<Vec2> WorldToGrid(Vec2 world_point) const;
  std::optional<Vec2> GridToWorld(Vec2 grid_point) const;
  std::optional<GridIndex> WorldToCell(Vec2 world_point) const;
  std::array<Vec2, 4> CellCornersInGrid(GridIndex logical_index) const;
  Vec2 CellCenter(GridIndex logical_index) const;
  const GridGeometry& geometry() const;
  double polygon_area_m2() const;
  std::span<const GridIndex> task_cells() const;
  std::span<const GridIndex> map_backed_cells() const;

 private:
  GridGeometry geometry_{};
  double polygon_area_m2_{0.0};
  GridIndex raster_min_{0, 0};
  std::size_t raster_width_{0U};
  std::size_t raster_height_{0U};
  std::vector<CellState> raster_states_;
  std::vector<GridIndex> task_cells_;
  std::vector<GridIndex> map_backed_cells_;
};

}  // namespace lunar::pure_exploration
