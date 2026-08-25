#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "lunar_pure_exploration_core/types.hpp"

namespace lunar::pure_exploration {

class OccupancyGridView {
 public:
  OccupancyGridView(GridGeometry geometry, std::span<const std::int8_t> data,
                    std::int8_t occupied_threshold);

  CellState Classify(GridIndex index) const;
  std::optional<std::int8_t> RawValue(GridIndex index) const;
  bool Contains(GridIndex index) const;
  static std::optional<Vec2> WorldToGrid(const GridGeometry& geometry,
                                         Vec2 world_point);
  static std::optional<Vec2> GridToWorld(const GridGeometry& geometry,
                                         Vec2 grid_point);
  static std::optional<GridIndex> WorldToCell(const GridGeometry& geometry,
                                              Vec2 world_point);
  static std::array<Vec2, 4> CellCornersInGrid(GridIndex logical_index);
  std::optional<Vec2> WorldToGrid(Vec2 point) const;
  std::optional<Vec2> GridToWorld(Vec2 point) const;
  std::optional<GridIndex> WorldToCell(Vec2 point) const;
  Vec2 CellCenter(GridIndex index) const;
  const GridGeometry& geometry() const;

 private:
  GridGeometry geometry_;
  std::vector<std::int8_t> data_;
  std::int8_t occupied_threshold_;
};

}  // namespace lunar::pure_exploration
