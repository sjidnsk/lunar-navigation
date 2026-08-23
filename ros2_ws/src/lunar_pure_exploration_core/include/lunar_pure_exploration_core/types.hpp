#pragma once

#include <compare>
#include <cstdint>
#include <vector>

namespace lunar::pure_exploration {

enum class CellState : std::uint8_t {
  kOutsideTask,
  kOutsideMap,
  kUnknown,
  kFree,
  kOccupied,
};

struct Vec2 {
  double x;
  double y;
};

struct Pose2 {
  double x;
  double y;
  double yaw;
};

struct GridIndex {
  std::int32_t x;
  std::int32_t y;
  auto operator<=>(const GridIndex&) const = default;
};

struct GridGeometry {
  std::uint32_t width;
  std::uint32_t height;
  double resolution;
  double origin_x;
  double origin_y;
  double origin_yaw;
};

struct Polygon2 {
  std::vector<Vec2> vertices;
};

}  // namespace lunar::pure_exploration
