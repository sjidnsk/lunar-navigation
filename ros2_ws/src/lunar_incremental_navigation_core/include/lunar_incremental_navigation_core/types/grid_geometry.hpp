#pragma once

#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "lunar_incremental_navigation_core/types/geometry.hpp"

namespace lunar::incremental_navigation {

inline constexpr std::int64_t kGridTileWidthCells = 256;
inline constexpr std::size_t kGridTileCellCount =
    static_cast<std::size_t>(kGridTileWidthCells * kGridTileWidthCells);

struct Pose2 final {
  Vec2 position_m;
  double yaw_rad{};

  auto operator<=>(const Pose2&) const = default;
};

struct GridIndex final {
  std::int64_t x{};
  std::int64_t y{};

  auto operator<=>(const GridIndex&) const = default;
};

struct TileIndex final {
  std::int64_t x{};
  std::int64_t y{};

  auto operator<=>(const TileIndex&) const = default;
};

[[nodiscard]] constexpr TileIndex TileForCell(
    const GridIndex index) noexcept {
  const auto tile_coordinate = [](const std::int64_t value) {
    const std::int64_t quotient = value / kGridTileWidthCells;
    const std::int64_t remainder = value % kGridTileWidthCells;
    return remainder < 0 ? quotient - 1 : quotient;
  };
  return TileIndex{.x = tile_coordinate(index.x),
                   .y = tile_coordinate(index.y)};
}

[[nodiscard]] constexpr std::size_t TileCellOffset(
    const GridIndex index) noexcept {
  const auto cell_coordinate = [](const std::int64_t value) {
    const std::int64_t remainder = value % kGridTileWidthCells;
    return remainder < 0 ? remainder + kGridTileWidthCells : remainder;
  };
  return static_cast<std::size_t>(
             cell_coordinate(index.y)) *
             static_cast<std::size_t>(kGridTileWidthCells) +
         static_cast<std::size_t>(cell_coordinate(index.x));
}

struct GridGeometry final {
  std::string frame_id;
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
  Vec3 origin_m;

  [[nodiscard]] bool valid() const noexcept {
    if (frame_id != "map" || width == 0U || height == 0U ||
        width > std::numeric_limits<std::size_t>::max() / height ||
        !std::isfinite(resolution_m) || resolution_m <= 0.0 ||
        !std::isfinite(origin_m.x) || !std::isfinite(origin_m.y) ||
        !std::isfinite(origin_m.z)) {
      return false;
    }
    return std::isfinite(std::fma(static_cast<double>(width), resolution_m,
                                  origin_m.x)) &&
           std::isfinite(std::fma(static_cast<double>(height), resolution_m,
                                  origin_m.y));
  }

  [[nodiscard]] constexpr std::size_t CellCount() const noexcept {
    if (width == 0U || height == 0U ||
        width > std::numeric_limits<std::size_t>::max() / height) {
      return 0U;
    }
    return width * height;
  }

  [[nodiscard]] bool Contains(const GridIndex index) const noexcept {
    return valid() && index.x >= 0 && index.y >= 0 &&
           static_cast<std::uint64_t>(index.x) < width &&
           static_cast<std::uint64_t>(index.y) < height;
  }

  [[nodiscard]] bool Contains(const TileIndex index) const noexcept {
    if (!valid() || index.x < 0 || index.y < 0) {
      return false;
    }
    const std::size_t max_tile_x =
        (width - 1U) / static_cast<std::size_t>(kGridTileWidthCells);
    const std::size_t max_tile_y =
        (height - 1U) / static_cast<std::size_t>(kGridTileWidthCells);
    return static_cast<std::uint64_t>(index.x) <= max_tile_x &&
           static_cast<std::uint64_t>(index.y) <= max_tile_y;
  }
};

}  // namespace lunar::incremental_navigation
