#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace lunar::observed_map {

inline constexpr double kL0ResolutionM = 0.2;
inline constexpr std::size_t kTileSideCells = 256U;
inline constexpr std::size_t kCellsPerTile =
    kTileSideCells * kTileSideCells;
inline constexpr std::size_t kMaximumTilesPerSession = 512U;
inline constexpr std::size_t kNeighborhoodCenter = 4U;

struct GridCellIndex final {
  std::int64_t x{};
  std::int64_t y{};

  bool operator==(const GridCellIndex&) const = default;
};

struct TileKey final {
  std::int64_t x{};
  std::int64_t y{};

  bool operator==(const TileKey&) const = default;
};

struct TileKeyHash final {
  [[nodiscard]] std::size_t operator()(const TileKey& key) const noexcept;
};

struct CellLocation final {
  TileKey tile;
  std::size_t local_x{};
  std::size_t local_y{};

  bool operator==(const CellLocation&) const = default;
};

struct CellEvidence final {
  double elevation_m{};
  double elevation_m2{};
  std::uint64_t observation_count{};
  std::int64_t last_observed_time_ns{};
  double obstacle_height_m{};
  bool valid{};
  bool obstacle{};
  bool forbidden{};

  [[nodiscard]] double elevation_variance() const noexcept {
    return observation_count > 1U
        ? elevation_m2 / static_cast<double>(observation_count - 1U)
        : 0.0;
  }

  [[nodiscard]] double observation_age_s(
      const std::int64_t simulation_time_ns) const noexcept {
    if (!valid || simulation_time_ns <= last_observed_time_ns) {
      return 0.0;
    }
    return static_cast<double>(
        simulation_time_ns - last_observed_time_ns) * 1.0e-9;
  }
};

struct ObservedPatch final {
  std::string session_id;
  std::int64_t simulation_time_ns{};
  GridCellIndex cell_zero;
  std::size_t width{};
  std::size_t height{};
  std::vector<double> elevation_m;
  std::vector<std::uint8_t> valid;
};

struct FuseResult final {
  bool accepted{};
  std::size_t updated_cells{};
  std::size_t allocated_tiles{};
  std::string reason_code;
};

struct ObservedBounds final {
  GridCellIndex minimum;
  GridCellIndex maximum;
};

enum class Occupancy : std::uint8_t {
  kUnknown = 0U,
  kFree = 1U,
  kOccupied = 2U,
};

struct CellClassification final {
  Occupancy occupancy{Occupancy::kUnknown};
  bool obstacle{};
  bool forbidden{};
  double obstacle_height_m{};
  double support_elevation_m{
      std::numeric_limits<double>::quiet_NaN()};
  double obstacle_variance{};
};

using CellNeighborhood =
    std::array<std::optional<CellEvidence>, 9U>;

}  // namespace lunar::observed_map
