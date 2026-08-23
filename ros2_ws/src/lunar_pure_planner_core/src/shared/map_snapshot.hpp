#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/types/world_snapshot.hpp"

namespace lunar::pure_planning::shared {

enum class MapContract : std::uint8_t {
  kGlobalOccupancy,
  kLocalElevation,
};

struct GridCell final {
  std::int32_t x{};
  std::int32_t y{};

  auto operator<=>(const GridCell&) const = default;
};

class MapSnapshot;

struct MapSnapshotBuildResult final {
  std::shared_ptr<const MapSnapshot> snapshot;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return snapshot != nullptr && reason_code.empty();
  }
};

class MapSnapshot final {
 public:
  [[nodiscard]] static MapSnapshotBuildResult Create(
      GridMap map, MapContract contract);
  [[nodiscard]] static MapSnapshotBuildResult Create(
      const GridMap& map, MapContract contract, SearchControl control);

  [[nodiscard]] std::string_view frame_id() const noexcept;
  [[nodiscard]] TimePoint stamp() const noexcept;
  [[nodiscard]] std::size_t width() const noexcept;
  [[nodiscard]] std::size_t height() const noexcept;
  [[nodiscard]] std::size_t cell_count() const noexcept;
  [[nodiscard]] double resolution_m() const noexcept;
  [[nodiscard]] const Vec3& origin_m() const noexcept;
  [[nodiscard]] bool InBounds(GridCell cell) const noexcept;
  [[nodiscard]] std::size_t Index(GridCell cell) const noexcept;
  [[nodiscard]] std::optional<GridCell> PositionToCell(
      Vec2 position_m) const noexcept;
  [[nodiscard]] std::optional<double> SampleElevationBilinear(
      Vec2 position_m) const noexcept;
  [[nodiscard]] Vec3 CellCenter(GridCell cell) const noexcept;
  [[nodiscard]] std::span<const float> FloatLayer(
      std::string_view name) const noexcept;
  [[nodiscard]] std::span<const std::int8_t> Int8Layer(
      std::string_view name) const noexcept;
  [[nodiscard]] std::span<const std::uint8_t> ByteLayer(
      std::string_view name) const noexcept;
  [[nodiscard]] std::span<const std::uint32_t> CountLayer(
      std::string_view name) const noexcept;

 private:
  explicit MapSnapshot(GridMap map,
                       std::optional<MapContract> contract = std::nullopt);

  GridMap map_;
  std::optional<MapContract> contract_;
};

}  // namespace lunar::pure_planning::shared
