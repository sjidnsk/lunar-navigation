#include "shared/map_snapshot.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::shared {
namespace {

enum class LayerValueType {
  kInt8,
  kFloat,
  kByte,
  kCount,
};

[[nodiscard]] MapSnapshotBuildResult Failure(
    std::string reason_code) {
  return MapSnapshotBuildResult{
      .snapshot = nullptr,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool IsFinite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool IsExpectedType(
    const GridLayer& layer, const LayerValueType type) noexcept {
  switch (type) {
    case LayerValueType::kInt8:
      return std::holds_alternative<std::vector<std::int8_t>>(layer.values);
    case LayerValueType::kFloat:
      return std::holds_alternative<std::vector<float>>(layer.values);
    case LayerValueType::kByte:
      return std::holds_alternative<std::vector<std::uint8_t>>(layer.values);
    case LayerValueType::kCount:
      return std::holds_alternative<std::vector<std::uint32_t>>(layer.values);
  }
  return false;
}

}  // namespace

MapSnapshotBuildResult MapSnapshot::Create(
    GridMap map, const MapContract contract) {
  if (map.frame_id.empty() || map.CellCount() == 0U ||
      map.width >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      map.height >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      !std::isfinite(map.resolution_m) || map.resolution_m <= 0.0 ||
      !IsFinite(map.origin_m)) {
    return Failure("MAP_FORMAT_INVALID");
  }

  const auto valid_layer = [&](const std::string_view name,
                               const LayerValueType type) {
    const auto found = map.layers.find(name);
    return found != map.layers.end() && IsExpectedType(found->second, type) &&
           found->second.size() == map.CellCount();
  };
  const bool valid_contract =
      contract == MapContract::kGlobalOccupancy
          ? valid_layer("occupancy", LayerValueType::kInt8)
          : valid_layer("occupancy", LayerValueType::kFloat) &&
                valid_layer("elevation", LayerValueType::kFloat);
  if (!valid_contract) {
    return Failure("MAP_FORMAT_INVALID");
  }

  return MapSnapshotBuildResult{
      .snapshot = std::shared_ptr<const MapSnapshot>{
          new MapSnapshot(std::move(map), contract)},
      .reason_code = {},
  };
}

MapSnapshotBuildResult MapSnapshot::Create(
    const GridMap& map, const MapContract contract, SearchControl control) {
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return Failure(std::string{*stopped});
  }

  GridMap copied{
      .frame_id = map.frame_id,
      .stamp = map.stamp,
      .width = map.width,
      .height = map.height,
      .resolution_m = map.resolution_m,
      .origin_m = map.origin_m,
      .layers = {},
  };
  std::string stopped_reason;
  for (const auto& [name, layer] : map.layers) {
    GridLayer copied_layer;
    copied_layer.values = std::visit(
        [&](const auto& source_values) -> GridLayerValues {
          using Values = std::decay_t<decltype(source_values)>;
          Values values;
          values.reserve(source_values.size());
          for (std::size_t offset = 0U; offset < source_values.size();
               offset += kControlCheckStride) {
            if (const auto stopped = StopReason(control);
                stopped.has_value()) {
              stopped_reason = *stopped;
              return values;
            }
            const std::size_t finish = std::min(
                source_values.size(), offset + kControlCheckStride);
            values.insert(values.end(), source_values.begin() + offset,
                          source_values.begin() + finish);
          }
          return values;
        },
        layer.values);
    if (!stopped_reason.empty()) {
      return Failure(std::move(stopped_reason));
    }
    copied.layers.emplace(name, std::move(copied_layer));
  }
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return Failure(std::string{*stopped});
  }
  return Create(std::move(copied), contract);
}

MapSnapshot::MapSnapshot(GridMap map, const std::optional<MapContract> contract)
    : map_(std::move(map)), contract_(contract) {}

std::string_view MapSnapshot::frame_id() const noexcept {
  return map_.frame_id;
}

TimePoint MapSnapshot::stamp() const noexcept {
  return map_.stamp;
}

std::size_t MapSnapshot::width() const noexcept {
  return map_.width;
}

std::size_t MapSnapshot::height() const noexcept {
  return map_.height;
}

std::size_t MapSnapshot::cell_count() const noexcept {
  return map_.CellCount();
}

double MapSnapshot::resolution_m() const noexcept {
  return map_.resolution_m;
}

const Vec3& MapSnapshot::origin_m() const noexcept {
  return map_.origin_m;
}

bool MapSnapshot::InBounds(const GridCell cell) const noexcept {
  return cell.x >= 0 && cell.y >= 0 &&
         static_cast<std::size_t>(cell.x) < map_.width &&
         static_cast<std::size_t>(cell.y) < map_.height;
}

std::size_t MapSnapshot::Index(const GridCell cell) const noexcept {
  if (!InBounds(cell)) {
    return cell_count();
  }
  return static_cast<std::size_t>(cell.y) * map_.width +
         static_cast<std::size_t>(cell.x);
}

std::optional<GridCell> MapSnapshot::PositionToCell(
    const Vec2 position_m) const noexcept {
  if (!std::isfinite(position_m.x) || !std::isfinite(position_m.y)) {
    return std::nullopt;
  }
  const double relative_x = (position_m.x - map_.origin_m.x) / map_.resolution_m;
  const double relative_y = (position_m.y - map_.origin_m.y) / map_.resolution_m;
  if (!std::isfinite(relative_x) || !std::isfinite(relative_y) ||
      relative_x < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
      relative_x > static_cast<double>(std::numeric_limits<std::int32_t>::max()) ||
      relative_y < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
      relative_y > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    return std::nullopt;
  }
  const GridCell cell{
      .x = static_cast<std::int32_t>(std::floor(relative_x)),
      .y = static_cast<std::int32_t>(std::floor(relative_y)),
  };
  return InBounds(cell) ? std::optional<GridCell>{cell} : std::nullopt;
}

std::optional<double> MapSnapshot::SampleElevationBilinear(
    const Vec2 position_m) const noexcept {
  if (!std::isfinite(position_m.x) || !std::isfinite(position_m.y)) {
    return std::nullopt;
  }
  const double sample_x =
      (position_m.x - map_.origin_m.x) / map_.resolution_m - 0.5;
  const double sample_y =
      (position_m.y - map_.origin_m.y) / map_.resolution_m - 0.5;
  if (!std::isfinite(sample_x) || !std::isfinite(sample_y)) {
    return std::nullopt;
  }
  const auto x0 = static_cast<std::int64_t>(std::floor(sample_x));
  const auto y0 = static_cast<std::int64_t>(std::floor(sample_y));
  const double fraction_x = sample_x - static_cast<double>(x0);
  const double fraction_y = sample_y - static_cast<double>(y0);
  const auto elevation = FloatLayer("elevation");
  if (elevation.size() != cell_count()) {
    return std::nullopt;
  }
  double result = 0.0;
  double total_weight = 0.0;
  for (std::int64_t dy = 0; dy <= 1; ++dy) {
    const double weight_y = dy == 0 ? 1.0 - fraction_y : fraction_y;
    for (std::int64_t dx = 0; dx <= 1; ++dx) {
      const double weight_x = dx == 0 ? 1.0 - fraction_x : fraction_x;
      const double weight = weight_x * weight_y;
      if (weight <= 1.0e-15) {
        continue;
      }
      const auto x = x0 + dx;
      const auto y = y0 + dy;
      if (x < 0 || y < 0 ||
          x > static_cast<std::int64_t>(
                  std::numeric_limits<std::int32_t>::max()) ||
          y > static_cast<std::int64_t>(
                  std::numeric_limits<std::int32_t>::max())) {
        return std::nullopt;
      }
      const GridCell cell{.x = static_cast<std::int32_t>(x),
                          .y = static_cast<std::int32_t>(y)};
      if (!InBounds(cell)) {
        return std::nullopt;
      }
      const std::size_t index = Index(cell);
      if (!std::isfinite(elevation[index])) {
        return std::nullopt;
      }
      result += weight * static_cast<double>(elevation[index]);
      total_weight += weight;
    }
  }
  if (total_weight <= 0.0 || !std::isfinite(result)) {
    return std::nullopt;
  }
  const double interpolated = result / total_weight;
  return std::isfinite(interpolated)
             ? std::optional<double>{interpolated}
             : std::nullopt;
}

Vec3 MapSnapshot::CellCenter(const GridCell cell) const noexcept {
  if (!InBounds(cell)) {
    return {};
  }
  const std::size_t index = Index(cell);
  const auto elevation = FloatLayer("elevation");
  const double elevation_m =
      contract_ != MapContract::kGlobalOccupancy &&
              elevation.size() == cell_count() &&
              std::isfinite(elevation[index])
          ? static_cast<double>(elevation[index])
          : map_.origin_m.z;
  return Vec3{
      .x = map_.origin_m.x +
           (static_cast<double>(cell.x) + 0.5) * map_.resolution_m,
      .y = map_.origin_m.y +
           (static_cast<double>(cell.y) + 0.5) * map_.resolution_m,
      .z = elevation_m,
  };
}

std::span<const float> MapSnapshot::FloatLayer(
    const std::string_view name) const noexcept {
  const auto found = map_.layers.find(name);
  if (found == map_.layers.end()) {
    return {};
  }
  const auto* values = std::get_if<std::vector<float>>(&found->second.values);
  return values == nullptr ? std::span<const float>{}
                           : std::span<const float>{*values};
}

std::span<const std::int8_t> MapSnapshot::Int8Layer(
    const std::string_view name) const noexcept {
  const auto found = map_.layers.find(name);
  if (found == map_.layers.end()) {
    return {};
  }
  const auto* values =
      std::get_if<std::vector<std::int8_t>>(&found->second.values);
  return values == nullptr ? std::span<const std::int8_t>{}
                           : std::span<const std::int8_t>{*values};
}

std::span<const std::uint8_t> MapSnapshot::ByteLayer(
    const std::string_view name) const noexcept {
  const auto found = map_.layers.find(name);
  if (found == map_.layers.end()) {
    return {};
  }
  const auto* values =
      std::get_if<std::vector<std::uint8_t>>(&found->second.values);
  return values == nullptr ? std::span<const std::uint8_t>{}
                           : std::span<const std::uint8_t>{*values};
}

std::span<const std::uint32_t> MapSnapshot::CountLayer(
    const std::string_view name) const noexcept {
  const auto found = map_.layers.find(name);
  if (found == map_.layers.end()) {
    return {};
  }
  const auto* values =
      std::get_if<std::vector<std::uint32_t>>(&found->second.values);
  return values == nullptr ? std::span<const std::uint32_t>{}
                           : std::span<const std::uint32_t>{*values};
}

}  // namespace lunar::pure_planning::shared
