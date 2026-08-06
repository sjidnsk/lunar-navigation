#include "shared/map_snapshot.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lunar::planning::shared {
namespace {

enum class LayerValueType {
  kFloat,
  kByte,
  kCount,
};

struct RequiredLayer final {
  std::string_view name;
  LayerValueType type;
};

constexpr std::array<RequiredLayer, 10> kRequiredLayers{
    RequiredLayer{"elevation", LayerValueType::kFloat},
    RequiredLayer{"valid_mask", LayerValueType::kByte},
    RequiredLayer{"obstacle", LayerValueType::kByte},
    RequiredLayer{"obstacle_height", LayerValueType::kFloat},
    RequiredLayer{"observation_age_s", LayerValueType::kFloat},
    RequiredLayer{"observation_quality", LayerValueType::kFloat},
    RequiredLayer{"elevation_variance", LayerValueType::kFloat},
    RequiredLayer{"obstacle_variance", LayerValueType::kFloat},
    RequiredLayer{"observation_count", LayerValueType::kCount},
    RequiredLayer{"forbidden", LayerValueType::kByte},
};

[[nodiscard]] std::string UpperName(const std::string_view name) {
  std::string result;
  result.reserve(name.size());
  for (const char character : name) {
    result.push_back(static_cast<char>(
        std::toupper(static_cast<unsigned char>(character))));
  }
  return result;
}

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
    case LayerValueType::kFloat:
      return std::holds_alternative<std::vector<float>>(layer.values);
    case LayerValueType::kByte:
      return std::holds_alternative<std::vector<std::uint8_t>>(layer.values);
    case LayerValueType::kCount:
      return std::holds_alternative<std::vector<std::uint32_t>>(layer.values);
  }
  return false;
}

[[nodiscard]] bool AllFinite(const std::vector<float>& values) noexcept {
  return std::ranges::all_of(values, [](const float value) {
    return std::isfinite(value);
  });
}

[[nodiscard]] bool AllNonNegative(
    const std::vector<float>& values) noexcept {
  return std::ranges::all_of(values, [](const float value) {
    return value >= 0.0F;
  });
}

[[nodiscard]] bool AllUnitInterval(
    const std::vector<float>& values) noexcept {
  return std::ranges::all_of(values, [](const float value) {
    return value >= 0.0F && value <= 1.0F;
  });
}

[[nodiscard]] bool AllBinary(
    const std::vector<std::uint8_t>& values) noexcept {
  return std::ranges::all_of(values, [](const std::uint8_t value) {
    return value <= 1U;
  });
}

}  // namespace

MapSnapshotBuildResult MapSnapshot::Create(const GridMap& map) {
  if (map.frame_id.empty()) {
    return Failure("MAP_FRAME_EMPTY");
  }
  if (map.stamp.nanoseconds_since_epoch <= 0) {
    return Failure("MAP_STAMP_INVALID");
  }
  if (map.CellCount() == 0U) {
    return Failure("MAP_GEOMETRY_EMPTY");
  }
  if (map.width >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      map.height >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return Failure("MAP_GEOMETRY_INDEX_RANGE");
  }
  if (!std::isfinite(map.resolution_m) || map.resolution_m <= 0.0 ||
      !IsFinite(map.origin_m)) {
    return Failure("MAP_GEOMETRY_INVALID");
  }

  const std::size_t expected_size = map.CellCount();
  for (const RequiredLayer& required : kRequiredLayers) {
    const auto found = map.layers.find(required.name);
    const std::string upper_name = UpperName(required.name);
    if (found == map.layers.end()) {
      return Failure("MISSING_MAP_LAYER_" + upper_name);
    }
    if (!IsExpectedType(found->second, required.type)) {
      return Failure("MAP_LAYER_TYPE_" + upper_name);
    }
    if (found->second.size() != expected_size) {
      return Failure("MAP_LAYER_SIZE_" + upper_name);
    }
  }

  for (const RequiredLayer& required : kRequiredLayers) {
    if (required.type != LayerValueType::kFloat) {
      continue;
    }
    const auto& values = std::get<std::vector<float>>(
        map.layers.find(required.name)->second.values);
    if (!AllFinite(values)) {
      return Failure("MAP_NONFINITE_" + UpperName(required.name));
    }
  }

  for (const std::string_view layer : {
           "obstacle_height", "observation_age_s", "elevation_variance",
           "obstacle_variance"}) {
    const auto& values =
        std::get<std::vector<float>>(map.layers.find(layer)->second.values);
    if (!AllNonNegative(values)) {
      return Failure("MAP_RANGE_" + UpperName(layer));
    }
  }
  const auto& quality = std::get<std::vector<float>>(
      map.layers.at("observation_quality").values);
  if (!AllUnitInterval(quality)) {
    return Failure("MAP_RANGE_OBSERVATION_QUALITY");
  }
  for (const std::string_view layer : {
           "valid_mask", "obstacle", "forbidden"}) {
    const auto& values = std::get<std::vector<std::uint8_t>>(
        map.layers.find(layer)->second.values);
    if (!AllBinary(values)) {
      return Failure("MAP_RANGE_" + UpperName(layer));
    }
  }

  return MapSnapshotBuildResult{
      .snapshot = std::shared_ptr<const MapSnapshot>{new MapSnapshot(map)},
      .reason_code = {},
  };
}

MapSnapshot::MapSnapshot(GridMap map) : map_(std::move(map)) {}

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
  const auto valid = ByteLayer("valid_mask");
  const auto elevation = FloatLayer("elevation");
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
      if (valid[index] == 0U) {
        return std::nullopt;
      }
      result += weight * static_cast<double>(elevation[index]);
      total_weight += weight;
    }
  }
  return total_weight > 0.0
             ? std::optional<double>{result / total_weight}
             : std::nullopt;
}

Vec3 MapSnapshot::CellCenter(const GridCell cell) const noexcept {
  if (!InBounds(cell)) {
    return {};
  }
  const std::size_t index = Index(cell);
  const auto elevation = FloatLayer("elevation");
  return Vec3{
      .x = map_.origin_m.x +
           (static_cast<double>(cell.x) + 0.5) * map_.resolution_m,
      .y = map_.origin_m.y +
           (static_cast<double>(cell.y) + 0.5) * map_.resolution_m,
      .z = static_cast<double>(elevation[index]),
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

}  // namespace lunar::planning::shared
