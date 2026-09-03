#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "lunar_incremental_navigation_core/types/geometry.hpp"

namespace lunar::incremental_navigation {

using GridLayerValues = std::variant<
    std::vector<std::int8_t>,
    std::vector<float>,
    std::vector<std::uint8_t>,
    std::vector<std::uint32_t>>;

struct GridLayer final {
  GridLayerValues values;

  [[nodiscard]] std::size_t size() const noexcept {
    return std::visit([](const auto& typed_values) { return typed_values.size(); }, values);
  }
};

struct GridMap final {
  std::string frame_id;
  TimePoint stamp;
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
  Vec3 origin_m;
  std::map<std::string, GridLayer, std::less<>> layers;

  [[nodiscard]] std::size_t CellCount() const noexcept {
    if (width == 0U || height == 0U ||
        height > std::numeric_limits<std::size_t>::max() / width) {
      return 0U;
    }
    return width * height;
  }

  [[nodiscard]] bool HasLayer(const std::string_view name) const {
    return layers.contains(name);
  }

  [[nodiscard]] bool HasConsistentLayerSizes() const noexcept {
    const std::size_t expected = CellCount();
    if (expected == 0U) {
      return false;
    }
    for (const auto& [name, layer] : layers) {
      static_cast<void>(name);
      if (layer.size() != expected) {
        return false;
      }
    }
    return true;
  }
};

struct RigidTransform final {
  std::string parent_frame;
  std::string child_frame;
  TimePoint stamp;
  Vec3 translation_m;
  Quaternion rotation;
};

struct WorldSnapshot final {
  GridMap global_map;
  GridMap local_map;
  RigidTransform map_from_odom;
};

}  // namespace lunar::incremental_navigation
