#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <grid_map_msgs/msg/grid_map.hpp>

#include "lunar_planner_core/types/world_snapshot.hpp"

namespace lunar::planning::ros {

enum class GridMapErrorCode : std::uint8_t {
  kWrongFrame,
  kInvalidStamp,
  kInvalidGeometry,
  kDuplicateLayer,
  kMissingLayer,
  kMalformedLayer,
  kInvalidStartIndex,
  kNonFiniteValue,
  kOutOfRangeValue,
};

struct GridMapError final {
  GridMapErrorCode code{GridMapErrorCode::kMalformedLayer};
  std::string reason_code;
  std::string layer;
};

struct GridMapAdaptResult final {
  std::optional<lunar::planning::GridMap> map;
  std::optional<GridMapError> error;
  std::uint64_t content_identity{};

  [[nodiscard]] bool ok() const noexcept {
    return map.has_value() && !error.has_value();
  }
};

class GridMapAdapter final {
 public:
  [[nodiscard]] GridMapAdaptResult Adapt(
      const grid_map_msgs::msg::GridMap& message,
      std::string_view expected_frame) const;
};

}  // namespace lunar::planning::ros
