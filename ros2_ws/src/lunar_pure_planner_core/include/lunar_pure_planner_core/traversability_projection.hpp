#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/types/planner_config.hpp"
#include "lunar_pure_planner_core/types/platform_capability.hpp"
#include "lunar_pure_planner_core/types/world_snapshot.hpp"

namespace lunar::pure_planning {

struct TraversabilityProjection final {
  PlatformType platform_type{};
  std::size_t width{};
  std::size_t height{};
  std::vector<std::uint8_t> known;
  std::vector<std::uint8_t> intrinsic_feasible;
  std::vector<std::uint8_t> hard_feasible;
  std::vector<float> clearance_m;
  std::vector<float> slope_rad;
  std::vector<float> roughness_m;
  std::vector<float> traversal_cost;
  std::vector<std::int32_t> connected_component;
};

struct TraversabilityProjectionResult final {
  std::optional<TraversabilityProjection> projection;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return projection.has_value() && reason_code.empty();
  }
};

[[nodiscard]] TraversabilityProjectionResult ProjectTraversability(
    const WorldSnapshot& world,
    const PlatformCapability& capability,
    const MapSafetyConfig& config,
    std::stop_token stop_token);

}  // namespace lunar::pure_planning
