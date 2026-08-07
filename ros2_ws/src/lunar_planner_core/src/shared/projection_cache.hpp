#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>

#include "lunar_planner_core/types/planner_config.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "lunar_planner_core/types/world_snapshot.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "shared/terrain_checks.hpp"

namespace lunar::planning::shared {

struct ProjectionCacheKey final {
  std::uint64_t map_generation{};
  PlatformType platform_type{PlatformType::kWheeled};
  std::string platform_id;
  std::string capability_version;
  std::string frame_id;
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
  double origin_x_m{};
  double origin_y_m{};
  double origin_z_m{};
  std::uint64_t map_content_fingerprint{};
  std::uint64_t capability_fingerprint{};
  std::uint64_t safety_fingerprint{};

  bool operator==(const ProjectionCacheKey&) const = default;
};

struct ProjectionContext final {
  std::shared_ptr<const MapSnapshot> map;
  std::shared_ptr<const SafeProjection> projection;
  TerrainLimits terrain_limits;
};

struct ProjectionContextResult final {
  std::shared_ptr<const ProjectionContext> context;
  bool cache_hit{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return context != nullptr && reason_code.empty();
  }
};

[[nodiscard]] ProjectionCacheKey MakeProjectionCacheKey(
    std::uint64_t map_generation,
    std::string platform_id,
    std::string capability_version,
    const GridMap& map,
    const PlatformCapability& capability,
    const MapSafetyConfig& safety);

class ProjectionCache final {
 public:
  [[nodiscard]] ProjectionContextResult GetOrBuild(
      const ProjectionCacheKey& key,
      const GridMap& map,
      const PlatformCapability& capability,
      const MapSafetyConfig& safety,
      std::stop_token stop_token);

 private:
  std::mutex mutex_;
  std::optional<ProjectionCacheKey> key_;
  std::shared_ptr<const ProjectionContext> context_;
};

}  // namespace lunar::planning::shared
