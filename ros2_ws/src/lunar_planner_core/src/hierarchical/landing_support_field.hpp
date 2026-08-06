#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_planner_core/types/platform_capability.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::hierarchical {

using LandingNodeId = std::size_t;

struct LandingSupportFieldBuildResult;

class LandingSupportField final {
public:
  [[nodiscard]] bool BaseSafe(shared::GridCell cell) const noexcept;
  [[nodiscard]] bool CenterSafe(shared::GridCell cell) const noexcept;
  [[nodiscard]] std::span<const LandingNodeId> SafeCenterIds() const noexcept;
  [[nodiscard]] std::shared_ptr<const shared::MapSnapshot>
  SourceMap() const noexcept;
  [[nodiscard]] double
  SupportRadiusMeters(shared::GridCell cell) const noexcept;
  [[nodiscard]] double RequiredRadiusMeters() const noexcept;
  [[nodiscard]] std::size_t EstimatedWorkMemoryBytes() const noexcept;

private:
  friend struct LandingSupportFieldBuildResult;
  friend LandingSupportFieldBuildResult
  BuildLandingSupportField(const shared::SafeProjection &,
                           const HopperCapability &, std::stop_token);

  std::shared_ptr<const shared::MapSnapshot> map_;
  std::vector<std::uint8_t> base_safe_;
  std::vector<std::uint8_t> center_safe_;
  std::vector<double> squared_distance_cells_;
  std::vector<double> support_radius_m_;
  std::vector<LandingNodeId> safe_center_ids_;
  double required_radius_m_{};
};

struct LandingSupportFieldBuildResult final {
  std::optional<LandingSupportField> field;
  std::chrono::nanoseconds elapsed{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return field.has_value() && reason_code.empty();
  }
};

[[nodiscard]] LandingSupportFieldBuildResult
BuildLandingSupportField(const shared::SafeProjection &projection,
                         const HopperCapability &capability,
                         std::stop_token stop_token);

} // namespace lunar::planning::hierarchical
