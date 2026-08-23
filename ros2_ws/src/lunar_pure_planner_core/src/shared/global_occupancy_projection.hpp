#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::shared {

struct GlobalOccupancyProjectionBuildResult;

struct GlobalOccupancyProjectionView final {
  const MapSnapshot* map{};
  std::span<const std::uint8_t> hard_feasible;
  std::span<const float> clearance_m;

  [[nodiscard]] bool Valid() const noexcept;
  [[nodiscard]] bool HardFeasible(GridCell cell) const noexcept;
  [[nodiscard]] float ClearanceMeters(GridCell cell) const noexcept;
};

class GlobalOccupancyProjection final {
 public:
  [[nodiscard]] GlobalOccupancyProjectionView View() const & noexcept;
  GlobalOccupancyProjectionView View() const && = delete;

 private:
  friend GlobalOccupancyProjectionBuildResult BuildGlobalOccupancyProjection(
      std::shared_ptr<const MapSnapshot>, std::int32_t, SearchControl);
  friend GlobalOccupancyProjectionBuildResult
  BuildInflatedGlobalOccupancyProjection(std::shared_ptr<const MapSnapshot>,
                                         std::int32_t, double, SearchControl);

  std::shared_ptr<const MapSnapshot> source_map_;
  std::vector<std::uint8_t> hard_feasible_;
  std::vector<float> clearance_m_;
};

struct GlobalOccupancyProjectionBuildResult final {
  std::optional<GlobalOccupancyProjection> projection;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return projection.has_value() && reason_code.empty();
  }
};

[[nodiscard]] GlobalOccupancyProjectionBuildResult
BuildGlobalOccupancyProjection(std::shared_ptr<const MapSnapshot> map,
                               std::int32_t obstacle_threshold_percent = 50,
                               SearchControl control = {});

[[nodiscard]] GlobalOccupancyProjectionBuildResult
BuildInflatedGlobalOccupancyProjection(
    std::shared_ptr<const MapSnapshot> map,
    std::int32_t obstacle_threshold_percent, double inflation_m,
    SearchControl control = {});

}  // namespace lunar::pure_planning::shared
