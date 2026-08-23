#pragma once

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/types/planning_request.hpp"
#include "shared/local_terrain_projection.hpp"
#include "shared/obstacle_height_estimator.hpp"
#include "wheel/anytime_wheel_planner.hpp"

namespace lunar::pure_planning::hopper {

struct HopperPlanRequest final {
  HopperState start;
  GoalRegion goal_odom;
  const shared::LocalTerrainProjection* terrain{};
  const HopperCapability* capability{};
  SearchControl control;
  AnytimeSearchConfig search;
  std::function<shared::ObstacleHeight(const shared::MapSnapshot&,
                                       shared::GridCell)>
      estimate_height;
};

struct HopperPlanResult final {
  LocalPlanStatus status{LocalPlanStatus::kInvalidInput};
  std::string reason_code;
  std::vector<HopSegment> hops;
  double cost{std::numeric_limits<double>::infinity()};
  LocalPlanMetrics metrics;

  [[nodiscard]] bool ok() const noexcept {
    return status == LocalPlanStatus::kSolved;
  }
};

[[nodiscard]] HopperPlanResult PlanHopper(const HopperPlanRequest& request);

}  // namespace lunar::pure_planning::hopper
