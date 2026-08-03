#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "lunar_planner_core/types/geometry.hpp"

namespace lunar::planning {

struct PointGoal final {
  Vec3 position_m;
  double tolerance_m{};
};

struct PlanarRegionGoal final {
  std::vector<Vec3> boundary_m;
  double normal_tolerance_m{};
};

using GoalTarget = std::variant<PointGoal, PlanarRegionGoal>;

struct GoalRegion final {
  std::string goal_id;
  GoalTarget target;
  std::optional<double> yaw_rad;
  double yaw_tolerance_rad{};
};

}  // namespace lunar::planning
