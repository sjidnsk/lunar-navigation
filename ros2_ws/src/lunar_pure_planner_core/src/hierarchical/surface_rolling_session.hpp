#pragma once

#include <optional>

#include "lunar_pure_planner_core/planner.hpp"

namespace lunar::pure_planning::hierarchical {

struct SurfaceRollingConfig final {
  double horizon_m{8.0};
  double max_deviation_m{2.0};
};

struct SurfaceRollingDecision final {
  enum class Kind {
    kNextGoal,
    kFinalGoalReached,
    kInvalidRoute,
  };

  Kind kind{Kind::kInvalidRoute};
  std::optional<GoalRegion> goal;
  double lateral_deviation_m{};
};

class SurfaceRollingSession final {
 public:
  SurfaceRollingSession(GlobalRoute route, GoalRegion final_goal,
                        SurfaceRollingConfig config);

  [[nodiscard]] SurfaceRollingDecision Decide(const Pose3& pose_map) const;

 private:
  GlobalRoute route_;
  GoalRegion final_goal_;
  SurfaceRollingConfig config_;
};

}  // namespace lunar::pure_planning::hierarchical
