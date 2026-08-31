#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/types/planning_request.hpp"

namespace lunar::pure_planning {

namespace shared {
class ActivePlannerCache;
}

struct GlobalRoute final {
  std::vector<Pose3> poses_map;
  std::uint64_t expanded_states{};
};

struct GlobalStageResult final {
  std::optional<GlobalRoute> route;
  std::string reason_code;
  bool snapshot_cache_hit{};
  bool projection_cache_hit{};
  bool route_cache_hit{};
};

struct LocalStageResult final {
  LocalPlanStatus status{LocalPlanStatus::kInvalidInput};
  std::optional<MotionReferenceData> data;
  std::string reason_code;
  std::optional<std::size_t> selected_goal_index;
  bool snapshot_cache_hit{};
  bool projection_cache_hit{};
  bool goal_field_cache_hit{};
  std::uint64_t expanded_states{};
  std::optional<double> best_cost;
  LeggedLocalDiagnostics legged_local;
};

struct PlannerBackends final {
  std::function<GlobalStageResult(const PlanningRequest&, SearchControl)> global;
  std::function<LocalStageResult(const PlanningRequest&, const LocalGoalSet&,
                                 SearchControl)>
      local;
};

class Planner final {
 public:
  Planner();
  explicit Planner(PlannerBackends backends);

  [[nodiscard]] LocalStageResult PlanLocal(
      const PlanningRequest& input, const LocalGoalSet& goals_odom,
      SearchControl control) noexcept;
  [[nodiscard]] PlanningResult Plan(const PlanningRequest& input) noexcept;

 private:
  std::shared_ptr<shared::ActivePlannerCache> cache_;
  PlannerBackends backends_;
};

}  // namespace lunar::pure_planning
