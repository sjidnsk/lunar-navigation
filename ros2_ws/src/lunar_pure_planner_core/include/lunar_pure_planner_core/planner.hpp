#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/types/planning_request.hpp"

namespace lunar::pure_planning {

struct GlobalRoute final {
  std::vector<Pose3> poses_map;
  std::uint64_t expanded_states{};
};

struct GlobalStageResult final {
  std::optional<GlobalRoute> route;
  std::string reason_code;
};

struct LocalStageResult final {
  LocalPlanStatus status{LocalPlanStatus::kInvalidInput};
  std::optional<MotionReferenceData> data;
  std::string reason_code;
  std::optional<std::size_t> selected_goal_index;
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

  [[nodiscard]] PlanningResult Plan(const PlanningRequest& input) noexcept;

 private:
  PlannerBackends backends_;
};

}  // namespace lunar::pure_planning
