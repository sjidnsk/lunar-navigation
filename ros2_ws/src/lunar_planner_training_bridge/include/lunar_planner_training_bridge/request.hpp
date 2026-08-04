#pragma once

#include <optional>
#include <string>

#include "lunar_planner_core/planner.hpp"
#include "lunar_planner_core/traversability_projection.hpp"

namespace lunar::planning::training {

struct TrainingPlanRequest final {
  std::string request_id;
  lunar::planning::TimePoint state_time;
  lunar::planning::PlatformState current_state;
  lunar::planning::GoalRegion goal;
  lunar::planning::WorldSnapshot world;
  lunar::planning::PlatformCapability capability;
  lunar::planning::PlannerConfig config;
  std::optional<lunar::planning::ExecutionContext> previous_execution;
};

class PlannerBridge final {
 public:
  [[nodiscard]] lunar::planning::PlannerOutput Plan(
      const TrainingPlanRequest &request) noexcept;
  [[nodiscard]] lunar::planning::TraversabilityProjectionResult
  ProjectTraversability(const TrainingPlanRequest &request) const {
    return lunar::planning::ProjectTraversability(
        request.world, request.capability, request.config.map_safety, {});
  }

 private:
  lunar::planning::Planner planner_;
};

}  // namespace lunar::planning::training
