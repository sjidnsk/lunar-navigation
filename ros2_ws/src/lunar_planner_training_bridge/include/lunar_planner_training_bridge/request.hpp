#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "lunar_planner_core/planner.hpp"
#include "lunar_planner_core/traversability_projection.hpp"

namespace lunar::planning::training {

struct TrainingPlanRequest final {
  std::string request_id;
  std::string mission_id;
  std::uint64_t mission_revision{};
  std::string platform_id;
  std::string capability_version;
  std::uint64_t global_map_generation{};
  std::uint64_t local_map_generation{};
  std::uint64_t map_from_odom_generation{};
  lunar::planning::TimePoint state_time;
  lunar::planning::PlatformState current_state;
  lunar::planning::GoalRegion goal;
  lunar::planning::WorldSnapshot world;
  lunar::planning::PlatformCapability capability;
  lunar::planning::PlannerConfig config;
  std::optional<lunar::planning::ExecutionContext> previous_execution;
  double position_uncertainty_m{};
  double velocity_uncertainty_mps{};
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
