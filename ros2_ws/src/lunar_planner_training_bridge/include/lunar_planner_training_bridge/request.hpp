#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "lunar_planner_core/planner.hpp"
#include "lunar_planner_core/primitive_reachability.hpp"
#include "lunar_planner_core/reachability_projection.hpp"
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
  [[nodiscard]] lunar::planning::ReachabilityProjectionResult
  ProjectReachability(const TrainingPlanRequest &request,
                      double maximum_edge_distance_m) const noexcept;
  [[nodiscard]] lunar::planning::ReachabilityProjectionResult
  ProjectReachability(
      const TrainingPlanRequest &request,
      double maximum_edge_distance_m,
      const lunar::planning::HopperLandingEvidenceGrid &evidence)
      const noexcept;
  [[nodiscard]] lunar::planning::ReachabilityProjectionResult
  ProjectDirectHopperReachability(
      const TrainingPlanRequest &request,
      double maximum_edge_distance_m,
      const lunar::planning::HopperLandingEvidenceGrid &evidence)
      const noexcept;
  [[nodiscard]] lunar::planning::HopperOpportunityContextResult
  ProjectHopperOpportunityContext(
      const TrainingPlanRequest &request,
      double maximum_edge_distance_m,
      const lunar::planning::HopperLandingEvidenceGrid &evidence)
      const noexcept;
  [[nodiscard]] lunar::planning::HopperOpportunityDistanceProjectionResult
  QueryHopperOpportunityDistance(
      lunar::planning::HopperOpportunityContext &context,
      const std::vector<std::uint8_t> &positive_opportunities) const noexcept;
  [[nodiscard]] lunar::planning::HopperLandingEvidenceProjectionResult
  ProjectHopperLandingEvidence(
      const TrainingPlanRequest &request,
      const std::vector<lunar::planning::Vec3> &target_positions_map)
      const noexcept;
  [[nodiscard]] lunar::planning::TraversabilityProjectionResult
  ProjectTraversability(const TrainingPlanRequest &request) const {
    return lunar::planning::ProjectTraversability(
        request.world, request.capability, request.config.map_safety, {});
  }

 private:
  lunar::planning::Planner planner_;
};

class TrainingPrimitiveReachabilityEngine final {
 public:
  [[nodiscard]] lunar::planning::PrimitiveReachabilityResult Update(
      const TrainingPlanRequest& request,
      std::optional<double> maximum_action_distance_m) noexcept;
  [[nodiscard]] lunar::planning::PrimitiveReachabilityResult Update(
      const TrainingPlanRequest& request,
      std::optional<double> maximum_action_distance_m,
      const lunar::planning::HopperLandingEvidenceGrid&
          hopper_landing_evidence) noexcept;
  void Reset() noexcept;

 private:
  lunar::planning::PrimitiveReachabilityEngine engine_;
};

}  // namespace lunar::planning::training
