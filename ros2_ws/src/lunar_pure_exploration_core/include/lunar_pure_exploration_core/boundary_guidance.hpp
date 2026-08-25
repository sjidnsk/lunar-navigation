#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lunar_pure_exploration_core/candidate_ranker.hpp"
#include "lunar_pure_exploration_core/goal_identity.hpp"
#include "lunar_pure_exploration_core/information_gain.hpp"
#include "lunar_pure_exploration_core/safe_pose_validator.hpp"

namespace lunar::pure_exploration {

enum class NavigationPhase : std::uint8_t { kApproachTask, kExploreTask };

enum class ApproachWaitReason : std::uint8_t {
  kNone,
  kWaitingForSafeStart,
  kWaitingForTaskMapCoverage,
  kNoGuidanceRoute,
  kNoSafeCandidate,
};

struct GuidanceCost {
  std::uint32_t unknown_cell_count;
  double path_length_m;
};

struct ApproachIntent {
  GridIndex cell;
  GuidanceCost total_cost;
  std::vector<GridIndex> route;
};

struct ApproachCandidate {
  std::uint64_t id;
  BoundaryApproachGoalIdentity identity;
  Pose2 pose;
  GuidanceCost remaining_cost;
  double task_unknown_area_m2;
  std::uint32_t guidance_unknown_cell_count;
  bool fully_inside_task;
  std::vector<Vec2> guidance_route;
};

struct BoundaryGuidanceResult {
  NavigationPhase phase;
  ApproachWaitReason wait_reason;
  bool fully_inside_task;
  std::vector<ApproachIntent> intents;
  std::vector<ApproachCandidate> candidates;
  std::size_t consumed_work_units;
};

class BoundaryGuidance final {
 public:
  struct Limits {
    std::size_t maximum_guidance_grid_cells;
    std::size_t maximum_guidance_work_units;
    std::size_t maximum_approach_candidates;
  };

  BoundaryGuidance(PlatformGeometry platform, SensorModel sensor_model,
                   double goal_yaw_tolerance_rad, Limits limits);

  BoundaryGuidanceResult Build(const OccupancyGridView& map,
                               const TaskRaster& raster,
                               Pose2 robot_pose) const;

  std::vector<std::size_t> CoarseOrder(
      const BoundaryGuidanceResult& result, Pose2 robot_pose) const;

  std::vector<std::size_t> FinalOrder(
      const BoundaryGuidanceResult& result,
      std::span<const PlannedCandidate> planned,
      Pose2 robot_pose) const;

  bool IsCandidateStillValid(const OccupancyGridView& latest_map,
                             const TaskRaster& latest_raster,
                             const ApproachCandidate& frozen_candidate) const;

 private:
  SafePoseValidator validator_;
  SensorModel sensor_model_;
  double goal_yaw_tolerance_rad_;
  Limits limits_;
};

}  // namespace lunar::pure_exploration
