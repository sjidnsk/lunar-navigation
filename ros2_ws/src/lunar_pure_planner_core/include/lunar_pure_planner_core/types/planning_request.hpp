#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "lunar_pure_planner_core/planning_timing.hpp"
#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/traversability_map.hpp"
#include "lunar_pure_planner_core/types/goal.hpp"
#include "lunar_pure_planner_core/types/motion_reference.hpp"
#include "lunar_pure_planner_core/types/platform_capability.hpp"
#include "lunar_pure_planner_core/types/world_snapshot.hpp"

namespace lunar::pure_planning {

struct AnytimeSearchConfig final {
  std::array<double, 4> epsilon_schedule{2.5, 2.0, 1.5, 1.0};
  bool stop_after_first_solution{true};
};

struct AnytimePlannerConfig final {
  std::int32_t global_occupancy_threshold{50};
  double local_occupancy_threshold{0.5};
  double platform_discretization_m{0.2};
  double capability_cost_scale{1.0};
  std::array<double, 5> cost_weights{1.0, 1.0, 1.0, 1.0, 1.0};
  WheelPlannerMode wheel_planner_mode{WheelPlannerMode::kLegacyCertified};
  LeggedGlobalMode legged_global_mode{
      LeggedGlobalMode::kGridTraversabilityV1};
  double grid_v1_local_horizon_m{8.0};
  AnytimeSearchConfig search;
};

struct MinimalWorldSnapshot final {
  std::optional<GridMap> global_map;
  GridMap local_map;
  RigidTransform map_from_odom;
  std::uint64_t global_map_sequence{};
  std::uint64_t local_map_sequence{};
  std::uint64_t odometry_sequence{};
  std::uint64_t tf_sequence{};
  std::shared_ptr<const TraversabilitySnapshot> traversability_snapshot;
};

struct LocalGoalSet final {
  std::vector<GoalRegion> goals_odom;
  bool exact_final_goal{};
};

struct WheeledState final {
  Pose3 pose;
  Twist3 velocity;
};

struct LeggedState final {
  Pose3 body_pose;
  Twist3 body_velocity;
};

struct HopperState final {
  Pose3 pose;
  Twist3 velocity;
};

using PlatformState = std::variant<WheeledState, LeggedState, HopperState>;

enum class LocalPlanStatus : std::uint8_t {
  kSolved,
  kNoPath,
  kTimedOut,
  kCanceled,
  kInvalidInput,
  kPlannerError,
};

struct PlanningRequest final {
  std::string request_id;
  EnvironmentMode environment_mode{EnvironmentMode::kLunarSurface};
  PlatformState current_state;
  GoalRegion goal_map;
  MinimalWorldSnapshot world;
  PlatformCapability capability;
  AnytimePlannerConfig config;
  SearchControl control;
  std::optional<SteadyClock::time_point> request_started_at;
  ProgressFn progress;
};

enum class PlanningStatus : std::uint8_t {
  kSuccess,
  kInvalidInput,
  kGoalOutsideLocalMap,
  kNoPath,
  kTimedOut,
  kCanceled,
  kPlannerError,
};

struct GridV1Diagnostics final {
  bool active{};
  std::uint64_t global_input_sequence{};
  std::uint64_t local_input_sequence{};
  std::uint64_t odometry_input_sequence{};
  std::uint64_t traversability_revision{};
  std::uint64_t publish_check_revision{};
  std::uint64_t profile_hash{};
  double canonical_resolution_m{};
  Vec3 map_origin_m;
  std::size_t allocated_tiles{};
  std::size_t estimated_map_bytes{};
  std::size_t updated_cells{};
  std::size_t dirty_tiles{};
  std::size_t halo_recomputed_cells{};
  std::size_t free_cells{};
  std::size_t blocked_cells{};
  std::size_t unknown_cells{};
  std::size_t prior_conflicts{};
  bool global_route_reused{};
  std::uint64_t global_expanded_states{};
  std::size_t global_open_peak{};
  std::uint64_t local_expanded_states{};
  std::size_t local_open_peak{};
  std::size_t local_candidate_count{};
  std::size_t local_attempt_count{};
  std::size_t selected_candidate_index{};
  std::size_t raw_path_points{};
  std::size_t shortcut_path_points{};
  std::size_t resampled_path_points{};
  std::size_t final_trajectory_points{};
  std::string direction;
  double forward_cost{};
  double reverse_cost{};
  std::size_t final_supercover_cells{};
  std::string postprocess_mode;
  std::chrono::nanoseconds map_fusion_elapsed{};
  std::chrono::nanoseconds traversability_elapsed{};
  std::chrono::nanoseconds postprocess_elapsed{};
};

struct LeggedLocalDiagnostics final {
  bool active{};
  bool traversal_projection_cache_hit{};
  std::size_t fast_path_accepts{};
  std::size_t exact_sweep_fallbacks{};
  std::size_t exact_sweep_cell_checks{};
  std::size_t edge_validation_cache_hits{};
};

struct PlanningResult final {
  PlanningStatus status{PlanningStatus::kInvalidInput};
  std::string reason_code;
  std::optional<MotionReference> reference;
  GlobalRoutePreview global_route_preview;
  PlannerCallTiming timing;
  std::uint64_t expanded_states{};
  std::optional<std::size_t> selected_goal_index;
  bool global_snapshot_cache_hit{};
  bool global_projection_cache_hit{};
  bool global_route_cache_hit{};
  bool local_snapshot_cache_hit{};
  bool local_projection_cache_hit{};
  bool goal_field_cache_hit{};
  std::optional<double> best_cost;
  LeggedLocalDiagnostics legged_local;
  GridV1Diagnostics grid_v1;
};

}  // namespace lunar::pure_planning
