#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "lunar_pure_planner_core/planning_timing.hpp"
#include "lunar_pure_planner_core/search_control.hpp"
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

struct PlanningResult final {
  PlanningStatus status{PlanningStatus::kInvalidInput};
  std::string reason_code;
  std::optional<MotionReference> reference;
  PlannerCallTiming timing;
  std::uint64_t expanded_states{};
  std::optional<std::size_t> selected_goal_index;
  bool global_snapshot_cache_hit{};
  bool global_projection_cache_hit{};
  bool global_route_cache_hit{};
  bool local_snapshot_cache_hit{};
  bool local_projection_cache_hit{};
  bool goal_field_cache_hit{};
};

}  // namespace lunar::pure_planning
