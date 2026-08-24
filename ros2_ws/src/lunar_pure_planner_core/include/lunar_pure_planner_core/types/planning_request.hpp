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

struct WheelPlanningMetrics final {
  std::uint64_t expanded_states{};
  std::size_t edge_validation_evaluations{};
  std::size_t edge_validation_cache_hits{};
  std::size_t broad_phase_rejects{};
  std::size_t full_certifications{};
  std::size_t full_invalidations{};
  std::size_t sweep_cell_checks{};
  std::size_t quantization_alias_states{};
  std::size_t quantized_state_reuses{};
  std::size_t quantized_endpoint_aliases{};
  std::size_t quantized_state_count{};
  std::size_t maximum_active_labels_per_key{};
  bool used_narrow_resolution{};
  double finest_xy_key_resolution_m{};
  std::size_t maximum_yaw_bins{};
  std::size_t ara_search_invocations{};
  std::size_t returned_edge_certificate_confirmations{};
  std::size_t mode_switch_edge_count{};
  std::size_t reverse_edge_count{};
  double start_heuristic_lower_bound{};
  bool has_certified_preferred_candidate{};
  std::size_t preferred_candidate_full_primitive_edge_count{};
  std::size_t preferred_candidate_terminal_connector_edge_count{};
  std::size_t preferred_candidate_certified_edge_count{};
  double preferred_candidate_cost{};
  std::size_t preferred_builder_invocations{};
  std::array<double, 5U> cost_components{};
  std::array<double, 5U> cost_scales{};
  std::size_t direct_unknown_or_unsupported_footprint_rejects{};
  std::size_t measured_obstacle_clearance_rejects{};
  std::size_t slope_or_roughness_rejects{};
  std::size_t relief_or_underbody_rejects{};
  std::size_t dynamics_or_primitive_shape_rejects{};
  std::size_t deadline_or_cancellation_interruptions{};
  std::size_t far_clearance_scan_skips{};
  std::size_t occupied_clearance_cell_checks{};
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
  std::optional<double> best_cost;
  std::optional<WheelPlanningMetrics> wheel_metrics;
};

}  // namespace lunar::pure_planning
