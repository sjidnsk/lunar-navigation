#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "lunar_planner_core/types/execution_context.hpp"
#include "lunar_planner_core/types/goal.hpp"
#include "lunar_planner_core/types/motion_reference.hpp"
#include "lunar_planner_core/types/planner_config.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "lunar_planner_core/types/world_snapshot.hpp"

namespace lunar::planning {

class RouteContinuation;

enum class PlanningOutcome : std::uint8_t {
  kNewReferenceAvailable = 0,
  kSafeFrontierReferenceAvailable = 1,
  kNoKnownSafeRoute = 2,
  kGoalInfeasible = 3,
  kInvalidRequest = 4,
  kStaleInput = 5,
  kNumericalFailure = 6,
  kResourceExhausted = 7,
  kActiveReferenceInvalidated = 8,
  kCanceled = 9,
};

enum class ExecutionDirective : std::uint8_t {
  kActivateNewReference = 0,
  kContinueActiveReference = 1,
  kHoldPosition = 2,
  kContinueCommittedHop = 3,
  kNoSafeReference = 4,
};

enum class TrajectoryMode : std::uint8_t {
  kStationary,
  kOptimized,
  kDiscreteFallback,
  kCertifiedHop,
};

enum class CollisionValidation : std::uint8_t {
  kCertified,
  kNotApplicable,
};

[[nodiscard]] constexpr std::string_view ToString(
    const TrajectoryMode mode) noexcept {
  switch (mode) {
    case TrajectoryMode::kStationary:
      return "STATIONARY";
    case TrajectoryMode::kOptimized:
      return "OPTIMIZED";
    case TrajectoryMode::kDiscreteFallback:
      return "DISCRETE_FALLBACK";
    case TrajectoryMode::kCertifiedHop:
      return "CERTIFIED_HOP";
  }
  return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view ToString(
    const CollisionValidation validation) noexcept {
  switch (validation) {
    case CollisionValidation::kCertified:
      return "CERTIFIED";
    case CollisionValidation::kNotApplicable:
      return "NOT_APPLICABLE";
  }
  return "UNKNOWN";
}

struct LocalTrajectoryDiagnostics final {
  TrajectoryMode trajectory_mode{TrajectoryMode::kStationary};
  double start_anchor_error_m{0.0};
  double endpoint_error_m{0.0};
  double maximum_curvature_per_m{0.0};
  CollisionValidation collision_validation{
      CollisionValidation::kNotApplicable};
  double smoothing_elapsed_s{0.0};
  double landing_field_elapsed_s{0.0};
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

struct CertifiedHopPreview final {
  std::string segment_id;
  Pose3 launch_pose_map;
  Pose3 landing_pose_map;
  Vec3 launch_velocity_mps;
  std::chrono::nanoseconds flight_time{};
  double flight_tube_radius_m{};
  std::vector<Vec3> landing_region_map;
  std::vector<Vec3> promotion_region_map;
  double position_uncertainty_m{};
  double velocity_uncertainty_mps{};
};

struct PlannerInput final {
  std::string request_id;
  std::string mission_id;
  std::uint64_t mission_revision{};
  std::string platform_id;
  std::string capability_version;
  std::uint64_t global_map_generation{};
  std::uint64_t local_map_generation{};
  std::uint64_t map_from_odom_generation{};
  TimePoint state_time;
  PlatformState current_state;
  GoalRegion goal_map;
  WorldSnapshot world;
  PlatformCapability capability;
  PlannerConfig config;
  std::optional<ExecutionContext> previous_execution;
  std::stop_token stop_token;
  double position_uncertainty_m{};
  double velocity_uncertainty_mps{};
  std::shared_ptr<const RouteContinuation> continuation;
};

struct HierarchicalPlannerMetrics final {
  std::size_t global_level{};
  double global_resolution_m{};
  std::size_t global_cells{};
  std::chrono::nanoseconds global_elapsed{};
  std::chrono::nanoseconds local_elapsed{};
  std::uint64_t global_expanded_states{};
  std::uint64_t local_expanded_states{};
  std::size_t global_open_peak{};
  std::size_t estimated_work_memory_bytes{};
  std::size_t raw_route_points{};
  std::size_t simplified_route_points{};
  double local_frontier_distance_m{};
  std::size_t local_attempts{};
  std::size_t local_search_runs{};
  std::size_t global_replans{};
  std::size_t global_projection_cache_hits{};
  std::size_t local_projection_cache_hits{};
  double corridor_width_m{};
  std::size_t hopper_graph_nodes{};
  std::size_t hopper_graph_edges{};
  std::size_t hopper_route_hops{};
  std::size_t hopper_certification_attempts{};
  std::chrono::nanoseconds landing_field_elapsed{};
  std::chrono::nanoseconds spatial_index_elapsed{};
  std::chrono::nanoseconds ballistic_solve_elapsed{};
  std::chrono::nanoseconds flight_tube_certification_elapsed{};
  std::size_t safe_landing_nodes{};
  std::size_t candidate_edges_evaluated{};
  std::size_t coarse_edges_rejected{};
  std::size_t full_edges_certified{};
  std::size_t full_edges_invalidated{};
  std::size_t edge_certificate_cache_hits{};
  bool route_reused{};
  std::size_t route_cursor{};
  std::uint64_t rolling_request_count{1U};
};

struct PlannerDiagnostics final {
  std::string planner_name{"cpp_v3"};
  std::chrono::nanoseconds elapsed{};
  std::uint64_t expanded_states{};
  std::optional<double> best_cost;
  std::vector<std::string> warning_codes;
  std::optional<HierarchicalPlannerMetrics> hierarchical;
  std::optional<LocalTrajectoryDiagnostics> local_trajectory;
};

struct PlannerOutput final {
  PlanningOutcome outcome{PlanningOutcome::kInvalidRequest};
  ExecutionDirective directive{ExecutionDirective::kNoSafeReference};
  std::string reason_code;
  std::optional<MotionReference> reference;
  PlannerDiagnostics diagnostics;
  std::vector<CertifiedHopPreview> certified_hops;
  std::shared_ptr<const RouteContinuation> continuation;
};

} // namespace lunar::planning
