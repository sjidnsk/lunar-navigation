#include "support/complex_terrain_benchmark.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
#include "lunar_incremental_navigation_core/global_guidance_builder.hpp"
#include "lunar_incremental_navigation_core/global_route_planner.hpp"
#include "lunar_incremental_navigation_core/legged_local_planner.hpp"
#include "lunar_incremental_navigation_core/local_target_selector.hpp"
#include "lunar_incremental_navigation_core/planning_session_coordinator.hpp"
#include "lunar_incremental_navigation_core/request_local_start_patch.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"
#include "local/local_target_selector_internal.hpp"

namespace lunar::incremental_navigation::test_support {
namespace {

[[nodiscard]] double ElapsedMilliseconds(
    const SteadyClock::time_point started) {
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - started)
      .count();
}

[[nodiscard]] double Milliseconds(const std::chrono::nanoseconds duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

[[nodiscard]] std::string StatusName(
    const ElevationUpdateResult::Status status) {
  switch (status) {
    case ElevationUpdateResult::Status::kApplied:
      return "APPLIED";
    case ElevationUpdateResult::Status::kDuplicate:
      return "DUPLICATE";
    case ElevationUpdateResult::Status::kRejected:
      return "REJECTED";
  }
  return "UNKNOWN";
}

[[nodiscard]] std::string StatusName(const GuidanceStatus status) {
  switch (status) {
    case GuidanceStatus::kAvailable:
      return "AVAILABLE";
    case GuidanceStatus::kUnavailable:
      return "UNAVAILABLE";
    case GuidanceStatus::kNoRoute:
      return "NO_ROUTE";
    case GuidanceStatus::kTimeout:
      return "TIMEOUT";
  }
  return "UNKNOWN";
}

[[nodiscard]] std::string StatusName(const LocalPlanResult::Status status) {
  switch (status) {
    case LocalPlanResult::Status::kPlanFound:
      return "PLAN_FOUND";
    case LocalPlanResult::Status::kNoPath:
      return "NO_PATH";
    case LocalPlanResult::Status::kTimeout:
      return "TIMEOUT";
    case LocalPlanResult::Status::kCanceled:
      return "CANCELED";
  }
  return "UNKNOWN";
}

[[nodiscard]] ComplexTerrainBenchmarkRecord BaseRecord(
    const ComplexTerrainScenario& scenario, const std::size_t iteration,
    std::string stage, std::string platform) {
  return ComplexTerrainBenchmarkRecord{
      .scenario = scenario.name,
      .stage = std::move(stage),
      .platform = std::move(platform),
      .width = scenario.geometry.width,
      .height = scenario.geometry.height,
      .resolution_m = scenario.geometry.resolution_m,
      .iteration = iteration,
  };
}

void CopyFineMetrics(const FineTraversabilitySnapshot& fine,
                     ComplexTerrainBenchmarkRecord& record) {
  const FineSnapshotMetrics& metrics = fine.metrics();
  record.updated_cells = metrics.updated_cells;
  record.elevation_cells_examined = metrics.elevation_cells_examined;
  record.allocated_cells = metrics.allocated_cells;
  record.allocated_bytes = metrics.allocated_bytes;
}

[[nodiscard]] GridIndex IncrementalUpdateCell(
    const ComplexTerrainScenario& scenario) {
  const std::int64_t min_x =
      static_cast<std::int64_t>(scenario.geometry.width / 4U);
  const std::int64_t max_x =
      static_cast<std::int64_t>(3U * scenario.geometry.width / 4U);
  const std::int64_t min_y =
      static_cast<std::int64_t>(scenario.geometry.height / 4U);
  const std::int64_t max_y =
      static_cast<std::int64_t>(3U * scenario.geometry.height / 4U);
  for (std::int64_t y = min_y; y < max_y; ++y) {
    for (std::int64_t x = min_x; x < max_x; ++x) {
      const std::size_t offset =
          static_cast<std::size_t>(y) * scenario.geometry.width +
          static_cast<std::size_t>(x);
      if (scenario.fine_states[offset] == FineCellState::kFree &&
          std::isfinite(scenario.elevation_m[offset])) {
        return GridIndex{.x = x, .y = y};
      }
    }
  }
  throw std::runtime_error(
      "complex terrain benchmark has no interior cell for incremental update");
}

[[nodiscard]] FinalGoal FarSelectionGoal(
    const ComplexTerrainScenario& scenario) {
  const LocalTarget goal = LocalTargetForGoal(scenario);
  return FinalGoal{
      .target_x_m = goal.center.x +
                    static_cast<double>(scenario.geometry.width) *
                        scenario.geometry.resolution_m,
      .target_y_m = goal.center.y,
  };
}

[[nodiscard]] GlobalRoute StraightGuidance(
    const ComplexTerrainScenario& scenario) {
  const Pose2 start = StartPose(scenario);
  const FinalGoal goal = FarSelectionGoal(scenario);
  return GlobalRoute{.poses_map = {
                         Pose3{.position_m = {.x = start.position_m.x,
                                              .y = start.position_m.y}},
                         Pose3{.position_m = {.x = goal.target_x_m,
                                              .y = goal.target_y_m}},
                     }};
}

[[nodiscard]] SearchDeadline DeadlineAfter(
    const std::chrono::milliseconds allowance) {
  return SteadyClock::now() + allowance;
}

[[nodiscard]] PlanningSessionPorts BindPorts(
    GlobalRoutePlanner& global_planner, LocalTargetSelector& target_selector,
    RequestLocalStartPatchBuilder& patch_builder,
    WheelLocalPlanner& wheel_planner, LeggedLocalPlanner& legged_planner) {
  return PlanningSessionPorts{
      .plan_global =
          [&global_planner](const GlobalGuidanceSnapshot& snapshot,
                            const Point2 start, const Point2 goal,
                            const SearchDeadline deadline,
                            const StopToken& stop) {
            return global_planner.Plan(snapshot, start, goal, deadline, stop);
          },
      .select_target =
          [&target_selector](const FineTraversabilitySnapshot& fine,
                             const SparseGridGeometry& local_window,
                             const Point2 start, const FinalGoal& goal,
                             const std::optional<GlobalRoute>& guidance) {
            return target_selector.Select(fine, local_window, start, goal,
                                          guidance);
          },
      .build_start_patch =
          [&patch_builder](
              std::shared_ptr<const FineTraversabilitySnapshot> fine,
              const SparseGridGeometry& local_window, const Pose2& start,
              const PlatformCapability& capability,
              const TraversabilityProfile& profile) {
            return patch_builder.Build(std::move(fine), local_window, start,
                                       capability, profile);
          },
      .plan_wheel =
          [&wheel_planner](const RequestLocalPlanningView& view,
                           const Pose2& start, const LocalTarget& target,
                           const SearchDeadline deadline,
                           const StopToken& stop) {
            return wheel_planner.Plan(view, start, target, deadline, stop);
          },
      .plan_legged =
          [&legged_planner](const RequestLocalPlanningView& view,
                            const Pose2& start, const LocalTarget& target,
                            const SearchDeadline deadline,
                            const StopToken& stop) {
            return legged_planner.Plan(view, start, target, deadline, stop);
          },
  };
}

[[nodiscard]] std::string CoordinatorStatus(const CycleOutput& output) {
  if (output.path_reference.has_value()) {
    return "PLAN_FOUND";
  }
  if (output.terminal.has_value()) {
    return output.terminal->result.reason_code.empty()
               ? "TERMINAL"
               : output.terminal->result.reason_code;
  }
  return output.feedback.reason_code.empty() ? "NO_OUTPUT"
                                             : output.feedback.reason_code;
}

[[nodiscard]] ComplexTerrainBenchmarkRecord RunCoordinatorCycle(
    const ComplexTerrainScenario& scenario,
    std::shared_ptr<const FineTraversabilitySnapshot> fine,
    std::shared_ptr<const GlobalGuidanceSnapshot> guidance,
    const std::size_t iteration, const bool legged,
    const std::chrono::milliseconds stage_deadline) {
  const WheeledCapability wheel_capability = BenchmarkWheelCapability();
  const LeggedCapability legged_capability = BenchmarkLeggedCapability();
  GlobalRoutePlanner global_planner(GlobalRoutePlannerConfig{
      .global_detour_margin_m =
          static_cast<double>(scenario.geometry.height) *
          scenario.geometry.resolution_m,
      .unknown_step_risk = 5.0,
  });
  LocalTargetSelector target_selector;
  RequestLocalStartPatchBuilder patch_builder;
  WheelLocalPlanner wheel_planner(wheel_capability);
  LeggedLocalPlanner legged_planner(legged_capability);
  const PlatformCapability capability =
      legged ? PlatformCapability{legged_capability}
             : PlatformCapability{wheel_capability};
  PlanningSessionCoordinator coordinator(
      capability, BenchmarkTraversabilityProfile(),
      PlanningSessionCoordinatorConfig{
          .goal_position_tolerance_m =
              0.5 * scenario.geometry.resolution_m,
          .goal_yaw_tolerance_rad = 0.25,
          .local_window_size_m = std::min(
              64.0, static_cast<double>(std::min(scenario.geometry.width,
                                                 scenario.geometry.height)) *
                        scenario.geometry.resolution_m),
          .global_subdeadline = stage_deadline,
      },
      BindPorts(global_planner, target_selector, patch_builder, wheel_planner,
                legged_planner));
  SessionId session_id;
  session_id.bytes.front() = static_cast<std::uint8_t>(iteration + 1U);
  const LocalTarget target = LocalTargetForGoal(scenario);
  const StartSessionResult start_result = coordinator.Start(
      session_id, FinalGoal{.target_x_m = target.center.x,
                            .target_y_m = target.center.y});
  if (!start_result.accepted) {
    throw std::runtime_error("benchmark coordinator rejected valid session");
  }

  ComplexTerrainBenchmarkRecord record = BaseRecord(
      scenario, iteration,
      legged ? "coordinator_legged_cycle" : "coordinator_wheel_cycle",
      legged ? "legged" : "wheel");
  const auto started = SteadyClock::now();
  const CycleOutput output = coordinator.PlanCycle(
      StateInput{.base_link_pose = StartPose(scenario)},
      SnapshotBundle{.fine = std::move(fine),
                     .guidance = std::move(guidance)},
      CycleTrigger::kContinue, DeadlineAfter(stage_deadline));
  record.elapsed_ms = ElapsedMilliseconds(started);
  record.status = CoordinatorStatus(output);
  if (output.path_reference.has_value()) {
    record.path_points = output.path_reference->path.poses.size();
  }
  return record;
}

[[nodiscard]] std::optional<GridIndex> CellForPoint(
    const ComplexTerrainScenario& scenario, const Point2 point) {
  const double resolution = scenario.geometry.resolution_m;
  const double cell_x =
      std::floor((point.x - scenario.geometry.origin_m.x) / resolution);
  const double cell_y =
      std::floor((point.y - scenario.geometry.origin_m.y) / resolution);
  if (!std::isfinite(cell_x) || !std::isfinite(cell_y)) {
    return std::nullopt;
  }
  const GridIndex index{.x = static_cast<std::int64_t>(cell_x),
                        .y = static_cast<std::int64_t>(cell_y)};
  if (index.x < 0 || index.y < 0 ||
      index.x >= static_cast<std::int64_t>(scenario.geometry.width) ||
      index.y >= static_cast<std::int64_t>(scenario.geometry.height)) {
    return std::nullopt;
  }
  return index;
}

[[nodiscard]] std::uint64_t CellKey(const GridIndex index) noexcept {
  return (static_cast<std::uint64_t>(index.x) << 32U) |
         static_cast<std::uint64_t>(index.y);
}

[[nodiscard]] std::vector<GridIndex> RouteCells(
    const ComplexTerrainScenario& scenario, const GlobalRoute& route) {
  std::vector<GridIndex> cells;
  for (const Pose3& pose : route.poses_map) {
    const auto cell = CellForPoint(
        scenario, {.x = pose.position_m.x, .y = pose.position_m.y});
    if (cell && (cells.empty() || cells.back() != *cell)) {
      cells.push_back(*cell);
    }
  }
  return cells;
}

[[nodiscard]] std::vector<GridIndex> PathCells(
    const ComplexTerrainScenario& scenario,
    const std::vector<Pose3>& poses) {
  std::vector<GridIndex> cells;
  for (const Pose3& pose : poses) {
    const auto cell = CellForPoint(
        scenario, {.x = pose.position_m.x, .y = pose.position_m.y});
    if (cell && (cells.empty() || cells.back() != *cell)) {
      cells.push_back(*cell);
    }
  }
  return cells;
}

[[nodiscard]] std::optional<GridIndex> FindOffRouteCandidate(
    const ComplexTerrainScenario& scenario,
    const std::vector<GridIndex>& route_cells) {
  std::unordered_set<std::uint64_t> influence;
  influence.reserve(route_cells.size() * 9U);
  for (const GridIndex route_cell : route_cells) {
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        const GridIndex cell{.x = route_cell.x + dx,
                             .y = route_cell.y + dy};
        if (cell.x >= 0 && cell.y >= 0 &&
            cell.x < static_cast<std::int64_t>(scenario.geometry.width) &&
            cell.y < static_cast<std::int64_t>(scenario.geometry.height)) {
          influence.insert(CellKey(cell));
        }
      }
    }
  }
  for (std::int64_t y = 1;
       y + 1 < static_cast<std::int64_t>(scenario.geometry.height); ++y) {
    for (std::int64_t x = 1;
         x + 1 < static_cast<std::int64_t>(scenario.geometry.width); ++x) {
      const GridIndex cell{.x = x, .y = y};
      const std::size_t offset =
          static_cast<std::size_t>(y) * scenario.geometry.width +
          static_cast<std::size_t>(x);
      if (!influence.contains(CellKey(cell)) &&
          scenario.guidance_states[offset] ==
              GuidanceCellState::kCandidate) {
        return cell;
      }
    }
  }
  return std::nullopt;
}

void SetScenarioRisk(ComplexTerrainScenario& scenario, const GridIndex cell,
                     const double risk) {
  const std::size_t offset =
      static_cast<std::size_t>(cell.y) * scenario.geometry.width +
      static_cast<std::size_t>(cell.x);
  scenario.traversal_costs[offset] = risk;
  scenario.guidance_risks[offset] = risk;
}

void SetScenarioBlocked(ComplexTerrainScenario& scenario,
                        const GridIndex cell) {
  const std::size_t offset =
      static_cast<std::size_t>(cell.y) * scenario.geometry.width +
      static_cast<std::size_t>(cell.x);
  scenario.elevation_m[offset] = 1.0F;
  scenario.fine_states[offset] = FineCellState::kBlocked;
  scenario.traversal_costs[offset] = 0.0;
  scenario.guidance_states[offset] = GuidanceCellState::kProvenBlocked;
  scenario.guidance_risks[offset] = 0.0;
}

[[nodiscard]] ComplexTerrainBenchmarkRecord GlobalResultRecord(
    const ComplexTerrainScenario& scenario, const std::size_t iteration,
    std::string stage, const GlobalRouteResult& result,
    const double elapsed_ms) {
  ComplexTerrainBenchmarkRecord record =
      BaseRecord(scenario, iteration, std::move(stage), "shared");
  record.status = StatusName(result.status);
  record.cache_reused = result.reused_cache;
  record.elapsed_ms = elapsed_ms;
  record.statistics = result.statistics;
  record.path_points = result.route ? result.route->poses_map.size() : 0U;
  return record;
}

void AppendIncrementalGlobalRecords(
    const ComplexTerrainScenario& scenario,
    const std::shared_ptr<const GlobalGuidanceSnapshot>& base_snapshot,
    const Point2 start, const Point2 goal, const std::size_t iteration,
    const std::chrono::milliseconds stage_deadline,
    const GlobalRouteResult& base_result,
    std::vector<ComplexTerrainBenchmarkRecord>& records,
    GlobalRoutePlanner& exact_cache_planner) {
  if (base_result.status != GuidanceStatus::kAvailable ||
      !base_result.route || base_result.route->poses_map.size() < 3U) {
    for (const std::string stage : {
             "global_cache_exact",
             "global_replan_off_route_update",
             "global_replan_on_route_update",
             "global_replan_moved_start",
         }) {
      ComplexTerrainBenchmarkRecord record =
          BaseRecord(scenario, iteration, stage, "shared");
      record.status = "NOT_APPLICABLE";
      records.push_back(std::move(record));
    }
    return;
  }

  auto started = SteadyClock::now();
  const GlobalRouteResult exact = exact_cache_planner.Plan(
      *base_snapshot, start, goal, DeadlineAfter(stage_deadline), {});
  records.push_back(GlobalResultRecord(
      scenario, iteration, "global_cache_exact", exact,
      ElapsedMilliseconds(started)));

  const std::vector<GridIndex> route_cells =
      RouteCells(scenario, *base_result.route);
  const auto off_route = FindOffRouteCandidate(scenario, route_cells);
  if (!off_route || route_cells.size() < 3U) {
    for (const std::string stage : {
             "global_replan_off_route_update",
             "global_replan_on_route_update",
             "global_replan_moved_start",
         }) {
      ComplexTerrainBenchmarkRecord record =
          BaseRecord(scenario, iteration, stage, "shared");
      record.status = "NOT_APPLICABLE";
      records.push_back(std::move(record));
    }
    return;
  }

  GlobalRoutePlanner update_planner(GlobalRoutePlannerConfig{
      .global_detour_margin_m =
          static_cast<double>(scenario.geometry.height) *
          scenario.geometry.resolution_m,
      .unknown_step_risk = 5.0,
  });
  static_cast<void>(update_planner.Plan(
      *base_snapshot, start, goal, DeadlineAfter(stage_deadline), {}));
  ComplexTerrainScenario off_route_scenario = scenario;
  SetScenarioRisk(off_route_scenario, *off_route, 3.0);
  const auto off_route_snapshot = MakeGuidanceSnapshot(
      off_route_scenario, "benchmark-profile", 2U,
      {TileForCell(*off_route)});
  started = SteadyClock::now();
  const GlobalRouteResult off_route_result = update_planner.Plan(
      *off_route_snapshot, start, goal, DeadlineAfter(stage_deadline), {});
  records.push_back(GlobalResultRecord(
      scenario, iteration, "global_replan_off_route_update", off_route_result,
      ElapsedMilliseconds(started)));

  const GridIndex on_route = route_cells[route_cells.size() / 2U];
  ComplexTerrainScenario on_route_scenario = off_route_scenario;
  SetScenarioBlocked(on_route_scenario, on_route);
  const auto on_route_snapshot = MakeGuidanceSnapshot(
      on_route_scenario, "benchmark-profile", 3U,
      {TileForCell(on_route)});
  started = SteadyClock::now();
  const GlobalRouteResult on_route_result = update_planner.Plan(
      *on_route_snapshot, start, goal, DeadlineAfter(stage_deadline), {});
  records.push_back(GlobalResultRecord(
      scenario, iteration, "global_replan_on_route_update", on_route_result,
      ElapsedMilliseconds(started)));

  GlobalRoutePlanner moved_start_planner(GlobalRoutePlannerConfig{
      .global_detour_margin_m =
          static_cast<double>(scenario.geometry.height) *
          scenario.geometry.resolution_m,
      .unknown_step_risk = 5.0,
  });
  static_cast<void>(moved_start_planner.Plan(
      *base_snapshot, start, goal, DeadlineAfter(stage_deadline), {}));
  const GridIndex moved_cell = route_cells[1U];
  const double resolution = scenario.geometry.resolution_m;
  const Point2 moved_start{
      .x = scenario.geometry.origin_m.x +
           (static_cast<double>(moved_cell.x) + 0.5) * resolution,
      .y = scenario.geometry.origin_m.y +
           (static_cast<double>(moved_cell.y) + 0.5) * resolution,
  };
  started = SteadyClock::now();
  const GlobalRouteResult moved_result = moved_start_planner.Plan(
      *base_snapshot, moved_start, goal, DeadlineAfter(stage_deadline), {});
  records.push_back(GlobalResultRecord(
      scenario, iteration, "global_replan_moved_start", moved_result,
      ElapsedMilliseconds(started)));
}

void AppendPathRevalidationRecords(
    const ComplexTerrainScenario& scenario,
    const std::shared_ptr<const FineTraversabilitySnapshot>& base_fine,
    const std::shared_ptr<const GlobalGuidanceSnapshot>& guidance,
    const std::size_t iteration,
    const std::chrono::milliseconds stage_deadline,
    std::vector<ComplexTerrainBenchmarkRecord>& records) {
  const WheeledCapability wheel_capability = BenchmarkWheelCapability();
  const LeggedCapability legged_capability = BenchmarkLeggedCapability();
  GlobalRoutePlanner global_planner(GlobalRoutePlannerConfig{
      .global_detour_margin_m =
          static_cast<double>(scenario.geometry.height) *
          scenario.geometry.resolution_m,
      .unknown_step_risk = 5.0,
  });
  LocalTargetSelector target_selector;
  RequestLocalStartPatchBuilder patch_builder;
  WheelLocalPlanner wheel_planner(wheel_capability);
  LeggedLocalPlanner legged_planner(legged_capability);
  PlanningSessionCoordinator coordinator(
      PlatformCapability{wheel_capability}, BenchmarkTraversabilityProfile(),
      PlanningSessionCoordinatorConfig{
          .goal_position_tolerance_m =
              0.5 * scenario.geometry.resolution_m,
          .goal_yaw_tolerance_rad = 0.25,
          .local_window_size_m = std::min(
              64.0, static_cast<double>(std::min(scenario.geometry.width,
                                                 scenario.geometry.height)) *
                        scenario.geometry.resolution_m),
          .global_subdeadline = stage_deadline,
      },
      BindPorts(global_planner, target_selector, patch_builder, wheel_planner,
                legged_planner));
  SessionId session_id;
  session_id.bytes.front() = static_cast<std::uint8_t>(iteration + 101U);
  const LocalTarget target = LocalTargetForGoal(scenario);
  const StartSessionResult start_result = coordinator.Start(
      session_id, FinalGoal{.target_x_m = target.center.x,
                            .target_y_m = target.center.y});
  const CycleOutput initial =
      start_result.accepted
          ? coordinator.PlanCycle(
                StateInput{.base_link_pose = StartPose(scenario)},
                SnapshotBundle{.fine = base_fine, .guidance = guidance},
                CycleTrigger::kContinue, DeadlineAfter(stage_deadline))
          : CycleOutput{};
  if (!initial.path_reference ||
      initial.path_reference->path.poses.size() < 3U) {
    for (const std::string stage : {
             "path_revalidate_off_route_update",
             "path_revalidate_on_route_update",
         }) {
      ComplexTerrainBenchmarkRecord record =
          BaseRecord(scenario, iteration, stage, "shared");
      record.status = "NOT_APPLICABLE";
      records.push_back(std::move(record));
    }
    return;
  }

  const std::vector<GridIndex> path_cells =
      PathCells(scenario, initial.path_reference->path.poses);
  const auto off_route = FindOffRouteCandidate(scenario, path_cells);
  if (!off_route || path_cells.size() < 3U) {
    for (const std::string stage : {
             "path_revalidate_off_route_update",
             "path_revalidate_on_route_update",
         }) {
      ComplexTerrainBenchmarkRecord record =
          BaseRecord(scenario, iteration, stage, "shared");
      record.status = "NOT_APPLICABLE";
      records.push_back(std::move(record));
    }
    return;
  }

  ComplexTerrainScenario off_route_scenario = scenario;
  SetScenarioRisk(off_route_scenario, *off_route, 3.0);
  const auto off_route_fine = MakeFineSnapshot(
      off_route_scenario, "benchmark-profile", 2U,
      {TileForCell(*off_route)});
  ComplexTerrainBenchmarkRecord off_record = BaseRecord(
      scenario, iteration, "path_revalidate_off_route_update", "shared");
  auto started = SteadyClock::now();
  const auto off_result = coordinator.OnFineSnapshot(off_route_fine);
  off_record.elapsed_ms = ElapsedMilliseconds(started);
  off_record.status = off_result ? "PATH_INVALIDATED" : "PATH_UNCHANGED";
  off_record.path_points = initial.path_reference->path.poses.size();
  records.push_back(std::move(off_record));

  const GridIndex on_route = path_cells[path_cells.size() / 2U];
  ComplexTerrainScenario on_route_scenario = off_route_scenario;
  SetScenarioBlocked(on_route_scenario, on_route);
  const auto on_route_fine = MakeFineSnapshot(
      on_route_scenario, "benchmark-profile", 3U,
      {TileForCell(on_route)});
  ComplexTerrainBenchmarkRecord on_record = BaseRecord(
      scenario, iteration, "path_revalidate_on_route_update", "shared");
  started = SteadyClock::now();
  const auto on_result = coordinator.OnFineSnapshot(on_route_fine);
  on_record.elapsed_ms = ElapsedMilliseconds(started);
  on_record.status = on_result ? "PATH_INVALIDATED" : "PATH_UNCHANGED";
  on_record.path_points = initial.path_reference->path.poses.size();
  records.push_back(std::move(on_record));
}

[[nodiscard]] std::string CsvField(const std::string_view value) {
  if (value.find_first_of(",\"\n\r") == std::string_view::npos) {
    return std::string(value);
  }
  std::string escaped{"\""};
  for (const char character : value) {
    if (character == '\"') {
      escaped.push_back('\"');
    }
    escaped.push_back(character);
  }
  escaped.push_back('\"');
  return escaped;
}

[[nodiscard]] std::size_t ParsePositiveSize(const std::string_view value,
                                            const std::string_view option) {
  std::size_t parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      parsed == 0U) {
    throw std::invalid_argument(std::string(option) +
                                " requires a positive integer");
  }
  return parsed;
}

[[nodiscard]] double ParsePositiveDouble(const std::string_view value,
                                         const std::string_view option) {
  double parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      !std::isfinite(parsed) || parsed <= 0.0) {
    throw std::invalid_argument(std::string(option) +
                                " requires a positive finite number");
  }
  return parsed;
}

[[nodiscard]] ComplexTerrainPattern ParsePattern(
    const std::string_view name) {
  if (name == "dense-rock") {
    return ComplexTerrainPattern::kDenseRockField;
  }
  if (name == "maze") {
    return ComplexTerrainPattern::kAlternatingWallMaze;
  }
  if (name == "dead-ends") {
    return ComplexTerrainPattern::kNarrowPassagesAndDeadEnds;
  }
  if (name == "risk-unknown") {
    return ComplexTerrainPattern::kRiskAndUnknownBands;
  }
  if (name == "enclosed") {
    return ComplexTerrainPattern::kEnclosedGoal;
  }
  if (name == "legged-step-gap") {
    return ComplexTerrainPattern::kLeggedStepGapField;
  }
  throw std::invalid_argument("unknown complex terrain scenario: " +
                              std::string(name));
}

[[nodiscard]] std::vector<ComplexTerrainPattern> AllPatterns() {
  return {
      ComplexTerrainPattern::kDenseRockField,
      ComplexTerrainPattern::kAlternatingWallMaze,
      ComplexTerrainPattern::kNarrowPassagesAndDeadEnds,
      ComplexTerrainPattern::kRiskAndUnknownBands,
      ComplexTerrainPattern::kEnclosedGoal,
      ComplexTerrainPattern::kLeggedStepGapField,
  };
}

}  // namespace

ComplexTerrainBenchmarkOptions ParseComplexTerrainBenchmarkArguments(
    const std::vector<std::string_view>& arguments) {
  std::size_t size = 320U;
  double resolution_m = 0.2;
  std::size_t iterations = 3U;
  std::size_t deadline_ms = 3000U;
  bool include_legged = true;
  bool include_coordinator = true;
  std::vector<ComplexTerrainPattern> patterns = AllPatterns();

  for (const std::string_view argument : arguments) {
    if (argument.starts_with("--size=")) {
      size = ParsePositiveSize(argument.substr(7U), "--size");
      if (size < 32U || size > kMaximumLocalPlanningWindowAxisCells) {
        throw std::invalid_argument("--size must be between 32 and 640");
      }
    } else if (argument.starts_with("--resolution=")) {
      resolution_m =
          ParsePositiveDouble(argument.substr(13U), "--resolution");
    } else if (argument.starts_with("--iterations=")) {
      iterations = ParsePositiveSize(argument.substr(13U), "--iterations");
      if (iterations > 100U) {
        throw std::invalid_argument("--iterations must not exceed 100");
      }
    } else if (argument.starts_with("--deadline-ms=")) {
      deadline_ms = ParsePositiveSize(argument.substr(14U), "--deadline-ms");
      if (deadline_ms > 600000U) {
        throw std::invalid_argument("--deadline-ms must not exceed 600000");
      }
    } else if (argument.starts_with("--scenario=")) {
      const std::string_view scenario = argument.substr(11U);
      patterns = scenario == "all"
                     ? AllPatterns()
                     : std::vector<ComplexTerrainPattern>{
                           ParsePattern(scenario)};
    } else if (argument == "--no-legged") {
      include_legged = false;
    } else if (argument == "--no-coordinator") {
      include_coordinator = false;
    } else {
      throw std::invalid_argument("unknown benchmark option: " +
                                  std::string(argument));
    }
  }

  std::vector<ComplexTerrainSpec> scenarios;
  scenarios.reserve(patterns.size());
  std::uint32_t seed = 0xC0FFEEU;
  for (const ComplexTerrainPattern pattern : patterns) {
    scenarios.push_back(ComplexTerrainSpec{
        .pattern = pattern,
        .width = size,
        .height = size,
        .resolution_m = resolution_m,
        .seed = seed++,
    });
  }
  return ComplexTerrainBenchmarkOptions{
      .scenarios = std::move(scenarios),
      .iterations = iterations,
      .stage_deadline = std::chrono::milliseconds(deadline_ms),
      .include_legged = include_legged,
      .include_coordinator = include_coordinator,
  };
}

std::vector<ComplexTerrainBenchmarkRecord> RunComplexTerrainBenchmarks(
    const ComplexTerrainBenchmarkOptions& options) {
  if (options.scenarios.empty() || options.iterations == 0U ||
      options.stage_deadline <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument(
        "complex terrain benchmark requires scenarios, iterations and deadline");
  }

  std::vector<ComplexTerrainBenchmarkRecord> records;
  for (const ComplexTerrainSpec& spec : options.scenarios) {
    const ComplexTerrainScenario scenario = MakeComplexTerrainScenario(spec);
    const auto direct_fine = MakeFineSnapshot(scenario, "benchmark-profile");
    const auto direct_guidance =
        MakeGuidanceSnapshot(scenario, "benchmark-profile");
    const Pose2 start = StartPose(scenario);
    const LocalTarget target = LocalTargetForGoal(scenario);
    const RequestLocalPlanningView view(direct_fine, start, 0.0, {});
    const WheeledCapability wheel_capability = BenchmarkWheelCapability();
    const LeggedCapability legged_capability = BenchmarkLeggedCapability();
    const TraversabilityProfile profile = BenchmarkTraversabilityProfile();
    WheelLocalPlanner wheel_planner(wheel_capability);
    LeggedLocalPlanner legged_planner(legged_capability);
    const GlobalRoute target_guidance = StraightGuidance(scenario);

    for (std::size_t iteration = 0U; iteration < options.iterations;
         ++iteration) {
      PersistentElevationMap elevation_map;
      ComplexTerrainBenchmarkRecord elevation_record =
          BaseRecord(scenario, iteration, "elevation_apply", "shared");
      auto started = SteadyClock::now();
      const ElevationUpdateResult elevation_update =
          elevation_map.Apply(ElevationEvidence{
              .geometry = scenario.geometry,
              .elevation_m = scenario.elevation_m,
              .map_from_source = RigidTransform{.parent_frame = "map",
                                                .child_frame = "map"},
          });
      elevation_record.elapsed_ms = ElapsedMilliseconds(started);
      elevation_record.status = StatusName(elevation_update.status);
      elevation_record.updated_cells = elevation_update.updated_cells;
      records.push_back(std::move(elevation_record));
      const auto raw = elevation_map.Snapshot();

      FineTraversabilityBuilder fine_builder;
      ComplexTerrainBenchmarkRecord fine_record =
          BaseRecord(scenario, iteration, "fine_derive_wheel", "wheel");
      started = SteadyClock::now();
      const auto derived_fine = fine_builder.Derive(
          raw, PlatformCapability{wheel_capability}, profile);
      fine_record.elapsed_ms = ElapsedMilliseconds(started);
      fine_record.status = derived_fine ? "READY" : "UNAVAILABLE";
      if (derived_fine) {
        CopyFineMetrics(*derived_fine, fine_record);
      }
      records.push_back(std::move(fine_record));

      const GridIndex update_cell = IncrementalUpdateCell(scenario);
      const double resolution_m = scenario.geometry.resolution_m;
      const std::size_t update_offset =
          static_cast<std::size_t>(update_cell.y) * scenario.geometry.width +
          static_cast<std::size_t>(update_cell.x);
      const std::array<float, 1U> update_value{
          scenario.elevation_m[update_offset] + 0.05F};
      const GridGeometry update_geometry{
          .frame_id = scenario.geometry.frame_id,
          .width = 1U,
          .height = 1U,
          .resolution_m = resolution_m,
          .origin_m = {
              .x = scenario.geometry.origin_m.x +
                   static_cast<double>(update_cell.x) * resolution_m,
              .y = scenario.geometry.origin_m.y +
                   static_cast<double>(update_cell.y) * resolution_m,
              .z = scenario.geometry.origin_m.z,
          },
      };
      ComplexTerrainBenchmarkRecord incremental_apply_record = BaseRecord(
          scenario, iteration, "elevation_apply_incremental_single_cell",
          "shared");
      started = SteadyClock::now();
      const ElevationUpdateResult incremental_update =
          elevation_map.Apply(ElevationEvidence{
              .geometry = update_geometry,
              .elevation_m = update_value,
              .map_from_source = RigidTransform{.parent_frame = "map",
                                                .child_frame = "map"},
          });
      incremental_apply_record.elapsed_ms = ElapsedMilliseconds(started);
      incremental_apply_record.status = StatusName(incremental_update.status);
      incremental_apply_record.updated_cells = incremental_update.updated_cells;
      records.push_back(std::move(incremental_apply_record));

      ComplexTerrainBenchmarkRecord incremental_fine_record = BaseRecord(
          scenario, iteration, "fine_derive_incremental_single_cell", "wheel");
      started = SteadyClock::now();
      const auto incremental_fine = fine_builder.Derive(
          elevation_map.Snapshot(), PlatformCapability{wheel_capability},
          profile, derived_fine);
      incremental_fine_record.elapsed_ms = ElapsedMilliseconds(started);
      incremental_fine_record.status =
          incremental_fine ? "READY" : "UNAVAILABLE";
      if (incremental_fine) {
        CopyFineMetrics(*incremental_fine, incremental_fine_record);
      }
      records.push_back(std::move(incremental_fine_record));

      const std::size_t patch_size = std::clamp(
          std::min(scenario.geometry.width, scenario.geometry.height) / 20U,
          std::size_t{2U}, std::size_t{32U});
      const std::size_t patch_x =
          (scenario.geometry.width - patch_size) / 2U;
      const std::size_t patch_y =
          (scenario.geometry.height - patch_size) / 2U;
      std::vector<float> patch_values(patch_size * patch_size);
      for (std::size_t y = 0U; y < patch_size; ++y) {
        for (std::size_t x = 0U; x < patch_size; ++x) {
          const float prior =
              scenario.elevation_m[(patch_y + y) * scenario.geometry.width +
                                   patch_x + x];
          patch_values[y * patch_size + x] =
              std::isfinite(prior) ? prior + 0.1F : 0.0F;
        }
      }
      const GridGeometry patch_geometry{
          .frame_id = scenario.geometry.frame_id,
          .width = patch_size,
          .height = patch_size,
          .resolution_m = resolution_m,
          .origin_m = {
              .x = scenario.geometry.origin_m.x +
                   static_cast<double>(patch_x) * resolution_m,
              .y = scenario.geometry.origin_m.y +
                   static_cast<double>(patch_y) * resolution_m,
              .z = scenario.geometry.origin_m.z,
          },
      };
      ComplexTerrainBenchmarkRecord patch_apply_record = BaseRecord(
          scenario, iteration, "elevation_apply_incremental_patch", "shared");
      started = SteadyClock::now();
      const ElevationUpdateResult patch_update =
          elevation_map.Apply(ElevationEvidence{
              .geometry = patch_geometry,
              .elevation_m = patch_values,
              .map_from_source = RigidTransform{.parent_frame = "map",
                                                .child_frame = "map"},
          });
      patch_apply_record.elapsed_ms = ElapsedMilliseconds(started);
      patch_apply_record.status = StatusName(patch_update.status);
      patch_apply_record.updated_cells = patch_update.updated_cells;
      records.push_back(std::move(patch_apply_record));

      ComplexTerrainBenchmarkRecord patch_fine_record = BaseRecord(
          scenario, iteration, "fine_derive_incremental_patch", "wheel");
      started = SteadyClock::now();
      const auto patch_fine = fine_builder.Derive(
          elevation_map.Snapshot(), PlatformCapability{wheel_capability},
          profile, incremental_fine);
      patch_fine_record.elapsed_ms = ElapsedMilliseconds(started);
      patch_fine_record.status = patch_fine ? "READY" : "UNAVAILABLE";
      if (patch_fine) {
        CopyFineMetrics(*patch_fine, patch_fine_record);
      }
      records.push_back(std::move(patch_fine_record));

      GlobalGuidanceBuilder guidance_builder(
          std::max(1.0, scenario.geometry.resolution_m));
      ComplexTerrainBenchmarkRecord guidance_record = BaseRecord(
          scenario, iteration, "guidance_derive_wheel", "wheel");
      started = SteadyClock::now();
      const auto derived_guidance =
          guidance_builder.Derive(derived_fine, std::nullopt);
      guidance_record.elapsed_ms = ElapsedMilliseconds(started);
      guidance_record.status = derived_guidance ? "READY" : "UNAVAILABLE";
      records.push_back(std::move(guidance_record));

      GlobalRoutePlanner global_planner(GlobalRoutePlannerConfig{
          .global_detour_margin_m =
              static_cast<double>(scenario.geometry.height) *
              scenario.geometry.resolution_m,
          .unknown_step_risk = 5.0,
      });
      ComplexTerrainBenchmarkRecord global_record =
          BaseRecord(scenario, iteration, "global_plan", "shared");
      started = SteadyClock::now();
      const GlobalRouteResult global_result = global_planner.Plan(
          *direct_guidance, start.position_m, target.center,
          DeadlineAfter(options.stage_deadline), {});
      global_record.elapsed_ms = ElapsedMilliseconds(started);
      global_record.status = StatusName(global_result.status);
      global_record.cache_reused = global_result.reused_cache;
      global_record.statistics = global_result.statistics;
      global_record.path_points =
          global_result.route ? global_result.route->poses_map.size() : 0U;
      records.push_back(std::move(global_record));
      AppendIncrementalGlobalRecords(
          scenario, direct_guidance, start.position_m, target.center,
          iteration, options.stage_deadline, global_result, records,
          global_planner);

      local_target_selector_internal::SelectionMetrics selection_metrics;
      ComplexTerrainBenchmarkRecord target_record = BaseRecord(
          scenario, iteration, "target_select_guided", "shared");
      started = SteadyClock::now();
      const auto selected = local_target_selector_internal::Select(
          *direct_fine, direct_fine->geometry(), start.position_m,
          FarSelectionGoal(scenario),
          target_guidance, &selection_metrics);
      target_record.elapsed_ms = ElapsedMilliseconds(started);
      target_record.status = selected ? "TARGET_SELECTED" : "NO_TARGET";
      target_record.candidate_tile_lookups =
          selection_metrics.candidate_tile_lookups;
      target_record.candidate_cells_examined =
          selection_metrics.candidate_cells_examined;
      target_record.guidance_cells_examined =
          selection_metrics.guidance_cells_examined;
      records.push_back(std::move(target_record));

      ComplexTerrainBenchmarkRecord wheel_record =
          BaseRecord(scenario, iteration, "wheel_plan", "wheel");
      started = SteadyClock::now();
      const LocalPlanResult wheel_result = wheel_planner.Plan(
          view, start, target, DeadlineAfter(options.stage_deadline), {});
      wheel_record.elapsed_ms = ElapsedMilliseconds(started);
      wheel_record.status = StatusName(wheel_result.status);
      wheel_record.postprocess_ms =
          Milliseconds(wheel_result.postprocess_elapsed);
      wheel_record.statistics = wheel_result.statistics;
      wheel_record.path_points = wheel_result.path.size();
      wheel_record.raw_path_points = wheel_result.raw_path.size();
      records.push_back(std::move(wheel_record));

      if (options.include_legged) {
        ComplexTerrainBenchmarkRecord legged_record =
            BaseRecord(scenario, iteration, "legged_plan", "legged");
        started = SteadyClock::now();
        const LocalPlanResult legged_result = legged_planner.Plan(
            view, start, target, DeadlineAfter(options.stage_deadline), {});
        legged_record.elapsed_ms = ElapsedMilliseconds(started);
        legged_record.status = StatusName(legged_result.status);
        legged_record.postprocess_ms =
            Milliseconds(legged_result.postprocess_elapsed);
        legged_record.statistics = legged_result.statistics;
        legged_record.path_points = legged_result.path.size();
        legged_record.raw_path_points = legged_result.raw_path.size();
        records.push_back(std::move(legged_record));
      }

      if (options.include_coordinator) {
        records.push_back(RunCoordinatorCycle(
            scenario, direct_fine, direct_guidance, iteration, false,
            options.stage_deadline));
        if (options.include_legged) {
          records.push_back(RunCoordinatorCycle(
              scenario, direct_fine, direct_guidance, iteration, true,
              options.stage_deadline));
        }
        AppendPathRevalidationRecords(
            scenario, direct_fine, direct_guidance, iteration,
            options.stage_deadline, records);
      }
    }
  }
  return records;
}

void WriteComplexTerrainBenchmarkCsv(
    const std::vector<ComplexTerrainBenchmarkRecord>& records,
    std::ostream& output) {
  output << "scenario,stage,platform,width,height,resolution_m,iteration,"
            "status,cache_reused,elapsed_ms,postprocess_ms,expanded_states,"
            "generated_states,evaluated_transitions,open_peak,path_points,"
            "raw_path_points,updated_cells,elevation_cells_examined,"
            "allocated_cells,allocated_bytes,candidate_tile_lookups,"
            "candidate_cells_examined,guidance_cells_examined\n";
  output << std::setprecision(9);
  for (const ComplexTerrainBenchmarkRecord& record : records) {
    output << CsvField(record.scenario) << ',' << CsvField(record.stage) << ','
           << CsvField(record.platform) << ',' << record.width << ','
           << record.height << ',' << record.resolution_m << ','
           << record.iteration << ',' << CsvField(record.status) << ','
           << (record.cache_reused ? 1 : 0) << ',' << record.elapsed_ms << ','
           << record.postprocess_ms << ','
           << record.statistics.expanded_states << ','
           << record.statistics.generated_states << ','
           << record.statistics.evaluated_transitions << ','
           << record.statistics.open_peak << ',' << record.path_points << ','
           << record.raw_path_points << ',' << record.updated_cells << ','
           << record.elevation_cells_examined << ',' << record.allocated_cells
           << ',' << record.allocated_bytes << ','
           << record.candidate_tile_lookups << ','
           << record.candidate_cells_examined << ','
           << record.guidance_cells_examined << '\n';
  }
}

}  // namespace lunar::incremental_navigation::test_support
