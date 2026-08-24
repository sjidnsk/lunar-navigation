#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <stop_token>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/surface_portal_set.hpp"
#include "hierarchical/reference_composer.hpp"
#include "legged/anytime_legged_planner.hpp"
#include "lunar_pure_planner_core/planner.hpp"
#include "shared/local_terrain_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning {
namespace {

using namespace std::chrono_literals;

class ManualClock final {
 public:
  [[nodiscard]] NowFn now() { return [this] { return current_; }; }

  void Advance(const std::chrono::milliseconds elapsed) {
    current_ += elapsed;
  }

 private:
  SteadyClock::time_point current_{};
};

[[nodiscard]] GridMap LocalMap() {
  constexpr std::size_t kWidth = 6U;
  constexpr std::size_t kHeight = 4U;
  return GridMap{
      .frame_id = "odom",
      .width = kWidth,
      .height = kHeight,
      .resolution_m = 1.0,
      .origin_m = {},
      .layers = {
          {"occupancy", GridLayer{.values = std::vector<float>(
                                       kWidth * kHeight, 0.0F)}},
          {"elevation", GridLayer{.values = std::vector<float>(
                                       kWidth * kHeight, 4.25F)}},
      },
  };
}

[[nodiscard]] GridMap GlobalMap() {
  constexpr std::size_t kWidth = 10U;
  constexpr std::size_t kHeight = 4U;
  return GridMap{
      .frame_id = "map",
      .width = kWidth,
      .height = kHeight,
      .resolution_m = 1.0,
      .origin_m = {},
      .layers = {
          {"occupancy", GridLayer{.values = std::vector<std::int8_t>(
                                       kWidth * kHeight, 0)}},
      },
  };
}

[[nodiscard]] GridMap WideLocalMap(const std::size_t width) {
  constexpr std::size_t kHeight = 4U;
  return GridMap{
      .frame_id = "odom",
      .width = width,
      .height = kHeight,
      .resolution_m = 1.0,
      .origin_m = {},
      .layers = {
          {"occupancy", GridLayer{.values = std::vector<float>(
                                       width * kHeight, 0.0F)}},
          {"elevation", GridLayer{.values = std::vector<float>(
                                       width * kHeight, 4.25F)}},
      },
  };
}

[[nodiscard]] GoalRegion PointTarget(const double x, const double y,
                                     const double z = 0.0) {
  return GoalRegion{
      .goal_id = "goal",
      .target = PointGoal{
          .position_m = {.x = x, .y = y, .z = z},
          .tolerance_m = 0.05,
      },
      .yaw_rad = 0.25,
      .yaw_tolerance_rad = 0.1,
  };
}

[[nodiscard]] PlanningRequest Request(const EnvironmentMode mode,
                                      ManualClock& clock) {
  return PlanningRequest{
      .request_id = "request-10",
      .environment_mode = mode,
      .current_state = WheeledState{
          .pose = {.position_m = {.x = 1.0, .y = 1.0, .z = 4.25}},
      },
      .goal_map = PointTarget(mode == EnvironmentMode::kLunarSurface ? 8.0
                                                                     : 4.0,
                              1.0),
      .world = MinimalWorldSnapshot{
          .global_map = mode == EnvironmentMode::kLunarSurface
                            ? std::optional<GridMap>{GlobalMap()}
                            : std::nullopt,
          .local_map = LocalMap(),
          .map_from_odom = RigidTransform{
              .parent_frame = "map",
              .child_frame = "odom",
          },
      },
      .capability = WheeledCapability{},
      .control = SearchControl{.now = clock.now()},
  };
}

[[nodiscard]] GlobalStageResult Route() {
  return GlobalStageResult{
      .route = GlobalRoute{
          .poses_map = {
              {.position_m = {.x = 1.0, .y = 1.0, .z = 4.25}},
              {.position_m = {.x = 4.0, .y = 1.0, .z = 4.25}},
              {.position_m = {.x = 8.0, .y = 1.0, .z = 4.25}},
          },
          .expanded_states = 17U,
      },
      .reason_code = {},
  };
}

[[nodiscard]] LocalStageResult TrajectoryTo(const GoalRegion& goal) {
  const auto* point = std::get_if<PointGoal>(&goal.target);
  if (point == nullptr) {
    return {.status = LocalPlanStatus::kInvalidInput,
            .reason_code = "INVALID_INPUT"};
  }
  return LocalStageResult{
      .status = LocalPlanStatus::kSolved,
      .data = TrajectoryReference{
          .semantics = TrajectorySemantics::kWheeledBase,
          .points = {{.pose = {.position_m = {.x = point->position_m.x,
                                              .y = point->position_m.y,
                                              .z = 4.25}}}},
      },
      .reason_code = {},
  };
}

[[nodiscard]] const GoalRegion& FirstGoal(const LocalGoalSet& goals) {
  return goals.goals_odom.front();
}

struct CountingBackends final {
  ManualClock clock;
  std::size_t global_calls{};
  std::size_t local_calls{};
  LocalGoalSet local_goals;
  SearchControl global_control;
  SearchControl local_control;
  Planner planner;

  CountingBackends()
      : planner(PlannerBackends{
            .global = [this](const PlanningRequest&, SearchControl control) {
              ++global_calls;
              global_control = std::move(control);
              clock.Advance(20ms);
              return Route();
            },
            .local = [this](const PlanningRequest&, const LocalGoalSet& goals,
                            SearchControl control) {
              ++local_calls;
              local_goals = goals;
              local_control = std::move(control);
              clock.Advance(30ms);
              return TrajectoryTo(FirstGoal(goals));
            },
        }) {}
};

TEST(DualModePlanner, ConvertsPortalsToSafeStrictlyAdvancingGoalRegions) {
  const GoalRegion advancing{
      .goal_id = "advancing",
      .target = PointGoal{
          .position_m = {.x = 8.1, .y = 2.1, .z = 0.0},
          .tolerance_m = 0.0,
      },
  };
  const GoalRegion current_cell{
      .goal_id = "current-cell",
      .target = PointGoal{
          .position_m = {.x = 1.1, .y = 2.1, .z = 0.0},
          .tolerance_m = 0.0,
      },
  };
  const hierarchical::SurfacePortalSetResult portals{
      .candidates = {
          hierarchical::SurfacePortalCandidate{
              .goal_odom = advancing,
              .route_progress_m = 18.0,
              .local_cell = {.x = 40, .y = 10},
          },
          hierarchical::SurfacePortalCandidate{
              .goal_odom = current_cell,
              .route_progress_m = 10.0,
              .local_cell = {.x = 5, .y = 10},
          },
      },
  };
  const hierarchical::SurfaceRollingDecision decision{
      .kind = hierarchical::SurfaceRollingDecision::Kind::kNextPortalSet,
      .projected_route_progress_m = 10.0,
      .desired_horizon_progress_m = 18.0,
      .targets_final_goal = false,
  };

  const hierarchical::LocalGoalSetResult converted =
      hierarchical::ConvertSurfacePortalsToLocalGoals(
          portals, decision,
          Pose3{.position_m = {.x = 1.1, .y = 2.1, .z = 0.0}},
          GridMap{.width = 100U, .height = 100U, .resolution_m = 1.0},
          GridMap{.width = 100U, .height = 100U, .resolution_m = 0.2});

  ASSERT_TRUE(converted.ok()) << converted.reason_code;
  ASSERT_EQ(converted.goals->goals_odom.size(), 1U);
  EXPECT_FALSE(converted.goals->exact_final_goal);
  const GoalRegion& retained = converted.goals->goals_odom.front();
  EXPECT_EQ(retained.goal_id, "advancing");
  EXPECT_FALSE(retained.yaw_rad.has_value());
  EXPECT_LT(std::get<PointGoal>(retained.target).tolerance_m, 0.1);
  EXPECT_GT(std::get<PointGoal>(retained.target).tolerance_m, 0.099999);
}

TEST(DualModePlanner, LeavesExactFinalPortalUntouched) {
  const GoalRegion mission_goal{
      .goal_id = "mission-final",
      .target = PointGoal{
          .position_m = {.x = 8.0, .y = 2.0, .z = 0.0},
          .tolerance_m = 0.23,
      },
      .yaw_rad = 0.4,
      .yaw_tolerance_rad = 0.05,
  };
  const hierarchical::SurfacePortalSetResult portals{
      .candidates = {hierarchical::SurfacePortalCandidate{
          .goal_odom = mission_goal,
          .route_progress_m = 18.0,
      }},
  };
  const hierarchical::SurfaceRollingDecision decision{
      .kind = hierarchical::SurfaceRollingDecision::Kind::kNextPortalSet,
      .projected_route_progress_m = 10.0,
      .desired_horizon_progress_m = 18.0,
      .targets_final_goal = true,
  };

  const hierarchical::LocalGoalSetResult converted =
      hierarchical::ConvertSurfacePortalsToLocalGoals(
          portals, decision,
          Pose3{.position_m = {.x = 1.0, .y = 2.0, .z = 0.0}},
          GridMap{.width = 100U, .height = 100U, .resolution_m = 1.0},
          GridMap{.width = 100U, .height = 100U, .resolution_m = 0.2});

  ASSERT_TRUE(converted.ok()) << converted.reason_code;
  ASSERT_TRUE(converted.goals->exact_final_goal);
  ASSERT_EQ(converted.goals->goals_odom.size(), 1U);
  const GoalRegion& retained = converted.goals->goals_odom.front();
  EXPECT_EQ(retained.goal_id, mission_goal.goal_id);
  EXPECT_EQ(retained.yaw_rad, mission_goal.yaw_rad);
  EXPECT_DOUBLE_EQ(std::get<PointGoal>(retained.target).tolerance_m, 0.23);
}

TEST(DualModePlanner, LavaTubeNeverTouchesGlobalMap) {
  CountingBackends backends;
  PlanningRequest input = Request(EnvironmentMode::kLavaTube, backends.clock);

  const PlanningResult output = backends.planner.Plan(input);

  ASSERT_TRUE(output.reference.has_value()) << output.reason_code;
  EXPECT_EQ(backends.global_calls, 0U);
  EXPECT_EQ(backends.local_calls, 1U);
  EXPECT_EQ(output.timing.global_elapsed, 0ns);
  EXPECT_EQ(output.timing.global_call_count, 0U);
  EXPECT_EQ(output.timing.local_call_count, 1U);
  EXPECT_EQ(output.timing.local_elapsed, 30ms);
  EXPECT_EQ(output.timing.total_elapsed, 30ms);
  EXPECT_EQ(backends.local_control.deadline,
            SteadyClock::time_point{3s});
  EXPECT_EQ(output.reference->preview.poses_map.back().position_m.x, 4.0);
}

TEST(DualModePlanner, LunarSurfaceUsesOneGlobalAndOneLocalStage) {
  CountingBackends backends;
  PlanningRequest input =
      Request(EnvironmentMode::kLunarSurface, backends.clock);

  const PlanningResult output = backends.planner.Plan(input);

  ASSERT_TRUE(output.reference.has_value()) << output.reason_code;
  EXPECT_EQ(backends.global_calls, 1U);
  EXPECT_EQ(backends.local_calls, 1U);
  EXPECT_EQ(backends.global_control.deadline,
            SteadyClock::time_point{3s});
  EXPECT_EQ(backends.local_control.deadline,
            SteadyClock::time_point{3s});
  ASSERT_FALSE(backends.local_goals.goals_odom.empty());
  const auto* subgoal = std::get_if<PointGoal>(
      &backends.local_goals.goals_odom.front().target);
  ASSERT_NE(subgoal, nullptr);
  EXPECT_GT(subgoal->position_m.x, 4.0);
  EXPECT_LT(subgoal->position_m.x, 6.0);
  EXPECT_DOUBLE_EQ(output.reference->preview.poses_map.back().position_m.x,
                   8.0);
  const auto* trajectory =
      std::get_if<TrajectoryReference>(&output.reference->data);
  ASSERT_NE(trajectory, nullptr);
  EXPECT_DOUBLE_EQ(trajectory->points.back().pose.position_m.x,
                   subgoal->position_m.x);
  EXPECT_EQ(output.timing.global_call_count, 1U);
  EXPECT_EQ(output.timing.local_call_count, 1U);
  EXPECT_EQ(output.timing.global_elapsed, 20ms);
  EXPECT_EQ(output.timing.local_elapsed, 30ms);
  EXPECT_EQ(output.timing.total_elapsed, 50ms);
}

TEST(DualModePlanner, GlobalBackendReceivesTheUnreducedHardDeadline) {
  ManualClock clock;
  PlanningRequest input = Request(EnvironmentMode::kLunarSurface, clock);
  const auto deadline = SteadyClock::time_point{3s};
  const auto result = hierarchical::PlanSurfaceGlobal(
      input,
      SearchControl{
          .deadline = deadline,
          .now = [deadline] { return deadline - 5ms; },
      },
      0.0);

  EXPECT_NE(result.reason_code, "TIMEOUT");
}

TEST(DualModePlanner,
     WorkerPreprocessingConsumesTotalBudgetButNotTheGlobalStageBudget) {
  ManualClock clock;
  clock.Advance(300ms);  // Worker t0 was 0; core is entered after adaptation.
  SearchControl global_control;
  SearchControl local_control;
  Planner planner(PlannerBackends{
      .global = [&clock, &global_control](const PlanningRequest&,
                                          SearchControl control) {
        global_control = std::move(control);
        clock.Advance(20ms);
        return Route();
      },
      .local = [&clock, &local_control](const PlanningRequest&,
                                        const LocalGoalSet& goals,
                                        SearchControl control) {
        local_control = std::move(control);
        clock.Advance(30ms);
        return TrajectoryTo(FirstGoal(goals));
      },
  });
  PlanningRequest input = Request(EnvironmentMode::kLunarSurface, clock);
  input.request_started_at = SteadyClock::time_point{};

  const PlanningResult output = planner.Plan(input);

  ASSERT_EQ(output.status, PlanningStatus::kSuccess) << output.reason_code;
  EXPECT_EQ(global_control.deadline, SteadyClock::time_point{3s});
  EXPECT_EQ(local_control.deadline, SteadyClock::time_point{3s});
  EXPECT_EQ(output.timing.global_elapsed, 20ms);
  EXPECT_EQ(output.timing.local_elapsed, 30ms);
  EXPECT_EQ(output.timing.total_elapsed, 350ms);
}

TEST(DualModePlanner, UsesExactThreeSecondHardBoundaryForEveryElapsedClass) {
  struct Case final {
    std::chrono::milliseconds elapsed;
    PlanningStatus status;
    bool has_reference;
  };
  for (const Case& test_case : {
           Case{999ms, PlanningStatus::kSuccess, true},
           Case{1000ms, PlanningStatus::kSuccess, true},
           Case{1999ms, PlanningStatus::kSuccess, true},
           Case{2000ms, PlanningStatus::kSuccess, true},
           Case{2999ms, PlanningStatus::kSuccess, true},
           Case{3000ms, PlanningStatus::kTimedOut, false},
       }) {
    ManualClock clock;
    Planner planner(PlannerBackends{
        .global = {},
        .local = [&clock, test_case](const PlanningRequest&,
                                     const LocalGoalSet& goals,
                                     SearchControl control) {
          EXPECT_EQ(control.deadline, SteadyClock::time_point{3s});
          clock.Advance(test_case.elapsed);
          return TrajectoryTo(FirstGoal(goals));
        },
    });

    const PlanningResult output =
        planner.Plan(Request(EnvironmentMode::kLavaTube, clock));

    EXPECT_EQ(output.status, test_case.status) << test_case.elapsed.count();
    EXPECT_EQ(output.reference.has_value(), test_case.has_reference)
        << test_case.elapsed.count();
  }
}

TEST(DualModePlanner,
     SuppliedRequestStartDoesNotEagerlyConsumeAClockTickAndReportsPhases) {
  ManualClock fixture_clock;
  PlanningRequest input =
      Request(EnvironmentMode::kLavaTube, fixture_clock);
  input.request_started_at = SteadyClock::time_point{};
  std::size_t clock_reads{};
  input.control.now = [&clock_reads] {
    ++clock_reads;
    return SteadyClock::time_point{};
  };
  std::vector<PlannerPhase> phases;
  input.progress = [&phases](const PlannerProgress& progress) {
    phases.push_back(progress.phase);
  };
  Planner planner(PlannerBackends{
      .global = {},
      .local = [&clock_reads](const PlanningRequest&, const LocalGoalSet& goals,
                              SearchControl) {
        EXPECT_EQ(clock_reads, 3U);
        return TrajectoryTo(FirstGoal(goals));
      },
  });

  const PlanningResult output = planner.Plan(input);

  ASSERT_EQ(output.status, PlanningStatus::kSuccess) << output.reason_code;
  EXPECT_EQ(phases,
            (std::vector<PlannerPhase>{
                PlannerPhase::kSnapshotProjection, PlannerPhase::kLocalGoal,
                PlannerPhase::kLocalSearch, PlannerPhase::kCertification,
                PlannerPhase::kOutput}));
}

TEST(DualModePlanner, LavaTubeOutsideGoalDoesNotFallBackToProvidedGlobalMap) {
  CountingBackends backends;
  PlanningRequest input = Request(EnvironmentMode::kLavaTube, backends.clock);
  input.goal_map = PointTarget(20.0, 1.0);
  input.world.global_map = GlobalMap();

  const PlanningResult output = backends.planner.Plan(input);

  EXPECT_EQ(output.status, PlanningStatus::kGoalOutsideLocalMap);
  EXPECT_EQ(output.reason_code, "GOAL_OUTSIDE_LOCAL_MAP");
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_EQ(backends.global_calls, 0U);
  EXPECT_EQ(backends.local_calls, 0U);
  EXPECT_EQ(output.timing.global_elapsed, 0ns);
}

TEST(DualModePlanner, RejectsGlobalIncumbentAfterTheStageDeadline) {
  ManualClock clock;
  std::size_t local_calls{};
  Planner planner(PlannerBackends{
      .global = [&clock](const PlanningRequest&, SearchControl control) {
        EXPECT_EQ(control.deadline, SteadyClock::time_point{3s});
        clock.Advance(3000ms);
        return Route();
      },
      .local = [&local_calls](const PlanningRequest&, const LocalGoalSet& goals,
                              SearchControl) {
        ++local_calls;
        return TrajectoryTo(FirstGoal(goals));
      },
  });

  const PlanningResult output =
      planner.Plan(Request(EnvironmentMode::kLunarSurface, clock));

  EXPECT_EQ(output.status, PlanningStatus::kTimedOut);
  EXPECT_EQ(output.reason_code, "TIMEOUT");
  EXPECT_EQ(local_calls, 0U);
  EXPECT_EQ(output.timing.global_call_count, 1U);
  EXPECT_EQ(output.timing.local_call_count, 0U);
  EXPECT_EQ(output.timing.global_elapsed, 3000ms);
  EXPECT_EQ(output.timing.total_elapsed, 3000ms);
}

TEST(DualModePlanner, CaveLocalStageMayUseTheFullHardWindow) {
  ManualClock clock;
  Planner planner(PlannerBackends{
      .global = {},
      .local = [&clock](const PlanningRequest&, const LocalGoalSet&,
                        SearchControl control) {
        EXPECT_EQ(control.deadline, SteadyClock::time_point{3s});
        clock.Advance(3000ms);
        return LocalStageResult{.status = LocalPlanStatus::kTimedOut,
                                .reason_code = "TIMEOUT"};
      },
  });

  const PlanningResult output =
      planner.Plan(Request(EnvironmentMode::kLavaTube, clock));

  EXPECT_EQ(output.status, PlanningStatus::kTimedOut);
  EXPECT_EQ(output.reason_code, "TIMEOUT");
  EXPECT_EQ(output.timing.local_call_count, 1U);
  EXPECT_EQ(output.timing.local_elapsed, 3000ms);
  EXPECT_EQ(output.timing.total_elapsed, 3000ms);
}

TEST(DualModePlanner, ReturnsAValidIncumbentEvenWhenTheDeadlineIsReached) {
  ManualClock clock;
  Planner planner(PlannerBackends{
      .global = {},
      .local = [&clock](const PlanningRequest&, const LocalGoalSet& goals,
                        SearchControl) {
        clock.Advance(950ms);
        return TrajectoryTo(FirstGoal(goals));
      },
  });

  const PlanningResult output =
      planner.Plan(Request(EnvironmentMode::kLavaTube, clock));

  EXPECT_EQ(output.status, PlanningStatus::kSuccess) << output.reason_code;
  EXPECT_TRUE(output.reference.has_value());
  EXPECT_EQ(output.timing.local_elapsed, 950ms);
  EXPECT_EQ(output.timing.total_elapsed, 950ms);
}

TEST(DualModePlanner, RouteSamplingTimeoutIsNotReportedAsNoPath) {
  ManualClock clock;
  PlanningRequest input = Request(EnvironmentMode::kLunarSurface, clock);
  input.world.local_map = WideLocalMap(4096U);
  input.goal_map = PointTarget(5000.0, 1.0);
  GlobalRoute route;
  route.poses_map.reserve(4096U);
  for (std::size_t index = 0U; index < 4096U; ++index) {
    route.poses_map.push_back(
        {.position_m = {.x = static_cast<double>(index), .y = 1.0}});
  }
  SteadyClock::time_point now{};
  SearchControl control{
      .deadline = SteadyClock::time_point{5ms},
      .now = [&now] {
        now += 1ms;
        return now;
      },
  };

  const auto selected =
      hierarchical::SelectSurfaceLocalGoals(input, route, control);

  EXPECT_FALSE(selected.ok());
  EXPECT_EQ(selected.reason_code, "TIMEOUT");
}

TEST(DualModePlanner, CompositionTimeoutIsNotReportedAsSuccess) {
  ManualClock clock;
  PlanningRequest input = Request(EnvironmentMode::kLunarSurface, clock);
  GlobalRoute route;
  route.poses_map.resize(4096U);
  for (std::size_t index = 0U; index < route.poses_map.size(); ++index) {
    route.poses_map[index].position_m = {
        .x = static_cast<double>(index), .y = 1.0, .z = 4.25};
  }
  SteadyClock::time_point now{};
  SearchControl control{
      .deadline = SteadyClock::time_point{5ms},
      .now = [&now] {
        now += 1ms;
        return now;
      },
  };

  const auto composed = hierarchical::ComposeSurfaceReference(
      input, route, *TrajectoryTo(input.goal_map).data, control);

  EXPECT_FALSE(composed.ok());
  EXPECT_EQ(composed.reason_code, "TIMEOUT");
}

TEST(DualModePlanner, PreservesCanceledNoPathAndExceptionSemantics) {
  struct Case final {
    LocalPlanStatus local_status;
    const char* local_reason;
    PlanningStatus expected_status;
    const char* expected_reason;
  };
  for (const Case& test_case : {
           Case{LocalPlanStatus::kCanceled, "REQUEST_CANCELED",
                PlanningStatus::kCanceled, "REQUEST_CANCELED"},
           Case{LocalPlanStatus::kNoPath, "NO_PATH", PlanningStatus::kNoPath,
                "NO_PATH"},
           Case{LocalPlanStatus::kInvalidInput, "backend detail",
                PlanningStatus::kInvalidInput, "INVALID_INPUT"},
           Case{LocalPlanStatus::kPlannerError, "backend detail",
                PlanningStatus::kPlannerError, "PLANNER_ERROR"},
       }) {
    ManualClock clock;
    Planner planner(PlannerBackends{
        .global = {},
        .local = [test_case, &clock](const PlanningRequest&,
                                     const LocalGoalSet&, SearchControl) {
          clock.Advance(7ms);
          return LocalStageResult{.status = test_case.local_status,
                                  .reason_code = test_case.local_reason};
        },
    });
    const PlanningResult output =
        planner.Plan(Request(EnvironmentMode::kLavaTube, clock));
    EXPECT_EQ(output.status, test_case.expected_status);
    EXPECT_EQ(output.reason_code, test_case.expected_reason);
    EXPECT_EQ(output.timing.local_call_count, 1U);
    EXPECT_EQ(output.timing.local_elapsed, 7ms);
    EXPECT_EQ(output.timing.total_elapsed, 7ms);
  }

  ManualClock clock;
  Planner throwing(PlannerBackends{
      .global = {},
      .local = [&clock](const PlanningRequest&, const LocalGoalSet&,
                        SearchControl) -> LocalStageResult {
        clock.Advance(11ms);
        throw std::runtime_error("backend-specific exception");
      },
  });
  const PlanningResult exception_output =
      throwing.Plan(Request(EnvironmentMode::kLavaTube, clock));
  EXPECT_EQ(exception_output.status, PlanningStatus::kPlannerError);
  EXPECT_EQ(exception_output.reason_code, "PLANNER_ERROR");
  EXPECT_EQ(exception_output.timing.local_call_count, 1U);
  EXPECT_EQ(exception_output.timing.local_elapsed, 11ms);
  EXPECT_EQ(exception_output.timing.total_elapsed, 11ms);
}

TEST(DualModePlanner, PreservesGlobalFailureAndExceptionSemantics) {
  struct Case final {
    const char* reason;
    PlanningStatus expected_status;
    const char* expected_reason;
  };
  for (const Case& test_case : {
           Case{"NO_PATH", PlanningStatus::kNoPath, "NO_PATH"},
           Case{"TIMEOUT", PlanningStatus::kTimedOut, "TIMEOUT"},
           Case{"REQUEST_CANCELED", PlanningStatus::kCanceled,
                "REQUEST_CANCELED"},
           Case{"INVALID_INPUT", PlanningStatus::kInvalidInput,
                "INVALID_INPUT"},
           Case{"backend detail", PlanningStatus::kPlannerError,
                "PLANNER_ERROR"},
       }) {
    ManualClock clock;
    std::size_t local_calls{};
    Planner planner(PlannerBackends{
        .global = [test_case, &clock](const PlanningRequest&, SearchControl) {
          clock.Advance(13ms);
          return GlobalStageResult{.route = std::nullopt,
                                   .reason_code = test_case.reason};
        },
        .local = [&local_calls](const PlanningRequest&, const LocalGoalSet&,
                                SearchControl) {
          ++local_calls;
          return LocalStageResult{};
        },
    });
    const PlanningResult output =
        planner.Plan(Request(EnvironmentMode::kLunarSurface, clock));
    EXPECT_EQ(output.status, test_case.expected_status);
    EXPECT_EQ(output.reason_code, test_case.expected_reason);
    EXPECT_EQ(local_calls, 0U);
    EXPECT_EQ(output.timing.global_call_count, 1U);
    EXPECT_EQ(output.timing.global_elapsed, 13ms);
    EXPECT_EQ(output.timing.local_call_count, 0U);
    EXPECT_EQ(output.timing.total_elapsed, 13ms);
  }

  ManualClock clock;
  Planner throwing(PlannerBackends{
      .global = [&clock](const PlanningRequest&,
                         SearchControl) -> GlobalStageResult {
        clock.Advance(9ms);
        throw std::runtime_error("backend-specific exception");
      },
      .local = {},
  });
  const PlanningResult exception_output =
      throwing.Plan(Request(EnvironmentMode::kLunarSurface, clock));
  EXPECT_EQ(exception_output.status, PlanningStatus::kPlannerError);
  EXPECT_EQ(exception_output.reason_code, "PLANNER_ERROR");
  EXPECT_EQ(exception_output.timing.global_call_count, 1U);
  EXPECT_EQ(exception_output.timing.global_elapsed, 9ms);
  EXPECT_EQ(exception_output.timing.local_call_count, 0U);
  EXPECT_EQ(exception_output.timing.total_elapsed, 9ms);
}

TEST(DualModePlanner, PreCanceledRequestNeverEntersABackend) {
  CountingBackends backends;
  PlanningRequest input = Request(EnvironmentMode::kLavaTube, backends.clock);
  std::stop_source stop;
  stop.request_stop();
  input.control.stop_token = stop.get_token();

  const PlanningResult output = backends.planner.Plan(input);

  EXPECT_EQ(output.status, PlanningStatus::kCanceled);
  EXPECT_EQ(output.reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(backends.global_calls, 0U);
  EXPECT_EQ(backends.local_calls, 0U);
}

TEST(DualModePlanner, InjectedClockExceptionsAlwaysReturnPlannerError) {
  const auto run = [](const std::size_t throw_on_call,
                      std::size_t* observed_calls = nullptr) {
    std::size_t calls{};
    ManualClock request_clock;
    PlanningRequest input =
        Request(EnvironmentMode::kLavaTube, request_clock);
    input.control.now = [&calls, throw_on_call] {
      ++calls;
      if (throw_on_call != 0U && calls == throw_on_call) {
        throw std::runtime_error("injected now failure");
      }
      return SteadyClock::time_point{};
    };
    Planner planner(PlannerBackends{
        .global = {},
        .local = [](const PlanningRequest&, const LocalGoalSet& goals,
                    SearchControl) { return TrajectoryTo(FirstGoal(goals)); },
    });
    PlanningResult output = planner.Plan(input);
    if (observed_calls != nullptr) {
      *observed_calls = calls;
    }
    return output;
  };

  EXPECT_EQ(run(1U).status, PlanningStatus::kPlannerError);
  EXPECT_EQ(run(3U).status, PlanningStatus::kPlannerError);

  std::size_t successful_call_count{};
  const PlanningResult baseline = run(0U, &successful_call_count);
  ASSERT_EQ(baseline.status, PlanningStatus::kSuccess);
  ASSERT_GT(successful_call_count, 3U);
  const PlanningResult final_clock_failure = run(successful_call_count);
  EXPECT_EQ(final_clock_failure.status, PlanningStatus::kPlannerError);
  EXPECT_EQ(final_clock_failure.reason_code, "PLANNER_ERROR");
}

TEST(DualModePlanner, IgnoresPointGoalZIncludingNan) {
  for (const double z : {0.0, 100.0,
                         std::numeric_limits<double>::quiet_NaN()}) {
    CountingBackends backends;
    PlanningRequest input = Request(EnvironmentMode::kLavaTube, backends.clock);
    input.goal_map = PointTarget(4.0, 1.0, z);

    const PlanningResult output = backends.planner.Plan(input);

    EXPECT_EQ(output.status, PlanningStatus::kSuccess) << output.reason_code;
    ASSERT_TRUE(output.reference.has_value());
    const auto* trajectory =
        std::get_if<TrajectoryReference>(&output.reference->data);
    ASSERT_NE(trajectory, nullptr);
    EXPECT_DOUBLE_EQ(trajectory->points.back().pose.position_m.z, 4.25);
  }
}

[[nodiscard]] WheelMotionPrimitive WheelForwardPrimitive() {
  return WheelMotionPrimitive{
      .primitive_id = "forward",
      .kind = WheelPrimitiveKind::kForward,
      .relative_end_pose = {
          .position_m = {.x = 0.2, .y = 0.0, .z = 0.0},
      },
  };
}

[[nodiscard]] WheeledCapability RealWheelCapability() {
  return WheeledCapability{
      .footprint_xy_m = {{-.1, -.1}, {.1, -.1}, {.1, .1}, {-.1, .1}},
      .body_extent_m = {.x = 0.2, .y = 0.2, .z = 0.2},
      .wheel_diameter_m = 0.1,
      .wheel_width_m = 0.05,
      .wheelbase_m = 0.1,
      .track_width_m = 0.1,
      .minimum_underbody_clearance_m = 0.05,
      .maximum_local_obstacle_relief_m = 0.1,
      .allow_unsupported_gap = false,
      .minimum_body_z_m = 0.0,
      .maximum_body_z_m = 1.0,
      .maximum_forward_speed_mps = 1.0,
      .maximum_reverse_speed_mps = 0.5,
      .maximum_spin_rate_radps = 1.0,
      .maximum_acceleration_mps2 = 1.0,
      .maximum_braking_deceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .maximum_lateral_acceleration_mps2 = 1.0,
      .maximum_curvature_per_m = 2.0,
      .maximum_slope_rad = 0.5,
      .minimum_clearance_m = 0.0,
      .motion_primitives = {WheelForwardPrimitive()},
  };
}

[[nodiscard]] LeggedCapability RealLeggedCapability() {
  return LeggedCapability{
      .body_extent_m = {0.2, 0.2, 0.2},
      .nominal_body_height_m = 0.5,
      .platform_mass_kg = 16.0,
      .nominal_payload_kg = 2.0,
      .maximum_payload_kg = 10.0,
      .maximum_slope_rad = 0.6,
      .maximum_step_height_m = 0.2,
      .maximum_gap_width_m = 0.0,
      .minimum_body_clearance_m = 0.0,
      .step_vertical_rate_mps = 0.2,
      .body_height_m = {0.4, 0.6},
      .forward_speed_mps = {-0.6, 0.8},
      .lateral_speed_mps = {-0.4, 0.4},
      .yaw_rate_radps = {-0.8, 0.8},
      .maximum_linear_acceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .unknown_is_traversable = false,
      .motion_primitives = {
          LeggedBodyPrimitive{
              .primitive_id = "forward",
              .kind = LeggedPrimitiveKind::kForward,
              .body_frame_displacement_m = {0.2, 0.0, 0.0}},
      },
  };
}

[[nodiscard]] HopperCapability RealHopperCapability() {
  return HopperCapability{
      .specific_impulse_s = 301.0,
      .reference_total_mass_kg = 20.0,
      .reference_propellant_mass_kg = 0.2,
      .gravity_mps2 = {0.0, 0.0, -1.62},
      .reference_horizontal_range_m = 8.0,
      .reference_elevation_delta_m = 0.0,
      .runtime_fallback_allowed = false,
      .landing_support_radius_m = 0.2,
      .flight_collision_radius_m = 0.1,
      .maximum_landing_plane_residual_m = 0.05,
      .landing_lateral_margin_m = 0.1,
      .flight_map_margin_m = 0.05,
      .reachability_delta_v_margin_ratio = 0.1,
      .standard_gravity_mps2 = 9.80665,
      .maximum_landing_slope_rad = 0.3,
  };
}

[[nodiscard]] PlanningRequest RealRequest(const PlatformType platform,
                                          const EnvironmentMode mode) {
  PlanningRequest input;
  input.request_id = "real-default-backend";
  input.environment_mode = mode;
  input.goal_map = PointTarget(1.0, 1.0,
                               std::numeric_limits<double>::quiet_NaN());
  input.goal_map.yaw_rad = std::nullopt;
  input.world.local_map = LocalMap();
  std::get<std::vector<float>>(
      input.world.local_map.layers.at("elevation").values)
      .assign(input.world.local_map.CellCount(), 0.0F);
  input.world.global_map = mode == EnvironmentMode::kLunarSurface
                               ? std::optional<GridMap>{GlobalMap()}
                               : std::nullopt;
  input.world.map_from_odom = {
      .parent_frame = "map",
      .child_frame = "odom",
  };
  switch (platform) {
    case PlatformType::kWheeled:
      input.current_state = WheeledState{
          .pose = {.position_m = {.x = 1.0, .y = 1.0, .z = 0.0}},
      };
      input.capability = RealWheelCapability();
      break;
    case PlatformType::kLegged:
      input.current_state = LeggedState{
          .body_pose = {.position_m = {.x = 1.0, .y = 1.0, .z = 0.5}},
      };
      input.capability = RealLeggedCapability();
      break;
    case PlatformType::kHopper:
      input.current_state = HopperState{
          .pose = {.position_m = {.x = 1.0, .y = 1.0, .z = 0.0}},
      };
      input.capability = RealHopperCapability();
      break;
  }
  return input;
}

TEST(DualModePlanner, DefaultBackendsSolveAllThreePlatformsInBothModes) {
  for (const PlatformType platform : {
           PlatformType::kWheeled,
           PlatformType::kLegged,
           PlatformType::kHopper,
       }) {
    for (const EnvironmentMode mode : {
             EnvironmentMode::kLunarSurface,
             EnvironmentMode::kLavaTube,
         }) {
      Planner planner;
      const PlanningResult output = planner.Plan(RealRequest(platform, mode));

      EXPECT_EQ(output.status, PlanningStatus::kSuccess)
          << static_cast<int>(platform) << ":" << static_cast<int>(mode)
          << ":" << output.reason_code;
      ASSERT_TRUE(output.reference.has_value());
      EXPECT_EQ(output.reference->platform_type, platform);
      EXPECT_EQ(output.timing.global_call_count,
                mode == EnvironmentMode::kLunarSurface ? 1U : 0U);
      EXPECT_EQ(output.timing.local_call_count, 1U);
      EXPECT_FALSE(output.reference->preview.poses_map.empty());
      EXPECT_DOUBLE_EQ(
          output.reference->preview.poses_map.back().position_m.x, 1.0);
      EXPECT_DOUBLE_EQ(
          output.reference->preview.poses_map.back().position_m.y, 1.0);
    }
  }
}

void SetCacheableSequences(PlanningRequest& input) {
  input.world.global_map_sequence = 11U;
  input.world.local_map_sequence = 12U;
  input.world.odometry_sequence = 13U;
  input.world.tf_sequence = 14U;
}

void ExpectEveryCacheMiss(const PlanningResult& result) {
  EXPECT_FALSE(result.global_snapshot_cache_hit);
  EXPECT_FALSE(result.global_projection_cache_hit);
  EXPECT_FALSE(result.global_route_cache_hit);
  EXPECT_FALSE(result.local_snapshot_cache_hit);
  EXPECT_FALSE(result.local_projection_cache_hit);
  EXPECT_FALSE(result.goal_field_cache_hit);
}

void ExpectEveryCacheHit(const PlanningResult& result) {
  EXPECT_TRUE(result.global_snapshot_cache_hit);
  EXPECT_TRUE(result.global_projection_cache_hit);
  EXPECT_TRUE(result.global_route_cache_hit);
  EXPECT_TRUE(result.local_snapshot_cache_hit);
  EXPECT_TRUE(result.local_projection_cache_hit);
  EXPECT_TRUE(result.goal_field_cache_hit);
}

TEST(DualModePlanner, DefaultPlannerReportsColdThenWarmArtifactHits) {
  Planner planner;
  PlanningRequest input =
      RealRequest(PlatformType::kWheeled, EnvironmentMode::kLunarSurface);
  SetCacheableSequences(input);

  const PlanningResult cold = planner.Plan(input);
  const PlanningResult warm = planner.Plan(input);

  ASSERT_EQ(cold.status, PlanningStatus::kSuccess) << cold.reason_code;
  ASSERT_EQ(warm.status, PlanningStatus::kSuccess) << warm.reason_code;
  ExpectEveryCacheMiss(cold);
  ExpectEveryCacheHit(warm);
}

TEST(DualModePlanner, ZeroSourceSequencesNeverReusePlannerArtifacts) {
  Planner planner;
  const PlanningRequest input =
      RealRequest(PlatformType::kWheeled, EnvironmentMode::kLunarSurface);

  const PlanningResult first = planner.Plan(input);
  const PlanningResult second = planner.Plan(input);

  ASSERT_EQ(first.status, PlanningStatus::kSuccess) << first.reason_code;
  ASSERT_EQ(second.status, PlanningStatus::kSuccess) << second.reason_code;
  ExpectEveryCacheMiss(first);
  ExpectEveryCacheMiss(second);
}

TEST(DualModePlanner, SourceRevisionChangesInvalidateOnlyDependentArtifacts) {
  struct Case final {
    void (*mutate)(PlanningRequest&);
    bool global_snapshot_hit;
    bool global_projection_hit;
    bool global_route_hit;
    bool local_snapshot_hit;
    bool local_projection_hit;
    bool goal_field_hit;
  };
  const Case cases[]{
      {[](PlanningRequest& request) { ++request.world.global_map_sequence; },
       false, false, false, true, true, true},
      {[](PlanningRequest& request) { ++request.world.local_map_sequence; },
       true, true, true, false, false, false},
      {[](PlanningRequest& request) { ++request.world.odometry_sequence; },
       true, true, false, true, true, true},
      {[](PlanningRequest& request) { ++request.world.tf_sequence; },
       true, true, false, true, true, true},
  };
  for (const Case& test_case : cases) {
    Planner planner;
    PlanningRequest input =
        RealRequest(PlatformType::kWheeled, EnvironmentMode::kLunarSurface);
    SetCacheableSequences(input);
    ASSERT_EQ(planner.Plan(input).status, PlanningStatus::kSuccess);
    test_case.mutate(input);

    const PlanningResult changed = planner.Plan(input);

    ASSERT_EQ(changed.status, PlanningStatus::kSuccess) << changed.reason_code;
    EXPECT_EQ(changed.global_snapshot_cache_hit,
              test_case.global_snapshot_hit);
    EXPECT_EQ(changed.global_projection_cache_hit,
              test_case.global_projection_hit);
    EXPECT_EQ(changed.global_route_cache_hit, test_case.global_route_hit);
    EXPECT_EQ(changed.local_snapshot_cache_hit,
              test_case.local_snapshot_hit);
    EXPECT_EQ(changed.local_projection_cache_hit,
              test_case.local_projection_hit);
    EXPECT_EQ(changed.goal_field_cache_hit, test_case.goal_field_hit);
  }
}

TEST(DualModePlanner, SemanticChangesInvalidateOnlyDependentArtifacts) {
  struct Case final {
    void (*mutate)(PlanningRequest&);
    bool global_projection_hit;
    bool global_route_hit;
    bool local_projection_hit;
    bool goal_field_hit;
  };
  const Case cases[]{
      {[](PlanningRequest& request) {
         ++request.config.global_occupancy_threshold;
       }, false, false, true, true},
      {[](PlanningRequest& request) {
         request.config.local_occupancy_threshold = 0.6;
       }, true, true, false, false},
      {[](PlanningRequest& request) {
         auto& capability = std::get<WheeledCapability>(request.capability);
         capability.maximum_forward_speed_mps = 0.9;
       }, false, false, true, false},
      {[](PlanningRequest& request) {
         request.goal_map.goal_id = "changed-goal";
         std::get<PointGoal>(request.goal_map.target).position_m.x = 1.1;
       }, true, false, true, false},
      {[](PlanningRequest& request) {
         request.config.search.stop_after_first_solution = false;
       }, true, false, true, false},
  };
  for (const Case& test_case : cases) {
    Planner planner;
    PlanningRequest input =
        RealRequest(PlatformType::kWheeled, EnvironmentMode::kLunarSurface);
    SetCacheableSequences(input);
    ASSERT_EQ(planner.Plan(input).status, PlanningStatus::kSuccess);
    test_case.mutate(input);

    const PlanningResult changed = planner.Plan(input);

    ASSERT_EQ(changed.status, PlanningStatus::kSuccess) << changed.reason_code;
    EXPECT_TRUE(changed.global_snapshot_cache_hit);
    EXPECT_EQ(changed.global_projection_cache_hit,
              test_case.global_projection_hit);
    EXPECT_EQ(changed.global_route_cache_hit, test_case.global_route_hit);
    EXPECT_TRUE(changed.local_snapshot_cache_hit);
    EXPECT_EQ(changed.local_projection_cache_hit,
              test_case.local_projection_hit);
    EXPECT_EQ(changed.goal_field_cache_hit, test_case.goal_field_hit);
  }
}

[[nodiscard]] RigidTransform NonUnitMapFromOdom() {
  constexpr double kHalfYaw = std::numbers::pi / 4.0;
  return RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .translation_m = {10.0, -3.0, 2.0},
      .rotation = {.w = std::cos(kHalfYaw), .z = std::sin(kHalfYaw)},
  };
}

[[nodiscard]] MotionReferenceData DetailedLocalReference(
    const PlatformType platform) {
  if (platform == PlatformType::kHopper) {
    return HopReference{
        .segments = {{
            .segment_id = "hop-odom",
            .launch_pose = {
                .position_m = {1.0, 2.0, 3.0},
                .orientation = {},
            },
            .landing_region_boundary_m = {{2.0, 3.0, 4.0},
                                          {-1.0, 4.0, 0.0}},
            .flight_time = 2250ms,
            .launch_velocity_mps = {8.0, 9.0, 10.0},
            .flight_tube_radius_m = 0.7,
            .nominal_landing_point_m = {5.0, 6.0, 7.0},
            .required_delta_v_mps = 11.0,
            .available_delta_v_mps = 12.0,
            .capability_version = "hopper-v1",
            .global_map_generation = 13U,
            .local_map_generation = 14U,
        }},
    };
  }
  return TrajectoryReference{
      .semantics = platform == PlatformType::kWheeled
                       ? TrajectorySemantics::kWheeledBase
                       : TrajectorySemantics::kLeggedBodyReference,
      .points = {{
          .time_from_start = 1500ms,
          .pose = {
              .position_m = {1.0, 2.0, 3.0},
              .orientation = {},
          },
          .velocity = {
              .linear_mps = {4.0, 5.0, 6.0},
              .angular_radps = {0.4, 0.5, 0.6},
          },
      }},
  };
}

void SetPlatform(PlanningRequest& input, const PlatformType platform) {
  switch (platform) {
    case PlatformType::kWheeled:
      input.current_state = WheeledState{};
      input.capability = WheeledCapability{};
      return;
    case PlatformType::kLegged:
      input.current_state = LeggedState{};
      input.capability = LeggedCapability{};
      return;
    case PlatformType::kHopper:
      input.current_state = HopperState{};
      input.capability = HopperCapability{};
      return;
  }
}

TEST(DualModePlanner,
     ComposesAllPlatformExecutionDataInMapExactlyOnceForBothModes) {
  constexpr double kSqrtHalf = 0.70710678118654752440;
  for (const PlatformType platform : {
           PlatformType::kWheeled,
           PlatformType::kLegged,
           PlatformType::kHopper,
       }) {
    for (const EnvironmentMode mode : {
             EnvironmentMode::kLunarSurface,
             EnvironmentMode::kLavaTube,
         }) {
      ManualClock clock;
      PlanningRequest input = Request(mode, clock);
      SetPlatform(input, platform);
      input.world.map_from_odom = NonUnitMapFromOdom();
      input.goal_map = PointTarget(9.0, -1.0);
      input.goal_map.yaw_rad = std::nullopt;
      const GlobalRoute route{
          .poses_map = {{.position_m = {50.0, 60.0, 70.0}}},
          .expanded_states = 17U,
      };
      Planner planner(PlannerBackends{
          .global = [route](const PlanningRequest&, SearchControl) {
            return GlobalStageResult{.route = route};
          },
          .local = [platform](const PlanningRequest&, const LocalGoalSet&,
                              SearchControl) {
            return LocalStageResult{
                .status = LocalPlanStatus::kSolved,
                .data = DetailedLocalReference(platform),
            };
          },
      });

      const PlanningResult output = planner.Plan(input);

      ASSERT_EQ(output.status, PlanningStatus::kSuccess)
          << static_cast<int>(platform) << ":" << static_cast<int>(mode)
          << ":" << output.reason_code;
      ASSERT_TRUE(output.reference.has_value());
      if (mode == EnvironmentMode::kLunarSurface) {
        ASSERT_EQ(output.reference->preview.poses_map.size(), 1U);
        EXPECT_EQ(output.reference->preview.poses_map.front(),
                  route.poses_map.front());
      }
      if (const auto* trajectory =
              std::get_if<TrajectoryReference>(&output.reference->data)) {
        ASSERT_NE(platform, PlatformType::kHopper);
        ASSERT_EQ(trajectory->points.size(), 1U);
        const TrajectoryPoint& point = trajectory->points.front();
        EXPECT_EQ(point.time_from_start, 1500ms);
        EXPECT_NEAR(point.pose.position_m.x, 8.0, 1.0e-12);
        EXPECT_NEAR(point.pose.position_m.y, -2.0, 1.0e-12);
        EXPECT_NEAR(point.pose.position_m.z, 5.0, 1.0e-12);
        EXPECT_NEAR(point.pose.orientation.w, kSqrtHalf, 1.0e-12);
        EXPECT_NEAR(point.pose.orientation.x, 0.0, 1.0e-12);
        EXPECT_NEAR(point.pose.orientation.y, 0.0, 1.0e-12);
        EXPECT_NEAR(point.pose.orientation.z, kSqrtHalf, 1.0e-12);
        EXPECT_NEAR(point.velocity.linear_mps.x, -5.0, 1.0e-12);
        EXPECT_NEAR(point.velocity.linear_mps.y, 4.0, 1.0e-12);
        EXPECT_NEAR(point.velocity.linear_mps.z, 6.0, 1.0e-12);
        EXPECT_NEAR(point.velocity.angular_radps.x, -0.5, 1.0e-12);
        EXPECT_NEAR(point.velocity.angular_radps.y, 0.4, 1.0e-12);
        EXPECT_NEAR(point.velocity.angular_radps.z, 0.6, 1.0e-12);
        if (mode == EnvironmentMode::kLavaTube) {
          ASSERT_EQ(output.reference->preview.poses_map.size(), 1U);
          EXPECT_EQ(output.reference->preview.poses_map.front(), point.pose);
        }
        continue;
      }

      ASSERT_EQ(platform, PlatformType::kHopper);
      const auto& hops = std::get<HopReference>(output.reference->data);
      ASSERT_EQ(hops.segments.size(), 1U);
      const HopSegment& segment = hops.segments.front();
      EXPECT_EQ(segment.segment_id, "hop-odom");
      EXPECT_NEAR(segment.launch_pose.position_m.x, 8.0, 1.0e-12);
      EXPECT_NEAR(segment.launch_pose.position_m.y, -2.0, 1.0e-12);
      EXPECT_NEAR(segment.launch_pose.position_m.z, 5.0, 1.0e-12);
      EXPECT_NEAR(segment.launch_pose.orientation.w, kSqrtHalf, 1.0e-12);
      EXPECT_NEAR(segment.launch_pose.orientation.z, kSqrtHalf, 1.0e-12);
      ASSERT_EQ(segment.landing_region_boundary_m.size(), 2U);
      EXPECT_NEAR(segment.landing_region_boundary_m[0].x, 7.0, 1.0e-12);
      EXPECT_NEAR(segment.landing_region_boundary_m[0].y, -1.0, 1.0e-12);
      EXPECT_NEAR(segment.landing_region_boundary_m[0].z, 6.0, 1.0e-12);
      EXPECT_NEAR(segment.landing_region_boundary_m[1].x, 6.0, 1.0e-12);
      EXPECT_NEAR(segment.landing_region_boundary_m[1].y, -4.0, 1.0e-12);
      EXPECT_NEAR(segment.landing_region_boundary_m[1].z, 2.0, 1.0e-12);
      EXPECT_EQ(segment.flight_time, 2250ms);
      EXPECT_NEAR(segment.launch_velocity_mps.x, -9.0, 1.0e-12);
      EXPECT_NEAR(segment.launch_velocity_mps.y, 8.0, 1.0e-12);
      EXPECT_NEAR(segment.launch_velocity_mps.z, 10.0, 1.0e-12);
      EXPECT_DOUBLE_EQ(segment.flight_tube_radius_m, 0.7);
      EXPECT_NEAR(segment.nominal_landing_point_m.x, 4.0, 1.0e-12);
      EXPECT_NEAR(segment.nominal_landing_point_m.y, 2.0, 1.0e-12);
      EXPECT_NEAR(segment.nominal_landing_point_m.z, 9.0, 1.0e-12);
      EXPECT_DOUBLE_EQ(segment.required_delta_v_mps, 11.0);
      EXPECT_DOUBLE_EQ(segment.available_delta_v_mps, 12.0);
      EXPECT_EQ(segment.capability_version, "hopper-v1");
      EXPECT_EQ(segment.global_map_generation, 13U);
      EXPECT_EQ(segment.local_map_generation, 14U);
      if (mode == EnvironmentMode::kLavaTube) {
        ASSERT_EQ(output.reference->preview.poses_map.size(), 2U);
        EXPECT_EQ(output.reference->preview.poses_map.front(),
                  segment.launch_pose);
        EXPECT_EQ(output.reference->preview.poses_map.back().position_m,
                  segment.nominal_landing_point_m);
      }
    }
  }
}

TEST(DualModePlanner,
     LeggedDefaultBackendStopsDuringItsSecondTrajectoryConversion) {
  PlanningRequest baseline =
      RealRequest(PlatformType::kLegged, EnvironmentMode::kLavaTube);
  baseline.goal_map = PointTarget(1.2, 1.0);
  baseline.goal_map.yaw_rad = std::nullopt;
  std::size_t backend_clock_reads = 0U;
  SearchControl backend_control{
      .deadline = SteadyClock::time_point::max(),
      .now = [&] {
        ++backend_clock_reads;
        return SteadyClock::time_point{};
      },
  };
  auto snapshot = shared::MapSnapshot::Create(
      baseline.world.local_map, shared::MapContract::kLocalElevation,
      backend_control);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  auto projection = shared::BuildLocalTerrainProjection(
      snapshot.snapshot,
      static_cast<float>(baseline.config.local_occupancy_threshold),
      backend_control);
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const auto& state = std::get<LeggedState>(baseline.current_state);
  const auto& capability = std::get<LeggedCapability>(baseline.capability);
  const legged::LeggedPlanResult backend_result = legged::PlanLegged({
      .start = state,
      .goal_odom = baseline.goal_map,
      .terrain = &*projection.value,
      .capability = &capability,
      .control = backend_control,
      .search = baseline.config.search,
  });
  ASSERT_TRUE(backend_result.ok()) << backend_result.reason_code;
  ASSERT_FALSE(backend_result.trajectory.empty());

  // Planner entry and local-stage timing each read the clock once before the
  // default backend performs the same map/projection/search work above.
  const std::size_t first_second_conversion_read = backend_clock_reads + 2U;
  std::stop_source stop;
  std::size_t cancel_reads = 0U;
  PlanningRequest canceled = baseline;
  canceled.control.stop_token = stop.get_token();
  canceled.control.deadline = SteadyClock::time_point::max();
  canceled.control.now = [&] {
    if (cancel_reads++ == first_second_conversion_read) {
      stop.request_stop();
    }
    return SteadyClock::time_point{};
  };
  const PlanningResult canceled_result = Planner{}.Plan(canceled);
  EXPECT_EQ(canceled_result.status, PlanningStatus::kCanceled)
      << "clock_reads=" << cancel_reads;
  EXPECT_EQ(canceled_result.reason_code, "REQUEST_CANCELED");
  EXPECT_FALSE(canceled_result.reference.has_value());

  std::size_t timeout_reads = 0U;
  PlanningRequest timed_out = baseline;
  timed_out.control.now = [&] {
    return timeout_reads++ < first_second_conversion_read
               ? SteadyClock::time_point{}
               : SteadyClock::time_point{3s};
  };
  const PlanningResult timeout_result = Planner{}.Plan(timed_out);
  EXPECT_EQ(timeout_result.status, PlanningStatus::kTimedOut)
      << "clock_reads=" << timeout_reads;
  EXPECT_EQ(timeout_result.reason_code, "TIMEOUT");
  EXPECT_FALSE(timeout_result.reference.has_value());
}

TEST(DualModePlanner, LargeGlobalProjectionRespectsTheHardDeadline) {
  constexpr std::size_t kWidth = 2048U;
  ManualClock fixture_clock;
  PlanningRequest input =
      Request(EnvironmentMode::kLunarSurface, fixture_clock);
  SteadyClock::time_point now{};
  input.request_started_at = now;
  input.control.now = [&now] {
    now += 250ms;
    return now;
  };
  input.world.global_map = GridMap{
      .frame_id = "map",
      .width = kWidth,
      .height = kWidth,
      .resolution_m = 1.0,
      .origin_m = {},
      .layers = {
          {"occupancy", GridLayer{.values = std::vector<std::int8_t>(
                                       kWidth * kWidth, 100)}},
      },
  };
  Planner planner;
  const PlanningResult output = planner.Plan(input);

  EXPECT_EQ(output.status, PlanningStatus::kTimedOut) << output.reason_code;
  EXPECT_EQ(output.reason_code, "TIMEOUT");
  EXPECT_GE(output.timing.total_elapsed, 3s);
  EXPECT_EQ(output.timing.global_call_count, 1U);
  EXPECT_EQ(output.timing.local_call_count, 0U);
}

}  // namespace
}  // namespace lunar::pure_planning
