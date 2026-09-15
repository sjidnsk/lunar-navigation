#include <gtest/gtest.h>
#include "lunar_incremental_navigation_core/local_goal_region.hpp"

#include <limits>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <numbers>
#include <string>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include "lunar_incremental_navigation_core/planning_session_coordinator.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] RigidTransform IdentityMapTransform() {
  return RigidTransform{.parent_frame = "map", .child_frame = "map"};
}

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot> MakeFine(
    const std::uint64_t revision = 1U,
    const std::optional<GridIndex> blocked = std::nullopt,
    const FineCellState changed_state = FineCellState::kBlocked) {
  PersistentElevationMap map;
  const GridGeometry geometry{.frame_id = "map",
                              .width = 8U,
                              .height = 8U,
                              .resolution_m = 1.0};
  const std::vector<float> elevation(geometry.CellCount(), 0.0F);
  const auto update = map.Apply(ElevationEvidence{
      .geometry = geometry,
      .elevation_m = elevation,
      .map_from_source = IdentityMapTransform(),
  });
  if (update.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("fine fixture elevation was rejected");
  }
  const auto raw = map.Snapshot();
  FineTraversabilityTile::StateArray states;
  states.fill(FineCellState::kUnknown);
  FineTraversabilityTile::CostArray costs;
  costs.fill(0.0);
  for (std::int64_t y = 0; y < 8; ++y) {
    for (std::int64_t x = 0; x < 8; ++x) {
      states[TileCellOffset({.x = x, .y = y})] = FineCellState::kFree;
    }
  }
  if (blocked) {
    states[TileCellOffset(*blocked)] = changed_state;
  }
  auto tile = std::make_shared<const FineTraversabilityTile>(
      std::move(states), std::move(costs));
  FineTraversabilityTileDirectory directory(raw->geometry());
  directory = directory.WithTile(TileIndex{}, std::move(tile));
  return std::make_shared<const FineTraversabilitySnapshot>(
      raw->geometry(), raw->raw_elevation_revision(), revision,
      "test-profile", 0.25, 0.0, TraversalCostWeights{}, raw,
      std::move(directory), std::vector<TileIndex>{TileIndex{}},
      std::vector<TileIndex>{TileIndex{}}, FineSnapshotMetrics{});
}

[[nodiscard]] std::shared_ptr<const GlobalGuidanceSnapshot> MakeGuidance(
    const FineTraversabilitySnapshot& fine) {
  GlobalGuidanceTile::StateArray states;
  states.fill(GuidanceCellState::kUnknown);
  GlobalGuidanceTile::RiskArray risks;
  risks.fill(0.0);
  states[TileCellOffset({.x = 0, .y = 0})] = GuidanceCellState::kCandidate;
  auto tile = std::make_shared<const GlobalGuidanceTile>(std::move(states),
                                                         std::move(risks));
  GlobalGuidanceTileDirectory directory(fine.geometry());
  directory = directory.WithTile(TileIndex{}, std::move(tile));
  return std::make_shared<const GlobalGuidanceSnapshot>(
      fine.geometry(), fine.raw_elevation_revision(),
      fine.fine_traversability_revision(), 1U, "test-profile",
      std::move(directory), std::vector<TileIndex>{TileIndex{}},
      std::vector<TileIndex>{TileIndex{}});
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw_rad) {
  return Quaternion{.w = std::cos(yaw_rad / 2.0),
                    .z = std::sin(yaw_rad / 2.0)};
}

struct FakePorts final {
  int global_calls{};
  int target_calls{};
  int patch_calls{};
  int wheel_calls{};
  int legged_calls{};
  GuidanceStatus guidance_status{GuidanceStatus::kAvailable};
  StartPatchResult::Status patch_status{StartPatchResult::Status::kNotNeeded};
  LocalPlanResult::Status local_status{LocalPlanResult::Status::kPlanFound};
  std::string local_reason_code;
  bool reaches_final_goal{};
  std::shared_ptr<const LocalGoalRegion> target_region;
  bool throw_from_target{};
  bool selector_received_guidance{};
  const FineTraversabilitySnapshot* fine_seen_by_target{};
  std::shared_ptr<const FineTraversabilitySnapshot> fine_seen_by_patch;
  std::shared_ptr<const FineTraversabilitySnapshot> fine_seen_by_local;
  std::optional<SparseGridGeometry> local_window_seen_by_target;
  std::optional<SparseGridGeometry> local_window_seen_by_patch;
  std::optional<LocalTarget> target_seen_by_local;

  [[nodiscard]] PlanningSessionPorts Bind() {
    return PlanningSessionPorts{
        .plan_global =
            [this](const GlobalGuidanceSnapshot&, Point2, Point2,
                   SearchDeadline, const StopToken&) {
              ++global_calls;
              GlobalRouteResult result{.status = guidance_status};
              if (guidance_status == GuidanceStatus::kAvailable) {
                result.route = GlobalRoute{
                    .poses_map = {Pose3{.position_m = {.x = 3.5,
                                                       .y = 0.5}}}};
              }
              return result;
            },
        .select_target =
            [this](const FineTraversabilitySnapshot& fine,
                   const SparseGridGeometry& local_window, Point2,
                   const FinalGoal& goal,
                   const std::optional<GlobalRoute>& guidance) {
              ++target_calls;
              if (throw_from_target) {
                throw std::runtime_error("target failure");
              }
              selector_received_guidance = guidance.has_value();
              fine_seen_by_target = &fine;
              local_window_seen_by_target = local_window;
              return std::optional<LocalTarget>(LocalTarget{
                  .center = {.x = goal.target_x_m, .y = goal.target_y_m},
                  .position_tolerance_m = 0.5,
                  .is_final_goal = reaches_final_goal,
                  .terminal_yaw_rad = goal.has_target_yaw
                                          ? std::optional(goal.target_yaw_rad)
                                          : std::nullopt,
                  .region = target_region,
              });
            },
        .build_start_patch =
            [this](std::shared_ptr<const FineTraversabilitySnapshot> fine,
                   const SparseGridGeometry& local_window,
                   const Pose2& p0,
                   const PlatformCapability&,
                   const TraversabilityProfile&) {
              ++patch_calls;
              fine_seen_by_patch = fine;
              local_window_seen_by_patch = local_window;
              if (patch_status == StartPatchResult::Status::kStartBlocked ||
                  patch_status == StartPatchResult::Status::kUnresolved) {
                return StartPatchResult{.status = patch_status};
              }
              auto view = std::make_shared<const RequestLocalPlanningView>(
                  fine, local_window, p0, 0.0,
                  std::vector<LocalCellOverride>{});
              return StartPatchResult{.status = patch_status,
                                      .view = std::move(view)};
            },
        .plan_wheel =
            [this](const RequestLocalPlanningView& view, const Pose2& start,
                   const LocalTarget& target, SearchDeadline,
                   const StopToken&) {
              ++wheel_calls;
              fine_seen_by_local = view.base();
              target_seen_by_local = target;
              return Result(start, target);
            },
        .plan_legged =
            [this](const RequestLocalPlanningView& view, const Pose2& start,
                   const LocalTarget& target, SearchDeadline,
                   const StopToken&) {
              ++legged_calls;
              fine_seen_by_local = view.base();
              target_seen_by_local = target;
              return Result(start, target);
            },
    };
  }

 private:
  [[nodiscard]] LocalPlanResult Result(const Pose2& start,
                                       const LocalTarget& target) const {
    LocalPlanResult result{.status = local_status,
                           .reaches_final_goal = reaches_final_goal,
                           .reason_code = local_reason_code};
    if (local_status == LocalPlanResult::Status::kPlanFound) {
      result.path = {
          PathPoint{.pose = Pose3{
                        .position_m = {.x = start.position_m.x,
                                       .y = start.position_m.y,
                                       .z = 0.0},
                        .orientation = YawQuaternion(start.yaw_rad)}},
          PathPoint{.pose = Pose3{
                        .position_m = {.x = target.center.x,
                                       .y = target.center.y,
                                       .z = 0.0},
                        .orientation = YawQuaternion(
                            target.terminal_yaw_rad.value_or(0.0))}},
      };
    }
    return result;
  }
};

[[nodiscard]] WheeledCapability Wheel() { return WheeledCapability{}; }

[[nodiscard]] LeggedCapability Legged() { return LeggedCapability{}; }

[[nodiscard]] PlanningSessionCoordinatorConfig Config() {
  return PlanningSessionCoordinatorConfig{
      .goal_position_tolerance_m = 0.2,
      .goal_yaw_tolerance_rad = 0.1,
  };
}

[[nodiscard]] StateInput At(const double x, const double y,
                            const double yaw = 0.0) {
  return StateInput{.base_link_pose = {
                        .position_m = {.x = x, .y = y}, .yaw_rad = yaw}};
}

TEST(PlanningSessionCoordinator, StartsInPlanningAndRejectsInvalidGoals) {
  PlanningSessionPorts ports;
  const PlatformCapability capability = WheeledCapability{};
  const TraversabilityProfile profile{};
  PlanningSessionCoordinator coordinator(capability, profile, {}, ports);

  const SessionId first{{1U}};
  const auto accepted = coordinator.Start(
      first, FinalGoal{.target_x_m = 3.0, .target_y_m = 4.0});
  EXPECT_TRUE(accepted.accepted);
  EXPECT_EQ(coordinator.state(), CoordinatorState::kPlanning);
  EXPECT_FALSE(accepted.terminal);

  const SessionId invalid{{2U}};
  const auto rejected = coordinator.Start(
      invalid, FinalGoal{.target_x_m = 0.0,
                         .target_y_m = 0.0,
                         .has_target_yaw = true,
                         .target_yaw_rad =
                             std::numeric_limits<double>::quiet_NaN()});
  EXPECT_FALSE(rejected.accepted);
  ASSERT_TRUE(rejected.terminal);
  EXPECT_EQ(rejected.terminal->result.outcome,
            SessionOutcome::kInvalidGoal);
  EXPECT_EQ(rejected.terminal->result.reason_code, "INVALID_GOAL");
  EXPECT_EQ(coordinator.state(), CoordinatorState::kPlanning);
}

TEST(PlanningSessionCoordinator,
     OneCycleUsesOneImmutableBundleAndAtomicallyReplacesRollingSegments) {
  FakePorts fake;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);
  const auto fine = MakeFine();
  const auto guidance = MakeGuidance(*fine);

  const CycleOutput first = coordinator.PlanCycle(
      At(0.5, 0.5), SnapshotBundle{.fine = fine, .guidance = guidance},
      CycleTrigger::kContinue, SearchDeadline::max());

  ASSERT_TRUE(first.path_reference);
  EXPECT_EQ(first.path_reference->state, PathState::kActive);
  EXPECT_EQ(first.path_reference->segment_revision, 1U);
  EXPECT_EQ(first.path_reference->traversability_revision, 1U);
  EXPECT_EQ(first.feedback.session_state, SessionState::kExecuting);
  EXPECT_EQ(first.feedback.planning_cycle, 1U);
  EXPECT_EQ(first.feedback.active_segment_revision, 1U);
  EXPECT_EQ(fake.global_calls, 1);
  EXPECT_EQ(fake.target_calls, 1);
  EXPECT_EQ(fake.patch_calls, 1);
  EXPECT_EQ(fake.wheel_calls, 1);
  EXPECT_EQ(fake.legged_calls, 0);
  EXPECT_EQ(fake.fine_seen_by_target, fine.get());
  EXPECT_EQ(fake.fine_seen_by_patch, fine);
  EXPECT_EQ(fake.fine_seen_by_local, fine);
  ASSERT_TRUE(fake.local_window_seen_by_target);
  ASSERT_TRUE(fake.local_window_seen_by_patch);
  EXPECT_EQ(fake.local_window_seen_by_target->min_inclusive(),
            fine->geometry().min_inclusive());
  EXPECT_EQ(fake.local_window_seen_by_target->max_exclusive(),
            fine->geometry().max_exclusive());
  EXPECT_EQ(fake.local_window_seen_by_patch->min_inclusive(),
            fake.local_window_seen_by_target->min_inclusive());
  EXPECT_EQ(fake.local_window_seen_by_patch->max_exclusive(),
            fake.local_window_seen_by_target->max_exclusive());

  const CycleOutput idle = coordinator.PlanCycle(
      At(1.0, 0.5), SnapshotBundle{.fine = fine, .guidance = guidance},
      CycleTrigger::kContinue, SearchDeadline::max());
  EXPECT_FALSE(idle.path_reference);
  EXPECT_EQ(fake.wheel_calls, 1);

  const CycleOutput replacement = coordinator.PlanCycle(
      At(1.0, 0.5), SnapshotBundle{.fine = fine, .guidance = guidance},
      CycleTrigger::kSegmentEnd, SearchDeadline::max());
  ASSERT_TRUE(replacement.path_reference);
  EXPECT_EQ(replacement.path_reference->state, PathState::kActive);
  EXPECT_EQ(replacement.path_reference->segment_revision, 2U);
  EXPECT_FALSE(replacement.terminal);
  EXPECT_EQ(fake.wheel_calls, 2);
  EXPECT_EQ(coordinator.state(), CoordinatorState::kExecuting);
}

TEST(PlanningSessionCoordinator,
     PassesTheConfiguredMeterWindowToLocalTargetAndStartPatchPorts) {
  FakePorts fake;
  auto config = Config();
  config.local_window_size_m = 4.0;
  PlanningSessionCoordinator coordinator(Wheel(), {}, config, fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 5.5,
                                          .target_y_m = 3.5})
                  .accepted);

  const CycleOutput output = coordinator.PlanCycle(
      At(3.5, 3.5), SnapshotBundle{.fine = MakeFine()},
      CycleTrigger::kContinue, SearchDeadline::max());

  ASSERT_TRUE(output.path_reference);
  ASSERT_TRUE(fake.local_window_seen_by_target);
  ASSERT_TRUE(fake.local_window_seen_by_patch);
  EXPECT_EQ(fake.local_window_seen_by_target->width(), 4U);
  EXPECT_EQ(fake.local_window_seen_by_target->height(), 4U);
  EXPECT_EQ(fake.local_window_seen_by_target->min_inclusive(),
            (GridIndex{.x = 1, .y = 1}));
  EXPECT_EQ(fake.local_window_seen_by_target->max_exclusive(),
            (GridIndex{.x = 5, .y = 5}));
  EXPECT_EQ(fake.local_window_seen_by_patch->min_inclusive(),
            fake.local_window_seen_by_target->min_inclusive());
  EXPECT_EQ(fake.local_window_seen_by_patch->max_exclusive(),
            fake.local_window_seen_by_target->max_exclusive());
}

TEST(PlanningSessionCoordinator,
     AppliesOneInternalGoalToleranceToTheSelectedLocalTarget) {
  FakePorts fake;
  auto config = Config();
  config.goal_position_tolerance_m = 0.05;
  PlanningSessionCoordinator coordinator(Wheel(), {}, config, fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);

  const CycleOutput output = coordinator.PlanCycle(
      At(0.5, 0.5), SnapshotBundle{.fine = MakeFine()},
      CycleTrigger::kContinue, SearchDeadline::max());

  ASSERT_TRUE(output.path_reference);
  ASSERT_TRUE(fake.target_seen_by_local);
  EXPECT_DOUBLE_EQ(fake.target_seen_by_local->position_tolerance_m, 0.05);
}

TEST(PlanningSessionCoordinator, EveryGuidanceStatusFallsBackToOneLocalPlan) {
  const auto fine = MakeFine();
  const auto guidance = MakeGuidance(*fine);
  for (const GuidanceStatus status : {
           GuidanceStatus::kAvailable, GuidanceStatus::kUnavailable,
           GuidanceStatus::kNoRoute, GuidanceStatus::kTimeout}) {
    FakePorts fake;
    fake.guidance_status = status;
    PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
    ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                  FinalGoal{.target_x_m = 3.5,
                                            .target_y_m = 0.5})
                    .accepted);

    const CycleOutput output = coordinator.PlanCycle(
        At(0.5, 0.5), SnapshotBundle{.fine = fine, .guidance = guidance},
        CycleTrigger::kContinue, SearchDeadline::max());

    ASSERT_TRUE(output.path_reference);
    EXPECT_EQ(output.guidance_status, status);
    EXPECT_EQ(fake.global_calls, 1);
    EXPECT_EQ(fake.target_calls, 1);
    EXPECT_EQ(fake.patch_calls, 1);
    EXPECT_EQ(fake.wheel_calls, 1);
    EXPECT_EQ(fake.selector_received_guidance,
              status == GuidanceStatus::kAvailable);
  }

  FakePorts missing;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(),
                                         missing.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);
  const CycleOutput output = coordinator.PlanCycle(
      At(0.5, 0.5), SnapshotBundle{.fine = fine}, CycleTrigger::kContinue,
      SearchDeadline::max());
  ASSERT_TRUE(output.path_reference);
  EXPECT_EQ(output.guidance_status, GuidanceStatus::kUnavailable);
  EXPECT_EQ(missing.global_calls, 0);
  EXPECT_EQ(missing.wheel_calls, 1);
}

TEST(PlanningSessionCoordinator,
     StartPatchFailuresPropagateStableNoPathReasonCodes) {
  const auto fine = MakeFine();
  for (const auto& [status, reason] :
       std::vector<std::pair<StartPatchResult::Status, const char*>>{
           {StartPatchResult::Status::kStartBlocked, "START_BLOCKED"},
           {StartPatchResult::Status::kUnresolved,
            "START_BLIND_ZONE_UNRESOLVED"}}) {
    FakePorts fake;
    fake.patch_status = status;
    PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
    ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                  FinalGoal{.target_x_m = 3.5,
                                            .target_y_m = 0.5})
                    .accepted);

    const CycleOutput output = coordinator.PlanCycle(
        At(0.5, 0.5), SnapshotBundle{.fine = fine}, CycleTrigger::kContinue,
        SearchDeadline::max());

    ASSERT_TRUE(output.terminal);
    EXPECT_EQ(output.terminal->result.outcome, SessionOutcome::kNoPath);
    EXPECT_EQ(output.terminal->result.reason_code, reason);
    EXPECT_EQ(fake.target_calls, 1);
    EXPECT_EQ(fake.patch_calls, 1);
    EXPECT_EQ(fake.wheel_calls, 0);
    EXPECT_EQ(coordinator.state(), CoordinatorState::kIdle);
  }
}

TEST(PlanningSessionCoordinator,
     PreservesCertifiedLocalNoPathDiagnostics) {
  FakePorts fake;
  fake.local_status = LocalPlanResult::Status::kNoPath;
  fake.local_reason_code = "TERMINAL_YAW_UNREACHABLE";
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5,
                                          .has_target_yaw = true,
                                          .target_yaw_rad = 0.5})
                  .accepted);

  const CycleOutput output = coordinator.PlanCycle(
      At(0.5, 0.5), SnapshotBundle{.fine = MakeFine()},
      CycleTrigger::kContinue, SearchDeadline::max());

  ASSERT_TRUE(output.terminal);
  EXPECT_EQ(output.terminal->result.outcome, SessionOutcome::kNoPath);
  EXPECT_EQ(output.terminal->result.reason_code,
            "TERMINAL_YAW_UNREACHABLE");
}

TEST(PlanningSessionCoordinator,
     DeviationInvalidatesBeforeExactlyOneReplanAndNoPathTerminal) {
  FakePorts fake;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);
  const auto fine = MakeFine();
  ASSERT_TRUE(coordinator
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);
  fake.local_status = LocalPlanResult::Status::kNoPath;

  const CycleOutput invalidated = coordinator.PlanCycle(
      At(1.5, 1.5), SnapshotBundle{.fine = fine}, CycleTrigger::kDeviation,
      SearchDeadline::max());

  ASSERT_TRUE(invalidated.path_reference);
  EXPECT_EQ(invalidated.path_reference->state, PathState::kInvalidated);
  EXPECT_TRUE(invalidated.path_reference->path.poses.empty());
  EXPECT_FALSE(invalidated.terminal);
  EXPECT_EQ(fake.wheel_calls, 1);
  EXPECT_EQ(coordinator.state(), CoordinatorState::kReplanning);

  const CycleOutput failed = coordinator.PlanCycle(
      At(1.5, 1.5), SnapshotBundle{.fine = fine}, CycleTrigger::kContinue,
      SearchDeadline::max());
  ASSERT_TRUE(failed.terminal);
  EXPECT_EQ(failed.terminal->result.outcome, SessionOutcome::kNoPath);
  EXPECT_EQ(failed.terminal->result.reason_code, "NO_PATH");
  EXPECT_FALSE(failed.terminal->invalidated);
  EXPECT_EQ(fake.wheel_calls, 2);
  EXPECT_EQ(coordinator.state(), CoordinatorState::kIdle);

  const CycleOutput after_terminal = coordinator.PlanCycle(
      At(1.5, 1.5), SnapshotBundle{.fine = fine}, CycleTrigger::kContinue,
      SearchDeadline::max());
  EXPECT_FALSE(after_terminal.path_reference);
  EXPECT_FALSE(after_terminal.terminal);
  EXPECT_EQ(fake.wheel_calls, 2);
}

TEST(PlanningSessionCoordinator,
     NewBlockedFineRevisionInvalidatesRemainingPathBeforeReplanning) {
  FakePorts fake;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);
  const auto fine = MakeFine(1U);
  ASSERT_TRUE(coordinator
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);

  EXPECT_FALSE(coordinator.OnFineSnapshot(MakeFine(2U)));
  EXPECT_FALSE(coordinator
                   .PlanCycle(At(2.4, 0.5),
                              SnapshotBundle{.fine = MakeFine(2U)},
                              CycleTrigger::kContinue, SearchDeadline::max())
                   .terminal);
  EXPECT_FALSE(coordinator.OnFineSnapshot(
      MakeFine(3U, GridIndex{.x = 0, .y = 0})));
  const auto invalidated =
      coordinator.OnFineSnapshot(MakeFine(4U, GridIndex{.x = 2, .y = 0}));

  ASSERT_TRUE(invalidated);
  EXPECT_EQ(invalidated->state, PathState::kInvalidated);
  EXPECT_EQ(invalidated->segment_revision, 1U);
  EXPECT_EQ(invalidated->traversability_revision, 4U);
  EXPECT_TRUE(invalidated->path.poses.empty());
  EXPECT_EQ(coordinator.state(), CoordinatorState::kReplanning);
  EXPECT_FALSE(coordinator.OnFineSnapshot(
      MakeFine(5U, GridIndex{.x = 2, .y = 0})));
}

TEST(PlanningSessionCoordinator, StationaryNonFinalPathCannotBeReissuedForever) {
  FakePorts fake;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}}, FinalGoal{.target_x_m = 0.5, .target_y_m = 0.5}).accepted);
  const auto result = coordinator.PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = MakeFine()},
      CycleTrigger::kContinue, SearchDeadline::max());
  ASSERT_TRUE(result.terminal);
  EXPECT_EQ(result.terminal->result.outcome, SessionOutcome::kNoPath);
  EXPECT_EQ(result.terminal->result.reason_code, "NO_LOCAL_PROGRESS");
}

TEST(PlanningSessionCoordinator, CompletedNonFinalTargetMustAdvanceOnReplan) {
  FakePorts fake;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}}, FinalGoal{.target_x_m = 3.5, .target_y_m = 0.5}).accepted);
  const auto fine = MakeFine();
  ASSERT_TRUE(coordinator.PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
      CycleTrigger::kContinue, SearchDeadline::max()).path_reference);
  const auto result = coordinator.PlanCycle(At(3.49, 0.5), SnapshotBundle{.fine = fine},
      CycleTrigger::kSegmentEnd, SearchDeadline::max());
  ASSERT_TRUE(result.terminal);
  EXPECT_EQ(result.terminal->result.outcome, SessionOutcome::kNoPath);
  EXPECT_EQ(result.terminal->result.reason_code, "NO_LOCAL_PROGRESS");
}

TEST(PlanningSessionCoordinator, ExecutionConfirmationGatesGoalSuccess) {
  FakePorts fake;
  fake.reaches_final_goal = true;
  auto config = Config();
  config.require_execution_confirmation = true;
  PlanningSessionCoordinator coordinator(Wheel(), {}, config, fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}}, FinalGoal{.target_x_m = 3.5, .target_y_m = 0.5}).accepted);
  const auto fine = MakeFine();
  ASSERT_TRUE(coordinator.PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
      CycleTrigger::kContinue, SearchDeadline::max()).path_reference);
  EXPECT_FALSE(coordinator.PlanCycle(At(3.5, 0.5), SnapshotBundle{.fine = fine},
      CycleTrigger::kContinue, SearchDeadline::max()).terminal);
  const auto result = coordinator.PlanCycle(At(3.5, 0.5), SnapshotBundle{.fine = fine},
      CycleTrigger::kSegmentEnd, SearchDeadline::max());
  ASSERT_TRUE(result.terminal);
  EXPECT_EQ(result.terminal->result.outcome, SessionOutcome::kGoalReached);
}

TEST(PlanningSessionCoordinator,
     GoalRequiresDerivedPositionAndOptionalYawThenInvalidatesBeforeSuccess) {
  FakePorts fake;
  fake.reaches_final_goal = true;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5,
                                          .has_target_yaw = true,
                                          .target_yaw_rad = 0.7})
                  .accepted);
  const auto fine = MakeFine();
  ASSERT_TRUE(coordinator
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);

  const CycleOutput wrong_yaw = coordinator.PlanCycle(
      At(3.5, 0.5, 0.9), SnapshotBundle{.fine = fine},
      CycleTrigger::kContinue, SearchDeadline::max());
  EXPECT_FALSE(wrong_yaw.terminal);
  EXPECT_EQ(fake.wheel_calls, 1);

  const CycleOutput reached = coordinator.PlanCycle(
      At(3.5, 0.5, 0.75), SnapshotBundle{.fine = fine},
      CycleTrigger::kContinue, SearchDeadline::max());
  ASSERT_TRUE(reached.terminal);
  EXPECT_EQ(reached.terminal->result.outcome, SessionOutcome::kGoalReached);
  EXPECT_EQ(reached.terminal->result.reason_code, "GOAL_REACHED");
  ASSERT_TRUE(reached.terminal->invalidated);
  EXPECT_EQ(reached.terminal->invalidated->state, PathState::kInvalidated);
  EXPECT_EQ(reached.terminal->invalidated->segment_revision, 1U);
  EXPECT_EQ(coordinator.state(), CoordinatorState::kIdle);
}

TEST(PlanningSessionCoordinator,
     CancelAndPreemptInvalidateTheActiveSegmentBeforeTerminal) {
  const auto fine = MakeFine();
  FakePorts fake;
  PlanningSessionCoordinator canceled(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(canceled.Start(SessionId{{1U}},
                            FinalGoal{.target_x_m = 3.5, .target_y_m = 0.5})
                  .accepted);
  ASSERT_TRUE(canceled
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);
  const SessionTerminal canceled_terminal =
      canceled.Cancel(CancelReason::kCanceled);
  EXPECT_EQ(canceled_terminal.result.outcome, SessionOutcome::kCanceled);
  EXPECT_EQ(canceled_terminal.result.reason_code, "CANCELED");
  ASSERT_TRUE(canceled_terminal.invalidated);
  EXPECT_EQ(canceled_terminal.invalidated->state, PathState::kInvalidated);
  EXPECT_EQ(canceled.state(), CoordinatorState::kIdle);

  FakePorts preempt_fake;
  PlanningSessionCoordinator preempted(Wheel(), {}, Config(),
                                       preempt_fake.Bind());
  ASSERT_TRUE(preempted.Start(SessionId{{1U}},
                             FinalGoal{.target_x_m = 3.5,
                                       .target_y_m = 0.5})
                  .accepted);
  ASSERT_TRUE(preempted
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);
  const StartSessionResult replacement = preempted.Start(
      SessionId{{2U}}, FinalGoal{.target_x_m = 6.5, .target_y_m = 0.5});
  EXPECT_TRUE(replacement.accepted);
  ASSERT_TRUE(replacement.replaced_session);
  EXPECT_EQ(replacement.replaced_session->result.outcome,
            SessionOutcome::kCanceled);
  EXPECT_EQ(replacement.replaced_session->result.reason_code, "PREEMPTED");
  ASSERT_TRUE(replacement.replaced_session->invalidated);
  EXPECT_EQ(replacement.replaced_session->invalidated->session_id,
            SessionId{{1U}});
  EXPECT_EQ(preempted.state(), CoordinatorState::kPlanning);
  EXPECT_EQ(preempted.active_segment_revision(), 0U);
}

TEST(PlanningSessionCoordinator,
     MissingFineAndExpiredDeadlineTerminateWithoutWaitingOrSearching) {
  FakePorts fake;
  auto now = SteadyClock::time_point(std::chrono::seconds(10));
  auto config = Config();
  config.now = [&now] { return now; };
  PlanningSessionCoordinator missing(Wheel(), {}, config, fake.Bind());
  ASSERT_TRUE(missing.Start(SessionId{{1U}},
                            FinalGoal{.target_x_m = 3.5, .target_y_m = 0.5})
                  .accepted);
  const CycleOutput no_map = missing.PlanCycle(
      At(0.5, 0.5), SnapshotBundle{}, CycleTrigger::kContinue,
      SearchDeadline::max());
  ASSERT_TRUE(no_map.terminal);
  EXPECT_EQ(no_map.terminal->result.outcome,
            SessionOutcome::kMapUnavailable);
  EXPECT_EQ(no_map.terminal->result.reason_code, "MAP_UNAVAILABLE");

  FakePorts timeout_fake;
  PlanningSessionCoordinator timeout(Wheel(), {}, config,
                                     timeout_fake.Bind());
  ASSERT_TRUE(timeout.Start(SessionId{{2U}},
                            FinalGoal{.target_x_m = 3.5, .target_y_m = 0.5})
                  .accepted);
  const CycleOutput expired = timeout.PlanCycle(
      At(0.5, 0.5), SnapshotBundle{.fine = MakeFine()},
      CycleTrigger::kContinue, now);
  ASSERT_TRUE(expired.terminal);
  EXPECT_EQ(expired.terminal->result.outcome, SessionOutcome::kTimeout);
  EXPECT_EQ(expired.terminal->result.reason_code, "TIMEOUT");
  EXPECT_EQ(timeout_fake.global_calls, 0);
  EXPECT_EQ(timeout_fake.target_calls, 0);
  EXPECT_EQ(timeout_fake.patch_calls, 0);
  EXPECT_EQ(timeout_fake.wheel_calls, 0);
}

TEST(PlanningSessionCoordinator,
     DeadlineReachedBeforeReferenceCommitReturnsTimeoutWithoutAPath) {
  const auto started = SteadyClock::time_point(std::chrono::seconds(10));
  const auto deadline = started + std::chrono::seconds(3);
  std::size_t clock_reads = 0U;
  auto config = Config();
  config.now = [&] {
    ++clock_reads;
    return clock_reads < 3U ? started : deadline;
  };
  FakePorts fake;
  PlanningSessionCoordinator coordinator(Wheel(), {}, config, fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);

  const CycleOutput output = coordinator.PlanCycle(
      At(0.5, 0.5), SnapshotBundle{.fine = MakeFine()},
      CycleTrigger::kContinue, deadline);

  ASSERT_TRUE(output.terminal);
  EXPECT_EQ(output.terminal->result.outcome, SessionOutcome::kTimeout);
  EXPECT_EQ(output.terminal->result.reason_code, "TIMEOUT");
  EXPECT_EQ(output.terminal->result.last_segment_revision, 0U);
  EXPECT_FALSE(output.path_reference);
  EXPECT_EQ(coordinator.active_segment_revision(), 0U);
  EXPECT_EQ(coordinator.state(), CoordinatorState::kIdle);
}

TEST(PlanningSessionCoordinator, FrozenPlatformSelectsExactlyOneLocalPlanner) {
  const auto fine = MakeFine();
  FakePorts wheel_fake;
  PlanningSessionCoordinator wheel(Wheel(), {}, Config(), wheel_fake.Bind());
  ASSERT_TRUE(wheel.Start(SessionId{{1U}},
                          FinalGoal{.target_x_m = 3.5, .target_y_m = 0.5})
                  .accepted);
  ASSERT_TRUE(wheel
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);
  EXPECT_EQ(wheel_fake.wheel_calls, 1);
  EXPECT_EQ(wheel_fake.legged_calls, 0);

  FakePorts legged_fake;
  PlanningSessionCoordinator legged(Legged(), {}, Config(),
                                    legged_fake.Bind());
  ASSERT_TRUE(legged.Start(SessionId{{2U}},
                           FinalGoal{.target_x_m = 3.5, .target_y_m = 0.5})
                  .accepted);
  ASSERT_TRUE(legged
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);
  EXPECT_EQ(legged_fake.wheel_calls, 0);
  EXPECT_EQ(legged_fake.legged_calls, 1);
}

TEST(PlanningSessionCoordinator,
     ComponentErrorInvalidatesAnActiveSegmentBeforeInternalErrorTerminal) {
  FakePorts fake;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);
  const auto fine = MakeFine();
  ASSERT_TRUE(coordinator
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);
  fake.throw_from_target = true;

  EXPECT_NO_THROW({
    const CycleOutput output = coordinator.PlanCycle(
        At(1.0, 0.5), SnapshotBundle{.fine = fine},
        CycleTrigger::kSegmentEnd, SearchDeadline::max());
    ASSERT_TRUE(output.terminal);
    EXPECT_EQ(output.terminal->result.outcome,
              SessionOutcome::kInternalError);
    EXPECT_EQ(output.terminal->result.reason_code, "INTERNAL_ERROR");
    ASSERT_TRUE(output.terminal->invalidated);
    EXPECT_EQ(output.terminal->invalidated->state,
              PathState::kInvalidated);
  });
}

TEST(PlanningSessionCoordinator,
     ThreadSafeCycleStopLetsSerialCancelOwnInvalidationAndTerminal) {
  FakePorts fake;
  PlanningSessionPorts ports = fake.Bind();
  std::mutex mutex;
  std::condition_variable condition;
  bool global_entered = false;
  bool global_observed_stop = false;
  ports.plan_global =
      [&](const GlobalGuidanceSnapshot&, Point2, Point2, SearchDeadline,
          const StopToken& stop) {
        std::unique_lock lock(mutex);
        global_entered = true;
        condition.notify_all();
        std::stop_callback notify(stop, [&] { condition.notify_all(); });
        condition.wait(lock, [&] { return stop.stop_requested(); });
        global_observed_stop = true;
        return GlobalRouteResult{.status = GuidanceStatus::kTimeout};
      };
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(),
                                         std::move(ports));
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);
  const auto fine = MakeFine();
  ASSERT_TRUE(coordinator
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);
  const auto guidance = MakeGuidance(*fine);
  CycleOutput interrupted;
  std::thread worker([&] {
    interrupted = coordinator.PlanCycle(
        At(1.0, 0.5), SnapshotBundle{.fine = fine, .guidance = guidance},
        CycleTrigger::kSegmentEnd, SearchDeadline::max());
  });
  {
    std::unique_lock lock(mutex);
    condition.wait(lock, [&] { return global_entered; });
  }

  EXPECT_TRUE(coordinator.RequestStopCurrentCycle());
  worker.join();

  EXPECT_TRUE(global_observed_stop);
  EXPECT_FALSE(interrupted.path_reference);
  EXPECT_FALSE(interrupted.terminal);
  const SessionTerminal terminal = coordinator.Cancel(CancelReason::kCanceled);
  EXPECT_EQ(terminal.result.reason_code, "CANCELED");
  ASSERT_TRUE(terminal.invalidated);
  EXPECT_EQ(terminal.invalidated->segment_revision, 1U);
  EXPECT_FALSE(coordinator.RequestStopCurrentCycle());
}

TEST(PlanningSessionCoordinator,
     ThreadSafeCycleStopLetsSerialStartOwnPreemptionAndNewSession) {
  FakePorts fake;
  PlanningSessionPorts ports = fake.Bind();
  auto immediate_plan = ports.plan_wheel;
  std::mutex mutex;
  std::condition_variable condition;
  int local_calls = 0;
  bool local_entered = false;
  bool local_observed_stop = false;
  ports.plan_wheel =
      [&](const RequestLocalPlanningView& view, const Pose2& start,
          const LocalTarget& target, SearchDeadline deadline,
          const StopToken& stop) {
        if (++local_calls == 1) {
          return immediate_plan(view, start, target, deadline, stop);
        }
        std::unique_lock lock(mutex);
        local_entered = true;
        condition.notify_all();
        std::stop_callback notify(stop, [&] { condition.notify_all(); });
        condition.wait(lock, [&] { return stop.stop_requested(); });
        local_observed_stop = true;
        return LocalPlanResult{.status = LocalPlanResult::Status::kCanceled};
      };
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(),
                                         std::move(ports));
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);
  const auto fine = MakeFine();
  ASSERT_TRUE(coordinator
                  .PlanCycle(At(0.5, 0.5), SnapshotBundle{.fine = fine},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);
  CycleOutput interrupted;
  std::thread worker([&] {
    interrupted = coordinator.PlanCycle(
        At(1.0, 0.5), SnapshotBundle{.fine = fine},
        CycleTrigger::kSegmentEnd, SearchDeadline::max());
  });
  {
    std::unique_lock lock(mutex);
    condition.wait(lock, [&] { return local_entered; });
  }

  EXPECT_TRUE(coordinator.RequestStopCurrentCycle());
  worker.join();

  EXPECT_TRUE(local_observed_stop);
  EXPECT_FALSE(interrupted.path_reference);
  EXPECT_FALSE(interrupted.terminal);
  const StartSessionResult replacement = coordinator.Start(
      SessionId{{2U}}, FinalGoal{.target_x_m = 6.5, .target_y_m = 0.5});
  ASSERT_TRUE(replacement.replaced_session);
  EXPECT_EQ(replacement.replaced_session->result.reason_code, "PREEMPTED");
  ASSERT_TRUE(replacement.replaced_session->invalidated);
  EXPECT_EQ(replacement.replaced_session->invalidated->session_id,
            SessionId{{1U}});
  EXPECT_EQ(coordinator.state(), CoordinatorState::kPlanning);
}

TEST(PlanningSessionCoordinator,
     RemainingPathSupercoverChecksBothSidesOfCornerAndGridLine) {
  const auto run_case = [](const StateInput start, const FinalGoal goal,
                           const GridIndex blocked) {
    FakePorts fake;
    PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
    EXPECT_TRUE(coordinator.Start(SessionId{{1U}}, goal).accepted);
    EXPECT_TRUE(coordinator
                    .PlanCycle(start, SnapshotBundle{.fine = MakeFine(1U)},
                               CycleTrigger::kContinue,
                               SearchDeadline::max())
                    .path_reference);
    return coordinator.OnFineSnapshot(MakeFine(2U, blocked));
  };

  const auto corner = run_case(
      At(0.5, 0.5), FinalGoal{.target_x_m = 2.5, .target_y_m = 2.5},
      GridIndex{.x = 1, .y = 0});
  ASSERT_TRUE(corner);
  EXPECT_EQ(corner->state, PathState::kInvalidated);

  const auto along_boundary = run_case(
      At(0.5, 1.0), FinalGoal{.target_x_m = 3.5, .target_y_m = 1.0},
      GridIndex{.x = 2, .y = 0});
  ASSERT_TRUE(along_boundary);
  EXPECT_EQ(along_boundary->state, PathState::kInvalidated);
}

TEST(PlanningSessionCoordinator,
     CancelUsesLastFullyEvaluatedFineRevisionNotMerelyCapturedRevision) {
  FakePorts fake;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{1U}},
                                FinalGoal{.target_x_m = 3.5,
                                          .target_y_m = 0.5})
                  .accepted);
  ASSERT_TRUE(coordinator
                  .PlanCycle(At(0.5, 0.5),
                             SnapshotBundle{.fine = MakeFine(1U)},
                             CycleTrigger::kContinue, SearchDeadline::max())
                  .path_reference);
  EXPECT_FALSE(coordinator.OnFineSnapshot(MakeFine(2U)));
  EXPECT_FALSE(coordinator
                   .PlanCycle(At(1.0, 0.5),
                              SnapshotBundle{.fine = MakeFine(3U)},
                              CycleTrigger::kContinue, SearchDeadline::max())
                   .terminal);

  const SessionTerminal terminal = coordinator.Cancel(CancelReason::kCanceled);

  ASSERT_TRUE(terminal.invalidated);
  EXPECT_EQ(terminal.invalidated->traversability_revision, 2U);
}

TEST(PlanningSessionCoordinator, NoLocalProgressWaitsWithoutRepeatingSearchUntilMapChanges) {
  FakePorts fake;
  fake.local_status = LocalPlanResult::Status::kNoPath;
  fake.local_reason_code = "LOCAL_WAITING_FOR_MAP";
  auto region = std::make_shared<LocalGoalRegion>();
  region->has_unknown_boundary = true;
  fake.target_region = region;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{90U}}, {.target_x_m=6.5, .target_y_m=1.5}).accepted);
  const auto fine = MakeFine(1U);
  auto wait = coordinator.PlanCycle(At(1.5,1.5), {.fine=fine}, CycleTrigger::kContinue, SearchDeadline::max());
  ASSERT_FALSE(wait.terminal);
  EXPECT_FALSE(wait.path_reference);
  EXPECT_EQ(wait.feedback.reason_code, "WAITING_FOR_MAP");
  for (int i=0;i<10;++i) {
    wait=coordinator.PlanCycle(At(1.5,1.5), {.fine=fine}, CycleTrigger::kSegmentEnd, SearchDeadline::max());
    EXPECT_FALSE(wait.terminal);
    EXPECT_FALSE(wait.path_reference);
    EXPECT_EQ(wait.feedback.reason_code,"WAITING_FOR_MAP");
  }
  EXPECT_EQ(fake.wheel_calls,1);
  EXPECT_EQ(fake.global_calls,0);
  fake.local_status=LocalPlanResult::Status::kPlanFound;
  fake.local_reason_code.clear();
  fake.reaches_final_goal=true;
  const auto updated=MakeFine(2U);
  const auto planned=coordinator.PlanCycle(At(1.5,1.5),{.fine=updated},CycleTrigger::kContinue,SearchDeadline::max());
  ASSERT_TRUE(planned.path_reference);
  EXPECT_EQ(fake.wheel_calls,2);
  const auto reached=coordinator.PlanCycle(At(6.5,1.5),{.fine=updated},CycleTrigger::kContinue,SearchDeadline::max());
  ASSERT_TRUE(reached.terminal);
  EXPECT_EQ(reached.terminal->result.outcome,SessionOutcome::kGoalReached);
}

TEST(PlanningSessionCoordinator, WaitingSessionRemainsCancelableAndNewGoalPlansAgain) {
  FakePorts fake;
  fake.local_status=LocalPlanResult::Status::kNoPath;
  fake.local_reason_code="LOCAL_WAITING_FOR_MAP";
  auto region = std::make_shared<LocalGoalRegion>();
  region->has_unknown_boundary = true;
  fake.target_region = region;
  PlanningSessionCoordinator coordinator(Wheel(),{},Config(),fake.Bind());
  const auto fine=MakeFine();
  ASSERT_TRUE(coordinator.Start(SessionId{{91U}},{.target_x_m=6.5,.target_y_m=1.5}).accepted);
  ASSERT_FALSE(coordinator.PlanCycle(At(1.5,1.5),{.fine=fine},CycleTrigger::kContinue,SearchDeadline::max()).terminal);
  EXPECT_EQ(coordinator.Cancel(CancelReason::kCanceled).result.outcome,SessionOutcome::kCanceled);
  ASSERT_TRUE(coordinator.Start(SessionId{{92U}},{.target_x_m=5.5,.target_y_m=1.5}).accepted);
  static_cast<void>(coordinator.PlanCycle(At(1.5,1.5),{.fine=fine},CycleTrigger::kContinue,SearchDeadline::max()));
  EXPECT_EQ(fake.wheel_calls,2);
}

TEST(PlanningSessionCoordinator, FinalArrivalUsesSameNumericalToleranceAsLocalSolver) {
  FakePorts fake;
  fake.reaches_final_goal=true;
  PlanningSessionCoordinator coordinator(Wheel(),{},Config(),fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{93U}},{.target_x_m=5.5,.target_y_m=1.5}).accepted);
  const auto fine=MakeFine();
  ASSERT_TRUE(coordinator.PlanCycle(At(1.5,1.5),{.fine=fine},CycleTrigger::kContinue,SearchDeadline::max()).path_reference);
  // The configured physical 0.2 m region applies even on this one-metre lattice.
  const auto reached=coordinator.PlanCycle(At(std::nextafter(5.7,7.0),1.5),{.fine=fine},CycleTrigger::kContinue,SearchDeadline::max());
  ASSERT_TRUE(reached.terminal);
  EXPECT_EQ(reached.terminal->result.outcome,SessionOutcome::kGoalReached);
}

TEST(PlanningSessionCoordinator, NoProgressWithoutUnknownEvidenceTerminatesNoPath) {
  for (const bool region_present : {false, true}) {
    FakePorts fake;
    fake.local_status = LocalPlanResult::Status::kNoPath;
    fake.local_reason_code = "LOCAL_NO_PROGRESS";
    if (region_present) {
      auto region = std::make_shared<LocalGoalRegion>();
      // A window-level hint alone is insufficient: the solver did not reach
      // a forward unknown boundary and returned LOCAL_NO_PROGRESS.
      region->has_unknown_boundary = true;
      fake.target_region = region;
    }
    PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
    ASSERT_TRUE(coordinator.Start(SessionId{{94U}},
        {.target_x_m=6.5, .target_y_m=1.5}).accepted);
    const auto result = coordinator.PlanCycle(At(1.5,1.5), {.fine=MakeFine()},
        CycleTrigger::kContinue, SearchDeadline::max());
    ASSERT_TRUE(result.terminal);
    EXPECT_EQ(result.terminal->result.outcome, SessionOutcome::kNoPath);
    EXPECT_EQ(result.feedback.reason_code, "LOCAL_NO_PROGRESS");
  }
}

TEST(PlanningSessionCoordinator, WaitingWakesOnMovementSmallerThanArrivalTolerance) {
  FakePorts fake;
  fake.local_status = LocalPlanResult::Status::kNoPath;
  fake.local_reason_code = "LOCAL_WAITING_FOR_MAP";
  auto region = std::make_shared<LocalGoalRegion>();
  region->has_unknown_boundary = true;
  fake.target_region = region;
  PlanningSessionCoordinator coordinator(Wheel(), {}, Config(), fake.Bind());
  ASSERT_TRUE(coordinator.Start(SessionId{{95U}},
      {.target_x_m=6.5, .target_y_m=1.5}).accepted);
  const auto fine = MakeFine();
  EXPECT_EQ(coordinator.PlanCycle(At(1.5,1.5), {.fine=fine},
      CycleTrigger::kContinue, SearchDeadline::max()).feedback.reason_code,
      "WAITING_FOR_MAP");
  fake.local_status = LocalPlanResult::Status::kPlanFound;
  fake.local_reason_code.clear();
  // A 0.1 m movement changes which routes can connect, even though it is
  // smaller than the fixture's configured 0.2 m final-arrival region.
  const auto moved = coordinator.PlanCycle(At(1.6,1.5), {.fine=fine},
      CycleTrigger::kContinue, SearchDeadline::max());
  EXPECT_EQ(fake.wheel_calls, 2);
  ASSERT_TRUE(moved.path_reference);
}

TEST(PlanningSessionCoordinator, OnePoseReferenceKeepsUnknownButInvalidatesNewBlocker) {
  FakePorts fake;
  auto ports=fake.Bind();
  ports.plan_wheel=[](const RequestLocalPlanningView&,const Pose2& start,const LocalTarget&,
                       SearchDeadline,const StopToken&) {
    const PathPoint point{.pose={.position_m={start.position_m.x,start.position_m.y,0.0},
        .orientation=YawQuaternion(1.57)}};
    return LocalPlanResult{.status=LocalPlanResult::Status::kPlanFound,.raw_path={point},.path={point}};
  };
  PlanningSessionCoordinator coordinator(Wheel(),{},Config(),std::move(ports));
  ASSERT_TRUE(coordinator.Start(SessionId{{96U}},{.target_x_m=3.5,.target_y_m=.5}).accepted);
  ASSERT_TRUE(coordinator.PlanCycle(At(.5,.5),{.fine=MakeFine()},CycleTrigger::kContinue,
      SearchDeadline::max()).path_reference);
  EXPECT_FALSE(coordinator.OnFineSnapshot(MakeFine(2U,GridIndex{0,0},FineCellState::kUnknown)));
  EXPECT_TRUE(coordinator.OnFineSnapshot(MakeFine(3U,GridIndex{0,0},FineCellState::kBlocked)));
}

}  // namespace
}  // namespace lunar::incremental_navigation
