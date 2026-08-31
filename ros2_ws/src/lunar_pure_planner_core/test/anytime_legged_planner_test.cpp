#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "legged/anytime_legged_planner.hpp"
#include "legged/legged_traversal_projection.hpp"
#include "shared/goal_distance_field.hpp"
#include "shared/local_terrain_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::legged {
namespace {

struct TerrainFixture final {
  std::shared_ptr<const shared::MapSnapshot> map;
  std::shared_ptr<const shared::LocalTerrainProjection> terrain;
};

[[nodiscard]] TerrainFixture MakeTerrain(
    const std::size_t width, const std::size_t height,
    const double resolution_m, std::vector<float> occupancy = {},
    std::vector<float> elevation = {}) {
  const std::size_t count = width * height;
  if (occupancy.empty()) {
    occupancy.assign(count, 0.0F);
  }
  if (elevation.empty()) {
    elevation.assign(count, 0.0F);
  }
  GridMap map{
      .frame_id = "odom",
      .stamp = TimePoint{},
      .width = width,
      .height = height,
      .resolution_m = resolution_m,
      .origin_m = {},
  };
  map.layers.emplace(
      "occupancy", GridLayer{.values = std::move(occupancy)});
  map.layers.emplace(
      "elevation", GridLayer{.values = std::move(elevation)});
  auto snapshot = shared::MapSnapshot::Create(
      std::move(map), shared::MapContract::kLocalElevation);
  EXPECT_TRUE(snapshot.ok()) << snapshot.reason_code;
  auto projection = shared::BuildLocalTerrainProjection(snapshot.snapshot);
  EXPECT_TRUE(projection.ok()) << projection.reason_code;
  return TerrainFixture{
      .map = std::move(snapshot.snapshot),
      .terrain = std::make_shared<const shared::LocalTerrainProjection>(
          std::move(*projection.value)),
  };
}

[[nodiscard]] LeggedCapability Capability() {
  return LeggedCapability{
      .body_extent_m = {0.6, 0.2, 0.3},
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
              .body_frame_displacement_m = {0.4, 0.0, 0.0}},
          LeggedBodyPrimitive{
              .primitive_id = "backward",
              .kind = LeggedPrimitiveKind::kBackward,
              .body_frame_displacement_m = {-0.4, 0.0, 0.0}},
          LeggedBodyPrimitive{
              .primitive_id = "left",
              .kind = LeggedPrimitiveKind::kLateralLeft,
              .body_frame_displacement_m = {0.0, 0.4, 0.0}},
          LeggedBodyPrimitive{
              .primitive_id = "right",
              .kind = LeggedPrimitiveKind::kLateralRight,
              .body_frame_displacement_m = {0.0, -0.4, 0.0}},
          LeggedBodyPrimitive{
              .primitive_id = "spin",
              .kind = LeggedPrimitiveKind::kSpin,
              .yaw_change_rad = std::numbers::pi / 4.0},
      },
  };
}

[[nodiscard]] LeggedPlanRequest RequestTo(
    const TerrainFixture& fixture, const LeggedCapability& capability,
    const Vec3 start, const Vec3 goal, const double goal_yaw = 0.0) {
  const auto traversal = BuildLeggedTraversalProjection(
      fixture.terrain, capability, {});
  EXPECT_TRUE(traversal.ok()) << traversal.reason_code;
  const GoalRegion goal_region{
    .goal_id = "goal",
    .target = PointGoal{.position_m = goal, .tolerance_m = 0.0},
    .yaw_rad = goal_yaw,
    .yaw_tolerance_rad = 0.0,
  };
  std::vector<std::uint8_t> feasible(fixture.map->cell_count(), 0U);
  for (std::size_t index = 0U; index < feasible.size(); ++index) {
    feasible[index] = static_cast<std::uint8_t>(
      traversal.value->hard_feasible[index] != 0U &&
      traversal.value->step_feasible[index] != 0U);
  }
  std::shared_ptr<const shared::GoalDistanceField> goal_field;
  if (const auto goal_cell = fixture.map->PositionToCell(
      {.x = goal.x, .y = goal.y});
    goal_cell.has_value())
  {
    const auto built = shared::BuildGoalDistanceField(
        *fixture.map, feasible,
        std::span<const shared::GridCell>{&*goal_cell, 1U});
    if (built.has_value()) {
      goal_field = std::make_shared<const shared::GoalDistanceField>(
          std::move(*built));
    }
  }
  if (goal_field == nullptr) {
    goal_field = std::make_shared<const shared::GoalDistanceField>(
        shared::GoalDistanceField{
          .distance_m = std::vector<double>(
                fixture.map->cell_count(),
                std::numeric_limits<double>::infinity()),
          .nearest_goal_index = std::vector<std::size_t>(
                fixture.map->cell_count(),
                std::numeric_limits<std::size_t>::max()),
        });
  }
  return LeggedPlanRequest{
      .start = LeggedState{
          .body_pose = Pose3{.position_m = start},
      },
      .goals_odom = LocalGoalSet{
          .goals_odom = {goal_region},
          .exact_final_goal = true,
      },
      .terrain = fixture.terrain.get(),
      .traversal = traversal.value,
      .goal_distance_field = std::move(goal_field),
      .capability = &capability,
  };
}

[[nodiscard]] LeggedCapability ProductionCapability() {
  LeggedCapability capability = Capability();
  capability.body_extent_m = {0.68, 0.33, 0.35};
  capability.nominal_body_height_m = 0.33;
  capability.body_height_m = {0.28, 0.38};
  capability.maximum_slope_rad = 0.5235987755982988;
  capability.maximum_step_height_m = 0.5;
  capability.maximum_gap_width_m = 0.3;
  capability.minimum_body_clearance_m = 0.3;
  capability.step_vertical_rate_mps = 0.1;
  capability.forward_speed_mps = {-1.5, 1.5};
  capability.lateral_speed_mps = {-0.8, 0.8};
  capability.yaw_rate_radps = {-1.0, 1.0};
  capability.motion_primitives = {
      {"forward", LeggedPrimitiveKind::kForward, {0.2, 0.0, 0.0}, 0.0},
      {"backward", LeggedPrimitiveKind::kBackward, {-0.2, 0.0, 0.0}, 0.0},
      {"lateral-left", LeggedPrimitiveKind::kLateralLeft,
       {0.0, 0.2, 0.0}, 0.0},
      {"lateral-right", LeggedPrimitiveKind::kLateralRight,
       {0.0, -0.2, 0.0}, 0.0},
      {"spin-left", LeggedPrimitiveKind::kSpin, {}, 0.09817477042468103},
      {"spin-right", LeggedPrimitiveKind::kSpin, {}, -0.09817477042468103},
  };
  return capability;
}

[[nodiscard]] std::vector<float> CorridorOccupancy(
    const std::size_t width, const std::size_t height,
    const std::size_t first_free_row, const std::size_t last_free_row) {
  std::vector<float> result(width * height, 1.0F);
  for (std::size_t y = first_free_row; y <= last_free_row; ++y) {
    std::fill_n(result.begin() + static_cast<std::ptrdiff_t>(y * width),
                width, 0.0F);
  }
  return result;
}

TEST(AnytimeLeggedPlanner, ReachesExactOffGridPoseAndYawWithPositiveCost) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives.push_back(LeggedBodyPrimitive{
      .primitive_id = "off-grid-coupled",
      .kind = LeggedPrimitiveKind::kCoupled,
      .body_frame_displacement_m = {0.27, -0.07, 0.0},
      .yaw_change_rad = 0.17,
  });
  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.37, 1.43, 0.0}, 0.17));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.trajectory.empty());
  const LeggedTransition& endpoint = result.trajectory.back();
  EXPECT_NEAR(endpoint.target_pose.position_m.x, 1.37, 1.0e-9);
  EXPECT_NEAR(endpoint.target_pose.position_m.y, 1.43, 1.0e-9);
  EXPECT_NEAR(endpoint.target_pose.yaw_rad, 0.17, 1.0e-9);
  EXPECT_TRUE(std::isfinite(result.cost));
  EXPECT_GT(result.cost, 0.0);
}

TEST(AnytimeLeggedPlanner, SelectsAReachableLaterPortalInOneSearch) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};
  LeggedPlanRequest request = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.1, 1.9, 0.0});
  request.goals_odom = LocalGoalSet{
    .goals_odom = {
      GoalRegion{
        .goal_id = "rank-0-lateral",
        .target = PointGoal{
          .position_m = {1.1, 1.9, 0.0}, .tolerance_m = 0.0},
      },
      GoalRegion{
        .goal_id = "rank-1-forward",
        .target = PointGoal{
          .position_m = {1.5, 1.5, 0.0}, .tolerance_m = 0.0},
      },
    },
    .exact_final_goal = false,
  };
  std::vector<std::uint8_t> feasible(fixture.map->cell_count(), 0U);
  for (std::size_t index = 0U; index < feasible.size(); ++index) {
    feasible[index] = static_cast<std::uint8_t>(
      request.traversal->hard_feasible[index] != 0U &&
      request.traversal->step_feasible[index] != 0U);
  }
  const std::vector<shared::GridCell> goal_cells{
    *fixture.map->PositionToCell({.x = 1.1, .y = 1.9}),
    *fixture.map->PositionToCell({.x = 1.5, .y = 1.5}),
  };
  const auto field = shared::BuildGoalDistanceField(
      *fixture.map, feasible, goal_cells);
  ASSERT_TRUE(field.has_value());
  request.goal_distance_field =
    std::make_shared<const shared::GoalDistanceField>(*field);

  const LeggedPlanResult result = PlanLegged(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_TRUE(result.selected_goal_index.has_value());
  EXPECT_EQ(*result.selected_goal_index, 1U);
  ASSERT_FALSE(result.trajectory.empty());
  EXPECT_NEAR(result.trajectory.back().target_pose.position_m.x, 1.5,
              1.0e-9);
  EXPECT_NEAR(result.trajectory.back().target_pose.position_m.y, 1.5,
              1.0e-9);
}

TEST(AnytimeLeggedPlanner, RejectsDisconnectedGoalsBeforeStateExpansion) {
  constexpr std::size_t kWidth = 20U;
  constexpr std::size_t kHeight = 15U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    occupancy[y * kWidth + 10U] = 1.0F;
  }
  const TerrainFixture fixture =
    MakeTerrain(kWidth, kHeight, 0.2, std::move(occupancy));
  const LeggedCapability capability = Capability();

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {0.7, 1.5, 0.5}, {3.1, 1.5, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath);
  EXPECT_EQ(result.reason_code, "LEGGED_NO_PATH");
  EXPECT_EQ(result.metrics.expanded_states, 0U);
}

TEST(AnytimeLeggedPlanner, RequestRelativeKeysPreserveRigidlyMovedSearch) {
  const TerrainFixture fixture = MakeTerrain(30U, 25U, 0.2);
  const LeggedCapability capability = Capability();
  LeggedPlanRequest baseline = RequestTo(
      fixture, capability, {1.11, 1.53, 0.5}, {2.31, 1.53, 0.0}, 0.0);
  LeggedPlanRequest shifted = RequestTo(
      fixture, capability, {1.18, 1.60, 0.5}, {2.38, 1.60, 0.0}, 0.0);
  constexpr double kRotatedYaw = 0.31;
  LeggedPlanRequest rotated = RequestTo(
      fixture, capability, {1.11, 1.53, 0.5},
    {1.11 + 1.2 * std::cos(kRotatedYaw),
      1.53 + 1.2 * std::sin(kRotatedYaw), 0.0},
      kRotatedYaw);
  baseline.start.body_pose.orientation = QuaternionFromYaw(0.0);
  shifted.start.body_pose.orientation = QuaternionFromYaw(0.0);
  rotated.start.body_pose.orientation = QuaternionFromYaw(kRotatedYaw);

  const LeggedPlanResult original = PlanLegged(baseline);
  const LeggedPlanResult translated = PlanLegged(shifted);
  const LeggedPlanResult turned = PlanLegged(rotated);

  ASSERT_TRUE(original.ok()) << original.reason_code;
  ASSERT_TRUE(translated.ok()) << translated.reason_code;
  ASSERT_TRUE(turned.ok()) << turned.reason_code;
  EXPECT_EQ(translated.trajectory.size(), original.trajectory.size());
  EXPECT_EQ(turned.trajectory.size(), original.trajectory.size());
  EXPECT_EQ(translated.quantized_state_count,
            original.quantized_state_count);
  EXPECT_EQ(turned.quantized_state_count, original.quantized_state_count);
  EXPECT_NEAR(translated.cost, original.cost, 1.0e-9);
  EXPECT_NEAR(turned.cost, original.cost, 1.0e-9);
}

TEST(AnytimeLeggedPlanner, StopsWhenControlTriggersOnlyDuringReconstruction) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives.push_back(LeggedBodyPrimitive{
      .primitive_id = "off-grid-coupled",
      .kind = LeggedPrimitiveKind::kCoupled,
      .body_frame_displacement_m = {0.27, -0.07, 0.0},
      .yaw_change_rad = 0.17,
  });
  std::size_t baseline_reads = 0U;
  LeggedPlanRequest baseline = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.37, 1.43, 0.0}, 0.17);
  baseline.control.now = [&] {
    ++baseline_reads;
    return SteadyClock::time_point{};
  };
  const LeggedPlanResult baseline_result = PlanLegged(baseline);
  ASSERT_TRUE(baseline_result.ok()) << baseline_result.reason_code;
  ASSERT_GT(baseline_reads, 2U);
  const std::size_t first_reconstruction_read = baseline_reads - 2U;
  const std::size_t final_reconstruction_read = baseline_reads - 1U;
  const std::size_t after_reconstruction_read = baseline_reads;

  std::stop_source stop;
  std::size_t cancel_reads = 0U;
  LeggedPlanRequest canceled = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.37, 1.43, 0.0}, 0.17);
  canceled.control.stop_token = stop.get_token();
  canceled.control.deadline = SteadyClock::time_point::max();
  canceled.control.now = [&] {
    if (cancel_reads++ == first_reconstruction_read) {
      stop.request_stop();
    }
    return SteadyClock::time_point{};
  };
  const LeggedPlanResult canceled_result = PlanLegged(canceled);
  EXPECT_EQ(canceled_result.status, LocalPlanStatus::kCanceled)
      << "clock_reads=" << cancel_reads;

  std::size_t timeout_reads = 0U;
  LeggedPlanRequest timed_out = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.37, 1.43, 0.0}, 0.17);
  timed_out.control.deadline = SteadyClock::time_point{
      std::chrono::milliseconds{1}};
  timed_out.control.now = [&] {
    return timeout_reads++ < final_reconstruction_read
               ? SteadyClock::time_point{}
               : SteadyClock::time_point{std::chrono::milliseconds{2}};
  };
  const LeggedPlanResult timeout_result = PlanLegged(timed_out);
  EXPECT_EQ(timeout_result.status, LocalPlanStatus::kTimedOut)
      << "clock_reads=" << timeout_reads;

  std::size_t completed_reads = 0U;
  LeggedPlanRequest completed = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.37, 1.43, 0.0}, 0.17);
  completed.control.deadline = SteadyClock::time_point{
      std::chrono::milliseconds{1}};
  completed.control.now = [&] {
    return completed_reads++ < after_reconstruction_read
               ? SteadyClock::time_point{}
               : SteadyClock::time_point{std::chrono::milliseconds{2}};
  };
  const LeggedPlanResult completed_result = PlanLegged(completed);
  ASSERT_TRUE(completed_result.ok())
      << completed_result.reason_code << " clock_reads=" << completed_reads;
  EXPECT_EQ(completed_result.trajectory.size(), 1U);
}

TEST(AnytimeLeggedPlanner, IgnoresGoalZAndDerivesTerminalZFromElevation) {
  constexpr double kElevationM = 0.25;
  const TerrainFixture fixture = MakeTerrain(
      20U, 15U, 0.2, {}, std::vector<float>(20U * 15U, kElevationM));
  LeggedCapability capability = Capability();
  capability.motion_primitives.push_back(LeggedBodyPrimitive{
      .primitive_id = "off-grid-coupled",
      .kind = LeggedPrimitiveKind::kCoupled,
      .body_frame_displacement_m = {0.27, -0.07, 0.0},
      .yaw_change_rad = 0.17,
  });
  LeggedPlanRequest request = RequestTo(
      fixture, capability, {1.1, 1.5, 0.75}, {1.37, 1.43, 0.0}, 0.17);
  auto& goal = std::get<PointGoal>(
      request.goals_odom.goals_odom.front().target);
  goal.position_m.z = std::numeric_limits<double>::quiet_NaN();

  const LeggedPlanResult result = PlanLegged(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.trajectory.empty());
  EXPECT_NEAR(result.trajectory.back().target_pose.position_m.z,
              kElevationM + 0.5, 1.0e-9);
}

TEST(AnytimeLeggedPlanner, DoesNotCoupleIndependentConnectorPrimitives) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {
      capability.motion_primitives.at(0U),
      LeggedBodyPrimitive{
          .primitive_id = "spin-small",
          .kind = LeggedPrimitiveKind::kSpin,
          .yaw_change_rad = 0.4,
      },
  };

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.5, 1.5, 0.0}, 0.2));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_EQ(result.trajectory.size(), 2U);
  for (const LeggedTransition& transition : result.trajectory) {
    const double translation = std::hypot(
        transition.target_pose.position_m.x -
            transition.source_pose.position_m.x,
        transition.target_pose.position_m.y -
            transition.source_pose.position_m.y);
    const double yaw = std::abs(ShortestYawDelta(
        transition.source_pose.yaw_rad, transition.target_pose.yaw_rad));
    EXPECT_TRUE(translation <= 1.0e-9 || yaw <= 1.0e-9);
  }
}

TEST(AnytimeLeggedPlanner, PreservesCurrentYawWhenGoalYawIsUnconstrained) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};
  constexpr double kStartYaw = 0.31;
  LeggedPlanRequest request = RequestTo(
      fixture, capability, {1.1, 1.1, 0.5},
      {1.1 + 0.25 * std::cos(kStartYaw),
       1.1 + 0.25 * std::sin(kStartYaw), 0.0});
  request.start.body_pose.orientation = QuaternionFromYaw(kStartYaw);
  request.goals_odom.goals_odom.front().yaw_rad.reset();

  const LeggedPlanResult result = PlanLegged(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.trajectory.empty());
  EXPECT_NEAR(result.trajectory.back().target_pose.yaw_rad,
              kStartYaw, 1.0e-9);
}

TEST(AnytimeLeggedPlanner, AcceptsPrimitiveEndpointInsideXyGoalTolerance) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};
  LeggedPlanRequest request = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.65, 1.5, 0.0});
  std::get<PointGoal>(request.goals_odom.goals_odom.front().target)
  .tolerance_m = 0.16;

  const LeggedPlanResult result = PlanLegged(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.trajectory.empty());
  EXPECT_NEAR(result.trajectory.back().target_pose.position_m.x,
              1.5, 1.0e-9);
}

TEST(AnytimeLeggedPlanner, AcceptsPrimitiveYawInsideGoalYawTolerance) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {LeggedBodyPrimitive{
      .primitive_id = "spin-tolerant",
      .kind = LeggedPrimitiveKind::kSpin,
      .yaw_change_rad = 0.4,
  }};
  LeggedPlanRequest request = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.1, 1.5, 0.0}, 0.5);
  request.goals_odom.goals_odom.front().yaw_tolerance_rad = 0.11;

  const LeggedPlanResult result = PlanLegged(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.trajectory.empty());
  EXPECT_NEAR(result.trajectory.back().target_pose.yaw_rad, 0.4, 1.0e-9);
}

TEST(AnytimeLeggedPlanner, UnconstrainedYawAcceptsTurningPrimitiveEndpoint) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {LeggedBodyPrimitive{
      .primitive_id = "turning-step",
      .kind = LeggedPrimitiveKind::kCoupled,
      .body_frame_displacement_m = {0.4, 0.0, 0.0},
      .yaw_change_rad = 0.3,
  }};
  LeggedPlanRequest request = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.5, 1.5, 0.0});
  request.goals_odom.goals_odom.front().yaw_rad.reset();

  const LeggedPlanResult result = PlanLegged(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.trajectory.empty());
  EXPECT_NEAR(result.trajectory.back().target_pose.yaw_rad, 0.3, 1.0e-9);
}

TEST(AnytimeLeggedPlanner, UnconstrainedYawKeepsShortenedPrimitiveYaw) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {LeggedBodyPrimitive{
      .primitive_id = "short-turning-step",
      .kind = LeggedPrimitiveKind::kCoupled,
      .body_frame_displacement_m = {0.4, 0.0, 0.0},
      .yaw_change_rad = 0.4,
  }};
  LeggedPlanRequest request = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.3, 1.5, 0.0});
  request.goals_odom.goals_odom.front().yaw_rad.reset();

  const LeggedPlanResult result = PlanLegged(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_EQ(result.trajectory.size(), 1U);
  EXPECT_NEAR(result.trajectory.back().target_pose.position_m.x,
              1.3, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().target_pose.yaw_rad, 0.2, 1.0e-9);
}

TEST(AnytimeLeggedPlanner,
     DoesNotReturnNearIncumbentWithoutTimeToReconstructLargeMapPlan) {
  const TerrainFixture fixture = MakeTerrain(320U, 320U, 0.2);
  const LeggedCapability capability = Capability();
  LeggedPlanRequest baseline = RequestTo(
      fixture, capability, {10.1, 10.1, 0.5}, {10.35, 10.1, 0.0});
  const auto baseline_reads = std::make_shared<std::int64_t>(0);
  baseline.control.now = [baseline_reads] {
    ++*baseline_reads;
    return SteadyClock::time_point{};
  };
  const LeggedPlanResult baseline_result = PlanLegged(baseline);
  ASSERT_TRUE(baseline_result.ok()) << baseline_result.reason_code;
  ASSERT_GT(*baseline_reads, 1);
  const auto final_reconstruction_clock_read =
      std::chrono::nanoseconds{*baseline_reads - 1};

  LeggedPlanRequest request = RequestTo(
      fixture, capability, {10.1, 10.1, 0.5}, {10.35, 10.1, 0.0});
  const auto clock_reads = std::make_shared<std::int64_t>(0);
  request.control.deadline =
      SteadyClock::time_point{final_reconstruction_clock_read};
  request.control.now = [clock_reads] {
    return SteadyClock::time_point{
        std::chrono::nanoseconds{(*clock_reads)++}};
  };

  const LeggedPlanResult result = PlanLegged(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kTimedOut)
      << result.reason_code << " clock_reads=" << *clock_reads;
  EXPECT_EQ(result.reason_code, "TIMEOUT");
  EXPECT_TRUE(result.trajectory.empty());
  EXPECT_GT(result.metrics.expanded_states, 0U);
  EXPECT_LT(*clock_reads, 40000);

  LeggedPlanRequest completed = RequestTo(
      fixture, capability, {10.1, 10.1, 0.5}, {10.35, 10.1, 0.0});
  const auto completed_reads = std::make_shared<std::int64_t>(0);
  completed.control.deadline = SteadyClock::time_point{
      final_reconstruction_clock_read + std::chrono::nanoseconds{1}};
  completed.control.now = [completed_reads] {
    return SteadyClock::time_point{
        std::chrono::nanoseconds{(*completed_reads)++}};
  };
  const LeggedPlanResult completed_result = PlanLegged(completed);
  ASSERT_TRUE(completed_result.ok())
      << completed_result.reason_code
      << " clock_reads=" << *completed_reads;
  EXPECT_FALSE(completed_result.trajectory.empty());
}

TEST(AnytimeLeggedPlanner, PlansFarAcrossLargeMapWithinOneSecond) {
  const TerrainFixture fixture = MakeTerrain(320U, 320U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {
      capability.motion_primitives.at(0U),
      capability.motion_primitives.at(2U),
  };
  const auto started = std::chrono::steady_clock::now();

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {10.1, 10.1, 0.5}, {30.1, 30.1, 0.0}));

  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.metrics.expanded_states, 100U);
#if !defined(__SANITIZE_ADDRESS__)
  EXPECT_LT(elapsed, std::chrono::seconds{1});
#else
  (void)elapsed;
#endif
}

TEST(AnytimeLeggedPlanner, OpenTerrainUsesOnlyTheFastBodySweepPath) {
  const TerrainFixture fixture = MakeTerrain(24U, 20U, 0.2);
  const LeggedCapability capability = Capability();

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {2.7, 1.5, 0.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.fast_path_accepts, 0U);
  EXPECT_EQ(result.exact_sweep_fallbacks, 0U);
  EXPECT_EQ(result.exact_sweep_cell_checks, 0U);
}

TEST(AnytimeLeggedPlanner,
     HazardInsideConservativeAabbFallsBackToOrientedSweep) {
  constexpr std::size_t kWidth = 24U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[9U * kWidth + 7U] = 1.0F;
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, 0.2, std::move(occupancy));
  const LeggedCapability capability = Capability();

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.5, 1.5, 0.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.exact_sweep_fallbacks, 0U);
  EXPECT_GT(result.exact_sweep_cell_checks, 0U);
}

TEST(AnytimeLeggedPlanner,
     ProductionOpenMapWithNonzeroYawPlansThreeMetersWithinOneSecond) {
  const TerrainFixture fixture = MakeTerrain(320U, 320U, 0.2);
  const LeggedCapability capability = ProductionCapability();
  constexpr double kYaw = 1.03242;
  LeggedPlanRequest request = RequestTo(
      fixture, capability, {10.1, 10.1, 0.33},
      {10.1 + 3.0 * std::cos(kYaw), 10.1 + 3.0 * std::sin(kYaw), 0.0},
      kYaw);
  request.start.body_pose.orientation = QuaternionFromYaw(kYaw);
  const auto started = std::chrono::steady_clock::now();
  request.control.deadline = started + std::chrono::seconds{3};

  const LeggedPlanResult result = PlanLegged(request);

  const auto elapsed = std::chrono::steady_clock::now() - started;
  RecordProperty(
      "local_search_elapsed_ms",
      std::to_string(
          std::chrono::duration<double, std::milli>(elapsed).count()));
  RecordProperty("expanded_states",
                 std::to_string(result.metrics.expanded_states));
  RecordProperty("fast_path_accepts",
                 std::to_string(result.fast_path_accepts));
  RecordProperty("exact_sweep_fallbacks",
                 std::to_string(result.exact_sweep_fallbacks));
  RecordProperty("exact_sweep_cell_checks",
                 std::to_string(result.exact_sweep_cell_checks));
  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.metrics.expanded_states, 0U);
  EXPECT_GT(result.fast_path_accepts, 0U);
  EXPECT_EQ(result.exact_sweep_fallbacks, 0U);
#if !defined(__SANITIZE_ADDRESS__)
  EXPECT_LT(elapsed, std::chrono::seconds{1});
#else
  (void)elapsed;
#endif
}

TEST(AnytimeLeggedPlanner, AcceptsAlreadySatisfiedGoalAfterBodyCheck) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  const LeggedCapability capability = Capability();

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.1, 1.5, 0.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_TRUE(result.trajectory.empty());
  EXPECT_DOUBLE_EQ(result.cost, 0.0);
  EXPECT_EQ(result.metrics.edge_validation_evaluations, 1U);
}

TEST(AnytimeLeggedPlanner, RejectsInvalidGoalYawTolerance) {
  const TerrainFixture fixture = MakeTerrain(20U, 15U, 0.2);
  const LeggedCapability capability = Capability();
  LeggedPlanRequest request = RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.5, 1.5, 0.0});
  request.goals_odom.goals_odom.front().yaw_tolerance_rad =
      std::numeric_limits<double>::quiet_NaN();

  const LeggedPlanResult non_finite = PlanLegged(request);

  EXPECT_EQ(non_finite.status, LocalPlanStatus::kInvalidInput);
  request.goals_odom.goals_odom.front().yaw_tolerance_rad = -0.1;

  const LeggedPlanResult negative = PlanLegged(request);

  EXPECT_EQ(negative.status, LocalPlanStatus::kInvalidInput);
}

TEST(AnytimeLeggedPlanner, UsesLateralAndBackwardCapabilityPrimitives) {
  const TerrainFixture fixture = MakeTerrain(24U, 20U, 0.2);
  LeggedCapability lateral = Capability();
  lateral.motion_primitives = {lateral.motion_primitives.at(2U)};
  const LeggedPlanResult lateral_result = PlanLegged(RequestTo(
      fixture, lateral, {2.1, 1.1, 0.5}, {2.1, 1.9, 0.0}));
  ASSERT_TRUE(lateral_result.ok()) << lateral_result.reason_code;
  EXPECT_TRUE(std::ranges::any_of(
      lateral_result.trajectory, [](const LeggedTransition& transition) {
        return transition.primitive_kind == LeggedPrimitiveKind::kLateralLeft;
      }));

  LeggedCapability backward = Capability();
  backward.motion_primitives = {backward.motion_primitives.at(1U)};
  const LeggedPlanResult backward_result = PlanLegged(RequestTo(
      fixture, backward, {2.1, 2.1, 0.5}, {1.3, 2.1, 0.0}));
  ASSERT_TRUE(backward_result.ok()) << backward_result.reason_code;
  EXPECT_TRUE(std::ranges::any_of(
      backward_result.trajectory, [](const LeggedTransition& transition) {
        return transition.primitive_kind == LeggedPrimitiveKind::kBackward;
      }));
}

TEST(AnytimeLeggedPlanner, RejectsQuantizedAliasWithDifferentPhysicalEndpoint) {
  const TerrainFixture fixture = MakeTerrain(24U, 20U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {
      LeggedBodyPrimitive{
          .primitive_id = "a-forward-031",
          .kind = LeggedPrimitiveKind::kForward,
          .body_frame_displacement_m = {0.31, 0.0, 0.0},
      },
      LeggedBodyPrimitive{
          .primitive_id = "b-forward-029",
          .kind = LeggedPrimitiveKind::kForward,
          .body_frame_displacement_m = {0.29, 0.0, 0.0},
      },
  };

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {2.03, 1.5, 0.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.quantized_endpoint_aliases, 0U);
  for (const LeggedTransition& transition : result.trajectory) {
    const double dx = transition.target_pose.position_m.x -
        transition.source_pose.position_m.x;
    EXPECT_TRUE(std::abs(dx - 0.31) <= 1.0e-9 || dx < 0.31);
  }
}

TEST(AnytimeLeggedPlanner, DistinguishesMotionModeAtSameSpatialKey) {
  const TerrainFixture fixture = MakeTerrain(24U, 20U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {
      LeggedBodyPrimitive{
          .primitive_id = "forward-mode",
          .kind = LeggedPrimitiveKind::kForward,
          .body_frame_displacement_m = {0.4, 0.0, 0.0},
      },
      LeggedBodyPrimitive{
          .primitive_id = "coupled-mode",
          .kind = LeggedPrimitiveKind::kCoupled,
          .body_frame_displacement_m = {0.4, 0.0, 0.0},
      },
  };

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.9, 1.5, 0.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GE(result.quantized_state_count, 3U);
  EXPECT_EQ(result.quantized_endpoint_aliases, 0U);
}

TEST(AnytimeLeggedPlanner, NormalizesCostAndChargesActualModeChange) {
  const TerrainFixture fixture = MakeTerrain(24U, 20U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.9, 1.5, 0.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  constexpr double kLengthScale = 0.6324555320336759;
  EXPECT_EQ(result.mode_change_edge_count, 1U);
  EXPECT_NEAR(result.cost_scales.at(0U), kLengthScale, 1.0e-12);
  EXPECT_NEAR(result.cost_scales.at(1U),
              kLengthScale / std::hypot(0.8, 0.4), 1.0e-12);
  EXPECT_NEAR(result.cost_scales.at(2U),
              0.6 + 0.2 / kLengthScale, 1.0e-12);
  EXPECT_DOUBLE_EQ(result.cost_scales.at(3U), 1.0);
  EXPECT_DOUBLE_EQ(result.cost_scales.at(4U), 3.0);
  EXPECT_GE(result.cost_components.at(4U), 0.1);
  double normalized = 0.0;
  for (std::size_t index = 0U; index < result.cost_components.size(); ++index) {
    EXPECT_TRUE(std::isfinite(result.cost_components.at(index)));
    EXPECT_GE(result.cost_components.at(index), 0.0);
    normalized += result.cost_components.at(index) /
        result.cost_scales.at(index);
  }
  EXPECT_NEAR(result.cost, normalized, 1.0e-9);
}

TEST(AnytimeLeggedPlanner, RejectsNonFiniteElevationInsideSearch) {
  constexpr std::size_t kWidth = 18U;
  constexpr std::size_t kHeight = 11U;
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    elevation[y * kWidth + 8U] = std::numeric_limits<float>::quiet_NaN();
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, 0.2, {}, std::move(elevation));
  const LeggedCapability capability = Capability();

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {0.9, 1.1, 0.5}, {2.7, 1.1, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath);
}

TEST(AnytimeLeggedPlanner, RejectsSlopeAboveCapabilityInsideSearch) {
  constexpr std::size_t kWidth = 18U;
  constexpr std::size_t kHeight = 11U;
  std::vector<float> elevation(kWidth * kHeight);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      elevation[y * kWidth + x] = static_cast<float>(0.15 * x);
    }
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, 0.2, {}, std::move(elevation));
  LeggedCapability capability = Capability();
  capability.maximum_slope_rad = 0.5;

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {0.9, 1.1, 1.1}, {2.7, 1.1, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath);
}

TEST(AnytimeLeggedPlanner, RejectsStepAboveCapabilityInsideSearch) {
  constexpr std::size_t kWidth = 18U;
  constexpr std::size_t kHeight = 11U;
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 9U; x < kWidth; ++x) {
      elevation[y * kWidth + x] = 0.21F;
    }
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, 0.2, {}, std::move(elevation));
  LeggedCapability capability = Capability();
  capability.maximum_slope_rad = 1.5;

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {0.9, 1.1, 0.5}, {2.7, 1.1, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath);
}

TEST(AnytimeLeggedPlanner, TraversesStepWithinCapabilityInsideSearch) {
  constexpr std::size_t kWidth = 18U;
  constexpr std::size_t kHeight = 11U;
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 9U; x < kWidth; ++x) {
      elevation[y * kWidth + x] = 0.19F;
    }
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, 0.2, {}, std::move(elevation));
  LeggedCapability capability = Capability();
  capability.maximum_slope_rad = 1.5;

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {0.9, 1.1, 0.5}, {2.7, 1.1, 0.0}));

  EXPECT_TRUE(result.ok()) << result.reason_code;
}

TEST(AnytimeLeggedPlanner, UsesRectangleInsteadOfCircumscribedCircleInCorridor) {
  constexpr std::size_t kWidth = 25U;
  constexpr std::size_t kHeight = 9U;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, 0.2,
      CorridorOccupancy(kWidth, kHeight, 3U, 5U));
  const LeggedCapability capability = Capability();
  const double corridor_width = 3.0 * fixture.map->resolution_m();
  const double circle_diameter = std::hypot(
      capability.body_extent_m.x, capability.body_extent_m.y);
  ASSERT_LT(capability.body_extent_m.y, corridor_width);
  ASSERT_GT(circle_diameter, corridor_width);

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {0.7, 0.9, 0.5}, {4.1, 0.9, 0.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_TRUE(result.metrics.used_narrow_resolution);
  EXPECT_NEAR(result.finest_xy_key_resolution_m, 0.1, 1.0e-12);
  EXPECT_EQ(result.maximum_yaw_bin_count, 128U);
}

TEST(AnytimeLeggedPlanner, RejectsCorridorNarrowerThanClearedBodyRectangle) {
  constexpr std::size_t kWidth = 25U;
  constexpr std::size_t kHeight = 9U;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, 0.2,
      CorridorOccupancy(kWidth, kHeight, 3U, 5U));
  LeggedCapability capability = Capability();
  capability.minimum_body_clearance_m = 0.21;

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {0.7, 0.9, 0.5}, {4.1, 0.9, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath);
}

TEST(AnytimeLeggedPlanner, RejectsEndpointBodyContactOnExactGoalConnector) {
  constexpr std::size_t kWidth = 20U;
  constexpr std::size_t kHeight = 15U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[7U * kWidth + 8U] = 1.0F;
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, 0.2, std::move(occupancy));
  const LeggedCapability capability = Capability();

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {1.1, 1.5, 0.5}, {1.3, 1.5, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath);
}

TEST(AnytimeLeggedPlanner, BoundsSweepSpacingAndCachesEachEdgeEvaluation) {
  const TerrainFixture fixture = MakeTerrain(22U, 15U, 0.2);
  const LeggedCapability capability = Capability();

  const LeggedPlanResult result = PlanLegged(RequestTo(
      fixture, capability, {0.9, 1.5, 0.5}, {3.3, 1.5, 0.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.metrics.edge_validation_evaluations, 0U);
  EXPECT_LE(result.maximum_sweep_translation_step_m,
            fixture.map->resolution_m() / 4.0 + 1.0e-12);
  EXPECT_GT(result.edge_validation_cache_hits, 0U);
  EXPECT_EQ(result.maximum_edge_sweep_evaluations, 1U);
  EXPECT_GT(result.fast_path_accepts, 0U);
  EXPECT_EQ(result.sweep_cell_checks, 0U);
  EXPECT_EQ(result.exact_sweep_cell_checks, 0U);
}

TEST(AnytimeLeggedPlanner, ReturnsCanceledAndTimedOutWithoutSearching) {
  const TerrainFixture fixture = MakeTerrain(12U, 12U, 0.2);
  const LeggedCapability capability = Capability();
  std::stop_source stop;
  stop.request_stop();
  LeggedPlanRequest canceled = RequestTo(
      fixture, capability, {0.9, 1.1, 0.5}, {1.7, 1.1, 0.0});
  canceled.control.stop_token = stop.get_token();
  EXPECT_EQ(PlanLegged(canceled).status, LocalPlanStatus::kCanceled);

  LeggedPlanRequest timed_out = RequestTo(
      fixture, capability, {0.9, 1.1, 0.5}, {1.7, 1.1, 0.0});
  timed_out.control.deadline = SteadyClock::time_point{};
  timed_out.control.now = [] {
    return SteadyClock::time_point{std::chrono::nanoseconds{1}};
  };
  EXPECT_EQ(PlanLegged(timed_out).status, LocalPlanStatus::kTimedOut);
}

TEST(AnytimeLeggedPlanner, StopsForCancellationAndDeadlineDuringSearch) {
  const TerrainFixture fixture = MakeTerrain(320U, 320U, 0.2);
  LeggedCapability capability = Capability();
  capability.motion_primitives = {
      capability.motion_primitives.at(0U),
      capability.motion_primitives.at(2U),
  };

  const auto timeout_reads = std::make_shared<std::int64_t>(0);
  LeggedPlanRequest timed_out = RequestTo(
      fixture, capability, {10.1, 10.1, 0.5}, {50.1, 50.1, 0.0});
  timed_out.control.deadline = SteadyClock::time_point{
      std::chrono::nanoseconds{20695}};
  timed_out.control.now = [timeout_reads] {
    return SteadyClock::time_point{
        std::chrono::nanoseconds{(*timeout_reads)++}};
  };

  const LeggedPlanResult timeout_result = PlanLegged(timed_out);

  EXPECT_EQ(timeout_result.status, LocalPlanStatus::kTimedOut);
  EXPECT_GT(timeout_result.metrics.expanded_states, 0U);
  EXPECT_GT(*timeout_reads, 100);

  std::stop_source stop;
  const auto cancel_reads = std::make_shared<std::int64_t>(0);
  LeggedPlanRequest canceled = RequestTo(
      fixture, capability, {10.1, 10.1, 0.5}, {50.1, 50.1, 0.0});
  canceled.control.stop_token = stop.get_token();
  canceled.control.deadline = SteadyClock::time_point::max();
  canceled.control.now = [cancel_reads, &stop] {
    if ((*cancel_reads)++ == 20695) {
      stop.request_stop();
    }
    return SteadyClock::time_point{};
  };

  const LeggedPlanResult canceled_result = PlanLegged(canceled);

  EXPECT_EQ(canceled_result.status, LocalPlanStatus::kCanceled);
  EXPECT_GT(canceled_result.metrics.expanded_states, 0U);
  EXPECT_GT(*cancel_reads, 100);
}

}  // namespace
}  // namespace lunar::pure_planning::legged
