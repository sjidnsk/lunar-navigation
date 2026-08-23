#include "hopper/anytime_hopper_planner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>
#include <stop_token>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "shared/local_terrain_projection.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/obstacle_height_estimator.hpp"

namespace lunar::pure_planning::hopper {
namespace {

using namespace std::chrono_literals;

struct TerrainFixture final {
  std::shared_ptr<const shared::MapSnapshot> map;
  shared::LocalTerrainProjection terrain;
};

struct CountingHeightEstimator final {
  shared::ObstacleHeight answer{.flyover_allowed = true, .height_m = 0.5};
  std::vector<std::pair<shared::GridCell, std::size_t>> calls;

  [[nodiscard]] std::function<shared::ObstacleHeight(
      const shared::MapSnapshot&, shared::GridCell)>
  Bind() {
    return [this](const shared::MapSnapshot&, const shared::GridCell cell) {
      for (auto& [recorded, count] : calls) {
        if (recorded == cell) {
          ++count;
          return answer;
        }
      }
      calls.emplace_back(cell, 1U);
      return answer;
    };
  }

  [[nodiscard]] std::size_t calls_for(const shared::GridCell cell) const {
    for (const auto& [recorded, count] : calls) {
      if (recorded == cell) {
        return count;
      }
    }
    return 0U;
  }
};

[[nodiscard]] TerrainFixture MakeTerrain(
    const std::size_t width, const std::size_t height,
    std::vector<float> occupancy, std::vector<float> elevation = {},
    const double resolution_m = 1.0) {
  if (elevation.empty()) {
    elevation.assign(width * height, 0.0F);
  }
  GridMap grid{
      .frame_id = "odom",
      .width = width,
      .height = height,
      .resolution_m = resolution_m,
      .origin_m = {},
      .layers = {
          {"occupancy", GridLayer{.values = std::move(occupancy)}},
          {"elevation", GridLayer{.values = std::move(elevation)}},
      },
  };
  auto snapshot = shared::MapSnapshot::Create(
      std::move(grid), shared::MapContract::kLocalElevation);
  EXPECT_TRUE(snapshot.ok()) << snapshot.reason_code;
  auto projection = shared::BuildLocalTerrainProjection(snapshot.snapshot);
  EXPECT_TRUE(projection.ok()) << projection.reason_code;
  return TerrainFixture{
      .map = std::move(snapshot.snapshot),
      .terrain = std::move(*projection.value),
  };
}

[[nodiscard]] TerrainFixture FlatTerrain(const std::size_t width = 16U,
                                         const std::size_t height = 9U) {
  return MakeTerrain(width, height,
                     std::vector<float>(width * height, 0.0F));
}

[[nodiscard]] HopperCapability Capability() {
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

[[nodiscard]] HopperCapability FormalCapability() {
  HopperCapability capability = Capability();
  capability.reference_horizontal_range_m = 100.0;
  capability.landing_support_radius_m = 0.45;
  capability.flight_collision_radius_m = 0.55;
  capability.landing_lateral_margin_m = 0.2;
  capability.flight_map_margin_m = 0.2;
  capability.maximum_landing_slope_rad = 0.174533;
  return capability;
}

[[nodiscard]] HopperPlanRequest RequestTo(
    const TerrainFixture& fixture, const HopperCapability& capability,
    const Vec3 goal, const Vec3 start = {1.5, 4.5, 0.0}) {
  return HopperPlanRequest{
      .start = HopperState{
          .pose = Pose3{.position_m = start},
      },
      .goal_odom = GoalRegion{
          .goal_id = "exact-point",
          .target = PointGoal{
              .position_m = goal,
              .tolerance_m = 1.0e-6,
          },
      },
      .terrain = &fixture.terrain,
      .capability = &capability,
      .control = SearchControl{
          .deadline = SteadyClock::now() + 2s,
      },
      .estimate_height = shared::EstimateObstacleHeight,
  };
}

TEST(AnytimeHopperPlanner, ReachesExactOffGridGoalThroughARealHop) {
  const TerrainFixture fixture = FlatTerrain();
  const HopperCapability capability = Capability();
  const Vec3 exact_goal{.x = 6.25, .y = 4.4, .z = 0.0};

  const HopperPlanResult result =
      PlanHopper(RequestTo(fixture, capability, exact_goal));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_EQ(result.hops.size(), 1U);
  EXPECT_DOUBLE_EQ(result.hops.front().nominal_landing_point_m.x,
                   exact_goal.x);
  EXPECT_DOUBLE_EQ(result.hops.front().nominal_landing_point_m.y,
                   exact_goal.y);
  EXPECT_DOUBLE_EQ(result.hops.front().nominal_landing_point_m.z,
                   exact_goal.z);
  EXPECT_TRUE(std::isfinite(result.cost));
  EXPECT_GT(result.cost, 0.0);
}

TEST(AnytimeHopperPlanner, StopsWhenControlTriggersOnlyDuringHopReconstruction) {
  const TerrainFixture fixture = FlatTerrain();
  const HopperCapability capability = Capability();
  const Vec3 goal{.x = 6.25, .y = 4.4, .z = 0.0};
  // All pre-reconstruction work consumes exactly this many deterministic
  // clock reads; the next read is the first hop checkpoint.
  constexpr std::size_t kFirstReconstructionClockRead = 129U;

  std::stop_source stop;
  std::size_t cancel_reads = 0U;
  HopperPlanRequest canceled = RequestTo(fixture, capability, goal);
  canceled.control.stop_token = stop.get_token();
  canceled.control.deadline = SteadyClock::time_point::max();
  canceled.control.now = [&] {
    if (cancel_reads++ == kFirstReconstructionClockRead) {
      stop.request_stop();
    }
    return SteadyClock::time_point{};
  };
  const HopperPlanResult canceled_result = PlanHopper(canceled);
  EXPECT_EQ(canceled_result.status, LocalPlanStatus::kCanceled)
      << "clock_reads=" << cancel_reads;

  std::size_t timeout_reads = 0U;
  HopperPlanRequest timed_out = RequestTo(fixture, capability, goal);
  timed_out.control.deadline = SteadyClock::time_point{1ms};
  timed_out.control.now = [&] {
    return timeout_reads++ < kFirstReconstructionClockRead
               ? SteadyClock::time_point{}
               : SteadyClock::time_point{2ms};
  };
  const HopperPlanResult timeout_result = PlanHopper(timed_out);
  EXPECT_EQ(timeout_result.status, LocalPlanStatus::kTimedOut)
      << "clock_reads=" << timeout_reads;
}

TEST(AnytimeHopperPlanner,
     ReturnsAZeroCostStationaryPlanWhenTheStartSatisfiesTheGoal) {
  const TerrainFixture fixture = FlatTerrain();
  const HopperCapability capability = Capability();
  HopperPlanRequest request = RequestTo(
      fixture, capability,
      Vec3{1.5, 4.5, std::numeric_limits<double>::quiet_NaN()});
  std::get<PointGoal>(request.goal_odom.target).tolerance_m = 0.1;
  request.goal_odom.yaw_rad = 0.05;
  request.goal_odom.yaw_tolerance_rad = 0.1;

  const HopperPlanResult result = PlanHopper(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.status, LocalPlanStatus::kSolved);
  EXPECT_EQ(result.reason_code, "HOPPER_STATIONARY_GOAL_SATISFIED");
  EXPECT_TRUE(result.hops.empty());
  EXPECT_DOUBLE_EQ(result.cost, 0.0);
}

TEST(AnytimeHopperPlanner,
     PlansAcrossAFormalThreeHundredTwentySquareMapWithinOneSecond) {
  constexpr std::size_t kWidth = 320U;
  constexpr std::size_t kHeight = 320U;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F), {},
      0.2);
  const HopperCapability capability = FormalCapability();
  HopperPlanRequest request = RequestTo(
      fixture, capability,
      Vec3{63.0, 32.0, std::numeric_limits<double>::quiet_NaN()},
      Vec3{1.0, 32.0, 0.0});
  request.control.deadline = SteadyClock::now() + 1100ms;

  const SteadyClock::time_point begin = SteadyClock::now();
  const HopperPlanResult result = PlanHopper(request);
  const SteadyClock::duration elapsed = SteadyClock::now() - begin;

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_LT(elapsed, 1s);
}

TEST(AnytimeHopperPlanner,
     FindsFormalMediumRangeHopsWhenTheDirectFlightTubeIsBlocked) {
  constexpr std::size_t kWidth = 320U;
  constexpr std::size_t kHeight = 320U;
  constexpr double kResolution = 0.2;
  const Vec3 start{4.1, 32.1, 0.0};
  const Vec3 intermediate{15.5, 37.9, 0.0};
  const Vec3 goal{26.9, 32.1, 0.0};
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  std::vector<float> elevation(kWidth * kHeight, 0.01F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      const double center_x = (static_cast<double>(x) + 0.5) * kResolution;
      const double center_y = (static_cast<double>(y) + 0.5) * kResolution;
      for (const Vec3 landing :
           std::array<Vec3, 3>{start, intermediate, goal}) {
        if (std::hypot(center_x - landing.x, center_y - landing.y) <= 1.2) {
          occupancy[y * kWidth + x] = 0.0F;
          elevation[y * kWidth + x] = 0.0F;
        }
      }
    }
  }
  occupancy[160U * kWidth + 77U] =
      std::numeric_limits<float>::quiet_NaN();
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), std::move(elevation),
      kResolution);
  const HopperCapability capability = FormalCapability();
  HopperPlanRequest request = RequestTo(fixture, capability, goal, start);
  request.estimate_height = [](const shared::MapSnapshot&,
                               const shared::GridCell) {
    return shared::ObstacleHeight{
        .flyover_allowed = true,
        .height_m = 0.01,
    };
  };

  for (std::size_t repetition = 0U; repetition < 5U; ++repetition) {
    SCOPED_TRACE(repetition);
    request.control.deadline = SteadyClock::now() + 1100ms;
    const SteadyClock::time_point begin = SteadyClock::now();
    const HopperPlanResult result = PlanHopper(request);
    const SteadyClock::duration elapsed = SteadyClock::now() - begin;

    ASSERT_TRUE(result.ok()) << result.reason_code;
    EXPECT_LT(elapsed, 1s);
    ASSERT_EQ(result.hops.size(), 2U);
    for (const HopSegment& hop : result.hops) {
      const double distance = std::hypot(
          hop.nominal_landing_point_m.x - hop.launch_pose.position_m.x,
          hop.nominal_landing_point_m.y - hop.launch_pose.position_m.y);
      EXPECT_GE(distance, 3.0);
      EXPECT_LE(distance, 25.0);
    }
    EXPECT_DOUBLE_EQ(result.hops.front().nominal_landing_point_m.x,
                     intermediate.x);
    EXPECT_DOUBLE_EQ(result.hops.front().nominal_landing_point_m.y,
                     intermediate.y);
  }
}

TEST(AnytimeHopperPlanner,
     FindsTheOnlyUnalignedReachableLandingWithinOneSecond) {
  constexpr std::size_t kWidth = 320U;
  constexpr std::size_t kHeight = 320U;
  constexpr double kResolution = 0.2;
  const Vec3 start{4.1, 32.1, 0.0};
  const Vec3 intermediate{14.9, 32.1, 0.0};
  const Vec3 goal{24.1, 42.1, 0.0};
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  std::vector<float> elevation(kWidth * kHeight, 0.01F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      const double center_x = (static_cast<double>(x) + 0.5) * kResolution;
      const double center_y = (static_cast<double>(y) + 0.5) * kResolution;
      for (const Vec3 landing :
           std::array<Vec3, 3>{start, intermediate, goal}) {
        const double landing_distance =
            std::hypot(center_x - landing.x, center_y - landing.y);
        if (landing_distance <= 1.4) {
          occupancy[y * kWidth + x] = 0.0F;
          elevation[y * kWidth + x] =
              landing_distance <= 0.9 ? 0.0F : -1.0F;
        }
      }
    }
  }
  for (std::size_t y = 175U; y <= 195U; ++y) {
    occupancy[y * kWidth + 70U] =
        std::numeric_limits<float>::quiet_NaN();
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), std::move(elevation),
      kResolution);
  const HopperCapability capability = FormalCapability();
  HopperPlanRequest request = RequestTo(fixture, capability, goal, start);
  request.estimate_height = [](const shared::MapSnapshot&,
                               const shared::GridCell) {
    return shared::ObstacleHeight{
        .flyover_allowed = true,
        .height_m = 0.01,
    };
  };
  for (std::size_t repetition = 0U; repetition < 5U; ++repetition) {
    SCOPED_TRACE(repetition);
    request.control.deadline = SteadyClock::now() + 1100ms;
    const SteadyClock::time_point begin = SteadyClock::now();
    const HopperPlanResult result = PlanHopper(request);
    const SteadyClock::duration elapsed = SteadyClock::now() - begin;

    ASSERT_TRUE(result.ok()) << result.reason_code << " evaluations="
                             << result.metrics.edge_validation_evaluations;
    EXPECT_LT(elapsed, 1s);
    ASSERT_GE(result.hops.size(), 2U);
    const auto uses_intermediate_support = std::ranges::any_of(
        result.hops, [&](const HopSegment& hop) {
          const Vec3 landing = hop.nominal_landing_point_m;
          return std::hypot(landing.x - intermediate.x,
                            landing.y - intermediate.y) <= 1.4;
        });
    EXPECT_TRUE(uses_intermediate_support);
  }
}

TEST(AnytimeHopperPlanner, IgnoresGoalZAndUsesTheLandingElevation) {
  constexpr std::size_t kWidth = 16U;
  constexpr std::size_t kHeight = 9U;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F),
      std::vector<float>(kWidth * kHeight, 2.0F));
  const HopperCapability capability = Capability();

  const HopperPlanResult result = PlanHopper(RequestTo(
      fixture, capability,
      Vec3{6.25, 4.4, std::numeric_limits<double>::quiet_NaN()},
      Vec3{1.5, 4.5, 2.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.hops.empty());
  EXPECT_DOUBLE_EQ(result.hops.back().nominal_landing_point_m.x, 6.25);
  EXPECT_DOUBLE_EQ(result.hops.back().nominal_landing_point_m.y, 4.4);
  EXPECT_DOUBLE_EQ(result.hops.back().nominal_landing_point_m.z, 2.0);
}

TEST(AnytimeHopperPlanner, UsesPointGoalToleranceInTheXYPlane) {
  constexpr std::size_t kWidth = 10U;
  constexpr std::size_t kHeight = 7U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[3U * kWidth + 7U] = 1.0F;
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const HopperCapability capability = Capability();
  HopperPlanRequest request = RequestTo(
      fixture, capability, Vec3{7.1, 3.5, 123.0},
      Vec3{1.5, 3.5, 0.0});
  std::get<PointGoal>(request.goal_odom.target).tolerance_m = 0.75;

  const HopperPlanResult result = PlanHopper(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.hops.empty());
  const Vec3 landing = result.hops.back().nominal_landing_point_m;
  EXPECT_LE(std::hypot(landing.x - 7.1, landing.y - 3.5), 0.75 + 1.0e-9);
  EXPECT_NE(landing.x, 7.1);
  EXPECT_DOUBLE_EQ(landing.z, 0.0);
}

TEST(AnytimeHopperPlanner,
     SearchesTheCompleteBoundedOneMeterGoalRegion) {
  constexpr std::size_t kWidth = 320U;
  constexpr std::size_t kHeight = 320U;
  constexpr double kResolution = 0.2;
  const shared::GridCell start_cell{.x = 20, .y = 160};
  const shared::GridCell terminal_cell{.x = 104, .y = 160};
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  std::vector<float> elevation(kWidth * kHeight, -10.0F);
  for (const shared::GridCell cell : {start_cell, terminal_cell}) {
    occupancy[static_cast<std::size_t>(cell.y) * kWidth +
              static_cast<std::size_t>(cell.x)] = 0.0F;
    elevation[static_cast<std::size_t>(cell.y) * kWidth +
              static_cast<std::size_t>(cell.x)] = 0.0F;
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), std::move(elevation),
      kResolution);
  const HopperCapability capability = FormalCapability();
  HopperPlanRequest request = RequestTo(
      fixture, capability,
      Vec3{20.1, 32.1, std::numeric_limits<double>::quiet_NaN()},
      fixture.map->CellCenter(start_cell));
  std::get<PointGoal>(request.goal_odom.target).tolerance_m = 1.0;
  request.estimate_height = [](const shared::MapSnapshot&,
                               const shared::GridCell) {
    return shared::ObstacleHeight{
        .flyover_allowed = true,
        .height_m = 0.01,
    };
  };

  request.control.deadline = SteadyClock::now() + 1100ms;

  const SteadyClock::time_point begin = SteadyClock::now();
  const HopperPlanResult result = PlanHopper(request);
  const SteadyClock::duration elapsed = SteadyClock::now() - begin;

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_LT(elapsed, 1s);
  ASSERT_EQ(result.hops.size(), 1U);
  const Vec3 landing = result.hops.front().nominal_landing_point_m;
  EXPECT_DOUBLE_EQ(landing.x, 20.9);
  EXPECT_DOUBLE_EQ(landing.y, 32.1);
  EXPECT_DOUBLE_EQ(landing.z, 0.0);
  EXPECT_LE(std::hypot(landing.x - 20.1, landing.y - 32.1), 1.0);
}

TEST(AnytimeHopperPlanner,
     AcceptsASatisfiedOptionalYawWithoutInventingRotation) {
  const TerrainFixture fixture = FlatTerrain();
  const HopperCapability capability = Capability();
  HopperPlanRequest request =
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 99.0});
  request.goal_odom.yaw_rad = 0.05;
  request.goal_odom.yaw_tolerance_rad = 0.1;

  const HopperPlanResult result = PlanHopper(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  for (const HopSegment& hop : result.hops) {
    EXPECT_EQ(hop.launch_pose.orientation, Quaternion{});
  }
}

TEST(AnytimeHopperPlanner,
     RejectsAnUnsatisfiedOptionalYawWithoutSynthesizingRotation) {
  const TerrainFixture fixture = FlatTerrain();
  const HopperCapability capability = Capability();
  HopperPlanRequest request =
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 0.0});
  request.goal_odom.yaw_rad = 1.0;
  request.goal_odom.yaw_tolerance_rad = 0.1;

  const HopperPlanResult result = PlanHopper(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_TRUE(result.hops.empty());
}

TEST(AnytimeHopperPlanner, RejectsAnOccupiedLandingCell) {
  constexpr std::size_t kWidth = 16U;
  constexpr std::size_t kHeight = 9U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[4U * kWidth + 6U] = 1.0F;
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const HopperCapability capability = Capability();

  const HopperPlanResult result = PlanHopper(
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_TRUE(result.hops.empty());
}

TEST(AnytimeHopperPlanner, RejectsALandingWithoutFiniteElevation) {
  constexpr std::size_t kWidth = 16U;
  constexpr std::size_t kHeight = 9U;
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  elevation[4U * kWidth + 6U] =
      std::numeric_limits<float>::quiet_NaN();
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F),
      std::move(elevation));
  const HopperCapability capability = Capability();

  const HopperPlanResult result = PlanHopper(
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_TRUE(result.hops.empty());
}

TEST(AnytimeHopperPlanner, RejectsALandingAboveTheCapabilitySlope) {
  constexpr std::size_t kWidth = 16U;
  constexpr std::size_t kHeight = 9U;
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 6U; x < kWidth; ++x) {
      elevation[y * kWidth + x] = static_cast<float>(2U * (x - 5U));
    }
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F),
      std::move(elevation));
  const HopperCapability capability = Capability();

  const HopperPlanResult result = PlanHopper(
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 2.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_TRUE(result.hops.empty());
}

TEST(AnytimeHopperPlanner,
     UsesDerivedObstacleHeightOnlyForCrossedOccupiedCells) {
  constexpr std::size_t kWidth = 16U;
  constexpr std::size_t kHeight = 9U;
  const shared::GridCell raised_cell{.x = 4, .y = 4};
  const shared::GridCell unrelated_occupied_cell{.x = 14, .y = 1};
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (const shared::GridCell cell :
       {raised_cell, unrelated_occupied_cell}) {
    occupancy[static_cast<std::size_t>(cell.y) * kWidth +
              static_cast<std::size_t>(cell.x)] = 1.0F;
    elevation[static_cast<std::size_t>(cell.y) * kWidth +
              static_cast<std::size_t>(cell.x)] = 0.5F;
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), std::move(elevation));
  const HopperCapability capability = Capability();
  CountingHeightEstimator estimator;
  HopperPlanRequest request =
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 0.0});
  request.estimate_height = estimator.Bind();

  const HopperPlanResult result = PlanHopper(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(estimator.calls_for(raised_cell), 1U);
  EXPECT_EQ(estimator.calls_for(unrelated_occupied_cell), 0U);
}

TEST(AnytimeHopperPlanner,
     RejectsAnIntermediateFreeHighlandThatIntersectsTheFlightTube) {
  constexpr std::size_t kWidth = 10U;
  constexpr std::size_t kHeight = 7U;
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    elevation[y * kWidth + 4U] = 100.0F;
    elevation[y * kWidth + 5U] = 100.0F;
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F),
      std::move(elevation));
  const HopperCapability capability = Capability();
  CountingHeightEstimator estimator;
  HopperPlanRequest request = RequestTo(
      fixture, capability, Vec3{7.25, 3.4, 0.0},
      Vec3{1.5, 3.5, 0.0});
  request.estimate_height = estimator.Bind();

  const HopperPlanResult result = PlanHopper(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_TRUE(estimator.calls.empty());
}

TEST(AnytimeHopperPlanner,
     UsesCapabilityBoundedHopsAndAnExactFinalGoalConnector) {
  const TerrainFixture fixture = FlatTerrain();
  HopperCapability capability = Capability();
  capability.reference_horizontal_range_m = 4.0;
  const Vec3 exact_goal{.x = 12.25, .y = 4.4, .z = 0.0};

  const HopperPlanResult result =
      PlanHopper(RequestTo(fixture, capability, exact_goal));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GE(result.hops.size(), 3U);
  Vec3 launch{1.5, 4.5, 0.0};
  for (const HopSegment& hop : result.hops) {
    EXPECT_LE(std::hypot(hop.nominal_landing_point_m.x - launch.x,
                         hop.nominal_landing_point_m.y - launch.y),
              capability.reference_horizontal_range_m + 1.0e-9);
    launch = hop.nominal_landing_point_m;
  }
  EXPECT_DOUBLE_EQ(result.hops.back().nominal_landing_point_m.x,
                   exact_goal.x);
  EXPECT_DOUBLE_EQ(result.hops.back().nominal_landing_point_m.y,
                   exact_goal.y);
  EXPECT_DOUBLE_EQ(result.hops.back().nominal_landing_point_m.z,
                   exact_goal.z);
}

TEST(AnytimeHopperPlanner,
     RejectsAnUnknownFlightColumnWithoutCallingTheHeightEstimator) {
  constexpr std::size_t kWidth = 10U;
  constexpr std::size_t kHeight = 7U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    occupancy[y * kWidth + 4U] =
        std::numeric_limits<float>::quiet_NaN();
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  HopperCapability capability = Capability();
  capability.reference_horizontal_range_m = 8.0;
  CountingHeightEstimator estimator;
  HopperPlanRequest request = RequestTo(
      fixture, capability, Vec3{7.25, 3.4, 0.0}, Vec3{1.5, 3.5, 0.0});
  request.estimate_height = estimator.Bind();

  const HopperPlanResult result = PlanHopper(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_TRUE(estimator.calls.empty());
}

TEST(AnytimeHopperPlanner,
     UsesTheCurvedFlightTubeWhenHorizontalGravityIsNonZero) {
  constexpr std::size_t kWidth = 80U;
  constexpr std::size_t kHeight = 50U;
  constexpr double kResolution = 0.2;
  const Vec3 start{1.1, 5.1, 0.0};
  const Vec3 goal{8.1, 5.1, 0.0};
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[25U * kWidth + 22U] =
      std::numeric_limits<float>::quiet_NaN();
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), {}, kResolution);
  HopperCapability capability = FormalCapability();
  capability.gravity_mps2 = {0.0, 1.62, -1.62};

  const HopperPlanResult result =
      PlanHopper(RequestTo(fixture, capability, goal, start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_EQ(result.hops.size(), 1U);
  EXPECT_DOUBLE_EQ(result.hops.front().nominal_landing_point_m.x, goal.x);
  EXPECT_DOUBLE_EQ(result.hops.front().nominal_landing_point_m.y, goal.y);
}

TEST(AnytimeHopperPlanner,
     RejectsAnUnflyableVerticalColumnAndCachesEachCellEstimate) {
  constexpr std::size_t kWidth = 10U;
  constexpr std::size_t kHeight = 7U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    occupancy[y * kWidth + 4U] = 1.0F;
    elevation[y * kWidth + 4U] = 0.5F;
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), std::move(elevation));
  HopperCapability capability = Capability();
  capability.reference_horizontal_range_m = 8.0;
  CountingHeightEstimator estimator;
  estimator.answer = {.flyover_allowed = false, .height_m = 0.0};
  HopperPlanRequest request = RequestTo(
      fixture, capability, Vec3{7.25, 3.4, 0.0}, Vec3{1.5, 3.5, 0.0});
  request.estimate_height = estimator.Bind();

  const HopperPlanResult result = PlanHopper(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_EQ(estimator.calls_for(shared::GridCell{.x = 4, .y = 3}), 1U);
}

TEST(AnytimeHopperPlanner,
     RejectsAPositiveObstacleWhenTheTrajectoryBottomDoesNotClearIt) {
  constexpr std::size_t kWidth = 10U;
  constexpr std::size_t kHeight = 7U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    occupancy[y * kWidth + 4U] = 1.0F;
    elevation[y * kWidth + 4U] = 100.0F;
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), std::move(elevation));
  HopperCapability capability = Capability();
  capability.reference_horizontal_range_m = 8.0;
  CountingHeightEstimator estimator;
  estimator.answer = {.flyover_allowed = true, .height_m = 100.0};
  HopperPlanRequest request = RequestTo(
      fixture, capability, Vec3{7.25, 3.4, 0.0}, Vec3{1.5, 3.5, 0.0});
  request.estimate_height = estimator.Bind();

  const HopperPlanResult result = PlanHopper(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_EQ(estimator.calls_for(shared::GridCell{.x = 4, .y = 3}), 1U);
}

TEST(AnytimeHopperPlanner, DoesNotInspectACaveRoofAboveFreeBottomCells) {
  const TerrainFixture fixture = FlatTerrain(10U, 7U);
  const HopperCapability capability = Capability();
  CountingHeightEstimator estimator;
  HopperPlanRequest request = RequestTo(
      fixture, capability, Vec3{7.25, 3.4, 0.0}, Vec3{1.5, 3.5, 0.0});
  request.estimate_height = estimator.Bind();

  const HopperPlanResult result = PlanHopper(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_TRUE(estimator.calls.empty());
}

TEST(AnytimeHopperPlanner, RejectsEveryEdgeAboveAvailableDeltaV) {
  const TerrainFixture fixture = FlatTerrain(8U, 5U);
  HopperCapability capability = Capability();
  capability.reference_propellant_mass_kg = 1.0e-9;

  const HopperPlanResult result = PlanHopper(RequestTo(
      fixture, capability, Vec3{6.25, 2.4, 0.0}, Vec3{1.5, 2.5, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_TRUE(result.hops.empty());
}

TEST(AnytimeHopperPlanner, MapsCancellationAndDeadlineBeforeExpansion) {
  const TerrainFixture fixture = FlatTerrain();
  const HopperCapability capability = Capability();
  std::stop_source stop;
  stop.request_stop();
  HopperPlanRequest canceled =
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 0.0});
  canceled.control.stop_token = stop.get_token();
  HopperPlanRequest timed_out =
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 0.0});
  timed_out.control.deadline = SteadyClock::time_point{};

  const HopperPlanResult canceled_result = PlanHopper(canceled);
  const HopperPlanResult timed_out_result = PlanHopper(timed_out);

  EXPECT_EQ(canceled_result.status, LocalPlanStatus::kCanceled);
  EXPECT_EQ(canceled_result.reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(timed_out_result.status, LocalPlanStatus::kTimedOut);
  EXPECT_EQ(timed_out_result.reason_code, "TIMEOUT");
}

TEST(AnytimeHopperPlanner,
     ReturnsTimeoutWhenTheReachableLandingScanIsNotExhausted) {
  constexpr std::size_t kWidth = 320U;
  constexpr std::size_t kHeight = 320U;
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  occupancy[160U * kWidth + 20U] = 0.0F;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), {}, 0.2);
  const HopperCapability capability = FormalCapability();
  HopperPlanRequest request = RequestTo(
      fixture, capability, Vec3{24.1, 42.1, 0.0},
      Vec3{4.1, 32.1, 0.0});
  std::size_t now_calls = 0U;
  request.control.deadline = SteadyClock::time_point{1ms};
  request.control.now = [&] {
    ++now_calls;
    return now_calls < 50U ? SteadyClock::time_point{0ms}
                             : SteadyClock::time_point{2ms};
  };

  const HopperPlanResult result = PlanHopper(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kTimedOut) << result.reason_code;
  EXPECT_EQ(result.reason_code, "TIMEOUT");
  EXPECT_GE(now_calls, 50U);
}

TEST(AnytimeHopperPlanner,
     GivesCancellationPriorityWhenDeadlineObservationRequestsStop) {
  const TerrainFixture fixture = FlatTerrain();
  const HopperCapability capability = Capability();
  std::stop_source stop;
  HopperPlanRequest request =
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 0.0});
  request.control.stop_token = stop.get_token();
  request.control.deadline = SteadyClock::time_point{0ms};
  request.control.now = [&] {
    stop.request_stop();
    return SteadyClock::time_point{0ms};
  };

  const HopperPlanResult result = PlanHopper(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
}

TEST(AnytimeHopperPlanner, RejectsAHopRangeThatCannotBeIndexedSafely) {
  const TerrainFixture fixture = FlatTerrain();
  HopperCapability capability = Capability();
  capability.reference_horizontal_range_m =
      static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 1.0;

  const HopperPlanResult result = PlanHopper(
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kInvalidInput)
      << result.reason_code;
  EXPECT_EQ(result.reason_code, "HOPPER_INPUT_INVALID");
}

TEST(AnytimeHopperPlanner, RejectsAFlightTubeThatCannotBeIndexedSafely) {
  const TerrainFixture fixture = FlatTerrain();
  HopperCapability capability = Capability();
  capability.flight_map_margin_m = std::numeric_limits<double>::max();

  const HopperPlanResult result = PlanHopper(
      RequestTo(fixture, capability, Vec3{6.25, 4.4, 0.0}));

  EXPECT_EQ(result.status, LocalPlanStatus::kInvalidInput)
      << result.reason_code;
  EXPECT_EQ(result.reason_code, "HOPPER_INPUT_INVALID");
}

TEST(AnytimeHopperPlanner, IncludesFlightMapMarginInTheActualTubeRadius) {
  constexpr std::size_t kWidth = 10U;
  constexpr std::size_t kHeight = 7U;
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  std::vector<float> elevation(kWidth * kHeight, 0.5F);
  for (std::size_t x = 0U; x < kWidth; ++x) {
    occupancy[3U * kWidth + x] = 0.0F;
    elevation[3U * kWidth + x] = 0.0F;
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), std::move(elevation));
  HopperCapability capability = Capability();
  capability.flight_collision_radius_m = 0.1;
  capability.flight_map_margin_m = 0.45;
  capability.reference_horizontal_range_m = 8.0;
  CountingHeightEstimator estimator;
  estimator.answer = {.flyover_allowed = false, .height_m = 0.0};
  HopperPlanRequest request = RequestTo(
      fixture, capability, Vec3{7.25, 3.4, 0.0}, Vec3{1.5, 3.5, 0.0});
  request.estimate_height = estimator.Bind();

  const HopperPlanResult result = PlanHopper(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_FALSE(estimator.calls.empty());
}

TEST(AnytimeHopperPlanner,
     IntegratesTheTask5SevenBySevenHeightEstimatorFromTwoLayers) {
  constexpr std::size_t kWidth = 10U;
  constexpr std::size_t kHeight = 7U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  occupancy[3U * kWidth + 4U] = 1.0F;
  elevation[3U * kWidth + 4U] = 0.5F;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), std::move(elevation));
  const HopperCapability capability = Capability();

  const HopperPlanResult result = PlanHopper(RequestTo(
      fixture, capability, Vec3{7.25, 3.4, 0.0}, Vec3{1.5, 3.5, 0.0}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.hops.empty());
  EXPECT_DOUBLE_EQ(
      result.hops.front().flight_tube_radius_m,
      capability.flight_collision_radius_m + capability.flight_map_margin_m);
}

}  // namespace
}  // namespace lunar::pure_planning::hopper
