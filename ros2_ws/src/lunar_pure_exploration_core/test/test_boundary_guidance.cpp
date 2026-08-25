#include "lunar_pure_exploration_core/boundary_guidance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_exploration_core/safe_pose_validator.hpp"

namespace lunar::pure_exploration {
namespace {

constexpr std::int8_t kFree = 0;
constexpr std::int8_t kUnknown = -1;
constexpr std::int8_t kOccupied = 100;
constexpr std::int8_t kOccupiedThreshold = 50;

PlatformGeometry TestPlatform() {
  return PlatformGeometry{
      .platform_id = "boundary-test",
      .platform_type = "WHEELED",
      .base_frame_id = "base_link",
      .footprint_vertices =
          {{-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1}},
      .minimum_clearance_m = 0.0,
  };
}

PlatformGeometry LongFootprintPlatform() {
  return PlatformGeometry{
      .platform_id = "boundary-long-footprint-test",
      .platform_type = "WHEELED",
      .base_frame_id = "base_link",
      .footprint_vertices =
          {{-0.6, -0.1}, {0.6, -0.1}, {0.6, 0.1}, {-0.6, 0.1}},
      .minimum_clearance_m = 0.0,
  };
}

PlatformGeometry LargeSquarePlatform() {
  return PlatformGeometry{
      .platform_id = "boundary-large-square-test",
      .platform_type = "WHEELED",
      .base_frame_id = "base_link",
      .footprint_vertices =
          {{-0.6, -0.6}, {0.6, -0.6}, {0.6, 0.6}, {-0.6, 0.6}},
      .minimum_clearance_m = 0.0,
  };
}

BoundaryGuidance::Limits GenerousLimits() {
  return BoundaryGuidance::Limits{
      .maximum_guidance_grid_cells = 4096U,
      .maximum_guidance_work_units = 1000000U,
      .maximum_approach_candidates = 256U,
  };
}

GridGeometry Geometry(std::uint32_t width, std::uint32_t height,
                      double origin_x = 0.0, double origin_y = 0.0,
                      double origin_yaw = 0.0) {
  return GridGeometry{
      .width = width,
      .height = height,
      .resolution = 1.0,
      .origin_x = origin_x,
      .origin_y = origin_y,
      .origin_yaw = origin_yaw,
  };
}

std::size_t Offset(const GridGeometry& geometry, std::int32_t x,
                   std::int32_t y) {
  return static_cast<std::size_t>(y) * geometry.width +
         static_cast<std::size_t>(x);
}

void Set(std::vector<std::int8_t>& data, const GridGeometry& geometry,
         std::int32_t x, std::int32_t y, std::int8_t value) {
  data.at(Offset(geometry, x, y)) = value;
}

std::vector<std::int8_t> Filled(const GridGeometry& geometry,
                                std::int8_t value) {
  return std::vector<std::int8_t>(
      static_cast<std::size_t>(geometry.width) * geometry.height, value);
}

OccupancyGridView Map(const GridGeometry& geometry,
                      std::span<const std::int8_t> data) {
  return OccupancyGridView(geometry, data, kOccupiedThreshold);
}

Vec2 GridPointToWorld(const GridGeometry& geometry, Vec2 grid_point) {
  const auto world = OccupancyGridView::GridToWorld(geometry, grid_point);
  if (!world.has_value()) {
    throw std::runtime_error("test grid point is not representable");
  }
  return *world;
}

Polygon2 GridPolygon(const GridGeometry& geometry,
                     std::initializer_list<Vec2> grid_vertices) {
  Polygon2 polygon;
  polygon.vertices.reserve(grid_vertices.size());
  for (const Vec2 vertex : grid_vertices) {
    polygon.vertices.push_back(GridPointToWorld(geometry, vertex));
  }
  return polygon;
}

Polygon2 GridPolygon(const GridGeometry& geometry,
                     std::span<const Vec2> grid_vertices) {
  Polygon2 polygon;
  polygon.vertices.reserve(grid_vertices.size());
  for (const Vec2 vertex : grid_vertices) {
    polygon.vertices.push_back(GridPointToWorld(geometry, vertex));
  }
  return polygon;
}

Pose2 PoseAt(const OccupancyGridView& map, GridIndex cell,
             double yaw = 0.0) {
  const Vec2 center = map.CellCenter(cell);
  return Pose2{center.x, center.y, yaw};
}

BoundaryGuidance Guidance(
    BoundaryGuidance::Limits limits = GenerousLimits(),
    SensorModel sensor = SensorModel{10.0, std::numbers::pi / 2.0},
    double yaw_tolerance = 0.05) {
  return BoundaryGuidance(TestPlatform(), sensor, yaw_tolerance, limits);
}

const ApproachIntent& FindIntent(const BoundaryGuidanceResult& result,
                                 GridIndex cell) {
  const auto found = std::find_if(
      result.intents.begin(), result.intents.end(),
      [cell](const ApproachIntent& intent) { return intent.cell == cell; });
  if (found == result.intents.end()) {
    throw std::runtime_error("expected intent was not produced");
  }
  return *found;
}

const ApproachCandidate& FindCandidate(const BoundaryGuidanceResult& result,
                                       GridIndex intent,
                                       ApproachCandidateKind kind) {
  const auto found = std::find_if(
      result.candidates.begin(), result.candidates.end(),
      [intent, kind](const ApproachCandidate& candidate) {
        return candidate.identity.intent_cell == intent &&
               candidate.identity.candidate_kind == kind;
      });
  if (found == result.candidates.end()) {
    throw std::runtime_error("expected approach candidate was not produced");
  }
  return *found;
}

void ExpectEveryCandidateExecutable(const OccupancyGridView& map,
                                    const BoundaryGuidanceResult& result) {
  SafePoseValidator validator(TestPlatform(), 1000000U);
  for (const ApproachCandidate& candidate : result.candidates) {
    std::size_t work = 0U;
    EXPECT_TRUE(validator.IsMapFree(map, candidate.pose, work));
    const auto cell =
        map.WorldToCell(Vec2{candidate.pose.x, candidate.pose.y});
    ASSERT_TRUE(cell.has_value());
    EXPECT_EQ(map.Classify(*cell), CellState::kFree);
    EXPECT_EQ(candidate.identity.candidate_key,
              MakeCandidateKey(candidate.pose));
  }
}

void ExpectEquivalent(const BoundaryGuidanceResult& left,
                      const BoundaryGuidanceResult& right) {
  EXPECT_EQ(left.phase, right.phase);
  EXPECT_EQ(left.wait_reason, right.wait_reason);
  EXPECT_EQ(left.fully_inside_task, right.fully_inside_task);
  EXPECT_EQ(left.consumed_work_units, right.consumed_work_units);
  ASSERT_EQ(left.intents.size(), right.intents.size());
  for (std::size_t i = 0U; i < left.intents.size(); ++i) {
    EXPECT_EQ(left.intents[i].cell, right.intents[i].cell);
    EXPECT_EQ(left.intents[i].total_cost.unknown_cell_count,
              right.intents[i].total_cost.unknown_cell_count);
    EXPECT_DOUBLE_EQ(left.intents[i].total_cost.path_length_m,
                     right.intents[i].total_cost.path_length_m);
    EXPECT_EQ(left.intents[i].route, right.intents[i].route);
  }
  ASSERT_EQ(left.candidates.size(), right.candidates.size());
  for (std::size_t i = 0U; i < left.candidates.size(); ++i) {
    const ApproachCandidate& a = left.candidates[i];
    const ApproachCandidate& b = right.candidates[i];
    EXPECT_EQ(a.id, b.id);
    EXPECT_EQ(a.identity, b.identity);
    EXPECT_DOUBLE_EQ(a.pose.x, b.pose.x);
    EXPECT_DOUBLE_EQ(a.pose.y, b.pose.y);
    EXPECT_DOUBLE_EQ(a.pose.yaw, b.pose.yaw);
    EXPECT_EQ(a.remaining_cost.unknown_cell_count,
              b.remaining_cost.unknown_cell_count);
    EXPECT_DOUBLE_EQ(a.remaining_cost.path_length_m,
                     b.remaining_cost.path_length_m);
    EXPECT_DOUBLE_EQ(a.task_unknown_area_m2, b.task_unknown_area_m2);
    EXPECT_EQ(a.guidance_unknown_cell_count,
              b.guidance_unknown_cell_count);
    EXPECT_EQ(a.fully_inside_task, b.fully_inside_task);
    ASSERT_EQ(a.guidance_route.size(), b.guidance_route.size());
    for (std::size_t route_index = 0U;
         route_index < a.guidance_route.size(); ++route_index) {
      EXPECT_DOUBLE_EQ(a.guidance_route[route_index].x,
                       b.guidance_route[route_index].x);
      EXPECT_DOUBLE_EQ(a.guidance_route[route_index].y,
                       b.guidance_route[route_index].y);
    }
  }
}

TEST(BoundaryGuidance, OutsideStartStopsBeforeUnknownAndEmitsOnlySafeFreePose) {
  const GridGeometry geometry = Geometry(8U, 5U);
  std::vector<std::int8_t> data = Filled(geometry, kFree);
  for (std::int32_t y = 1; y <= 3; ++y) {
    for (std::int32_t x = 5; x <= 6; ++x) {
      Set(data, geometry, x, y, kOccupied);
    }
  }
  Set(data, geometry, 5, 2, kUnknown);
  Set(data, geometry, 4, 1, kUnknown);  // Visible, but outside the task.
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{5.0, 1.0}, {7.0, 1.0},
                                  {7.0, 4.0}, {5.0, 4.0}}));

  const BoundaryGuidanceResult result =
      Guidance().Build(map, raster, PoseAt(map, {1, 2}));

  ASSERT_EQ(result.phase, NavigationPhase::kApproachTask);
  ASSERT_EQ(result.wait_reason, ApproachWaitReason::kNone);
  ASSERT_EQ(result.intents.size(), 1U);
  EXPECT_EQ(result.intents.front().cell, (GridIndex{5, 2}));
  const ApproachCandidate& candidate = FindCandidate(
      result, {5, 2}, ApproachCandidateKind::kTranslation);
  const auto candidate_cell =
      map.WorldToCell(Vec2{candidate.pose.x, candidate.pose.y});
  ASSERT_TRUE(candidate_cell.has_value());
  EXPECT_EQ(*candidate_cell, (GridIndex{4, 2}));
  EXPECT_EQ(candidate.remaining_cost.unknown_cell_count, 1U);
  EXPECT_DOUBLE_EQ(candidate.remaining_cost.path_length_m, 1.0);
  EXPECT_EQ(candidate.guidance_unknown_cell_count, 1U);
  EXPECT_DOUBLE_EQ(candidate.task_unknown_area_m2, 1.0);
  EXPECT_FALSE(candidate.fully_inside_task);
  ASSERT_EQ(candidate.guidance_route.size(), 2U);
  EXPECT_EQ(map.WorldToCell(candidate.guidance_route.back()),
            std::optional<GridIndex>(GridIndex{5, 2}));
  ExpectEveryCandidateExecutable(map, result);
}

TEST(BoundaryGuidance, OccupiedClosestEntranceLeavesAnotherBoundaryIntent) {
  const GridGeometry geometry = Geometry(8U, 6U);
  std::vector<std::int8_t> data = Filled(geometry, kFree);
  for (std::int32_t y = 1; y <= 4; ++y) {
    for (std::int32_t x = 5; x <= 6; ++x) {
      Set(data, geometry, x, y, kOccupied);
    }
  }
  Set(data, geometry, 5, 4, kFree);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{5.0, 1.0}, {7.0, 1.0},
                                  {7.0, 5.0}, {5.0, 5.0}}));

  const BoundaryGuidanceResult result =
      Guidance().Build(map, raster, PoseAt(map, {1, 2}));

  ASSERT_EQ(result.wait_reason, ApproachWaitReason::kNone);
  ASSERT_EQ(result.intents.size(), 1U);
  EXPECT_EQ(result.intents.front().cell, (GridIndex{5, 4}));
  const ApproachCandidate& candidate = FindCandidate(
      result, {5, 4}, ApproachCandidateKind::kTranslation);
  EXPECT_TRUE(candidate.fully_inside_task);
  ExpectEveryCandidateExecutable(map, result);
}

TEST(BoundaryGuidance, KnownDetourBeatsShortUnknownRouteLexicographically) {
  const GridGeometry geometry = Geometry(7U, 5U);
  std::vector<std::int8_t> data = Filled(geometry, kFree);
  Set(data, geometry, 2, 2, kUnknown);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{6.0, 2.0}, {7.0, 2.0},
                                  {7.0, 3.0}, {6.0, 3.0}}));

  const BoundaryGuidanceResult result =
      Guidance().Build(map, raster, PoseAt(map, {0, 2}));
  const ApproachIntent& intent = FindIntent(result, {6, 2});

  EXPECT_EQ(intent.total_cost.unknown_cell_count, 0U);
  EXPECT_DOUBLE_EQ(intent.total_cost.path_length_m, 8.0);
  EXPECT_EQ(std::count(intent.route.begin(), intent.route.end(),
                       GridIndex{2, 2}),
            0);
  EXPECT_NE(std::count_if(intent.route.begin(), intent.route.end(),
                          [](GridIndex cell) { return cell.y == 1; }),
            0);
}

TEST(BoundaryGuidance,
     RotatedNegativeWorldOriginConcaveTaskIsVertexOrderDeterministic) {
  const GridGeometry geometry = Geometry(9U, 9U, -10.0, -7.0, 0.37);
  const std::vector<std::int8_t> data = Filled(geometry, kFree);
  const OccupancyGridView map = Map(geometry, data);
  const std::vector<Vec2> vertices{{5.0, 2.0}, {8.0, 2.0}, {8.0, 7.0},
                                   {6.0, 7.0}, {6.0, 4.0}, {5.0, 4.0}};
  std::vector<Vec2> reversed(vertices.rbegin(), vertices.rend());
  std::rotate(reversed.begin(), reversed.begin() + 2, reversed.end());
  const TaskRaster first =
      TaskRaster::Build(map, GridPolygon(geometry, vertices));
  const TaskRaster second =
      TaskRaster::Build(map, GridPolygon(geometry, reversed));
  const Pose2 robot = PoseAt(map, {1, 3}, -0.2);

  const BoundaryGuidanceResult a = Guidance().Build(map, first, robot);
  const BoundaryGuidanceResult b = Guidance().Build(map, second, robot);

  ASSERT_FALSE(a.candidates.empty());
  ExpectEquivalent(a, b);
  ExpectEveryCandidateExecutable(map, a);
}

TEST(BoundaryGuidance, SameXyRotationRequiresYawChangeStrictlyBeyondTolerance) {
  const GridGeometry geometry = Geometry(5U, 3U);
  std::vector<std::int8_t> data = Filled(geometry, kFree);
  Set(data, geometry, 2, 1, kUnknown);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{2.0, 1.0}, {3.0, 1.0},
                                  {3.0, 2.0}, {2.0, 2.0}}));
  const Pose2 robot = PoseAt(map, {1, 1}, std::numbers::pi / 2.0);

  const BoundaryGuidanceResult accepted =
      Guidance(GenerousLimits(), SensorModel{10.0, std::numbers::pi / 2.0},
               0.05)
          .Build(map, raster, robot);
  ASSERT_EQ(accepted.candidates.size(), 1U);
  const ApproachCandidate& rotation = accepted.candidates.front();
  EXPECT_EQ(rotation.identity.candidate_kind,
            ApproachCandidateKind::kRotation);
  EXPECT_DOUBLE_EQ(rotation.pose.x, robot.x);
  EXPECT_DOUBLE_EQ(rotation.pose.y, robot.y);
  EXPECT_GT(std::abs(std::remainder(rotation.pose.yaw - robot.yaw,
                                    2.0 * std::numbers::pi)),
            0.05);
  EXPECT_EQ(rotation.guidance_unknown_cell_count, 1U);

  const BoundaryGuidanceResult rejected =
      Guidance(GenerousLimits(), SensorModel{10.0, std::numbers::pi / 2.0},
               std::numbers::pi / 2.0)
          .Build(map, raster, robot);
  EXPECT_TRUE(rejected.candidates.empty());
  EXPECT_EQ(rejected.wait_reason, ApproachWaitReason::kNoSafeCandidate);
}

TEST(BoundaryGuidance, SameXyRotationWithZeroObservationIsRejected) {
  const GridGeometry geometry = Geometry(5U, 3U);
  std::vector<std::int8_t> data = Filled(geometry, kFree);
  Set(data, geometry, 2, 1, kUnknown);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{2.0, 1.0}, {3.0, 1.0},
                                  {3.0, 2.0}, {2.0, 2.0}}));

  const BoundaryGuidanceResult result =
      Guidance(GenerousLimits(), SensorModel{0.49, std::numbers::pi / 2.0})
          .Build(map, raster,
                 PoseAt(map, {1, 1}, std::numbers::pi / 2.0));

  EXPECT_TRUE(result.candidates.empty());
  EXPECT_EQ(result.wait_reason, ApproachWaitReason::kNoSafeCandidate);
}

TEST(BoundaryGuidance,
     FullMapVisibilityPassesOutsideTaskButOccupiedStillBlocksTheSameUnknown) {
  const GridGeometry geometry = Geometry(7U, 3U);
  std::vector<std::int8_t> data = Filled(geometry, kOccupied);
  for (std::int32_t x = 0; x <= 5; ++x) {
    Set(data, geometry, x, 1, kFree);
  }
  Set(data, geometry, 3, 1, kUnknown);  // Outside-task guidance value.
  const OccupancyGridView map = Map(geometry, data);
  const Polygon2 task = GridPolygon(
      geometry, {{5.0, 1.0}, {6.0, 1.0}, {6.0, 2.0}, {5.0, 2.0}});
  const TaskRaster raster = TaskRaster::Build(map, task);
  BoundaryGuidance guidance = Guidance();

  const BoundaryGuidanceResult built =
      guidance.Build(map, raster, PoseAt(map, {0, 1}));
  const ApproachCandidate& generated = FindCandidate(
      built, {5, 1}, ApproachCandidateKind::kTranslation);
  EXPECT_EQ(generated.guidance_unknown_cell_count, 1U);
  EXPECT_DOUBLE_EQ(generated.task_unknown_area_m2, 0.0);
  EXPECT_FALSE(generated.fully_inside_task);

  const Pose2 frozen_pose = PoseAt(map, {1, 1});
  const CandidateKey frozen_key = MakeCandidateKey(frozen_pose);
  ApproachCandidate frozen{
      .id = 91U,
      .identity = BoundaryApproachGoalIdentity{
          .intent_cell = {5, 1},
          .candidate_key = frozen_key,
          .candidate_kind = ApproachCandidateKind::kTranslation,
      },
      .pose = frozen_pose,
      .remaining_cost = GuidanceCost{1U, 4.0},
      .task_unknown_area_m2 = 0.0,
      .guidance_unknown_cell_count = 1U,
      .fully_inside_task = false,
      .guidance_route = {},
  };
  frozen.guidance_route = {
      Vec2{frozen_pose.x, frozen_pose.y}, map.CellCenter({3, 1}),
      map.CellCenter({5, 1})};
  EXPECT_TRUE(guidance.IsCandidateStillValid(map, raster, frozen));

  std::vector<std::int8_t> blocked_data = data;
  Set(blocked_data, geometry, 2, 1, kOccupied);
  const OccupancyGridView blocked_map = Map(geometry, blocked_data);
  const TaskRaster blocked_raster = TaskRaster::Build(blocked_map, task);
  EXPECT_FALSE(guidance.IsCandidateStillValid(blocked_map, blocked_raster,
                                              frozen));
}

TEST(BoundaryGuidance,
     OffRouteTaskUnknownCannotReplaceOccludedRouteUnknownValue) {
  const GridGeometry geometry = Geometry(9U, 6U);
  std::vector<std::int8_t> data = Filled(geometry, kFree);
  Set(data, geometry, 1, 1, kOccupied);
  Set(data, geometry, 1, 3, kOccupied);
  for (std::int32_t y = 0; y <= 4; ++y) {
    for (std::int32_t x = 4; x <= 7; ++x) {
      const bool boundary = x == 4 || x == 7 || y == 0 || y == 4;
      if (boundary) {
        Set(data, geometry, x, y, kOccupied);
      }
    }
  }
  Set(data, geometry, 4, 0, kFree);      // Keep the safe-prefix corner clear.
  Set(data, geometry, 4, 1, kFree);      // Legal boundary intent and clear ray.
  Set(data, geometry, 5, 1, kUnknown);   // Visible task gain, off the routes.
  Set(data, geometry, 7, 2, kUnknown);   // Route intent hidden by (4, 2).
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{4.0, 0.0}, {8.0, 0.0},
                                  {8.0, 5.0}, {4.0, 5.0}}));
  BoundaryGuidance guidance(
      LongFootprintPlatform(),
      SensorModel{10.0, std::numbers::pi / 2.0}, 0.05,
      GenerousLimits());

  const BoundaryGuidanceResult result =
      guidance.Build(map, raster, PoseAt(map, {1, 2}));

  EXPECT_TRUE(result.candidates.empty());
  EXPECT_EQ(result.wait_reason, ApproachWaitReason::kNoSafeCandidate);
}

TEST(BoundaryGuidance,
     KnownRouteFallsBackToLatestFullyInsideSafePrefixPose) {
  const GridGeometry geometry = Geometry(12U, 8U);
  const std::vector<std::int8_t> data = Filled(geometry, kFree);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{3.0, 1.0}, {9.0, 1.0},
                                  {9.0, 6.0}, {3.0, 6.0}}));
  BoundaryGuidance guidance(
      LargeSquarePlatform(),
      SensorModel{10.0, std::numbers::pi / 2.0}, 0.05,
      GenerousLimits());

  const BoundaryGuidanceResult result =
      guidance.Build(map, raster, PoseAt(map, {1, 3}));

  ASSERT_FALSE(result.candidates.empty());
  SafePoseValidator validator(LargeSquarePlatform(), 1000000U);
  bool found_latest_interior = false;
  for (const ApproachCandidate& candidate : result.candidates) {
    std::size_t work = 0U;
    EXPECT_TRUE(validator.IsMapFree(map, candidate.pose, work));
    EXPECT_TRUE(validator.IsTaskFree(raster, candidate.pose, work));
    EXPECT_TRUE(candidate.fully_inside_task);
    const auto cell =
        map.WorldToCell(Vec2{candidate.pose.x, candidate.pose.y});
    ASSERT_TRUE(cell.has_value());
    found_latest_interior = found_latest_interior || *cell == GridIndex{7, 3};
  }
  EXPECT_TRUE(found_latest_interior);
}

TEST(BoundaryGuidance, FullyInsideFootprintSwitchesToExploreTask) {
  const GridGeometry geometry = Geometry(5U, 5U);
  const std::vector<std::int8_t> data = Filled(geometry, kFree);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{1.0, 1.0}, {4.0, 1.0},
                                  {4.0, 4.0}, {1.0, 4.0}}));

  const BoundaryGuidanceResult result =
      Guidance().Build(map, raster, PoseAt(map, {2, 2}));

  EXPECT_EQ(result.phase, NavigationPhase::kExploreTask);
  EXPECT_EQ(result.wait_reason, ApproachWaitReason::kNone);
  EXPECT_TRUE(result.fully_inside_task);
  EXPECT_TRUE(result.intents.empty());
  EXPECT_TRUE(result.candidates.empty());
}

TEST(BoundaryGuidance, OutsideMapTaskBoundaryReturnsCoverageWait) {
  const GridGeometry geometry = Geometry(4U, 4U);
  const std::vector<std::int8_t> data = Filled(geometry, kFree);
  const OccupancyGridView map = Map(geometry, data);
  const std::vector<Vec2> vertices{{-1.0, 1.0}, {2.0, 1.0},
                                   {2.0, 3.0},  {-1.0, 3.0}};
  std::vector<Vec2> reversed(vertices.rbegin(), vertices.rend());
  const TaskRaster first =
      TaskRaster::Build(map, GridPolygon(geometry, vertices));
  const TaskRaster second =
      TaskRaster::Build(map, GridPolygon(geometry, reversed));

  const BoundaryGuidanceResult a =
      Guidance().Build(map, first, PoseAt(map, {0, 0}));
  const BoundaryGuidanceResult b =
      Guidance().Build(map, second, PoseAt(map, {0, 0}));

  EXPECT_EQ(a.phase, NavigationPhase::kApproachTask);
  EXPECT_EQ(a.wait_reason,
            ApproachWaitReason::kWaitingForTaskMapCoverage);
  EXPECT_TRUE(a.candidates.empty());
  ExpectEquivalent(a, b);
}

TEST(BoundaryGuidance, UnsafeRobotPoseReturnsSafeStartWait) {
  const GridGeometry geometry = Geometry(5U, 4U);
  std::vector<std::int8_t> data = Filled(geometry, kFree);
  Set(data, geometry, 0, 1, kOccupied);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{3.0, 1.0}, {5.0, 1.0},
                                  {5.0, 3.0}, {3.0, 3.0}}));

  const BoundaryGuidanceResult result =
      Guidance().Build(map, raster, PoseAt(map, {0, 1}));

  EXPECT_EQ(result.phase, NavigationPhase::kApproachTask);
  EXPECT_EQ(result.wait_reason,
            ApproachWaitReason::kWaitingForSafeStart);
  EXPECT_TRUE(result.intents.empty());
  EXPECT_TRUE(result.candidates.empty());
}

TEST(BoundaryGuidance, EntirelyOccupiedBoundaryReturnsNoGuidanceRoute) {
  const GridGeometry geometry = Geometry(6U, 4U);
  std::vector<std::int8_t> data = Filled(geometry, kFree);
  for (std::int32_t y = 1; y <= 2; ++y) {
    for (std::int32_t x = 4; x <= 5; ++x) {
      Set(data, geometry, x, y, kOccupied);
    }
  }
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{4.0, 1.0}, {6.0, 1.0},
                                  {6.0, 3.0}, {4.0, 3.0}}));

  const BoundaryGuidanceResult result =
      Guidance().Build(map, raster, PoseAt(map, {0, 1}));

  EXPECT_EQ(result.wait_reason, ApproachWaitReason::kNoGuidanceRoute);
  EXPECT_TRUE(result.intents.empty());
  EXPECT_TRUE(result.candidates.empty());
}

TEST(BoundaryGuidance, ExactResourceLimitsPassAndOneLessFailsClosed) {
  const GridGeometry geometry = Geometry(6U, 6U);
  const std::vector<std::int8_t> data = Filled(geometry, kFree);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{3.0, 1.0}, {6.0, 1.0},
                                  {6.0, 5.0}, {3.0, 5.0}}));
  const Pose2 robot = PoseAt(map, {0, 3});
  const BoundaryGuidanceResult measured =
      Guidance().Build(map, raster, robot);
  ASSERT_GT(measured.candidates.size(), 1U);
  ASSERT_GT(measured.consumed_work_units, 1U);

  const BoundaryGuidance::Limits exact{
      .maximum_guidance_grid_cells = 36U,
      .maximum_guidance_work_units = measured.consumed_work_units,
      .maximum_approach_candidates = measured.candidates.size(),
  };
  const BoundaryGuidanceResult repeated =
      Guidance(exact).Build(map, raster, robot);
  ExpectEquivalent(measured, repeated);

  auto short_grid = exact;
  --short_grid.maximum_guidance_grid_cells;
  EXPECT_THROW(Guidance(short_grid).Build(map, raster, robot),
               std::length_error);
  auto short_work = exact;
  --short_work.maximum_guidance_work_units;
  EXPECT_THROW(Guidance(short_work).Build(map, raster, robot),
               std::length_error);
  auto short_candidates = exact;
  --short_candidates.maximum_approach_candidates;
  EXPECT_THROW(Guidance(short_candidates).Build(map, raster, robot),
               std::length_error);
}

TEST(BoundaryGuidance,
     DuplicateActualRotationAcrossIntentsKeepsDeterministicWinningIntent) {
  const GridGeometry geometry = Geometry(7U, 3U);
  std::vector<std::int8_t> data = Filled(geometry, kOccupied);
  Set(data, geometry, 0, 1, kFree);
  Set(data, geometry, 1, 1, kUnknown);
  for (std::int32_t x = 2; x <= 4; ++x) {
    Set(data, geometry, x, 1, kFree);
  }
  Set(data, geometry, 4, 0, kFree);
  Set(data, geometry, 4, 2, kFree);
  Set(data, geometry, 5, 0, kFree);
  Set(data, geometry, 5, 2, kFree);
  Set(data, geometry, 6, 0, kFree);
  Set(data, geometry, 6, 2, kFree);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{5.0, 0.0}, {7.0, 0.0},
                                  {7.0, 3.0}, {5.0, 3.0}}));

  const BoundaryGuidanceResult result = Guidance().Build(
      map, raster, PoseAt(map, {0, 1}, std::numbers::pi / 2.0));

  ASSERT_GE(result.intents.size(), 2U);
  ASSERT_EQ(result.candidates.size(), 1U);
  const ApproachCandidate& candidate = result.candidates.front();
  EXPECT_EQ(candidate.identity.candidate_kind,
            ApproachCandidateKind::kRotation);
  EXPECT_EQ(candidate.identity.intent_cell, (GridIndex{5, 0}));
  EXPECT_EQ(candidate.identity.candidate_key,
            MakeCandidateKey(candidate.pose));
}

ApproachCandidate RankedFixture(std::uint64_t id, Pose2 pose,
                                GridIndex intent, bool fully_inside,
                                std::uint32_t remaining_unknown,
                                double remaining_length, double task_gain) {
  const CandidateKey key = MakeCandidateKey(pose);
  return ApproachCandidate{
      .id = id,
      .identity = BoundaryApproachGoalIdentity{
          .intent_cell = intent,
          .candidate_key = key,
          .candidate_kind = ApproachCandidateKind::kTranslation,
      },
      .pose = pose,
      .remaining_cost = GuidanceCost{remaining_unknown, remaining_length},
      .task_unknown_area_m2 = task_gain,
      .guidance_unknown_cell_count = remaining_unknown,
      .fully_inside_task = fully_inside,
      .guidance_route = {},
  };
}

TEST(BoundaryGuidanceRanking, CoarseOrderFreezesEverySpecifiedTieLevel) {
  BoundaryGuidanceResult result{
      .phase = NavigationPhase::kApproachTask,
      .wait_reason = ApproachWaitReason::kNone,
      .fully_inside_task = false,
      .intents = {},
      .candidates = {
          RankedFixture(0U, {2.0, 0.0, 0.0}, {5, 0}, false, 1U, 2.0, 5.0),
          RankedFixture(1U, {9.0, 0.0, 0.0}, {9, 0}, true, 9U, 9.0, 0.0),
          RankedFixture(2U, {8.0, 0.0, 0.0}, {8, 0}, false, 0U, 8.0, 0.0),
          RankedFixture(3U, {7.0, 0.0, 0.0}, {7, 0}, false, 1U, 1.0, 0.0),
          RankedFixture(4U, {6.0, 0.0, 0.0}, {6, 0}, false, 1U, 2.0, 6.0),
          RankedFixture(5U, {1.0, 0.0, 0.0}, {4, 0}, false, 1U, 2.0, 5.0),
          RankedFixture(6U, {-1.0, 0.0, 0.0}, {3, 0}, false, 1U, 2.0, 5.0),
      },
      .consumed_work_units = 0U,
  };

  const std::vector<std::size_t> order =
      Guidance().CoarseOrder(result, Pose2{0.0, 0.0, 0.0});

  EXPECT_EQ(order, (std::vector<std::size_t>{1U, 2U, 3U, 4U, 6U, 5U, 0U}));
}

TEST(BoundaryGuidanceRanking, FinalOrderUsesPlannedLengthThenHeadingAndIdentity) {
  BoundaryGuidanceResult result{
      .phase = NavigationPhase::kApproachTask,
      .wait_reason = ApproachWaitReason::kNone,
      .fully_inside_task = false,
      .intents = {},
      .candidates = {
          RankedFixture(0U, {1.0, 0.0, 0.0}, {3, 0}, false, 1U, 2.0, 5.0),
          RankedFixture(1U, {0.0, 1.0, 1.0}, {2, 0}, false, 1U, 2.0, 5.0),
          RankedFixture(2U, {0.0, -1.0, 0.5}, {1, 0}, false, 1U, 2.0, 5.0),
          RankedFixture(3U, {-1.0, 0.0, 0.5}, {0, 0}, false, 1U, 2.0, 5.0),
      },
      .consumed_work_units = 0U,
  };
  const std::vector<PlannedCandidate> planned{
      {0U, 0.5}, {1U, 0.25}, {2U, 0.25}, {3U, 0.25}};

  const std::vector<std::size_t> order =
      Guidance().FinalOrder(result, planned, Pose2{0.0, 0.0, 0.0});

  EXPECT_EQ(order, (std::vector<std::size_t>{3U, 2U, 1U, 0U}));
}

TEST(BoundaryGuidanceRanking, RejectsNonfiniteCandidateRobotAndPlanMetrics) {
  BoundaryGuidanceResult result{
      .phase = NavigationPhase::kApproachTask,
      .wait_reason = ApproachWaitReason::kNone,
      .fully_inside_task = false,
      .intents = {},
      .candidates = {RankedFixture(0U, {1.0, 0.0, 0.0}, {3, 0}, false,
                                   1U, 2.0, 5.0)},
      .consumed_work_units = 0U,
  };
  BoundaryGuidance guidance = Guidance();

  EXPECT_THROW(guidance.CoarseOrder(
                   result, Pose2{std::numeric_limits<double>::quiet_NaN(),
                                 0.0, 0.0}),
               std::invalid_argument);
  result.candidates.front().remaining_cost.path_length_m =
      std::numeric_limits<double>::infinity();
  EXPECT_THROW(guidance.CoarseOrder(result, Pose2{0.0, 0.0, 0.0}),
               std::invalid_argument);
  result.candidates.front().remaining_cost.path_length_m = 2.0;
  EXPECT_THROW(
      guidance.FinalOrder(
          result,
          std::vector<PlannedCandidate>{
              {0U, std::numeric_limits<double>::infinity()}},
          Pose2{0.0, 0.0, 0.0}),
      std::invalid_argument);
}

TEST(BoundaryGuidanceValidity,
     SafePrefixExtensionDoesNotInvalidateStillUsefulFrozenCandidate) {
  const GridGeometry geometry = Geometry(7U, 3U);
  std::vector<std::int8_t> initial_data = Filled(geometry, kOccupied);
  Set(initial_data, geometry, 0, 1, kFree);
  Set(initial_data, geometry, 1, 1, kUnknown);
  Set(initial_data, geometry, 2, 1, kUnknown);
  Set(initial_data, geometry, 3, 1, kFree);
  Set(initial_data, geometry, 4, 1, kFree);
  Set(initial_data, geometry, 5, 1, kUnknown);
  const OccupancyGridView initial_map = Map(geometry, initial_data);
  const Polygon2 task = GridPolygon(
      geometry, {{5.0, 1.0}, {6.0, 1.0}, {6.0, 2.0}, {5.0, 2.0}});
  const TaskRaster initial_raster = TaskRaster::Build(initial_map, task);
  const Pose2 robot =
      PoseAt(initial_map, {0, 1}, std::numbers::pi / 2.0);
  BoundaryGuidance guidance = Guidance();
  const BoundaryGuidanceResult initial =
      guidance.Build(initial_map, initial_raster, robot);
  ASSERT_EQ(initial.candidates.size(), 1U);
  const ApproachCandidate frozen = initial.candidates.front();
  ASSERT_EQ(frozen.identity.candidate_kind,
            ApproachCandidateKind::kRotation);

  std::vector<std::int8_t> latest_data = initial_data;
  Set(latest_data, geometry, 1, 1, kFree);
  const OccupancyGridView latest_map = Map(geometry, latest_data);
  const TaskRaster latest_raster = TaskRaster::Build(latest_map, task);
  const BoundaryGuidanceResult rebuilt =
      guidance.Build(latest_map, latest_raster, robot);
  ASSERT_FALSE(rebuilt.candidates.empty());
  EXPECT_EQ(std::count_if(
                rebuilt.candidates.begin(), rebuilt.candidates.end(),
                [&frozen](const ApproachCandidate& candidate) {
                  return candidate.identity.candidate_key ==
                             frozen.identity.candidate_key &&
                         candidate.identity.candidate_kind ==
                             frozen.identity.candidate_kind;
                }),
            0);

  EXPECT_TRUE(guidance.IsCandidateStillValid(latest_map, latest_raster,
                                             frozen));
}

TEST(BoundaryGuidanceValidity, RejectsUnsafePoseIllegalIntentAndLostRouteValue) {
  const GridGeometry geometry = Geometry(7U, 3U);
  std::vector<std::int8_t> data = Filled(geometry, kOccupied);
  Set(data, geometry, 0, 1, kFree);
  Set(data, geometry, 1, 1, kUnknown);
  Set(data, geometry, 2, 1, kUnknown);
  Set(data, geometry, 3, 1, kFree);
  Set(data, geometry, 4, 1, kFree);
  Set(data, geometry, 5, 1, kUnknown);
  const OccupancyGridView map = Map(geometry, data);
  const Polygon2 task = GridPolygon(
      geometry, {{5.0, 1.0}, {6.0, 1.0}, {6.0, 2.0}, {5.0, 2.0}});
  const TaskRaster raster = TaskRaster::Build(map, task);
  BoundaryGuidance guidance = Guidance();
  const BoundaryGuidanceResult built = guidance.Build(
      map, raster, PoseAt(map, {0, 1}, std::numbers::pi / 2.0));
  ASSERT_EQ(built.candidates.size(), 1U);
  const ApproachCandidate frozen = built.candidates.front();

  std::vector<std::int8_t> unsafe_data = data;
  Set(unsafe_data, geometry, 0, 1, kOccupied);
  const OccupancyGridView unsafe_map = Map(geometry, unsafe_data);
  const TaskRaster unsafe_raster = TaskRaster::Build(unsafe_map, task);
  EXPECT_FALSE(guidance.IsCandidateStillValid(unsafe_map, unsafe_raster,
                                              frozen));

  std::vector<std::int8_t> illegal_intent_data = data;
  Set(illegal_intent_data, geometry, 5, 1, kOccupied);
  const OccupancyGridView illegal_intent_map =
      Map(geometry, illegal_intent_data);
  const TaskRaster illegal_intent_raster =
      TaskRaster::Build(illegal_intent_map, task);
  EXPECT_FALSE(guidance.IsCandidateStillValid(
      illegal_intent_map, illegal_intent_raster, frozen));

  std::vector<std::int8_t> known_data = data;
  Set(known_data, geometry, 1, 1, kFree);
  Set(known_data, geometry, 2, 1, kFree);
  Set(known_data, geometry, 5, 1, kFree);
  const OccupancyGridView known_map = Map(geometry, known_data);
  const TaskRaster known_raster = TaskRaster::Build(known_map, task);
  EXPECT_FALSE(guidance.IsCandidateStillValid(known_map, known_raster,
                                              frozen));
}

TEST(BoundaryGuidanceValidity, FullyInsideCandidateNeedsNoRemainingUnknown) {
  const GridGeometry geometry = Geometry(5U, 5U);
  const std::vector<std::int8_t> data = Filled(geometry, kFree);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{1.0, 1.0}, {4.0, 1.0},
                                  {4.0, 4.0}, {1.0, 4.0}}));
  const Pose2 pose = PoseAt(map, {2, 2});
  const ApproachCandidate frozen = RankedFixture(
      7U, pose, {1, 2}, true, 0U, 0.0, 0.0);

  EXPECT_TRUE(Guidance().IsCandidateStillValid(map, raster, frozen));
}

TEST(BoundaryGuidanceValidation, RejectsInvalidConstructorAndBuildInputs) {
  auto limits = GenerousLimits();
  limits.maximum_guidance_grid_cells = 0U;
  EXPECT_THROW(Guidance(limits), std::invalid_argument);
  EXPECT_THROW(Guidance(GenerousLimits(), SensorModel{0.0, 1.0}),
               std::invalid_argument);
  EXPECT_THROW(Guidance(GenerousLimits(), SensorModel{1.0, 1.0},
                        std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);

  const GridGeometry geometry = Geometry(4U, 4U);
  const std::vector<std::int8_t> data = Filled(geometry, kFree);
  const OccupancyGridView map = Map(geometry, data);
  const TaskRaster raster = TaskRaster::Build(
      map, GridPolygon(geometry, {{2.0, 1.0}, {4.0, 1.0},
                                  {4.0, 3.0}, {2.0, 3.0}}));
  EXPECT_THROW(Guidance().Build(
                   map, raster,
                   Pose2{std::numeric_limits<double>::infinity(), 0.0, 0.0}),
               std::invalid_argument);
}

}  // namespace
}  // namespace lunar::pure_exploration
