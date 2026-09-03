#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lunar_pure_exploration_core/failure_memory.hpp"

namespace lunar::pure_exploration {
namespace {

constexpr std::int8_t kFree = 0;
constexpr std::int8_t kOccupied = 100;
constexpr std::int8_t kUnknown = -1;

std::vector<std::int8_t> UniformData(const GridGeometry& geometry,
                                     std::int8_t value = kFree) {
  return std::vector<std::int8_t>(
      static_cast<std::size_t>(geometry.width) * geometry.height, value);
}

void SetCell(std::vector<std::int8_t>& data, const GridGeometry& geometry,
             GridIndex index, std::int8_t value) {
  ASSERT_GE(index.x, 0);
  ASSERT_GE(index.y, 0);
  ASSERT_LT(static_cast<std::uint32_t>(index.x), geometry.width);
  ASSERT_LT(static_cast<std::uint32_t>(index.y), geometry.height);
  data[static_cast<std::size_t>(index.y) * geometry.width +
       static_cast<std::size_t>(index.x)] = value;
}

TaskRaster BuildRaster(const GridGeometry& geometry,
                       std::vector<std::int8_t> data,
                       const std::vector<Vec2>& polygon_grid) {
  OccupancyGridView map(geometry, data, 50);
  Polygon2 polygon;
  for (const Vec2 grid_vertex : polygon_grid) {
    const auto world = OccupancyGridView::GridToWorld(geometry, grid_vertex);
    if (!world.has_value()) {
      throw std::overflow_error("test polygon transform overflow");
    }
    polygon.vertices.push_back(*world);
  }
  return TaskRaster::Build(map, polygon, TaskRaster::Limits{1048576U});
}

TaskRaster RectangularRaster(
    const GridGeometry& geometry, std::vector<std::int8_t> data,
    double minimum_x = -2.0, double minimum_y = -2.0,
    double maximum_x = 9.0, double maximum_y = 9.0) {
  return BuildRaster(
      geometry, std::move(data),
      {{minimum_x, minimum_y}, {maximum_x, minimum_y},
       {maximum_x, maximum_y}, {minimum_x, maximum_y}});
}

CandidateView CandidateAt(const TaskRaster& raster, Vec2 grid_center,
                          CandidateKey key = {1, 2, 3},
                          std::uint64_t display_id = 7U) {
  const auto world = raster.GridToWorld(grid_center);
  if (!world.has_value()) {
    throw std::overflow_error("test candidate transform overflow");
  }
  return CandidateView{display_id, 11U, 0U, key,
                       Pose2{world->x, world->y, 0.0}, 0.0};
}

FailureMemoryLimits Limits(std::size_t entries = 8U,
                           std::size_t per_entry = 256U,
                           std::size_t total = 2048U) {
  return {entries, per_entry, total};
}

constexpr PersistentFailureReason kPersistent =
    PersistentFailureReason::kExecutionReplansExhausted;

TEST(FailureMemory, NavigationFailuresUseTheExistingEvidencePatchLifetime) {
  const GridGeometry geometry{8U, 8U, 1.0, 0.0, 0.0, 0.0};
  const auto base = RectangularRaster(geometry, UniformData(geometry));
  const auto no_path_candidate =
      CandidateAt(base, {2.5, 2.5}, {10, 20, 30});
  const auto timeout_candidate =
      CandidateAt(base, {5.5, 5.5}, {40, 50, 60});

  FailureMemory memory(1.0, Limits());
  memory.BeginTask("navigation-failures");
  memory.RecordPersistentFailure(
      no_path_candidate, PersistentFailureReason::kNavigationNoPath, base);
  memory.RecordPersistentFailure(
      timeout_candidate, PersistentFailureReason::kNavigationTimeout, base);

  EXPECT_TRUE(memory.IsSuppressed(no_path_candidate, base));
  EXPECT_TRUE(memory.IsSuppressed(timeout_candidate, base));

  auto changed_data = UniformData(geometry);
  SetCell(changed_data, geometry, {2, 2}, kOccupied);
  const auto changed =
      RectangularRaster(geometry, std::move(changed_data));
  EXPECT_FALSE(memory.IsSuppressed(no_path_candidate, changed));
  EXPECT_TRUE(memory.IsSuppressed(timeout_candidate, changed));
}

TEST(FailureMemory, ValidatesLimitsReasonAndAcceptedStartScope) {
  EXPECT_THROW(FailureMemory(0.0, Limits()), std::invalid_argument);
  EXPECT_THROW(FailureMemory(-1.0, Limits()), std::invalid_argument);
  EXPECT_THROW(FailureMemory(std::numeric_limits<double>::infinity(), Limits()),
               std::invalid_argument);
  EXPECT_THROW(FailureMemory(1.0, Limits(0U, 1U, 1U)),
               std::invalid_argument);
  EXPECT_THROW(FailureMemory(1.0, Limits(1U, 0U, 1U)),
               std::invalid_argument);
  EXPECT_THROW(FailureMemory(1.0, Limits(1U, 1U, 0U)),
               std::invalid_argument);

  const GridGeometry geometry{6U, 6U, 1.0, 0.0, 0.0, 0.0};
  const auto raster = RectangularRaster(geometry, UniformData(geometry));
  const auto candidate = CandidateAt(raster, {2.5, 2.5});
  FailureMemory memory(2.0, Limits());
  memory.BeginTask("same-task");
  memory.RecordPersistentFailure(candidate, kPersistent, raster);
  ASSERT_EQ(memory.size(), 1U);

  EXPECT_THROW(memory.BeginTask(""), std::invalid_argument);
  EXPECT_EQ(memory.size(), 1U);
  memory.BeginTask("same-task");
  EXPECT_EQ(memory.size(), 0U);

  memory.RecordPersistentFailure(candidate, kPersistent, raster);
  EXPECT_THROW(memory.RecordPersistentFailure(
                   candidate, static_cast<PersistentFailureReason>(255U),
                   raster),
               std::invalid_argument);
  EXPECT_EQ(memory.size(), 1U);
}

TEST(FailureMemory, KeysByCompleteCandidateKeyAndAtomicallyReplaces) {
  const GridGeometry geometry{8U, 8U, 1.0, 0.0, 0.0, 0.0};
  const auto raster = RectangularRaster(geometry, UniformData(geometry));
  const auto first = CandidateAt(raster, {2.5, 2.5}, {10, 20, 30}, 99U);
  const auto same_display_different_key =
      CandidateAt(raster, {5.5, 5.5}, {10, 20, 31}, 99U);
  FailureMemory memory(2.0, Limits());
  memory.BeginTask("keys");
  memory.RecordPersistentFailure(first, kPersistent, raster);

  EXPECT_FALSE(memory.IsSuppressed(same_display_different_key, raster));
  EXPECT_EQ(memory.size(), 1U);
  memory.RecordPersistentFailure(same_display_different_key, kPersistent,
                                 raster);
  EXPECT_EQ(memory.size(), 2U);

  auto replacement = same_display_different_key;
  replacement.id = 123456U;
  replacement.pose.x = std::nextafter(
      replacement.pose.x, std::numeric_limits<double>::infinity());
  memory.RecordPersistentFailure(replacement, kPersistent, raster);
  EXPECT_EQ(memory.size(), 2U);
  EXPECT_TRUE(memory.IsSuppressed(replacement, raster));

  auto tangent_data = UniformData(geometry);
  SetCell(tangent_data, geometry, {3, 5}, kOccupied);
  const auto first_center_tangent_changed =
      RectangularRaster(geometry, std::move(tangent_data));
  EXPECT_FALSE(memory.IsSuppressed(replacement, first_center_tangent_changed));

  auto moved_query = replacement;
  moved_query.pose = CandidateAt(raster, {0.5, 0.5}).pose;
  memory.RecordPersistentFailure(replacement, kPersistent, raster);
  EXPECT_TRUE(memory.IsSuppressed(moved_query, raster));
}

TEST(FailureMemory, DuplicateKeyPreservesTheFirstRecordedRadius) {
  const GridGeometry first_geometry{16U, 16U, 1.0, 0.0, 0.0, 0.0};
  const auto first_raster = RectangularRaster(
      first_geometry, UniformData(first_geometry), -2.0, -2.0, 17.0, 17.0);
  const auto first = CandidateAt(first_raster, {11.0, 11.0}, {8, 9, 10});

  FailureMemory memory(0.5, Limits(1U, 25U, 13U));
  memory.BeginTask("saved-radius");
  memory.RecordPersistentFailure(first, kPersistent, first_raster);

  const GridGeometry second_geometry{12U, 12U, 2.0, 0.0, 0.0, 0.0};
  const auto second_raster = RectangularRaster(
      second_geometry, UniformData(second_geometry), -2.0, -2.0, 13.0, 13.0);
  const auto same_key = CandidateAt(second_raster, {5.5, 5.5}, first.key);
  memory.RecordPersistentFailure(same_key, kPersistent, second_raster);

  auto changed_data = UniformData(second_geometry);
  SetCell(changed_data, second_geometry, {3, 5}, kOccupied);
  const auto recomputed_radius_only_change = RectangularRaster(
      second_geometry, std::move(changed_data), -2.0, -2.0, 13.0, 13.0);
  EXPECT_TRUE(
      memory.IsSuppressed(same_key, recomputed_radius_only_change));
}

TEST(FailureMemory, UsesSharedRotatedGeometryAndFrozenPointTwoRadius) {
  const GridGeometry rotated_geometry{8U, 8U, 0.2, 10.0, -4.0, 0.7};
  const auto rotated =
      RectangularRaster(rotated_geometry, UniformData(rotated_geometry));
  const auto rotated_candidate = CandidateAt(rotated, {2.5, 2.5});
  FailureMemory shared_transform(0.4, Limits());
  shared_transform.BeginTask("rotated");
  shared_transform.RecordPersistentFailure(rotated_candidate, kPersistent,
                                           rotated);
  EXPECT_TRUE(shared_transform.IsSuppressed(rotated_candidate, rotated));

  auto rotated_near_data = UniformData(rotated_geometry);
  SetCell(rotated_near_data, rotated_geometry, {1, 2}, kOccupied);
  const auto rotated_near_changed =
      RectangularRaster(rotated_geometry, std::move(rotated_near_data));
  EXPECT_FALSE(shared_transform.IsSuppressed(rotated_candidate,
                                             rotated_near_changed));

  FailureMemory far_only(0.4, Limits());
  far_only.BeginTask("rotated-far");
  far_only.RecordPersistentFailure(rotated_candidate, kPersistent, rotated);
  auto rotated_far_data = UniformData(rotated_geometry);
  SetCell(rotated_far_data, rotated_geometry, {7, 7}, kOccupied);
  const auto rotated_far_changed =
      RectangularRaster(rotated_geometry, std::move(rotated_far_data));
  EXPECT_TRUE(far_only.IsSuppressed(rotated_candidate, rotated_far_changed));

  const GridGeometry exact_geometry{8U, 8U, 0.2, 0.0, 0.0, 0.0};
  const auto exact =
      RectangularRaster(exact_geometry, UniformData(exact_geometry));
  const auto exact_candidate = CandidateAt(exact, {2.5, 2.5});
  FailureMemory exact_limit(0.4, Limits(1U, 25U, 13U));
  exact_limit.BeginTask("point-two");
  exact_limit.RecordPersistentFailure(exact_candidate, kPersistent, exact);
  EXPECT_TRUE(exact_limit.IsSuppressed(exact_candidate, exact));

  FailureMemory one_less(0.4, Limits(1U, 24U, 13U));
  one_less.BeginTask("point-two");
  EXPECT_THROW(one_less.RecordPersistentFailure(exact_candidate, kPersistent,
                                                exact),
               std::length_error);
  EXPECT_EQ(one_less.size(), 0U);
}

TEST(FailureMemory, IncludesClosedCircleTangencyAndExcludesNextafterOutside) {
  const GridGeometry geometry{7U, 7U, 1.0, 0.0, 0.0, 0.0};
  const auto base = RectangularRaster(geometry, UniformData(geometry));
  auto changed_data = UniformData(geometry);
  SetCell(changed_data, geometry, {0, 2}, kOccupied);
  const auto tangent_changed =
      RectangularRaster(geometry, std::move(changed_data));

  const auto tangent = CandidateAt(base, {2.5, 2.5}, {1, 1, 1});
  FailureMemory tangent_memory(1.0, Limits());
  tangent_memory.BeginTask("tangent");
  tangent_memory.RecordPersistentFailure(tangent, kPersistent, base);
  EXPECT_FALSE(tangent_memory.IsSuppressed(tangent, tangent_changed));

  const double just_inside = std::nextafter(2.5, 3.0);
  const auto outside = CandidateAt(base, {just_inside, 2.5}, {2, 2, 2});
  FailureMemory outside_memory(2.0, Limits());
  outside_memory.BeginTask("outside");
  outside_memory.RecordPersistentFailure(outside, kPersistent, base);
  EXPECT_TRUE(outside_memory.IsSuppressed(outside, tangent_changed));

  auto platform_changed_data = UniformData(geometry);
  SetCell(platform_changed_data, geometry, {0, 3}, kOccupied);
  const auto platform_changed =
      RectangularRaster(geometry, std::move(platform_changed_data));
  const auto platform_dominates =
      CandidateAt(base, {3.5, 3.5}, {3, 3, 3});
  FailureMemory platform_memory(3.0, Limits());
  platform_memory.BeginTask("platform-radius");
  platform_memory.RecordPersistentFailure(platform_dominates, kPersistent,
                                          base);
  EXPECT_FALSE(
      platform_memory.IsSuppressed(platform_dominates, platform_changed));
}

TEST(FailureMemory, FreezesDoubleThenPromoteRadiusArithmetic) {
  const GridGeometry geometry{32U, 12U, 0.1, 0.0, 0.0, 0.0};
  const auto base = RectangularRaster(geometry, UniformData(geometry),
                                      0.0, 0.0, 32.0, 12.0);
  const auto candidate = CandidateAt(base, {19.5, 5.5}, {4, 5, 6});
  auto changed_data = UniformData(geometry);
  SetCell(changed_data, geometry, {10, 5}, kOccupied);
  const auto tangent_changed = RectangularRaster(
      geometry, std::move(changed_data), 0.0, 0.0, 32.0, 12.0);

  FailureMemory memory(0.9, Limits(1U, 361U, 253U));
  memory.BeginTask("double-then-promote");
  memory.RecordPersistentFailure(candidate, kPersistent, base);
  EXPECT_FALSE(memory.IsSuppressed(candidate, tangent_changed));
}

TEST(FailureMemory, SavesFiveStatesAndOnlyInvalidatesOnNearChanges) {
  const GridGeometry geometry{6U, 6U, 1.0, 0.0, 0.0, 0.0};
  auto data = UniformData(geometry);
  SetCell(data, geometry, {0, 0}, kUnknown);
  SetCell(data, geometry, {1, 0}, kFree);
  SetCell(data, geometry, {2, 0}, kOccupied);
  const std::vector<Vec2> polygon{{-2.0, 0.0}, {6.0, 0.0},
                                  {6.0, 6.0}, {-2.0, 6.0}};
  const auto base = BuildRaster(geometry, data, polygon);
  const auto candidate = CandidateAt(base, {0.5, 0.5});

  FailureMemory unchanged(2.0, Limits());
  unchanged.BeginTask("five-states");
  unchanged.RecordPersistentFailure(candidate, kPersistent, base);
  EXPECT_TRUE(unchanged.IsSuppressed(candidate, base));

  auto distant_data = data;
  SetCell(distant_data, geometry, {5, 5}, kOccupied);
  const auto distant = BuildRaster(geometry, distant_data, polygon);
  EXPECT_TRUE(unchanged.IsSuppressed(candidate, distant));

  for (const GridIndex index : {GridIndex{0, 0}, GridIndex{1, 0},
                                GridIndex{2, 0}}) {
    auto near_data = data;
    SetCell(near_data, geometry, index,
            near_data[static_cast<std::size_t>(index.y) * geometry.width +
                              static_cast<std::size_t>(index.x)] == kOccupied
                ? kFree
                : kOccupied);
    const auto near = BuildRaster(geometry, near_data, polygon);
    FailureMemory memory(2.0, Limits());
    memory.BeginTask("near");
    memory.RecordPersistentFailure(candidate, kPersistent, base);
    EXPECT_FALSE(memory.IsSuppressed(candidate, near));
    EXPECT_EQ(memory.size(), 0U);
  }

  const std::vector<Vec2> expanded_polygon{{-2.0, -2.0}, {6.0, -2.0},
                                           {6.0, 6.0}, {-2.0, 6.0}};
  const auto outside_task_changed =
      BuildRaster(geometry, data, expanded_polygon);
  FailureMemory task_state(2.0, Limits());
  task_state.BeginTask("outside-task");
  task_state.RecordPersistentFailure(candidate, kPersistent, base);
  EXPECT_FALSE(task_state.IsSuppressed(candidate, outside_task_changed));
}

TEST(FailureMemory, ErasesEntryWhenAnyGeometryFieldChanges) {
  const GridGeometry base_geometry{6U, 6U, 1.0, 0.0, 0.0, 0.0};
  const auto base = RectangularRaster(base_geometry,
                                      UniformData(base_geometry));
  const auto candidate = CandidateAt(base, {2.5, 2.5});
  const std::vector<GridGeometry> changed_geometries{
      {7U, 6U, 1.0, 0.0, 0.0, 0.0}, {6U, 7U, 1.0, 0.0, 0.0, 0.0},
      {6U, 6U, 0.5, 0.0, 0.0, 0.0}, {6U, 6U, 1.0, 0.25, 0.0, 0.0},
      {6U, 6U, 1.0, 0.0, -0.25, 0.0}, {6U, 6U, 1.0, 0.0, 0.0, 0.25}};

  for (const GridGeometry geometry : changed_geometries) {
    const auto changed = RectangularRaster(geometry, UniformData(geometry));
    FailureMemory memory(2.0, Limits());
    memory.BeginTask("geometry");
    memory.RecordPersistentFailure(candidate, kPersistent, base);
    EXPECT_FALSE(memory.IsSuppressed(candidate, changed));
    EXPECT_EQ(memory.size(), 0U);
  }
}

TEST(FailureMemory, EnforcesExactEntryAndTotalRetainedLimitsWithoutEviction) {
  const GridGeometry geometry{12U, 8U, 1.0, 0.0, 0.0, 0.0};
  const auto raster = RectangularRaster(geometry, UniformData(geometry),
                                        -2.0, -2.0, 13.0, 9.0);
  const auto first = CandidateAt(raster, {2.5, 2.5}, {1, 0, 0});
  const auto second = CandidateAt(raster, {8.5, 2.5}, {2, 0, 0});

  FailureMemory exact(2.0, Limits(2U, 25U, 26U));
  exact.BeginTask("exact");
  exact.RecordPersistentFailure(first, kPersistent, raster);
  exact.RecordPersistentFailure(second, kPersistent, raster);
  EXPECT_EQ(exact.size(), 2U);

  const auto third = CandidateAt(raster, {5.5, 5.5}, {3, 0, 0});
  EXPECT_THROW(exact.RecordPersistentFailure(third, kPersistent, raster),
               std::length_error);
  EXPECT_EQ(exact.size(), 2U);
  EXPECT_TRUE(exact.IsSuppressed(first, raster));
  EXPECT_TRUE(exact.IsSuppressed(second, raster));

  FailureMemory total_over(2.0, Limits(2U, 25U, 25U));
  total_over.BeginTask("total");
  total_over.RecordPersistentFailure(first, kPersistent, raster);
  EXPECT_THROW(total_over.RecordPersistentFailure(second, kPersistent, raster),
               std::length_error);
  EXPECT_EQ(total_over.size(), 1U);
  EXPECT_TRUE(total_over.IsSuppressed(first, raster));
  EXPECT_FALSE(total_over.IsSuppressed(second, raster));
}

TEST(FailureMemory, CountsOutsideMapCellsAgainstTheTotalLimit) {
  const GridGeometry geometry{12U, 8U, 1.0, 0.0, 0.0, 0.0};
  const auto raster = RectangularRaster(geometry, UniformData(geometry),
                                        -2.0, -2.0, 13.0, 9.0);
  const auto boundary = CandidateAt(raster, {0.5, 2.5}, {1, 0, 0});
  const auto interior = CandidateAt(raster, {8.5, 2.5}, {2, 0, 0});

  FailureMemory memory(2.0, Limits(2U, 25U, 25U));
  memory.BeginTask("outside-map-count");
  memory.RecordPersistentFailure(boundary, kPersistent, raster);
  EXPECT_THROW(memory.RecordPersistentFailure(interior, kPersistent, raster),
               std::length_error);
  EXPECT_EQ(memory.size(), 1U);
  EXPECT_TRUE(memory.IsSuppressed(boundary, raster));
}

TEST(FailureMemory, RejectsUncutWorkBeforeInt32ConversionAndTinyResolution) {
  const GridGeometry geometry{6U, 6U, 1.0, 0.0, 0.0, 0.0};
  const auto raster = RectangularRaster(geometry, UniformData(geometry));
  auto beyond = CandidateAt(raster, {2.5, 2.5}, {9, 9, 9});
  beyond.pose.x = static_cast<double>(std::numeric_limits<std::int32_t>::max()) +
                  0.5;
  beyond.pose.y = 0.5;

  FailureMemory work_first(2.0, Limits(1U, 24U, 100U));
  work_first.BeginTask("work-first");
  EXPECT_THROW(work_first.RecordPersistentFailure(beyond, kPersistent, raster),
               std::length_error);
  EXPECT_EQ(work_first.size(), 0U);

  FailureMemory index_second(2.0, Limits(1U, 25U, 100U));
  index_second.BeginTask("index-second");
  EXPECT_THROW(index_second.RecordPersistentFailure(beyond, kPersistent, raster),
               std::overflow_error);
  EXPECT_EQ(index_second.size(), 0U);

  const GridGeometry tiny_geometry{100U, 100U, 1.0e-6, 0.0, 0.0, 0.0};
  const auto tiny = RectangularRaster(
      tiny_geometry, UniformData(tiny_geometry), 0.0, 0.0, 100.0, 100.0);
  const auto tiny_candidate = CandidateAt(tiny, {50.5, 50.5});
  FailureMemory attacked(1.0, Limits(1U, 1000000U, 1000000U));
  attacked.BeginTask("tiny");
  EXPECT_THROW(attacked.RecordPersistentFailure(tiny_candidate, kPersistent,
                                                tiny),
               std::length_error);
  EXPECT_EQ(attacked.size(), 0U);
}

TEST(FailureMemory, FailedReplacementPreservesOriginalEntryAndTotal) {
  const GridGeometry base_geometry{8U, 8U, 1.0, 0.0, 0.0, 0.0};
  const auto base = RectangularRaster(base_geometry,
                                      UniformData(base_geometry));
  const auto original = CandidateAt(base, {2.5, 2.5}, {7, 8, 9});

  FailureMemory build_failure(2.0, Limits(1U, 25U, 13U));
  build_failure.BeginTask("build-failure");
  build_failure.RecordPersistentFailure(original, kPersistent, base);
  const GridGeometry overflowing_geometry{
      8U, 8U, 1.0,
      -static_cast<double>(std::numeric_limits<std::int32_t>::max()),
      0.0, 0.0};
  const auto overflowing_raster = RectangularRaster(
      overflowing_geometry, UniformData(overflowing_geometry));
  EXPECT_THROW(build_failure.RecordPersistentFailure(original, kPersistent,
                                                     overflowing_raster),
               std::overflow_error);
  EXPECT_EQ(build_failure.size(), 1U);
  EXPECT_TRUE(build_failure.IsSuppressed(original, base));
  EXPECT_NO_THROW(
      build_failure.RecordPersistentFailure(original, kPersistent, base));

  const GridGeometry dense_geometry{16U, 16U, 0.5, 0.0, 0.0, 0.0};
  const auto dense = RectangularRaster(dense_geometry,
                                       UniformData(dense_geometry), -2.0,
                                       -2.0, 17.0, 17.0);
  auto larger_patch = CandidateAt(dense, {4.5, 4.5}, original.key);
  FailureMemory total_failure(2.0, Limits(1U, 81U, 13U));
  total_failure.BeginTask("total-failure");
  total_failure.RecordPersistentFailure(original, kPersistent, base);
  EXPECT_THROW(total_failure.RecordPersistentFailure(larger_patch, kPersistent,
                                                     dense),
               std::length_error);
  EXPECT_EQ(total_failure.size(), 1U);
  EXPECT_TRUE(total_failure.IsSuppressed(original, base));
  EXPECT_NO_THROW(
      total_failure.RecordPersistentFailure(original, kPersistent, base));
}

TEST(FailureMemory, RejectsNonfiniteFirstWorldCenterWithoutMutation) {
  const GridGeometry geometry{6U, 6U, 1.0, 0.0, 0.0, 0.0};
  const auto raster = RectangularRaster(geometry, UniformData(geometry));
  auto candidate = CandidateAt(raster, {2.5, 2.5});
  FailureMemory memory(2.0, Limits());
  memory.BeginTask("finite");
  candidate.pose.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(memory.RecordPersistentFailure(candidate, kPersistent, raster),
               std::invalid_argument);
  EXPECT_EQ(memory.size(), 0U);
}

}  // namespace
}  // namespace lunar::pure_exploration
