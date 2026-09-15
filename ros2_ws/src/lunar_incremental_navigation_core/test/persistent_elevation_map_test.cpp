#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/elevation_map.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] GridGeometry Geometry(const std::size_t width,
                                    const std::size_t height,
                                    const double resolution_m,
                                    const Vec3 origin_m = {}) {
  return GridGeometry{
      .frame_id = "odom",
      .width = width,
      .height = height,
      .resolution_m = resolution_m,
      .origin_m = origin_m,
  };
}

[[nodiscard]] RigidTransform MapFromSource(
    const Vec3 translation_m = {}, const double yaw_rad = 0.0) {
  return RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 123},
      .translation_m = translation_m,
      .rotation = Quaternion{.w = std::cos(yaw_rad / 2.0),
                             .z = std::sin(yaw_rad / 2.0)},
  };
}

[[nodiscard]] ElevationUpdateResult Apply(
    PersistentElevationMap& map, const GridGeometry& geometry,
    const std::vector<float>& elevation_m,
    const RigidTransform& map_from_source = MapFromSource()) {
  return map.Apply(ElevationEvidence{
      .geometry = geometry,
      .elevation_m = std::span<const float>(elevation_m),
      .map_from_source = map_from_source,
  });
}

TEST(PersistentElevationMap, InsertsFiniteValuesInCanonicalMap) {
  PersistentElevationMap map;
  const ElevationUpdateResult result =
      Apply(map, Geometry(2U, 1U, 1.0), {1.0F, 2.0F},
            MapFromSource(Vec3{.x = 10.0, .y = -2.0, .z = 3.0}));
  const auto snapshot = map.Snapshot();

  ASSERT_EQ(result.status, ElevationUpdateResult::Status::kApplied);
  EXPECT_EQ(result.raw_elevation_revision, 1U);
  EXPECT_EQ(result.updated_cells, 2U);
  ASSERT_TRUE(snapshot);
  EXPECT_TRUE(snapshot->valid());
  EXPECT_EQ(snapshot->geometry().frame_id(), "map");
  EXPECT_DOUBLE_EQ(snapshot->geometry().resolution_m(), 1.0);
  EXPECT_EQ(snapshot->geometry().min_inclusive(),
            (GridIndex{.x = 0, .y = 0}));
  EXPECT_EQ(snapshot->geometry().max_exclusive(),
            (GridIndex{.x = 2, .y = 1}));
  EXPECT_EQ(snapshot->geometry().width(), 2U);
  EXPECT_EQ(snapshot->geometry().height(), 1U);
  EXPECT_EQ(snapshot->geometry().CellCount(), 2U);
  EXPECT_EQ(snapshot->raw_elevation_revision(), 1U);
  EXPECT_EQ(snapshot->ElevationAtWorld(10.5, -1.5),
            std::optional<float>(4.0F));
  EXPECT_EQ(snapshot->ElevationAtWorld(11.5, -1.5),
            std::optional<float>(5.0F));
}

TEST(PersistentElevationMap, UnknownValuesDoNotEraseFiniteHistory) {
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(3U, 1U, 1.0), {1.0F, 2.0F, 3.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto before = map.Snapshot();
  ASSERT_TRUE(before);

  const ElevationUpdateResult result = Apply(
      map, Geometry(3U, 1U, 1.0),
      {std::numeric_limits<float>::quiet_NaN(),
       std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity()});
  const auto after = map.Snapshot();

  EXPECT_EQ(result.status, ElevationUpdateResult::Status::kDuplicate);
  EXPECT_EQ(result.raw_elevation_revision, before->raw_elevation_revision());
  EXPECT_EQ(result.updated_cells, 0U);
  EXPECT_TRUE(result.dirty_tiles.empty());
  ASSERT_TRUE(after);
  EXPECT_EQ(after->ElevationAt(GridIndex{.x = 0, .y = 0}),
            std::optional<float>(1.0F));
  EXPECT_EQ(after->ElevationAt(GridIndex{.x = 1, .y = 0}),
            std::optional<float>(2.0F));
  EXPECT_EQ(after->ElevationAt(GridIndex{.x = 2, .y = 0}),
            std::optional<float>(3.0F));
}

TEST(PersistentElevationMap, IdenticalWindowIsDuplicate) {
  PersistentElevationMap map;
  const GridGeometry geometry = Geometry(2U, 1U, 0.25);
  ASSERT_EQ(Apply(map, geometry, {1.0F, 2.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const std::uint64_t revision = map.Snapshot()->raw_elevation_revision();

  const ElevationUpdateResult duplicate =
      Apply(map, geometry, {1.0F, 2.0F});

  EXPECT_EQ(duplicate.status, ElevationUpdateResult::Status::kDuplicate);
  EXPECT_EQ(duplicate.raw_elevation_revision, revision);
  EXPECT_EQ(duplicate.updated_cells, 0U);
  EXPECT_TRUE(duplicate.dirty_tiles.empty());
  const ElevationMapCounters counters = map.Counters();
  EXPECT_EQ(counters.applied_updates, 1U);
  EXPECT_EQ(counters.duplicate_updates, 1U);
  EXPECT_EQ(counters.rejected_updates, 0U);
}

TEST(PersistentElevationMap, RejectsMalformedGeometryWithoutChangingSnapshot) {
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 0.5), {7.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto before = map.Snapshot();
  const ElevationMapCounters counters_before = map.Counters();
  GridGeometry malformed = Geometry(2U, 1U, 0.5);

  const ElevationUpdateResult rejected = Apply(map, malformed, {8.0F});
  const auto after = map.Snapshot();
  const ElevationMapCounters counters_after = map.Counters();

  EXPECT_EQ(rejected.status, ElevationUpdateResult::Status::kRejected);
  EXPECT_EQ(rejected.raw_elevation_revision,
            before->raw_elevation_revision());
  EXPECT_EQ(rejected.updated_cells, 0U);
  EXPECT_TRUE(rejected.dirty_tiles.empty());
  EXPECT_EQ(after, before);
  EXPECT_EQ(counters_after.applied_updates, counters_before.applied_updates);
  EXPECT_EQ(counters_after.duplicate_updates,
            counters_before.duplicate_updates);
  EXPECT_EQ(counters_after.rejected_updates,
            counters_before.rejected_updates + 1U);
  EXPECT_EQ(counters_after.allocated_tiles, counters_before.allocated_tiles);
  EXPECT_EQ(counters_after.tile_allocations, counters_before.tile_allocations);
  EXPECT_EQ(counters_after.tile_copies, counters_before.tile_copies);
  EXPECT_EQ(counters_after.updated_cells, counters_before.updated_cells);
  EXPECT_EQ(counters_after.estimated_bytes, counters_before.estimated_bytes);
}

TEST(PersistentElevationMap, FreezesFirstValidResolution) {
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 0.25), {1.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto before = map.Snapshot();

  const ElevationUpdateResult rejected =
      Apply(map, Geometry(1U, 1U, 0.5), {2.0F});

  EXPECT_EQ(rejected.status, ElevationUpdateResult::Status::kRejected);
  EXPECT_EQ(map.Snapshot(), before);
  EXPECT_DOUBLE_EQ(map.Snapshot()->geometry().resolution_m(), 0.25);
}

TEST(PersistentElevationMap, RejectsAnyDifferentPositiveResolution) {
  PersistentElevationMap map;
  constexpr double kResolution = 1.0e-9;
  ASSERT_EQ(Apply(map, Geometry(1U, 1U, kResolution), {1.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto before = map.Snapshot();

  const ElevationUpdateResult rejected = Apply(
      map,
      Geometry(1U, 1U,
               std::nextafter(kResolution,
                              std::numeric_limits<double>::infinity())),
      {2.0F});

  EXPECT_EQ(rejected.status, ElevationUpdateResult::Status::kRejected);
  EXPECT_EQ(map.Snapshot(), before);
}

TEST(PersistentElevationMap,
     SnapshotExposesOnlyItsOwnChangedCellsAndDirectPredecessor) {
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(4U, 2U, 0.2),
                  std::vector<float>(8U, 0.0F))
                .status,
            ElevationUpdateResult::Status::kApplied);
  const auto first = map.Snapshot();
  ASSERT_TRUE(first);
  EXPECT_EQ(first->changed_cells().size(), 8U);
  EXPECT_FALSE(first->IsDirectSuccessorOf(*first));

  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 0.2, Vec3{.x = 0.4}), {0.25F})
                .status,
            ElevationUpdateResult::Status::kApplied);
  const auto second = map.Snapshot();

  ASSERT_TRUE(second);
  ASSERT_EQ(second->changed_cells().size(), 1U);
  EXPECT_EQ(second->changed_cells().front(), (GridIndex{.x = 2, .y = 0}));
  ASSERT_EQ(second->changed_tiles().size(), 1U);
  EXPECT_EQ(second->changed_tiles().front(), (TileIndex{.x = 0, .y = 0}));
  EXPECT_TRUE(second->IsDirectSuccessorOf(*first));

  PersistentElevationMap independent;
  ASSERT_EQ(Apply(independent, Geometry(4U, 2U, 0.2),
                  std::vector<float>(8U, 1.0F))
                .status,
            ElevationUpdateResult::Status::kApplied);
  EXPECT_FALSE(independent.Snapshot()->IsDirectSuccessorOf(*first));
}

TEST(PersistentElevationMap, RejectsCanonicalElevationOutsideFloatRange) {
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0), {1.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto before = map.Snapshot();

  const ElevationUpdateResult rejected =
      Apply(map, Geometry(1U, 1U, 1.0, Vec3{.x = 4.0}), {2.0F},
            MapFromSource(Vec3{.z = std::numeric_limits<double>::max()}));

  EXPECT_EQ(rejected.status, ElevationUpdateResult::Status::kRejected);
  EXPECT_EQ(map.Snapshot(), before);
  EXPECT_FALSE(map.Snapshot()->ElevationAt(GridIndex{.x = 4, .y = 0}));
}

TEST(PersistentElevationMap, ProjectsRotatedSourceCellsIntoCanonicalGrid) {
  PersistentElevationMap map;

  const ElevationUpdateResult result =
      Apply(map, Geometry(1U, 1U, 1.0), {6.0F},
            MapFromSource({}, std::numbers::pi / 2.0));
  const auto snapshot = map.Snapshot();

  ASSERT_EQ(result.status, ElevationUpdateResult::Status::kApplied);
  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->geometry().min_inclusive(),
            (GridIndex{.x = -1, .y = 0}));
  EXPECT_EQ(snapshot->geometry().max_exclusive(),
            (GridIndex{.x = 0, .y = 1}));
  EXPECT_TRUE(snapshot->geometry().Contains(GridIndex{.x = -1, .y = 0}));
  EXPECT_FALSE(snapshot->geometry().Contains(GridIndex{.x = 0, .y = 0}));
  EXPECT_EQ(snapshot->ElevationAt(GridIndex{.x = -1, .y = 0}),
            std::optional<float>(6.0F));
  EXPECT_EQ(snapshot->ElevationAtWorld(-0.5, 0.5),
            std::optional<float>(6.0F));
}

TEST(PersistentElevationMap,
     FiveDegreeProjectionPreservesCollidingExtremaIndependentOfOrder) {
  const float unknown = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> first_values(36U, unknown);
  first_values[5U * 6U + 4U] = -10.0F;
  first_values[5U * 6U + 5U] = 50.0F;
  PersistentElevationMap first_map;
  ASSERT_EQ(Apply(first_map, Geometry(6U, 6U, 1.0), first_values,
                  MapFromSource({}, 5.0 * std::numbers::pi / 180.0))
                .status,
            ElevationUpdateResult::Status::kApplied);

  std::vector<float> reversed_values(36U, unknown);
  reversed_values[5U * 6U + 4U] = 50.0F;
  reversed_values[5U * 6U + 5U] = -10.0F;
  PersistentElevationMap reversed_map;
  ASSERT_EQ(Apply(reversed_map, Geometry(6U, 6U, 1.0), reversed_values,
                  MapFromSource({}, 5.0 * std::numbers::pi / 180.0))
                .status,
            ElevationUpdateResult::Status::kApplied);

  const auto first_range =
      first_map.Snapshot()->ElevationRangeAt(GridIndex{.x = 4, .y = 5});
  const auto reversed_range =
      reversed_map.Snapshot()->ElevationRangeAt(GridIndex{.x = 4, .y = 5});
  ASSERT_TRUE(first_range);
  ASSERT_TRUE(reversed_range);
  EXPECT_FLOAT_EQ(first_range->min_m, -10.0F);
  EXPECT_FLOAT_EQ(first_range->max_m, 50.0F);
  EXPECT_EQ(reversed_range, first_range);
  EXPECT_EQ(first_map.Snapshot()->ElevationAt(GridIndex{.x = 4, .y = 5}),
            std::optional<float>(20.0F));
}

TEST(PersistentElevationMap,
     ThirtyDegreeProjectionKeepsHighObstacleAndDepressionInSameCell) {
  const float unknown = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> elevation{unknown, -6.0F, 14.0F};
  PersistentElevationMap map;

  ASSERT_EQ(Apply(map, Geometry(3U, 1U, 1.0), elevation,
                  MapFromSource({}, 30.0 * std::numbers::pi / 180.0))
                .status,
            ElevationUpdateResult::Status::kApplied);
  const auto range =
      map.Snapshot()->ElevationRangeAt(GridIndex{.x = 1, .y = 1});

  ASSERT_TRUE(range);
  EXPECT_FLOAT_EQ(range->min_m, -6.0F);
  EXPECT_FLOAT_EQ(range->max_m, 14.0F);
  EXPECT_EQ(map.Snapshot()->ElevationRangeAtWorld(1.5, 1.5), range);
  EXPECT_EQ(map.Snapshot()->ElevationAt(GridIndex{.x = 1, .y = 1}),
            std::optional<float>(4.0F));
}

TEST(PersistentElevationMap, FiniteEvidenceReplacesPriorCollidingRange) {
  const float unknown = std::numeric_limits<float>::quiet_NaN();
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(3U, 1U, 1.0), {unknown, -6.0F, 14.0F},
                  MapFromSource({}, 30.0 * std::numbers::pi / 180.0))
                .status,
            ElevationUpdateResult::Status::kApplied);
  const auto old_snapshot = map.Snapshot();

  ASSERT_EQ(Apply(map, Geometry(3U, 1U, 1.0), {unknown, 2.0F, unknown},
                  MapFromSource({}, 30.0 * std::numbers::pi / 180.0))
                .status,
            ElevationUpdateResult::Status::kApplied);
  const auto new_snapshot = map.Snapshot();

  ASSERT_TRUE(old_snapshot);
  ASSERT_TRUE(new_snapshot);
  EXPECT_EQ(old_snapshot->ElevationRangeAt(GridIndex{.x = 1, .y = 1}),
            (ElevationRange{.min_m = -6.0F, .max_m = 14.0F}));
  EXPECT_EQ(new_snapshot->ElevationRangeAt(GridIndex{.x = 1, .y = 1}),
            (ElevationRange{.min_m = 2.0F, .max_m = 2.0F}));
}

TEST(PersistentElevationMap, RangeMidpointDoesNotOverflowFloat) {
  const float unknown = std::numeric_limits<float>::quiet_NaN();
  const float maximum = std::numeric_limits<float>::max();
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(3U, 1U, 1.0),
                  {unknown, -maximum, maximum},
                  MapFromSource({}, 30.0 * std::numbers::pi / 180.0))
                .status,
            ElevationUpdateResult::Status::kApplied);

  const auto range =
      map.Snapshot()->ElevationRangeAt(GridIndex{.x = 1, .y = 1});
  const auto midpoint =
      map.Snapshot()->ElevationAt(GridIndex{.x = 1, .y = 1});

  ASSERT_TRUE(range);
  ASSERT_TRUE(midpoint);
  EXPECT_FLOAT_EQ(range->min_m, -maximum);
  EXPECT_FLOAT_EQ(range->max_m, maximum);
  EXPECT_TRUE(std::isfinite(*midpoint));
  EXPECT_FLOAT_EQ(*midpoint, 0.0F);
}

TEST(PersistentElevationMap, AdvancesRevisionOnlyForChangedCellsAndTiles) {
  PersistentElevationMap map;
  const ElevationUpdateResult first =
      Apply(map, Geometry(1U, 1U, 1.0), {1.0F});
  ASSERT_EQ(first.status, ElevationUpdateResult::Status::kApplied);
  ASSERT_EQ(first.dirty_tiles.size(), 1U);
  EXPECT_EQ(first.dirty_tiles.front(), (TileIndex{.x = 0, .y = 0}));

  const ElevationUpdateResult second =
      Apply(map, Geometry(1U, 1U, 1.0, Vec3{.x = 256.0}), {2.0F});

  EXPECT_EQ(second.status, ElevationUpdateResult::Status::kApplied);
  EXPECT_EQ(second.raw_elevation_revision, first.raw_elevation_revision + 1U);
  EXPECT_EQ(second.updated_cells, 1U);
  ASSERT_EQ(second.dirty_tiles.size(), 1U);
  EXPECT_EQ(second.dirty_tiles.front(), (TileIndex{.x = 1, .y = 0}));
}

TEST(PersistentElevationMap, OldSnapshotRemainsStableAfterCopyOnWrite) {
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0), {1.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto old_snapshot = map.Snapshot();

  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0), {9.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto new_snapshot = map.Snapshot();

  ASSERT_TRUE(old_snapshot);
  ASSERT_TRUE(new_snapshot);
  EXPECT_EQ(old_snapshot->ElevationAt(GridIndex{.x = 0, .y = 0}),
            std::optional<float>(1.0F));
  EXPECT_EQ(new_snapshot->ElevationAt(GridIndex{.x = 0, .y = 0}),
            std::optional<float>(9.0F));
  EXPECT_EQ(old_snapshot->raw_elevation_revision(), 1U);
  EXPECT_EQ(new_snapshot->raw_elevation_revision(), 2U);
}

TEST(PersistentElevationMap, ExpandsSparseBoundsWithoutMutatingOldSnapshot) {
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0), {1.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto old_snapshot = map.Snapshot();

  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0, Vec3{.x = 256.0}), {2.0F})
                .status,
            ElevationUpdateResult::Status::kApplied);
  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0, Vec3{.x = -2.0}), {3.0F})
                .status,
            ElevationUpdateResult::Status::kApplied);
  const auto expanded = map.Snapshot();

  ASSERT_TRUE(old_snapshot);
  ASSERT_TRUE(expanded);
  EXPECT_EQ(old_snapshot->geometry().min_inclusive(),
            (GridIndex{.x = 0, .y = 0}));
  EXPECT_EQ(old_snapshot->geometry().max_exclusive(),
            (GridIndex{.x = 1, .y = 1}));
  EXPECT_EQ(old_snapshot->geometry().CellCount(), 1U);
  EXPECT_EQ(expanded->geometry().min_inclusive(),
            (GridIndex{.x = -2, .y = 0}));
  EXPECT_EQ(expanded->geometry().max_exclusive(),
            (GridIndex{.x = 257, .y = 1}));
  EXPECT_EQ(expanded->geometry().width(), 259U);
  EXPECT_EQ(expanded->geometry().height(), 1U);
  EXPECT_EQ(expanded->geometry().CellCount(), 259U);
}

TEST(PersistentElevationMap, WorldQueriesRespectSparseBounds) {
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0), {4.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto snapshot = map.Snapshot();

  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->ElevationAtWorld(0.5, 0.5),
            std::optional<float>(4.0F));
  EXPECT_FALSE(snapshot->ElevationAtWorld(-0.001, 0.5));
  EXPECT_FALSE(snapshot->ElevationAtWorld(1.0, 0.5));
  EXPECT_FALSE(snapshot->ElevationAtWorld(0.5, -0.001));
  EXPECT_FALSE(snapshot->ElevationAtWorld(0.5, 1.0));
}

TEST(PersistentElevationMap, DefaultSnapshotHasEmptySparseGeometry) {
  const ElevationSnapshot snapshot;

  EXPECT_FALSE(snapshot.valid());
  EXPECT_TRUE(snapshot.geometry().empty());
  EXPECT_EQ(snapshot.geometry().CellCount(), 0U);
}

TEST(PersistentElevationMap, ReportsTileAllocationAndCopyCounters) {
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0), {1.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  ElevationMapCounters counters = map.Counters();
  EXPECT_EQ(counters.allocated_tiles, 1U);
  EXPECT_EQ(counters.tile_allocations, 1U);
  EXPECT_EQ(counters.tile_copies, 0U);
  EXPECT_EQ(counters.updated_cells, 1U);
  EXPECT_GT(counters.estimated_bytes, 0U);

  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0), {2.0F}).status,
            ElevationUpdateResult::Status::kApplied);
  counters = map.Counters();
  EXPECT_EQ(counters.allocated_tiles, 1U);
  EXPECT_EQ(counters.tile_allocations, 1U);
  EXPECT_EQ(counters.tile_copies, 1U);
  EXPECT_EQ(counters.updated_cells, 2U);

  ASSERT_EQ(Apply(map, Geometry(1U, 1U, 1.0, Vec3{.x = 256.0}), {3.0F})
                .status,
            ElevationUpdateResult::Status::kApplied);
  counters = map.Counters();
  EXPECT_EQ(counters.allocated_tiles, 2U);
  EXPECT_EQ(counters.tile_allocations, 2U);
  EXPECT_EQ(counters.tile_copies, 1U);
  EXPECT_EQ(counters.updated_cells, 3U);
  EXPECT_GT(counters.estimated_bytes, 256U * 256U * sizeof(float));
}


TEST(PersistentElevationMap, CenterStatisticsRevisionDuplicateAndSnapshotIsolation) {
  PersistentElevationMap map;
  const auto geometry = Geometry(1, 1, 1.0);
  std::vector<float> heights{0};
  std::vector<LocalTerrainMeasurements> stats{{true, true, 0.2, 0.3, 0.1}};
  const auto apply = [&] { return map.Apply({.geometry = geometry,
    .elevation_m = heights, .map_from_source = MapFromSource(), .terrain_measurements = stats}); };
  ASSERT_EQ(apply().status, ElevationUpdateResult::Status::kApplied);
  const auto before = map.Snapshot();
  const auto measured_bytes = map.Counters().estimated_bytes;
  EXPECT_EQ(apply().status, ElevationUpdateResult::Status::kDuplicate);
  stats[0].relief_m = 0.5;
  auto changed = apply();
  EXPECT_EQ(changed.status, ElevationUpdateResult::Status::kApplied);
  EXPECT_EQ(changed.updated_cells, 1U);
  EXPECT_EQ(changed.raw_elevation_revision, 2U);
  EXPECT_EQ(map.Snapshot()->changed_cells().size(), 1U);
  ASSERT_TRUE(before->TerrainMeasurementsAt({0, 0}));
  EXPECT_DOUBLE_EQ(before->TerrainMeasurementsAt({0, 0})->relief_m, 0.3);
  EXPECT_DOUBLE_EQ(map.Snapshot()->TerrainMeasurementsAt({0, 0})->relief_m, 0.5);
  // Incomplete recomputation and identical height-only evidence cannot erase a valid measurement.
  stats[0] = {true, false, 0, 0, 0};
  EXPECT_EQ(apply().status, ElevationUpdateResult::Status::kDuplicate);
  EXPECT_EQ(Apply(map, geometry, heights).status, ElevationUpdateResult::Status::kDuplicate);
  EXPECT_DOUBLE_EQ(map.Snapshot()->TerrainMeasurementsAt({0, 0})->relief_m, 0.5);
  heights[0] = 1;
  EXPECT_EQ(Apply(map, geometry, heights).status, ElevationUpdateResult::Status::kApplied);
  EXPECT_EQ(map.Snapshot()->TerrainMeasurementsAt({0, 0}), std::nullopt);
  EXPECT_LT(map.Counters().estimated_bytes, measured_bytes);
  EXPECT_DOUBLE_EQ(before->TerrainMeasurementsAt({0, 0})->relief_m, 0.3);
}

TEST(PersistentElevationMap, ScalarMeasurementsRejectUnsupportedGeometryAtomically) {
  const auto geometry = Geometry(1, 1, 1.0);
  const std::vector<float> heights{0};
  std::vector<LocalTerrainMeasurements> stats{{true, true, 0.2, 0.3, 0.1}};
  PersistentElevationMap map;
  ASSERT_EQ(Apply(map, geometry, heights).status, ElevationUpdateResult::Status::kApplied);
  auto evidence = ElevationEvidence{.geometry = geometry, .elevation_m = heights,
      .map_from_source = MapFromSource(), .terrain_measurements = stats};
  evidence.map_from_source.rotation = {.w = std::cos(0.2), .x = std::sin(0.2)};
  EXPECT_EQ(map.Apply(evidence).status, ElevationUpdateResult::Status::kRejected);
  evidence.map_from_source = MapFromSource({}, 0.3);
  EXPECT_EQ(map.Apply(evidence).status, ElevationUpdateResult::Status::kRejected);
  evidence.map_from_source = MapFromSource({.x = 0.25});
  EXPECT_EQ(map.Apply(evidence).status, ElevationUpdateResult::Status::kRejected);
  evidence.map_from_source = MapFromSource();
  stats[0].relief_m = -1;
  EXPECT_EQ(map.Apply(evidence).status, ElevationUpdateResult::Status::kRejected);
  EXPECT_EQ(map.Snapshot()->raw_elevation_revision(), 1U);
  EXPECT_EQ(map.Snapshot()->TerrainMeasurementsAt({0, 0}), std::nullopt);
}

}  // namespace
}  // namespace lunar::incremental_navigation
