#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] GridGeometry Geometry(const std::size_t width,
                                    const std::size_t height,
                                    const double resolution_m = 0.2) {
  return GridGeometry{.frame_id = "map",
                      .width = width,
                      .height = height,
                      .resolution_m = resolution_m};
}

[[nodiscard]] RigidTransform IdentityMapTransform() {
  return RigidTransform{.parent_frame = "map", .child_frame = "map"};
}

[[nodiscard]] ElevationUpdateResult Apply(
    PersistentElevationMap& map, const GridGeometry& geometry,
    const std::vector<float>& elevation_m) {
  return map.Apply(ElevationEvidence{
      .geometry = geometry,
      .elevation_m = elevation_m,
      .map_from_source = IdentityMapTransform(),
  });
}

[[nodiscard]] std::shared_ptr<const ElevationSnapshot> Snapshot(
    PersistentElevationMap& map, const GridGeometry& geometry,
    const std::vector<float>& elevation_m) {
  const ElevationUpdateResult update = Apply(map, geometry, elevation_m);
  if (update.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("test elevation fixture was rejected");
  }
  return map.Snapshot();
}

[[nodiscard]] WheeledCapability WheelCapability() {
  return WheeledCapability{
      .footprint_xy_m = {{-0.3, -0.2}, {0.3, -0.2},
                         {0.3, 0.2}, {-0.3, 0.2}},
      .body_extent_m = {.x = 0.6, .y = 0.4, .z = 0.3},
      .wheel_diameter_m = 0.3,
      .wheel_width_m = 0.08,
      .wheelbase_m = 0.4,
      .track_width_m = 0.3,
      .minimum_underbody_clearance_m = 0.5,
      .maximum_local_obstacle_relief_m = 0.2,
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
      .maximum_slope_rad = std::numbers::pi / 2.0,
      .minimum_clearance_m = 0.2,
  };
}

[[nodiscard]] LeggedCapability LeggedCapabilityForTest() {
  return LeggedCapability{
      .body_extent_m = {.x = 0.6, .y = 0.4, .z = 0.3},
      .nominal_body_height_m = 0.5,
      .platform_mass_kg = 20.0,
      .nominal_payload_kg = 2.0,
      .maximum_payload_kg = 10.0,
      .maximum_slope_rad = std::numbers::pi / 2.0,
      .maximum_step_height_m = 0.6,
      .maximum_gap_width_m = 0.2,
      .minimum_body_clearance_m = 0.3,
      .step_vertical_rate_mps = 0.2,
      .body_height_m = {.lower = 0.4, .upper = 0.6},
      .forward_speed_mps = {.lower = -0.5, .upper = 0.8},
      .lateral_speed_mps = {.lower = -0.4, .upper = 0.4},
      .yaw_rate_radps = {.lower = -0.8, .upper = 0.8},
      .maximum_linear_acceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .unknown_is_traversable = false,
  };
}

[[nodiscard]] TraversabilityProfile Profile(
    const double half_x_m = 0.3, const double half_y_m = 0.2,
    const double preferred_clearance_m = 0.4,
    const TraversalCostWeights weights = {.slope = 1.0,
                                          .relief = 1.0,
                                          .clearance = 1.0}) {
  return TraversabilityProfile{
      .planar_envelope_xy_m = {{-half_x_m, -half_y_m},
                               {half_x_m, -half_y_m},
                               {half_x_m, half_y_m},
                               {-half_x_m, half_y_m}},
      .preferred_clearance_m = preferred_clearance_m,
      .slope_weight = weights.slope,
      .relief_weight = weights.relief,
      .clearance_weight = weights.clearance,
      .start_blind_zone_margin_m = 0.1,
  };
}

[[nodiscard]] std::size_t Offset(const GridGeometry& geometry,
                                 const GridIndex index) {
  return static_cast<std::size_t>(index.y) * geometry.width +
         static_cast<std::size_t>(index.x);
}

TEST(FineTraversabilityGeometry, UsesActualEnvelopeRadiusWithoutPadding) {
  const TraversabilityProfile profile = Profile(0.3, 0.4);

  EXPECT_DOUBLE_EQ(CircumscribedRadius(profile.planar_envelope_xy_m), 0.5);
}

TEST(FineTraversabilityGeometry,
     CircleIntersectionUsesCellAreaAtEveryResolution) {
  for (const double resolution_m : {0.1, 0.2, 0.75}) {
    const SparseGridGeometry geometry(
        "map", resolution_m, {}, {.x = 0, .y = 0}, {.x = 8, .y = 8});
    const Point2 center{.x = 2.5 * resolution_m,
                        .y = 2.5 * resolution_m};

    EXPECT_TRUE(CircleIntersectsCellArea(
        center, 0.5 * resolution_m, geometry, {.x = 3, .y = 2}));
    EXPECT_FALSE(CircleIntersectsCellArea(
        center, std::nextafter(0.5 * resolution_m, 0.0), geometry,
        {.x = 3, .y = 2}));
    EXPECT_FALSE(CircleIntersectsCellArea(
        center, 0.7 * resolution_m, geometry, {.x = 3, .y = 3}));
    EXPECT_TRUE(CircleIntersectsCellArea(
        center, std::sqrt(0.5) * resolution_m, geometry,
        {.x = 3, .y = 3}));
  }
}

class ExtremeElevationView final : public ElevationRangeView {
 public:
  explicit ExtremeElevationView(SparseGridGeometry geometry)
      : geometry_(std::move(geometry)) {}

  [[nodiscard]] const SparseGridGeometry& geometry() const noexcept override {
    return geometry_;
  }

  [[nodiscard]] std::optional<ElevationRange> ElevationRangeAt(
      const GridIndex index) const noexcept override {
    return geometry_.Contains(index)
               ? std::optional<ElevationRange>(
                     ElevationRange{.min_m = 0.0F, .max_m = 0.0F})
               : std::nullopt;
  }

 private:
  SparseGridGeometry geometry_;
};

TEST(FineTraversabilityGeometry,
     SharedEvaluatorHandlesBothInt64GeometryBoundaries) {
  constexpr std::int64_t kMinimum =
      std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t kMaximum =
      std::numeric_limits<std::int64_t>::max();
  const TraversabilityProfile profile = Profile(0.1, 0.1, 0.5);
  const ExtremeElevationView near_min(SparseGridGeometry(
      "map", 1.0, {}, {.x = kMinimum, .y = kMinimum},
      {.x = kMinimum + 4, .y = kMinimum + 4}));
  const ExtremeElevationView near_max(SparseGridGeometry(
      "map", 1.0, {}, {.x = kMaximum - 4, .y = kMaximum - 4},
      {.x = kMaximum, .y = kMaximum}));

  FineCellEvaluator minimum_evaluator(near_min, WheelCapability(), profile);
  FineCellEvaluator maximum_evaluator(near_max, WheelCapability(), profile);

  EXPECT_EQ(minimum_evaluator.Evaluate({.x = kMinimum, .y = kMinimum}).state,
            FineCellState::kUnknown);
  EXPECT_EQ(
      maximum_evaluator
          .Evaluate({.x = kMaximum - 1, .y = kMaximum - 1})
          .state,
      FineCellState::kUnknown);
}

TEST(FineTraversabilityBuilder,
     DenseDeriveReportsTileBoundedIntrinsicScratchStorage) {
  const GridGeometry geometry = Geometry(520U, 16U);
  PersistentElevationMap map;
  const auto raw = Snapshot(
      map, geometry, std::vector<float>(geometry.CellCount(), 0.0F));
  const TraversabilityProfile profile = Profile();
  FineCellEvaluator evaluator(*raw, WheelCapability(), profile);

  for (std::int64_t y = 0; y < static_cast<std::int64_t>(geometry.height);
       ++y) {
    for (std::int64_t x = 0; x < static_cast<std::int64_t>(geometry.width);
         ++x) {
      static_cast<void>(evaluator.Evaluate({.x = x, .y = y}));
    }
  }

  EXPECT_EQ(evaluator.evaluated_elevation_cells(), geometry.CellCount());
  EXPECT_EQ(evaluator.cached_elevation_tiles(), 3U);

  const auto fine = FineTraversabilityBuilder().Derive(
      raw, WheelCapability(), profile);
  ASSERT_TRUE(fine);
  EXPECT_EQ(fine->metrics().elevation_cells_examined, geometry.CellCount());
  EXPECT_EQ(fine->metrics().elevation_cache_tiles, 3U);
}

TEST(FineTraversabilityBuilder, PreservesWheelAndLeggedPhysicsDifferences) {
  const GridGeometry geometry = Geometry(21U, 21U);
  std::vector<float> values(geometry.CellCount(), 0.0F);
  const GridIndex bump{.x = 10, .y = 10};
  values[Offset(geometry, bump)] = 0.4F;
  PersistentElevationMap map;
  const auto raw = Snapshot(map, geometry, values);
  const TraversabilityProfile profile = Profile(0.01, 0.01, 0.0, {});
  FineTraversabilityBuilder builder;

  const auto wheel = builder.Derive(raw, WheelCapability(), profile);
  const auto legged =
      builder.Derive(raw, LeggedCapabilityForTest(), profile);

  ASSERT_TRUE(wheel);
  ASSERT_TRUE(legged);
  EXPECT_EQ(wheel->State(bump), FineCellState::kBlocked);
  EXPECT_EQ(legged->State(bump), FineCellState::kFree);
}

TEST(FineTraversabilityBuilder, BlockedTakesPriorityOverInflatedUnknown) {
  const GridGeometry geometry = Geometry(25U, 25U);
  std::vector<float> values(geometry.CellCount(), 0.0F);
  const GridIndex center{.x = 12, .y = 12};
  values[Offset(geometry, center)] = 0.5F;
  values[Offset(geometry, {.x = 13, .y = 12})] =
      std::numeric_limits<float>::quiet_NaN();
  PersistentElevationMap map;
  const auto raw = Snapshot(map, geometry, values);
  FineTraversabilityBuilder builder;

  const auto fine = builder.Derive(raw, WheelCapability(),
                                   Profile(0.25, 0.01, 0.0, {}));

  ASSERT_TRUE(fine);
  EXPECT_EQ(fine->State({.x = 13, .y = 12}), FineCellState::kBlocked);
}

TEST(FineTraversabilityBuilder, InflatesUnknownWithoutChangingItToBlocked) {
  const GridGeometry geometry = Geometry(25U, 25U);
  std::vector<float> values(geometry.CellCount(), 0.0F);
  const GridIndex missing{.x = 12, .y = 12};
  values[Offset(geometry, missing)] =
      std::numeric_limits<float>::quiet_NaN();
  PersistentElevationMap map;
  const auto raw = Snapshot(map, geometry, values);
  FineTraversabilityBuilder builder;

  const auto fine = builder.Derive(raw, WheelCapability(),
                                   Profile(0.25, 0.01, 0.0, {}));

  ASSERT_TRUE(fine);
  EXPECT_EQ(fine->State({.x = 13, .y = 12}), FineCellState::kUnknown);
  EXPECT_EQ(fine->State({.x = 15, .y = 12}), FineCellState::kFree);
}

TEST(FineTraversabilityBuilder, ZeroWeightsProduceZeroFiniteTraversalCost) {
  const GridGeometry geometry = Geometry(25U, 25U);
  std::vector<float> values(geometry.CellCount(), 0.0F);
  values[Offset(geometry, {.x = 12, .y = 12})] = 0.05F;
  PersistentElevationMap map;
  const auto raw = Snapshot(map, geometry, values);
  FineTraversabilityBuilder builder;

  const auto fine = builder.Derive(raw, LeggedCapabilityForTest(),
                                   Profile(0.01, 0.01, 0.8, {}));

  ASSERT_TRUE(fine);
  ASSERT_EQ(fine->State({.x = 12, .y = 12}), FineCellState::kFree);
  EXPECT_DOUBLE_EQ(fine->TraversalCost({.x = 12, .y = 12}), 0.0);
  EXPECT_TRUE(std::isfinite(fine->TraversalCost({.x = 12, .y = 12})));
}

TEST(FineTraversabilityBuilder,
     SaturatesLargeFiniteSoftWeightsWithoutChangingTopology) {
  const GridGeometry geometry = Geometry(25U, 25U);
  std::vector<float> values(geometry.CellCount(), 0.0F);
  const GridIndex raised{.x = 12, .y = 12};
  values[Offset(geometry, raised)] = 0.59F;
  PersistentElevationMap map;
  const auto raw = Snapshot(map, geometry, values);
  FineTraversabilityBuilder builder;
  const TraversabilityProfile geometric = Profile(0.01, 0.01, 10.0, {});
  const double largest_finite = std::numeric_limits<double>::max();
  const TraversabilityProfile saturated =
      Profile(0.01, 0.01, 10.0,
              {.slope = largest_finite,
               .relief = largest_finite,
               .clearance = largest_finite});
  const auto geometric_fine =
      builder.Derive(raw, LeggedCapabilityForTest(), geometric);

  std::shared_ptr<const FineTraversabilitySnapshot> saturated_fine;
  EXPECT_NO_THROW(saturated_fine = builder.Derive(
                      raw, LeggedCapabilityForTest(), saturated,
                      geometric_fine));

  ASSERT_TRUE(saturated_fine);
  for (std::int64_t y = 0; y < 25; ++y) {
    for (std::int64_t x = 0; x < 25; ++x) {
      EXPECT_EQ(saturated_fine->State({.x = x, .y = y}),
                geometric_fine->State({.x = x, .y = y}));
    }
  }
  ASSERT_EQ(saturated_fine->State(raised), FineCellState::kFree);
  EXPECT_TRUE(std::isfinite(saturated_fine->TraversalCost(raised)));
  EXPECT_GE(saturated_fine->TraversalCost(raised), 0.0);
  EXPECT_DOUBLE_EQ(saturated_fine->TraversalCost(raised), largest_finite);
}

TEST(FineTraversabilityBuilder,
     PreferredClearanceChangesCostWithoutChangingTopology) {
  const GridGeometry geometry = Geometry(31U, 31U);
  std::vector<float> values(geometry.CellCount(), 0.0F);
  values[Offset(geometry, {.x = 15, .y = 15})] = 0.5F;
  PersistentElevationMap map;
  const auto raw = Snapshot(map, geometry, values);
  FineTraversabilityBuilder builder;
  TraversabilityProfile short_clearance = Profile(0.01, 0.01, 0.2, {});
  short_clearance.clearance_weight = 2.0;
  TraversabilityProfile long_clearance = short_clearance;
  long_clearance.preferred_clearance_m = 1.0;

  const auto short_fine =
      builder.Derive(raw, WheelCapability(), short_clearance);
  const auto long_fine = builder.Derive(raw, WheelCapability(), long_clearance,
                                        short_fine);

  ASSERT_TRUE(short_fine);
  ASSERT_TRUE(long_fine);
  for (std::int64_t y = 0; y < 31; ++y) {
    for (std::int64_t x = 0; x < 31; ++x) {
      EXPECT_EQ(short_fine->State({.x = x, .y = y}),
                long_fine->State({.x = x, .y = y}));
    }
  }
  const GridIndex free_near_hazard{.x = 20, .y = 15};
  ASSERT_EQ(short_fine->State(free_near_hazard), FineCellState::kFree);
  EXPECT_GT(long_fine->TraversalCost(free_near_hazard),
            short_fine->TraversalCost(free_near_hazard));
}

TEST(FineTraversabilityBuilder, RejectsInvalidEnvelopeAndSoftWeights) {
  const GridGeometry geometry = Geometry(5U, 5U);
  PersistentElevationMap map;
  const auto raw = Snapshot(
      map, geometry, std::vector<float>(geometry.CellCount(), 0.0F));
  FineTraversabilityBuilder builder;
  TraversabilityProfile invalid_envelope = Profile();
  invalid_envelope.planar_envelope_xy_m = {
      {1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}, {1.0, 2.0}};
  TraversabilityProfile invalid_weight = Profile();
  invalid_weight.slope_weight = -1.0;

  EXPECT_THROW(builder.Derive(raw, WheelCapability(), invalid_envelope),
               std::invalid_argument);
  EXPECT_THROW(builder.Derive(raw, WheelCapability(), invalid_weight),
               std::invalid_argument);
}

TEST(FineTraversabilityBuilder, DuplicateRawSnapshotPreservesFineRevision) {
  const GridGeometry geometry = Geometry(20U, 20U);
  PersistentElevationMap map;
  std::vector<float> values(geometry.CellCount(), 0.0F);
  const auto raw = Snapshot(map, geometry, values);
  FineTraversabilityBuilder builder;
  const auto first = builder.Derive(raw, WheelCapability(), Profile());
  const ElevationUpdateResult duplicate = Apply(map, geometry, values);

  const auto second =
      builder.Derive(map.Snapshot(), WheelCapability(), Profile(), first);

  ASSERT_EQ(duplicate.status, ElevationUpdateResult::Status::kDuplicate);
  EXPECT_EQ(second, first);
  EXPECT_EQ(second->fine_traversability_revision(), 1U);
}

TEST(FineTraversabilityBuilder,
     EqualRevisionFromIndependentMapDoesNotReuseOldObstacleState) {
  const GridGeometry geometry = Geometry(25U, 25U);
  const GridIndex center{.x = 12, .y = 12};
  std::vector<float> obstructed_values(geometry.CellCount(), 0.0F);
  obstructed_values[Offset(geometry, center)] = 0.5F;
  std::vector<float> flat_values(geometry.CellCount(), 0.0F);
  PersistentElevationMap obstructed_map;
  PersistentElevationMap flat_map;
  const auto obstructed_raw =
      Snapshot(obstructed_map, geometry, obstructed_values);
  const auto flat_raw = Snapshot(flat_map, geometry, flat_values);
  ASSERT_EQ(obstructed_raw->raw_elevation_revision(),
            flat_raw->raw_elevation_revision());
  FineTraversabilityBuilder builder;
  const TraversabilityProfile profile = Profile(0.01, 0.01, 0.0, {});
  const auto obstructed =
      builder.Derive(obstructed_raw, WheelCapability(), profile);
  ASSERT_EQ(obstructed->State(center), FineCellState::kBlocked);

  const auto flat =
      builder.Derive(flat_raw, WheelCapability(), profile, obstructed);

  ASSERT_TRUE(flat);
  EXPECT_NE(flat, obstructed);
  EXPECT_EQ(flat->elevation(), flat_raw);
  EXPECT_EQ(flat->State(center), FineCellState::kFree);
}

TEST(FineTraversabilityBuilder,
     StartBlindZoneMarginDoesNotChangeBaseFineSnapshotIdentity) {
  const GridGeometry geometry = Geometry(20U, 20U);
  PersistentElevationMap map;
  const auto raw = Snapshot(
      map, geometry, std::vector<float>(geometry.CellCount(), 0.0F));
  FineTraversabilityBuilder builder;
  TraversabilityProfile first_profile = Profile();
  first_profile.start_blind_zone_margin_m = 0.1;
  const auto first =
      builder.Derive(raw, WheelCapability(), first_profile);
  TraversabilityProfile request_only_change = first_profile;
  request_only_change.start_blind_zone_margin_m =
      std::numeric_limits<double>::quiet_NaN();

  std::shared_ptr<const FineTraversabilitySnapshot> second;
  EXPECT_NO_THROW(second = builder.Derive(
                      raw, WheelCapability(), request_only_change, first));

  EXPECT_EQ(second, first);
  EXPECT_EQ(second->fine_traversability_revision(), 1U);
}

TEST(FineTraversabilityBuilder,
     ChangedTerrainRecomputesBoundedHaloAndSharesFarTile) {
  const GridGeometry geometry = Geometry(520U, 16U);
  PersistentElevationMap map;
  std::vector<float> values(geometry.CellCount(), 0.0F);
  const auto first_raw = Snapshot(map, geometry, values);
  FineTraversabilityBuilder builder;
  const TraversabilityProfile profile = Profile(0.1, 0.1, 0.4);
  const auto first =
      builder.Derive(first_raw, LeggedCapabilityForTest(), profile);
  const auto far_tile = first->FindTile({.x = 2, .y = 0});
  ASSERT_TRUE(far_tile);

  const GridGeometry one_cell = GridGeometry{
      .frame_id = "map",
      .width = 1U,
      .height = 1U,
      .resolution_m = geometry.resolution_m,
      .origin_m = {.x = 10.0 * geometry.resolution_m,
                   .y = 8.0 * geometry.resolution_m},
  };
  ASSERT_EQ(Apply(map, one_cell, {0.1F}).status,
            ElevationUpdateResult::Status::kApplied);

  const auto second = builder.Derive(map.Snapshot(),
                                     LeggedCapabilityForTest(), profile, first);

  ASSERT_TRUE(second);
  EXPECT_EQ(second->fine_traversability_revision(), 2U);
  EXPECT_EQ(second->raw_elevation_revision(), 2U);
  EXPECT_LT(second->metrics().updated_cells, geometry.CellCount());
  EXPECT_LT(second->metrics().elevation_cells_examined,
            geometry.CellCount() / 4U);
  ASSERT_EQ(second->changed_halo_tiles().size(), 1U);
  EXPECT_EQ(second->changed_halo_tiles().front(), (TileIndex{.x = 0, .y = 0}));
  EXPECT_LT(second->tile_directory().last_update_copied_nodes(),
            second->tile_directory().tile_count());
  EXPECT_EQ(second->FindTile({.x = 2, .y = 0}), far_tile);
  EXPECT_EQ(second->State({.x = 510, .y = 8}),
            first->State({.x = 510, .y = 8}));
  EXPECT_DOUBLE_EQ(second->TraversalCost({.x = 510, .y = 8}),
                   first->TraversalCost({.x = 510, .y = 8}));
}

TEST(FineTraversabilityBuilder,
     MergedMultiRevisionChangesStayIncrementalAndShareFarTile) {
  const GridGeometry geometry = Geometry(520U, 16U);
  PersistentElevationMap map;
  const auto first_raw = Snapshot(
      map, geometry, std::vector<float>(geometry.CellCount(), 0.0F));
  FineTraversabilityBuilder builder;
  const TraversabilityProfile profile = Profile(0.1, 0.1, 0.4);
  const auto first =
      builder.Derive(first_raw, LeggedCapabilityForTest(), profile);
  const auto far_tile = first->FindTile({.x = 2, .y = 0});
  ASSERT_TRUE(far_tile);

  std::vector<GridIndex> merged_changes;
  for (const std::int64_t x : {10, 100}) {
    const GridGeometry one_cell{
        .frame_id = "map",
        .width = 1U,
        .height = 1U,
        .resolution_m = geometry.resolution_m,
        .origin_m = {.x = static_cast<double>(x) * geometry.resolution_m,
                     .y = 8.0 * geometry.resolution_m},
    };
    ASSERT_EQ(Apply(map, one_cell, {0.1F}).status,
              ElevationUpdateResult::Status::kApplied);
    const auto raw = map.Snapshot();
    merged_changes.insert(merged_changes.end(), raw->changed_cells().begin(),
                          raw->changed_cells().end());
  }

  const auto second = builder.Derive(
      map.Snapshot(), LeggedCapabilityForTest(), profile, first,
      FineElevationChangeSet{
          .base_raw_elevation_revision = first_raw->raw_elevation_revision(),
          .changed_cells = merged_changes});

  ASSERT_TRUE(second);
  EXPECT_EQ(second->raw_elevation_revision(), 3U);
  EXPECT_EQ(second->fine_traversability_revision(), 2U);
  EXPECT_LT(second->metrics().updated_cells, geometry.CellCount());
  EXPECT_LT(second->metrics().elevation_cells_examined,
            geometry.CellCount() / 2U);
  EXPECT_EQ(second->FindTile({.x = 2, .y = 0}), far_tile);
  EXPECT_GT(second->TraversalCost({.x = 10, .y = 8}),
            first->TraversalCost({.x = 10, .y = 8}));
  EXPECT_GT(second->TraversalCost({.x = 100, .y = 8}),
            first->TraversalCost({.x = 100, .y = 8}));
}

TEST(FineTraversabilityBuilder,
     MergedChangesRejectAnIndependentMapWithMatchingRevisionsAndLattice) {
  const GridGeometry geometry = Geometry(16U, 16U);
  PersistentElevationMap first_map;
  PersistentElevationMap independent_map;
  const auto first_raw = Snapshot(
      first_map, geometry, std::vector<float>(geometry.CellCount(), 0.0F));
  const auto independent_first = Snapshot(
      independent_map, geometry,
      std::vector<float>(geometry.CellCount(), 0.0F));
  FineTraversabilityBuilder builder;
  const auto first =
      builder.Derive(first_raw, LeggedCapabilityForTest(), Profile());
  const GridGeometry one_cell{
      .frame_id = "map",
      .width = 1U,
      .height = 1U,
      .resolution_m = geometry.resolution_m,
      .origin_m = {.x = 4.0 * geometry.resolution_m,
                   .y = 4.0 * geometry.resolution_m},
  };
  ASSERT_EQ(Apply(independent_map, one_cell, {0.1F}).status,
            ElevationUpdateResult::Status::kApplied);
  const auto independent_latest = independent_map.Snapshot();
  ASSERT_EQ(independent_latest->raw_elevation_revision(), 2U);
  ASSERT_FALSE(independent_latest->SharesLineageWith(*first_raw));
  const std::vector<GridIndex> claimed_changes{{.x = 4, .y = 4}};

  EXPECT_THROW(
      builder.Derive(
          independent_latest, LeggedCapabilityForTest(), Profile(), first,
          FineElevationChangeSet{
              .base_raw_elevation_revision =
                  first_raw->raw_elevation_revision(),
              .changed_cells = claimed_changes}),
      std::invalid_argument);
  EXPECT_EQ(independent_first->raw_elevation_revision(), 1U);
}

TEST(FineTraversabilityBuilder, PublishesCompleteImmutableSnapshotLineage) {
  const GridGeometry geometry = Geometry(12U, 12U);
  PersistentElevationMap map;
  const auto raw = Snapshot(
      map, geometry, std::vector<float>(geometry.CellCount(), 0.0F));
  FineTraversabilityBuilder builder;

  const auto fine = builder.Derive(raw, WheelCapability(), Profile());

  ASSERT_TRUE(fine);
  EXPECT_EQ(fine->elevation(), raw);
  EXPECT_EQ(fine->raw_elevation_revision(), raw->raw_elevation_revision());
  EXPECT_EQ(fine->fine_traversability_revision(), 1U);
  EXPECT_FALSE(fine->platform_profile_hash().empty());
  EXPECT_DOUBLE_EQ(fine->hard_inflation_radius_m(),
                   std::sqrt(0.3 * 0.3 + 0.2 * 0.2));
  EXPECT_DOUBLE_EQ(fine->preferred_clearance_m(), 0.4);
  EXPECT_DOUBLE_EQ(fine->cost_weights().slope, 1.0);
  EXPECT_DOUBLE_EQ(fine->cost_weights().relief, 1.0);
  EXPECT_DOUBLE_EQ(fine->cost_weights().clearance, 1.0);
  EXPECT_GT(fine->metrics().allocated_cells, 0U);
  EXPECT_GT(fine->metrics().allocated_bytes, 0U);
  EXPECT_TRUE(std::isfinite(fine->metrics().derivation_ms));
}

}  // namespace
}  // namespace lunar::incremental_navigation
