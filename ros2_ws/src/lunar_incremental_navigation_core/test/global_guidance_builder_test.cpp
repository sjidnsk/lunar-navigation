#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/global_guidance_builder.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] RigidTransform IdentityMapTransform() {
  return RigidTransform{.parent_frame = "map", .child_frame = "map"};
}

[[nodiscard]] GridGeometry Geometry(const std::size_t width,
                                    const std::size_t height,
                                    const double resolution_m = 0.5,
                                    const double origin_x_m = 0.0,
                                    const double origin_y_m = 0.0) {
  return GridGeometry{.frame_id = "map",
                      .width = width,
                      .height = height,
                      .resolution_m = resolution_m,
                      .origin_m = Vec3{.x = origin_x_m,
                                       .y = origin_y_m,
                                       .z = 0.0}};
}

[[nodiscard]] std::shared_ptr<const ElevationSnapshot> MakeElevation(
    const GridGeometry& geometry, const std::vector<float>& values) {
  PersistentElevationMap map;
  const ElevationUpdateResult result = map.Apply(ElevationEvidence{
      .geometry = geometry,
      .elevation_m = values,
      .map_from_source = IdentityMapTransform(),
  });
  if (result.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("elevation test fixture was not applied");
  }
  return map.Snapshot();
}

[[nodiscard]] std::shared_ptr<const ElevationSnapshot> MakeSparseElevation() {
  PersistentElevationMap map;
  const std::array<float, 1U> value{1.0F};
  for (const double origin_x_m : {0.0, 2.0}) {
    const ElevationUpdateResult result = map.Apply(ElevationEvidence{
        .geometry = Geometry(1U, 1U, 1.0, origin_x_m),
        .elevation_m = value,
        .map_from_source = IdentityMapTransform(),
    });
    if (result.status != ElevationUpdateResult::Status::kApplied) {
      throw std::runtime_error("sparse elevation fixture was not applied");
    }
  }
  return map.Snapshot();
}

[[nodiscard]] std::shared_ptr<const ElevationSnapshot> MakeSparseElevationAt(
    const std::vector<double>& origins_x_m, const double resolution_m = 1.0) {
  PersistentElevationMap map;
  const std::array<float, 1U> value{1.0F};
  for (const double origin_x_m : origins_x_m) {
    const ElevationUpdateResult result = map.Apply(ElevationEvidence{
        .geometry = Geometry(1U, 1U, resolution_m, origin_x_m),
        .elevation_m = value,
        .map_from_source = IdentityMapTransform(),
    });
    if (result.status != ElevationUpdateResult::Status::kApplied) {
      throw std::runtime_error("wide sparse elevation fixture was not applied");
    }
  }
  return map.Snapshot();
}

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot> MakeFine(
    std::shared_ptr<const ElevationSnapshot> elevation,
    const std::vector<FineCellState>& states,
    const std::vector<double>& costs = {},
    const std::uint64_t fine_revision = 1U,
    std::vector<TileIndex> changed_tiles = {},
    std::vector<TileIndex> changed_halo_tiles = {}) {
  const SparseGridGeometry& geometry = elevation->geometry();
  if (states.size() != geometry.CellCount() ||
      (!costs.empty() && costs.size() != states.size())) {
    throw std::invalid_argument("fine fixture dimensions do not match");
  }

  struct MutableFineTile final {
    FineTraversabilityTile::StateArray states;
    FineTraversabilityTile::CostArray costs;

    MutableFineTile() {
      states.fill(FineCellState::kUnknown);
      costs.fill(0.0);
    }
  };
  std::map<TileIndex, MutableFineTile> mutable_tiles;
  const GridIndex min = geometry.min_inclusive();
  const GridIndex max = geometry.max_exclusive();
  std::size_t input_offset = 0U;
  for (std::int64_t y = min.y; y < max.y; ++y) {
    for (std::int64_t x = min.x; x < max.x; ++x, ++input_offset) {
      const GridIndex index{.x = x, .y = y};
      MutableFineTile& tile = mutable_tiles[TileForCell(index)];
      tile.states[TileCellOffset(index)] = states[input_offset];
      tile.costs[TileCellOffset(index)] =
          costs.empty() ? 0.0 : costs[input_offset];
    }
  }
  FineTraversabilityTileDirectory directory(geometry);
  for (auto& [index, tile] : mutable_tiles) {
    directory = directory.WithTile(
        index, std::make_shared<const FineTraversabilityTile>(
                   std::move(tile.states), std::move(tile.costs)));
  }
  if (changed_tiles.empty()) {
    changed_tiles = directory.tile_indices();
  }
  if (changed_halo_tiles.empty()) {
    changed_halo_tiles = changed_tiles;
  }
  return std::make_shared<const FineTraversabilitySnapshot>(
      geometry, elevation->raw_elevation_revision(), fine_revision,
      "wheel-profile", 0.6, 0.25,
      TraversalCostWeights{.slope = 1.0, .relief = 1.0, .clearance = 1.0},
      std::move(elevation), std::move(directory), std::move(changed_tiles),
      std::move(changed_halo_tiles), FineSnapshotMetrics{});
}

TEST(GlobalGuidanceBuilderTest,
     FreezesPositiveCoarseResolutionAndRunsWithoutExternalPrior) {
  EXPECT_THROW(GlobalGuidanceBuilder(0.0), std::invalid_argument);
  EXPECT_THROW(GlobalGuidanceBuilder(
                   std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);

  const auto raw = MakeElevation(Geometry(2U, 2U),
                                 std::vector<float>(4U, 0.0F));
  const auto fine = MakeFine(raw, std::vector<FineCellState>(
                                      4U, FineCellState::kFree));
  const GlobalGuidanceBuilder builder(1.0);
  const auto guidance = builder.Derive(fine, std::nullopt);

  ASSERT_NE(guidance, nullptr);
  EXPECT_DOUBLE_EQ(builder.coarse_resolution_m(), 1.0);
  EXPECT_DOUBLE_EQ(guidance->geometry().resolution_m(), 1.0);
  EXPECT_EQ(guidance->State(GridIndex{.x = 0, .y = 0}),
            GuidanceCellState::kCandidate);
  EXPECT_EQ(guidance->global_guidance_revision(), 1U);
  EXPECT_THROW(GlobalGuidanceBuilder(2.0).Derive(fine, std::nullopt,
                                                guidance),
               std::invalid_argument);
}

TEST(GlobalGuidanceBuilderTest,
     FineHistoryHasPriorityAndLatePriorOnlyFillsUncoveredCoarseCells) {
  const auto raw = MakeElevation(Geometry(2U, 2U),
                                 std::vector<float>(4U, 1.0F));
  const auto fine = MakeFine(raw, std::vector<FineCellState>(
                                      4U, FineCellState::kBlocked));
  const GlobalGuidanceBuilder builder(1.0);
  const auto without_prior = builder.Derive(fine, std::nullopt);
  ASSERT_EQ(without_prior->State(GridIndex{.x = 0, .y = 0}),
            GuidanceCellState::kProvenBlocked);

  const auto prior = MakeElevation(Geometry(2U, 1U, 1.0), {2.0F, 3.0F});
  const auto with_late_prior = builder.Derive(
      fine, GlobalElevationPrior{.elevation = prior}, without_prior);
  ASSERT_NE(with_late_prior, without_prior);
  EXPECT_EQ(with_late_prior->State(GridIndex{.x = 0, .y = 0}),
            GuidanceCellState::kProvenBlocked);
  EXPECT_EQ(with_late_prior->State(GridIndex{.x = 1, .y = 0}),
            GuidanceCellState::kCandidate);

  const auto after_prior_stops =
      builder.Derive(fine, std::nullopt, with_late_prior);
  EXPECT_EQ(after_prior_stops, with_late_prior);
  EXPECT_EQ(after_prior_stops->State(GridIndex{.x = 1, .y = 0}),
            GuidanceCellState::kCandidate);
}

TEST(GlobalGuidanceBuilderTest,
     MixedFineEvidenceIsCandidateAndOnlyCompleteBlockedCoverageProvesBlockage) {
  const auto raw = MakeElevation(Geometry(4U, 2U),
                                 std::vector<float>(8U, 0.0F));
  const std::vector<FineCellState> states{
      FineCellState::kBlocked, FineCellState::kBlocked,
      FineCellState::kBlocked, FineCellState::kFree,
      FineCellState::kBlocked, FineCellState::kBlocked,
      FineCellState::kBlocked, FineCellState::kBlocked};
  const std::vector<double> costs{0.0, 0.0, 0.0, 4.0,
                                  0.0, 0.0, 0.0, 0.0};
  const auto guidance =
      GlobalGuidanceBuilder(1.0).Derive(MakeFine(raw, states, costs),
                                        std::nullopt);

  EXPECT_EQ(guidance->State(GridIndex{.x = 0, .y = 0}),
            GuidanceCellState::kProvenBlocked);
  EXPECT_EQ(guidance->State(GridIndex{.x = 1, .y = 0}),
            GuidanceCellState::kCandidate);
  EXPECT_DOUBLE_EQ(guidance->TerrainRisk(GridIndex{.x = 1, .y = 0}), 4.0);
}

TEST(GlobalGuidanceBuilderTest,
     MissingEvidenceStaysUnknownWhileFiniteFineHistoryRemainsCandidate) {
  const auto raw = MakeSparseElevation();
  const auto fine = MakeFine(raw, std::vector<FineCellState>(
                                      3U, FineCellState::kUnknown));
  const auto guidance =
      GlobalGuidanceBuilder(1.0).Derive(fine, std::nullopt);

  EXPECT_EQ(guidance->State(GridIndex{.x = 0, .y = 0}),
            GuidanceCellState::kCandidate);
  EXPECT_EQ(guidance->State(GridIndex{.x = 1, .y = 0}),
            GuidanceCellState::kUnknown);
  EXPECT_EQ(guidance->State(GridIndex{.x = 2, .y = 0}),
            GuidanceCellState::kCandidate);
  EXPECT_DOUBLE_EQ(guidance->TerrainRisk(GridIndex{.x = 1, .y = 0}), 0.0);
}

TEST(GlobalGuidanceBuilderTest,
     DuplicateDerivationReusesSnapshotAndChangedCellsPublishBoundedHalo) {
  const auto raw = MakeElevation(Geometry(2U, 2U),
                                 std::vector<float>(4U, 0.0F));
  const auto free_fine = MakeFine(raw, std::vector<FineCellState>(
                                          4U, FineCellState::kFree));
  const GlobalGuidanceBuilder builder(1.0);
  const auto first = builder.Derive(free_fine, std::nullopt);
  const auto duplicate = builder.Derive(free_fine, std::nullopt, first);
  ASSERT_EQ(duplicate, first);
  EXPECT_EQ(duplicate->global_guidance_revision(), 1U);

  const auto blocked_fine = MakeFine(
      raw, std::vector<FineCellState>(4U, FineCellState::kBlocked), {}, 2U);
  const auto changed = builder.Derive(blocked_fine, std::nullopt, duplicate);
  ASSERT_NE(changed, duplicate);
  EXPECT_EQ(changed->global_guidance_revision(), 2U);
  ASSERT_EQ(changed->changed_tiles().size(), 1U);
  EXPECT_EQ(changed->changed_tiles().front(), (TileIndex{.x = 0, .y = 0}));
  ASSERT_EQ(changed->changed_halo_tiles().size(), 1U);
  EXPECT_EQ(changed->changed_halo_tiles().front(),
            (TileIndex{.x = 0, .y = 0}));
}

TEST(GlobalGuidanceBuilderTest,
     IncrementalFineChangeSharesUnaffectedGuidanceTile) {
  const auto raw = MakeSparseElevationAt({0.0, 300.0});
  std::vector<FineCellState> states(raw->geometry().CellCount(),
                                    FineCellState::kUnknown);
  states[0U] = FineCellState::kFree;
  states[300U] = FineCellState::kFree;
  const GlobalGuidanceBuilder builder(1.0);
  const auto first = builder.Derive(MakeFine(raw, states), std::nullopt);
  const auto first_tile_zero = first->FindTile(TileIndex{.x = 0, .y = 0});
  const auto first_tile_one = first->FindTile(TileIndex{.x = 1, .y = 0});
  ASSERT_NE(first_tile_zero, nullptr);
  ASSERT_NE(first_tile_one, nullptr);

  states[300U] = FineCellState::kBlocked;
  const auto changed_fine = MakeFine(
      raw, states, {}, 2U, {{.x = 1, .y = 0}}, {{.x = 1, .y = 0}});
  const auto second = builder.Derive(changed_fine, std::nullopt, first);

  EXPECT_EQ(second->State(GridIndex{.x = 300, .y = 0}),
            GuidanceCellState::kProvenBlocked);
  EXPECT_EQ(second->FindTile(TileIndex{.x = 0, .y = 0}), first_tile_zero);
  EXPECT_NE(second->FindTile(TileIndex{.x = 1, .y = 0}), first_tile_one);
}

TEST(GlobalGuidanceBuilderTest,
     ExternalPriorExpansionOnlyReplacesActuallyChangedGuidanceTile) {
  const auto fine_raw = MakeElevation(Geometry(1U, 1U, 1.0), {0.0F});
  const auto fine = MakeFine(fine_raw, {FineCellState::kFree});
  const GlobalGuidanceBuilder builder(1.0);

  PersistentElevationMap prior_map;
  const std::array<float, 1U> value{2.0F};
  ASSERT_EQ(prior_map.Apply(ElevationEvidence{
                .geometry = Geometry(1U, 1U, 1.0, 300.0),
                .elevation_m = value,
                .map_from_source = IdentityMapTransform(),
            }).status,
            ElevationUpdateResult::Status::kApplied);
  const auto first_prior = prior_map.Snapshot();
  const auto first = builder.Derive(
      fine, GlobalElevationPrior{.elevation = first_prior});
  const auto first_tile_one = first->FindTile(TileIndex{.x = 1, .y = 0});
  ASSERT_NE(first_tile_one, nullptr);

  ASSERT_EQ(prior_map.Apply(ElevationEvidence{
                .geometry = Geometry(1U, 1U, 1.0, 600.0),
                .elevation_m = value,
                .map_from_source = IdentityMapTransform(),
            }).status,
            ElevationUpdateResult::Status::kApplied);
  const auto expanded = builder.Derive(
      fine, GlobalElevationPrior{.elevation = prior_map.Snapshot()}, first);
  EXPECT_EQ(expanded->State(GridIndex{.x = 600, .y = 0}),
            GuidanceCellState::kCandidate);
  EXPECT_EQ(expanded->FindTile(TileIndex{.x = 1, .y = 0}), first_tile_one);
  EXPECT_NE(expanded->FindTile(TileIndex{.x = 2, .y = 0}), nullptr);

  const auto after_missing_prior =
      builder.Derive(fine, std::nullopt, expanded);
  EXPECT_EQ(after_missing_prior, expanded);
  EXPECT_EQ(after_missing_prior->State(GridIndex{.x = 600, .y = 0}),
            GuidanceCellState::kCandidate);
}

TEST(GlobalGuidanceBuilderTest,
     NearIntegerWorldBoundaryDoesNotCreateAnEighthCoarseCell) {
  const auto raw = MakeElevation(Geometry(7U, 1U, 0.3),
                                 std::vector<float>(7U, 0.0F));
  const auto guidance = GlobalGuidanceBuilder(0.3).Derive(
      MakeFine(raw,
               std::vector<FineCellState>(7U, FineCellState::kFree)),
      std::nullopt);

  EXPECT_EQ(guidance->geometry().min_inclusive(),
            (GridIndex{.x = 0, .y = 0}));
  EXPECT_EQ(guidance->geometry().max_exclusive(),
            (GridIndex{.x = 7, .y = 1}));
  EXPECT_EQ(guidance->State(GridIndex{.x = 6, .y = 0}),
            GuidanceCellState::kCandidate);
  EXPECT_EQ(guidance->State(GridIndex{.x = 7, .y = 0}),
            GuidanceCellState::kUnknown);
}

TEST(GlobalGuidanceBuilderTest,
     RejectsFiniteResolutionWhoseCellAreaOverflows) {
  const double overflowing_resolution =
      std::sqrt(std::numeric_limits<double>::max()) * 2.0;
  ASSERT_TRUE(std::isfinite(overflowing_resolution));
  ASSERT_FALSE(std::isfinite(overflowing_resolution * overflowing_resolution));
  EXPECT_THROW((void)GlobalGuidanceBuilder(overflowing_resolution),
               std::invalid_argument);
}

TEST(GlobalGuidanceBuilderTest, RejectsMissingOrInvalidInputs) {
  const auto raw = MakeElevation(Geometry(2U, 2U),
                                 std::vector<float>(4U, 0.0F));
  const auto fine = MakeFine(raw, std::vector<FineCellState>(
                                      4U, FineCellState::kFree));
  const GlobalGuidanceBuilder builder(1.0);
  EXPECT_THROW(builder.Derive(nullptr, std::nullopt), std::invalid_argument);
  EXPECT_THROW(builder.Derive(
                   fine, GlobalElevationPrior{.elevation = nullptr}),
               std::invalid_argument);
}

}  // namespace
}  // namespace lunar::incremental_navigation
