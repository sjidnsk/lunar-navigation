#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/global_guidance_snapshot.hpp"
#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#include "lunar_incremental_navigation_core/types/grid_geometry.hpp"
#include "lunar_incremental_navigation_core/types/persistent_tile_directory.hpp"

namespace lunar::incremental_navigation {
namespace {

static_assert(std::is_same_v<decltype(GridIndex::x), std::int64_t>);
static_assert(std::is_same_v<decltype(GridIndex::y), std::int64_t>);
static_assert(std::is_same_v<decltype(TileIndex::x), std::int64_t>);
static_assert(std::is_same_v<decltype(TileIndex::y), std::int64_t>);
static_assert(std::is_same_v<decltype(Pose2::position_m), Vec2>);
static_assert(std::is_same_v<decltype(Pose2::yaw_rad), double>);
static_assert(std::is_same_v<decltype(PathPoint::pose), Pose3>);
static_assert(std::is_same_v<decltype(PathPoint::phase), StartPhase>);
static_assert(std::is_same_v<decltype(LocalPlanResult::raw_path),
                             std::vector<PathPoint>>);
static_assert(std::is_same_v<decltype(LocalPlanResult::path),
                             std::vector<PathPoint>>);
static_assert(std::is_same_v<decltype(LocalPlanResult::statistics),
                             SearchStatistics>);

template <typename T>
concept HasMutableHeader = requires(T value) { value.header; };

template <typename T>
concept HasTimestamp = requires(T value) { value.timestamp; };

static_assert(!HasMutableHeader<FineTraversabilitySnapshot>);
static_assert(!HasMutableHeader<GlobalGuidanceSnapshot>);
static_assert(!HasMutableHeader<RequestLocalPlanningView>);
static_assert(!HasTimestamp<FineTraversabilitySnapshot>);
static_assert(!HasTimestamp<GlobalGuidanceSnapshot>);
static_assert(!HasTimestamp<RequestLocalPlanningView>);
static_assert(std::is_same_v<
              decltype(std::declval<const FineTraversabilitySnapshot&>()
                           .geometry()),
              const SparseGridGeometry&>);
static_assert(std::is_same_v<
              decltype(std::declval<const FineTraversabilitySnapshot&>()
                           .elevation()),
              std::shared_ptr<const ElevationSnapshot>>);
static_assert(std::is_same_v<
              decltype(std::declval<const FineTraversabilitySnapshot&>()
                           .cost_weights()),
              const TraversalCostWeights&>);
static_assert(std::is_same_v<
              decltype(std::declval<const FineTraversabilitySnapshot&>()
                           .metrics()),
              const FineSnapshotMetrics&>);
static_assert(std::is_same_v<
              decltype(std::declval<const FineTraversabilitySnapshot&>()
                           .FindTile(TileIndex{})),
              std::shared_ptr<const FineTraversabilityTile>>);
static_assert(std::is_same_v<
              decltype(std::declval<const FineTraversabilitySnapshot&>()
                           .tile_directory()),
              const FineTraversabilityTileDirectory&>);
static_assert(std::is_same_v<
              decltype(std::declval<const GlobalGuidanceSnapshot&>()
                           .FindTile(TileIndex{})),
              std::shared_ptr<const GlobalGuidanceTile>>);
static_assert(!std::is_assignable_v<FineTraversabilityTile&,
                                    const FineTraversabilityTile&>);
static_assert(!std::is_assignable_v<FineTraversabilityTile&,
                                    FineTraversabilityTile&&>);
static_assert(!std::is_assignable_v<GlobalGuidanceTile&,
                                    const GlobalGuidanceTile&>);
static_assert(!std::is_assignable_v<GlobalGuidanceTile&,
                                    GlobalGuidanceTile&&>);
static_assert(std::is_same_v<
              decltype(std::declval<const FineTraversabilityTile&>()
                           .IsUnobserved(std::size_t{})),
              bool>);
static_assert(std::is_same_v<
              decltype(std::declval<const GlobalGuidanceTile&>()
                           .IsUnobserved(std::size_t{})),
              bool>);
static_assert(std::is_same_v<
              FineTraversabilityTileDirectory,
              PersistentTileDirectory<FineTraversabilityTile>>);
static_assert(std::is_same_v<GlobalGuidanceTileDirectory,
                             PersistentTileDirectory<GlobalGuidanceTile>>);
static_assert(!std::is_default_constructible_v<
              FineTraversabilityTileDirectory>);
static_assert(!std::is_default_constructible_v<GlobalGuidanceTileDirectory>);
static_assert(std::is_same_v<
              decltype(std::declval<const RequestLocalPlanningView&>().base()),
              std::shared_ptr<const FineTraversabilitySnapshot>>);
static_assert(std::is_same_v<decltype(SnapshotBundle::fine),
                             std::shared_ptr<const FineTraversabilitySnapshot>>);
static_assert(std::is_same_v<
              decltype(SnapshotBundle::guidance),
              std::shared_ptr<const GlobalGuidanceSnapshot>>);

[[nodiscard]] GridGeometry DenseGeometry(
    const std::size_t width = 3U, const std::size_t height = 2U,
    const double resolution_m = 0.2,
    const Vec3 origin_m = Vec3{.x = -2.0, .y = 4.5, .z = 1.25}) {
  return GridGeometry{.frame_id = "map",
                      .width = width,
                      .height = height,
                      .resolution_m = resolution_m,
                      .origin_m = origin_m};
}

[[nodiscard]] RigidTransform IdentityMapTransform() {
  return RigidTransform{.parent_frame = "map", .child_frame = "map"};
}

[[nodiscard]] std::shared_ptr<const ElevationSnapshot> MakeElevationSnapshot(
    const GridGeometry& geometry) {
  PersistentElevationMap map;
  const std::vector<float> values(geometry.CellCount(), 1.0F);
  const auto result = map.Apply(ElevationEvidence{
      .geometry = geometry,
      .elevation_m = values,
      .map_from_source = IdentityMapTransform(),
  });
  if (result.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("test elevation fixture was rejected");
  }
  return map.Snapshot();
}

[[nodiscard]] std::shared_ptr<const ElevationSnapshot> MakeWideElevation(
    const double far_x_m) {
  PersistentElevationMap map;
  const std::array<float, 1U> value{1.0F};
  const auto apply = [&](const double origin_x_m) {
    return map.Apply(ElevationEvidence{
        .geometry = DenseGeometry(1U, 1U, 0.2,
                                  Vec3{.x = origin_x_m, .y = 0.0, .z = 0.0}),
        .elevation_m = value,
        .map_from_source = IdentityMapTransform(),
    });
  };
  if (apply(0.0).status != ElevationUpdateResult::Status::kApplied ||
      apply(far_x_m).status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("wide elevation fixture was rejected");
  }
  return map.Snapshot();
}

[[nodiscard]] std::shared_ptr<const ElevationSnapshot> MakeNegativeElevation() {
  PersistentElevationMap map;
  const std::array<float, 1U> value{1.0F};
  const auto apply = [&](const double origin_x_m) {
    return map.Apply(ElevationEvidence{
        .geometry = DenseGeometry(1U, 1U, 0.2,
                                  Vec3{.x = origin_x_m, .y = 0.0, .z = 0.0}),
        .elevation_m = value,
        .map_from_source = IdentityMapTransform(),
    });
  };
  if (apply(0.0).status != ElevationUpdateResult::Status::kApplied ||
      apply(-0.4).status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("negative elevation fixture was rejected");
  }
  return map.Snapshot();
}

[[nodiscard]] std::shared_ptr<const FineTraversabilityTile> MakeFineTile(
    const FineCellState initial_state = FineCellState::kUnknown,
    const double initial_cost = 0.0) {
  FineTraversabilityTile::StateArray states;
  states.fill(initial_state);
  FineTraversabilityTile::CostArray costs;
  costs.fill(initial_cost);
  return std::make_shared<const FineTraversabilityTile>(std::move(states),
                                                        std::move(costs));
}

[[nodiscard]] std::shared_ptr<const FineTraversabilityTile> MakeFineTileAt(
    const GridIndex index, const FineCellState state,
    const double traversal_cost = 0.0) {
  FineTraversabilityTile::StateArray states;
  states.fill(FineCellState::kUnknown);
  FineTraversabilityTile::CostArray costs;
  costs.fill(0.0);
  states[TileCellOffset(index)] = state;
  costs[TileCellOffset(index)] = traversal_cost;
  return std::make_shared<const FineTraversabilityTile>(std::move(states),
                                                        std::move(costs));
}

[[nodiscard]] std::shared_ptr<const FineTraversabilityTile> MakeMixedFineTile() {
  FineTraversabilityTile::StateArray states;
  states.fill(FineCellState::kUnknown);
  FineTraversabilityTile::CostArray costs;
  costs.fill(0.0);
  states[TileCellOffset(GridIndex{.x = 1, .y = 0})] = FineCellState::kFree;
  costs[TileCellOffset(GridIndex{.x = 1, .y = 0})] = 1.25;
  states[TileCellOffset(GridIndex{.x = 2, .y = 0})] = FineCellState::kBlocked;
  return std::make_shared<const FineTraversabilityTile>(std::move(states),
                                                        std::move(costs));
}

[[nodiscard]] std::shared_ptr<const GlobalGuidanceTile> MakeGuidanceTile(
    const GuidanceCellState initial_state = GuidanceCellState::kCandidate,
    const double initial_risk = 0.0) {
  GlobalGuidanceTile::StateArray states;
  states.fill(initial_state);
  GlobalGuidanceTile::RiskArray risks;
  risks.fill(initial_risk);
  return std::make_shared<const GlobalGuidanceTile>(std::move(states),
                                                    std::move(risks));
}

[[nodiscard]] std::shared_ptr<const GlobalGuidanceTile> MakeGuidanceTileAt(
    const GridIndex index, const GuidanceCellState state,
    const double terrain_risk = 0.0) {
  GlobalGuidanceTile::StateArray states;
  states.fill(GuidanceCellState::kUnknown);
  GlobalGuidanceTile::RiskArray risks;
  risks.fill(0.0);
  states[TileCellOffset(index)] = state;
  risks[TileCellOffset(index)] = terrain_risk;
  return std::make_shared<const GlobalGuidanceTile>(std::move(states),
                                                    std::move(risks));
}

[[nodiscard]] FineTraversabilityTileDirectory MakeFineDirectory(
    const SparseGridGeometry& geometry,
    std::initializer_list<std::pair<
        TileIndex, std::shared_ptr<const FineTraversabilityTile>>> entries) {
  FineTraversabilityTileDirectory directory(geometry);
  for (const auto& [index, tile] : entries) {
    directory = directory.WithTile(index, tile);
  }
  return directory;
}

[[nodiscard]] GlobalGuidanceTileDirectory MakeGuidanceDirectory(
    const SparseGridGeometry& geometry,
    std::initializer_list<
        std::pair<TileIndex, std::shared_ptr<const GlobalGuidanceTile>>>
        entries) {
  GlobalGuidanceTileDirectory directory(geometry);
  for (const auto& [index, tile] : entries) {
    directory = directory.WithTile(index, tile);
  }
  return directory;
}

[[nodiscard]] FineTraversabilitySnapshot MakeFineSnapshot(
    const std::shared_ptr<const ElevationSnapshot>& elevation,
    const SparseGridGeometry& geometry,
    FineTraversabilityTileDirectory tiles,
    const std::uint64_t fine_revision = 1U,
    std::vector<TileIndex> changed_tiles = {},
    std::vector<TileIndex> changed_halo_tiles = {}) {
  return FineTraversabilitySnapshot(
      geometry, elevation->raw_elevation_revision(), fine_revision,
      "wheel-profile", 0.6, 0.25,
      TraversalCostWeights{.slope = 1.0, .relief = 2.0, .clearance = 3.0},
      elevation, std::move(tiles), std::move(changed_tiles),
      std::move(changed_halo_tiles),
      FineSnapshotMetrics{.updated_cells = 1U,
                          .allocated_cells = kGridTileCellCount,
                          .allocated_bytes = sizeof(FineTraversabilityTile),
                          .derivation_ms = 1.5});
}

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot>
MakeBasicFineSnapshot() {
  const auto elevation = MakeElevationSnapshot(DenseGeometry());
  const auto tiles = MakeFineDirectory(
      elevation->geometry(),
      {{TileIndex{.x = 0, .y = 0}, MakeMixedFineTile()}});
  return std::make_shared<const FineTraversabilitySnapshot>(MakeFineSnapshot(
      elevation, elevation->geometry(), tiles, 7U,
      {{.x = 0, .y = 0}}, {{.x = 0, .y = 0}}));
}

TEST(PlanningSnapshotContractTest, DenseGeometryRejectsNonFiniteWorldExtent) {
  GridGeometry geometry = DenseGeometry();
  EXPECT_TRUE(geometry.valid());
  geometry.origin_m.x = std::numeric_limits<double>::max();
  geometry.resolution_m = std::numeric_limits<double>::max();
  EXPECT_FALSE(geometry.valid());
  geometry = DenseGeometry();
  geometry.width = std::numeric_limits<std::size_t>::max();
  geometry.height = 2U;
  EXPECT_FALSE(geometry.valid());
  EXPECT_EQ(geometry.CellCount(), 0U);
}

TEST(PlanningSnapshotContractTest,
     SparseGeometrySupportsNegativeBoundsAndRejectsNonFiniteWorldExtent) {
  const SparseGridGeometry valid("map", 0.2, Vec3{},
                                 GridIndex{.x = -3, .y = -2},
                                 GridIndex{.x = 4, .y = 5});
  EXPECT_TRUE(valid.valid());
  EXPECT_TRUE(valid.Contains(GridIndex{.x = -3, .y = -2}));
  EXPECT_TRUE(valid.Contains(TileIndex{.x = -1, .y = -1}));

  const SparseGridGeometry overflow(
      "map", std::numeric_limits<double>::max(), Vec3{},
      GridIndex{.x = -2, .y = 0}, GridIndex{.x = 2, .y = 1});
  EXPECT_FALSE(overflow.valid());
}

TEST(PlanningSnapshotContractTest, TileConstructorsRejectInvalidCosts) {
  FineTraversabilityTile::StateArray fine_states;
  fine_states.fill(FineCellState::kFree);
  FineTraversabilityTile::CostArray fine_costs;
  fine_costs.fill(0.0);
  fine_costs[17U] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(FineTraversabilityTile(fine_states, fine_costs),
               std::invalid_argument);
  fine_costs[17U] = 0.0;
  fine_states[17U] = static_cast<FineCellState>(255U);
  EXPECT_THROW(FineTraversabilityTile(fine_states, fine_costs),
               std::invalid_argument);

  GlobalGuidanceTile::StateArray guidance_states;
  guidance_states.fill(GuidanceCellState::kCandidate);
  GlobalGuidanceTile::RiskArray risks;
  risks.fill(0.0);
  risks[19U] = -0.1;
  EXPECT_THROW(GlobalGuidanceTile(guidance_states, risks),
               std::invalid_argument);
  risks[19U] = 0.0;
  guidance_states[19U] = static_cast<GuidanceCellState>(255U);
  EXPECT_THROW(GlobalGuidanceTile(guidance_states, risks),
               std::invalid_argument);
}

TEST(PlanningSnapshotContractTest,
     FineSnapshotSharesUnchangedTilesAcrossRevisions) {
  const auto elevation = MakeElevationSnapshot(DenseGeometry(300U, 1U));
  const auto unchanged = MakeFineTileAt(GridIndex{.x = 0, .y = 0},
                                        FineCellState::kFree, 0.2);
  const auto old_changed = MakeFineTile(FineCellState::kUnknown, 0.0);
  const auto first_tiles = MakeFineDirectory(elevation->geometry(), {
      {TileIndex{.x = 0, .y = 0}, unchanged},
      {TileIndex{.x = 1, .y = 0}, old_changed}});
  const auto first = MakeFineSnapshot(elevation, elevation->geometry(),
                                      first_tiles, 1U, {{.x = 1, .y = 0}});

  const auto replacement = MakeFineTileAt(GridIndex{.x = 256, .y = 0},
                                          FineCellState::kBlocked);
  const auto second_tiles =
      first_tiles.WithTile(TileIndex{.x = 1, .y = 0}, replacement);
  const auto second = MakeFineSnapshot(elevation, elevation->geometry(),
                                       second_tiles, 2U,
                                       {{.x = 1, .y = 0}});

  EXPECT_EQ(first.FindTile(TileIndex{.x = 0, .y = 0}),
            second.FindTile(TileIndex{.x = 0, .y = 0}));
  EXPECT_NE(first.FindTile(TileIndex{.x = 1, .y = 0}),
            second.FindTile(TileIndex{.x = 1, .y = 0}));
  EXPECT_EQ(second.FindTile(TileIndex{.x = 1, .y = 0}), replacement);
}

TEST(PlanningSnapshotContractTest,
     GlobalSnapshotSharesUnchangedTilesAcrossRevisions) {
  const SparseGridGeometry geometry("map", 1.0, Vec3{},
                                    GridIndex{.x = 0, .y = 0},
                                    GridIndex{.x = 300, .y = 1});
  const auto unchanged = MakeGuidanceTileAt(
      GridIndex{.x = 0, .y = 0}, GuidanceCellState::kCandidate);
  const auto old_changed = MakeGuidanceTile(GuidanceCellState::kUnknown);
  const auto first_tiles = MakeGuidanceDirectory(geometry, {
      {TileIndex{.x = 0, .y = 0}, unchanged},
      {TileIndex{.x = 1, .y = 0}, old_changed}});
  const GlobalGuidanceSnapshot first(
      geometry, 1U, 1U, 1U, "wheel-profile", first_tiles,
      {{.x = 1, .y = 0}}, {});

  const auto replacement = MakeGuidanceTileAt(
      GridIndex{.x = 256, .y = 0}, GuidanceCellState::kProvenBlocked);
  const auto second_tiles =
      first_tiles.WithTile(TileIndex{.x = 1, .y = 0}, replacement);
  const GlobalGuidanceSnapshot second(
      geometry, 1U, 1U, 2U, "wheel-profile", second_tiles,
      {{.x = 1, .y = 0}}, {});

  EXPECT_EQ(first.FindTile(TileIndex{.x = 0, .y = 0}),
            second.FindTile(TileIndex{.x = 0, .y = 0}));
  EXPECT_NE(first.FindTile(TileIndex{.x = 1, .y = 0}),
            second.FindTile(TileIndex{.x = 1, .y = 0}));
}

TEST(PlanningSnapshotContractTest,
     PersistentDirectoriesCopyOnlyTheUpdatedPathAt4096Tiles) {
  constexpr std::int64_t kDirectoryTileCount = 4096;
  const SparseGridGeometry directory_geometry(
      "map", 1.0, Vec3{}, GridIndex{.x = 0, .y = 0},
      GridIndex{.x = kDirectoryTileCount * kGridTileWidthCells,
                .y = kGridTileWidthCells});
  const auto fine_common = MakeFineTile(FineCellState::kFree, 0.2);
  FineTraversabilityTileDirectory fine_directory(directory_geometry);
  for (std::int64_t x = 0; x < kDirectoryTileCount; ++x) {
    fine_directory =
        fine_directory.WithTile(TileIndex{.x = x, .y = 0}, fine_common);
  }
  const auto old_fine_directory = fine_directory;
  const auto fine_replacement = MakeFineTile(FineCellState::kBlocked);
  const auto next_fine_directory = old_fine_directory.WithTile(
      TileIndex{.x = 2048, .y = 0}, fine_replacement);

  EXPECT_EQ(old_fine_directory.tile_count(), 4096U);
  EXPECT_EQ(next_fine_directory.tile_count(), 4096U);
  EXPECT_LE(next_fine_directory.last_update_copied_nodes(), 32U);
  EXPECT_EQ(old_fine_directory.Find(TileIndex{.x = 2048, .y = 0}),
            fine_common);
  EXPECT_EQ(next_fine_directory.Find(TileIndex{.x = 2048, .y = 0}),
            fine_replacement);
  EXPECT_EQ(old_fine_directory.Find(TileIndex{.x = 17, .y = 0}),
            next_fine_directory.Find(TileIndex{.x = 17, .y = 0}));

  const auto guidance_common = MakeGuidanceTile();
  GlobalGuidanceTileDirectory guidance_directory(directory_geometry);
  for (std::int64_t x = 0; x < kDirectoryTileCount; ++x) {
    guidance_directory = guidance_directory.WithTile(
        TileIndex{.x = x, .y = 0}, guidance_common);
  }
  const auto old_guidance_directory = guidance_directory;
  const auto guidance_replacement =
      MakeGuidanceTile(GuidanceCellState::kProvenBlocked);
  const auto next_guidance_directory = old_guidance_directory.WithTile(
      TileIndex{.x = 2048, .y = 0}, guidance_replacement);

  EXPECT_EQ(old_guidance_directory.tile_count(), 4096U);
  EXPECT_EQ(next_guidance_directory.tile_count(), 4096U);
  EXPECT_LE(next_guidance_directory.last_update_copied_nodes(), 32U);
  EXPECT_EQ(old_guidance_directory.Find(TileIndex{.x = 2048, .y = 0}),
            guidance_common);
  EXPECT_EQ(next_guidance_directory.Find(TileIndex{.x = 2048, .y = 0}),
            guidance_replacement);
  EXPECT_EQ(old_guidance_directory.Find(TileIndex{.x = 17, .y = 0}),
            next_guidance_directory.Find(TileIndex{.x = 17, .y = 0}));

  EXPECT_THROW(
      FineTraversabilityTileDirectory(directory_geometry).WithTile(
          TileIndex{}, std::shared_ptr<const FineTraversabilityTile>{}),
      std::invalid_argument);
  EXPECT_THROW(
      GlobalGuidanceTileDirectory(directory_geometry).WithTile(
          TileIndex{}, std::shared_ptr<const GlobalGuidanceTile>{}),
      std::invalid_argument);
}

TEST(PlanningSnapshotContractTest,
     PersistentDirectoryBindsBoundsAndExpandsBySharingItsRoot) {
  const SparseGridGeometry original_geometry(
      "map", 0.2, Vec3{}, GridIndex{.x = 0, .y = 0},
      GridIndex{.x = 256, .y = 1});
  const SparseGridGeometry expanded_geometry(
      "map", 0.2, Vec3{}, GridIndex{.x = -256, .y = 0},
      GridIndex{.x = 512, .y = 1});
  const auto common = MakeFineTile(FineCellState::kUnknown);
  const auto original = FineTraversabilityTileDirectory(original_geometry)
                            .WithTile(TileIndex{.x = 0, .y = 0}, common);
  const auto expanded = original.WithGeometry(expanded_geometry);

  EXPECT_TRUE(original.shares_root_with(expanded));
  EXPECT_EQ(expanded.tile_count(), original.tile_count());
  EXPECT_EQ(expanded.last_update_copied_nodes(), 0U);
  EXPECT_EQ(original.Find(TileIndex{.x = 0, .y = 0}),
            expanded.Find(TileIndex{.x = 0, .y = 0}));
  EXPECT_EQ(original.geometry().min_inclusive(),
            (GridIndex{.x = 0, .y = 0}));
  EXPECT_EQ(expanded.geometry().min_inclusive(),
            (GridIndex{.x = -256, .y = 0}));
  EXPECT_TRUE(expanded.WithTile(TileIndex{.x = -1, .y = 0}, common)
                  .Find(TileIndex{.x = -1, .y = 0}));

  const SparseGridGeometry shrink(
      "map", 0.2, Vec3{}, GridIndex{.x = 0, .y = 0},
      GridIndex{.x = 128, .y = 1});
  EXPECT_THROW(original.WithGeometry(shrink), std::invalid_argument);
  const SparseGridGeometry wrong_resolution(
      "map", 0.25, Vec3{}, GridIndex{.x = -256, .y = 0},
      GridIndex{.x = 512, .y = 1});
  EXPECT_THROW(original.WithGeometry(wrong_resolution),
               std::invalid_argument);
  const SparseGridGeometry wrong_origin(
      "map", 0.2, Vec3{.x = 0.2}, GridIndex{.x = -256, .y = 0},
      GridIndex{.x = 512, .y = 1});
  EXPECT_THROW(original.WithGeometry(wrong_origin), std::invalid_argument);
  EXPECT_THROW(
      FineTraversabilityTileDirectory(SparseGridGeometry{}),
      std::invalid_argument);
}

TEST(PlanningSnapshotContractTest,
     BoundDirectoriesRejectOutOfBoundsTilesAndSnapshotGeometryMismatch) {
  const SparseGridGeometry one_tile_geometry(
      "map", 0.2, Vec3{}, GridIndex{.x = 0, .y = 0},
      GridIndex{.x = 256, .y = 1});
  EXPECT_THROW(
      FineTraversabilityTileDirectory(one_tile_geometry)
          .WithTile(TileIndex{.x = 1, .y = 0}, MakeFineTile()),
      std::invalid_argument);
  EXPECT_THROW(
      GlobalGuidanceTileDirectory(one_tile_geometry)
          .WithTile(TileIndex{.x = -1, .y = 0}, MakeGuidanceTile()),
      std::invalid_argument);

  const auto elevation = MakeElevationSnapshot(DenseGeometry(300U, 1U));
  const auto fine_tiles =
      FineTraversabilityTileDirectory(one_tile_geometry)
          .WithTile(TileIndex{.x = 0, .y = 0}, MakeFineTile());
  EXPECT_THROW(
      MakeFineSnapshot(elevation, elevation->geometry(), fine_tiles),
      std::invalid_argument);

  const SparseGridGeometry expanded_geometry(
      "map", 0.2, Vec3{}, GridIndex{.x = -256, .y = 0},
      GridIndex{.x = 512, .y = 1});
  const auto guidance_tiles =
      GlobalGuidanceTileDirectory(expanded_geometry)
          .WithTile(TileIndex{.x = 0, .y = 0},
                    MakeGuidanceTile(GuidanceCellState::kUnknown));
  EXPECT_THROW(GlobalGuidanceSnapshot(one_tile_geometry, 1U, 1U, 1U,
                                      "profile", guidance_tiles, {}, {}),
               std::invalid_argument);
}

TEST(PlanningSnapshotContractTest,
     PartialBoundaryTilesKeepCellsOutsideOldBoundsUnobserved) {
  const auto elevation = MakeElevationSnapshot(DenseGeometry(3U, 1U));
  const SparseGridGeometry& expanded_fine_geometry = elevation->geometry();
  const SparseGridGeometry old_fine_geometry(
      expanded_fine_geometry.frame_id(), expanded_fine_geometry.resolution_m(),
      expanded_fine_geometry.origin_m(), GridIndex{.x = 0, .y = 0},
      GridIndex{.x = 2, .y = 1});
  FineTraversabilityTile::StateArray hidden_fine_states;
  hidden_fine_states.fill(FineCellState::kUnknown);
  FineTraversabilityTile::CostArray hidden_fine_costs;
  hidden_fine_costs.fill(0.0);
  hidden_fine_states[TileCellOffset(GridIndex{.x = 2, .y = 0})] =
      FineCellState::kFree;
  const auto hidden_fine = std::make_shared<const FineTraversabilityTile>(
      std::move(hidden_fine_states), std::move(hidden_fine_costs));
  EXPECT_THROW(
      FineTraversabilityTileDirectory(old_fine_geometry)
          .WithTile(TileIndex{.x = 0, .y = 0}, hidden_fine),
      std::invalid_argument);

  const auto old_fine_directory =
      FineTraversabilityTileDirectory(old_fine_geometry)
          .WithTile(TileIndex{.x = 0, .y = 0}, MakeFineTile());
  const auto expanded_fine_directory =
      old_fine_directory.WithGeometry(expanded_fine_geometry);
  const auto before_fine_replacement = MakeFineSnapshot(
      elevation, expanded_fine_geometry, expanded_fine_directory);
  EXPECT_EQ(before_fine_replacement.State(GridIndex{.x = 2, .y = 0}),
            FineCellState::kUnknown);
  EXPECT_DOUBLE_EQ(
      before_fine_replacement.TraversalCost(GridIndex{.x = 2, .y = 0}),
      0.0);

  FineTraversabilityTile::StateArray replacement_fine_states;
  replacement_fine_states.fill(FineCellState::kUnknown);
  FineTraversabilityTile::CostArray replacement_fine_costs;
  replacement_fine_costs.fill(0.0);
  replacement_fine_states[TileCellOffset(GridIndex{.x = 2, .y = 0})] =
      FineCellState::kFree;
  replacement_fine_costs[TileCellOffset(GridIndex{.x = 2, .y = 0})] = 0.5;
  const auto replacement_fine =
      std::make_shared<const FineTraversabilityTile>(
          std::move(replacement_fine_states),
          std::move(replacement_fine_costs));
  const auto updated_fine_directory = expanded_fine_directory.WithTile(
      TileIndex{.x = 0, .y = 0}, replacement_fine);
  const auto after_fine_replacement = MakeFineSnapshot(
      elevation, expanded_fine_geometry, updated_fine_directory, 2U);
  EXPECT_EQ(after_fine_replacement.State(GridIndex{.x = 2, .y = 0}),
            FineCellState::kFree);
  EXPECT_DOUBLE_EQ(
      after_fine_replacement.TraversalCost(GridIndex{.x = 2, .y = 0}),
      0.5);

  const SparseGridGeometry old_guidance_geometry(
      "map", 1.0, Vec3{}, GridIndex{.x = 0, .y = 0},
      GridIndex{.x = 2, .y = 1});
  const SparseGridGeometry expanded_guidance_geometry(
      "map", 1.0, Vec3{}, GridIndex{.x = 0, .y = 0},
      GridIndex{.x = 3, .y = 1});
  GlobalGuidanceTile::StateArray hidden_guidance_states;
  hidden_guidance_states.fill(GuidanceCellState::kUnknown);
  GlobalGuidanceTile::RiskArray hidden_guidance_risks;
  hidden_guidance_risks.fill(0.0);
  hidden_guidance_states[TileCellOffset(GridIndex{.x = 2, .y = 0})] =
      GuidanceCellState::kCandidate;
  const auto hidden_guidance = std::make_shared<const GlobalGuidanceTile>(
      std::move(hidden_guidance_states), std::move(hidden_guidance_risks));
  EXPECT_THROW(
      GlobalGuidanceTileDirectory(old_guidance_geometry)
          .WithTile(TileIndex{.x = 0, .y = 0}, hidden_guidance),
      std::invalid_argument);

  const auto old_guidance_directory =
      GlobalGuidanceTileDirectory(old_guidance_geometry)
          .WithTile(TileIndex{.x = 0, .y = 0},
                    MakeGuidanceTile(GuidanceCellState::kUnknown));
  const auto expanded_guidance_directory =
      old_guidance_directory.WithGeometry(expanded_guidance_geometry);
  const GlobalGuidanceSnapshot before_guidance_replacement(
      expanded_guidance_geometry, 1U, 1U, 1U, "profile",
      expanded_guidance_directory, {}, {});
  EXPECT_EQ(before_guidance_replacement.State(GridIndex{.x = 2, .y = 0}),
            GuidanceCellState::kUnknown);
  EXPECT_DOUBLE_EQ(
      before_guidance_replacement.TerrainRisk(GridIndex{.x = 2, .y = 0}),
      0.0);

  GlobalGuidanceTile::StateArray replacement_guidance_states;
  replacement_guidance_states.fill(GuidanceCellState::kUnknown);
  GlobalGuidanceTile::RiskArray replacement_guidance_risks;
  replacement_guidance_risks.fill(0.0);
  replacement_guidance_states[TileCellOffset(GridIndex{.x = 2, .y = 0})] =
      GuidanceCellState::kCandidate;
  replacement_guidance_risks[TileCellOffset(GridIndex{.x = 2, .y = 0})] =
      0.75;
  const auto replacement_guidance =
      std::make_shared<const GlobalGuidanceTile>(
          std::move(replacement_guidance_states),
          std::move(replacement_guidance_risks));
  const auto updated_guidance_directory =
      expanded_guidance_directory.WithTile(TileIndex{.x = 0, .y = 0},
                                           replacement_guidance);
  const GlobalGuidanceSnapshot after_guidance_replacement(
      expanded_guidance_geometry, 1U, 1U, 2U, "profile",
      updated_guidance_directory, {}, {});
  EXPECT_EQ(after_guidance_replacement.State(GridIndex{.x = 2, .y = 0}),
            GuidanceCellState::kCandidate);
  EXPECT_DOUBLE_EQ(
      after_guidance_replacement.TerrainRisk(GridIndex{.x = 2, .y = 0}),
      0.75);
}

TEST(PlanningSnapshotContractTest,
     KilometerBoundsRemainSparseAndMissingTilesStayUnknown) {
  const auto elevation = MakeWideElevation(1000.0);
  const auto tiles = MakeFineDirectory(
      elevation->geometry(),
      {{TileIndex{.x = 0, .y = 0},
        MakeFineTileAt(GridIndex{.x = 0, .y = 0}, FineCellState::kFree)}});
  const auto snapshot = MakeFineSnapshot(elevation, elevation->geometry(),
                                         tiles);

  EXPECT_EQ(snapshot.metrics().allocated_cells, kGridTileCellCount);
  EXPECT_TRUE(snapshot.FindTile(TileIndex{.x = 0, .y = 0}));
  EXPECT_FALSE(snapshot.FindTile(TileIndex{.x = 10, .y = 0}));
  EXPECT_EQ(snapshot.State(GridIndex{.x = 2500, .y = 0}),
            FineCellState::kUnknown);
  EXPECT_DOUBLE_EQ(snapshot.TraversalCost(GridIndex{.x = 2500, .y = 0}),
                   0.0);
}

TEST(PlanningSnapshotContractTest,
     FineSnapshotRequiresExactCanonicalLatticeAndRawBounds) {
  const auto elevation = MakeElevationSnapshot(DenseGeometry());
  const auto tiles = MakeFineDirectory(
      elevation->geometry(),
      {{TileIndex{.x = 0, .y = 0}, MakeFineTile()}});
  const auto construct = [&](const SparseGridGeometry& geometry) {
    return MakeFineSnapshot(elevation, geometry, tiles);
  };

  EXPECT_THROW(
      construct(SparseGridGeometry("map", 0.25,
                                   elevation->geometry().origin_m(),
                                   GridIndex{.x = 0, .y = 0},
                                   GridIndex{.x = 3, .y = 2})),
      std::invalid_argument);
  Vec3 wrong_origin = elevation->geometry().origin_m();
  wrong_origin.x += 0.2;
  EXPECT_THROW(construct(SparseGridGeometry(
                   "map", 0.2, wrong_origin, GridIndex{.x = 0, .y = 0},
                   GridIndex{.x = 3, .y = 2})),
               std::invalid_argument);
  EXPECT_THROW(
      construct(SparseGridGeometry("map", 0.2,
                                   elevation->geometry().origin_m(),
                                   GridIndex{.x = 0, .y = 0},
                                   GridIndex{.x = 4, .y = 2})),
      std::invalid_argument);
}

TEST(PlanningSnapshotContractTest,
     SnapshotLineageWeightsMetricsAndChangedTilesRemainValidated) {
  const auto elevation = MakeElevationSnapshot(DenseGeometry());
  const auto tiles = MakeFineDirectory(
      elevation->geometry(),
      {{TileIndex{.x = 0, .y = 0}, MakeMixedFineTile()}});
  const auto snapshot = MakeFineSnapshot(
      elevation, elevation->geometry(), tiles, 7U,
      {{.x = 0, .y = 0}, {.x = 0, .y = 0}},
      {{.x = 0, .y = 0}, {.x = 0, .y = 0}});

  EXPECT_EQ(snapshot.raw_elevation_revision(),
            elevation->raw_elevation_revision());
  EXPECT_EQ(snapshot.fine_traversability_revision(), 7U);
  EXPECT_DOUBLE_EQ(snapshot.cost_weights().slope, 1.0);
  EXPECT_DOUBLE_EQ(snapshot.metrics().derivation_ms, 1.5);
  EXPECT_EQ(snapshot.State(GridIndex{.x = 1, .y = 0}), FineCellState::kFree);
  EXPECT_DOUBLE_EQ(snapshot.TraversalCost(GridIndex{.x = 1, .y = 0}), 1.25);
  ASSERT_EQ(snapshot.changed_tiles().size(), 1U);
  ASSERT_EQ(snapshot.changed_halo_tiles().size(), 1U);

  EXPECT_THROW(
      FineTraversabilitySnapshot(
          elevation->geometry(), elevation->raw_elevation_revision(), 0U,
          "wheel-profile", 0.6, 0.25, TraversalCostWeights{}, elevation,
          tiles, {}, {}, FineSnapshotMetrics{}),
      std::invalid_argument);
  EXPECT_THROW(
      FineTraversabilitySnapshot(
          elevation->geometry(), elevation->raw_elevation_revision(), 1U, "",
          0.6, 0.25, TraversalCostWeights{}, elevation, tiles, {}, {},
          FineSnapshotMetrics{}),
      std::invalid_argument);
  EXPECT_THROW(
      FineTraversabilitySnapshot(
          elevation->geometry(), elevation->raw_elevation_revision(), 1U,
          "wheel-profile", 0.6, 0.25,
          TraversalCostWeights{.clearance = -1.0}, elevation, tiles, {}, {},
          FineSnapshotMetrics{}),
      std::invalid_argument);
  EXPECT_THROW(
      MakeFineSnapshot(elevation, elevation->geometry(), tiles, 2U,
                       {{.x = 1, .y = 0}}),
      std::invalid_argument);
  EXPECT_THROW(
      MakeFineSnapshot(elevation, elevation->geometry(), tiles, 2U, {},
                       {{.x = 1, .y = 0}}),
      std::invalid_argument);
  EXPECT_THROW(
      FineTraversabilitySnapshot(
          elevation->geometry(), 0U, 1U, "wheel-profile", 0.6, 0.25,
          TraversalCostWeights{}, elevation, tiles, {}, {},
          FineSnapshotMetrics{}),
      std::invalid_argument);
  EXPECT_THROW(
      FineTraversabilitySnapshot(
          elevation->geometry(), elevation->raw_elevation_revision() + 1U,
          1U, "wheel-profile", 0.6, 0.25, TraversalCostWeights{}, elevation,
          tiles, {}, {}, FineSnapshotMetrics{}),
      std::invalid_argument);
  EXPECT_THROW(
      FineTraversabilitySnapshot(
          elevation->geometry(), elevation->raw_elevation_revision(), 1U,
          "wheel-profile", 0.6, 0.25, TraversalCostWeights{}, nullptr, tiles,
          {}, {}, FineSnapshotMetrics{}),
      std::invalid_argument);
  EXPECT_THROW(
      FineTraversabilitySnapshot(
          elevation->geometry(), 0U, 1U, "wheel-profile", 0.6, 0.25,
          TraversalCostWeights{},
          std::make_shared<const ElevationSnapshot>(), tiles, {}, {},
          FineSnapshotMetrics{}),
      std::invalid_argument);

}

TEST(PlanningSnapshotContractTest,
     GlobalSnapshotKeepsSparseQueriesAndPublishedLineage) {
  const SparseGridGeometry geometry("map", 2.0, Vec3{},
                                    GridIndex{.x = -300, .y = -1},
                                    GridIndex{.x = 301, .y = 2});
  const auto tiles = MakeGuidanceDirectory(geometry, {
      {TileIndex{.x = -1, .y = 0},
       MakeGuidanceTileAt(GridIndex{.x = -1, .y = 0},
                          GuidanceCellState::kCandidate, 0.75)}});
  const GlobalGuidanceSnapshot snapshot(
      geometry, 17U, 13U, 5U, "legged-profile", tiles,
      {{.x = -1, .y = 0}}, {{.x = -1, .y = 0}});

  EXPECT_EQ(snapshot.source_raw_elevation_revision(), 17U);
  EXPECT_EQ(snapshot.source_fine_traversability_revision(), 13U);
  EXPECT_EQ(snapshot.global_guidance_revision(), 5U);
  EXPECT_EQ(snapshot.State(GridIndex{.x = -1, .y = 0}),
            GuidanceCellState::kCandidate);
  EXPECT_DOUBLE_EQ(snapshot.TerrainRisk(GridIndex{.x = -1, .y = 0}), 0.75);
  EXPECT_EQ(snapshot.State(GridIndex{.x = 300, .y = 1}),
            GuidanceCellState::kUnknown);

  EXPECT_THROW(GlobalGuidanceSnapshot(geometry, 0U, 13U, 5U, "profile",
                                      tiles, {}, {}),
               std::invalid_argument);
  EXPECT_THROW(GlobalGuidanceSnapshot(geometry, 17U, 0U, 5U, "profile",
                                      tiles, {}, {}),
               std::invalid_argument);
  EXPECT_THROW(GlobalGuidanceSnapshot(geometry, 17U, 13U, 0U, "profile",
                                      tiles, {}, {}),
               std::invalid_argument);
  EXPECT_THROW(GlobalGuidanceSnapshot(geometry, 17U, 13U, 5U, "", tiles, {},
                                      {}),
               std::invalid_argument);
  EXPECT_THROW(GlobalGuidanceSnapshot(geometry, 17U, 13U, 5U, "profile",
                                      tiles, {}, {{.x = 2, .y = 0}}),
               std::invalid_argument);
}

TEST(PlanningSnapshotContractTest,
     RequestViewSupportsNegativeIndicesAndKeepsPatchAuthority) {
  const auto elevation = MakeNegativeElevation();
  const auto tiles = MakeFineDirectory(elevation->geometry(), {
      {TileIndex{.x = -1, .y = 0}, MakeFineTile(FineCellState::kUnknown)},
      {TileIndex{.x = 0, .y = 0},
       MakeFineTileAt(GridIndex{.x = 0, .y = 0}, FineCellState::kFree)}});
  const auto fine = std::make_shared<const FineTraversabilitySnapshot>(
      MakeFineSnapshot(elevation, elevation->geometry(), tiles));
  const RequestLocalPlanningView view(
      fine, Pose2{.position_m = Vec2{.x = -0.3, .y = 0.1}}, 0.0,
      std::vector<LocalCellOverride>{
          {.index = GridIndex{.x = -2, .y = 0},
           .source = LocalCellSource::kStartAssumedFree,
           .traversal_cost = 0.4}});

  EXPECT_EQ(view.Source(GridIndex{.x = -2, .y = 0}),
            LocalCellSource::kStartAssumedFree);
  EXPECT_EQ(view.Source(GridIndex{.x = 0, .y = 0}),
            LocalCellSource::kEvidenceFree);
  EXPECT_THROW(
      RequestLocalPlanningView(
          fine, Pose2{.position_m = Vec2{.x = -0.3, .y = 0.1}}, 0.0,
          {{.index = GridIndex{.x = 0, .y = 0},
            .source = LocalCellSource::kStartAssumedFree}}),
      std::invalid_argument);
  EXPECT_THROW(
      RequestLocalPlanningView(
          fine, Pose2{.position_m = Vec2{.x = -0.3, .y = 0.1}}, 0.0,
          {{.index = GridIndex{.x = -2, .y = 0},
            .source = LocalCellSource::kEvidenceFree}}),
      std::invalid_argument);
}

TEST(PlanningSnapshotContractTest, SnapshotBundleAllowsMissingGuidance) {
  const auto fine = MakeBasicFineSnapshot();
  const SnapshotBundle bundle{.fine = fine, .guidance = nullptr};
  EXPECT_EQ(bundle.fine->fine_traversability_revision(), 7U);
  EXPECT_FALSE(bundle.guidance);
}

}  // namespace
}  // namespace lunar::incremental_navigation
