#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lunar_incremental_navigation_core/global_guidance_snapshot.hpp"
#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_core/local_target_selector.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#include "lunar_incremental_navigation_core/types/grid_geometry.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation::test_support {

enum class ComplexTerrainPattern : std::uint8_t {
  kDenseRockField,
  kAlternatingWallMaze,
  kNarrowPassagesAndDeadEnds,
  kRiskAndUnknownBands,
  kEnclosedGoal,
  kLeggedStepGapField,
};

struct ComplexTerrainSpec final {
  ComplexTerrainPattern pattern{ComplexTerrainPattern::kAlternatingWallMaze};
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
  std::uint32_t seed{};
};

struct ComplexTerrainScenario final {
  std::string name;
  GridGeometry geometry;
  std::vector<float> elevation_m;
  std::vector<FineCellState> fine_states;
  std::vector<double> traversal_costs;
  std::vector<GuidanceCellState> guidance_states;
  std::vector<double> guidance_risks;
  GridIndex start_cell;
  GridIndex goal_cell;
  bool expected_reachable{};

  [[nodiscard]] FineCellState State(GridIndex index) const;
};

[[nodiscard]] ComplexTerrainScenario MakeComplexTerrainScenario(
    const ComplexTerrainSpec& spec);

[[nodiscard]] std::shared_ptr<const ElevationSnapshot> MakeElevationSnapshot(
    const ComplexTerrainScenario& scenario);

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot>
MakeFineSnapshot(const ComplexTerrainScenario& scenario,
                 std::string profile_hash,
                 std::uint64_t fine_revision = 1U,
                 std::vector<TileIndex> changed_tiles = {});

[[nodiscard]] std::shared_ptr<const GlobalGuidanceSnapshot>
MakeGuidanceSnapshot(const ComplexTerrainScenario& scenario,
                     std::string profile_hash,
                     std::uint64_t guidance_revision = 1U,
                     std::vector<TileIndex> changed_tiles = {});

[[nodiscard]] Pose2 StartPose(const ComplexTerrainScenario& scenario);
[[nodiscard]] LocalTarget LocalTargetForGoal(
    const ComplexTerrainScenario& scenario);
[[nodiscard]] WheeledCapability BenchmarkWheelCapability();
[[nodiscard]] LeggedCapability BenchmarkLeggedCapability();
[[nodiscard]] TraversabilityProfile BenchmarkTraversabilityProfile();

}  // namespace lunar::incremental_navigation::test_support
