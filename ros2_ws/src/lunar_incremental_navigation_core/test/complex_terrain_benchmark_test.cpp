#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "support/complex_terrain_benchmark.hpp"

namespace lunar::incremental_navigation::test_support {
namespace {

[[nodiscard]] const ComplexTerrainBenchmarkRecord* FindStage(
    const std::vector<ComplexTerrainBenchmarkRecord>& records,
    const std::string& stage) {
  const auto found = std::find_if(
      records.begin(), records.end(), [&](const auto& record) {
        return record.stage == stage;
      });
  return found == records.end() ? nullptr : &*found;
}

TEST(ComplexTerrainBenchmark,
     OneReachableScenarioReportsEveryCoreStageWithoutTimingThresholds) {
  const ComplexTerrainBenchmarkOptions options{
      .scenarios = {{.pattern = ComplexTerrainPattern::kAlternatingWallMaze,
                     .width = 48U,
                     .height = 48U,
                     .resolution_m = 0.2,
                     .seed = 29U}},
      .iterations = 1U,
      .stage_deadline = std::chrono::seconds(5),
      .include_legged = true,
      .include_coordinator = true,
  };

  const auto records = RunComplexTerrainBenchmarks(options);

  for (const std::string stage : {
           "elevation_apply",
           "fine_derive_wheel",
           "elevation_apply_incremental_single_cell",
           "fine_derive_incremental_single_cell",
           "elevation_apply_incremental_patch",
           "fine_derive_incremental_patch",
           "guidance_derive_wheel",
           "global_plan",
           "global_cache_exact",
           "global_replan_off_route_update",
           "global_replan_on_route_update",
           "global_replan_moved_start",
           "target_select_guided",
           "wheel_plan",
           "legged_plan",
           "coordinator_wheel_cycle",
           "coordinator_legged_cycle",
           "path_revalidate_off_route_update",
           "path_revalidate_on_route_update",
       }) {
    const auto* record = FindStage(records, stage);
    ASSERT_NE(record, nullptr) << stage;
    EXPECT_GE(record->elapsed_ms, 0.0) << stage;
    EXPECT_EQ(record->scenario, "alternating_wall_maze") << stage;
  }
  EXPECT_EQ(FindStage(records, "global_plan")->status, "AVAILABLE");
  EXPECT_TRUE(FindStage(records, "global_cache_exact")->cache_reused);
  EXPECT_TRUE(
      FindStage(records, "global_replan_moved_start")->cache_reused);
  EXPECT_EQ(FindStage(records, "global_replan_moved_start")
                ->statistics.expanded_states,
            0U);
  EXPECT_EQ(FindStage(records, "wheel_plan")->status, "PLAN_FOUND");
  EXPECT_EQ(FindStage(records, "legged_plan")->status, "PLAN_FOUND");
  EXPECT_EQ(FindStage(records, "elevation_apply_incremental_single_cell")
                ->status,
            "APPLIED");
  EXPECT_EQ(FindStage(records, "elevation_apply_incremental_patch")->status,
            "APPLIED");
  EXPECT_GE(FindStage(records, "wheel_plan")->raw_path_points,
            FindStage(records, "wheel_plan")->path_points);
  EXPECT_GT(FindStage(records, "target_select_guided")
                ->candidate_cells_examined,
            0U);
  EXPECT_GT(FindStage(records, "target_select_guided")
                ->guidance_cells_examined,
            0U);
  EXPECT_EQ(FindStage(records, "coordinator_wheel_cycle")->status,
            "PLAN_FOUND");
  EXPECT_EQ(FindStage(records, "coordinator_legged_cycle")->status,
            "PLAN_FOUND");
  EXPECT_EQ(FindStage(records, "path_revalidate_off_route_update")->status,
            "PATH_UNCHANGED");
  EXPECT_EQ(FindStage(records, "path_revalidate_on_route_update")->status,
            "PATH_INVALIDATED");
  EXPECT_GT(FindStage(records, "fine_derive_wheel")->updated_cells, 0U);
  EXPECT_GT(FindStage(records, "fine_derive_wheel")
                ->elevation_cells_examined,
            0U);
  EXPECT_GT(FindStage(records, "fine_derive_incremental_single_cell")
                ->updated_cells,
            0U);
  EXPECT_LT(FindStage(records, "fine_derive_incremental_single_cell")
                ->updated_cells,
            FindStage(records, "fine_derive_wheel")->updated_cells);
  EXPECT_GT(FindStage(records, "elevation_apply_incremental_patch")
                ->updated_cells,
            FindStage(records, "elevation_apply_incremental_single_cell")
                ->updated_cells);
  EXPECT_GT(FindStage(records, "fine_derive_incremental_patch")
                ->updated_cells,
            FindStage(records, "fine_derive_incremental_single_cell")
                ->updated_cells);
  EXPECT_LT(FindStage(records, "fine_derive_incremental_patch")
                ->updated_cells,
            FindStage(records, "fine_derive_wheel")->updated_cells);
}

TEST(ComplexTerrainBenchmark, NoPathIsARecordedOutcomeNotAHarnessFailure) {
  const ComplexTerrainBenchmarkOptions options{
      .scenarios = {{.pattern = ComplexTerrainPattern::kEnclosedGoal,
                     .width = 48U,
                     .height = 48U,
                     .resolution_m = 0.2,
                     .seed = 31U}},
      .iterations = 1U,
      .stage_deadline = std::chrono::seconds(5),
      .include_legged = false,
      .include_coordinator = false,
  };

  const auto records = RunComplexTerrainBenchmarks(options);

  ASSERT_NE(FindStage(records, "global_plan"), nullptr);
  ASSERT_NE(FindStage(records, "wheel_plan"), nullptr);
  EXPECT_EQ(FindStage(records, "global_plan")->status, "NO_ROUTE");
  EXPECT_EQ(FindStage(records, "wheel_plan")->status, "NO_PATH");
}

TEST(ComplexTerrainBenchmark, CsvContainsStableColumnsAndOneRowPerRecord) {
  const ComplexTerrainBenchmarkOptions options{
      .scenarios = {{.pattern = ComplexTerrainPattern::kDenseRockField,
                     .width = 48U,
                     .height = 48U,
                     .resolution_m = 0.2,
                     .seed = 37U}},
      .iterations = 1U,
      .stage_deadline = std::chrono::seconds(5),
      .include_legged = false,
      .include_coordinator = false,
  };
  const auto records = RunComplexTerrainBenchmarks(options);
  std::ostringstream output;

  WriteComplexTerrainBenchmarkCsv(records, output);

  const std::string csv = output.str();
  EXPECT_EQ(csv.substr(0U, csv.find('\n')),
            "scenario,stage,platform,width,height,resolution_m,iteration,"
            "status,cache_reused,elapsed_ms,postprocess_ms,expanded_states,"
            "generated_states,evaluated_transitions,open_peak,path_points,"
            "raw_path_points,updated_cells,elevation_cells_examined,"
            "allocated_cells,allocated_bytes,candidate_tile_lookups,"
            "candidate_cells_examined,guidance_cells_examined");
  EXPECT_EQ(static_cast<std::size_t>(std::count(csv.begin(), csv.end(), '\n')),
            records.size() + 1U);
}

TEST(ComplexTerrainBenchmark,
     CommandLineOptionsSelectAReproducibleScenarioMatrix) {
  const std::vector<std::string_view> arguments{
      "--size=640",       "--resolution=0.1", "--iterations=2",
      "--deadline-ms=4000", "--scenario=maze", "--no-legged",
      "--no-coordinator",
  };

  const ComplexTerrainBenchmarkOptions options =
      ParseComplexTerrainBenchmarkArguments(arguments);

  ASSERT_EQ(options.scenarios.size(), 1U);
  EXPECT_EQ(options.scenarios.front().pattern,
            ComplexTerrainPattern::kAlternatingWallMaze);
  EXPECT_EQ(options.scenarios.front().width, 640U);
  EXPECT_EQ(options.scenarios.front().height, 640U);
  EXPECT_DOUBLE_EQ(options.scenarios.front().resolution_m, 0.1);
  EXPECT_EQ(options.iterations, 2U);
  EXPECT_EQ(options.stage_deadline, std::chrono::milliseconds(4000));
  EXPECT_FALSE(options.include_legged);
  EXPECT_FALSE(options.include_coordinator);
}

TEST(ComplexTerrainBenchmark, CommandLineRejectsUnknownOrUnsafeValues) {
  EXPECT_THROW(ParseComplexTerrainBenchmarkArguments({"--size=31"}),
               std::invalid_argument);
  EXPECT_THROW(ParseComplexTerrainBenchmarkArguments({"--iterations=0"}),
               std::invalid_argument);
  EXPECT_THROW(ParseComplexTerrainBenchmarkArguments({"--scenario=typo"}),
               std::invalid_argument);
  EXPECT_THROW(ParseComplexTerrainBenchmarkArguments({"--unknown-option"}),
               std::invalid_argument);
}

}  // namespace
}  // namespace lunar::incremental_navigation::test_support
