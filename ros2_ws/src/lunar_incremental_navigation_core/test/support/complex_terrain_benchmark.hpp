#pragma once

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

#include "lunar_incremental_navigation_core/search_control.hpp"
#include "support/complex_terrain_scenarios.hpp"

namespace lunar::incremental_navigation::test_support {

struct ComplexTerrainBenchmarkOptions final {
  std::vector<ComplexTerrainSpec> scenarios;
  std::size_t iterations{1U};
  std::chrono::milliseconds stage_deadline{std::chrono::seconds(3)};
  bool include_legged{true};
  bool include_coordinator{true};
};

struct ComplexTerrainBenchmarkRecord final {
  std::string scenario;
  std::string stage;
  std::string platform;
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
  std::size_t iteration{};
  std::string status;
  bool cache_reused{};
  double elapsed_ms{};
  double postprocess_ms{};
  SearchStatistics statistics;
  std::size_t path_points{};
  std::size_t raw_path_points{};
  std::size_t updated_cells{};
  std::size_t elevation_cells_examined{};
  std::size_t allocated_cells{};
  std::size_t allocated_bytes{};
  std::size_t candidate_tile_lookups{};
  std::size_t candidate_cells_examined{};
  std::size_t guidance_cells_examined{};
};

[[nodiscard]] std::vector<ComplexTerrainBenchmarkRecord>
RunComplexTerrainBenchmarks(const ComplexTerrainBenchmarkOptions& options);

[[nodiscard]] ComplexTerrainBenchmarkOptions
ParseComplexTerrainBenchmarkArguments(
    const std::vector<std::string_view>& arguments);

void WriteComplexTerrainBenchmarkCsv(
    const std::vector<ComplexTerrainBenchmarkRecord>& records,
    std::ostream& output);

}  // namespace lunar::incremental_navigation::test_support
