#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

#include "support/complex_terrain_benchmark.hpp"

int main(const int argc, const char* const argv[]) {
  std::vector<std::string_view> arguments;
  arguments.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0U);
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }

  try {
    const auto options =
        lunar::incremental_navigation::test_support::
            ParseComplexTerrainBenchmarkArguments(arguments);
    const auto records =
        lunar::incremental_navigation::test_support::
            RunComplexTerrainBenchmarks(options);
    lunar::incremental_navigation::test_support::
        WriteComplexTerrainBenchmarkCsv(records, std::cout);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "complex terrain benchmark: " << error.what() << '\n'
              << "options: --size=32..640 --resolution=<meters> "
                 "--iterations=1..100 --deadline-ms=1..600000 "
                 "--scenario=all|dense-rock|maze|dead-ends|risk-unknown|"
                 "enclosed|legged-step-gap --no-legged --no-coordinator\n";
    return 2;
  }
}
