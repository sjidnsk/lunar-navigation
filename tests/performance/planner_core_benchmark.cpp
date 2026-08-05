#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::benchmark {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kWarmupCount = 10U;
constexpr std::size_t kMeasuredCount = 100U;
constexpr std::uint64_t kRandomSeed = 549'924'309'330ULL;

struct Arguments final {
  std::string output_path;
  std::string device_fingerprint_sha256;
  std::string git_commit;
};

[[nodiscard]] bool IsHexDigest(
    const std::string_view value, const std::size_t length) {
  return value.size() == length &&
         std::ranges::all_of(value, [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f') ||
                  (character >= 'A' && character <= 'F');
         });
}

[[nodiscard]] Arguments ParseArguments(const int argc, char** argv) {
  if (argc != 7) {
    throw std::invalid_argument{
        "usage: planner_core_benchmark --output PATH "
        "--device-fingerprint-sha256 SHA256 --git-commit COMMIT"};
  }
  Arguments arguments;
  for (int index = 1; index < argc; index += 2) {
    const std::string_view option{argv[index]};
    const std::string value{argv[index + 1]};
    if (option == "--output") {
      arguments.output_path = value;
    } else if (option == "--device-fingerprint-sha256") {
      arguments.device_fingerprint_sha256 = value;
    } else if (option == "--git-commit") {
      arguments.git_commit = value;
    } else {
      throw std::invalid_argument{"unknown option: " + std::string{option}};
    }
  }
  if (arguments.output_path.empty()) {
    throw std::invalid_argument{"benchmark output path is empty"};
  }
  if (!IsHexDigest(arguments.device_fingerprint_sha256, 64U)) {
    throw std::invalid_argument{
        "device fingerprint SHA-256 must contain 64 hexadecimal characters"};
  }
  if (!IsHexDigest(arguments.git_commit, 40U)) {
    throw std::invalid_argument{
        "Git commit must contain 40 hexadecimal characters"};
  }
  return arguments;
}

void RequireValidResult(const PlannerOutput& output) {
  if (output.outcome != PlanningOutcome::kNewReferenceAvailable ||
      !output.reference.has_value() ||
      output.reference->platform_type != PlatformType::kWheeled) {
    throw std::runtime_error{
        "fixed benchmark fixture did not produce a wheeled reference: " +
        output.reason_code};
  }
}

[[nodiscard]] double Median(const std::vector<double>& sorted) {
  const std::size_t middle = sorted.size() / 2U;
  if (sorted.size() % 2U == 1U) {
    return sorted[middle];
  }
  return (sorted[middle - 1U] + sorted[middle]) / 2.0;
}

[[nodiscard]] double Percentile95(const std::vector<double>& sorted) {
  const auto rank = static_cast<std::size_t>(
      std::ceil(0.95 * static_cast<double>(sorted.size())));
  return sorted[std::max<std::size_t>(1U, rank) - 1U];
}

[[nodiscard]] Json Run(const Arguments& arguments) {
  Planner planner;
  PlannerInput input = test::MakeValidWheelInput();
  std::mt19937_64 random{kRandomSeed};

  for (std::size_t index = 0U; index < kWarmupCount; ++index) {
    input.request_id = "benchmark-warmup-" + std::to_string(random());
    RequireValidResult(planner.Plan(input));
  }

  std::vector<double> elapsed_ms;
  elapsed_ms.reserve(kMeasuredCount);
  PlannerOutput last_output;
  for (std::size_t index = 0U; index < kMeasuredCount; ++index) {
    input.request_id = "benchmark-measured-" + std::to_string(random());
    const auto begin = std::chrono::steady_clock::now();
    last_output = planner.Plan(input);
    const auto end = std::chrono::steady_clock::now();
    RequireValidResult(last_output);
    elapsed_ms.push_back(
        std::chrono::duration<double, std::milli>{end - begin}.count());
  }
  std::ranges::sort(elapsed_ms);

  const auto& state = std::get<WheeledState>(input.current_state);
  const auto& goal = std::get<PointGoal>(input.goal_map.target);
  return Json{
      {"schema_version", "lunar-planner-core-benchmark/v1"},
      {"fixture_id", "wheeled-flat-map-v1"},
      {"platform", "WHEELED"},
      {"map",
       {
           {"frame_id", input.world.global_map.frame_id},
           {"width", input.world.global_map.width},
           {"height", input.world.global_map.height},
           {"resolution_m", input.world.global_map.resolution_m},
       }},
      {"start_xyz_m",
       {state.pose.position_m.x,
        state.pose.position_m.y,
        state.pose.position_m.z}},
      {"goal_xyz_m",
       {goal.position_m.x, goal.position_m.y, goal.position_m.z}},
      {"random_seed", kRandomSeed},
      {"warmup_count", kWarmupCount},
      {"count", kMeasuredCount},
      {"median", Median(elapsed_ms)},
      {"p95", Percentile95(elapsed_ms)},
      {"max", elapsed_ms.back()},
      {"timing_unit", "ms"},
      {"device_fingerprint_sha256", arguments.device_fingerprint_sha256},
      {"git_commit", arguments.git_commit},
      {"planning_outcome", "NEW_REFERENCE_AVAILABLE"},
      {"expanded_states", last_output.diagnostics.expanded_states},
      {"release_gate", false},
  };
}

void Write(const std::string& output_path, const Json& document) {
  std::ofstream stream{output_path, std::ios::binary | std::ios::trunc};
  if (!stream) {
    throw std::runtime_error{"cannot open benchmark output: " + output_path};
  }
  stream << document.dump(2) << '\n';
  stream.flush();
  if (!stream) {
    throw std::runtime_error{"cannot write benchmark output: " + output_path};
  }
}

}  // namespace
}  // namespace lunar::planning::benchmark

int main(int argc, char** argv) {
  try {
    const auto arguments = lunar::planning::benchmark::ParseArguments(argc, argv);
    lunar::planning::benchmark::Write(
        arguments.output_path,
        lunar::planning::benchmark::Run(arguments));
  } catch (const std::exception& error) {
    std::cerr << "planner core benchmark error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
