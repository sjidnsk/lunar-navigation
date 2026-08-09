#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "lunar_planner_training_bridge/visibility.hpp"

#ifndef LUNAR_VISIBILITY_BUILD_TYPE
#define LUNAR_VISIBILITY_BUILD_TYPE ""
#endif
#ifndef LUNAR_VISIBILITY_COMPILER_ID
#define LUNAR_VISIBILITY_COMPILER_ID ""
#endif
#ifndef LUNAR_VISIBILITY_COMPILER_VERSION
#define LUNAR_VISIBILITY_COMPILER_VERSION ""
#endif

namespace lunar::planning::training {
namespace {

constexpr std::size_t kCandidateHeight = 601U;
constexpr std::size_t kCandidateWidth = 601U;
constexpr std::size_t kCandidateCount = 64U;
constexpr double kCandidateResolutionM = 0.2;
constexpr double kRangeM = 30.0;
constexpr std::size_t kRevealHeight = 320U;
constexpr std::size_t kRevealWidth = 320U;
constexpr double kRevealResolutionM = 0.2;

struct Timings final {
  double p50_ms{};
  double p95_ms{};
};

[[nodiscard]] std::size_t ParsePositiveCount(const char *const text,
                                             const char *const label) {
  if (text == nullptr || *text == '\0') {
    throw std::invalid_argument(std::string{label} + " is missing");
  }
  if (!std::ranges::all_of(std::string_view{text}, [](const char character) {
        return character >= '0' && character <= '9';
      })) {
    throw std::invalid_argument(std::string{label} + " must be positive");
  }
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || parsed == 0U ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string{label} + " must be positive");
  }
  return static_cast<std::size_t>(parsed);
}

[[nodiscard]] std::size_t PercentileIndex(const std::size_t count,
                                          const double percentile) {
  return std::min(count - 1U, static_cast<std::size_t>(std::ceil(
                                  percentile * static_cast<double>(count))) -
                                  1U);
}

template <typename Operation>
[[nodiscard]] Timings Measure(const std::size_t warmup_count,
                              const std::size_t sample_count,
                              Operation &&operation) {
  for (std::size_t index = 0U; index < warmup_count; ++index) {
    operation();
  }
  std::vector<double> samples;
  samples.reserve(sample_count);
  for (std::size_t index = 0U; index < sample_count; ++index) {
    const auto start = std::chrono::steady_clock::now();
    operation();
    const auto stop = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(stop - start).count());
  }
  std::ranges::sort(samples);
  return Timings{
      .p50_ms = samples[PercentileIndex(sample_count, 0.50)],
      .p95_ms = samples[PercentileIndex(sample_count, 0.95)],
  };
}

[[nodiscard]] std::uint32_t NextRandom(std::uint32_t &state) noexcept {
  state ^= state << 13U;
  state ^= state >> 17U;
  state ^= state << 5U;
  return state;
}

void PrintJson(const std::size_t warmup_count, const std::size_t sample_count,
               const Timings candidate, const Timings reveal,
               const std::uint64_t checksum) {
  std::cout << std::setprecision(17)
            << "{\"schema_version\":\"native-visibility-benchmark/v1\","
            << "\"build_type\":\"" << LUNAR_VISIBILITY_BUILD_TYPE << "\","
            << "\"compiler_id\":\"" << LUNAR_VISIBILITY_COMPILER_ID << "\","
            << "\"compiler_version\":\"" << LUNAR_VISIBILITY_COMPILER_VERSION
            << "\","
            << "\"warmup_count\":" << warmup_count << ','
            << "\"sample_count\":" << sample_count << ','
            << "\"timing_unit\":\"ms\","
            << "\"candidate_fixture\":{\"height\":601,\"width\":601,"
               "\"resolution_m\":0.2,\"range_m\":30.0,"
               "\"candidate_count\":64},"
            << "\"reveal_fixture\":{\"height\":320,\"width\":320,"
               "\"resolution_m\":0.2,\"range_m\":30.0},"
            << "\"latency_ms\":{\"candidate_gains\":{\"p50\":"
            << candidate.p50_ms << ",\"p95\":" << candidate.p95_ms
            << "},\"reveal\":{\"p50\":" << reveal.p50_ms
            << ",\"p95\":" << reveal.p95_ms << "}},"
            << "\"checksum\":" << checksum << "}\n";
}

} // namespace
} // namespace lunar::planning::training

int main(int argc, char **argv) {
  using lunar::planning::training::GridCell;
  using lunar::planning::training::GridShape;
  using lunar::planning::training::kCandidateCount;
  using lunar::planning::training::kCandidateHeight;
  using lunar::planning::training::kCandidateResolutionM;
  using lunar::planning::training::kCandidateWidth;
  using lunar::planning::training::kRangeM;
  using lunar::planning::training::kRevealHeight;
  using lunar::planning::training::kRevealResolutionM;
  using lunar::planning::training::kRevealWidth;
  using lunar::planning::training::Measure;
  using lunar::planning::training::NextRandom;
  using lunar::planning::training::PrintJson;
  using lunar::planning::training::VisibilityKernel;
  try {
    std::size_t warmup_count = 50U;
    std::size_t sample_count = 200U;
    for (int index = 1; index < argc; index += 2) {
      if (index + 1 >= argc) {
        throw std::invalid_argument("benchmark option value is missing");
      }
      const std::string_view option{argv[index]};
      if (option == "--warmup") {
        warmup_count = lunar::planning::training::ParsePositiveCount(
            argv[index + 1], "warmup count");
      } else if (option == "--samples") {
        sample_count = lunar::planning::training::ParsePositiveCount(
            argv[index + 1], "sample count");
      } else {
        throw std::invalid_argument("unknown benchmark option");
      }
    }

    constexpr GridShape candidate_shape{.height = kCandidateHeight,
                                        .width = kCandidateWidth};
    const std::size_t candidate_cells =
        candidate_shape.height * candidate_shape.width;
    std::vector<std::uint8_t> observed(candidate_cells, 0U);
    std::vector<float> obstacles(candidate_cells, 0.0F);
    std::vector<float> roi(candidate_cells, 0.0F);
    std::vector<float> priority(candidate_cells, 0.0F);
    std::uint32_t random_state = 4080U;
    constexpr std::int32_t center = 300;
    constexpr std::int32_t observed_radius = 150;
    for (std::int32_t row = 0;
         row < static_cast<std::int32_t>(candidate_shape.height); ++row) {
      for (std::int32_t column = 0;
           column < static_cast<std::int32_t>(candidate_shape.width);
           ++column) {
        const std::size_t index =
            static_cast<std::size_t>(row) * candidate_shape.width +
            static_cast<std::size_t>(column);
        const std::uint32_t sample = NextRandom(random_state);
        const std::int64_t delta_row = row - center;
        const std::int64_t delta_column = column - center;
        observed[index] = delta_row * delta_row + delta_column * delta_column <=
                                  observed_radius * observed_radius
                              ? 1U
                              : 0U;
        obstacles[index] = 0.0F;
        roi[index] = static_cast<float>((sample >> 8U) % 101U) / 100.0F;
        priority[index] = static_cast<float>((sample >> 16U) % 101U) / 100.0F;
      }
    }
    std::vector<GridCell> candidates;
    candidates.reserve(kCandidateCount);
    constexpr double candidate_radius = 120.0;
    for (std::size_t index = 0U; index < kCandidateCount; ++index) {
      const double angle = 2.0 * std::numbers::pi * static_cast<double>(index) /
                           static_cast<double>(kCandidateCount);
      const GridCell candidate{
          .row = center + static_cast<std::int32_t>(
                              std::llround(candidate_radius * std::sin(angle))),
          .column = center + static_cast<std::int32_t>(std::llround(
                                 candidate_radius * std::cos(angle))),
      };
      candidates.push_back(candidate);
      const std::size_t candidate_index =
          static_cast<std::size_t>(candidate.row) * candidate_shape.width +
          static_cast<std::size_t>(candidate.column);
      observed[candidate_index] = 1U;
      obstacles[candidate_index] = 0.0F;
    }
    const VisibilityKernel candidate_kernel{kCandidateResolutionM, kRangeM};
    std::uint64_t checksum = 0U;
    const auto candidate_timing = Measure(warmup_count, sample_count, [&] {
      const auto gains = candidate_kernel.EstimateCandidateGains(
          candidate_shape, observed, obstacles, roi, priority, candidates);
      for (const auto gain : gains) {
        checksum += static_cast<std::uint64_t>(
            std::llround((gain.roi + gain.priority) * 1000.0F));
      }
    });

    constexpr GridShape reveal_shape{.height = kRevealHeight,
                                     .width = kRevealWidth};
    const std::size_t reveal_cells = reveal_shape.height * reveal_shape.width;
    std::vector<float> truth_obstacles(reveal_cells, 0.0F);
    random_state = 4080U;
    for (std::size_t index = 0U; index < reveal_cells; ++index) {
      truth_obstacles[index] =
          (NextRandom(random_state) % 127U == 0U) ? 1.0F : 0.0F;
    }
    constexpr GridCell reveal_pose{.row = 160, .column = 160};
    truth_obstacles[static_cast<std::size_t>(reveal_pose.row) *
                        reveal_shape.width +
                    static_cast<std::size_t>(reveal_pose.column)] = 0.0F;
    const VisibilityKernel reveal_kernel{kRevealResolutionM, kRangeM};
    const auto reveal_timing = Measure(warmup_count, sample_count, [&] {
      const auto visible = reveal_kernel.RevealFromPose(
          reveal_shape, reveal_pose, truth_obstacles);
      checksum += static_cast<std::uint64_t>(
          std::ranges::count(visible, std::uint8_t{1}));
    });

    PrintJson(warmup_count, sample_count, candidate_timing, reveal_timing,
              checksum);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  return 0;
}
