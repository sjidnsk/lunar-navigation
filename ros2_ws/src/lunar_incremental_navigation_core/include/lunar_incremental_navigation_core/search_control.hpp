#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stop_token>

namespace lunar::incremental_navigation {

enum class EnvironmentMode : std::uint8_t {
  kLunarSurface = 1,
  kLavaTube = 2,
};

using SteadyClock = std::chrono::steady_clock;
using NowFn = std::function<SteadyClock::time_point()>;
using SearchDeadline = SteadyClock::time_point;
using StopToken = std::stop_token;

struct SearchStatistics final {
  std::uint64_t expanded_states{};
  std::uint64_t generated_states{};
  std::uint64_t evaluated_transitions{};
  std::size_t open_peak{};
};

struct SearchControl final {
  SteadyClock::time_point deadline{SteadyClock::time_point::max()};
  std::stop_token stop_token;
  NowFn now{[] { return SteadyClock::now(); }};

  [[nodiscard]] bool canceled() const;
  [[nodiscard]] bool expired() const;
};

}  // namespace lunar::incremental_navigation
