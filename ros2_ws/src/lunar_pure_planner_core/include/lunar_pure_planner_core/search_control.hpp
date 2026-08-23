#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <stop_token>

namespace lunar::pure_planning {

enum class EnvironmentMode : std::uint8_t {
  kLunarSurface = 1,
  kLavaTube = 2,
};

using SteadyClock = std::chrono::steady_clock;
using NowFn = std::function<SteadyClock::time_point()>;

struct SearchControl final {
  SteadyClock::time_point deadline{SteadyClock::time_point::max()};
  std::stop_token stop_token;
  NowFn now{[] { return SteadyClock::now(); }};

  [[nodiscard]] bool canceled() const;
  [[nodiscard]] bool expired() const;
};

}  // namespace lunar::pure_planning
