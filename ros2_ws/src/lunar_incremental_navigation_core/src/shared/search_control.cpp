#include "lunar_incremental_navigation_core/search_control.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] SteadyClock::time_point ReadNow(const NowFn& now) {
  return now ? now() : SteadyClock::now();
}

}  // namespace

bool SearchControl::canceled() const {
  return stop_token.stop_requested();
}

bool SearchControl::expired() const {
  return ReadNow(now) >= deadline;
}

}  // namespace lunar::incremental_navigation
