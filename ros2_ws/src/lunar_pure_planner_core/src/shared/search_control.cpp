#include "lunar_pure_planner_core/search_control.hpp"

namespace lunar::pure_planning {
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

}  // namespace lunar::pure_planning
