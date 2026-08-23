#include "shared/active_planner_cache.hpp"

namespace lunar::pure_planning::shared::detail {

std::string ActivePlannerCacheStopReason(
    const SearchControl& control) noexcept {
  try {
    if (control.canceled()) {
      return "REQUEST_CANCELED";
    }
    if (control.expired()) {
      return "TIMEOUT";
    }
  } catch (...) {
    return "PLANNER_ERROR";
  }
  return {};
}

}  // namespace lunar::pure_planning::shared::detail
