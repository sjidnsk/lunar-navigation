#pragma once

#include <memory>
#include <optional>

#include "lunar_incremental_navigation_core/global_guidance_snapshot.hpp"
#include "lunar_incremental_navigation_core/search_control.hpp"
#include "lunar_incremental_navigation_core/types/global_route.hpp"
#include "lunar_incremental_navigation_core/types/planning_cycle.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation {

struct GlobalRoutePlannerConfig final {
  double global_detour_margin_m{};
  double unknown_step_risk{1.0};
  NowFn now{[] { return SteadyClock::now(); }};
};

struct GlobalRouteResult final {
  GuidanceStatus status{GuidanceStatus::kUnavailable};
  std::optional<GlobalRoute> route;
  bool reused_cache{};
  SearchStatistics statistics;
};

class GlobalRoutePlanner final {
 public:
  explicit GlobalRoutePlanner(GlobalRoutePlannerConfig config = {});
  ~GlobalRoutePlanner();

  GlobalRoutePlanner(const GlobalRoutePlanner&) = delete;
  GlobalRoutePlanner& operator=(const GlobalRoutePlanner&) = delete;
  GlobalRoutePlanner(GlobalRoutePlanner&&) noexcept;
  GlobalRoutePlanner& operator=(GlobalRoutePlanner&&) noexcept;

  [[nodiscard]] GlobalRouteResult Plan(
      const GlobalGuidanceSnapshot& snapshot, Point2 start, Point2 goal,
      SearchDeadline deadline, const StopToken& stop);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::incremental_navigation
