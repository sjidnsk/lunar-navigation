#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_core/local_planning_window.hpp"
#include "lunar_incremental_navigation_core/local_target_selector.hpp"
#include "lunar_incremental_navigation_core/search_control.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation {

struct WheelLocalPlannerConfig final {
  std::size_t maximum_width_cells{kMaximumLocalPlanningWindowAxisCells};
  std::size_t maximum_height_cells{kMaximumLocalPlanningWindowAxisCells};
  double terminal_yaw_tolerance_rad{0.2617993877991494};
  NowFn now{[] { return SteadyClock::now(); }};
};

class WheelLocalPlanner final {
 public:
  WheelLocalPlanner();
  explicit WheelLocalPlanner(
      WheeledCapability capability,
      WheelLocalPlannerConfig config = WheelLocalPlannerConfig{});

  [[nodiscard]] LocalPlanResult Plan(
      const RequestLocalPlanningView& view, const Pose2& start,
      const LocalTarget& target, SearchDeadline deadline,
      const StopToken& stop);

  [[nodiscard]] std::size_t buffer_capacity_cells() const noexcept;
  [[nodiscard]] std::size_t buffer_allocation_count() const noexcept;

 private:
  WheeledCapability capability_;
  WheelLocalPlannerConfig config_;
  std::vector<long double> g_cost_;
  std::vector<std::size_t> parent_;
  std::vector<std::uint8_t> state_;
  std::vector<StartPhase> phase_;
  std::vector<std::uint32_t> generation_;
  std::uint32_t current_generation_{};
  std::size_t buffer_capacity_cells_{};
  std::size_t buffer_allocation_count_{};
};

[[nodiscard]] std::vector<PathPoint> SimplifyPhaseAwarePath(
    const RequestLocalPlanningView& view,
    std::span<const PathPoint> raw_path);
[[nodiscard]] std::vector<PathPoint> SimplifyPhaseAwarePath(
    const RequestLocalPlanningView& view,
    std::span<const PathPoint> raw_path, const SearchControl& control);

}  // namespace lunar::incremental_navigation
