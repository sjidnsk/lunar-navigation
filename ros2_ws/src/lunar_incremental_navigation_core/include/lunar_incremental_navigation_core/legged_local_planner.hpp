#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_core/local_target_selector.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation {

struct LeggedLocalPlannerConfig final {
  double terminal_yaw_tolerance_rad{0.17453292519943295};
};

class LeggedLocalPlanner final {
 public:
  explicit LeggedLocalPlanner(
      LeggedCapability capability,
      LeggedLocalPlannerConfig config = LeggedLocalPlannerConfig{});

  [[nodiscard]] LocalPlanResult Plan(
      const RequestLocalPlanningView& view, const Pose2& start,
      const LocalTarget& target, SearchDeadline deadline,
      const StopToken& stop);

 private:
  LeggedCapability capability_;
  LeggedLocalPlannerConfig config_;
  double maximum_translation_m_{};
  std::vector<double> spin_deltas_rad_;
  // The spatial state is a cell plus start-prefix and one-time continuous
  // start-anchor markers. It intentionally does not contain a discretized yaw
  // dimension.
  std::vector<double> g_cost_;
  std::vector<std::uint32_t> parent_;
  std::vector<std::uint8_t> state_;
  std::vector<std::uint32_t> generation_;
  std::uint32_t current_generation_{};
  std::size_t buffer_capacity_states_{};
};

}  // namespace lunar::incremental_navigation
