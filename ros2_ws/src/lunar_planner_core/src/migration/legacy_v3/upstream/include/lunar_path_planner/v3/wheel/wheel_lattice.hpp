#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"
#include "lunar_path_planner/v3/map/safe_projection.hpp"
#include "lunar_path_planner/v3/search/platform_adapter.hpp"
#include "lunar_path_planner/v3/wheel/wheel_primitive.hpp"

namespace lunar::planning::v3 {

struct WheelLatticeState final {
  std::int32_t ix{};
  std::int32_t iy{};
  std::int32_t iyaw{};
  WheelMotionMode motion_mode{WheelMotionMode::kStart};

  auto operator<=>(const WheelLatticeState&) const = default;
};

struct WheelLatticeEdge final {
  WheelLatticeState source;
  WheelLatticeState target;
  PrimitiveId primitive_id;
  PrimitiveId capability_primitive_id;
  WheelPrimitiveKind primitive_kind{WheelPrimitiveKind::kDriveLine};
  PoseXyzYaw source_pose;
  PoseXyzYaw target_pose;
  DurationNanoseconds transition_time;
  SecondaryCostVector secondary_costs;
  ContentRef validation_ref;
};

[[nodiscard]] StateKey EncodeWheelStateKey(
    std::int32_t ix,
    std::int32_t iy,
    std::int32_t iyaw,
    WheelMotionMode mode) noexcept;

class WheelLatticeAdapter final {
 public:
  WheelLatticeAdapter(
      const WheelPrimitiveCatalog& catalog,
      const SafeProjection& projection,
      WheelCapabilityView capability,
      std::span<const TerminalCandidate> terminals,
      GridConfig grid_config);

  [[nodiscard]] StateKey Key(
      const WheelLatticeState& state) const noexcept;
  [[nodiscard]] std::vector<
      SearchTransition<WheelLatticeState, WheelLatticeEdge>>
  Expand(const WheelLatticeState& state) const;
  [[nodiscard]] bool HardFeasible(
      const SearchTransition<WheelLatticeState, WheelLatticeEdge>&
          transition) const;
  [[nodiscard]] DurationNanoseconds TransitionTime(
      const SearchTransition<WheelLatticeState, WheelLatticeEdge>&
          transition) const noexcept;
  [[nodiscard]] DurationNanoseconds AdmissibleTimeHeuristic(
      const WheelLatticeState& state,
      const SearchProblem<WheelLatticeState>& problem) const noexcept;
  [[nodiscard]] SecondaryCostVector SecondaryCosts(
      const SearchTransition<WheelLatticeState, WheelLatticeEdge>&
          transition) const noexcept;
  [[nodiscard]] bool IsTerminal(
      const WheelLatticeState& state,
      const SearchProblem<WheelLatticeState>& problem) const noexcept;

  [[nodiscard]] PoseXyzYaw Pose(
      const WheelLatticeState& state) const noexcept;
  [[nodiscard]] WheelLatticeState Quantize(
      const PoseXyzYaw& pose,
      WheelMotionMode mode) const noexcept;

 private:
  const WheelPrimitiveCatalog* catalog_{};
  const SafeProjection* projection_{};
  WheelCapabilityView capability_;
  std::vector<TerminalCandidate> terminals_;
  GridConfig grid_config_;
};

static_assert(
    PlatformSearchAdapter<WheelLatticeAdapter,
                          WheelLatticeState,
                          WheelLatticeEdge>);

}  // namespace lunar::planning::v3
