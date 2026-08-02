#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"
#include "lunar_path_planner/v3/legged/legged_terrain.hpp"
#include "lunar_path_planner/v3/search/platform_adapter.hpp"

namespace lunar::planning::v3 {

struct LeggedLatticeState final {
  std::int32_t ix{};
  std::int32_t iy{};
  std::int32_t iyaw{};
  HeightInterval reachable_z;
};

struct LeggedLatticeEdge final {
  LeggedLatticeState source;
  LeggedLatticeState target;
  PrimitiveId primitive_id;
  BodyMotionKind motion_kind{BodyMotionKind::kForward};
  ContentRef sampled_body_sweep_ref;
  DurationNanoseconds transition_time;
  SecondaryCostVector secondary_costs;
};

struct HeightLabelRecord final {
  std::string stable_label_id;
  StateKey pose_key{};
  DurationNanoseconds arrival_time;
  HeightInterval reachable_z;
};

[[nodiscard]] bool HeightLabelDominates(
    DurationNanoseconds lhs_time,
    const HeightInterval& lhs_interval,
    DurationNanoseconds rhs_time,
    const HeightInterval& rhs_interval) noexcept;

[[nodiscard]] std::vector<HeightLabelRecord>
FilterNonDominatedHeightLabels(
    std::span<const HeightLabelRecord> labels);

[[nodiscard]] std::optional<LeggedLatticeState>
ApplyLeggedPrimitiveKinematics(
    const LeggedLatticeState& source,
    const BodyMotionPrimitive& primitive,
    const GridConfig& grid) noexcept;

class LeggedLatticeAdapter final {
 public:
  LeggedLatticeAdapter(
      const SafeProjection& projection,
      std::span<const TerminalCandidate> terminals,
      GridConfig grid,
      const BodyMotionPrimitiveCatalog& catalog) noexcept;

  [[nodiscard]] StateKey Key(
      const LeggedLatticeState& state) const noexcept;
  [[nodiscard]] std::vector<
      SearchTransition<LeggedLatticeState, LeggedLatticeEdge>>
  Expand(const LeggedLatticeState& state) const;
  [[nodiscard]] bool HardFeasible(
      const SearchTransition<LeggedLatticeState,
                             LeggedLatticeEdge>& transition) const;
  [[nodiscard]] DurationNanoseconds TransitionTime(
      const SearchTransition<LeggedLatticeState,
                             LeggedLatticeEdge>& transition) const noexcept;
  [[nodiscard]] DurationNanoseconds AdmissibleTimeHeuristic(
      const LeggedLatticeState& state,
      const SearchProblem<LeggedLatticeState>& problem) const noexcept;
  [[nodiscard]] SecondaryCostVector SecondaryCosts(
      const SearchTransition<LeggedLatticeState,
                             LeggedLatticeEdge>& transition) const noexcept;
  [[nodiscard]] bool IsTerminal(
      const LeggedLatticeState& state,
      const SearchProblem<LeggedLatticeState>& problem) const noexcept;

 private:
  const SafeProjection* projection_;
  std::span<const TerminalCandidate> terminals_;
  GridConfig grid_;
  const BodyMotionPrimitiveCatalog* catalog_;
};

static_assert(
    PlatformSearchAdapter<LeggedLatticeAdapter, LeggedLatticeState,
                          LeggedLatticeEdge>);

}  // namespace lunar::planning::v3
