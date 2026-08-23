#pragma once

#include <cstddef>
#include <optional>
#include <stop_token>

#include "lunar_pure_planner_core/types/planner_io.hpp"
#include "shared/primitive_reachability_graph.hpp"
#include "shared/safe_projection.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::pure_planning::wheel {

[[nodiscard]] bool ValidateWheelPrimitiveCapability(
    const WheeledCapability& capability,
    const PlannerConfig& config) noexcept;

[[nodiscard]] WheelLatticeState WheelPrimitivePoseKey(
    const WheelPose& pose, WheelMotionMode mode,
    const shared::MapSnapshot& map,
    std::size_t yaw_bin_count) noexcept;

[[nodiscard]] std::optional<WheelTransition> ApplyWheelPrimitiveKinematics(
    const WheelLatticeState& source_state, const WheelPose& source,
    const WheelMotionPrimitive& primitive, std::size_t primitive_index,
    const shared::MapSnapshot& map, std::size_t yaw_bin_count);

[[nodiscard]] double WheelPrimitiveEdgeCost(
    const WheelTransition& transition,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability) noexcept;

[[nodiscard]] double WheelPrimitiveCostLowerBound(
    const WheelTransition& transition,
    const WheeledCapability& capability) noexcept;

[[nodiscard]] bool ExistingTargetDominates(
    double source_cost, double edge_cost_lower_bound,
    double existing_target_cost) noexcept;

[[nodiscard]] shared::PrimitiveGraphBuildResult BuildWheelPrimitiveGraph(
    const WheeledState& current_state,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);

}  // namespace lunar::pure_planning::wheel
