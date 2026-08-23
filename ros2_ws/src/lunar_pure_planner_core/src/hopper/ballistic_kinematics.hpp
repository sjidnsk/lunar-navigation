#pragma once

#include "hopper/hopper_types.hpp"

namespace lunar::pure_planning::hopper {

[[nodiscard]] BallisticSolveResult SolveBallisticArc(
    Vec3 launch_position_m, Vec3 landing_position_m, Vec3 gravity_mps2,
    double flight_time_s) noexcept;

[[nodiscard]] BallisticState EvaluateBallisticState(
    const BallisticArc& arc, double time_s) noexcept;

[[nodiscard]] double BallisticApexTime(
    const BallisticArc& arc) noexcept;

}  // namespace lunar::pure_planning::hopper
