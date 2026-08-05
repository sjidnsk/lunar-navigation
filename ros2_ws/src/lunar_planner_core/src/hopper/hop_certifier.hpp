#pragma once

#include "hierarchical/local_planning_problem.hpp"
#include "hopper/hopper_types.hpp"

namespace lunar::planning::hopper {

[[nodiscard]] HopCertificationResult CertifyFirstHop(
    const hierarchical::LocalPlanningProblem& problem,
    const CertifiedLandingRegion& source_region,
    const CertifiedLandingRegion& target_region);

}  // namespace lunar::planning::hopper
