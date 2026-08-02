#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "lunar_path_planner/v3/hopper/ballistic_kinematics.hpp"

namespace lunar::planning::v3 {

struct LandingPlaneUncertainty final {
  AxisAlignedBox3 origin_error_m;
  RotationVectorBall normal_error;
  SymmetricScalarInterval residual_error_m;
};

struct LandingSetPropagationInput final {
  NominalBallisticArc arc;
  LandingPlane landing_plane;
  AxisAlignedBox3 initial_position_error_m;
  AxisAlignedBox3 initial_velocity_error_mps;
  AxisAlignedBox3 gravity_error_mps2;
  AxisAlignedBox3 launch_execution_velocity_error_mps;
  LandingPlaneUncertainty landing_plane_error;
  CircularYawInterval certified_landing_yaw_interval;
  ContentRef source_error_model_ref;
};

struct LandingSetPropagationDiagnostics final {
  std::size_t impact_root_iteration_count{};
  std::size_t support_direction_count{};
  double worst_case_downward_normal_speed_mps{};
  std::string rejection_reason;
};

struct LandingSetPropagationResult final {
  std::optional<PredictedLandingFootprint> footprint;
  LandingSetPropagationDiagnostics diagnostics;
};

class LandingSetPropagator final {
 public:
  LandingSetPropagator(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  [[nodiscard]] LandingSetPropagationResult propagate(
      const LandingSetPropagationInput& input) const;

 private:
  HopperCapabilityView capability_;
  HopperPlannerLimits limits_;
};

[[nodiscard]] ValidationReport validate_landing_containment(
    const PredictedLandingFootprint& footprint,
    const NextLandingRegion& region,
    const AttitudeBoundary& attitude_boundary);

}  // namespace lunar::planning::v3
