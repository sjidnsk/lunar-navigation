#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "lunar_path_planner/v3/hopper/ballistic_kinematics.hpp"
#include "lunar_path_planner/v3/hopper/landing_region.hpp"

namespace lunar::planning::v3 {

struct FlightTubeCertificationInput final {
  NominalBallisticArc arc;
  const ImmutableMapSnapshot* map{};
  TerrainCertifiedLandingRegion source_region;
  TerrainCertifiedLandingRegion target_region;
  AxisAlignedBox3 initial_position_error_m;
  AxisAlignedBox3 initial_velocity_error_mps;
  AxisAlignedBox3 launch_execution_velocity_error_mps;
  AxisAlignedBox3 gravity_error_mps2;
  ConvexPolytope3 arbitrary_attitude_body_envelope;
  ContentRef source_snapshot_ref;
  ContentRef body_rotation_envelope_ref;
  ContentRef error_model_ref;
};

struct FlightTubeCertificationDiagnostics final {
  std::size_t time_slab_count{};
  std::size_t overlapped_cell_count{};
  std::size_t subdivision_count{};
  double minimum_clearance_m{};
  std::string rejection_reason;
};

struct FlightTubeCertificationResult final {
  std::optional<CertifiedFlightTube> certified_tube;
  FlightTubeCertificationDiagnostics diagnostics;
};

class FlightTubeCertifier final {
 public:
  FlightTubeCertifier(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  [[nodiscard]] FlightTubeCertificationResult certify(
      const FlightTubeCertificationInput& input) const;

 private:
  HopperCapabilityView capability_;
  HopperPlannerLimits limits_;
};

}  // namespace lunar::planning::v3
