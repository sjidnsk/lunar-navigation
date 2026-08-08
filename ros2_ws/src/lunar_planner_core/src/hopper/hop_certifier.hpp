#pragma once

#include <stop_token>

#include "hopper/ballistic_envelope.hpp"
#include "hopper/hopper_types.hpp"
#include "lunar_planner_core/types/planner_config.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::planning::hopper {

struct SingleHopCertificationProblem final {
  Vec3 launch_position_m;
  Vec3 landing_position_m;
  Vec3 gravity_mps2{0.0, 0.0, -1.62};
  const shared::MapSnapshot* flight_map{};
  const HopperCapability* capability{};
  const MapSafetyConfig* map_safety{};
  std::stop_token stop_token;
};

struct CertifiedSingleHop final {
  BallisticArc arc;
  SingleHopEnvelopeEvidence envelope;
  FlightTubeCertificationResult flight_tube;
};

struct SingleHopCertificationResult final {
  HopCertificationStatus status{HopCertificationStatus::kInvalid};
  std::optional<CertifiedSingleHop> certification;
  std::size_t examined_intervals{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == HopCertificationStatus::kCertified &&
        certification.has_value() && reason_code.empty();
  }
};

[[nodiscard]] SingleHopCertificationResult CertifySingleHop(
    const SingleHopCertificationProblem& problem);

}  // namespace lunar::planning::hopper
