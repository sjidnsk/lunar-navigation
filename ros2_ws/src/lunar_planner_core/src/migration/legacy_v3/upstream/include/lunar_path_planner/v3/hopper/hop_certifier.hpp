#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/hopper/attitude_certifier.hpp"
#include "lunar_path_planner/v3/hopper/ballistic_time_solver.hpp"
#include "lunar_path_planner/v3/hopper/flight_tube_certifier.hpp"
#include "lunar_path_planner/v3/hopper/landing_graph.hpp"
#include "lunar_path_planner/v3/hopper/landing_set_propagator.hpp"

namespace lunar::planning::v3 {

struct HopCertificationContext final {
  ClockStamp reference_time_origin;
  HopperState launch_state;
  const ImmutableMapSnapshot* map{};
  HopperCapabilityView capability;
  HopperPlannerLimits limits;
  std::string source_request_id;
  ContentRef source_error_model_ref;
};

struct HopCertificationDiagnostics final {
  std::size_t ballistic_candidate_count{};
  std::size_t fully_attempted_candidate_count{};
  std::size_t certified_candidate_count{};
  std::vector<std::string> candidate_rejection_reasons;
  std::string selected_candidate_id;
};

class HopCertifier final : public NextHopCertificationOracle {
 public:
  explicit HopCertifier(HopCertificationContext context);

  [[nodiscard]] FirstEdgeCertificationResult certify_first_edge(
      const LandingGraphNode& from,
      const LandingGraphNode& to) override;

  [[nodiscard]] const HopCertificationDiagnostics& diagnostics()
      const noexcept;

 private:
  HopCertificationContext context_;
  HopCertificationDiagnostics diagnostics_;
};

[[nodiscard]] ValidationReport validate_final_hopper_reference(
    const HopperReference& reference,
    const HopCertificationContext& context);

}  // namespace lunar::planning::v3
