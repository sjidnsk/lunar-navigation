#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/types/planner_io.hpp"

namespace lunar::pure_planning {

struct HopperOpportunityContextStorage;
struct GroundEndpointReachabilityContextStorage;

struct HopperLandingEvidence final {
  std::uint8_t certified{};
  Vec3 aim_position_on_surface_m;
  std::array<Vec3, 4U> boundary_m;
  double area_m2{};
};

struct HopperLandingEvidenceProjection final {
  std::vector<HopperLandingEvidence> landings;
  std::string algorithm_id;
  std::size_t candidates_evaluated{};
  std::size_t certified_count{};
};

struct HopperLandingEvidenceProjectionResult final {
  std::optional<HopperLandingEvidenceProjection> projection;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return projection.has_value() && reason_code.empty();
  }
};

struct HopperLandingEvidenceGrid final {
  std::size_t width{};
  std::size_t height{};
  std::vector<HopperLandingEvidence> landings;
  std::string algorithm_id;
};

struct HopperSingleHopEnvelopeProjection final {
  std::size_t width{};
  std::size_t height{};
  std::vector<std::uint8_t> eligible;
  std::vector<double> required_delta_v_mps;
  std::vector<double> nominal_flight_time_s;
  std::string algorithm_id;
  std::size_t raw_known_landing_count{};
  std::size_t candidates_evaluated{};
  std::size_t eligible_count{};
  std::uint8_t complete{};
};

struct HopperSingleHopEnvelopeProjectionResult final {
  std::optional<HopperSingleHopEnvelopeProjection> projection;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return projection.has_value() && reason_code.empty();
  }
};

enum class HopperEdgeEvidenceDisposition : std::uint8_t {
  kCertified,
  kStablePhysicalRejection,
  kWaitingEvidence,
};

struct HopperIncrementalEdgeDiagnostic final {
  std::size_t target_index{};
  HopperEdgeEvidenceDisposition disposition{};
  std::vector<std::size_t> dependency_tile_indices;
  Vec3 exact_target_position_m;
  double nominal_flight_time_s{};
  std::string reason_code;
};

struct HopperIncrementalEdgeProjection final {
  std::size_t width{};
  std::size_t height{};
  std::vector<HopperIncrementalEdgeDiagnostic> edges;
  std::string algorithm_id;
};

struct HopperIncrementalEdgeProjectionResult final {
  std::optional<HopperIncrementalEdgeProjection> projection;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return projection.has_value() && reason_code.empty();
  }
};

struct ReachabilityProjection final {
  PlatformType platform_type{};
  std::size_t width{};
  std::size_t height{};
  std::vector<std::uint8_t> reachable;
  std::string algorithm_id;
  double maximum_edge_distance_m{};
  std::size_t candidate_edges_evaluated{};
  std::size_t certified_edges{};
  std::size_t rejected_edges{};
  double maximum_certified_edge_distance_m{};
  std::vector<double> minimum_cost;
  std::vector<std::size_t> parent_index;
  std::size_t start_index{};
  double search_elapsed_s{};
};

struct ReachabilityProjectionResult final {
  std::optional<ReachabilityProjection> projection;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return projection.has_value() && reason_code.empty();
  }
};

struct GroundEndpointReachabilityContext final {
  ReachabilityProjection projection;
  std::shared_ptr<GroundEndpointReachabilityContextStorage> storage;
};

struct GroundEndpointReachabilityContextResult final {
  std::optional<GroundEndpointReachabilityContext> context;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return context.has_value() && reason_code.empty();
  }
};

struct GroundExactEndpointProjection final {
  std::vector<std::uint8_t> reachable;
  std::vector<double> minimum_cost_m;
  std::vector<std::string> reason_codes;
};

struct GroundExactEndpointProjectionResult final {
  std::optional<GroundExactEndpointProjection> projection;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return projection.has_value() && reason_code.empty();
  }
};

struct HopperOpportunityContext final {
  std::size_t width{};
  std::size_t height{};
  std::vector<std::uint8_t> direct;
  std::vector<std::uint8_t> reachable;
  std::vector<std::int32_t> hop_distance_from_current;
  std::string algorithm_id;
  std::size_t candidate_edges_evaluated{};
  std::size_t certified_edges{};
  std::size_t rejected_edges{};
  std::shared_ptr<HopperOpportunityContextStorage> storage;
};

struct HopperOpportunityContextResult final {
  std::optional<HopperOpportunityContext> context;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return context.has_value() && reason_code.empty();
  }
};

struct HopperOpportunityDistanceProjection final {
  std::size_t width{};
  std::size_t height{};
  std::vector<std::uint8_t> direct_progress;
  std::vector<std::uint8_t> reachable_opportunities;
  std::vector<double> direct_progress_total_cost;
  std::vector<std::size_t> represented_opportunity_index;
  std::int32_t current_hop_distance{-1};
  std::uint8_t has_reachable_opportunity{};
  std::string algorithm_id;
};

struct HopperOpportunityDistanceProjectionResult final {
  std::optional<HopperOpportunityDistanceProjection> projection;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return projection.has_value() && reason_code.empty();
  }
};

[[nodiscard]] ReachabilityProjectionResult ProjectReachability(
    const PlannerInput& input,
    double maximum_edge_distance_m);

[[nodiscard]] GroundEndpointReachabilityContextResult
ProjectGroundEndpointReachabilityContext(
    const PlannerInput& input,
    double maximum_edge_distance_m);

[[nodiscard]] GroundExactEndpointProjectionResult
QueryGroundExactEndpoints(
    const GroundEndpointReachabilityContext& context,
    std::span<const Vec3> target_positions_map,
    double tolerance_m);

[[nodiscard]] ReachabilityProjectionResult ProjectReachability(
    const PlannerInput& input,
    double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid& hopper_landing_evidence);

[[nodiscard]] ReachabilityProjectionResult ProjectDirectHopperReachability(
    const PlannerInput& input,
    double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid& hopper_landing_evidence);

[[nodiscard]] HopperOpportunityContextResult
ProjectHopperOpportunityContext(
    const PlannerInput& input,
    double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid& hopper_landing_evidence);

[[nodiscard]] HopperOpportunityDistanceProjectionResult
QueryHopperOpportunityDistance(
    HopperOpportunityContext& context,
    std::span<const std::uint8_t> positive_opportunities,
    bool enumerate_all_reachable_opportunities = true);

[[nodiscard]] HopperLandingEvidenceProjectionResult
ProjectHopperLandingEvidence(
    const PlannerInput& input,
    std::span<const Vec3> target_positions_map);

[[nodiscard]] HopperSingleHopEnvelopeProjectionResult
ProjectHopperSingleHopEnvelope(
    const PlannerInput& input,
    const HopperLandingEvidenceGrid& hopper_landing_evidence);

[[nodiscard]] HopperIncrementalEdgeProjectionResult
ProjectHopperIncrementalEdges(
    const PlannerInput& input,
    const HopperLandingEvidenceGrid& hopper_landing_evidence,
    std::span<const std::uint8_t> task_target_mask);

}  // namespace lunar::pure_planning
