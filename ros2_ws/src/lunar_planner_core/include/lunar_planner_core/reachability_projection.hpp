#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning {

struct HopperOpportunityContextStorage;

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
};

struct ReachabilityProjectionResult final {
  std::optional<ReachabilityProjection> projection;
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

}  // namespace lunar::planning
