#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning {

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

[[nodiscard]] ReachabilityProjectionResult ProjectReachability(
    const PlannerInput& input,
    double maximum_edge_distance_m);

}  // namespace lunar::planning
