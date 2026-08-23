#pragma once

#include <cstddef>
#include <cstdint>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/primitive_reachability.hpp"

namespace lunar::pure_planning::shared {

struct PrimitiveGraphPotentialEdge final {
  std::size_t source_state_index{};
  std::size_t target_state_index{};
  std::uint32_t primitive_index{};
  std::string primitive_id;
  double cost{};
  std::vector<std::uint64_t> swept_tile_ids;
  bool certified{};
  std::string rejection_reason;
};

struct PrimitiveGraphBuildResult final {
  PlatformType platform_type{};
  std::size_t width{};
  std::size_t height{};
  std::vector<PrimitiveReachabilityState> states;
  std::vector<PrimitiveGraphPotentialEdge> potential_edges;
  std::size_t anchor_state_index{};
  std::string algorithm_id;
  std::string state_schema;
  std::vector<std::uint8_t> primitive_set_canonical_bytes;
  std::uint64_t revision{};
  std::size_t invalidated_edge_count{};
  std::size_t revalidated_edge_count{};
  std::string reason_code;
};

[[nodiscard]] PrimitiveReachabilityResult FinalizePrimitiveGraph(
    PrimitiveGraphBuildResult graph, std::stop_token stop_token);

}  // namespace lunar::pure_planning::shared
