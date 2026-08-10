#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning {

struct PrimitiveReachabilityState final {
  std::uint64_t state_id{};
  Vec3 position_m;
  double yaw_rad{};
  std::int32_t cell_x{};
  std::int32_t cell_y{};
  std::int32_t yaw_bin{};
  std::int32_t motion_mode{};
  Interval body_z_m;
  double path_cost{};
  std::uint8_t forward_reachable{};
  std::uint8_t returnable{};
  std::uint8_t observation_state{};
  std::uint8_t direct_successor{};

  auto operator<=>(const PrimitiveReachabilityState&) const = default;
};

struct PrimitiveReachabilityEdge final {
  std::uint64_t source_state_id{};
  std::uint64_t target_state_id{};
  std::uint32_t primitive_index{};
  std::string primitive_id;
  double cost{};

  auto operator<=>(const PrimitiveReachabilityEdge&) const = default;
};

struct PrimitiveReachabilitySnapshot final {
  PlatformType platform_type{};
  std::size_t width{};
  std::size_t height{};
  std::vector<PrimitiveReachabilityState> states;
  std::vector<PrimitiveReachabilityEdge> edges;
  std::vector<std::uint8_t> reachable;
  std::string algorithm_id;
  std::string state_schema;
  std::string primitive_set_sha256;
  std::string world_evidence_sha256;
  std::string graph_sha256;
  std::uint64_t revision{};
  std::size_t invalidated_edge_count{};
  std::size_t revalidated_edge_count{};
};

struct PrimitiveReachabilityResult final {
  std::optional<PrimitiveReachabilitySnapshot> snapshot;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return snapshot.has_value() && reason_code.empty();
  }
};

class PrimitiveReachabilityEngine final {
 public:
  PrimitiveReachabilityEngine();
  ~PrimitiveReachabilityEngine();
  PrimitiveReachabilityEngine(PrimitiveReachabilityEngine&&) noexcept;
  PrimitiveReachabilityEngine& operator=(
      PrimitiveReachabilityEngine&&) noexcept;
  PrimitiveReachabilityEngine(const PrimitiveReachabilityEngine&) = delete;
  PrimitiveReachabilityEngine& operator=(
      const PrimitiveReachabilityEngine&) = delete;

  [[nodiscard]] PrimitiveReachabilityResult Update(
      const PlannerInput& input,
      std::optional<double> maximum_action_distance_m);
  void Reset() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::planning
