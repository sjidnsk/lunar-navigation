#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "hierarchical/global_route.hpp"
#include "lunar_pure_planner_core/types/planner_io.hpp"

namespace lunar::pure_planning {

class RouteContinuation final {
 public:
  RouteContinuation(std::string route_id, std::string current_reference_plan_id,
                    const PlannerInput &input,
                    hierarchical::GlobalRoute global_route,
                    std::vector<CertifiedHopPreview> certified_hops,
                    std::size_t route_cursor = 0U,
                    double corridor_half_width_m = 0.0,
                    std::uint64_t rolling_request_count = 1U);

  [[nodiscard]] const std::string &route_id() const noexcept;
  [[nodiscard]] const std::string &current_reference_plan_id() const noexcept;
  [[nodiscard]] PlatformType platform_type() const noexcept;
  [[nodiscard]] const std::string &mission_id() const noexcept;
  [[nodiscard]] std::uint64_t mission_revision() const noexcept;
  [[nodiscard]] const std::string &platform_id() const noexcept;
  [[nodiscard]] const GoalRegion &goal_map() const noexcept;
  [[nodiscard]] const std::string &capability_version() const noexcept;
  [[nodiscard]] std::uint64_t global_map_generation() const noexcept;
  [[nodiscard]] std::uint64_t local_map_generation_at_issue() const noexcept;
  [[nodiscard]] std::uint64_t map_from_odom_generation() const noexcept;
  [[nodiscard]] const RigidTransform &map_from_odom_at_issue() const noexcept;
  [[nodiscard]] const hierarchical::GlobalRoute &global_route() const noexcept;
  [[nodiscard]] const std::vector<CertifiedHopPreview> &certified_hops()
      const noexcept;
  [[nodiscard]] std::size_t route_cursor() const noexcept;
  [[nodiscard]] double corridor_half_width_m() const noexcept;
  [[nodiscard]] std::uint64_t rolling_request_count() const noexcept;

 private:
  std::string route_id_;
  std::string current_reference_plan_id_;
  PlatformType platform_type_{PlatformType::kWheeled};
  std::string mission_id_;
  std::uint64_t mission_revision_{};
  std::string platform_id_;
  GoalRegion goal_map_;
  std::string capability_version_;
  std::uint64_t global_map_generation_{};
  std::uint64_t local_map_generation_at_issue_{};
  std::uint64_t map_from_odom_generation_{};
  RigidTransform map_from_odom_at_issue_;
  hierarchical::GlobalRoute global_route_;
  std::vector<CertifiedHopPreview> certified_hops_;
  std::size_t route_cursor_{};
  double corridor_half_width_m_{};
  std::uint64_t rolling_request_count_{1U};
};

namespace hierarchical {

struct GroundRouteReuseResult final {
  std::optional<GlobalRoute> route;
  std::size_t route_cursor{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return route.has_value() && reason_code == "GROUND_ROUTE_REUSED";
  }
};

struct HopperHopPromotionResult final {
  std::optional<CertifiedHopPreview> hop;
  std::size_t route_cursor{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return hop.has_value() && reason_code == "HOPPER_HOP_PROMOTED";
  }
};

[[nodiscard]] GroundRouteReuseResult TryReuseGroundRoute(
    const PlannerInput &input, const RouteContinuation &continuation);

[[nodiscard]] HopperHopPromotionResult TryPromoteHopperHop(
    const PlannerInput &input, const RouteContinuation &continuation);

}  // namespace hierarchical

}  // namespace lunar::pure_planning
