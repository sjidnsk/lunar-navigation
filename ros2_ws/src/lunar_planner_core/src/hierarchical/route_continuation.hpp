#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "hierarchical/global_route.hpp"
#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning {

class RouteContinuation final {
public:
  RouteContinuation(std::string route_id, PlatformType platform_type,
                    hierarchical::GlobalRoute global_route,
                    std::vector<CertifiedHopPreview> certified_hops,
                    std::size_t route_cursor = 0U);

  [[nodiscard]] const std::string &route_id() const noexcept;
  [[nodiscard]] PlatformType platform_type() const noexcept;
  [[nodiscard]] const hierarchical::GlobalRoute &global_route() const noexcept;
  [[nodiscard]] const std::vector<CertifiedHopPreview> &
  certified_hops() const noexcept;
  [[nodiscard]] std::size_t route_cursor() const noexcept;

private:
  std::string route_id_;
  PlatformType platform_type_{PlatformType::kWheeled};
  hierarchical::GlobalRoute global_route_;
  std::vector<CertifiedHopPreview> certified_hops_;
  std::size_t route_cursor_{};
};

} // namespace lunar::planning
