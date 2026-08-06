#include "hierarchical/route_continuation.hpp"

#include <utility>

namespace lunar::planning {

RouteContinuation::RouteContinuation(
    std::string route_id, const PlatformType platform_type,
    hierarchical::GlobalRoute global_route,
    std::vector<CertifiedHopPreview> certified_hops,
    const std::size_t route_cursor)
    : route_id_(std::move(route_id)), platform_type_(platform_type),
      global_route_(std::move(global_route)),
      certified_hops_(std::move(certified_hops)), route_cursor_(route_cursor) {}

const std::string &RouteContinuation::route_id() const noexcept {
  return route_id_;
}

PlatformType RouteContinuation::platform_type() const noexcept {
  return platform_type_;
}

const hierarchical::GlobalRoute &
RouteContinuation::global_route() const noexcept {
  return global_route_;
}

const std::vector<CertifiedHopPreview> &
RouteContinuation::certified_hops() const noexcept {
  return certified_hops_;
}

std::size_t RouteContinuation::route_cursor() const noexcept {
  return route_cursor_;
}

} // namespace lunar::planning
