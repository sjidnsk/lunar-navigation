#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <tuple>

#include <builtin_interfaces/msg/time.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_planner_core/planner.hpp"

namespace lunar::planning::ros {

class ProvisionalRouteMarkerPublisher final {
 public:
  [[nodiscard]] visualization_msgs::msg::MarkerArray Replace(
      const lunar::planning::ProvisionalGlobalRoute& route,
      const builtin_interfaces::msg::Time& stamp);

  [[nodiscard]] visualization_msgs::msg::MarkerArray DeleteOwned(
      const builtin_interfaces::msg::Time& stamp);

 private:
  using OwnedMarker = std::tuple<std::string, std::int32_t, std::string>;
  std::set<OwnedMarker> owned_markers_;
};

}  // namespace lunar::planning::ros
