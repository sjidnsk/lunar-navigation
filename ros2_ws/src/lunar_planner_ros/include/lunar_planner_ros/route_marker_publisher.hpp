#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::ros {

struct RouteMarkerContext final {
  builtin_interfaces::msg::Time stamp;
  std::string route_id;
};

class RouteMarkerPublisher final {
 public:
  [[nodiscard]] visualization_msgs::msg::MarkerArray Replace(
      const lunar::planning::PlannerOutput& output,
      const lunar::planning::PlannerInput& input,
      const RouteMarkerContext& context);

  [[nodiscard]] visualization_msgs::msg::MarkerArray DeleteOwned(
      const builtin_interfaces::msg::Time& stamp);

 private:
  using OwnedMarker = std::tuple<std::string, std::int32_t, std::string>;
  std::set<OwnedMarker> owned_markers_;
};

}  // namespace lunar::planning::ros
