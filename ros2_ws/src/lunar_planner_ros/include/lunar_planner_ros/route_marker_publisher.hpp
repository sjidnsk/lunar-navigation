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
  std::string reference_plan_id;
  std::string authorized_segment_id;
  lunar::planning::RigidTransform map_from_odom;
  lunar::planning::Vec3 gravity_mps2;
};

class RouteMarkerPublisher final {
 public:
  [[nodiscard]] visualization_msgs::msg::MarkerArray Replace(
      const std::vector<lunar::planning::CertifiedHopPreview>& hops,
      const RouteMarkerContext& context);

  [[nodiscard]] visualization_msgs::msg::MarkerArray DeleteOwned(
      const builtin_interfaces::msg::Time& stamp);

 private:
  using OwnedMarker = std::tuple<std::string, std::int32_t, std::string>;
  std::set<OwnedMarker> owned_markers_;
};

}  // namespace lunar::planning::ros
