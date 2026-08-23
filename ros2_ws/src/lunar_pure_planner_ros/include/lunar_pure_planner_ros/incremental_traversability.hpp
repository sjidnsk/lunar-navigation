#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <grid_map_msgs/msg/grid_map.hpp>

#include "lunar_pure_planner_core/types/platform_capability.hpp"

namespace lunar::pure_planner_ros {

struct TraversabilityProfile final {
  double support_radius_m{};
  double maximum_slope_rad{};
  double occupancy_threshold{0.5};
};

[[nodiscard]] TraversabilityProfile MakeTraversabilityProfile(
    const lunar::pure_planning::PlatformCapability& capability,
    double occupancy_threshold);


struct TraversabilityUpdate final {
  grid_map_msgs::msg::GridMap map;
  bool full_rebuild{};
  std::size_t recomputed_cells{};
};

class IncrementalTraversability final {
 public:
  explicit IncrementalTraversability(TraversabilityProfile profile);

  [[nodiscard]] std::optional<TraversabilityUpdate> Update(
      const grid_map_msgs::msg::GridMap& local_map);

 private:
  TraversabilityProfile profile_;
  bool initialized_{};
  std::string frame_id_;
  std::size_t width_{};
  std::size_t height_{};
  double resolution_m_{};
  double origin_x_m_{};
  double origin_y_m_{};
  double origin_z_m_{};
  std::vector<float> occupancy_;
  std::vector<float> elevation_;
  std::vector<float> traversability_;
};

}  // namespace lunar::pure_planner_ros
