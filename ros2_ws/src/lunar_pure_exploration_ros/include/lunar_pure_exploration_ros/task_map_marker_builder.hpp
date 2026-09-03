#pragma once

#include <string_view>

#include <lunar_pure_exploration_core/task_raster.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace lunar::pure_exploration_ros {

class TaskMapMarkerBuilder final {
 public:
  visualization_msgs::msg::MarkerArray Build(
      const lunar::pure_exploration::TaskRaster& raster,
      std::string_view frame_id) const;
};

}  // namespace lunar::pure_exploration_ros
