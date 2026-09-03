#pragma once

#include <cstdint>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace lunar::incremental_navigation_ros {

struct IncrementalDemoPose final {
  double x_m{};
  double y_m{};
  double yaw_rad{};
};

struct IncrementalDemoScenarioConfig final {
  double fine_resolution_m{0.2};
  double local_window_size_m{16.0};
  double task_size_m{300.0};
};

class IncrementalDemoScenario final {
 public:
  explicit IncrementalDemoScenario(IncrementalDemoScenarioConfig config);

  [[nodiscard]] double fine_resolution_m() const noexcept;
  [[nodiscard]] double local_window_size_m() const noexcept;
  [[nodiscard]] grid_map_msgs::msg::GridMap MakeLocalObservation(
      IncrementalDemoPose pose) const;
  [[nodiscard]] visualization_msgs::msg::MarkerArray MakeLocalWindow(
      IncrementalDemoPose pose) const;
  [[nodiscard]] nav_msgs::msg::OccupancyGrid MakeGroundTruth() const;

 private:
  [[nodiscard]] float ElevationAt(double x_m, double y_m) const noexcept;

  IncrementalDemoScenarioConfig config_;
};

}  // namespace lunar::incremental_navigation_ros
