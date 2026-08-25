#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <geometry_msgs/msg/twist.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_pure_exploration_sim/coordinated_steering_plant.hpp"
#include "lunar_pure_exploration_sim/lunar_scene.hpp"
#include "lunar_pure_exploration_sim/visibility.hpp"

namespace lunar::pure_exploration_sim {

struct SimulationMessages {
  nav_msgs::msg::OccupancyGrid global_overview;
  grid_map_msgs::msg::GridMap local_grid_map;
  nav_msgs::msg::Odometry odometry;
  tf2_msgs::msg::TFMessage transforms;
  visualization_msgs::msg::Marker sensor_fov;
  visualization_msgs::msg::MarkerArray vehicle_markers;
  visualization_msgs::msg::MarkerArray local_map_markers;
  nav_msgs::msg::Path actual_path;
  std_msgs::msg::Float64 sim_elapsed;
};

class SimulationNode : public rclcpp::Node {
 public:
  SimulationNode();
  explicit SimulationNode(const rclcpp::NodeOptions& options,
                          bool start_timer = true);

  // Deterministic entry point used by tests and by the steady-clock timer.
  // A single wall step is capped at 0.1 seconds before simulation scaling.
  void Tick(double wall_dt_s);
  void SetCommand(const geometry_msgs::msg::Twist& command);

  [[nodiscard]] const SimulationMessages& latest_messages() const noexcept;
  [[nodiscard]] std::size_t known_global_count() const noexcept;
  [[nodiscard]] std::uint64_t tick_count() const noexcept;
  [[nodiscard]] std::uint64_t deadline_miss_count() const noexcept;
  [[nodiscard]] double last_tick_duration_s() const noexcept;
  [[nodiscard]] double max_tick_duration_s() const noexcept;
  [[nodiscard]] double deadline_period_s() const noexcept;

 private:
  void OnTimer();
  void BuildMessages(const rclcpp::Time& stamp);
  void PublishLatest();

  LunarScene scene_;
  ObservationState observations_;
  PlantState plant_state_;
  PlantParameters plant_parameters_;
  SensorModel sensor_model_;
  BodyCommand command_;
  SimulationMessages messages_;

  double update_rate_hz_{20.0};
  std::uint64_t tick_count_{0U};
  std::uint64_t deadline_miss_count_{0U};
  double last_tick_duration_s_{0.0};
  double max_tick_duration_s_{0.0};
  std::chrono::steady_clock::time_point last_wall_tick_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr local_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr fov_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      vehicle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      local_markers_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr elapsed_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace lunar::pure_exploration_sim
