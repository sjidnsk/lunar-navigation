#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_planning_msgs/msg/demo_map_ack.hpp>
#include <lunar_planning_msgs/msg/demo_plan_segment.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_pure_planner_ros/lunar_surface_demo_state.hpp"
#include "lunar_pure_planner_ros/lunar_surface_local_map.hpp"
#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"
#include "lunar_pure_planner_ros/lunar_surface_traversability_viz.hpp"

namespace lunar::pure_planner_ros {
namespace {

constexpr std::size_t kLocalWidth = 320U;
constexpr std::size_t kLocalHeight = 320U;
constexpr double kLocalResolutionM = 0.2;
constexpr double kLocalMapUpdateDistanceM = 4.0;
constexpr double kRoverStepM = 0.5;
constexpr double kLeggedNominalBodyHeightM = 0.33;

std_msgs::msg::Float32MultiArray MakeLayer(const std::vector<float>& values,
                                           const std::size_t width,
                                           const std::size_t height) {
  std_msgs::msg::Float32MultiArray layer;
  layer.layout.dim.resize(2U);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = height;
  layer.layout.dim[0].stride = width * height;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = width;
  layer.layout.dim[1].stride = width;
  layer.data.resize(values.size());
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const std::size_t logical = y * width + x;
      const std::size_t physical = (height - 1U - y) * width + (width - 1U - x);
      layer.data[physical] = values[logical];
    }
  }
  return layer;
}

class LunarSurfaceDemoNode final : public rclcpp::Node {
 public:
  LunarSurfaceDemoNode()
      : Node("lunar_surface_demo"),
        scenario_(BuildLunarSurfaceScenario(
            static_cast<std::uint32_t>(declare_parameter<std::int64_t>("seed", 20260823)))),
        platform_type_(declare_parameter<std::string>("platform_type", "wheel")),
        auto_goal_(declare_parameter<bool>("auto_goal", false)),
        delivery_protocol_enabled_(
            declare_parameter<bool>("demo_delivery_protocol_enabled", false)),
        demo_map_ack_topic_(declare_parameter<std::string>(
            "demo_map_ack_topic", "/lunar_demo/map_ack")),
        demo_plan_segment_topic_(declare_parameter<std::string>(
            "demo_plan_segment_topic", "/lunar_demo/plan_segment")),
        rolling_replan_distance_m_(
            declare_parameter<double>("rolling_replan_distance_m",
                                      kLocalMapUpdateDistanceM)) {
    (void)declare_parameter<double>("rolling_horizon_m", 12.0);
    // RViz and tf2 listeners request reliable delivery by default.  A reliable
    // writer also remains compatible with the planner's best-effort readers.
    const auto input_qos = rclcpp::QoS{10}.reliable();
    global_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/lunar_demo/global_overview", input_qos);
    global_grid_pub_ = create_publisher<grid_map_msgs::msg::GridMap>(
        "/lunar_demo/global_grid_map",
        rclcpp::QoS{1}.reliable().transient_local());
    local_pub_ = create_publisher<grid_map_msgs::msg::GridMap>(
        "/lunar_demo/grid_map", input_qos);
    local_viz_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/lunar_demo/local_map_viz", input_qos);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/lunar_demo/odometry", input_qos);
    tf_pub_ = create_publisher<tf2_msgs::msg::TFMessage>("/tf", input_qos);
    goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/lunar_demo/rviz_goal", rclcpp::QoS{1}.reliable());
    default_goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/lunar_demo/default_goal", rclcpp::QoS{1}.transient_local());
    accepted_start_pub_ =
        create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/lunar_demo/accepted_start", rclcpp::QoS{1}.transient_local());
    state_.Reset(scenario_.origin_x_m +
                     (static_cast<double>(scenario_.start_cell.x) + 0.5) *
                         scenario_.resolution_m,
                 scenario_.origin_y_m +
                     (static_cast<double>(scenario_.start_cell.y) + 0.5) *
                         scenario_.resolution_m);
    if (delivery_protocol_enabled_) {
      const auto platform = platform_type_ == "legged"
          ? lunar_planning_msgs::msg::DemoPlanSegment::LEGGED
          : lunar_planning_msgs::msg::DemoPlanSegment::WHEELED;
      state_.EnableDeliveryProtocol(platform);
      map_ack_sub_ = create_subscription<lunar_planning_msgs::msg::DemoMapAck>(
          demo_map_ack_topic_, rclcpp::QoS{10}.reliable(),
          [this](lunar_planning_msgs::msg::DemoMapAck::ConstSharedPtr ack) {
            if (state_.HandleMapAck(*ack)) {
              static_inputs_acknowledged_ = true;
              delivery_map_token_.reset();
            }
          });
      plan_segment_sub_ =
          create_subscription<lunar_planning_msgs::msg::DemoPlanSegment>(
              demo_plan_segment_topic_, rclcpp::QoS{10}.reliable(),
              [this](lunar_planning_msgs::msg::DemoPlanSegment::ConstSharedPtr segment) {
                const bool is_stop =
                    segment->command == lunar_planning_msgs::msg::DemoPlanSegment::STOP;
                if (state_.HandleSegment(*segment) && is_stop) {
                  static_inputs_acknowledged_ = false;
                  const auto map_token = static_cast<builtin_interfaces::msg::Time>(now());
                  state_.BeginMapDelivery(map_token);
                  if (state_.map_delivery_pending()) {
                    delivery_map_token_ = map_token;
                  }
                }
              });
    } else {
      const auto accept_path = [this](nav_msgs::msg::Path::ConstSharedPtr path) {
        state_.AcceptPath(*path);
      };
      wheeled_path_sub_ = create_subscription<nav_msgs::msg::Path>(
          "/lunar_demo/wheeled_path", rclcpp::QoS{10}.reliable(),
          accept_path);
      legged_path_sub_ = create_subscription<nav_msgs::msg::Path>(
          "/lunar_demo/legged_path", rclcpp::QoS{10}.reliable(), accept_path);
    }
    start_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/lunar_demo/start_pose", rclcpp::QoS{10}.reliable(),
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr start) {
          SetStart(*start);
        });
    timer_ = create_wall_timer(std::chrono::milliseconds{500}, [this] { Publish(); });
  }

 private:
  [[nodiscard]] geometry_msgs::msg::PoseStamped Goal(const rclcpp::Time& stamp) const {
    geometry_msgs::msg::PoseStamped message;
    message.header.stamp = stamp;
    message.header.frame_id = "map";
    message.pose.position.x = scenario_.origin_x_m +
                              (static_cast<double>(scenario_.default_goal_cell.x) + 0.5) * scenario_.resolution_m;
    message.pose.position.y = scenario_.origin_y_m +
                              (static_cast<double>(scenario_.default_goal_cell.y) + 0.5) * scenario_.resolution_m;
    message.pose.orientation.w = 1.0;
    return message;
  }

  void SetStart(const geometry_msgs::msg::PoseWithCovarianceStamped& start) {
    const double x_m = start.pose.pose.position.x;
    const double y_m = start.pose.pose.position.y;
    const auto sample = scenario_.Sample(x_m, y_m);
    if ((start.header.frame_id != "map" && start.header.frame_id != "odom") ||
        !std::isfinite(x_m) || !std::isfinite(y_m) || !sample.has_value() ||
        sample->occupied) {
      RCLCPP_WARN(get_logger(),
                  "Rejected RViz start pose at (%.2f, %.2f): outside known FREE terrain",
                  x_m, y_m);
      return;
    }
    const auto& orientation = start.pose.pose.orientation;
    const double sine = 2.0 *
        (orientation.w * orientation.z + orientation.x * orientation.y);
    const double cosine = 1.0 - 2.0 *
        (orientation.y * orientation.y + orientation.z * orientation.z);
    state_.Reset(x_m, y_m, std::atan2(sine, cosine));
    auto accepted = start;
    accepted.header.frame_id = "map";
    accepted_start_pub_->publish(accepted);
    RCLCPP_INFO(get_logger(), "Accepted RViz start pose at (%.2f, %.2f)",
                state_.x_m(), state_.y_m());
  }

  void Publish() {
    state_.Advance(kRoverStepM);
    const rclcpp::Time stamp = now();
    if (delivery_protocol_enabled_ &&
        state_.ShouldBeginMapDelivery(rolling_replan_distance_m_)) {
      state_.BeginMapDelivery(static_cast<builtin_interfaces::msg::Time>(stamp));
      if (state_.map_delivery_pending()) {
        delivery_map_token_ = static_cast<builtin_interfaces::msg::Time>(stamp);
      }
    }
    builtin_interfaces::msg::Time input_stamp = stamp;
    if (delivery_protocol_enabled_ && state_.map_delivery_pending() &&
        delivery_map_token_.has_value()) {
      input_stamp = *delivery_map_token_;
    }
    const double start_x = state_.x_m();
    const double start_y = state_.y_m();
    nav_msgs::msg::OccupancyGrid global;
    global.header.stamp = input_stamp;
    global.header.frame_id = "map";
    global.info.resolution = scenario_.resolution_m;
    global.info.width = scenario_.width;
    global.info.height = scenario_.height;
    global.info.origin.position.x = scenario_.origin_x_m;
    global.info.origin.position.y = scenario_.origin_y_m;
    global.info.origin.orientation.w = 1.0;
    global.data = scenario_.occupancy;
    if (!global_grid_published_) {
      auto global_grid = MakeGlobalTraversabilityInput(scenario_);
      global_grid.header.stamp = stamp;
      global_grid_pub_->publish(global_grid);
      global_grid_published_ = true;
    }
    // Global terrain and map->odom are static in this test-only scene.
    // Republishing them after rolling has started increments their input
    // sequences and forces an unnecessary global-route rebuild every cycle.
    const bool planner_inputs_discovered =
        global_pub_->get_subscription_count() > 0U &&
        local_pub_->get_subscription_count() >= 2U &&
        odom_pub_->get_subscription_count() >= 2U &&
        tf_pub_->get_subscription_count() > 0U;
    const bool protocol_static_delivery =
        delivery_protocol_enabled_ && state_.map_delivery_pending() &&
        !static_inputs_acknowledged_;
    const bool deliver_global_input =
        (!delivery_protocol_enabled_ && static_delivery_count_ < 12U) ||
        protocol_static_delivery;
    if (deliver_global_input) {
      global_pub_->publish(global);
    }

    const bool should_publish_local = delivery_protocol_enabled_
        ? state_.map_delivery_pending() && local_pub_->get_subscription_count() >= 2U
        : LocalMapPublicationReady(
              state_.LocalMapDue(rolling_replan_distance_m_),
              static_delivery_count_ < 12U,
              local_pub_->get_subscription_count());
    if (should_publish_local) {
      const LunarSurfaceLocalRaster raster =
          BuildLunarSurfaceLocalRaster(scenario_, start_x, start_y);
      grid_map_msgs::msg::GridMap local;
      local.header.stamp = input_stamp;
      local.header.frame_id = "odom";
      local.info.resolution = kLocalResolutionM;
      local.info.length_x = raster.length_x_m();
      local.info.length_y = raster.length_y_m();
      local.info.pose.position.x = start_x;
      local.info.pose.position.y = start_y;
      local.info.pose.orientation.w = 1.0;
      local.layers = {"occupancy", "elevation"};
      local.basic_layers = local.layers;
      local.data = {MakeLayer(raster.occupancy, kLocalWidth, kLocalHeight),
                    MakeLayer(raster.elevation_m, kLocalWidth, kLocalHeight)};
      local_pub_->publish(local);

      nav_msgs::msg::OccupancyGrid local_viz;
      local_viz.header = local.header;
      local_viz.info.resolution = kLocalResolutionM;
      local_viz.info.width = kLocalWidth;
      local_viz.info.height = kLocalHeight;
      local_viz.info.origin.position.x =
          start_x - raster.length_x_m() / 2.0;
      local_viz.info.origin.position.y =
          start_y - raster.length_y_m() / 2.0;
      local_viz.info.origin.position.z = 0.05;
      local_viz.info.origin.orientation.w = 1.0;
      local_viz.data.resize(raster.occupancy.size(), -1);
      for (std::size_t index = 0U; index < raster.occupancy.size(); ++index) {
        if (std::isfinite(raster.occupancy[index])) {
          local_viz.data[index] = raster.occupancy[index] >= 0.5F ? 100 : 0;
        }
      }
      local_viz_pub_->publish(local_viz);
      if (!state_.map_delivery_pending()) {
        state_.MarkLocalMapPublished();
      }
    }

    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = input_stamp;
    odometry.header.frame_id = "odom";
    odometry.child_frame_id = "base_link";
    odometry.pose.pose.position.x = start_x;
    odometry.pose.pose.position.y = start_y;
    const auto rover_sample = scenario_.Sample(start_x, start_y);
    odometry.pose.pose.position.z =
        (rover_sample.has_value() ? rover_sample->elevation_m : 0.0) +
        (platform_type_ == "legged" ? kLeggedNominalBodyHeightM : 0.0);
    odometry.pose.pose.orientation.z = std::sin(0.5 * state_.yaw_rad());
    odometry.pose.pose.orientation.w = std::cos(0.5 * state_.yaw_rad());
    odom_pub_->publish(odometry);

    const bool deliver_tf_input =
        static_delivery_count_ < 12U || protocol_static_delivery;
    if (deliver_tf_input) {
      tf2_msgs::msg::TFMessage transforms;
      transforms.transforms.resize(1U);
      auto& transform = transforms.transforms.front();
      transform.header.stamp = input_stamp;
      transform.header.frame_id = "map";
      transform.child_frame_id = "odom";
      transform.transform.rotation.w = 1.0;
      tf_pub_->publish(transforms);
    }

    const auto goal = Goal(stamp);
    default_goal_pub_->publish(goal);
    // The 1 km map contains one million cells.  Wait for twelve 2 Hz
    // publications before issuing the automatic goal so every planner input
    // has reached its callback cache; the former 1.5 s delay raced startup
    // and produced an INVALID_INPUT result.
    if (planner_inputs_discovered && static_delivery_count_ < 12U) {
      ++static_delivery_count_;
      if (static_delivery_count_ == 12U && auto_goal_) {
        goal_pub_->publish(goal);
      }
    }
  }

  LunarSurfaceScenario scenario_;
  LunarSurfaceDemoState state_;
  std::string platform_type_;
  bool auto_goal_{};
  bool delivery_protocol_enabled_{};
  std::string demo_map_ack_topic_;
  std::string demo_plan_segment_topic_;
  double rolling_replan_distance_m_{kLocalMapUpdateDistanceM};
  std::optional<builtin_interfaces::msg::Time> delivery_map_token_;
  std::size_t static_delivery_count_{};
  bool static_inputs_acknowledged_{};
  bool global_grid_published_{};
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr global_grid_pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr local_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr local_viz_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr default_goal_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      accepted_start_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr wheeled_path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr legged_path_sub_;
  rclcpp::Subscription<lunar_planning_msgs::msg::DemoMapAck>::SharedPtr
      map_ack_sub_;
  rclcpp::Subscription<lunar_planning_msgs::msg::DemoPlanSegment>::SharedPtr
      plan_segment_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      start_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace
}  // namespace lunar::pure_planner_ros

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lunar::pure_planner_ros::LunarSurfaceDemoNode>());
  rclcpp::shutdown();
  return 0;
}
