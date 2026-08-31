#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <lunar_planning_msgs/msg/demo_map_ack.hpp>
#include <lunar_planning_msgs/msg/demo_plan_segment.hpp>
#include <nav_msgs/msg/path.hpp>

namespace lunar::pure_planner_ros {

[[nodiscard]] bool LocalMapPublicationReady(
    bool local_map_due, bool startup_delivery_active,
    std::size_t subscriber_count) noexcept;

class LunarSurfaceDemoState final {
 public:
  void Reset(double x_m, double y_m, double yaw_rad = 0.0) noexcept;
  void AcceptPath(const nav_msgs::msg::Path& path);
  void Advance(double step_m) noexcept;

  void EnableDeliveryProtocol(std::uint8_t platform_type) noexcept;
  [[nodiscard]] bool HandleSegment(
      const lunar_planning_msgs::msg::DemoPlanSegment& segment);
  void BeginMapDelivery(const builtin_interfaces::msg::Time& map_token);
  [[nodiscard]] bool HandleMapAck(
      const lunar_planning_msgs::msg::DemoMapAck& ack);
  [[nodiscard]] bool ShouldBeginMapDelivery(
      double update_distance_m) const noexcept;

  [[nodiscard]] bool LocalMapDue(double update_distance_m) const noexcept;
  void MarkLocalMapPublished() noexcept;

  [[nodiscard]] double x_m() const noexcept { return x_m_; }
  [[nodiscard]] double y_m() const noexcept { return y_m_; }
  [[nodiscard]] double yaw_rad() const noexcept { return yaw_rad_; }
  [[nodiscard]] bool has_active_path() const noexcept;
  [[nodiscard]] bool can_advance() const noexcept;
  [[nodiscard]] bool map_delivery_pending() const noexcept {
    return map_delivery_pending_;
  }

 private:
  enum class DeliveryState : std::uint8_t {
    kLegacy,
    kIdle,
    kWaitingForMap,
    kWaitingForAck,
    kWaitingForSegment,
    kExecuting,
  };

  void ClearActivePath() noexcept;
  void ActivateSegment(
      const lunar_planning_msgs::msg::DemoPlanSegment& segment);

  double x_m_{};
  double y_m_{};
  double yaw_rad_{};
  nav_msgs::msg::Path active_path_;
  std::size_t next_path_pose_{};
  std::optional<double> last_local_x_m_;
  std::optional<double> last_local_y_m_;
  DeliveryState delivery_state_{DeliveryState::kLegacy};
  std::uint8_t platform_type_{};
  std::string request_id_;
  std::optional<builtin_interfaces::msg::Time> map_token_;
  std::optional<std::uint32_t> last_segment_index_;
  std::optional<lunar_planning_msgs::msg::DemoPlanSegment> cached_segment_;
  double distance_since_map_delivery_m_{};
  bool map_delivery_pending_{};
};

}  // namespace lunar::pure_planner_ros
