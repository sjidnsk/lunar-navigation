#include "lunar_pure_planner_ros/lunar_surface_demo_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "lunar_pure_planner_ros/lunar_surface_demo_motion.hpp"

namespace lunar::pure_planner_ros {
namespace {

bool SameToken(const builtin_interfaces::msg::Time& left,
               const builtin_interfaces::msg::Time& right) noexcept {
  return left.sec == right.sec && left.nanosec == right.nanosec;
}

std::size_t FirstPoseAfterNearestProjection(const nav_msgs::msg::Path& path,
                                            const double x_m,
                                            const double y_m) noexcept {
  if (path.poses.size() <= 1U) {
    return 0U;
  }

  double best_distance_squared = std::numeric_limits<double>::infinity();
  std::size_t best_next_pose = 1U;
  for (std::size_t next_pose = 1U; next_pose < path.poses.size();
       ++next_pose) {
    const auto& start = path.poses[next_pose - 1U].pose.position;
    const auto& finish = path.poses[next_pose].pose.position;
    const double dx = finish.x - start.x;
    const double dy = finish.y - start.y;
    const double length_squared = dx * dx + dy * dy;
    const double projection = length_squared > 0.0
        ? std::clamp(((x_m - start.x) * dx + (y_m - start.y) * dy) /
                         length_squared,
                     0.0, 1.0)
        : 0.0;
    const double projected_x = start.x + projection * dx;
    const double projected_y = start.y + projection * dy;
    const double offset_x = x_m - projected_x;
    const double offset_y = y_m - projected_y;
    const double distance_squared =
        offset_x * offset_x + offset_y * offset_y;
    if (distance_squared <= best_distance_squared) {
      best_distance_squared = distance_squared;
      best_next_pose = next_pose;
    }
  }
  return best_next_pose;
}

}  // namespace

bool LocalMapPublicationReady(const bool local_map_due,
                              const bool startup_delivery_active,
                              const std::size_t subscriber_count) noexcept {
  // The isolated launch has two required consumers: the planner and the
  // local traversability visualizer. Waiting for both prevents the visualizer
  // from consuming the only startup sample before the planner is discovered.
  return (local_map_due || startup_delivery_active) &&
         subscriber_count >= 2U;
}

void LunarSurfaceDemoState::Reset(const double x_m, const double y_m,
                                  const double yaw_rad) noexcept {
  x_m_ = x_m;
  y_m_ = y_m;
  yaw_rad_ = yaw_rad;
  active_path_ = nav_msgs::msg::Path{};
  next_path_pose_ = 0U;
  last_local_x_m_.reset();
  last_local_y_m_.reset();
  if (delivery_state_ != DeliveryState::kLegacy) {
    delivery_state_ = DeliveryState::kIdle;
    request_id_.clear();
    map_token_.reset();
    last_segment_index_.reset();
    cached_segment_.reset();
    distance_since_map_delivery_m_ = 0.0;
    map_delivery_pending_ = false;
  }
}

void LunarSurfaceDemoState::AcceptPath(const nav_msgs::msg::Path& path) {
  active_path_ = path;
  next_path_pose_ = active_path_.poses.size() > 1U ? 1U : 0U;
}

void LunarSurfaceDemoState::Advance(const double step_m) noexcept {
  if (!std::isfinite(step_m) || step_m <= 0.0 || !can_advance()) {
    return;
  }
  const double previous_x_m = x_m_;
  const double previous_y_m = y_m_;
  AdvanceAlongDemoPath(active_path_, next_path_pose_, x_m_, y_m_, step_m);
  const double dx = x_m_ - previous_x_m;
  const double dy = y_m_ - previous_y_m;
  const double moved_m = std::hypot(dx, dy);
  if (moved_m > 1.0e-9) {
    yaw_rad_ = std::atan2(dy, dx);
    if (delivery_state_ != DeliveryState::kLegacy) {
      distance_since_map_delivery_m_ += moved_m;
    }
  }
}

void LunarSurfaceDemoState::EnableDeliveryProtocol(
    const std::uint8_t platform_type) noexcept {
  platform_type_ = platform_type;
  delivery_state_ = DeliveryState::kIdle;
  request_id_.clear();
  map_token_.reset();
  last_segment_index_.reset();
  cached_segment_.reset();
  distance_since_map_delivery_m_ = 0.0;
  map_delivery_pending_ = false;
  ClearActivePath();
}

bool LunarSurfaceDemoState::HandleSegment(
    const lunar_planning_msgs::msg::DemoPlanSegment& segment) {
  if (delivery_state_ == DeliveryState::kLegacy ||
      segment.platform_type != platform_type_) {
    return false;
  }

  if (segment.command == lunar_planning_msgs::msg::DemoPlanSegment::STOP) {
    if (segment.request_id.empty()) {
      return false;
    }
    request_id_ = segment.request_id;
    map_token_.reset();
    last_segment_index_.reset();
    cached_segment_.reset();
    distance_since_map_delivery_m_ = 0.0;
    map_delivery_pending_ = false;
    ClearActivePath();
    delivery_state_ = DeliveryState::kWaitingForMap;
    return true;
  }

  if (segment.command != lunar_planning_msgs::msg::DemoPlanSegment::EXECUTE ||
      request_id_.empty() || segment.request_id != request_id_ ||
      !map_token_.has_value() ||
      !SameToken(segment.map_token, *map_token_) ||
      (last_segment_index_.has_value() &&
       segment.segment_index <= *last_segment_index_)) {
    return false;
  }

  last_segment_index_ = segment.segment_index;
  if (map_delivery_pending_) {
    cached_segment_ = segment;
    return true;
  }
  ActivateSegment(segment);
  return true;
}

void LunarSurfaceDemoState::BeginMapDelivery(
    const builtin_interfaces::msg::Time& map_token) {
  if (delivery_state_ == DeliveryState::kLegacy || request_id_.empty()) {
    return;
  }
  map_token_ = map_token;
  cached_segment_.reset();
  distance_since_map_delivery_m_ = 0.0;
  map_delivery_pending_ = true;
  delivery_state_ = DeliveryState::kWaitingForAck;
}

bool LunarSurfaceDemoState::HandleMapAck(
    const lunar_planning_msgs::msg::DemoMapAck& ack) {
  if (delivery_state_ != DeliveryState::kWaitingForAck ||
      !map_delivery_pending_ || ack.platform_type != platform_type_ ||
      ack.request_id != request_id_ || !map_token_.has_value() ||
      !SameToken(ack.map_token, *map_token_)) {
    return false;
  }

  map_delivery_pending_ = false;
  if (cached_segment_.has_value()) {
    ActivateSegment(*cached_segment_);
    cached_segment_.reset();
  } else {
    delivery_state_ = DeliveryState::kWaitingForSegment;
  }
  return true;
}

bool LunarSurfaceDemoState::ShouldBeginMapDelivery(
    const double update_distance_m) const noexcept {
  if (delivery_state_ != DeliveryState::kExecuting || map_delivery_pending_ ||
      !std::isfinite(update_distance_m) || update_distance_m <= 0.0) {
    return false;
  }
  return distance_since_map_delivery_m_ >= update_distance_m ||
         !has_active_path();
}

bool LunarSurfaceDemoState::LocalMapDue(
    const double update_distance_m) const noexcept {
  if (!last_local_x_m_.has_value() || !last_local_y_m_.has_value()) {
    return true;
  }
  return std::isfinite(update_distance_m) && update_distance_m > 0.0 &&
         std::hypot(x_m_ - *last_local_x_m_, y_m_ - *last_local_y_m_) >=
             update_distance_m;
}

void LunarSurfaceDemoState::MarkLocalMapPublished() noexcept {
  last_local_x_m_ = x_m_;
  last_local_y_m_ = y_m_;
}

bool LunarSurfaceDemoState::has_active_path() const noexcept {
  return next_path_pose_ < active_path_.poses.size();
}

bool LunarSurfaceDemoState::can_advance() const noexcept {
  return has_active_path() &&
         (delivery_state_ == DeliveryState::kLegacy ||
          delivery_state_ == DeliveryState::kExecuting ||
          delivery_state_ == DeliveryState::kWaitingForSegment);
}

void LunarSurfaceDemoState::ClearActivePath() noexcept {
  active_path_ = nav_msgs::msg::Path{};
  next_path_pose_ = 0U;
}

void LunarSurfaceDemoState::ActivateSegment(
    const lunar_planning_msgs::msg::DemoPlanSegment& segment) {
  active_path_ = segment.executable_path;
  next_path_pose_ = FirstPoseAfterNearestProjection(active_path_, x_m_, y_m_);
  delivery_state_ = DeliveryState::kExecuting;
}

}  // namespace lunar::pure_planner_ros
