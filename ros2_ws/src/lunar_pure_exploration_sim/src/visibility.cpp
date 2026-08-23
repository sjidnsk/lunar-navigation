#include "lunar_pure_exploration_sim/visibility.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

namespace lunar::pure_exploration_sim {
namespace {

constexpr std::size_t kGlobalWidth = 300U;
constexpr std::size_t kGlobalHeight = 300U;
constexpr double kTolerance = 1.0e-9;

double NormalizeAngle(double angle_rad) {
  return std::remainder(angle_rad, 2.0 * std::numbers::pi);
}

bool FinitePose(const Pose2& pose) {
  return std::isfinite(pose.x_m) && std::isfinite(pose.y_m) &&
         std::isfinite(pose.yaw_rad);
}

bool ValidSensor(const SensorModel& sensor) {
  return std::isfinite(sensor.range_m) && sensor.range_m > 0.0 &&
         std::isfinite(sensor.horizontal_fov_rad) &&
         sensor.horizontal_fov_rad > 0.0 &&
         sensor.horizontal_fov_rad <= 2.0 * std::numbers::pi &&
         std::isfinite(sensor.radial_step_m) && sensor.radial_step_m > 0.0 &&
         std::isfinite(sensor.angular_step_rad) &&
         sensor.angular_step_rad > 0.0 &&
         sensor.angular_step_rad <= std::atan(0.1 / 10.0) + kTolerance;
}

}  // namespace

ObservationState::ObservationState()
    : known_global_(kGlobalWidth * kGlobalHeight, false) {}

void ObservationState::Observe(const LunarScene& scene, Pose2 pose,
                               SensorModel sensor) {
  if (!FinitePose(pose) || !ValidSensor(sensor)) {
    has_observation_ = false;
    return;
  }
  current_pose_ = pose;
  current_sensor_ = sensor;
  has_observation_ = true;

  const std::size_t ray_intervals = static_cast<std::size_t>(std::ceil(
      sensor.horizontal_fov_rad / sensor.angular_step_rad));
  const double angular_step =
      sensor.horizontal_fov_rad / static_cast<double>(ray_intervals);
  const double first_angle = pose.yaw_rad - sensor.horizontal_fov_rad / 2.0;
  for (std::size_t ray = 0U; ray <= ray_intervals; ++ray) {
    const double angle = first_angle + static_cast<double>(ray) * angular_step;
    for (double distance_m = 0.0;
         distance_m <= sensor.range_m + kTolerance;
         distance_m += sensor.radial_step_m) {
      const double world_x_m = pose.x_m + distance_m * std::cos(angle);
      const double world_y_m = pose.y_m + distance_m * std::sin(angle);
      if (world_x_m < scene.min_x_m() || world_x_m >= scene.max_x_m() ||
          world_y_m < scene.min_x_m() || world_y_m >= scene.max_x_m()) {
        break;
      }
      MarkKnown(scene, world_x_m, world_y_m);
      if (scene.Sample(world_x_m, world_y_m).occupied) {
        break;
      }
    }
  }

  const auto minimum_x = static_cast<long>(std::floor(
      (pose.x_m - sensor.range_m - scene.min_x_m()) /
      scene.global_resolution_m()));
  const auto maximum_x = static_cast<long>(std::floor(
      (pose.x_m + sensor.range_m - scene.min_x_m()) /
      scene.global_resolution_m()));
  const auto minimum_y = static_cast<long>(std::floor(
      (pose.y_m - sensor.range_m - scene.min_x_m()) /
      scene.global_resolution_m()));
  const auto maximum_y = static_cast<long>(std::floor(
      (pose.y_m + sensor.range_m - scene.min_x_m()) /
      scene.global_resolution_m()));
  for (long y = std::max(0L, minimum_y);
       y <= std::min(static_cast<long>(scene.global_height()) - 1L, maximum_y);
       ++y) {
    for (long x = std::max(0L, minimum_x);
         x <= std::min(static_cast<long>(scene.global_width()) - 1L, maximum_x);
         ++x) {
      const double world_x_m =
          scene.min_x_m() + (static_cast<double>(x) + 0.5) *
                                scene.global_resolution_m();
      const double world_y_m =
          scene.min_x_m() + (static_cast<double>(y) + 0.5) *
                                scene.global_resolution_m();
      if (IsCurrentlyVisible(scene, world_x_m, world_y_m)) {
        known_global_[static_cast<std::size_t>(y) * scene.global_width() +
                      static_cast<std::size_t>(x)] = true;
      }
    }
  }
}

bool ObservationState::IsCurrentlyVisible(const LunarScene& scene,
                                          double world_x_m,
                                          double world_y_m) const {
  if (!has_observation_ || !std::isfinite(world_x_m) ||
      !std::isfinite(world_y_m) || world_x_m < scene.min_x_m() ||
      world_x_m >= scene.max_x_m() || world_y_m < scene.min_x_m() ||
      world_y_m >= scene.max_x_m()) {
    return false;
  }
  const double delta_x_m = world_x_m - current_pose_.x_m;
  const double delta_y_m = world_y_m - current_pose_.y_m;
  const double distance_m = std::hypot(delta_x_m, delta_y_m);
  if (distance_m > current_sensor_.range_m + kTolerance) {
    return false;
  }
  if (distance_m <= kTolerance) {
    return true;
  }
  const double yaw_offset =
      NormalizeAngle(std::atan2(delta_y_m, delta_x_m) - current_pose_.yaw_rad);
  if (std::abs(yaw_offset) > current_sensor_.horizontal_fov_rad / 2.0 +
                                 kTolerance) {
    return false;
  }

  for (double ray_distance_m = current_sensor_.radial_step_m;
       ray_distance_m < distance_m - kTolerance;
       ray_distance_m += current_sensor_.radial_step_m) {
    const double ratio = ray_distance_m / distance_m;
    if (scene.Sample(current_pose_.x_m + ratio * delta_x_m,
                     current_pose_.y_m + ratio * delta_y_m)
            .occupied) {
      return false;
    }
  }
  return true;
}

bool ObservationState::IsKnownGlobalCell(std::size_t x,
                                         std::size_t y) const noexcept {
  return x < kGlobalWidth && y < kGlobalHeight &&
         known_global_[y * kGlobalWidth + x];
}

std::size_t ObservationState::KnownGlobalCount() const noexcept {
  return static_cast<std::size_t>(
      std::count(known_global_.begin(), known_global_.end(), true));
}

const std::vector<bool>& ObservationState::KnownGlobalMask() const noexcept {
  return known_global_;
}

void ObservationState::MarkKnown(const LunarScene& scene, double world_x_m,
                                 double world_y_m) {
  const auto x = static_cast<std::size_t>(std::floor(
      (world_x_m - scene.min_x_m()) / scene.global_resolution_m()));
  const auto y = static_cast<std::size_t>(std::floor(
      (world_y_m - scene.min_x_m()) / scene.global_resolution_m()));
  if (x < scene.global_width() && y < scene.global_height()) {
    known_global_[y * scene.global_width() + x] = true;
  }
}

}  // namespace lunar::pure_exploration_sim
