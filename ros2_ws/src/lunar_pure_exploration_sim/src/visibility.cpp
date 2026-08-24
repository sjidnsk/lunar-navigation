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
constexpr std::size_t kLocalWidth = 320U;
constexpr std::size_t kLocalHeight = 320U;
constexpr double kLocalResolutionM = 0.2;
constexpr double kLocalHalfLengthM = 32.0;
constexpr double kFrozenRangeM = 10.0;
constexpr double kFrozenFovRad = std::numbers::pi / 2.0;
constexpr double kFrozenRadialStepM = 0.1;
constexpr double kWheelbaseM = 0.8175;
constexpr double kTrackWidthM = 0.67;
constexpr double kMinimumClearanceM = 0.2;
constexpr double kGlobalCellHalfDiagonalM = std::numbers::sqrt2 / 2.0;
// At the origin the continuous vehicle pose lies on a 1 m global-cell
// corner. Seed the complete 3x3 coarse-cell stencil as the initial contact
// prior so the coarse global map represents the known physical start state.
constexpr double kInitialGlobalStartStencilRadiusM =
    3.0 * kGlobalCellHalfDiagonalM;
constexpr double kGlobalKnownEnvelopeMarginM =
    std::hypot(kWheelbaseM / 2.0, kTrackWidthM / 2.0) +
    kMinimumClearanceM + kGlobalCellHalfDiagonalM;
constexpr double kInitialKnownStartRadiusM = 4.0;
constexpr double kFootprintHalfLengthM = 0.591;
constexpr double kFootprintHalfWidthM = 0.409;
constexpr double kLocalCellHalfDiagonalM =
    kLocalResolutionM * std::numbers::sqrt2 / 2.0;
constexpr double kLocalContactEnvelopeRadiusM =
    std::hypot(kFootprintHalfLengthM, kFootprintHalfWidthM) +
    kMinimumClearanceM + kLocalCellHalfDiagonalM;
constexpr double kTolerance = 1.0e-9;
constexpr double kContractTolerance = 1.0e-12;

double NormalizeAngle(double angle_rad) {
  return std::remainder(angle_rad, 2.0 * std::numbers::pi);
}

bool FinitePose(const Pose2& pose) {
  return std::isfinite(pose.x_m) && std::isfinite(pose.y_m) &&
         std::isfinite(pose.yaw_rad);
}

bool ValidSensor(const SensorModel& sensor) {
  return std::isfinite(sensor.range_m) &&
         std::abs(sensor.range_m - kFrozenRangeM) <= kContractTolerance &&
         std::isfinite(sensor.horizontal_fov_rad) &&
         std::abs(sensor.horizontal_fov_rad - kFrozenFovRad) <=
             kContractTolerance &&
         std::isfinite(sensor.radial_step_m) &&
         std::abs(sensor.radial_step_m - kFrozenRadialStepM) <=
             kContractTolerance &&
         std::isfinite(sensor.angular_step_rad) &&
         sensor.angular_step_rad > 0.0 &&
         sensor.angular_step_rad <= std::atan(0.1 / 10.0) +
                                        kContractTolerance;
}

bool SamePose(const Pose2& left, const Pose2& right) {
  return std::abs(left.x_m - right.x_m) <= kContractTolerance &&
         std::abs(left.y_m - right.y_m) <= kContractTolerance &&
         std::abs(NormalizeAngle(left.yaw_rad - right.yaw_rad)) <=
             kContractTolerance;
}

}  // namespace

double GlobalKnownEnvelopeMarginM() noexcept {
  return kGlobalKnownEnvelopeMarginM;
}

double LocalContactEnvelopeRadiusM() noexcept {
  return kLocalContactEnvelopeRadiusM;
}

ObservationState::ObservationState()
    : known_global_(kGlobalWidth * kGlobalHeight, false),
      current_local_(kLocalWidth * kLocalHeight) {}

void ObservationState::Observe(const LunarScene& scene, Pose2 pose,
                               SensorModel sensor) {
  if (!FinitePose(pose) || !ValidSensor(sensor)) {
    has_observation_ = false;
    std::fill(current_local_.begin(), current_local_.end(), CachedLocalCell{});
    return;
  }
  current_pose_ = pose;
  current_sensor_ = sensor;
  has_observation_ = true;
  std::fill(current_local_.begin(), current_local_.end(), CachedLocalCell{});
  CacheCurrentContactEnvelope(scene);
  SeedInitialKnownStart(scene, pose);

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
      if (scene.IsOccupied(world_x_m, world_y_m)) {
        MarkKnown(scene, world_x_m, world_y_m);
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
      const std::size_t index =
          static_cast<std::size_t>(y) * scene.global_width() +
          static_cast<std::size_t>(x);
      const bool occupied = scene.GlobalOccupancy()[index] == 100;
      if ((occupied && IsCurrentlyVisible(scene, world_x_m, world_y_m)) ||
          (!occupied && GlobalFreeCellEnvelopeVisible(
                            scene, world_x_m, world_y_m))) {
        known_global_[index] = true;
      }
    }
  }

  constexpr std::size_t kFirstCandidate = 109U;
  constexpr std::size_t kLastCandidate = 210U;
  const double local_origin_x_m = pose.x_m - kLocalHalfLengthM;
  const double local_origin_y_m = pose.y_m - kLocalHalfLengthM;
  for (std::size_t logical_y = kFirstCandidate;
       logical_y <= kLastCandidate; ++logical_y) {
    const double world_y_m =
        local_origin_y_m + (static_cast<double>(logical_y) + 0.5) *
                               kLocalResolutionM;
    for (std::size_t logical_x = kFirstCandidate;
         logical_x <= kLastCandidate; ++logical_x) {
      const double world_x_m =
          local_origin_x_m + (static_cast<double>(logical_x) + 0.5) *
                                 kLocalResolutionM;
      if (IsCurrentlyVisible(scene, world_x_m, world_y_m)) {
        CacheCurrentLocalCell(scene, logical_x, logical_y);
      }
    }
  }
}

bool ObservationState::GlobalFreeCellEnvelopeVisible(
    const LunarScene& scene, const double world_x_m,
    const double world_y_m) const {
  if (!IsCurrentlyVisible(scene, world_x_m, world_y_m)) {
    return false;
  }
  const double delta_x_m = world_x_m - current_pose_.x_m;
  const double delta_y_m = world_y_m - current_pose_.y_m;
  const double forward_m =
      delta_x_m * std::cos(current_pose_.yaw_rad) +
      delta_y_m * std::sin(current_pose_.yaw_rad);
  const double lateral_m =
      -delta_x_m * std::sin(current_pose_.yaw_rad) +
      delta_y_m * std::cos(current_pose_.yaw_rad);
  const double distance_m = std::hypot(delta_x_m, delta_y_m);
  const double halfspace_margin_m =
      kGlobalKnownEnvelopeMarginM * std::numbers::sqrt2;
  if (distance_m + kGlobalKnownEnvelopeMarginM >
          current_sensor_.range_m + kTolerance ||
      forward_m - std::abs(lateral_m) + kTolerance < halfspace_margin_m) {
    return false;
  }

  constexpr double kLosSampleStepM = kLocalResolutionM;
  for (double offset_y_m = -kGlobalKnownEnvelopeMarginM;
       offset_y_m <= kGlobalKnownEnvelopeMarginM + kTolerance;
       offset_y_m += kLosSampleStepM) {
    for (double offset_x_m = -kGlobalKnownEnvelopeMarginM;
         offset_x_m <= kGlobalKnownEnvelopeMarginM + kTolerance;
         offset_x_m += kLosSampleStepM) {
      if (std::hypot(offset_x_m, offset_y_m) >
          kGlobalKnownEnvelopeMarginM + kTolerance) {
        continue;
      }
      if (!IsCurrentlyVisible(scene, world_x_m + offset_x_m,
                              world_y_m + offset_y_m)) {
        return false;
      }
    }
  }
  return true;
}

void ObservationState::SeedInitialKnownStart(const LunarScene& scene,
                                             const Pose2 pose) {
  if (std::abs(pose.x_m) > kContractTolerance ||
      std::abs(pose.y_m) > kContractTolerance ||
      std::ranges::any_of(known_global_, [](const bool known) { return known; })) {
    return;
  }
  for (std::size_t y = 0U; y < scene.global_height(); ++y) {
    const double world_y_m =
        scene.min_x_m() + (static_cast<double>(y) + 0.5) *
                              scene.global_resolution_m();
    for (std::size_t x = 0U; x < scene.global_width(); ++x) {
      const double world_x_m =
          scene.min_x_m() + (static_cast<double>(x) + 0.5) *
                                scene.global_resolution_m();
      const double delta_x_m = world_x_m - pose.x_m;
      const double delta_y_m = world_y_m - pose.y_m;
      const double distance_m = std::hypot(delta_x_m, delta_y_m);
      const double forward_m =
          delta_x_m * std::cos(pose.yaw_rad) +
          delta_y_m * std::sin(pose.yaw_rad);
      const double lateral_m =
          -delta_x_m * std::sin(pose.yaw_rad) +
          delta_y_m * std::cos(pose.yaw_rad);
      const bool inside_contact_prior =
          distance_m <= kInitialGlobalStartStencilRadiusM + kTolerance;
      const bool complete_cell_inside_start_fov =
          distance_m + kGlobalCellHalfDiagonalM <=
              kInitialKnownStartRadiusM + kTolerance &&
          forward_m - std::abs(lateral_m) + kTolerance >=
              kGlobalCellHalfDiagonalM * std::numbers::sqrt2;
      if (inside_contact_prior || complete_cell_inside_start_fov) {
        known_global_[y * scene.global_width() + x] = true;
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
    if (scene.IsOccupied(current_pose_.x_m + ratio * delta_x_m,
                         current_pose_.y_m + ratio * delta_y_m)) {
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

const TruthSample* ObservationState::CurrentLocalSample(
    Pose2 map_pose, std::size_t logical_x, std::size_t logical_y) const noexcept {
  if (!has_observation_ || !SamePose(map_pose, current_pose_) ||
      logical_x >= kLocalWidth || logical_y >= kLocalHeight) {
    return nullptr;
  }
  const CachedLocalCell& cell =
      current_local_[logical_y * kLocalWidth + logical_x];
  return cell.visible ? &cell.sample : nullptr;
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

void ObservationState::CacheCurrentLocalCell(const LunarScene& scene,
                                             std::size_t logical_x,
                                             std::size_t logical_y) {
  const double origin_x_m = current_pose_.x_m - kLocalHalfLengthM;
  const double origin_y_m = current_pose_.y_m - kLocalHalfLengthM;
  if (logical_x >= kLocalWidth || logical_y >= kLocalHeight) {
    return;
  }
  CachedLocalCell& cell =
      current_local_[logical_y * kLocalWidth + logical_x];
  if (cell.visible) {
    return;
  }
  const double center_x_m =
      origin_x_m + (static_cast<double>(logical_x) + 0.5) * kLocalResolutionM;
  const double center_y_m =
      origin_y_m + (static_cast<double>(logical_y) + 0.5) * kLocalResolutionM;
  if (center_x_m < scene.min_x_m() || center_x_m >= scene.max_x_m() ||
      center_y_m < scene.min_x_m() || center_y_m >= scene.max_x_m()) {
    return;
  }
  cell.visible = true;
  cell.sample = scene.Sample(center_x_m, center_y_m);
}

void ObservationState::CacheCurrentContactEnvelope(const LunarScene& scene) {
  const double local_origin_x_m = current_pose_.x_m - kLocalHalfLengthM;
  const double local_origin_y_m = current_pose_.y_m - kLocalHalfLengthM;
  const auto minimum_x = static_cast<long>(std::floor(
      (current_pose_.x_m - kLocalContactEnvelopeRadiusM - local_origin_x_m) /
      kLocalResolutionM));
  const auto maximum_x = static_cast<long>(std::floor(
      (current_pose_.x_m + kLocalContactEnvelopeRadiusM - local_origin_x_m) /
      kLocalResolutionM));
  const auto minimum_y = static_cast<long>(std::floor(
      (current_pose_.y_m - kLocalContactEnvelopeRadiusM - local_origin_y_m) /
      kLocalResolutionM));
  const auto maximum_y = static_cast<long>(std::floor(
      (current_pose_.y_m + kLocalContactEnvelopeRadiusM - local_origin_y_m) /
      kLocalResolutionM));
  for (long y = std::max(0L, minimum_y);
       y <= std::min(static_cast<long>(kLocalHeight) - 1L, maximum_y); ++y) {
    for (long x = std::max(0L, minimum_x);
         x <= std::min(static_cast<long>(kLocalWidth) - 1L, maximum_x); ++x) {
      const double center_x_m =
          local_origin_x_m + (static_cast<double>(x) + 0.5) *
                                 kLocalResolutionM;
      const double center_y_m =
          local_origin_y_m + (static_cast<double>(y) + 0.5) *
                                 kLocalResolutionM;
      if (std::hypot(center_x_m - current_pose_.x_m,
                     center_y_m - current_pose_.y_m) <=
          kLocalContactEnvelopeRadiusM + kTolerance) {
        CacheCurrentLocalCell(scene, static_cast<std::size_t>(x),
                              static_cast<std::size_t>(y));
      }
    }
  }
}

}  // namespace lunar::pure_exploration_sim
