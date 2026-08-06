#include "lunar_planner_ros/snapshot_builder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace lunar::planning::ros {
namespace {

constexpr double kQuaternionTolerance = 1.0e-3;
constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

[[nodiscard]] SnapshotBuildResult Failure(
    const SnapshotErrorCode code,
    std::string reason_code,
    std::string detail = {}) {
  return SnapshotBuildResult{
      .input = std::nullopt,
      .error = SnapshotError{
          .code = code,
          .reason_code = std::move(reason_code),
          .detail = std::move(detail),
      },
  };
}

[[nodiscard]] std::optional<std::int64_t> StampNanoseconds(
    const builtin_interfaces::msg::Time& stamp) noexcept {
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond ||
      (stamp.sec == 0 && stamp.nanosec == 0U)) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
      static_cast<std::int64_t>(stamp.nanosec);
}

[[nodiscard]] builtin_interfaces::msg::Time ToStamp(
    const std::int64_t nanoseconds) noexcept {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(
      nanoseconds / kNanosecondsPerSecond);
  stamp.nanosec = static_cast<std::uint32_t>(
      nanoseconds % kNanosecondsPerSecond);
  return stamp;
}

[[nodiscard]] bool Fresh(
    const std::int64_t now,
    const std::int64_t stamp,
    const std::chrono::nanoseconds maximum_age) noexcept {
  return stamp <= now && now - stamp <= maximum_age.count();
}

[[nodiscard]] bool IsFinite(const double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] bool IsFinite(
    const geometry_msgs::msg::Vector3& value) noexcept {
  return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
}

[[nodiscard]] bool IsFinite(
    const geometry_msgs::msg::Point& value) noexcept {
  return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
}

[[nodiscard]] bool IsFinite(
    const geometry_msgs::msg::Quaternion& value) noexcept {
  return IsFinite(value.w) && IsFinite(value.x) &&
      IsFinite(value.y) && IsFinite(value.z);
}

[[nodiscard]] double Norm(
    const geometry_msgs::msg::Quaternion& value) noexcept {
  return std::sqrt(
      value.w * value.w + value.x * value.x +
      value.y * value.y + value.z * value.z);
}

[[nodiscard]] bool IsUnitQuaternion(
    const geometry_msgs::msg::Quaternion& value) noexcept {
  const double norm = Norm(value);
  return IsFinite(value) && IsFinite(norm) &&
      std::abs(norm - 1.0) <= kQuaternionTolerance;
}

[[nodiscard]] geometry_msgs::msg::Quaternion Normalize(
    geometry_msgs::msg::Quaternion value) noexcept {
  const double norm = Norm(value);
  value.w /= norm;
  value.x /= norm;
  value.y /= norm;
  value.z /= norm;
  return value;
}

[[nodiscard]] geometry_msgs::msg::Quaternion Conjugate(
    const geometry_msgs::msg::Quaternion& value) noexcept {
  return geometry_msgs::msg::Quaternion{}
      .set__x(-value.x)
      .set__y(-value.y)
      .set__z(-value.z)
      .set__w(value.w);
}

[[nodiscard]] geometry_msgs::msg::Quaternion Multiply(
    const geometry_msgs::msg::Quaternion& left,
    const geometry_msgs::msg::Quaternion& right) noexcept {
  geometry_msgs::msg::Quaternion result;
  result.w = left.w * right.w - left.x * right.x -
      left.y * right.y - left.z * right.z;
  result.x = left.w * right.x + left.x * right.w +
      left.y * right.z - left.z * right.y;
  result.y = left.w * right.y - left.x * right.z +
      left.y * right.w + left.z * right.x;
  result.z = left.w * right.z + left.x * right.y -
      left.y * right.x + left.z * right.w;
  return result;
}

[[nodiscard]] lunar::planning::Vec3 Rotate(
    const geometry_msgs::msg::Quaternion& rotation,
    const lunar::planning::Vec3 value) noexcept {
  geometry_msgs::msg::Quaternion vector;
  vector.x = value.x;
  vector.y = value.y;
  vector.z = value.z;
  vector.w = 0.0;
  const auto rotated = Multiply(
      Multiply(rotation, vector), Conjugate(rotation));
  return lunar::planning::Vec3{
      .x = rotated.x,
      .y = rotated.y,
      .z = rotated.z,
  };
}

[[nodiscard]] lunar::planning::Vec3 TransformPoint(
    const geometry_msgs::msg::Transform& map_from_odom,
    const lunar::planning::Vec3 point_in_odom) noexcept {
  const lunar::planning::Vec3 rotated =
      Rotate(map_from_odom.rotation, point_in_odom);
  return lunar::planning::Vec3{
      .x = rotated.x + map_from_odom.translation.x,
      .y = rotated.y + map_from_odom.translation.y,
      .z = rotated.z + map_from_odom.translation.z,
  };
}

[[nodiscard]] bool IsFiniteOdometry(
    const nav_msgs::msg::Odometry& odometry,
    const std::string_view base_frame_id) noexcept {
  if (odometry.header.frame_id != "odom" ||
      odometry.child_frame_id != base_frame_id ||
      !IsFinite(odometry.pose.pose.position) ||
      !IsUnitQuaternion(odometry.pose.pose.orientation) ||
      !IsFinite(odometry.twist.twist.linear) ||
      !IsFinite(odometry.twist.twist.angular)) {
    return false;
  }
  for (std::size_t index = 0U; index < 36U; ++index) {
    if (!IsFinite(odometry.pose.covariance[index]) ||
        !IsFinite(odometry.twist.covariance[index])) {
      return false;
    }
  }
  for (std::size_t axis = 0U; axis < 6U; ++axis) {
    const std::size_t diagonal = axis * 6U + axis;
    if (odometry.pose.covariance[diagonal] < 0.0 ||
        odometry.twist.covariance[diagonal] < 0.0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] double ThreeSigmaPosition(
    const nav_msgs::msg::Odometry& odometry) noexcept {
  const double maximum = std::max({
      odometry.pose.covariance[0],
      odometry.pose.covariance[7],
      odometry.pose.covariance[14],
  });
  return 3.0 * std::sqrt(maximum);
}

[[nodiscard]] double ThreeSigmaLinearVelocity(
    const nav_msgs::msg::Odometry& odometry) noexcept {
  const double maximum = std::max({
      odometry.twist.covariance[0],
      odometry.twist.covariance[7],
      odometry.twist.covariance[14],
  });
  return 3.0 * std::sqrt(maximum);
}

[[nodiscard]] std::uint64_t TransformIdentity(
    const geometry_msgs::msg::TransformStamped& transform) noexcept {
  std::uint64_t value = 14'695'981'039'346'656'037ULL;
  const auto add_byte = [&value](const std::uint8_t byte) {
    value ^= byte;
    value *= 1'099'511'628'211ULL;
  };
  const auto add_u64 = [&add_byte](const std::uint64_t number) {
    for (std::size_t index = 0U; index < sizeof(number); ++index) {
      add_byte(static_cast<std::uint8_t>(
          number >> static_cast<unsigned>(index * 8U)));
    }
  };
  const auto add_string = [&add_byte, &add_u64](const std::string_view text) {
    add_u64(text.size());
    for (const unsigned char character : text) {
      add_byte(character);
    }
  };
  const auto add_double = [&add_u64](double number) {
    if (number == 0.0) {
      number = 0.0;
    }
    add_u64(std::bit_cast<std::uint64_t>(number));
  };
  add_string(transform.header.frame_id);
  add_string(transform.child_frame_id);
  add_double(transform.transform.translation.x);
  add_double(transform.transform.translation.y);
  add_double(transform.transform.translation.z);
  add_double(transform.transform.rotation.w);
  add_double(transform.transform.rotation.x);
  add_double(transform.transform.rotation.y);
  add_double(transform.transform.rotation.z);
  return value == 0U ? 1U : value;
}

[[nodiscard]] bool CovarianceWithinDegradedLimits(
    const nav_msgs::msg::Odometry& odometry,
    const SnapshotPolicy& policy) noexcept {
  for (std::size_t axis = 0U; axis < 6U; ++axis) {
    const std::size_t diagonal = axis * 6U + axis;
    if (odometry.pose.covariance[diagonal] >
            policy.degraded_pose_covariance_limit ||
        odometry.twist.covariance[diagonal] >
            policy.degraded_twist_covariance_limit) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ValidTransform(
    const geometry_msgs::msg::TransformStamped& transform) noexcept {
  return !transform.header.frame_id.empty() &&
      !transform.child_frame_id.empty() &&
      StampNanoseconds(transform.header.stamp).has_value() &&
      IsFinite(transform.transform.translation) &&
      IsUnitQuaternion(transform.transform.rotation);
}

[[nodiscard]] geometry_msgs::msg::TransformStamped Interpolate(
    const geometry_msgs::msg::TransformStamped& lower,
    const geometry_msgs::msg::TransformStamped& upper,
    const std::int64_t target_stamp) {
  const std::int64_t lower_stamp = *StampNanoseconds(lower.header.stamp);
  const std::int64_t upper_stamp = *StampNanoseconds(upper.header.stamp);
  const double ratio = static_cast<double>(target_stamp - lower_stamp) /
      static_cast<double>(upper_stamp - lower_stamp);
  geometry_msgs::msg::TransformStamped result;
  result.header.frame_id = lower.header.frame_id;
  result.child_frame_id = lower.child_frame_id;
  result.header.stamp = ToStamp(target_stamp);
  result.transform.translation.x = lower.transform.translation.x +
      ratio * (upper.transform.translation.x - lower.transform.translation.x);
  result.transform.translation.y = lower.transform.translation.y +
      ratio * (upper.transform.translation.y - lower.transform.translation.y);
  result.transform.translation.z = lower.transform.translation.z +
      ratio * (upper.transform.translation.z - lower.transform.translation.z);

  auto lower_rotation = Normalize(lower.transform.rotation);
  auto upper_rotation = Normalize(upper.transform.rotation);
  const double dot = lower_rotation.w * upper_rotation.w +
      lower_rotation.x * upper_rotation.x +
      lower_rotation.y * upper_rotation.y +
      lower_rotation.z * upper_rotation.z;
  if (dot < 0.0) {
    upper_rotation.w = -upper_rotation.w;
    upper_rotation.x = -upper_rotation.x;
    upper_rotation.y = -upper_rotation.y;
    upper_rotation.z = -upper_rotation.z;
  }
  result.transform.rotation.w =
      lower_rotation.w + ratio * (upper_rotation.w - lower_rotation.w);
  result.transform.rotation.x =
      lower_rotation.x + ratio * (upper_rotation.x - lower_rotation.x);
  result.transform.rotation.y =
      lower_rotation.y + ratio * (upper_rotation.y - lower_rotation.y);
  result.transform.rotation.z =
      lower_rotation.z + ratio * (upper_rotation.z - lower_rotation.z);
  result.transform.rotation = Normalize(result.transform.rotation);
  return result;
}

[[nodiscard]] std::optional<geometry_msgs::msg::TransformStamped>
SelectTransform(
    const std::vector<geometry_msgs::msg::TransformStamped>& transforms,
    const std::string_view parent,
    const std::string_view child,
    const std::int64_t state_stamp,
    const std::int64_t now,
    const SnapshotPolicy& policy) {
  std::vector<geometry_msgs::msg::TransformStamped> candidates;
  for (const auto& transform : transforms) {
    if (transform.header.frame_id == parent &&
        transform.child_frame_id == child) {
      if (!ValidTransform(transform)) {
        return std::nullopt;
      }
      candidates.push_back(transform);
    }
  }
  std::ranges::sort(candidates, [](const auto& left, const auto& right) {
    return *StampNanoseconds(left.header.stamp) <
        *StampNanoseconds(right.header.stamp);
  });
  for (auto iterator = candidates.rbegin(); iterator != candidates.rend();
       ++iterator) {
    const std::int64_t stamp = *StampNanoseconds(iterator->header.stamp);
    if (stamp == state_stamp &&
        Fresh(now, stamp, policy.tf_max_age)) {
      auto selected = *iterator;
      selected.transform.rotation = Normalize(selected.transform.rotation);
      return selected;
    }
  }

  const geometry_msgs::msg::TransformStamped* lower = nullptr;
  const geometry_msgs::msg::TransformStamped* upper = nullptr;
  for (const auto& candidate : candidates) {
    const std::int64_t stamp = *StampNanoseconds(candidate.header.stamp);
    if (stamp < state_stamp) {
      lower = &candidate;
    } else if (stamp > state_stamp) {
      upper = &candidate;
      break;
    }
  }
  if (lower == nullptr || upper == nullptr) {
    return std::nullopt;
  }
  const std::int64_t lower_stamp = *StampNanoseconds(lower->header.stamp);
  const std::int64_t upper_stamp = *StampNanoseconds(upper->header.stamp);
  if (state_stamp - lower_stamp > policy.max_pairwise_skew.count() ||
      upper_stamp - state_stamp > policy.max_pairwise_skew.count() ||
      !Fresh(now, lower_stamp, policy.tf_max_age) ||
      !Fresh(now, upper_stamp, policy.tf_max_age)) {
    return std::nullopt;
  }
  return Interpolate(*lower, *upper, state_stamp);
}

[[nodiscard]] bool FiniteCoreVector(
    const lunar::planning::Vec3& value) noexcept {
  return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
}

[[nodiscard]] std::optional<lunar::planning::GoalRegion> ConvertGoal(
    const GoalRequest& request,
    const geometry_msgs::msg::Transform& map_from_odom) {
  if (request.request_id.empty() || request.goal.goal_id.empty() ||
      request.stamp.nanoseconds() <= 0 ||
      (request.frame_id != "map" && request.frame_id != "odom") ||
      !IsFinite(request.goal.yaw_tolerance_rad) ||
      request.goal.yaw_tolerance_rad < 0.0 ||
      (request.goal.yaw_rad.has_value() &&
       !IsFinite(*request.goal.yaw_rad))) {
    return std::nullopt;
  }

  lunar::planning::GoalRegion goal = request.goal;
  const bool transform = request.frame_id == "odom";
  bool target_valid = std::visit(
      [&](auto& target) {
        using Target = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Target, lunar::planning::PointGoal>) {
          if (!FiniteCoreVector(target.position_m) ||
              !IsFinite(target.tolerance_m) || target.tolerance_m < 0.0) {
            return false;
          }
          if (transform) {
            target.position_m = TransformPoint(
                map_from_odom, target.position_m);
          }
          return FiniteCoreVector(target.position_m);
        } else {
          if (target.boundary_m.size() < 3U ||
              !IsFinite(target.normal_tolerance_m) ||
              target.normal_tolerance_m < 0.0) {
            return false;
          }
          for (auto& point : target.boundary_m) {
            if (!FiniteCoreVector(point)) {
              return false;
            }
            if (transform) {
              point = TransformPoint(map_from_odom, point);
            }
            if (!FiniteCoreVector(point)) {
              return false;
            }
          }
          return true;
        }
      },
      goal.target);
  if (!target_valid) {
    return std::nullopt;
  }

  if (transform && goal.yaw_rad.has_value()) {
    const lunar::planning::Vec3 heading_in_odom{
        .x = std::cos(*goal.yaw_rad),
        .y = std::sin(*goal.yaw_rad),
        .z = 0.0,
    };
    const auto heading_in_map = Rotate(
        map_from_odom.rotation, heading_in_odom);
    if (!FiniteCoreVector(heading_in_map) ||
        std::hypot(heading_in_map.x, heading_in_map.y) <= 1.0e-9) {
      return std::nullopt;
    }
    goal.yaw_rad = std::atan2(heading_in_map.y, heading_in_map.x);
  }
  return goal;
}

[[nodiscard]] lunar::planning::Pose3 ToCorePose(
    const geometry_msgs::msg::Pose& pose) noexcept {
  return lunar::planning::Pose3{
      .position_m = {pose.position.x, pose.position.y, pose.position.z},
      .orientation = {
          .w = pose.orientation.w,
          .x = pose.orientation.x,
          .y = pose.orientation.y,
          .z = pose.orientation.z,
      },
  };
}

[[nodiscard]] lunar::planning::Twist3 ToCoreTwist(
    const geometry_msgs::msg::Twist& twist) noexcept {
  return lunar::planning::Twist3{
      .linear_mps = {twist.linear.x, twist.linear.y, twist.linear.z},
      .angular_radps = {twist.angular.x, twist.angular.y, twist.angular.z},
  };
}

[[nodiscard]] lunar::planning::PlatformState ToPlatformState(
    const nav_msgs::msg::Odometry& odometry,
    const lunar::planning::PlatformType platform) {
  const auto pose = ToCorePose(odometry.pose.pose);
  const auto twist = ToCoreTwist(odometry.twist.twist);
  if (platform == lunar::planning::PlatformType::kWheeled) {
    return lunar::planning::WheeledState{.pose = pose, .velocity = twist};
  }
  if (platform == lunar::planning::PlatformType::kLegged) {
    return lunar::planning::LeggedState{
        .body_pose = pose,
        .body_velocity = twist,
    };
  }
  return lunar::planning::HopperState{.pose = pose, .velocity = twist};
}

}  // namespace

bool ValidateSnapshotPolicy(const SnapshotPolicy& policy) noexcept {
  return policy.global_map_max_age.count() > 0 &&
      policy.local_map_max_age.count() > 0 &&
      policy.odometry_max_age.count() > 0 &&
      policy.localization_status_max_age.count() > 0 &&
      policy.propellant_state_max_age.count() > 0 &&
      policy.tf_max_age.count() > 0 &&
      policy.max_pairwise_skew.count() > 0 &&
      std::isfinite(policy.degraded_pose_covariance_limit) &&
      policy.degraded_pose_covariance_limit >= 0.0 &&
      std::isfinite(policy.degraded_twist_covariance_limit) &&
      policy.degraded_twist_covariance_limit >= 0.0;
}

SnapshotBuilder::SnapshotBuilder(
    std::shared_ptr<SnapshotStore> store,
    SnapshotPolicy policy,
    lunar::planning::PlatformCapability capability,
    lunar::planning::PlannerConfig planner_config,
    std::string base_frame_id)
    : store_(std::move(store)),
      policy_(policy),
      capability_(std::move(capability)),
      planner_config_(std::move(planner_config)),
      base_frame_id_(std::move(base_frame_id)) {}

SnapshotBuildResult SnapshotBuilder::Freeze(
    const GoalRequest& request,
    const rclcpp::Time now) const {
  if (store_ == nullptr || !ValidateSnapshotPolicy(policy_) ||
      now.nanoseconds() <= 0 || base_frame_id_.empty() ||
      request.mission_id.empty() || request.mission_revision == 0U ||
      request.platform_id.empty() || request.capability_version.empty()) {
    return Failure(
        SnapshotErrorCode::kConfigurationInvalid,
        "SNAPSHOT_CONFIGURATION_INVALID");
  }
  const SnapshotStoreView view = store_->Capture();
  const bool hopper =
      lunar::planning::CapabilityPlatform(capability_) ==
      lunar::planning::PlatformType::kHopper;
  if (!view.global_map.has_value()) {
    return Failure(SnapshotErrorCode::kMissingGlobalMap, "GLOBAL_MAP_MISSING");
  }
  if (!view.local_map.has_value()) {
    return Failure(SnapshotErrorCode::kMissingLocalMap, "LOCAL_MAP_MISSING");
  }
  if (!view.odometry.has_value()) {
    return Failure(SnapshotErrorCode::kMissingOdometry, "ODOMETRY_MISSING");
  }
  if (!view.localization_status.has_value()) {
    return Failure(
        SnapshotErrorCode::kMissingLocalizationStatus,
        "LOCALIZATION_STATUS_MISSING");
  }

  const GridMapAdaptResult global =
      grid_map_adapter_.Adapt(*view.global_map, "map");
  if (!global.ok()) {
    return Failure(
        SnapshotErrorCode::kInvalidGlobalMap,
        "GLOBAL_MAP_INVALID",
        global.error.has_value() ? global.error->reason_code : std::string{});
  }
  const GridMapAdaptResult local =
      grid_map_adapter_.Adapt(*view.local_map, "odom");
  if (!local.ok()) {
    return Failure(
        SnapshotErrorCode::kInvalidLocalMap,
        "LOCAL_MAP_INVALID",
        local.error.has_value() ? local.error->reason_code : std::string{});
  }

  lunar::planning::WorldSnapshot world{
      .global_map = std::move(*global.map),
      .local_map = std::move(*local.map),
      .map_from_odom = {},
  };
  const auto map_levels = lunar::planning::hierarchical::ValidateMapLevels(
      world, planner_config_.global_map);
  if (!map_levels.ok()) {
    if (map_levels.reason_code == "GLOBAL_MAP_CONFIGURATION_INVALID") {
      return Failure(
          SnapshotErrorCode::kConfigurationInvalid,
          "SNAPSHOT_CONFIGURATION_INVALID",
          map_levels.reason_code);
    }
    const bool local_level_invalid =
        map_levels.reason_code == "LOCAL_MAP_LEVEL_INVALID";
    return Failure(
        local_level_invalid ? SnapshotErrorCode::kInvalidLocalMap :
                              SnapshotErrorCode::kInvalidGlobalMap,
        map_levels.reason_code);
  }

  const auto odometry_stamp = StampNanoseconds(view.odometry->header.stamp);
  const auto localization_stamp =
      StampNanoseconds(view.localization_status->header.stamp);
  if (!odometry_stamp.has_value() ||
      !IsFiniteOdometry(*view.odometry, base_frame_id_)) {
    return Failure(
        SnapshotErrorCode::kInvalidOdometry, "ODOMETRY_INVALID");
  }
  if (!localization_stamp.has_value() ||
      view.localization_status->header.frame_id != "odom") {
    return Failure(
        SnapshotErrorCode::kInvalidLocalization,
        "LOCALIZATION_STATUS_INVALID");
  }

  const std::int64_t now_stamp = now.nanoseconds();
  std::optional<lunar::planning::HopperPropellantState> hopper_propellant;
  std::optional<std::int64_t> propellant_stamp;
  if (hopper) {
    if (!view.hopper_propellant_state.has_value()) {
      return Failure(
          SnapshotErrorCode::kInvalidHopperPropellant,
          "HOPPER_PROPELLANT_STATE_INVALID", "state is missing");
    }
    const auto& propellant = *view.hopper_propellant_state;
    propellant_stamp = StampNanoseconds(propellant.header.stamp);
    if (!propellant_stamp.has_value() || *propellant_stamp > now_stamp ||
        propellant.header.frame_id != base_frame_id_ ||
        propellant.platform_id != request.platform_id ||
        propellant.capability_version != request.capability_version ||
        !std::isfinite(propellant.total_mass_kg) ||
        !std::isfinite(propellant.remaining_usable_fuel_mass_kg) ||
        propellant.total_mass_kg <= 0.0 ||
        propellant.remaining_usable_fuel_mass_kg <= 0.0 ||
        propellant.remaining_usable_fuel_mass_kg >=
            propellant.total_mass_kg) {
      return Failure(
          SnapshotErrorCode::kInvalidHopperPropellant,
          "HOPPER_PROPELLANT_STATE_INVALID");
    }
    if (!Fresh(
            now_stamp, *propellant_stamp,
            policy_.propellant_state_max_age)) {
      return Failure(
          SnapshotErrorCode::kStaleHopperPropellant,
          "HOPPER_PROPELLANT_STATE_STALE");
    }
    hopper_propellant = lunar::planning::HopperPropellantState{
        .stamp = lunar::planning::TimePoint{
            .nanoseconds_since_epoch = *propellant_stamp,
        },
        .platform_id = propellant.platform_id,
        .capability_version = propellant.capability_version,
        .total_mass_kg = propellant.total_mass_kg,
        .remaining_usable_fuel_mass_kg =
            propellant.remaining_usable_fuel_mass_kg,
    };
  }
  if (!Fresh(
          now_stamp, world.global_map.stamp.nanoseconds_since_epoch,
          policy_.global_map_max_age)) {
    return Failure(
        SnapshotErrorCode::kStaleGlobalMap, "GLOBAL_MAP_STALE");
  }
  if (!Fresh(
          now_stamp, world.local_map.stamp.nanoseconds_since_epoch,
          policy_.local_map_max_age)) {
    return Failure(SnapshotErrorCode::kStaleLocalMap, "LOCAL_MAP_STALE");
  }
  if (!Fresh(now_stamp, *odometry_stamp, policy_.odometry_max_age)) {
    return Failure(SnapshotErrorCode::kStaleOdometry, "ODOMETRY_STALE");
  }
  if (!Fresh(
          now_stamp, *localization_stamp,
          policy_.localization_status_max_age)) {
    return Failure(
        SnapshotErrorCode::kStaleLocalizationStatus,
        "LOCALIZATION_STATUS_STALE");
  }

  const std::array<std::int64_t, 4> input_stamps{
      world.global_map.stamp.nanoseconds_since_epoch,
      world.local_map.stamp.nanoseconds_since_epoch,
      *odometry_stamp,
      *localization_stamp,
  };
  const auto [minimum_stamp, maximum_stamp] =
      std::ranges::minmax_element(input_stamps);
  if (*maximum_stamp - *minimum_stamp > policy_.max_pairwise_skew.count()) {
    return Failure(SnapshotErrorCode::kInputSkew, "INPUT_TIME_SKEW");
  }
  if (propellant_stamp.has_value() &&
      std::ranges::any_of(input_stamps, [&](const std::int64_t stamp) {
        return std::abs(stamp - *propellant_stamp) >
            policy_.max_pairwise_skew.count();
      })) {
    return Failure(
        SnapshotErrorCode::kStaleHopperPropellant,
        "HOPPER_PROPELLANT_STATE_STALE", "input time skew");
  }

  const std::uint8_t localization_state = view.localization_status->status;
  if (localization_state !=
          lunar_navigation_msgs::msg::LocalizationStatus::VALID &&
      localization_state !=
          lunar_navigation_msgs::msg::LocalizationStatus::DEGRADED) {
    return Failure(
        SnapshotErrorCode::kInvalidLocalization,
        "LOCALIZATION_NOT_PLANNABLE");
  }
  if (localization_state ==
          lunar_navigation_msgs::msg::LocalizationStatus::DEGRADED &&
      !CovarianceWithinDegradedLimits(*view.odometry, policy_)) {
    return Failure(
        SnapshotErrorCode::kCovarianceLimit,
        "DEGRADED_LOCALIZATION_COVARIANCE_LIMIT");
  }

  const auto map_from_odom = SelectTransform(
      view.transforms, "map", "odom", *odometry_stamp,
      now_stamp, policy_);
  const auto odom_from_base = SelectTransform(
      view.transforms, "odom", base_frame_id_, *odometry_stamp,
      now_stamp, policy_);
  if (!map_from_odom.has_value() || !odom_from_base.has_value()) {
    return Failure(SnapshotErrorCode::kStaleTf, "STALE_TF");
  }
  if (std::abs(
          world.local_map.stamp.nanoseconds_since_epoch - *odometry_stamp) >
      policy_.max_pairwise_skew.count()) {
    return Failure(SnapshotErrorCode::kStaleTf, "STALE_TF");
  }

  const auto goal = ConvertGoal(request, map_from_odom->transform);
  if (!goal.has_value()) {
    return Failure(SnapshotErrorCode::kInvalidGoal, "GOAL_INVALID");
  }
  if (hopper) {
    const auto* point = std::get_if<lunar::planning::PointGoal>(&goal->target);
    if (point == nullptr || point->tolerance_m != 0.0 ||
        goal->yaw_rad.has_value()) {
      return Failure(
          SnapshotErrorCode::kInvalidGoal, "HOPPER_GOAL_INVALID");
    }
  }

  lunar::planning::PlannerConfig planner_config = planner_config_;
  planner_config.maximum_input_skew = policy_.max_pairwise_skew;
  world.map_from_odom = lunar::planning::RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = lunar::planning::TimePoint{
          .nanoseconds_since_epoch = *odometry_stamp,
      },
      .translation_m = {
          map_from_odom->transform.translation.x,
          map_from_odom->transform.translation.y,
          map_from_odom->transform.translation.z,
      },
      .rotation = {
          .w = map_from_odom->transform.rotation.w,
          .x = map_from_odom->transform.rotation.x,
          .y = map_from_odom->transform.rotation.y,
          .z = map_from_odom->transform.rotation.z,
      },
  };
  const SnapshotContentGenerations generations =
      store_->ResolveContentGenerations(
          global.content_identity,
          local.content_identity,
          TransformIdentity(*map_from_odom));
  lunar::planning::PlannerInput input{
      .request_id = request.request_id,
      .mission_id = request.mission_id,
      .mission_revision = request.mission_revision,
      .platform_id = request.platform_id,
      .capability_version = request.capability_version,
      .global_map_generation = generations.global_map,
      .local_map_generation = generations.local_map,
      .map_from_odom_generation = generations.map_from_odom,
      .state_time = lunar::planning::TimePoint{
          .nanoseconds_since_epoch = *odometry_stamp,
      },
      .current_state = ToPlatformState(
          *view.odometry,
          lunar::planning::CapabilityPlatform(capability_)),
      .hopper_propellant = std::move(hopper_propellant),
      .goal_map = *goal,
      .world = std::move(world),
      .capability = capability_,
      .config = std::move(planner_config),
      .previous_execution = request.previous_execution,
      .stop_token = request.stop_token,
      .position_uncertainty_m = ThreeSigmaPosition(*view.odometry),
      .velocity_uncertainty_mps = ThreeSigmaLinearVelocity(*view.odometry),
      .continuation = request.continuation,
  };
  return SnapshotBuildResult{.input = std::move(input), .error = std::nullopt};
}

}  // namespace lunar::planning::ros
