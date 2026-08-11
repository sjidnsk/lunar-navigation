#include "lunar_unreal_tcp_bridge/ros_conversions.hpp"

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
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_navigation_msgs/msg/motion_execution_feedback.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/multi_array_dimension.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_unreal_tcp_bridge/metadata_validation.hpp"

namespace lunar::unreal_tcp {
namespace {

constexpr std::size_t kMapWidth = 320U;
constexpr std::size_t kMapHeight = 320U;
constexpr std::size_t kMapCellCount = kMapWidth * kMapHeight;
constexpr std::size_t kElevationBytes = kMapCellCount * sizeof(float);
constexpr std::size_t kValidityBytes = kMapCellCount / 8U;
constexpr std::size_t kMapPayloadBytes = kElevationBytes + kValidityBytes;
constexpr std::size_t kTrajectoryRecordBytes =
    sizeof(std::int64_t) + 13U * sizeof(double);
constexpr double kGeometryTolerance = 1.0e-8;

template<typename Value>
[[nodiscard]] ConversionResult<Value> Failure(std::string reason_code) {
  return ConversionResult<Value>{
      .value = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

template<typename Value>
[[nodiscard]] ConversionResult<Value> Success(Value value) {
  return ConversionResult<Value>{
      .value = std::move(value),
      .reason_code = {},
  };
}

[[nodiscard]] std::optional<builtin_interfaces::msg::Time> Stamp(
    const std::int64_t nanoseconds) {
  if (nanoseconds <= 0) {
    return std::nullopt;
  }
  const std::int64_t seconds = nanoseconds / 1'000'000'000LL;
  if (seconds > std::numeric_limits<std::int32_t>::max()) {
    return std::nullopt;
  }
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(seconds);
  stamp.nanosec = static_cast<std::uint32_t>(
      nanoseconds % 1'000'000'000LL);
  return stamp;
}

[[nodiscard]] std::optional<std::int64_t> Nanoseconds(
    const builtin_interfaces::msg::Time& stamp) {
  if (stamp.sec < 0 || stamp.nanosec >= 1'000'000'000U) {
    return std::nullopt;
  }
  const std::int64_t value =
      static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
      static_cast<std::int64_t>(stamp.nanosec);
  if (value <= 0) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] std::optional<std::int64_t> Nanoseconds(
    const builtin_interfaces::msg::Duration& duration) {
  if (duration.sec < 0 || duration.nanosec >= 1'000'000'000U) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(duration.sec) * 1'000'000'000LL +
      static_cast<std::int64_t>(duration.nanosec);
}

[[nodiscard]] Eigen::Vector3d Vector3(const nlohmann::json& value) {
  return {
      value.at(0).get<double>(),
      value.at(1).get<double>(),
      value.at(2).get<double>(),
  };
}

[[nodiscard]] Eigen::Quaterniond Quaternion(const nlohmann::json& value) {
  return Eigen::Quaterniond{
      value.at(3).get<double>(),
      value.at(0).get<double>(),
      value.at(1).get<double>(),
      value.at(2).get<double>(),
  };
}

[[nodiscard]] bool Finite(const Eigen::Vector3d& value) {
  return value.array().isFinite().all();
}

[[nodiscard]] bool Finite(const Eigen::Quaterniond& value) {
  return value.coeffs().array().isFinite().all() &&
      value.squaredNorm() > kGeometryTolerance;
}

[[nodiscard]] geometry_msgs::msg::Pose RosPose(
    const Eigen::Isometry3d& pose) {
  geometry_msgs::msg::Pose result;
  result.position.x = pose.translation().x();
  result.position.y = pose.translation().y();
  result.position.z = pose.translation().z();
  Eigen::Quaterniond orientation{pose.rotation()};
  orientation.normalize();
  result.orientation.x = orientation.x();
  result.orientation.y = orientation.y();
  result.orientation.z = orientation.z();
  result.orientation.w = orientation.w();
  return result;
}

[[nodiscard]] geometry_msgs::msg::Transform RosTransform(
    const Eigen::Isometry3d& transform) {
  geometry_msgs::msg::Transform result;
  result.translation.x = transform.translation().x();
  result.translation.y = transform.translation().y();
  result.translation.z = transform.translation().z();
  Eigen::Quaterniond orientation{transform.rotation()};
  orientation.normalize();
  result.rotation.x = orientation.x();
  result.rotation.y = orientation.y();
  result.rotation.z = orientation.z();
  result.rotation.w = orientation.w();
  return result;
}

[[nodiscard]] Eigen::Isometry3d EigenPose(
    const geometry_msgs::msg::Transform& transform) {
  const Eigen::Vector3d translation{
      transform.translation.x,
      transform.translation.y,
      transform.translation.z,
  };
  const Eigen::Quaterniond orientation{
      transform.rotation.w,
      transform.rotation.x,
      transform.rotation.y,
      transform.rotation.z,
  };
  if (!Finite(translation) || !Finite(orientation)) {
    throw std::invalid_argument("REFERENCE_POSE_NONFINITE");
  }
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() = translation;
  result.linear() = orientation.normalized().toRotationMatrix();
  return result;
}

[[nodiscard]] bool ValidCovariance(
    const std::array<double, 36>& covariance) {
  for (const double value : covariance) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < 6U; ++index) {
    if (covariance[index * 6U + index] <= 0.0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::array<double, 36> JsonCovariance(
    const nlohmann::json& value) {
  std::array<double, 36> result{};
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[index] = value.at(index).get<double>();
  }
  return result;
}

[[nodiscard]] std::uint32_t ReadUint32(
    const std::vector<std::byte>& bytes, const std::size_t offset) {
  std::uint32_t value{};
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    value |= static_cast<std::uint32_t>(
        std::to_integer<std::uint8_t>(bytes[offset + index])) <<
        (index * 8U);
  }
  return value;
}

[[nodiscard]] float ReadFloat32(
    const std::vector<std::byte>& bytes, const std::size_t offset) {
  return std::bit_cast<float>(ReadUint32(bytes, offset));
}

[[nodiscard]] bool ValidBit(
    const std::vector<std::byte>& payload, const std::size_t index) {
  const std::uint8_t value = std::to_integer<std::uint8_t>(
      payload[kElevationBytes + index / 8U]);
  return (value & static_cast<std::uint8_t>(1U << (index % 8U))) != 0U;
}

[[nodiscard]] std_msgs::msg::Float32MultiArray EncodeGridMapLayer(
    const std::vector<float>& row_major) {
  std_msgs::msg::Float32MultiArray result;
  result.layout.dim.resize(2U);
  result.layout.dim[0].label = "column_index";
  result.layout.dim[0].size = kMapHeight;
  result.layout.dim[0].stride = kMapCellCount;
  result.layout.dim[1].label = "row_index";
  result.layout.dim[1].size = kMapWidth;
  result.layout.dim[1].stride = kMapWidth;
  result.data.assign(kMapCellCount, 0.0F);
  for (std::size_t y = 0U; y < kMapHeight; ++y) {
    for (std::size_t x = 0U; x < kMapWidth; ++x) {
      const std::size_t physical_row = kMapWidth - 1U - x;
      const std::size_t physical_column = kMapHeight - 1U - y;
      result.data[physical_column * kMapWidth + physical_row] =
          row_major[y * kMapWidth + x];
    }
  }
  return result;
}

template<typename Integer>
void AppendLittleInteger(
    std::vector<std::byte>& output, const Integer value) {
  using Unsigned = std::make_unsigned_t<Integer>;
  const Unsigned bits = static_cast<Unsigned>(value);
  for (std::size_t index = 0U; index < sizeof(Integer); ++index) {
    output.push_back(static_cast<std::byte>(
        (bits >> (index * 8U)) & static_cast<Unsigned>(0xffU)));
  }
}

void AppendDouble(std::vector<std::byte>& output, const double value) {
  AppendLittleInteger(
      output, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] bool FiniteTwist(const geometry_msgs::msg::Twist& twist) {
  return std::isfinite(twist.linear.x) &&
      std::isfinite(twist.linear.y) &&
      std::isfinite(twist.linear.z) &&
      std::isfinite(twist.angular.x) &&
      std::isfinite(twist.angular.y) &&
      std::isfinite(twist.angular.z);
}

[[nodiscard]] std::uint8_t FeedbackState(const std::string_view state) {
  using Feedback = lunar_navigation_msgs::msg::MotionExecutionFeedback;
  if (state == "ACCEPTED") {
    return Feedback::ACCEPTED;
  }
  if (state == "EXECUTING") {
    return Feedback::EXECUTING;
  }
  if (state == "SEGMENT_COMPLETE") {
    return Feedback::SEGMENT_COMPLETE;
  }
  if (state == "FAILED") {
    return Feedback::FAILED;
  }
  return Feedback::CANCELED;
}

}  // namespace

ConversionResult<RobotStateMessages> ConvertRobotState(
    const Frame& frame,
    const CoordinateTransform& transform,
    const SimulationCovarianceProfile& covariance_profile) {
  if (frame.header.message_type != MessageType::kRobotState ||
      !frame.payload.empty()) {
    return Failure<RobotStateMessages>("ROBOT_STATE_FRAME_INVALID");
  }
  if (const auto error = ValidateMetadata(
          MessageType::kRobotState, frame.metadata)) {
    return Failure<RobotStateMessages>(error->reason_code);
  }
  const auto stamp = Stamp(frame.header.simulation_time_ns);
  if (!stamp.has_value()) {
    return Failure<RobotStateMessages>("ROBOT_STATE_STAMP_INVALID");
  }

  std::array<double, 36> pose_covariance = covariance_profile.pose;
  std::array<double, 36> twist_covariance = covariance_profile.twist;
  if (frame.metadata.contains("pose_covariance")) {
    pose_covariance = JsonCovariance(frame.metadata.at("pose_covariance"));
    twist_covariance = JsonCovariance(frame.metadata.at("twist_covariance"));
  }
  if (!ValidCovariance(pose_covariance) ||
      !ValidCovariance(twist_covariance)) {
    return Failure<RobotStateMessages>("ROBOT_STATE_COVARIANCE_INVALID");
  }

  try {
    const Eigen::Vector3d root_position =
        Vector3(frame.metadata.at("position_cm"));
    const Eigen::Quaterniond root_orientation =
        Quaternion(frame.metadata.at("quaternion_xyzw"));
    const Eigen::Isometry3d map_from_root =
        transform.MapFromUnrealRoot(root_position, root_orientation);
    const Eigen::Isometry3d map_from_base_link =
        transform.MapFromBaseLink(root_position, root_orientation);
    const Eigen::Isometry3d map_from_base_footprint =
        transform.MapFromBaseFootprint(root_position, root_orientation);
    const Eigen::Vector3d root_linear_map =
        transform.UnrealLinearVelocityToMap(
            Vector3(frame.metadata.at("linear_velocity_cmps")));
    const Eigen::Vector3d angular_map =
        transform.UnrealAngularVelocityToMap(
            Vector3(frame.metadata.at("angular_velocity_radps")));

    const auto point_velocity = [&root_linear_map, &angular_map,
                                 &map_from_root](
        const Eigen::Isometry3d& pose) {
      return root_linear_map + angular_map.cross(
          pose.translation() - map_from_root.translation());
    };
    const Eigen::Vector3d base_link_linear =
        map_from_base_link.rotation().transpose() *
        point_velocity(map_from_base_link);
    const Eigen::Vector3d base_link_angular =
        map_from_base_link.rotation().transpose() * angular_map;
    const Eigen::Vector3d footprint_linear =
        map_from_base_footprint.rotation().transpose() *
        point_velocity(map_from_base_footprint);
    const Eigen::Vector3d footprint_angular =
        map_from_base_footprint.rotation().transpose() * angular_map;

    RobotStateMessages output;
    output.external_odometry.header.stamp = *stamp;
    output.external_odometry.header.frame_id = "odom";
    output.external_odometry.child_frame_id = "base_link";
    output.external_odometry.pose.pose = RosPose(map_from_base_link);
    output.external_odometry.pose.covariance = pose_covariance;
    output.external_odometry.twist.twist.linear.x = base_link_linear.x();
    output.external_odometry.twist.twist.linear.y = base_link_linear.y();
    output.external_odometry.twist.twist.linear.z = base_link_linear.z();
    output.external_odometry.twist.twist.angular.x = base_link_angular.x();
    output.external_odometry.twist.twist.angular.y = base_link_angular.y();
    output.external_odometry.twist.twist.angular.z = base_link_angular.z();
    output.external_odometry.twist.covariance = twist_covariance;

    output.wheeled_odometry = output.external_odometry;
    output.wheeled_odometry.child_frame_id = "base_footprint";
    output.wheeled_odometry.pose.pose = RosPose(map_from_base_footprint);
    output.wheeled_odometry.twist.twist.linear.x = footprint_linear.x();
    output.wheeled_odometry.twist.twist.linear.y = footprint_linear.y();
    output.wheeled_odometry.twist.twist.linear.z = footprint_linear.z();
    output.wheeled_odometry.twist.twist.angular.x = footprint_angular.x();
    output.wheeled_odometry.twist.twist.angular.y = footprint_angular.y();
    output.wheeled_odometry.twist.twist.angular.z = footprint_angular.z();

    geometry_msgs::msg::TransformStamped map_from_odom;
    map_from_odom.header.stamp = *stamp;
    map_from_odom.header.frame_id = "map";
    map_from_odom.child_frame_id = "odom";
    map_from_odom.transform.rotation.w = 1.0;
    geometry_msgs::msg::TransformStamped odom_from_base_link;
    odom_from_base_link.header.stamp = *stamp;
    odom_from_base_link.header.frame_id = "odom";
    odom_from_base_link.child_frame_id = "base_link";
    odom_from_base_link.transform = RosTransform(map_from_base_link);
    output.dynamic_tf.transforms = {
        std::move(map_from_odom), std::move(odom_from_base_link)};

    geometry_msgs::msg::TransformStamped base_link_from_footprint;
    base_link_from_footprint.header.stamp = *stamp;
    base_link_from_footprint.header.frame_id = "base_link";
    base_link_from_footprint.child_frame_id = "base_footprint";
    base_link_from_footprint.transform = RosTransform(
        transform.base_footprint_from_base_link().inverse());
    output.static_tf.transforms = {std::move(base_link_from_footprint)};
    return Success(std::move(output));
  } catch (const std::exception&) {
    return Failure<RobotStateMessages>("ROBOT_STATE_TRANSFORM_INVALID");
  }
}

ConversionResult<grid_map_msgs::msg::GridMap> ConvertObservedElevation(
    const Frame& frame, const CoordinateTransform& transform) {
  if (frame.header.message_type != MessageType::kLocalElevationMap) {
    return Failure<grid_map_msgs::msg::GridMap>("LOCAL_MAP_FRAME_INVALID");
  }
  if (const auto error = ValidateMetadata(
          MessageType::kLocalElevationMap, frame.metadata)) {
    return Failure<grid_map_msgs::msg::GridMap>(error->reason_code);
  }
  if (frame.payload.size() != kMapPayloadBytes) {
    return Failure<grid_map_msgs::msg::GridMap>(
        "LOCAL_MAP_PAYLOAD_SIZE_INVALID");
  }
  const auto stamp = Stamp(frame.header.simulation_time_ns);
  if (!stamp.has_value()) {
    return Failure<grid_map_msgs::msg::GridMap>("LOCAL_MAP_STAMP_INVALID");
  }

  try {
    const Eigen::Vector3d cell_zero = transform.UnrealPositionToMap(
        Vector3(frame.metadata.at("cell_zero_center_world_cm")));
    Eigen::Vector3d u = transform.UnrealDirectionToMap(
        Vector3(frame.metadata.at("u_axis_world")));
    Eigen::Vector3d v = transform.UnrealDirectionToMap(
        Vector3(frame.metadata.at("v_axis_world")));
    if (std::abs(u.norm() - 1.0) > kGeometryTolerance ||
        std::abs(v.norm() - 1.0) > kGeometryTolerance ||
        std::abs(u.dot(v)) > kGeometryTolerance ||
        std::abs(u.z()) > kGeometryTolerance ||
        std::abs(v.z()) > kGeometryTolerance) {
      return Failure<grid_map_msgs::msg::GridMap>(
          "LOCAL_MAP_AXES_INVALID");
    }
    u.normalize();
    v.normalize();
    const double handedness = u.cross(v).z();
    if (std::abs(std::abs(handedness) - 1.0) > kGeometryTolerance) {
      return Failure<grid_map_msgs::msg::GridMap>(
          "LOCAL_MAP_AXES_INVALID");
    }
    const bool flip_v = handedness < 0.0;
    const Eigen::Vector3d grid_v = flip_v ? -v : v;
    Eigen::Matrix3d grid_orientation;
    grid_orientation.col(0) = u;
    grid_orientation.col(1) = grid_v;
    grid_orientation.col(2) = Eigen::Vector3d::UnitZ();
    const double resolution_m =
        frame.metadata.at("resolution_cm").get<double>() *
        transform.length_unit_to_m();
    if (std::abs(resolution_m - 0.2) > kGeometryTolerance) {
      return Failure<grid_map_msgs::msg::GridMap>(
          "LOCAL_MAP_RESOLUTION_INVALID");
    }

    std::vector<float> elevations(kMapCellCount, 0.0F);
    std::vector<float> valid_mask(kMapCellCount, 0.0F);
    for (std::size_t source_y = 0U; source_y < kMapHeight; ++source_y) {
      const std::size_t target_y =
          flip_v ? kMapHeight - 1U - source_y : source_y;
      for (std::size_t x = 0U; x < kMapWidth; ++x) {
        const std::size_t source_index = source_y * kMapWidth + x;
        const float elevation_native =
            ReadFloat32(frame.payload, source_index * sizeof(float));
        const bool valid = ValidBit(frame.payload, source_index);
        if (valid && !std::isfinite(elevation_native)) {
          return Failure<grid_map_msgs::msg::GridMap>(
              "LOCAL_MAP_VALID_ELEVATION_NONFINITE");
        }
        if (!valid && elevation_native != 0.0F) {
          return Failure<grid_map_msgs::msg::GridMap>(
              "LOCAL_MAP_INVALID_CELL_ENCODING");
        }
        const std::size_t target_index = target_y * kMapWidth + x;
        if (valid) {
          const Eigen::Vector3d elevation_point =
              transform.UnrealPositionToMap(
                  {0.0, 0.0, static_cast<double>(elevation_native)});
          elevations[target_index] =
              static_cast<float>(elevation_point.z());
          valid_mask[target_index] = 1.0F;
        }
      }
    }

    grid_map_msgs::msg::GridMap output;
    output.header.stamp = *stamp;
    output.header.frame_id = "map";
    output.info.resolution = resolution_m;
    output.info.length_x = resolution_m * static_cast<double>(kMapWidth);
    output.info.length_y = resolution_m * static_cast<double>(kMapHeight);
    const Eigen::Vector3d center = cell_zero +
        u * (0.5 * static_cast<double>(kMapWidth - 1U) * resolution_m) +
        v * (0.5 * static_cast<double>(kMapHeight - 1U) * resolution_m);
    output.info.pose.position.x = center.x();
    output.info.pose.position.y = center.y();
    output.info.pose.position.z = center.z();
    Eigen::Quaterniond orientation{grid_orientation};
    orientation.normalize();
    output.info.pose.orientation.x = orientation.x();
    output.info.pose.orientation.y = orientation.y();
    output.info.pose.orientation.z = orientation.z();
    output.info.pose.orientation.w = orientation.w();
    output.layers = {"elevation", "valid_mask"};
    output.basic_layers = output.layers;
    output.data = {
        EncodeGridMapLayer(elevations),
        EncodeGridMapLayer(valid_mask),
    };
    output.outer_start_index = 0U;
    output.inner_start_index = 0U;
    return Success(std::move(output));
  } catch (const std::exception&) {
    return Failure<grid_map_msgs::msg::GridMap>(
        "LOCAL_MAP_TRANSFORM_INVALID");
  }
}

ConversionResult<Frame> ConvertMotionReference(
    const lunar_planning_msgs::msg::MotionReference& message,
    const CoordinateTransform& transform,
    const FrozenSession& session,
    const std::uint64_t outgoing_sequence) {
  if (message.platform_type != message.WHEELED) {
    return Failure<Frame>("REFERENCE_PLATFORM_NOT_WHEELED");
  }
  if (session.session_id.empty() || message.plan_id.empty() ||
      outgoing_sequence == 0U) {
    return Failure<Frame>("REFERENCE_IDENTITY_INVALID");
  }
  if (session.calibration_hash != transform.calibration_hash()) {
    return Failure<Frame>("REFERENCE_CALIBRATION_HASH_MISMATCH");
  }
  if (message.trajectory.header.frame_id != "odom" ||
      message.trajectory.points.empty()) {
    return Failure<Frame>("REFERENCE_TRAJECTORY_INVALID");
  }
  const auto input_time_ns = Nanoseconds(message.input_time);
  if (!input_time_ns.has_value()) {
    return Failure<Frame>("REFERENCE_INPUT_TIME_INVALID");
  }

  Frame frame;
  frame.header.message_type = MessageType::kMotionReference;
  frame.header.sequence = outgoing_sequence;
  frame.header.simulation_time_ns = *input_time_ns;
  frame.payload.reserve(
      message.trajectory.points.size() * kTrajectoryRecordBytes);
  std::int64_t previous_time = -1;
  for (std::size_t index = 0U;
       index < message.trajectory.points.size(); ++index) {
    const auto& point = message.trajectory.points[index];
    const auto time = Nanoseconds(point.time_from_start);
    if (!time.has_value() || (index == 0U && *time != 0) ||
        (index != 0U && *time <= previous_time) ||
        point.transforms.size() != 1U || point.velocities.size() != 1U ||
        !point.accelerations.empty() || !FiniteTwist(point.velocities[0])) {
      return Failure<Frame>("REFERENCE_TRAJECTORY_POINT_INVALID");
    }
    try {
      const Eigen::Isometry3d map_from_footprint =
          EigenPose(point.transforms[0]);
      const Eigen::Isometry3d unreal_from_root =
          transform.UnrealFromRootForMapBaseFootprint(map_from_footprint);
      const Eigen::Isometry3d map_from_root =
          map_from_footprint *
          transform.base_footprint_from_base_link() *
          transform.base_link_from_unreal_root();
      const Eigen::Vector3d linear_footprint_map{
          point.velocities[0].linear.x,
          point.velocities[0].linear.y,
          point.velocities[0].linear.z,
      };
      const Eigen::Vector3d angular_map{
          point.velocities[0].angular.x,
          point.velocities[0].angular.y,
          point.velocities[0].angular.z,
      };
      const Eigen::Vector3d root_linear_map = linear_footprint_map +
          angular_map.cross(
              map_from_root.translation() -
              map_from_footprint.translation());
      const Eigen::Vector3d linear_unreal =
          transform.MapLinearVelocityToUnreal(root_linear_map);
      const Eigen::Vector3d angular_unreal =
          transform.MapAngularVelocityToUnreal(angular_map);
      const Eigen::Quaterniond orientation_unreal{
          unreal_from_root.rotation()};

      AppendLittleInteger(frame.payload, *time);
      for (std::size_t axis = 0U; axis < 3U; ++axis) {
        AppendDouble(frame.payload, unreal_from_root.translation()[axis]);
      }
      AppendDouble(frame.payload, orientation_unreal.x());
      AppendDouble(frame.payload, orientation_unreal.y());
      AppendDouble(frame.payload, orientation_unreal.z());
      AppendDouble(frame.payload, orientation_unreal.w());
      for (std::size_t axis = 0U; axis < 3U; ++axis) {
        AppendDouble(frame.payload, linear_unreal[axis]);
      }
      for (std::size_t axis = 0U; axis < 3U; ++axis) {
        AppendDouble(frame.payload, angular_unreal[axis]);
      }
    } catch (const std::exception&) {
      return Failure<Frame>("REFERENCE_TRANSFORM_INVALID");
    }
    previous_time = *time;
  }
  frame.metadata = {
      {"session_id", session.session_id},
      {"plan_id", message.plan_id},
      {"platform_type", "WHEELED"},
      {"input_time_ns", *input_time_ns},
      {"point_count", message.trajectory.points.size()},
      {"source_frame", "base_footprint"},
      {"target_unreal_reference", "robot_root"},
      {"execution_directive", "ACTIVATE_NEW_REFERENCE"},
      {"calibration_hash", session.calibration_hash},
  };
  if (const auto error = ValidateMetadata(
          MessageType::kMotionReference, frame.metadata)) {
    return Failure<Frame>(error->reason_code);
  }
  return Success(std::move(frame));
}

ConversionResult<lunar_navigation_msgs::msg::MotionExecutionFeedback>
ConvertExecutionFeedback(
    const Frame& frame, const std::string_view expected_session_id) {
  using Feedback = lunar_navigation_msgs::msg::MotionExecutionFeedback;
  if (frame.header.message_type != MessageType::kExecutionFeedback ||
      !frame.payload.empty()) {
    return Failure<Feedback>("EXECUTION_FEEDBACK_FRAME_INVALID");
  }
  if (const auto error = ValidateMetadata(
          MessageType::kExecutionFeedback, frame.metadata)) {
    return Failure<Feedback>(error->reason_code);
  }
  if (frame.metadata.at("session_id").get<std::string>() !=
      expected_session_id) {
    return Failure<Feedback>("EXECUTION_FEEDBACK_SESSION_MISMATCH");
  }
  if (frame.metadata.at("plan_id") != frame.metadata.at("segment_id")) {
    return Failure<Feedback>("EXECUTION_FEEDBACK_SEGMENT_MISMATCH");
  }
  const auto stamp = Stamp(frame.header.simulation_time_ns);
  if (!stamp.has_value()) {
    return Failure<Feedback>("EXECUTION_FEEDBACK_STAMP_INVALID");
  }

  Feedback output;
  output.header.stamp = *stamp;
  output.header.frame_id = "base_footprint";
  output.sequence = frame.metadata.at("sequence").get<std::uint64_t>();
  output.platform_type = Feedback::WHEELED;
  output.plan_id = frame.metadata.at("plan_id").get<std::string>();
  output.segment_id = frame.metadata.at("segment_id").get<std::string>();
  output.state = FeedbackState(
      frame.metadata.at("state").get_ref<const std::string&>());
  output.reason_code =
      frame.metadata.at("reason_code").get<std::string>();
  return Success(std::move(output));
}

}  // namespace lunar::unreal_tcp
