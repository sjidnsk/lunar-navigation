#include "lunar_unreal_tcp_bridge/ros_conversions.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <gtest/gtest.h>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>

namespace lunar::unreal_tcp {
namespace {

constexpr char kSessionId[] = "123e4567-e89b-12d3-a456-426614174000";
constexpr char kCalibrationHash[] = "sha256:ros-conversion";
constexpr std::size_t kMapWidth = 320U;
constexpr std::size_t kMapHeight = 320U;
constexpr std::size_t kMapCellCount = kMapWidth * kMapHeight;
constexpr std::size_t kValidityBytes = kMapCellCount / 8U;

CoordinateTransform Transform() {
  CoordinateTransformConfig config;
  config.basis_map_from_unreal =
      (Eigen::Vector3d{1.0, -1.0, 1.0}).asDiagonal();
  config.length_unit_to_m = 0.01;
  config.calibration_hash = kCalibrationHash;
  return CoordinateTransform{config, kCalibrationHash};
}

SimulationCovarianceProfile Covariance() {
  SimulationCovarianceProfile profile;
  for (std::size_t index = 0U; index < 6U; ++index) {
    profile.pose[index * 6U + index] = 0.1;
    profile.twist[index * 6U + index] = 0.2;
  }
  return profile;
}

Frame RobotStateFrame() {
  Frame frame;
  frame.header.message_type = MessageType::kRobotState;
  frame.header.sequence = 2U;
  frame.header.simulation_time_ns = 2'000'000'000LL;
  frame.metadata = {
      {"session_id", kSessionId},
      {"position_cm", {100.0, 200.0, 300.0}},
      {"quaternion_xyzw", {0.0, 0.0, 0.0, 1.0}},
      {"linear_velocity_cmps", {100.0, 0.0, 0.0}},
      {"angular_velocity_radps", {0.0, 0.0, 1.0}},
      {"control_state", "READY"},
      {"simulation_covariance_profile", "deterministic"},
  };
  return frame;
}

void AppendFloat32(std::vector<std::byte>& output, const float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  for (std::size_t offset = 0U; offset < 4U; ++offset) {
    output.push_back(static_cast<std::byte>((bits >> (offset * 8U)) & 0xffU));
  }
}

Frame LocalMapFrame() {
  Frame frame;
  frame.header.message_type = MessageType::kLocalElevationMap;
  frame.header.sequence = 3U;
  frame.header.simulation_time_ns = 3'000'000'000LL;
  const nlohmann::json pose{
      {"position_cm", {0.0, 0.0, 0.0}},
      {"quaternion_xyzw", {0.0, 0.0, 0.0, 1.0}},
  };
  frame.metadata = {
      {"session_id", kSessionId},
      {"width", 320},
      {"height", 320},
      {"resolution_cm", 20.0},
      {"cell_zero_center_world_cm", {-205160.0, -511559.0, 50.0}},
      {"u_axis_world", {1.0, 0.0, 0.0}},
      {"v_axis_world", {0.0, 1.0, 0.0}},
      {"sensor_pose_world", pose},
      {"robot_pose_world", pose},
      {"elevation_encoding", "float32_le"},
      {"valid_encoding", "bitset_lsb0"},
  };
  frame.payload.reserve(kMapCellCount * 4U + kValidityBytes);
  for (std::size_t index = 0U; index < kMapCellCount; ++index) {
    const float value = index == 0U ? 123.0F :
        (index + 1U == kMapCellCount ? 200.0F : 100.0F);
    AppendFloat32(frame.payload, value);
  }
  frame.payload.insert(frame.payload.end(), kValidityBytes, std::byte{0xff});
  return frame;
}

std::vector<float> UnwrapLayer(
    const std_msgs::msg::Float32MultiArray& layer) {
  std::vector<float> output(kMapCellCount);
  for (std::size_t y = 0U; y < kMapHeight; ++y) {
    for (std::size_t x = 0U; x < kMapWidth; ++x) {
      const std::size_t physical_row = kMapWidth - 1U - x;
      const std::size_t physical_column = kMapHeight - 1U - y;
      output[y * kMapWidth + x] =
          layer.data[physical_column * kMapWidth + physical_row];
    }
  }
  return output;
}

FrozenSession Session() {
  FrozenSession session;
  session.session_id = kSessionId;
  session.calibration_hash = kCalibrationHash;
  return session;
}

lunar_planning_msgs::msg::MotionReference MotionReference() {
  lunar_planning_msgs::msg::MotionReference message;
  message.header.frame_id = "map";
  message.header.stamp.sec = 10;
  message.plan_id = "wheel-plan";
  message.platform_type = message.WHEELED;
  message.input_time.sec = 10;
  message.trajectory.header.frame_id = "odom";
  message.trajectory.joint_names = {"base_link"};

  trajectory_msgs::msg::MultiDOFJointTrajectoryPoint first;
  geometry_msgs::msg::Transform first_transform;
  first_transform.translation.x = 1.0;
  first_transform.translation.y = 2.0;
  first_transform.translation.z = 3.0;
  first_transform.rotation.w = 1.0;
  first.transforms.push_back(first_transform);
  geometry_msgs::msg::Twist first_twist;
  first_twist.linear.x = 1.0;
  first_twist.linear.y = 2.0;
  first_twist.linear.z = 3.0;
  first_twist.angular.z = 0.5;
  first.velocities.push_back(first_twist);
  first.time_from_start.sec = 0;

  auto second = first;
  second.transforms[0].translation.x = 2.0;
  second.time_from_start.sec = 1;
  message.trajectory.points = {first, second};
  return message;
}

template<typename Integer>
Integer ReadLittleInteger(
    const std::vector<std::byte>& bytes, const std::size_t offset) {
  using Unsigned = std::make_unsigned_t<Integer>;
  Unsigned value{};
  for (std::size_t index = 0U; index < sizeof(Integer); ++index) {
    value |= static_cast<Unsigned>(std::to_integer<std::uint8_t>(
        bytes.at(offset + index))) << (index * 8U);
  }
  return static_cast<Integer>(value);
}

double ReadDouble(
    const std::vector<std::byte>& bytes, const std::size_t offset) {
  return std::bit_cast<double>(ReadLittleInteger<std::uint64_t>(bytes, offset));
}

Frame FeedbackFrame(std::string state, std::string reason_code) {
  Frame frame;
  frame.header.message_type = MessageType::kExecutionFeedback;
  frame.header.sequence = 9U;
  frame.header.simulation_time_ns = 4'000'000'000LL;
  frame.metadata = {
      {"session_id", kSessionId},
      {"platform_type", "WHEELED"},
      {"plan_id", "wheel-plan"},
      {"segment_id", "wheel-plan"},
      {"sequence", 2U},
      {"state", std::move(state)},
      {"reason_code", std::move(reason_code)},
  };
  return frame;
}

TEST(RosConversions, RobotStateProducesExternalAndWheeledOdometryFromOneSnapshot) {
  const auto result = ConvertRobotState(
      RobotStateFrame(), Transform(), Covariance());
  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto& output = *result.value;
  EXPECT_EQ(output.external_odometry.header.frame_id, "odom");
  EXPECT_EQ(output.external_odometry.child_frame_id, "base_link");
  EXPECT_EQ(output.wheeled_odometry.child_frame_id, "base_footprint");
  EXPECT_EQ(output.external_odometry.header.stamp.sec, 2);
  EXPECT_DOUBLE_EQ(output.external_odometry.pose.pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(output.external_odometry.pose.pose.position.y, -2.0);
  EXPECT_DOUBLE_EQ(output.external_odometry.pose.pose.position.z, 3.0);
  EXPECT_DOUBLE_EQ(output.external_odometry.twist.twist.linear.x, 1.0);
  EXPECT_DOUBLE_EQ(output.external_odometry.twist.twist.angular.z, -1.0);
  EXPECT_DOUBLE_EQ(output.external_odometry.pose.covariance[0], 0.1);
  EXPECT_DOUBLE_EQ(output.external_odometry.twist.covariance[0], 0.2);
  ASSERT_EQ(output.dynamic_tf.transforms.size(), 2U);
  EXPECT_EQ(output.dynamic_tf.transforms[0].header.frame_id, "map");
  EXPECT_EQ(output.dynamic_tf.transforms[0].child_frame_id, "odom");
  EXPECT_EQ(output.dynamic_tf.transforms[1].header.frame_id, "odom");
  EXPECT_EQ(output.dynamic_tf.transforms[1].child_frame_id, "base_link");
  ASSERT_EQ(output.static_tf.transforms.size(), 1U);
  EXPECT_EQ(output.static_tf.transforms[0].header.frame_id, "base_link");
  EXPECT_EQ(output.static_tf.transforms[0].child_frame_id, "base_footprint");
}

TEST(RosConversions, ObservedMapUsesCellZeroAndUvAxesNotLandscapeOrigin) {
  const auto result = ConvertObservedElevation(LocalMapFrame(), Transform());
  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto& map = *result.value;
  EXPECT_EQ(map.header.frame_id, "map");
  EXPECT_EQ(map.layers, (std::vector<std::string>{"elevation", "valid_mask"}));
  EXPECT_EQ(map.basic_layers, map.layers);
  EXPECT_DOUBLE_EQ(map.info.resolution, 0.2);
  EXPECT_DOUBLE_EQ(map.info.length_x, 64.0);
  EXPECT_DOUBLE_EQ(map.info.length_y, 64.0);
  EXPECT_NEAR(map.info.pose.position.x, -2019.7, 1.0e-9);
  EXPECT_NEAR(map.info.pose.position.y, 5083.69, 1.0e-9);
  EXPECT_NEAR(map.info.pose.orientation.w, 1.0, 1.0e-12);
  ASSERT_EQ(map.data.size(), 2U);
  const std::vector<float> elevation = UnwrapLayer(map.data[0]);
  const std::vector<float> valid = UnwrapLayer(map.data[1]);
  EXPECT_FLOAT_EQ(elevation[(kMapHeight - 1U) * kMapWidth], 1.23F);
  EXPECT_FLOAT_EQ(elevation[kMapWidth - 1U], 2.0F);
  EXPECT_FLOAT_EQ(valid.front(), 1.0F);
  EXPECT_FLOAT_EQ(valid.back(), 1.0F);
}

TEST(RosConversions, ObservedMapRejectsBadBitsetAndNonfiniteValidElevation) {
  auto short_payload = LocalMapFrame();
  short_payload.payload.pop_back();
  const auto short_result =
      ConvertObservedElevation(short_payload, Transform());
  EXPECT_FALSE(short_result.ok());
  EXPECT_EQ(short_result.reason_code, "LOCAL_MAP_PAYLOAD_SIZE_INVALID");

  auto nonfinite = LocalMapFrame();
  const std::uint32_t nan_bits = std::bit_cast<std::uint32_t>(
      std::numeric_limits<float>::quiet_NaN());
  for (std::size_t index = 0U; index < 4U; ++index) {
    nonfinite.payload[index] =
        static_cast<std::byte>((nan_bits >> (index * 8U)) & 0xffU);
  }
  const auto nan_result = ConvertObservedElevation(nonfinite, Transform());
  EXPECT_FALSE(nan_result.ok());
  EXPECT_EQ(
      nan_result.reason_code, "LOCAL_MAP_VALID_ELEVATION_NONFINITE");
}

TEST(RosConversions, MotionReferenceEncodes112ByteStrictlyIncreasingRecords) {
  const auto result = ConvertMotionReference(
      MotionReference(), Transform(), Session(), 7U);
  ASSERT_TRUE(result.ok()) << result.reason_code;
  const Frame& frame = *result.value;
  EXPECT_EQ(frame.header.message_type, MessageType::kMotionReference);
  EXPECT_EQ(frame.header.sequence, 7U);
  EXPECT_EQ(frame.metadata.at("point_count"), 2U);
  EXPECT_EQ(frame.metadata.at("source_frame"), "base_footprint");
  EXPECT_EQ(frame.metadata.at("calibration_hash"), kCalibrationHash);
  ASSERT_EQ(frame.payload.size(), 224U);
  EXPECT_EQ(ReadLittleInteger<std::int64_t>(frame.payload, 0U), 0);
  EXPECT_DOUBLE_EQ(ReadDouble(frame.payload, 8U), 100.0);
  EXPECT_DOUBLE_EQ(ReadDouble(frame.payload, 16U), -200.0);
  EXPECT_DOUBLE_EQ(ReadDouble(frame.payload, 24U), 300.0);
  EXPECT_DOUBLE_EQ(ReadDouble(frame.payload, 64U), 100.0);
  EXPECT_DOUBLE_EQ(ReadDouble(frame.payload, 72U), -200.0);
  EXPECT_DOUBLE_EQ(ReadDouble(frame.payload, 80U), 300.0);
  EXPECT_DOUBLE_EQ(ReadDouble(frame.payload, 104U), -0.5);
  EXPECT_EQ(
      ReadLittleInteger<std::int64_t>(frame.payload, 112U),
      1'000'000'000LL);
}

TEST(RosConversions, MotionReferenceRejectsNonWheeledOrWrongCalibration) {
  auto wrong_platform = MotionReference();
  wrong_platform.platform_type = wrong_platform.LEGGED;
  const auto platform_result = ConvertMotionReference(
      wrong_platform, Transform(), Session(), 7U);
  EXPECT_FALSE(platform_result.ok());
  EXPECT_EQ(platform_result.reason_code, "REFERENCE_PLATFORM_NOT_WHEELED");

  auto wrong_session = Session();
  wrong_session.calibration_hash = "sha256:different";
  const auto calibration_result = ConvertMotionReference(
      MotionReference(), Transform(), wrong_session, 7U);
  EXPECT_FALSE(calibration_result.ok());
  EXPECT_EQ(
      calibration_result.reason_code, "REFERENCE_CALIBRATION_HASH_MISMATCH");
}

TEST(RosConversions, ExecutionFeedbackMapsStableStatesAndReasons) {
  const auto accepted = ConvertExecutionFeedback(
      FeedbackFrame("ACCEPTED", ""), kSessionId);
  ASSERT_TRUE(accepted.ok()) << accepted.reason_code;
  EXPECT_EQ(
      accepted.value->state,
      lunar_navigation_msgs::msg::MotionExecutionFeedback::ACCEPTED);
  EXPECT_EQ(accepted.value->header.frame_id, "base_footprint");
  EXPECT_EQ(accepted.value->header.stamp.sec, 4);
  EXPECT_EQ(accepted.value->sequence, 2U);
  EXPECT_EQ(accepted.value->plan_id, "wheel-plan");

  const auto failed = ConvertExecutionFeedback(
      FeedbackFrame("FAILED", "TRACKING_ERROR"), kSessionId);
  ASSERT_TRUE(failed.ok()) << failed.reason_code;
  EXPECT_EQ(
      failed.value->state,
      lunar_navigation_msgs::msg::MotionExecutionFeedback::FAILED);
  EXPECT_EQ(failed.value->reason_code, "TRACKING_ERROR");
}

}  // namespace
}  // namespace lunar::unreal_tcp
