#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Geometry>

#include "lunar_path_planner/v3/hopper/hopper_config.hpp"

namespace lunar::planning::v3 {

struct ArbitraryAxisAttitudeCapability final {
  double maximum_angular_speed_radps{};
  double maximum_angular_acceleration_radps2{};
  double maximum_initial_angular_speed_radps{};
  DurationNanoseconds minimum_settle_guard;
  ContentRef source_safety_capability_ref;
  std::optional<ContentRef> attitude_tightening_table_ref;
  std::shared_ptr<const AttitudeTighteningTable>
      resolved_attitude_tightening_table;
  HopperErrorBounds certified_state_error_bounds;
};

struct AttitudeCertificationInput final {
  Eigen::Quaterniond initial_orientation_body_to_frame{
      Eigen::Quaterniond::Identity()};
  Eigen::Vector3d initial_angular_velocity_radps{
      Eigen::Vector3d::Zero()};
  RotationVectorBall initial_orientation_error_set;
  DeterministicVectorSet3
      initial_angular_velocity_error_set_radps;
  LandingPlane landing_plane;
  CircularYawInterval allowed_yaw_interval;
  double flight_time_s{};
};

struct AttitudeCertificationResult final {
  std::optional<AttitudeBoundary> boundary;
  double shortest_rotation_angle_rad{};
  double certified_required_rotation_time_s{};
  double worst_case_initial_angular_speed_radps{};
  double effective_maximum_speed_radps{};
  double effective_maximum_acceleration_radps2{};
  std::string rejection_reason;
};

[[nodiscard]] ArbitraryAxisAttitudeCapability
attitude_capability_view(const HopperCapabilityView& capability);

class AttitudeCertifier final {
 public:
  explicit AttitudeCertifier(
      ArbitraryAxisAttitudeCapability capability);

  [[nodiscard]] Result<double> conservative_required_time_s(
      const Eigen::Quaterniond& initial_orientation_body_to_frame,
      const Eigen::Vector3d& initial_angular_velocity_radps,
      const RotationVectorBall& initial_orientation_error_set,
      const DeterministicVectorSet3&
          initial_angular_velocity_error_set_radps,
      const LandingPlane& landing_plane,
      const CircularYawInterval& allowed_yaw_interval) const;

  [[nodiscard]] AttitudeCertificationResult certify(
      const AttitudeCertificationInput& input) const;

 private:
  ArbitraryAxisAttitudeCapability capability_;
};

[[nodiscard]] double shortest_quaternion_angle_rad(
    const Eigen::Quaterniond& from,
    const Eigen::Quaterniond& to);

[[nodiscard]] double bang_bang_rotation_time_s(
    double angle_rad,
    double maximum_speed_radps,
    double maximum_acceleration_radps2);

}  // namespace lunar::planning::v3
