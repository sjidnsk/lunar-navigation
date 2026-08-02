#pragma once

#include <string>
#include <string_view>

#include <Eigen/Core>

#include "lunar_path_planner/v3/hopper/aim_point_generator.hpp"

namespace lunar::planning::v3 {

struct SpatialValidityRegion final {
  AxisAlignedBox3 bounds_m;
};

struct TimeValidityInterval final {
  ClockStamp valid_from;
  ClockStamp valid_until;
};

struct GravityModelView final {
  ContentRef model_ref;
  FrameId frame_id;
  Eigen::Vector3d nominal_acceleration_mps2{
      Eigen::Vector3d::Zero()};
  AxisAlignedBox3 acceleration_error_mps2;
  SpatialValidityRegion spatial_validity;
  TimeValidityInterval time_validity;
};

struct BallisticState final {
  Eigen::Vector3d position_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity_mps{Eigen::Vector3d::Zero()};
};

struct NominalBallisticArc final {
  std::string aim_point_id;
  FrameId frame_id;
  Eigen::Vector3d launch_position_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d aim_position_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gravity_mps2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d launch_velocity_mps{Eigen::Vector3d::Zero()};
  Eigen::Vector3d landing_velocity_mps{Eigen::Vector3d::Zero()};
  double flight_time_s{};
};

[[nodiscard]] GravityModelView gravity_model_view(
    const GravityModel& model);

[[nodiscard]] Result<NominalBallisticArc>
make_nominal_ballistic_arc(
    const Eigen::Vector3d& launch_position_m,
    std::string_view launch_frame_id,
    const AimPointCandidate& aim_point,
    const GravityModelView& gravity,
    double flight_time_s);

[[nodiscard]] BallisticState evaluate_ballistic_state(
    const NominalBallisticArc& arc, double time_s);

[[nodiscard]] double ballistic_apex_time_s(
    const NominalBallisticArc& arc);

}  // namespace lunar::planning::v3
