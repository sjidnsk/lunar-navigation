#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lunar::unreal_tcp {

struct CoordinateTransformConfig final {
  Eigen::Matrix3d basis_map_from_unreal{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d unreal_world_origin_in_map_m{Eigen::Vector3d::Zero()};
  double length_unit_to_m{0.01};
  Eigen::Matrix4d base_link_from_unreal_root{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d base_footprint_from_base_link{Eigen::Matrix4d::Identity()};
  std::string calibration_hash;
};

class CoordinateTransform final {
 public:
  CoordinateTransform(
      CoordinateTransformConfig config,
      const std::string& declared_calibration_hash);

  [[nodiscard]] Eigen::Vector3d UnrealPositionToMap(
      const Eigen::Vector3d& position_native) const;
  [[nodiscard]] Eigen::Vector3d MapPositionToUnreal(
      const Eigen::Vector3d& position_m) const;
  [[nodiscard]] Eigen::Vector3d UnrealDirectionToMap(
      const Eigen::Vector3d& direction) const;
  [[nodiscard]] Eigen::Vector3d MapDirectionToUnreal(
      const Eigen::Vector3d& direction) const;
  [[nodiscard]] Eigen::Vector3d UnrealLinearVelocityToMap(
      const Eigen::Vector3d& velocity_native_per_s) const;
  [[nodiscard]] Eigen::Vector3d MapLinearVelocityToUnreal(
      const Eigen::Vector3d& velocity_mps) const;
  [[nodiscard]] Eigen::Vector3d UnrealAngularVelocityToMap(
      const Eigen::Vector3d& angular_velocity_radps) const;
  [[nodiscard]] Eigen::Vector3d MapAngularVelocityToUnreal(
      const Eigen::Vector3d& angular_velocity_radps) const;
  [[nodiscard]] Eigen::Quaterniond UnrealOrientationToMap(
      const Eigen::Quaterniond& orientation) const;
  [[nodiscard]] Eigen::Quaterniond MapOrientationToUnreal(
      const Eigen::Quaterniond& orientation) const;

  [[nodiscard]] Eigen::Isometry3d MapFromUnrealRoot(
      const Eigen::Vector3d& position_native,
      const Eigen::Quaterniond& orientation) const;
  [[nodiscard]] Eigen::Isometry3d MapFromBaseLink(
      const Eigen::Vector3d& root_position_native,
      const Eigen::Quaterniond& root_orientation) const;
  [[nodiscard]] Eigen::Isometry3d MapFromBaseFootprint(
      const Eigen::Vector3d& root_position_native,
      const Eigen::Quaterniond& root_orientation) const;
  [[nodiscard]] Eigen::Isometry3d UnrealFromRootForMapBaseFootprint(
      const Eigen::Isometry3d& map_from_base_footprint) const;

  [[nodiscard]] const Eigen::Isometry3d& base_link_from_unreal_root() const;
  [[nodiscard]] const Eigen::Isometry3d&
  base_footprint_from_base_link() const;
  [[nodiscard]] double length_unit_to_m() const noexcept;
  [[nodiscard]] const std::string& calibration_hash() const noexcept;

 private:
  CoordinateTransformConfig config_;
  Eigen::Matrix3d inverse_basis_;
  double basis_determinant_{};
  Eigen::Isometry3d base_link_from_unreal_root_;
  Eigen::Isometry3d base_footprint_from_base_link_;
};

}  // namespace lunar::unreal_tcp
